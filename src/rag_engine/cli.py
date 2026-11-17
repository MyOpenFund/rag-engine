"""The ``rag-engine`` command.

``rag-engine --version`` prints the version. ``rag-engine eval`` measures the search of a
collection on a question file (:mod:`rag_engine.eval`) and prints the report::

    rag-engine eval --questions eval/known_item_v1.jsonl --collection documents \\
        --k 10 --chunk-limit 50 --backend onnx --output report.json --fail-under-hit-at-k 0.9

This module is the one place of the engine that reads the settings by itself: the Qdrant
connection, the model and the search maximum come from the environment
(:meth:`rag_engine.config.Settings.from_env`). A ``.env`` file is not read.

Exit codes of ``eval``: 0 when the evaluation ran (and met ``--fail-under-hit-at-k`` when
given); 1 when ``--fail-under-hit-at-k`` is given and not met; 2 for anything that kept the
evaluation from running to its end: bad arguments, a bad setting, an unusable question
file, an unreachable server, a missing collection or embedding backend, a failure midway.

Importing this module loads no embedding backend.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .config import BACKENDS, Settings
from .embedding import Embedder, make_embedder
from .errors import RagEngineError
from .eval import DEFAULT_CHUNK_LIMIT, DEFAULT_K, evaluate, format_report, load_questions
from .store import QdrantStore

__all__ = ["main"]

PROGRAM = "rag-engine"


class _Refused(Exception):
    """The command cannot run; the message is for the user."""


def _share(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{text!r} is not between 0 and 1")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM, description="Ingest documents into Qdrant and search them: the tools of the engine."
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    evaluation = commands.add_parser(
        "eval",
        help="measure the search of a collection on a question file",
        description=(
            "Run the search for every question of a JSON Lines question file and report, on documents, "
            "hit@1, hit@k, MRR and recall@k. The Qdrant connection and the model are read from the "
            "environment (QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_HTTPS, RAG_ENGINE_MODEL, "
            "RAG_ENGINE_BACKEND, ...). Exit code: 0, 1 when --fail-under-hit-at-k is not met, "
            "2 when the evaluation could not run."
        ),
    )
    evaluation.add_argument(
        "--questions", required=True, metavar="FILE", help="the question file (JSON Lines)"
    )
    evaluation.add_argument(
        "--collection", required=True, metavar="NAME", help="the Qdrant collection to search"
    )
    evaluation.add_argument(
        "--k", type=int, default=DEFAULT_K, help=f"documents the metrics look at (default {DEFAULT_K})"
    )
    evaluation.add_argument(
        "--chunk-limit",
        type=int,
        default=DEFAULT_CHUNK_LIMIT,
        metavar="N",
        help=(
            f"chunks asked to the search per question, then grouped into documents (default "
            f"{DEFAULT_CHUNK_LIMIT}; at most RAG_ENGINE_SEARCH_MAX_LIMIT)"
        ),
    )
    evaluation.add_argument(
        "--backend", choices=BACKENDS, help="embedding backend (default: RAG_ENGINE_BACKEND, else torch)"
    )
    evaluation.add_argument("--output", metavar="FILE", help="also write the full report there, as JSON")
    evaluation.add_argument(
        "--fail-under-hit-at-k",
        type=_share,
        metavar="X",
        help="exit with code 1 when the overall hit@k is below X (between 0 and 1)",
    )
    return parser


def _store(settings: Settings) -> QdrantStore:
    return QdrantStore(settings=settings)


def _embedder(settings: Settings) -> Embedder:
    return make_embedder(settings)


def _check_counts(k: int, chunk_limit: int, maximum: int) -> None:
    """Refuse, before a model is loaded, the counts ``evaluate`` would refuse."""
    for name, value in (("--k", k), ("--chunk-limit", chunk_limit)):
        if value < 1:
            raise _Refused(f"{name} must be at least 1, got {value}")
    if chunk_limit > maximum:
        raise _Refused(
            f"chunk_limit {chunk_limit} is above the maximum of {maximum} chunks a search may return: "
            "lower --chunk-limit or raise RAG_ENGINE_SEARCH_MAX_LIMIT"
        )
    if k > chunk_limit:
        raise _Refused(f"--k {k} documents cannot be ranked from --chunk-limit {chunk_limit} chunks")


def _check_collection(store: Any, collection: str) -> int | None:
    """Make sure the server answers and holds ``collection``; returns the size of its vectors."""
    if not store.is_reachable():
        raise _Refused(
            f"Qdrant at {store.host}:{store.port} does not answer, or refuses the API key "
            "(check QDRANT_HOST, QDRANT_PORT, QDRANT_HTTPS and QDRANT_API_KEY)"
        )
    if not store.client.collection_exists(collection_name=collection):
        raise _Refused(f"collection {collection!r} does not exist on that server")
    vectors = store.client.get_collection(collection_name=collection).config.params.vectors
    size = getattr(vectors, "size", None)
    return int(size) if size is not None else None


def _evaluate(arguments: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if arguments.backend is not None:
        settings = dataclasses.replace(settings, backend=arguments.backend)
    _check_counts(arguments.k, arguments.chunk_limit, settings.search_max_limit)
    questions = load_questions(arguments.questions)
    if arguments.output is not None:
        folder = os.path.dirname(os.path.abspath(arguments.output))
        if not os.path.isdir(folder):
            raise _Refused(f"the directory of --output does not exist: {folder}")
    store = _store(settings)
    size = _check_collection(store, arguments.collection)
    try:
        embedder = _embedder(settings)
    except ValueError as exc:  # a model the backend does not know, or of another dimension
        raise _Refused(str(exc)) from exc
    try:
        if size != embedder.dim:
            raise _Refused(
                f"collection {arguments.collection!r} holds {size}-dimensional vectors, and the model "
                f"{embedder.model_name!r} produces {embedder.dim}-dimensional ones "
                "(check RAG_ENGINE_MODEL and RAG_ENGINE_DIM)"
            )
        try:
            report = evaluate(
                questions,
                collection=arguments.collection,
                store=store,
                embedder=embedder,
                k=arguments.k,
                chunk_limit=arguments.chunk_limit,
                max_limit=settings.search_max_limit,
            )
        except RagEngineError:
            raise
        except Exception as exc:  # exit code 1 is kept for the threshold
            raise _Refused(f"the evaluation failed: {type(exc).__name__}: {exc}") from exc
    finally:
        embedder.release()
    print(format_report(report))
    if arguments.output is not None:
        with open(arguments.output, "w", encoding="utf-8") as handle:
            handle.write(report.to_json() + "\n")
    threshold = arguments.fail_under_hit_at_k
    if threshold is None:
        return 0
    reached = report.overall.hit_at_k
    if reached is None:
        print(
            f"{PROGRAM}: no question could be scored: --fail-under-hit-at-k {threshold} is not met",
            file=sys.stderr,
        )
        return 1
    if reached < threshold:
        print(
            f"{PROGRAM}: hit@{arguments.k} is {reached:.3f}, below the required {threshold}", file=sys.stderr
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command with ``argv`` (default: the arguments of the process); returns the exit code.

    It never raises for a problem it can name: the message goes to the standard error
    stream, after ``rag-engine: error:``, and the code is 2. See the module for the codes.
    """
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as exit_:  # argparse has printed its message: usage errors are code 2
        return exit_.code if isinstance(exit_.code, int) else 2
    if arguments.command is None:
        parser.print_usage(sys.stderr)
        print(f"{PROGRAM}: error: a command is required (eval)", file=sys.stderr)
        return 2
    try:
        return _evaluate(arguments)
    except (_Refused, RagEngineError, OSError, ImportError) as exc:
        print(f"{PROGRAM}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
