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
file, an unreachable server, a missing collection or embedding backend, a collection of
another dimension, of another distance than cosine or with named vectors
(:meth:`rag_engine.store.QdrantStore.check_collection`, which only reads), a collection
that holds none of the expected documents (unless ``--allow-empty``), a failure midway.

``--fail-under-hit-at-k`` looks at the headline of the report: the hit@k of the kinds of
question asked without filters, averaged per document
(:attr:`rag_engine.eval.EvalReport.headline`), not at the pooled mean of all questions.

What is cheap is checked first: the arguments, the question file, the server, the
collection, its dimension and its distance, the presence of the expected documents; the embedding model is
loaded last.

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
from .eval import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CHUNK_LIMIT,
    DEFAULT_K,
    QuestionSet,
    evaluate,
    format_report,
    load_questions,
)
from .store import CollectionMismatchError, CollectionNotFoundError, QdrantStore

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
            "hit@1, hit@k, MRR and recall@k: per kind of question, then the headline (the kinds asked "
            "without filters, averaged per document, with a 95%% interval) and all questions pooled. "
            "The Qdrant connection and the model are read from the environment (QDRANT_HOST, "
            "QDRANT_PORT, QDRANT_API_KEY, QDRANT_HTTPS, RAG_ENGINE_MODEL, RAG_ENGINE_DIM, "
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
        help=(
            "exit with code 1 when the headline hit@k is below X (between 0 and 1): the hit@k of the "
            "kinds asked without filters, averaged per document"
        ),
    )
    evaluation.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        metavar="N",
        help=(
            "times the documents are drawn again for the 95%% intervals "
            f"(default {DEFAULT_BOOTSTRAP_RESAMPLES})"
        ),
    )
    evaluation.add_argument(
        "--allow-empty",
        action="store_true",
        help=(
            "run even when the collection holds none of the expected documents (nothing is scored); "
            "without it this is an error, exit code 2"
        ),
    )
    return parser


def _store(settings: Settings) -> QdrantStore:
    return QdrantStore(settings=settings)


def _embedder(settings: Settings) -> Embedder:
    return make_embedder(settings)


def _check_counts(k: int, chunk_limit: int, maximum: int, resamples: int) -> None:
    """Refuse, before a model is loaded, the counts ``evaluate`` would refuse."""
    for name, value in (("--k", k), ("--chunk-limit", chunk_limit), ("--bootstrap-resamples", resamples)):
        if value < 1:
            raise _Refused(f"{name} must be at least 1, got {value}")
    if chunk_limit > maximum:
        raise _Refused(
            f"chunk_limit {chunk_limit} is above the maximum of {maximum} chunks a search may return: "
            "lower --chunk-limit or raise RAG_ENGINE_SEARCH_MAX_LIMIT"
        )
    if k > chunk_limit:
        raise _Refused(f"--k {k} documents cannot be ranked from --chunk-limit {chunk_limit} chunks")


def _check_collection(store: Any, collection: str, dim: int) -> None:
    """Make sure the server answers and holds ``collection``: one unnamed cosine vector of ``dim`` values."""
    if not store.is_reachable():
        raise _Refused(
            f"Qdrant at {store.host}:{store.port} does not answer, or refuses the API key "
            "(check QDRANT_HOST, QDRANT_PORT, QDRANT_HTTPS and QDRANT_API_KEY)"
        )
    try:
        store.check_collection(collection, dim)  # read-only: the command never creates a collection
    except CollectionNotFoundError:
        raise _Refused(f"collection {collection!r} does not exist on that server") from None
    except CollectionMismatchError as exc:
        raise _Refused(
            f"{exc}. The engine searches the single unnamed vector of a collection, compared by "
            f"cosine, as QdrantStore.ensure_collection creates it; its dimension is the one "
            f"RAG_ENGINE_DIM announces, {dim} (check RAG_ENGINE_MODEL and RAG_ENGINE_DIM)"
        ) from exc
    except RagEngineError:
        raise  # a refused API key: reported as it is
    except Exception as exc:  # what the Qdrant client raised, e.g. for a name the server refuses
        raise _Refused(
            f"collection {collection!r} could not be checked: {type(exc).__name__}: {exc}"
        ) from exc


def _dimensions(collection: str, size: int, model: str) -> str:
    return (
        f"collection {collection!r} holds {size}-dimensional vectors, and {model} "
        "(check RAG_ENGINE_MODEL and RAG_ENGINE_DIM)"
    )


def _check_documents(store: Any, collection: str, questions: QuestionSet, allow_empty: bool) -> None:
    """Refuse a collection that holds none of the expected documents, unless ``allow_empty``."""
    wanted = sorted({doc_id for question in questions for doc_id in question.expected_doc_ids})
    if store.existing_doc_ids(collection, wanted):
        return
    message = (
        f"none of the {len(wanted)} documents the questions expect is in collection {collection!r}: "
        "no question can be scored (is it the collection these questions were written for?)"
    )
    if not allow_empty:
        raise _Refused(f"{message}; --allow-empty runs the evaluation all the same")
    print(f"{PROGRAM}: warning: {message}", file=sys.stderr)


def _evaluate(arguments: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if arguments.backend is not None:
        settings = dataclasses.replace(settings, backend=arguments.backend)
    _check_counts(
        arguments.k, arguments.chunk_limit, settings.search_max_limit, arguments.bootstrap_resamples
    )
    questions = load_questions(arguments.questions)
    if arguments.output is not None:
        folder = os.path.dirname(os.path.abspath(arguments.output))
        if not os.path.isdir(folder):
            raise _Refused(f"the directory of --output does not exist: {folder}")
    store = _store(settings)
    # Cheap checks first: the model is loaded only for an evaluation that can run.
    _check_collection(store, arguments.collection, settings.model_dim)
    _check_documents(store, arguments.collection, questions, arguments.allow_empty)
    try:
        embedder = _embedder(settings)
    except ValueError as exc:  # a model the backend does not know, or of another dimension
        raise _Refused(str(exc)) from exc
    try:
        if embedder.dim != settings.model_dim:
            raise _Refused(
                _dimensions(
                    arguments.collection,
                    settings.model_dim,
                    f"the model {embedder.model_name!r} produces {embedder.dim}-dimensional ones",
                )
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
                bootstrap_resamples=arguments.bootstrap_resamples,
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
    headline = report.headline.per_document
    if headline is None:
        print(
            f"{PROGRAM}: no question asked without filters could be scored: there is no headline, and "
            f"--fail-under-hit-at-k {threshold} is not met",
            file=sys.stderr,
        )
        return 1
    if headline.hit_at_k < threshold:
        print(
            f"{PROGRAM}: the headline hit@{arguments.k} is {headline.hit_at_k:.3f}, below the required "
            f"{threshold}",
            file=sys.stderr,
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
