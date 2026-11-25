#!/usr/bin/env python
"""Passage-level evaluation of the local index on SEC filings: chunking x retrieval.

For a directory of Markdown filings (``<doc_id>.md`` and a sidecar ``<doc_id>.json`` holding
``title``, ``form`` and ``date``) and a question file with ``expected_spans``
(``eval/filings_README.md``), the script builds one local index per chunking profile (v1 and v2) in
a temporary directory, runs every question through dense, hybrid and hybrid + eigenmind retrieval,
and prints hit@5, hit@10 and MRR@10 with intervals, the seconds each 10-K took to index, and the
milliseconds each search took. It then applies the two decision rules of the plan::

    v2 >= v1 on hit@10 (hybrid), and
    eigenmind is the default only if hybrid+eigenmind beats hybrid on hit@10 with an interval
    that does not overlap the interval of hybrid (otherwise it stays off).

Run it::

    python scripts/filings_eval.py FILINGS_DIR eval/filings_v1.jsonl --backend onnx
    python scripts/filings_eval.py FILINGS_DIR QUESTIONS --backend fake      # smoke test, no model

``--backend fake`` embeds with a hash-based ``FakeEmbedder`` and counts tokens by words: it checks
the plumbing, its numbers mean nothing. Every threshold is an option; the sweeps
(``--sweep-threshold``, ``--sweep-keep-top``, ``--sweep-solver``) add one eigenmind row per
combination.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from rag_engine.chunking import Chunk, ChunkingOptions, chunk_markdown, load_tokenizer
from rag_engine.config import Settings
from rag_engine.embedding import Embedder, FakeEmbedder, ModelSpec, make_embedder
from rag_engine.eval import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_INTERVAL_MIN_QUESTIONS,
    PassageReport,
    Question,
    evaluate_passages,
    load_questions,
)
from rag_engine.local import DEFAULT_CANDIDATES, DEFAULT_RRF_K, LocalIndex, collection_name, hybrid_search
from rag_engine.search import Hit, search
from rag_engine.select import DEFAULT_THRESHOLD, select_diverse

#: The chunking profiles compared, by label.
PROFILES: dict[str, Callable[[], ChunkingOptions]] = {"v1": ChunkingOptions, "v2": ChunkingOptions.v2}
#: The depth of the passages scored (hit@5 and hit@10 are read from it).
DEFAULT_K = 10
#: The metric the two decision rules read.
DECISION_METRIC = "hit_at_10"
_FAKE_SPEC = ModelSpec(name="fake", dim=16, passage_prefix="passage: ", query_prefix="query: ")


class _WordTokenizer:
    """Counts a word as a token: the stand-in of the model tokenizer with ``--backend fake``."""

    def encode(self, text: str) -> list[str]:
        return text.split()


@dataclass(frozen=True)
class Filing:
    """One filing: its id, Markdown text and the metadata of its sidecar."""

    doc_id: str
    text: str
    title: str
    form: str
    date: str


@dataclass
class Row:
    """One line of the table: a chunking x a retrieval, with what it cost."""

    chunking: str
    retrieval: str
    report: PassageReport
    seconds_per_filing: float
    ms_per_search: float
    chunks: int
    params: dict[str, Any] = field(default_factory=dict)


def load_filings(directory: str | Path) -> list[Filing]:
    """Every ``<doc_id>.md`` of ``directory`` with its ``<doc_id>.json`` sidecar, sorted by id.

    The sidecar holds ``title``, ``form`` and ``date`` (``YYYY-MM-DD``), all required.
    ``ValueError`` names the file when one is missing or the directory holds no filing.
    """
    root = Path(directory)
    filings = []
    for path in sorted(root.glob("*.md")):
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"{path}: no sidecar {sidecar.name} (title, form, date)")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        missing = [key for key in ("title", "form", "date") if not isinstance(meta.get(key), str)]
        if missing:
            raise ValueError(f"{sidecar}: missing or not a string: {', '.join(missing)}")
        filings.append(
            Filing(path.stem, path.read_text(encoding="utf-8"), meta["title"], meta["form"], meta["date"])
        )
    if not filings:
        raise ValueError(f"{root}: no Markdown filing")
    return filings


def build_index(
    root: Path,
    filings: Sequence[Filing],
    *,
    label: str,
    embedder: Embedder,
    tokenizer: Any,
    model_name: str,
) -> tuple[LocalIndex, float, int]:
    """Chunk, embed and write ``filings`` with the profile ``label``; returns the index, the
    mean seconds per filing and the number of chunks."""
    options = PROFILES[label]()
    name = collection_name(model_name, "eval", label, "cpu")
    index = LocalIndex(root / label, collection=name, dim=embedder.spec.dim)
    chunks_total = 0
    started = time.perf_counter()
    for filing in filings:
        chunks: list[Chunk] = chunk_markdown(
            filing.text, options=options, title=filing.title, tokenizer=tokenizer
        )
        if not chunks:
            continue
        vectors = embedder.encode_passage([chunk.embedded_text for chunk in chunks])
        payload = {"title": filing.title, "doc_type": filing.form, "date": filing.date}
        index.add(filing.doc_id, chunks, vectors, payload, first_index=0)
        chunks_total += len(chunks)
    seconds = (time.perf_counter() - started) / len(filings)
    return index, seconds, chunks_total


def retrievals(
    index: LocalIndex,
    embedder: Embedder,
    *,
    k: int,
    candidates: int,
    rrf_k: int,
    thresholds: Sequence[float],
    keep_tops: Sequence[int],
    solvers: Sequence[str],
    time_limit: float,
) -> dict[str, tuple[Callable[[Question], list[Hit]], dict[str, Any]]]:
    """The retrievals to compare, by name, each with its parameters: dense, hybrid and one
    hybrid+eigenmind per combination of the sweeps."""

    def dense(question: Question) -> list[Hit]:
        return search(
            question.question,
            collection=index.collection,
            store=index.store,
            embedder=embedder,
            filters=question.filters,
            limit=k,
            max_limit=max(k, candidates),
        )

    def hybrid(question: Question, selector: Any = None) -> list[Hit]:
        return hybrid_search(
            question.question,
            index=index,
            embedder=embedder,
            filters=question.filters,
            limit=k,
            candidates=candidates,
            rrf_k=rrf_k,
            selector=selector,
        )

    result: dict[str, tuple[Callable[[Question], list[Hit]], dict[str, Any]]] = {
        "dense": (dense, {}),
        "hybrid": (hybrid, {"candidates": candidates, "rrf_k": rrf_k}),
    }
    for threshold, keep_top, solver in itertools.product(thresholds, keep_tops, solvers):
        params = {"threshold": threshold, "keep_top": keep_top, "solver": solver, "time_limit": time_limit}
        selector = partial(
            select_diverse, threshold=threshold, keep_top=keep_top, solver=solver, time_limit=time_limit
        )
        name = f"hybrid+eigenmind[t={threshold:g},keep={keep_top},{solver}]"
        result[name] = (
            partial(hybrid, selector=selector),
            {"candidates": candidates, "rrf_k": rrf_k, **params},
        )
    return result


def run_matrix(
    filings: Sequence[Filing],
    questions: Sequence[Question],
    embedder: Embedder,
    tokenizer: Any,
    *,
    model_name: str,
    profiles: Sequence[str] = ("v1", "v2"),
    k: int = DEFAULT_K,
    candidates: int = DEFAULT_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
    thresholds: Sequence[float] = (DEFAULT_THRESHOLD,),
    keep_tops: Sequence[int] = (0,),
    solvers: Sequence[str] = ("milp",),
    time_limit: float = 2.0,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    interval_min_questions: int = DEFAULT_INTERVAL_MIN_QUESTIONS,
) -> list[Row]:
    """Build one index per profile in a temporary directory and score every retrieval on it."""
    rows: list[Row] = []
    with tempfile.TemporaryDirectory(prefix="filings_eval_") as tmp:
        for label in profiles:
            index, seconds, chunks = build_index(
                Path(tmp), filings, label=label, embedder=embedder, tokenizer=tokenizer, model_name=model_name
            )
            try:
                for name, (run, params) in retrievals(
                    index,
                    embedder,
                    k=k,
                    candidates=candidates,
                    rrf_k=rrf_k,
                    thresholds=thresholds,
                    keep_tops=keep_tops,
                    solvers=solvers,
                    time_limit=time_limit,
                ).items():
                    searches = 0

                    def timed(question: Question, run=run) -> list[Hit]:
                        nonlocal searches
                        searches += 1
                        return run(question)

                    started = time.perf_counter()
                    report = evaluate_passages(
                        questions,
                        timed,
                        k=k,
                        bootstrap_resamples=bootstrap_resamples,
                        interval_min_questions=interval_min_questions,
                    )
                    elapsed = time.perf_counter() - started
                    ms = 1000.0 * elapsed / max(searches, 1)
                    rows.append(Row(label, name, report, seconds, ms, chunks, params))
            finally:
                index.close()
    return rows


def _cell(report: PassageReport, metric: str) -> str:
    value = report.value(metric)
    interval = report.interval(metric)
    if interval is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{interval[0]:.2f},{interval[1]:.2f}]"


def format_table(rows: Sequence[Row]) -> str:
    """The rows as an aligned table: hit@5, hit@10, MRR@10 with intervals, cost, abstention tops."""
    header = ("chunking", "retrieval", "hit@5", "hit@10", "MRR@10", "s/10-K", "ms/search", "chunks")
    body = [
        (
            row.chunking,
            row.retrieval,
            _cell(row.report, "hit_at_5"),
            _cell(row.report, "hit_at_10"),
            _cell(row.report, "mrr_10"),
            f"{row.seconds_per_filing:.1f}",
            f"{row.ms_per_search:.0f}",
            str(row.chunks),
        )
        for row in rows
    ]
    widths = [max(len(line[i]) for line in (header, *body)) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True))
        for line in (header, *body)
    ]
    return "\n".join(lines)


def eigenmind_verdict(
    hybrid: PassageReport, eigenmind: PassageReport, metric: str = DECISION_METRIC
) -> tuple[bool, str]:
    """Whether eigenmind passes the default rule against ``hybrid``, and why.

    On only if its ``metric`` is higher than hybrid's **and** its interval lies wholly above
    hybrid's (no overlap); an interval missing (too few questions) means no.
    """
    a, b = hybrid.value(metric), eigenmind.value(metric)
    low_a, low_b = hybrid.interval(metric), eigenmind.interval(metric)
    if low_a is None or low_b is None:
        return False, f"undetermined: too few questions for an interval ({a:.3f} vs {b:.3f})"
    if b > a and low_b[0] > low_a[1]:
        return True, f"better with a non-overlapping interval ({a:.3f} -> {b:.3f})"
    if b > a:
        return False, f"better but the intervals overlap ({a:.3f} -> {b:.3f})"
    return False, f"not better ({a:.3f} -> {b:.3f})"


def decisions(rows: Sequence[Row]) -> list[str]:
    """The two rules of the plan, applied to ``rows``."""
    lines = []
    by = {(r.chunking, r.retrieval): r for r in rows}
    if ("v1", "hybrid") in by and ("v2", "hybrid") in by:
        v1 = by[("v1", "hybrid")].report.value(DECISION_METRIC)
        v2 = by[("v2", "hybrid")].report.value(DECISION_METRIC)
        lines.append(f"v2 >= v1 on hit@10 (hybrid): {'PASS' if v2 >= v1 else 'FAIL'} ({v1:.3f} -> {v2:.3f})")
    for chunking in dict.fromkeys(r.chunking for r in rows):
        hybrid = by.get((chunking, "hybrid"))
        for row in rows:
            if row.chunking == chunking and hybrid is not None and "eigenmind" in row.retrieval:
                ok, why = eigenmind_verdict(hybrid.report, row.report)
                lines.append(f"eigenmind ON for {chunking} {row.retrieval}: {'YES' if ok else 'no'}, {why}")
    return lines


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("filings", type=Path, help="directory of <doc_id>.md filings with <doc_id>.json sidecars")
    p.add_argument("questions", type=Path, help="question file with expected_spans (JSON Lines)")
    p.add_argument("--backend", choices=("onnx", "fake"), default="onnx")
    p.add_argument("--model", default=Settings().model_name, help="embedding model (default: the engine's)")
    p.add_argument("--model-cache", default=None, help="directory of the model files")
    p.add_argument("--profiles", default="v1,v2", help="chunking profiles to compare (default: v1,v2)")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="passages read per question (default: 10)")
    p.add_argument(
        "--candidates", type=int, default=DEFAULT_CANDIDATES, help="candidates per side of the fusion"
    )
    p.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF constant")
    p.add_argument(
        "--sweep-threshold",
        type=_floats,
        default=[DEFAULT_THRESHOLD],
        help="select_diverse thresholds, e.g. 0.5,0.65,0.8",
    )
    p.add_argument(
        "--sweep-keep-top", type=_ints, default=[0], help="select_diverse keep_top values, e.g. 0,1,2"
    )
    p.add_argument("--sweep-solver", default="milp", help="solvers, e.g. milp,lp_greedy")
    p.add_argument("--time-limit", type=float, default=2.0, help="seconds of the MILP stage")
    p.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES, help="bootstrap draws")
    p.add_argument(
        "--min-questions",
        type=int,
        default=DEFAULT_INTERVAL_MIN_QUESTIONS,
        help="fewest questions for an interval",
    )
    p.add_argument("--json", type=Path, default=None, help="also write the rows as JSON to this file")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    filings = load_filings(args.filings)
    questions = load_questions(args.questions)
    if args.backend == "fake":
        embedder: Embedder = FakeEmbedder(_FAKE_SPEC)
        tokenizer: Any = _WordTokenizer()
        model_name = _FAKE_SPEC.name
    else:
        settings = Settings(backend="onnx", model_name=args.model, model_cache=args.model_cache)
        embedder = make_embedder(settings)
        tokenizer = load_tokenizer(args.model_cache, args.model)
        model_name = args.model
    rows = run_matrix(
        filings,
        questions,
        embedder,
        tokenizer,
        model_name=model_name,
        profiles=[p.strip() for p in args.profiles.split(",") if p.strip()],
        k=args.k,
        candidates=args.candidates,
        rrf_k=args.rrf_k,
        thresholds=args.sweep_threshold,
        keep_tops=args.sweep_keep_top,
        solvers=[s.strip() for s in args.sweep_solver.split(",") if s.strip()],
        time_limit=args.time_limit,
        bootstrap_resamples=args.resamples,
        interval_min_questions=args.min_questions,
    )
    print(f"{len(filings)} filings, {len(questions)} questions ({args.questions}), backend {args.backend}\n")
    print(format_table(rows))
    print()
    print("\n".join(decisions(rows)))
    if args.json:
        payload = [
            {"chunking": r.chunking, "retrieval": r.retrieval, "params": r.params, "chunks": r.chunks,
             "seconds_per_filing": r.seconds_per_filing, "ms_per_search": r.ms_per_search,
             "report": r.report.to_dict()}
            for r in rows
        ]  # fmt: skip
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
