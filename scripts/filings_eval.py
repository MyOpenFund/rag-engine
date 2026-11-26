#!/usr/bin/env python
"""Passage-level evaluation of the local index on SEC filings: chunking x retrieval.

For a directory of Markdown filings (``<doc_id>.md`` and a sidecar ``<doc_id>.json`` holding
``company`` or ``title``, ``form`` and ``date``) and a question file with ``expected_spans``
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
    python scripts/filings_eval.py FILINGS_DIR QUESTIONS --cuts document,section

``--cuts section`` chunks each Item section of a filing on its own, as the no-vault MCP indexes
(``myopenfund_mcp.filings.indexer.document_chunks``): the Markdown is cut at the boundaries the
sidecar's ``sections`` gives (``[{"name", "start", "end"}]``, character offsets into the Markdown,
the ``FilingDocument.sections`` of company-corpus), the text outside every section is cut at them
too, and each piece is chunked alone, so no chunk spans two sections. Its rows are labelled
``<profile>/section``; ``--cuts document`` (the default) chunks the whole Markdown at once.

``--backend fake`` embeds with a hash-based ``FakeEmbedder`` and counts tokens by words: it checks
the plumbing, its numbers mean nothing. Every threshold is an option; the sweeps
(``--sweep-threshold``, ``--sweep-keep-top``, ``--sweep-solver``) add one eigenmind row per
combination.
"""

from __future__ import annotations

import argparse
import hashlib
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
#: How a filing is cut before chunking: whole (``document``) or one Item section at a time (``section``).
CUTS = ("document", "section")
#: The depth of the passages scored (hit@5 and hit@10 are read from it).
DEFAULT_K = 10
#: The smallest ``k`` the table can be printed with: it reads hit@10.
MIN_K = 10
#: The metric the two decision rules read.
DECISION_METRIC = "hit_at_10"
#: The question attributes the breakdown tables split every row by.
BREAKDOWNS = ("difficulty", "language")
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
    #: ``(name, start, end)`` of each Item section (sidecar ``sections``), ``None`` when not given.
    sections: tuple[tuple[str, int, int], ...] | None = None


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


def filing_title(company: str, form: str, report_date: str | None, filed: str | None) -> str:
    """The title the no-vault MCP gives a filing's chunks, which is their context (embedded and
    FTS-indexed): ``<company> <form> for the period ended <report_date>, filed <filed>``, ISO
    dates, a missing date leaving its clause out. It mirrors ``myopenfund_mcp.filings.indexer.
    filing_title`` (ruling R40): keep the two in step, or the eval no longer measures what ships."""
    title = f"{company} {form}"
    if report_date:
        title += f" for the period ended {report_date}"
    if filed:
        title += f", filed {filed}"
    return title


def load_filings(directory: str | Path) -> list[Filing]:
    """Every ``<doc_id>.md`` of ``directory`` with its ``<doc_id>.json`` sidecar, sorted by id.

    The sidecar holds ``form`` and ``date`` (the filing date, ``YYYY-MM-DD``), both required, and
    ``company`` (the SEC name, e.g. ``AMAZON COM INC``) or ``title``. With ``company``, the title is
    built as the MCP builds it (``filing_title``, with the optional ``report_date``); a ``title``
    given too must equal it. It may hold ``sections`` (a list of ``{"name", "start", "end"}``,
    character offsets into the Markdown) for ``--cuts section``. ``ValueError`` names the file when
    one is missing or malformed, or the directory holds no filing.
    """
    root = Path(directory)
    filings = []
    for path in sorted(root.glob("*.md")):
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"{path}: no sidecar {sidecar.name} (company or title, form, date)")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        missing = [key for key in ("form", "date") if not isinstance(meta.get(key), str)]
        if missing:
            raise ValueError(f"{sidecar}: missing or not a string: {', '.join(missing)}")
        title = _title(sidecar, meta)
        text = path.read_text(encoding="utf-8")
        sections = _sections(sidecar, meta, len(text))
        filings.append(Filing(path.stem, text, title, meta["form"], meta["date"], sections))
    if not filings:
        raise ValueError(f"{root}: no Markdown filing")
    return filings


def _title(sidecar: Path, meta: dict[str, Any]) -> str:
    company, title = meta.get("company"), meta.get("title")
    if not isinstance(company, str):
        if isinstance(title, str):
            return title
        raise ValueError(f"{sidecar}: missing or not a string: title or company")
    report_date = meta.get("report_date") if isinstance(meta.get("report_date"), str) else None
    built = filing_title(company, meta["form"], report_date, meta["date"])
    if title is not None and title != built:
        raise ValueError(f"{sidecar}: title {title!r} is not the MCP title {built!r} of its company")
    return built


def _sections(sidecar: Path, meta: dict[str, Any], length: int) -> tuple[tuple[str, int, int], ...] | None:
    raw = meta.get("sections")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"{sidecar}: 'sections' must be a list of {{name, start, end}}")  # noqa: TRY004
    out = []
    for item in raw:
        ok = (
            isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and all(type(item.get(key)) is int for key in ("start", "end"))
        )
        if not ok or not 0 <= item["start"] <= item["end"] <= length:
            raise ValueError(f"{sidecar}: bad section {item!r} (name, 0 <= start <= end <= {length})")
        out.append((item["name"], item["start"], item["end"]))
    return tuple(out)


def section_pieces(text: str, sections: Sequence[tuple[str, int, int]]) -> list[tuple[str | None, str]]:
    """``text`` cut at the boundaries of ``sections``, as the no-vault MCP cuts a filing.

    The same rule as ``myopenfund_mcp.filings.indexer.document_chunks``: sections in order of
    start, each clipped to start after the previous one, the text before, between and after them
    kept as pieces without a name, blank pieces dropped.
    """
    pieces: list[tuple[str | None, int, int]] = []
    cursor = 0
    for name, start, end in sorted(sections, key=lambda s: s[1]):
        start, end = max(start, cursor), min(end, len(text))
        if start >= end:
            continue
        if start > cursor:
            pieces.append((None, cursor, start))
        pieces.append((name, start, end))
        cursor = end
    if cursor < len(text):
        pieces.append((None, cursor, len(text)))
    return [(name, text[start:end]) for name, start, end in pieces if text[start:end].strip()]


def filing_chunks(filing: Filing, *, options: ChunkingOptions, tokenizer: Any, cut: str) -> list[Chunk]:
    """The chunks of ``filing``: the whole Markdown at once (``document``) or section by section."""
    if cut == "document":
        return chunk_markdown(filing.text, options=options, title=filing.title, tokenizer=tokenizer)
    if filing.sections is None:
        raise ValueError(f"{filing.doc_id}: --cuts section needs 'sections' in the sidecar")
    chunks: list[Chunk] = []
    for _, piece in section_pieces(filing.text, filing.sections):
        chunks.extend(chunk_markdown(piece, options=options, title=filing.title, tokenizer=tokenizer))
    return chunks


def label_of(profile: str, cut: str) -> str:
    """The chunking label of a row: the profile, with ``/section`` for the per-section cut."""
    return profile if cut == "document" else f"{profile}/{cut}"


def build_index(
    root: Path,
    filings: Sequence[Filing],
    *,
    label: str,
    embedder: Embedder,
    tokenizer: Any,
    model_name: str,
    cut: str = "document",
) -> tuple[LocalIndex, float, int]:
    """Chunk, embed and write ``filings`` with the profile ``label`` and the ``cut``; returns the
    index, the mean seconds per filing and the number of chunks."""
    options = PROFILES[label]()
    name = collection_name(model_name, "eval", f"{label}-{cut}", "cpu")
    index = LocalIndex(root / f"{label}-{cut}", collection=name, dim=embedder.spec.dim)
    chunks_total = 0
    started = time.perf_counter()
    for filing in filings:
        chunks = filing_chunks(filing, options=options, tokenizer=tokenizer, cut=cut)
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


def check_inputs(
    questions: Sequence[Question],
    profiles: Sequence[str],
    k: int,
    cuts: Sequence[str] = ("document",),
    filings: Sequence[Filing] | None = None,
) -> None:
    """``ValueError`` before any work when the run could not finish or would be meaningless.

    Every profile and cut is known, ``k`` is at least :data:`MIN_K` (the table reads hit@5,
    hit@10 and MRR@10), every question that does not abstain has ``expected_spans``, and with
    the ``section`` cut every filing (when given) has its sections.
    """
    unknown = [p for p in profiles if p not in PROFILES]
    if unknown:
        raise ValueError(f"unknown chunking profile {', '.join(unknown)} (known: {', '.join(PROFILES)})")
    bad_cuts = [c for c in cuts if c not in CUTS]
    if bad_cuts or not cuts:
        raise ValueError(f"unknown or no cut {', '.join(bad_cuts)} (known: {', '.join(CUTS)})")
    if "section" in cuts and filings is not None:
        bare_filings = [f.doc_id for f in filings if f.sections is None]
        if bare_filings:
            raise ValueError(f"--cuts section: no 'sections' in the sidecar of {', '.join(bare_filings[:5])}")
    if not isinstance(k, int) or k < MIN_K:
        raise ValueError(f"k must be an integer >= {MIN_K} (the table reads hit@10), got {k!r}")
    bare = [q.id for q in questions if q.expected_behavior != "abstain" and not q.expected_spans]
    if bare:
        raise ValueError(
            f"questions without expected_spans: {', '.join(bare[:5])}{' ...' if len(bare) > 5 else ''}"
        )


def run_matrix(
    filings: Sequence[Filing],
    questions: Sequence[Question],
    embedder: Embedder,
    tokenizer: Any,
    *,
    model_name: str,
    profiles: Sequence[str] = ("v1", "v2"),
    cuts: Sequence[str] = ("document",),
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
    """Build one index per cut x profile in a temporary directory and score every retrieval on it."""
    check_inputs(questions, profiles, k, cuts, filings)
    # One encode before any timer: the first call of a model loads and warms it.
    embedder.encode_passage(["warm-up"])
    embedder.encode_query("warm-up")
    rows: list[Row] = []
    with tempfile.TemporaryDirectory(prefix="filings_eval_") as tmp:
        for cut, label in itertools.product(cuts, profiles):
            index, seconds, chunks = build_index(
                Path(tmp),
                filings,
                label=label,
                embedder=embedder,
                tokenizer=tokenizer,
                model_name=model_name,
                cut=cut,
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
                    started = time.perf_counter()
                    hits = {question.id: run(question) for question in questions}
                    ms = 1000.0 * (time.perf_counter() - started) / max(len(questions), 1)
                    report = evaluate_passages(
                        questions,
                        lambda question, hits=hits: hits[question.id],
                        k=k,
                        bootstrap_resamples=bootstrap_resamples,
                        interval_min_questions=interval_min_questions,
                    )
                    rows.append(Row(label_of(label, cut), name, report, seconds, ms, chunks, params))
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


def breakdown(report: PassageReport, attribute: str) -> dict[str, dict[str, Any]]:
    """hit@5, hit@10 and MRR@10 of ``report`` per value of ``attribute`` (``difficulty``,
    ``language``): ``{label: {"questions": n, "hit_at_5": ..., "hit_at_10": ..., "mrr_10": ...}}``.
    Groups are small: their values carry no interval and are read as indications."""
    return {
        label: {"questions": len(group.results)}
        | {metric: group.value(metric) for metric in ("hit_at_5", "hit_at_10", "mrr_10")}
        for label, group in report.groups(attribute).items()
    }


def format_breakdown(rows: Sequence[Row], attribute: str) -> str:
    """One line per row and value of ``attribute``: the questions of that group and their metrics."""
    header = ("chunking", "retrieval", attribute, "n", "hit@5", "hit@10", "MRR@10")
    body = [
        (row.chunking, row.retrieval, label, str(m["questions"]),
         f"{m['hit_at_5']:.3f}", f"{m['hit_at_10']:.3f}", f"{m['mrr_10']:.3f}")
        for row in rows
        for label, m in breakdown(row.report, attribute).items()
    ]  # fmt: skip
    widths = [max(len(line[i]) for line in (header, *body)) for i in range(len(header))]
    lines = (
        "  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True))
        for line in (header, *body)
    )
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
    """The two rules of the plan, applied to ``rows`` (the first once per cut)."""
    lines = []
    by = {(r.chunking, r.retrieval): r for r in rows}
    for cut in CUTS:
        a, b = label_of("v1", cut), label_of("v2", cut)
        if (a, "hybrid") in by and (b, "hybrid") in by:
            v1 = by[(a, "hybrid")].report.value(DECISION_METRIC)
            v2 = by[(b, "hybrid")].report.value(DECISION_METRIC)
            where = "" if cut == "document" else f", {cut} cut"
            verdict = "PASS" if v2 >= v1 else "FAIL"
            lines.append(f"v2 >= v1 on hit@10 (hybrid{where}): {verdict} ({v1:.3f} -> {v2:.3f})")
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
    p.add_argument(
        "--cuts",
        default="document",
        help="how filings are cut before chunking: document, section, or both (default: document)",
    )
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
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    cuts = [c.strip() for c in args.cuts.split(",") if c.strip()]
    check_inputs(questions, profiles, args.k, cuts, filings)
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
        profiles=profiles,
        cuts=cuts,
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
    questions_sha256 = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    print(
        f"{len(filings)} filings, {len(questions)} questions ({args.questions}, "
        f"sha256 {questions_sha256[:12]}), backend {args.backend}\n"
    )
    print(format_table(rows))
    for attribute in BREAKDOWNS:
        print(f"\nby {attribute} (no interval: small groups)\n")
        print(format_breakdown(rows, attribute))
    print()
    print("\n".join(decisions(rows)))
    if args.json:
        payload = [
            {"chunking": r.chunking, "retrieval": r.retrieval, "params": r.params, "chunks": r.chunks,
             "seconds_per_filing": r.seconds_per_filing, "ms_per_search": r.ms_per_search,
             "questions_sha256": questions_sha256, "report": r.report.to_dict(),
             **{f"by_{a}": breakdown(r.report, a) for a in BREAKDOWNS}}
            for r in rows
        ]  # fmt: skip
        args.json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
