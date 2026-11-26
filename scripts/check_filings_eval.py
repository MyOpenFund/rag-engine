#!/usr/bin/env python
"""Check the ``expected_spans`` of a filings question file against the Markdown filings.

The rules of ``eval/filings_README.md`` ("How a question is checked"), applied mechanically:

- every ``expected_doc_ids`` of a question names a ``<doc_id>.md`` of ``FILINGS_DIR``;
- every span is at least ``--min-chars`` long once normalised and holds a letter (no dash-only or
  tiny span);
- every span is found in one of the question's expected filings, at most ``--max-per-filing``
  times in each, with the comparison of ``rag_engine.eval.span_hit`` (NFKD, typography folded,
  case-folded, white space collapsed);
- a span found in a filing the question does not expect is reported: it would be a hit for the
  wrong filing (boilerplate, or a figure repeated by the next year's 10-K);
- every expected filing holds at least one of the question's spans (a filing listed without a span
  could never give a hit);
- with ``--require-difficulty``, every question carries a ``difficulty``;
- an abstention's ``absent:<term>`` tag names a term found, as a whole word, in no filing, and an
  ``absent@<issuer>:<term>`` tag a term found in no filing of that issuer (doc ids
  ``<issuer>-...``): the search terms of the abstention, checked rather than asserted.

``--reachable v1,v2`` also chunks every expected filing with each named profile of
``scripts/filings_eval.py`` (cut ``--reach-cuts``, ``section`` by default, as the MCP ships) and
reports a question none of whose spans (with ``span_match: "all"``: none of the spans of one
group) is held whole by one chunk of its expected filings: no retrieval could score it. A
question tagged ``chunk-loss`` is expected to be unreachable (its answer sits in text the
chunking loses): it is never a problem, and a note says when it has become reachable. The
chunks need the model tokenizer (``--model-cache``, ``--model``); ``--word-tokenizer`` counts a
word as a token instead (tests, a quick look: chunk boundaries then differ from the real ones).

``--difficulty`` measures, for every answering question, the retrieval-difficulty factors of
``eval/filings_README.md`` (lexical overlap with the answering passage, year-template exposure, a
table answer, no year, ``span_match: "all"``), prints them, and reports a question whose
``difficulty`` is not the one the factors give.

``--report`` also prints, for every span, how often it occurs in each expected filing, in how many
other filings it occurs, and in how many other filings its *wording* occurs: the longest stretch of
the span between figures (at least 15 characters), the sentence a neighbouring year's 10-K repeats
with other numbers. That last count is the year-confusion exposure of the question.

Run it::

    python scripts/check_filings_eval.py eval/filings_v1.jsonl FILINGS_DIR

It prints one line per problem and a summary, and exits 1 when there is any problem.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_engine.eval import Question, _squash, load_questions

#: The shortest span accepted, in characters once normalised.
DEFAULT_MIN_CHARS = 20
#: The most times a span may appear in one expected filing ("once or twice").
DEFAULT_MAX_PER_FILING = 2
#: The shortest stretch of wording between figures the ``--report`` looks for in other filings.
WORDING_MIN_CHARS = 15
#: The tag of a question whose answer sits in text the chunking loses (unreachable by design).
CHUNK_LOSS_TAG = "chunk-loss"
#: The cut the reachability check uses unless told otherwise: the per-section cut the MCP ships.
DEFAULT_REACH_CUTS = ("section",)
#: A figure of a span: digits with their separators, an optional dollar sign and scale word.
_FIGURE = re.compile(r"\(?\$?\d[\d.,]*\)?(?: (?:billion|million))?%?")


@dataclass(frozen=True)
class Problem:
    """One failed rule: the question, the span (``None`` for a question-level problem), why."""

    question_id: str
    span: str | None
    reason: str

    def format(self) -> str:
        where = self.question_id if self.span is None else f"{self.question_id} {self.span!r}"
        return f"{where}: {self.reason}"


def load_markdown(directory: str | Path) -> dict[str, str]:
    """Every ``<doc_id>.md`` of ``directory``, normalised as ``span_hit`` compares, by doc id."""
    paths = sorted(Path(directory).glob("*.md"))
    texts = {path.stem: _squash(path.read_text(encoding="utf-8")) for path in paths}
    if not texts:
        raise ValueError(f"{directory}: no Markdown filing")
    return texts


def check_question(
    question: Question,
    texts: Mapping[str, str],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_per_filing: int = DEFAULT_MAX_PER_FILING,
) -> list[Problem]:
    """The problems of one question (empty when it passes)."""
    if question.expected_behavior == "abstain":
        return check_absent(question, texts)
    problems = [
        Problem(question.id, None, f"expected filing {doc_id!r} not in the directory")
        for doc_id in question.expected_doc_ids
        if doc_id not in texts
    ]
    if not question.expected_doc_ids:
        problems.append(Problem(question.id, None, "no expected_doc_ids"))
    if not question.expected_spans:
        problems.append(Problem(question.id, None, "no expected_spans"))
    expected = [doc_id for doc_id in question.expected_doc_ids if doc_id in texts]
    for span in question.expected_spans:
        needle = _squash(span)
        if len(needle) < min_chars or not any(ch.isalpha() for ch in needle):
            problems.append(Problem(question.id, span, f"too short or no letter (< {min_chars} characters)"))
            continue
        counts = {doc_id: texts[doc_id].count(needle) for doc_id in expected}
        if not any(counts.values()):
            problems.append(Problem(question.id, span, "not found in any expected filing"))
        problems.extend(
            Problem(question.id, span, f"found {n} times in {doc_id} (at most {max_per_filing})")
            for doc_id, n in counts.items()
            if n > max_per_filing
        )
        elsewhere = [doc_id for doc_id, text in texts.items() if doc_id not in counts and needle in text]
        if elsewhere:
            others = ", ".join(elsewhere)
            problems.append(Problem(question.id, span, f"also in filings not expected: {others}"))
    needles = [_squash(span) for span in question.expected_spans]
    problems.extend(
        Problem(question.id, None, f"expected filing {doc_id!r} holds none of the spans")
        for doc_id in expected
        if not any(needle and needle in texts[doc_id] for needle in needles)
    )
    return problems


def absent_terms(question: Question) -> list[tuple[str | None, str]]:
    """The ``(issuer or None, term)`` of the ``absent:`` and ``absent@<issuer>:`` tags of ``question``."""
    terms: list[tuple[str | None, str]] = []
    for tag in question.tags:
        if tag.startswith("absent:"):
            terms.append((None, tag[len("absent:") :]))
        elif tag.startswith("absent@") and ":" in tag:
            issuer, term = tag[len("absent@") :].split(":", 1)
            terms.append((issuer, term))
    return terms


def check_absent(question: Question, texts: Mapping[str, str]) -> list[Problem]:
    """The ``absent`` tags of an abstention whose term is found, as a whole word, where it must not be."""
    problems = []
    for issuer, term in absent_terms(question):
        needle = _squash(term)
        pattern = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)")
        if not needle:
            problems.append(Problem(question.id, None, f"blank absent term in {term!r}"))
            continue
        found = [
            doc_id
            for doc_id, text in texts.items()
            if (issuer is None or doc_id.startswith(f"{issuer}-")) and pattern.search(text)
        ]
        if found:
            where = "" if issuer is None else f" ({issuer} filings)"
            problems.append(
                Problem(question.id, None, f"absent term {term!r}{where} found in {', '.join(found)}")
            )
    return problems


def wording(span: str) -> str:
    """The longest stretch of ``span`` between figures, normalised; empty when every stretch is short."""
    parts = [part.strip(" ,.;") for part in _FIGURE.split(_squash(span))]
    longest = max(parts, key=len, default="")
    return longest if len(longest) >= WORDING_MIN_CHARS else ""


@dataclass(frozen=True)
class SpanOccurrence:
    """Where one span of one question occurs (``--report``)."""

    question_id: str
    span: str
    in_expected: dict[str, int]
    other_filings: tuple[str, ...]
    wording_filings: tuple[str, ...]

    def format(self) -> str:
        expected = ", ".join(f"{doc_id}x{n}" for doc_id, n in self.in_expected.items())
        others = ", ".join(self.other_filings) or "-"
        return (
            f"{self.question_id}: {self.span[:60]!r} expected [{expected}] "
            f"other filings {len(self.other_filings)} ({others}); "
            f"same wording in {len(self.wording_filings)} other filing(s)"
        )


def occurrences(questions: Sequence[Question], texts: Mapping[str, str]) -> list[SpanOccurrence]:
    """Every span of every answering question with where it, and its wording, occur."""
    result = []
    for question in questions:
        if question.expected_behavior == "abstain":
            continue
        expected = set(question.expected_doc_ids or ())
        for span in question.expected_spans:
            needle, words = _squash(span), wording(span)
            result.append(
                SpanOccurrence(
                    question.id,
                    span,
                    {doc_id: texts[doc_id].count(needle) for doc_id in sorted(expected) if doc_id in texts},
                    tuple(d for d, t in texts.items() if d not in expected and needle in t),
                    tuple(d for d, t in texts.items() if d not in expected and words and words in t),
                )
            )
    return result


def reachable(question: Question, chunks: Mapping[str, Sequence[str]]) -> tuple[bool, list[str]]:
    """Whether ``question`` can be scored a hit on ``chunks`` (chunk texts by doc id), and its
    spans held whole by no chunk of its expected filings.

    With ``span_match="any"`` one span held by one chunk is enough; with ``"all"`` every group
    needs one of its spans held by one chunk (not necessarily the same chunk for two groups).
    """
    texts = [_squash(text) for doc_id in question.expected_doc_ids or () for text in chunks.get(doc_id, ())]
    held = {span: any(_squash(span) in text for text in texts) for span in question.expected_spans}
    missing = [span for span, ok in held.items() if not ok]
    if question.span_match == "all":
        groups = question.span_groups or tuple((span,) for span in question.expected_spans)
        return all(any(held[span] for span in group) for group in groups), missing
    return any(held.values()), missing


def check_reachable(
    questions: Sequence[Question], chunks: Mapping[str, Sequence[str]], label: str
) -> tuple[list[Problem], list[str]]:
    """The unreachable questions of one chunking ``label`` as problems, and the notes: spans no chunk
    holds (one line each) and ``chunk-loss`` questions that have become reachable."""
    problems: list[Problem] = []
    notes: list[str] = []
    for question in questions:
        if question.expected_behavior == "abstain":
            continue
        ok, missing = reachable(question, chunks)
        lost = CHUNK_LOSS_TAG in question.tags
        notes.extend(f"{question.id} {span[:60]!r}: held by no {label} chunk" for span in missing)
        if not ok and not lost:
            problems.append(Problem(question.id, None, f"no {label} chunk holds a span (unreachable)"))
        elif ok and lost:
            notes.append(f"{question.id}: tagged {CHUNK_LOSS_TAG} but reachable with {label} chunks")
    return problems, notes


def chunk_texts(
    directory: str | Path, profile: str, cut: str, tokenizer: object, doc_ids: Sequence[str] | None = None
) -> dict[str, list[str]]:
    """The chunk texts of every filing of ``directory`` (or of ``doc_ids``), cut as ``filings_eval``
    cuts them for the run: the text a hit carries, without its context."""
    runner = _runner()
    if profile not in runner.PROFILES:
        raise ValueError(f"unknown chunking profile {profile!r} (known: {', '.join(runner.PROFILES)})")
    if cut not in runner.CUTS:
        raise ValueError(f"unknown cut {cut!r} (known: {', '.join(runner.CUTS)})")
    result: dict[str, list[str]] = {}
    for filing in runner.load_filings(directory):
        if doc_ids is not None and filing.doc_id not in doc_ids:
            continue
        chunks = runner.filing_chunks(
            filing, options=runner.PROFILES[profile](), tokenizer=tokenizer, cut=cut
        )
        result[filing.doc_id] = [chunk.text for chunk in chunks]
    return result


def _runner():
    """``scripts/filings_eval.py``, imported on demand (it needs the ingest extra)."""
    import importlib.util

    name = "filings_eval"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "filings_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


#: Lexical overlap bands of the difficulty criteria: high from the first, low under the second.
OVERLAP_HIGH, OVERLAP_LOW = 0.6, 0.35
#: Characters read on each side of a span for the overlap (about half a v2 chunk).
OVERLAP_WINDOW = 300
#: Share of a span's four-word shingles (figures masked) another filing must hold to repeat its wording.
SHINGLE_SHARE = 0.5
_STOP_WORDS = """
a an the of in on at to for and or did do does what how much many was were is are its it s by from that
which had has have with as any into during end year fiscal their this after before according annual
report whose le la les de des du d l un une et en au aux il elle t qu que quel quels quelle quelles
combien comment pourquoi est sont ses son sa sur dans pour par avait ont elles ils sans encore depuis
quand y ce cette fin entre
"""
_STOP = frozenset(_STOP_WORDS.split())
_ISSUERS = frozenset(("microsoft", "alphabet", "google", "amazon", "meta", "oracle"))
_WORD = re.compile(r"[a-zà-ÿ0-9]+")
_TOKEN = re.compile(r"[a-z]+|[$\d][\d.,]*")


def _content_words(text: str) -> set[str]:
    words = (w for w in _WORD.findall(text.lower()) if w not in _STOP and w not in _ISSUERS)
    return {
        re.sub(r"(ies|es|s|ed|ing|e)$", "", w) if len(w) > 4 else w
        for w in words
        if not re.fullmatch(r"20\d\d", w)
    }


def _shingles(text: str, n: int = 4) -> set[tuple[str, ...]]:
    tokens = ["#" if t[0] in "$0123456789" else t for t in _TOKEN.findall(text)]
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass(frozen=True)
class Difficulty:
    """The retrieval-difficulty factors of one question and the label they give."""

    overlap: float
    exposure: int
    table: bool
    no_year: bool
    multi: bool

    @property
    def factors(self) -> list[str]:
        found = []
        if self.overlap < OVERLAP_LOW:
            found.append("low overlap")
        if self.exposure >= 1 and not self.table:  # every table recurs yearly: counted once, as a table
            found.append(f"year template x{self.exposure}")
        if self.table:
            found.append("table")
        if self.no_year:
            found.append("no year")
        return found

    @property
    def label(self) -> str:
        if self.multi or len(self.factors) >= 2:
            return "hard"
        return "medium" if self.factors or self.overlap < OVERLAP_HIGH else "easy"


def difficulty(
    question: Question, texts: Mapping[str, str], shingles: Mapping[str, set] | None = None
) -> Difficulty:
    """The factors of an answering ``question``: the criteria of ``eval/filings_README.md``.

    Overlap: the share of the question's content words (stop words, issuer names and years left out,
    light stemming) within :data:`OVERLAP_WINDOW` characters of its best span. Exposure: how many
    filings of the issuer, not expected, repeat the wording of its first span (:data:`SHINGLE_SHARE`
    of its shingles, figures masked). Table: tagged ``table``. No year: no year in the question while
    the template recurs, or "latest". Multi: ``span_match`` is ``all``.
    """
    if shingles is None:
        shingles = {doc_id: _shingles(text) for doc_id, text in texts.items()}
    wanted = _content_words(question.question)
    expected = set(question.expected_doc_ids or ())
    overlap = 0.0
    for span in question.expected_spans:
        needle = _squash(span)
        for doc_id in expected & texts.keys():
            at = texts[doc_id].find(needle)
            if at >= 0 and wanted:
                window = texts[doc_id][max(0, at - OVERLAP_WINDOW) : at + len(needle) + OVERLAP_WINDOW]
                overlap = max(overlap, len(wanted & _content_words(window)) / len(wanted))
    first = _shingles(_squash(question.expected_spans[0])) if question.expected_spans else set()
    issuer = (question.expected_doc_ids or ("",))[0].split("-")[0]
    exposure = sum(
        1
        for doc_id in texts
        if first
        and doc_id not in expected
        and doc_id.startswith(f"{issuer}-")
        and len(first & shingles[doc_id]) / len(first) >= SHINGLE_SHARE
    )
    no_year = not re.search(r"20\d\d", question.question) and (
        exposure >= 1 or "latest" in question.question.lower()
    )
    return Difficulty(
        round(overlap, 2), exposure, "table" in question.tags, no_year, question.span_match == "all"
    )


def check(
    questions: Sequence[Question],
    texts: Mapping[str, str],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_per_filing: int = DEFAULT_MAX_PER_FILING,
) -> list[Problem]:
    """The problems of every question, in file order."""
    return [
        problem
        for question in questions
        for problem in check_question(question, texts, min_chars=min_chars, max_per_filing=max_per_filing)
    ]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("questions", type=Path, help="question file with expected_spans (JSON Lines)")
    p.add_argument("filings", type=Path, help="directory of <doc_id>.md filings")
    p.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help="shortest span accepted")
    p.add_argument(
        "--max-per-filing", type=int, default=DEFAULT_MAX_PER_FILING, help="most occurrences in one filing"
    )
    p.add_argument("--require-difficulty", action="store_true", help="every question must carry a difficulty")
    p.add_argument("--report", action="store_true", help="also print where every span and its wording occur")
    p.add_argument(
        "--difficulty",
        action="store_true",
        help="measure the difficulty factors; a label they contradict fails",
    )
    p.add_argument(
        "--reachable",
        default="",
        help="comma-separated chunking profiles (v1,v2): every question must have a span held by a chunk",
    )
    p.add_argument(
        "--reach-cuts",
        default=",".join(DEFAULT_REACH_CUTS),
        help="cuts of the reachability check (section,document)",
    )
    p.add_argument("--model", default=None, help="tokenizer model of the chunks (default: the engine's)")
    p.add_argument("--model-cache", default=None, help="directory of the model files")
    p.add_argument(
        "--word-tokenizer", action="store_true", help="count a word as a token (no model; boundaries differ)"
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    questions = load_questions(args.questions)
    texts = load_markdown(args.filings)
    problems = check(questions, texts, min_chars=args.min_chars, max_per_filing=args.max_per_filing)
    if args.require_difficulty:
        problems.extend(Problem(q.id, None, "no difficulty") for q in questions if q.difficulty is None)
    if args.report:
        for occurrence in occurrences(questions, texts):
            print(occurrence.format())
    if args.difficulty:
        shingles = {doc_id: _shingles(text) for doc_id, text in texts.items()}
        for question in questions:
            if question.expected_behavior == "abstain":
                continue
            rated = difficulty(question, texts, shingles)
            print(
                f"{question.id}: overlap {rated.overlap:.2f}, {', '.join(rated.factors) or 'no factor'}"
                f"{', all' if rated.multi else ''} -> {rated.label}"
            )
            if question.difficulty is not None and question.difficulty != rated.label:
                problems.append(
                    Problem(
                        question.id,
                        None,
                        f"difficulty {question.difficulty!r}, the factors give {rated.label!r}",
                    )
                )
    profiles = [p.strip() for p in args.reachable.split(",") if p.strip()]
    if profiles:
        runner = _runner()
        if args.word_tokenizer:
            tokenizer: object = runner._WordTokenizer()
        else:
            from rag_engine.chunking import load_tokenizer
            from rag_engine.config import Settings

            tokenizer = load_tokenizer(args.model_cache, args.model or Settings().model_name)
        wanted = sorted({d for q in questions for d in q.expected_doc_ids or ()})
        for profile in profiles:
            for cut in (c.strip() for c in args.reach_cuts.split(",") if c.strip()):
                label = runner.label_of(profile, cut)
                chunks = chunk_texts(args.filings, profile, cut, tokenizer, wanted)
                found, notes = check_reachable(questions, chunks, label)
                problems.extend(found)
                if args.report:
                    for note in notes:
                        print(note)
                else:
                    for note in notes:
                        if CHUNK_LOSS_TAG in note:
                            print(note)
    for problem in problems:
        print(problem.format())
    answering = [q for q in questions if q.expected_behavior != "abstain"]
    spans = sum(len(q.expected_spans) for q in answering)
    abstaining = len(questions) - len(answering)
    print(
        f"{len(questions)} questions ({len(answering)} answering, {abstaining} abstaining), "
        f"{spans} spans, {len(texts)} filings: {len(problems)} problem(s)"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
