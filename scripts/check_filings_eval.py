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
  wrong filing (boilerplate, or a figure repeated by the next year's 10-K).

Run it::

    python scripts/check_filings_eval.py eval/filings_v1.jsonl FILINGS_DIR

It prints one line per problem and a summary, and exits 1 when there is any problem.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_engine.eval import Question, _squash, load_questions

#: The shortest span accepted, in characters once normalised.
DEFAULT_MIN_CHARS = 20
#: The most times a span may appear in one expected filing ("once or twice").
DEFAULT_MAX_PER_FILING = 2


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
        return []
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
    return problems


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
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    questions = load_questions(args.questions)
    texts = load_markdown(args.filings)
    problems = check(questions, texts, min_chars=args.min_chars, max_per_filing=args.max_per_filing)
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
