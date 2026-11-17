"""Build the automatic "known item" questions from the manifests of a document corpus.

A known-item question asks for one document that is known to exist: the question is made
from the metadata of the document (its title, its bank, its type, its date), and the answer
is the document itself. Nothing has to be written or judged by hand, so the batch can be
large; it measures whether the search finds a document a user can already describe, not
whether it answers a question about the content.

Usage::

    python eval/make_known_item.py \\
        --manifests '/path/to/central-bank-corpus/data/manifest/*.jsonl' \\
        --doc-types A1 A2 A3 B1 E1 F1 --mime application/pdf \\
        --per-cell 12 --seed 1 --output eval/known_item_v1.jsonl

A manifest is JSON Lines, one document per line, with ``doc_id``, ``bank_code``,
``doc_type``, ``title``, ``date``, ``mime_type`` (and optionally ``date_precision``). The
output is a question file for ``rag-engine eval`` (see ``rag_engine.eval``); a summary of
what was built, skipped and found ambiguous is printed.

Sampling. The documents of the scope (the given doc types and MIME type, optionally the
given banks) are grouped in cells, one per (bank, doc type). In each cell the documents are
put in an order that depends only on the seed and on their ``doc_id``, and taken in that
order until ``--per-cell`` of them have given at least one new question. The same manifests and
seed always give the same file, whatever the order of the lines.

Kinds of question, written in the ``kind`` field:

``title``
    The title of the document as the manifest has it, on one line and without a trailing
    note about the file ("[PDF 108KB]").
``described``
    The bank, the type of document and the date, in words, in English and in French
    (``language``): "Federal Reserve FOMC minutes, 20 June 2012".
``described+filters``
    The English description again, with the filters a caller who knows these facts would
    set: the bank, the doc type, and a window of dates around the date.

What a question may say about a date. A manifest does not always know the day. A date
written ``YYYY-01-01`` is read as "some day of that year", and a date on the first of
another month as "some day of that month", unless the manifest says the date is precise
to the day (``date_precision: day``). A question never states more than that:

- a date known to the day is written in full ("20 June 2012");
- a date known to the month is written as a month ("June 2012"), and the question expects
  every document of the same bank and type dated in that month;
- a date known to the year only, or no date, is not written at all: the description uses
  the wording of the title instead ("Bank of England MPC minutes: Minutes of the MPC
  Meeting held on 9 and 10 April 2008"), and its filters carry no date window;
- a title that itself spells out a placeholder date (the stored ``2015-05-01`` of a
  document known to the month) is not used as a question.

The date is given as the manifest gives it, without saying what it is the date of: for some
banks it is the day of the meeting, for others the day of publication.

Ambiguity. A question is ambiguous when several documents of the scope fit it: the same
title, or the same bank, type and date. All of them are then expected (``expected_doc_ids``
lists them, and the question is tagged ``several-expected``), up to ``--max-expected``
documents; beyond that the question is not written. Both cases are counted in the summary.

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Any

# Bank code -> (name in an English phrase, name after a French noun: "... de la Banque du Canada").
BANKS: dict[str, tuple[str, str]] = {
    "ecb": ("European Central Bank", "de la Banque centrale européenne"),
    "us": ("Federal Reserve FOMC", "du FOMC de la Réserve fédérale"),
    "gb": ("Bank of England MPC", "du Comité de politique monétaire de la Banque d'Angleterre"),
    "au": ("Reserve Bank of Australia", "de la Banque de réserve d'Australie"),
    "ca": ("Bank of Canada", "de la Banque du Canada"),
    "jp": ("Bank of Japan", "de la Banque du Japon"),
}

# Doc type -> (English words, French words), after the taxonomy of the corpus.
DOC_TYPES: dict[str, tuple[str, str]] = {
    "A1": ("rate decision press release", "Communiqué de décision de taux"),
    "A2": ("monetary policy statement", "Déclaration de politique monétaire"),
    "A3": ("minutes", "Compte rendu de la réunion de politique monétaire"),
    "B1": ("press conference transcript", "Transcription de la conférence de presse"),
    "E1": ("monetary policy report", "Rapport sur la politique monétaire"),
    "F1": ("economic projections", "Projections économiques"),
}

MONTHS_EN = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MONTHS_FR = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)

KINDS = ("title", "described", "described+filters")
DEFAULT_MIME = "application/pdf"
DEFAULT_PER_CELL = 12
DEFAULT_SEED = 1
DEFAULT_DATE_WINDOW_DAYS = 45
DEFAULT_MAX_EXPECTED = 3

_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_FILE_NAME = re.compile(r"\S+\.(pdf|html?|docx?|xlsx?|pptx?|txt)", re.IGNORECASE)
_FILE_NOTE = re.compile(r"\s*\[\s*PDF[^\]]*\]\s*$", re.IGNORECASE)


class ManifestError(ValueError):
    """A manifest cannot be read as one JSON object per line with the fields a document needs."""


def read_manifests(patterns: list[str]) -> list[dict[str, Any]]:
    """The rows of the manifest files matching ``patterns`` (paths or glob patterns), files in name order."""
    paths: list[str] = []
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if not found:
            raise ManifestError(f"no manifest file matches {pattern!r}")
        paths.extend(path for path in found if path not in paths)
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ManifestError(f"{path}, line {number}: not valid JSON ({exc})") from None
                if not isinstance(row, dict):
                    raise ManifestError(f"{path}, line {number}: not a JSON object")
                for name in ("doc_id", "bank_code", "doc_type"):
                    if not isinstance(row.get(name), str) or not row[name]:
                        raise ManifestError(f"{path}, line {number}: {name!r} is missing or not a string")
                rows.append(row)
    return rows


def parsed_date(row: dict[str, Any]) -> datetime.date | None:
    """The ``date`` of a row when it is a full calendar date written ``YYYY-MM-DD``, else ``None``."""
    value = row.get("date")
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def date_precision(row: dict[str, Any]) -> str:
    """What the manifest really knows of the date of a row: ``day``, ``month``, ``year`` or ``none``.

    A ``date_precision`` field of the manifest (``day``, ``month`` or ``year``) is believed.
    Without it, 1 January stands for a year and the first of another month for a month.
    """
    date = parsed_date(row)
    if date is None:
        return "none"
    stated = row.get("date_precision")
    if stated in ("day", "month", "year"):
        return stated
    if date.month == 1 and date.day == 1:
        return "year"
    if date.day == 1:
        return "month"
    return "day"


def clean_title(row: dict[str, Any]) -> str:
    """The title on one line, without a trailing note about the file ("[PDF 108KB]")."""
    title = row.get("title")
    if not isinstance(title, str):
        return ""
    return _FILE_NOTE.sub("", " ".join(title.split())).strip()


def title_problem(row: dict[str, Any]) -> str | None:
    """Why the title of a row cannot be used in a question; ``None`` when it can."""
    title = clean_title(row)
    if not title:
        return "no title"
    if _FILE_NAME.fullmatch(title):
        return "the title is a file name"
    date = parsed_date(row)
    if date is not None and date_precision(row) != "day" and date.isoformat() in title:
        return "the title spells out a placeholder date"
    return None


def date_in_words(date: datetime.date, precision: str, language: str) -> str:
    """``date`` as a reader writes it, to the day or to the month."""
    months = MONTHS_FR if language == "fr" else MONTHS_EN
    month = f"{months[date.month - 1]} {date.year}"
    if precision == "month":
        return month
    day = "1er" if language == "fr" and date.day == 1 else str(date.day)
    return f"{day} {month}"


def _month_bounds(date: datetime.date) -> tuple[datetime.date, datetime.date]:
    first = date.replace(day=1)
    following = (first + datetime.timedelta(days=32)).replace(day=1)
    return first, following - datetime.timedelta(days=1)


def sampling_key(seed: int, doc_id: str) -> str:
    """The place of a document in the order of its cell: a hash of the seed and of its id."""
    return hashlib.sha256(f"{seed}\0{doc_id}".encode()).hexdigest()


class _Scope:
    """The documents of the scope, indexed by what a question can say of them."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.by_title: dict[str, list[str]] = defaultdict(list)
        self.by_day: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        self.by_month: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        self.by_cell_title: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for row in rows:
            cell = (row["bank_code"], row["doc_type"])
            title = clean_title(row)
            if title:
                self.by_title[title].append(row["doc_id"])
                self.by_cell_title[(*cell, title)].append(row["doc_id"])
            date, precision = parsed_date(row), date_precision(row)
            if date is not None and precision == "day":
                self.by_day[(*cell, date.isoformat())].append(row["doc_id"])
            if date is not None and precision in ("day", "month"):
                self.by_month[(*cell, date.isoformat()[:7])].append(row["doc_id"])


def _candidates(
    row: dict[str, Any], scope: _Scope, date_window_days: int
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """The questions one document can give, each with every document that fits it, and what cannot be built.

    Returns ``(questions, skipped)``: a question is a dict with ``kind``, ``language``,
    ``question``, ``fits`` (the ids of the documents of the scope that fit it) and optional
    ``filters``; ``skipped`` holds ``(kind, reason)`` pairs.
    """
    bank, doc_type, doc_id = row["bank_code"], row["doc_type"], row["doc_id"]
    questions: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    title = clean_title(row)
    problem = title_problem(row)

    if problem is None:
        questions.append(
            {"kind": "title", "language": None, "question": title, "fits": scope.by_title[title]}
        )
    else:
        skipped.append(("title", problem))

    reason = None
    if bank not in BANKS:
        reason = "no name for the bank"
    elif doc_type not in DOC_TYPES:
        reason = "no words for the doc type"
    date, precision = parsed_date(row), date_precision(row)
    if reason is None and precision in ("year", "none") and problem is not None:
        reason = f"no date known to the month, and {problem}"
    if reason is not None:
        skipped.extend((kind, reason) for kind in ("described", "described+filters"))
        return questions, skipped

    bank_en, bank_fr = BANKS[bank]
    type_en, type_fr = DOC_TYPES[doc_type]
    filters: dict[str, Any] = {"source_codes": [bank], "doc_types": [doc_type]}
    window = datetime.timedelta(days=date_window_days)
    if precision == "day":
        assert date is not None
        english = f"{bank_en} {type_en}, {date_in_words(date, 'day', 'en')}"
        french = f"{type_fr} {bank_fr}, {date_in_words(date, 'day', 'fr')}"
        fits = scope.by_day[(bank, doc_type, date.isoformat())]
        filters["date_from"] = (date - window).isoformat()
        filters["date_to"] = (date + window).isoformat()
    elif precision == "month":
        assert date is not None
        english = f"{bank_en} {type_en}, {date_in_words(date, 'month', 'en')}"
        french = f"{type_fr} {bank_fr}, {date_in_words(date, 'month', 'fr')}"
        fits = scope.by_month[(bank, doc_type, date.isoformat()[:7])]
        first, last = _month_bounds(date)
        filters["date_from"] = (first - window).isoformat()
        filters["date_to"] = (last + window).isoformat()
    else:
        english = f"{bank_en} {type_en}: {title}"
        french = f"{type_fr} {bank_fr} : {title}"
        fits = scope.by_cell_title[(bank, doc_type, title)]
    assert doc_id in fits
    questions.append({"kind": "described", "language": "en", "question": english, "fits": fits})
    questions.append({"kind": "described", "language": "fr", "question": french, "fits": fits})
    questions.append(
        {"kind": "described+filters", "language": "en", "question": english, "fits": fits, "filters": filters}
    )
    return questions, skipped


def _question_id(kind: str, language: str | None, doc_id: str) -> str:
    parts = ["ki", kind.replace("+", "-")]
    if language is not None:
        parts.append(language)
    return "-".join([*parts, doc_id])


def build(
    rows: list[dict[str, Any]],
    *,
    doc_types: list[str],
    mime: str | None = DEFAULT_MIME,
    banks: list[str] | None = None,
    per_cell: int = DEFAULT_PER_CELL,
    seed: int = DEFAULT_SEED,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
    max_expected: int = DEFAULT_MAX_EXPECTED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The questions for ``rows`` and the account of how they were built.

    ``doc_types`` (and ``mime``, ``banks`` when given) define the scope; ``per_cell`` is the
    number of documents taken per (bank, doc type), ``seed`` decides which; ``date_window_days``
    is the half-width of the date window of ``described+filters``; ``max_expected`` the
    largest number of documents an ambiguous question may expect. Returns ``(questions,
    summary)``: the questions as dicts ready to be written, one per line, and a dict of counts.
    """
    if per_cell < 1 or max_expected < 1 or date_window_days < 0:
        raise ValueError("per_cell and max_expected must be at least 1, and date_window_days at least 0")
    wanted_types, wanted_banks = set(doc_types), set(banks) if banks else None
    scope_rows: dict[str, dict[str, Any]] = {}
    repeated = 0
    for row in rows:
        if row["doc_type"] not in wanted_types:
            continue
        if mime is not None and row.get("mime_type") != mime:
            continue
        if wanted_banks is not None and row["bank_code"] not in wanted_banks:
            continue
        if row["doc_id"] in scope_rows:
            repeated += 1
            continue
        scope_rows[row["doc_id"]] = row
    scope = _Scope(list(scope_rows.values()))

    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scope.rows:
        cells[(row["bank_code"], row["doc_type"])].append(row)

    questions: list[dict[str, Any]] = []
    written: set[str] = set()
    skipped: Counter[tuple[str, str]] = Counter()
    several_expected: Counter[str] = Counter()
    too_ambiguous: Counter[str] = Counter()
    cell_table: list[dict[str, Any]] = []
    for cell in sorted(cells):
        members = sorted(cells[cell], key=lambda row: sampling_key(seed, row["doc_id"]))
        precisions = Counter(date_precision(row) for row in members)
        dates = [date for date in map(parsed_date, members) if date is not None]
        january_first = sum(1 for date in dates if (date.month, date.day) == (1, 1))
        first_of_month = sum(1 for date in dates if date.day == 1)
        taken = visited = 0
        for row in members:
            if taken == per_cell:
                break
            visited += 1
            candidates, not_built = _candidates(row, scope, date_window_days)
            skipped.update(not_built)
            gave_one = False
            for candidate in candidates:
                kind, fits = candidate["kind"], sorted(candidate["fits"])
                if len(fits) > max_expected:
                    too_ambiguous[kind] += 1
                    skipped[(kind, f"ambiguous: more than {max_expected} documents fit")] += 1
                    continue
                # One question per group of documents, whichever of them was drawn.
                identifier = _question_id(kind, candidate["language"], fits[0])
                if identifier in written:
                    continue
                written.add(identifier)
                gave_one = True
                if len(fits) > 1:
                    several_expected[kind] += 1
                question: dict[str, Any] = {
                    "id": identifier,
                    "question": candidate["question"],
                    "expected_doc_ids": fits,
                }
                if "filters" in candidate:
                    question["filters"] = candidate["filters"]
                question["kind"] = kind
                if candidate["language"] is not None:
                    question["language"] = candidate["language"]
                tags = [f"bank:{cell[0]}", f"doc_type:{cell[1]}", f"date:{date_precision(row)}"]
                if len(fits) > 1:
                    tags.append("several-expected")
                question["tags"] = tags
                questions.append(question)
            taken += gave_one
        cell_table.append(
            {
                "bank": cell[0],
                "doc_type": cell[1],
                "documents": len(members),
                "january_first": january_first,
                "first_of_month": first_of_month,
                "precision": {name: precisions.get(name, 0) for name in ("day", "month", "year", "none")},
                "visited": visited,
                "sampled": taken,
            }
        )

    def tagged(prefix: str) -> dict[str, int]:
        counts = Counter(tag[len(prefix) :] for q in questions for tag in q["tags"] if tag.startswith(prefix))
        return dict(sorted(counts.items()))

    def shared(groups: dict[Any, list[str]]) -> dict[str, int]:
        several = [members for members in groups.values() if len(members) > 1]
        return {"groups": len(several), "documents": sum(len(members) for members in several)}

    summary = {
        "scope_documents": len(scope.rows),
        "repeated_doc_ids": repeated,
        "scope_shared_day": shared(scope.by_day),
        "scope_shared_title": shared(scope.by_title),
        "cells": cell_table,
        "documents_visited": sum(entry["visited"] for entry in cell_table),
        "documents_sampled": sum(entry["sampled"] for entry in cell_table),
        "questions": len(questions),
        "by_kind": {kind: sum(1 for q in questions if q["kind"] == kind) for kind in KINDS},
        "by_language": dict(sorted(Counter(q.get("language", "(title)") for q in questions).items())),
        "by_bank": tagged("bank:"),
        "by_doc_type": tagged("doc_type:"),
        "by_date_precision": tagged("date:"),
        "several_expected": dict(sorted(several_expected.items())),
        "too_ambiguous": dict(sorted(too_ambiguous.items())),
        "skipped": {f"{kind}: {reason}": count for (kind, reason), count in sorted(skipped.items())},
    }
    return questions, summary


def format_summary(summary: dict[str, Any]) -> str:
    """The summary as text: the table of dates per cell, then the counts."""
    scope = (
        f"Scope: {summary['scope_documents']} documents in {len(summary['cells'])} cells "
        f"({summary['repeated_doc_ids']} repeated doc_id ignored)"
    )
    header = (
        f"{'bank':<6}{'type':<6}{'docs':>6}{'01-01':>7}{'share':>8}{'day 01':>8}"
        f"{'day':>7}{'month':>7}{'year':>6}{'none':>6}{'sampled':>9}"
    )
    lines = [
        scope,
        "",
        "Dates per cell: documents dated 1 January (and their share), documents dated the first of a",
        "month, then the precision each date is read with (day, month, year, none):",
        header,
    ]
    for cell in summary["cells"]:
        precision = cell["precision"]
        share = cell["january_first"] / cell["documents"]
        lines.append(
            f"{cell['bank']:<6}{cell['doc_type']:<6}{cell['documents']:>6}{cell['january_first']:>7}"
            f"{share:>8.1%}{cell['first_of_month']:>8}{precision['day']:>7}{precision['month']:>7}"
            f"{precision['year']:>6}{precision['none']:>6}{cell['sampled']:>9}"
        )
    day, title = summary["scope_shared_day"], summary["scope_shared_title"]
    lines.append("")
    lines.append(
        f"Ambiguity in the scope: {day['documents']} documents share their bank, type and day with "
        f"another ({day['groups']} groups); {title['documents']} share their title ({title['groups']} groups)"
    )
    lines.append("")
    lines.append(
        f"Questions: {summary['questions']} from {summary['documents_sampled']} documents "
        f"({summary['documents_visited']} looked at)"
    )
    for label, key in (
        ("by kind", "by_kind"),
        ("by language", "by_language"),
        ("by bank", "by_bank"),
        ("by doc type", "by_doc_type"),
        ("by date precision", "by_date_precision"),
        ("expecting several documents", "several_expected"),
        ("not written, too many documents fit", "too_ambiguous"),
    ):
        counts = ", ".join(f"{name} {count}" for name, count in summary[key].items()) or "none"
        lines.append(f"  {label}: {counts}")
    lines.append("Not built, among the documents looked at:")
    lines += [f"  {count:>4}  {reason}" for reason, count in summary["skipped"].items()] or ["  nothing"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build known-item questions for 'rag-engine eval' from the manifests of a corpus."
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        required=True,
        metavar="GLOB",
        help="manifest files (paths or glob patterns)",
    )
    parser.add_argument(
        "--doc-types", nargs="+", required=True, metavar="TYPE", help="doc types of the scope"
    )
    parser.add_argument(
        "--mime", default=DEFAULT_MIME, help=f"MIME type of the scope (default {DEFAULT_MIME}; 'any' for all)"
    )
    parser.add_argument("--banks", nargs="+", metavar="CODE", help="bank codes of the scope (default: all)")
    parser.add_argument(
        "--per-cell",
        type=int,
        default=DEFAULT_PER_CELL,
        metavar="N",
        help=f"documents per (bank, doc type) (default {DEFAULT_PER_CELL})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"sampling seed (default {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--date-window-days",
        type=int,
        default=DEFAULT_DATE_WINDOW_DAYS,
        metavar="DAYS",
        help=f"half-width of the date window of described+filters (default {DEFAULT_DATE_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--max-expected",
        type=int,
        default=DEFAULT_MAX_EXPECTED,
        metavar="N",
        help=f"most documents an ambiguous question may expect (default {DEFAULT_MAX_EXPECTED})",
    )
    parser.add_argument("--output", required=True, metavar="FILE", help="the question file to write")
    arguments = parser.parse_args(argv)
    try:
        rows = read_manifests(arguments.manifests)
        questions, summary = build(
            rows,
            doc_types=arguments.doc_types,
            mime=None if arguments.mime == "any" else arguments.mime,
            banks=arguments.banks,
            per_cell=arguments.per_cell,
            seed=arguments.seed,
            date_window_days=arguments.date_window_days,
            max_expected=arguments.max_expected,
        )
    except (ManifestError, ValueError, OSError) as exc:
        print(f"make_known_item: error: {exc}", file=sys.stderr)
        return 2
    if not questions:
        print("make_known_item: error: no question could be built for that scope", file=sys.stderr)
        return 2
    with open(arguments.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(json.dumps(question, ensure_ascii=False) + "\n" for question in questions)
    print(format_summary(summary))
    print(f"Wrote {len(questions)} questions to {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
