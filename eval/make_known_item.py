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
    A real title: the wording the bank itself gives the document, on one line and without a
    trailing note about the file ("[PDF 108KB]").
``label``
    A catalogue label: a title made by the scraper from the type and the date of the
    document ("FOMC minutes 1997-11-12"), which is printed nowhere on the document.
    :func:`is_catalogue_label` holds the rule.
``described``
    The bank, the type of document and the date, in words, in English and in French
    (``language``): "Minutes of the Federal Reserve FOMC meeting of 20 June 2012".
``described+filters``
    The English description again, with the filters a caller who knows these facts would
    set: the bank, the doc type, and a window of dates around the date.

Wording. What a doc type is differs from bank to bank: the ECB documents of type ``B1`` are
the combined decisions and statement, not a transcript; the type ``E1`` is a report of the
Board to Congress at the Federal Reserve and staff projections at the ECB. The words are
therefore kept in one table per (bank, doc type), :data:`CELL_WORDING`, with a default per
doc type, :data:`TYPE_WORDING`, for a bank the table does not know.

What a question may say about a date. A manifest does not always know the day. A date
written ``YYYY-01-01`` is read as "some day of that year", and a date on the first of
another month as "some day of that month", unless the manifest says the date is precise
to the day (``date_precision: day``). A question never states more than that:

- a date known to the day is written in full ("20 June 2012");
- a date known to the month is written as a month ("June 2012"), and the question expects
  every document of the same bank and type dated in that month;
- a date known to the year only, or no date, is not written at all: the description uses
  the wording of the title instead ("Bank of England Monetary Policy Committee minutes:
  Minutes of the MPC Meeting held on 9 and 10 April 2008"), in English only since the title
  is English, and its filters carry no date window; when that title is a catalogue label
  there is nothing to describe the document with, and no description is written;
- a label that itself spells out a placeholder date (the stored ``2015-05-01`` of a
  document known to the month) is not used as a question.

What the date is the date of. :data:`DATE_MEANING` says it per (bank, doc type): the day of
the ``meeting`` (its last day), the day of ``publication``, the ``period`` an issue of a
report is named after, or ``unknown``. The wording follows: "... meeting of 20 June 2012",
"..., published in October 2025", and a bare "..., May 2015" for a period or an unknown
meaning. Each question is tagged ``date-meaning:<value>``, or ``date-meaning:not-stated``
when its text carries no date.

Ambiguity. A question is ambiguous when several documents of the scope fit it: the same
title, or the same bank, type and date. All of them are then expected (``expected_doc_ids``
lists them, and the question is tagged ``several-expected``), up to ``--max-expected``
documents; beyond that the question is not written. Both cases are counted in the summary.

Equivalent documents. The catalogue holds some texts twice, under two doc types:
:data:`COMBINED_DOCUMENTS` declares the types that are the combination of others (the ECB
``B1`` holds the ``A1`` decision and the ``A2`` statement of its day). A question about a
part also expects the combined document of the same day, and a question about the combined
document also expects its parts; such a question is tagged ``equivalent-documents``, and
the ``doc_types`` filter of ``described+filters`` admits the types of every expected
document. Equivalent documents do not count against ``--max-expected``.

Filters. ``described+filters`` is tagged with the number of documents of the scope that
pass its filters (``filter-candidates:1``, ``2-3``, ``4-10``, ``>10``): with few candidates
a hit among the first ten says nothing.

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
# Used by the default wording of a doc type; a cell of CELL_WORDING names its bank itself.
BANKS: dict[str, tuple[str, str]] = {
    "ecb": ("European Central Bank", "de la Banque centrale européenne"),
    "us": ("Federal Reserve", "de la Réserve fédérale"),
    "gb": ("Bank of England", "de la Banque d'Angleterre"),
    "au": ("Reserve Bank of Australia", "de la Banque de réserve d'Australie"),
    "ca": ("Bank of Canada", "de la Banque du Canada"),
    "jp": ("Bank of Japan", "de la Banque du Japon"),
}

# The words for a document: (English, French, agreement of the French past participle:
# "" masculine, "e" feminine, "s" and "es" plural, as in "publié", "publiée", "publiées").

# Doc type -> default words, after the taxonomy of the corpus, for a (bank, doc type) that
# CELL_WORDING does not hold. "{bank}" and "{de_bank}" stand for the names of BANKS.
TYPE_WORDING: dict[str, tuple[str, str, str]] = {
    "A1": ("{bank} rate decision press release", "Communiqué de décision de taux {de_bank}", ""),
    "A2": ("{bank} monetary policy statement", "Déclaration de politique monétaire {de_bank}", "e"),
    "A3": (
        "{bank} monetary policy meeting minutes",
        "Compte rendu de la réunion de politique monétaire {de_bank}",
        "",
    ),
    "B1": ("{bank} press conference transcript", "Transcription de la conférence de presse {de_bank}", "e"),
    "E1": ("{bank} monetary policy report", "Rapport sur la politique monétaire {de_bank}", ""),
    "F1": ("{bank} economic projections", "Projections économiques {de_bank}", "es"),
}

# (bank, doc type) -> the words for what the documents of that cell really are, established
# from the titles and the addresses of the manifests. A phrase of a cell whose date is a
# meeting date is written so that "of <date>" / "du <date>" can follow it.
CELL_WORDING: dict[tuple[str, str], tuple[str, str, str]] = {
    ("au", "A1"): (
        "Reserve Bank of Australia monetary policy decision media release",
        "Communiqué de décision de politique monétaire de la Banque de réserve d'Australie",
        "",
    ),
    ("au", "A3"): (
        "Minutes of the Reserve Bank of Australia Board monetary policy meeting",
        "Compte rendu de la réunion de politique monétaire du Conseil de la Banque de réserve d'Australie",
        "",
    ),
    ("au", "E1"): (
        "Reserve Bank of Australia Statement on Monetary Policy",
        (
            "Rapport trimestriel de politique monétaire (Statement on Monetary Policy) "
            "de la Banque de réserve d'Australie"
        ),
        "",
    ),
    ("ca", "A3"): (
        "Bank of Canada summary of Governing Council deliberations",
        "Résumé des délibérations du Conseil de direction de la Banque du Canada",
        "",
    ),
    ("ca", "E1"): (
        "Bank of Canada Monetary Policy Report",
        "Rapport sur la politique monétaire de la Banque du Canada",
        "",
    ),
    ("ecb", "A1"): (
        "European Central Bank monetary policy decisions",
        "Décisions de politique monétaire de la Banque centrale européenne",
        "es",
    ),
    ("ecb", "A2"): (
        "European Central Bank monetary policy statement at the press conference",
        "Déclaration de politique monétaire à la conférence de presse de la Banque centrale européenne",
        "e",
    ),
    ("ecb", "A3"): (
        "European Central Bank account of the monetary policy meeting",
        "Compte rendu de la réunion de politique monétaire de la Banque centrale européenne",
        "",
    ),
    ("ecb", "B1"): (
        "European Central Bank combined monetary policy decisions and statement",
        (
            "Document combiné des décisions et de la déclaration de politique monétaire "
            "de la Banque centrale européenne"
        ),
        "",
    ),
    ("ecb", "E1"): (
        "Eurosystem and ECB staff macroeconomic projections for the euro area",
        "Projections macroéconomiques pour la zone euro des services de l'Eurosystème et de la BCE",
        "es",
    ),
    ("gb", "A3"): (
        "Bank of England Monetary Policy Committee minutes",
        "Procès-verbal de la réunion du Comité de politique monétaire de la Banque d'Angleterre",
        "",
    ),
    ("gb", "E1"): (
        "Bank of England Monetary Policy Report",
        "Rapport sur la politique monétaire de la Banque d'Angleterre",
        "",
    ),
    ("jp", "A3"): (
        "Minutes of the Bank of Japan Monetary Policy Meeting",
        "Compte rendu de la réunion de politique monétaire de la Banque du Japon",
        "",
    ),
    ("us", "A2"): ("Federal Reserve FOMC statement", "Communiqué du FOMC de la Réserve fédérale", ""),
    ("us", "A3"): (
        "Minutes of the Federal Reserve FOMC meeting",
        "Compte rendu de la réunion du FOMC de la Réserve fédérale",
        "",
    ),
    ("us", "B1"): (
        "Transcript of the Federal Reserve FOMC press conference",
        "Transcription de la conférence de presse du FOMC de la Réserve fédérale",
        "e",
    ),
    ("us", "E1"): (
        "Federal Reserve Monetary Policy Report to the Congress",
        "Rapport sur la politique monétaire de la Réserve fédérale au Congrès",
        "",
    ),
    ("us", "F1"): (
        "Federal Reserve FOMC Summary of Economic Projections",
        "Résumé des projections économiques du FOMC de la Réserve fédérale",
        "",
    ),
}

#: What the ``date`` of a manifest row can be the date of.
DATE_MEANINGS = ("meeting", "publication", "period", "unknown")

# (bank, doc type) -> what the date of the documents of that cell is the date of:
# "meeting": the day of the policy meeting, its last day when it lasts two (a decision, a
#     statement or a press conference of that day carries it as well);
# "publication": the day, or the month, the document was published, later than the meeting
#     it reports when there is one;
# "period": the month an issue of a report is named after;
# "unknown": not established; a cell that is not in the table is "unknown" too.
# The evidence, from the manifests, is in eval/README.md.
DATE_MEANING: dict[tuple[str, str], str] = {
    ("au", "A1"): "publication",
    ("au", "A3"): "meeting",
    ("au", "E1"): "period",
    ("ca", "A3"): "publication",
    ("ca", "E1"): "period",
    ("ecb", "A1"): "meeting",
    ("ecb", "A2"): "meeting",
    ("ecb", "A3"): "publication",
    ("ecb", "B1"): "meeting",
    ("ecb", "E1"): "period",
    ("gb", "A3"): "meeting",
    ("gb", "E1"): "period",
    ("jp", "A3"): "meeting",
    ("us", "A2"): "meeting",
    ("us", "A3"): "meeting",
    ("us", "B1"): "meeting",
    ("us", "E1"): "publication",
    ("us", "F1"): "meeting",
}

# (bank, doc type of a combined document) -> the doc types of its parts. The combined
# document of a day holds the text of the documents of these types dated the same day, so
# each is an answer to a question about the other (see "Equivalent documents" above).
# ECB: the 33 documents of type B1 are the file "ecb.ds<date>.en.pdf", the combined monetary
# policy decisions (A1) and statement (A2) of a meeting day.
COMBINED_DOCUMENTS: dict[tuple[str, str], tuple[str, ...]] = {("ecb", "B1"): ("A1", "A2")}

# Bank -> the wordings the scraper gives a document it found no title for, each followed by
# the date of the document written YYYY-MM-DD or YYYY-MM. Such a "title" is a catalogue
# label: it is printed nowhere on the document. The banks that are not here (gb, jp) carry
# the wording of the bank itself.
LABEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "au": ("RBA monetary policy decision", "RBA board minutes", "RBA Statement on Monetary Policy"),
    "ca": ("BoC governing council deliberations", "BoC Monetary Policy Report"),
    "ecb": (
        "Monetary policy decision",
        "Monetary policy statement",
        "Monetary policy account",
        "ECB press conference",
        "ECB staff projections",
        "ECB E1",
    ),
    "us": (
        "FOMC statement",
        "FOMC minutes",
        "FOMC press conference transcript",
        "FOMC Monetary Policy Report",
        "Summary of Economic Projections",
    ),
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

KINDS = ("title", "label", "described", "described+filters")
DEFAULT_MIME = "application/pdf"
DEFAULT_PER_CELL = 12
DEFAULT_SEED = 1
DEFAULT_DATE_WINDOW_DAYS = 45
DEFAULT_MAX_EXPECTED = 3

_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_FILE_NAME = re.compile(r"\S+\.(pdf|html?|docx?|xlsx?|pptx?|txt)", re.IGNORECASE)
_FILE_NOTE = re.compile(r"\s*\[\s*PDF[^\]]*\]\s*$", re.IGNORECASE)
_ISO_ENDING = re.compile(r"(?:^|\s)[0-9]{4}-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12][0-9]|3[01]))?$")
_FRENCH_ONLY_TITLE = "only an English title describes the document"


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


def is_catalogue_label(bank: str, title: str) -> bool:
    """Whether ``title`` is a label made by the scraper, not a title the bank gave the document.

    The rule: the title ends, after a space, with a date written the ISO way, ``YYYY-MM-DD``
    or ``YYYY-MM`` (a month from 01 to 12). The scraper names a document it found no title
    for after its type and its date written that way (:data:`LABEL_PREFIXES` lists its
    wordings per bank, and :func:`has_label_prefix` tells them); no title of the catalogue
    given by a bank ends like that. The rule is applied to every bank, so that a wording the
    table does not list yet is still a label.
    """
    return bool(title) and _ISO_ENDING.search(title) is not None


def has_label_prefix(bank: str, title: str) -> bool:
    """Whether ``title`` is one of the label wordings of ``bank`` followed by an ISO date."""
    if not is_catalogue_label(bank, title):
        return False
    return _ISO_ENDING.sub("", title) in LABEL_PREFIXES.get(bank, ())


def wording(bank: str, doc_type: str) -> tuple[str, str, str] | None:
    """The words for the documents of a cell: ``(English, French, French agreement)``.

    Those of :data:`CELL_WORDING`, else the default of the doc type with the name of the
    bank; ``None`` when the doc type has no words or the bank no name.
    """
    cell = CELL_WORDING.get((bank, doc_type))
    if cell is not None:
        return cell
    if bank not in BANKS or doc_type not in TYPE_WORDING:
        return None
    english, french, agreement = TYPE_WORDING[doc_type]
    name, de_name = BANKS[bank]
    return english.format(bank=name), french.format(de_bank=de_name), agreement


def date_meaning(bank: str, doc_type: str) -> str:
    """What the date of the documents of a cell is the date of (:data:`DATE_MEANINGS`)."""
    return DATE_MEANING.get((bank, doc_type), "unknown")


def equivalent_types(bank: str, doc_type: str) -> tuple[str, ...]:
    """The doc types that hold the text of a document of ``doc_type`` on the same day.

    The parts of a combined document, or the combined documents a part belongs to
    (:data:`COMBINED_DOCUMENTS`). Two parts of one combined document are not equivalent to
    each other: they are different texts.
    """
    parts = COMBINED_DOCUMENTS.get((bank, doc_type))
    if parts is not None:
        return parts
    return tuple(
        combined
        for (owner, combined), parts in sorted(COMBINED_DOCUMENTS.items())
        if owner == bank and doc_type in parts
    )


def candidates_bucket(count: int) -> str:
    """The value of the ``filter-candidates`` tag for ``count`` documents passing the filters."""
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    return "4-10" if count <= 10 else ">10"


def date_in_words(date: datetime.date, precision: str, language: str) -> str:
    """``date`` as a reader writes it, to the day or to the month."""
    months = MONTHS_FR if language == "fr" else MONTHS_EN
    month = f"{months[date.month - 1]} {date.year}"
    if precision == "month":
        return month
    day = "1er" if language == "fr" and date.day == 1 else str(date.day)
    return f"{day} {month}"


def dated(
    english: str, french: str, agreement: str, date: datetime.date, precision: str, meaning: str
) -> tuple[str, str]:
    """A description in English and in French with its date, worded after what the date is.

    ``precision`` is ``day`` or ``month``; ``meaning`` one of :data:`DATE_MEANINGS`:

    - ``meeting``: "X of 20 June 2012", "Y du 20 juin 2012" ("Y de juin 2012" to the month);
    - ``publication``: "X, published on 7 April 2016" or "published in October 2025",
      "Y, publié le 7 avril 2016" or "publié en octobre 2025", the participle agreeing;
    - ``period`` and ``unknown``: the bare date after a comma, which asserts nothing.
    """
    in_english = date_in_words(date, precision, "en")
    in_french = date_in_words(date, precision, "fr")
    if meaning == "meeting":
        if precision == "month":
            elided = in_french[0] in "aeiouyéèêàâîôûh"
            return (
                f"{english} of {in_english}",
                f"{french} d'{in_french}" if elided else f"{french} de {in_french}",
            )
        return f"{english} of {in_english}", f"{french} du {in_french}"
    if meaning == "publication":
        if precision == "month":
            return f"{english}, published in {in_english}", f"{french}, publié{agreement} en {in_french}"
        return f"{english}, published on {in_english}", f"{french}, publié{agreement} le {in_french}"
    return f"{english}, {in_english}", f"{french}, {in_french}"


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
        self.by_id: dict[str, dict[str, Any]] = {row["doc_id"]: row for row in rows}
        self.by_title: dict[str, list[str]] = defaultdict(list)
        self.by_day: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        self.by_month: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        self.by_cell_title: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        self.dated: dict[tuple[str, str], list[datetime.date]] = defaultdict(list)
        self.cell_sizes: Counter[tuple[str, str]] = Counter()
        for row in rows:
            cell = (row["bank_code"], row["doc_type"])
            self.cell_sizes[cell] += 1
            title = clean_title(row)
            if title:
                self.by_title[title].append(row["doc_id"])
                self.by_cell_title[(*cell, title)].append(row["doc_id"])
            date, precision = parsed_date(row), date_precision(row)
            if date is not None:
                self.dated[cell].append(date)
            if date is not None and precision == "day":
                self.by_day[(*cell, date.isoformat())].append(row["doc_id"])
            if date is not None and precision in ("day", "month"):
                self.by_month[(*cell, date.isoformat()[:7])].append(row["doc_id"])

    def same_day_equivalents(self, doc_ids: list[str]) -> list[str]:
        """The documents of an equivalent type dated the same day as one of ``doc_ids``.

        Only a date known to the day has equivalents: of a month or a year, nothing says
        that two documents are of the same day.
        """
        found: set[str] = set()
        for doc_id in doc_ids:
            row = self.by_id[doc_id]
            date = parsed_date(row)
            if date is None or date_precision(row) != "day":
                continue
            bank = row["bank_code"]
            for doc_type in equivalent_types(bank, row["doc_type"]):
                found.update(self.by_day.get((bank, doc_type, date.isoformat()), ()))
        return sorted(found - set(doc_ids))

    def passing(self, filters: dict[str, Any]) -> int:
        """Number of documents of the scope that pass ``filters``, as the search applies them.

        The bank and the doc type must be among those listed; with a date bound, a document
        passes when its stored date is within the bounds, both included, and a document
        without a date does not pass.
        """
        first = datetime.date.fromisoformat(filters["date_from"]) if "date_from" in filters else None
        last = datetime.date.fromisoformat(filters["date_to"]) if "date_to" in filters else None
        count = 0
        for bank in filters["source_codes"]:
            for doc_type in filters["doc_types"]:
                if first is None and last is None:
                    count += self.cell_sizes[(bank, doc_type)]
                    continue
                count += sum(
                    1
                    for date in self.dated.get((bank, doc_type), ())
                    if (first is None or date >= first) and (last is None or date <= last)
                )
        return count


def _candidates(
    row: dict[str, Any], scope: _Scope, date_window_days: int
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """The questions one document can give, each with every document that fits it, and what cannot be built.

    Returns ``(questions, skipped)``: a question is a dict with ``kind``, ``language``,
    ``question``, ``fits`` (the ids of the documents of the scope that fit it),
    ``equivalents`` (the ids of the documents of an equivalent type that hold the same text),
    ``meaning`` (what the date written in the question is the date of, ``not-stated``
    without one) and optional ``filters``; ``skipped`` holds ``(kind, reason)`` pairs.
    """
    bank, doc_type, doc_id = row["bank_code"], row["doc_type"], row["doc_id"]
    questions: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    title = clean_title(row)
    problem = title_problem(row)
    label = is_catalogue_label(bank, title)
    cell_meaning = date_meaning(bank, doc_type)

    if problem is None:
        fits = scope.by_title[title]
        questions.append(
            {
                "kind": "label" if label else "title",
                "language": None,
                "question": title,
                "fits": fits,
                "equivalents": scope.same_day_equivalents(fits),
                # A label spells the stored date; a real title is the wording of the bank.
                "meaning": cell_meaning if label else "not-stated",
            }
        )
    else:
        skipped.append(("label" if label else "title", problem))

    reason = None
    words = wording(bank, doc_type)
    if bank not in BANKS:
        reason = "no name for the bank"
    elif words is None:
        reason = "no words for the doc type"
    date, precision = parsed_date(row), date_precision(row)
    if reason is None and precision in ("year", "none"):
        if problem is not None:
            reason = f"no date known to the month, and {problem}"
        elif label:
            # Describing the document by its label would only repeat the "label" question.
            reason = "no date known to the month, and the title is a catalogue label"
    if reason is not None or words is None:
        skipped.extend((kind, reason or "") for kind in ("described", "described+filters"))
        return questions, skipped

    type_en, type_fr, agreement = words
    window = datetime.timedelta(days=date_window_days)
    date_filters: dict[str, str] = {}
    french: str | None
    if precision in ("day", "month"):
        assert date is not None
        english, french = dated(type_en, type_fr, agreement, date, precision, cell_meaning)
        meaning = cell_meaning
        if precision == "day":
            index, key = scope.by_day, date.isoformat()
            first = last = date
        else:
            index, key = scope.by_month, date.isoformat()[:7]
            first, last = _month_bounds(date)
        fits = index[(bank, doc_type, key)]
        # The documents of an equivalent type that the same words describe: same day, or same month.
        equivalents = sorted(
            {
                found
                for other in equivalent_types(bank, doc_type)
                for found in index.get((bank, other, key), ())
            }
        )
        date_filters = {"date_from": (first - window).isoformat(), "date_to": (last + window).isoformat()}
    else:
        english, french = f"{type_en}: {title}", None  # an English title: no French question
        meaning = "not-stated"
        fits = scope.by_cell_title[(bank, doc_type, title)]
        equivalents = scope.same_day_equivalents(fits)
        skipped.append(("described (French)", _FRENCH_ONLY_TITLE))
    assert doc_id in fits
    # The filters admit the doc type of every expected document, the equivalent ones included.
    types = sorted({doc_type, *(scope.by_id[found]["doc_type"] for found in equivalents)})
    filters: dict[str, Any] = {"source_codes": [bank], "doc_types": types, **date_filters}
    common = {"fits": fits, "equivalents": equivalents, "meaning": meaning}
    questions.append({"kind": "described", "language": "en", "question": english, **common})
    if french is not None:
        questions.append({"kind": "described", "language": "fr", "question": french, **common})
    questions.append(
        {"kind": "described+filters", "language": "en", "question": english, "filters": filters, **common}
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
    largest number of documents of its own cell an ambiguous question may expect (the
    equivalent documents of another doc type come on top). Returns ``(questions, summary)``:
    the questions as dicts ready to be written, one per line, and a dict of counts.

    ``expected_doc_ids`` lists the documents that fit the question, sorted, then their
    equivalents, sorted. Tags, in this order: ``bank:``, ``doc_type:``, ``date:`` (the
    precision of the date), ``date-meaning:``, ``filter-candidates:`` (``described+filters``
    only), ``several-expected`` and ``equivalent-documents`` when they apply.
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
    equivalents_added: Counter[str] = Counter()
    too_ambiguous: Counter[str] = Counter()
    cell_table: list[dict[str, Any]] = []
    for cell in sorted(cells):
        members = sorted(cells[cell], key=lambda row: sampling_key(seed, row["doc_id"]))
        precisions = Counter(date_precision(row) for row in members)
        dates = [date for date in map(parsed_date, members) if date is not None]
        january_first = sum(1 for date in dates if (date.month, date.day) == (1, 1))
        first_of_month = sum(1 for date in dates if date.day == 1)
        titles = [clean_title(row) for row in members]
        labels = sum(1 for title in titles if is_catalogue_label(cell[0], title))
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
                equivalents = [found for found in candidate["equivalents"] if found not in fits]
                expected = [*fits, *equivalents]
                if len(expected) > 1:
                    several_expected[kind] += 1
                if equivalents:
                    equivalents_added[kind] += 1
                question: dict[str, Any] = {
                    "id": identifier,
                    "question": candidate["question"],
                    "expected_doc_ids": expected,
                }
                if "filters" in candidate:
                    question["filters"] = candidate["filters"]
                question["kind"] = kind
                if candidate["language"] is not None:
                    question["language"] = candidate["language"]
                tags = [
                    f"bank:{cell[0]}",
                    f"doc_type:{cell[1]}",
                    f"date:{date_precision(row)}",
                    f"date-meaning:{candidate['meaning']}",
                ]
                if "filters" in candidate:
                    tags.append(f"filter-candidates:{candidates_bucket(scope.passing(candidate['filters']))}")
                if len(expected) > 1:
                    tags.append("several-expected")
                if equivalents:
                    tags.append("equivalent-documents")
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
                "date_meaning": date_meaning(*cell),
                "titles": {
                    "label": labels,
                    "label_outside_the_table": labels
                    - sum(1 for title in titles if has_label_prefix(cell[0], title)),
                    "none": sum(1 for title in titles if not title),
                    "title": sum(1 for title in titles if title) - labels,
                },
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

    candidates_order = {name: place for place, name in enumerate(("1", "2-3", "4-10", ">10"))}
    by_candidates = tagged("filter-candidates:")
    summary = {
        "scope_documents": len(scope.rows),
        "repeated_doc_ids": repeated,
        "scope_shared_day": shared(scope.by_day),
        "scope_shared_title": shared(scope.by_title),
        "scope_with_equivalents": sum(1 for row in scope.rows if scope.same_day_equivalents([row["doc_id"]])),
        "cells": cell_table,
        "documents_visited": sum(entry["visited"] for entry in cell_table),
        "documents_sampled": sum(entry["sampled"] for entry in cell_table),
        "questions": len(questions),
        "by_kind": {kind: sum(1 for q in questions if q["kind"] == kind) for kind in KINDS},
        "by_language": dict(sorted(Counter(q.get("language", "(title)") for q in questions).items())),
        "by_bank": tagged("bank:"),
        "by_doc_type": tagged("doc_type:"),
        "by_date_precision": tagged("date:"),
        "by_date_meaning": tagged("date-meaning:"),
        "by_filter_candidates": dict(
            sorted(by_candidates.items(), key=lambda item: candidates_order[item[0]])
        ),
        "several_expected": dict(sorted(several_expected.items())),
        "equivalents_added": dict(sorted(equivalents_added.items())),
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
        f"{'day':>7}{'month':>7}{'year':>6}{'none':>6}{'sampled':>9}  {'date is':<12}"
        f"{'titles':>7}{'labels':>8}"
    )
    lines = [
        scope,
        "",
        "Dates per cell: documents dated 1 January (and their share), documents dated the first of a",
        "month, the precision each date is read with (day, month, year, none), what the date is the date",
        "of, and how many titles are real titles and catalogue labels:",
        header,
    ]
    for cell in summary["cells"]:
        precision = cell["precision"]
        share = cell["january_first"] / cell["documents"]
        lines.append(
            f"{cell['bank']:<6}{cell['doc_type']:<6}{cell['documents']:>6}{cell['january_first']:>7}"
            f"{share:>8.1%}{cell['first_of_month']:>8}{precision['day']:>7}{precision['month']:>7}"
            f"{precision['year']:>6}{precision['none']:>6}{cell['sampled']:>9}  {cell['date_meaning']:<12}"
            f"{cell['titles']['title']:>7}{cell['titles']['label']:>8}"
        )
    day, title = summary["scope_shared_day"], summary["scope_shared_title"]
    lines.append("")
    lines.append(
        f"Ambiguity in the scope: {day['documents']} documents share their bank, type and day with "
        f"another ({day['groups']} groups); {title['documents']} share their title ({title['groups']} groups)"
    )
    outside = sum(cell["titles"]["label_outside_the_table"] for cell in summary["cells"])
    lines.append(
        f"Equivalent documents: {summary['scope_with_equivalents']} documents of the scope have the same "
        "text under another doc type on the same day"
    )
    lines.append(
        f"Catalogue labels: {sum(cell['titles']['label'] for cell in summary['cells'])} titles end with an "
        f"ISO date, {outside} of them with a wording that LABEL_PREFIXES does not list"
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
        ("by date meaning", "by_date_meaning"),
        ("by documents passing the filters", "by_filter_candidates"),
        ("expecting several documents", "several_expected"),
        ("of which with equivalent documents", "equivalents_added"),
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
