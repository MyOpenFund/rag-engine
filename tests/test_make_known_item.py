"""``eval/make_known_item.py``: the automatic questions, built from manifest rows.

The rows of these tests are imperfect on purpose: placeholder dates, missing dates and
titles, file names as titles, documents that share a date or a title, a bank the script
has no name for, cells smaller than the sample.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

from rag_engine.eval import load_questions

EVAL = Path(__file__).resolve().parent.parent / "eval"
SCRIPT = EVAL / "make_known_item.py"
KNOWN_ITEM = EVAL / "known_item_v1.jsonl"


def _load():
    spec = importlib.util.spec_from_file_location("make_known_item", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mki = _load()


def row(doc_id, bank="us", doc_type="A3", date="2012-06-20", title=..., **extra):
    title = f"Title of {doc_id}" if title is ... else title
    return {
        "doc_id": doc_id,
        "bank_code": bank,
        "doc_type": doc_type,
        "date": date,
        "title": title,
        "mime_type": "application/pdf",
        "language": "en",
        **extra,
    }


def build(rows, **kwargs):
    kwargs.setdefault("doc_types", ["A1", "A2", "A3", "B1", "E1", "F1"])
    return mki.build(rows, **kwargs)


def by_id(questions):
    return {question["id"]: question for question in questions}


# ---------------------------------------------------------------------------
# What the manifest knows of a date
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("date", "extra", "precision"),
    [
        ("2012-06-20", {}, "day"),
        ("2012-06-01", {}, "month"),
        ("2012-01-01", {}, "year"),
        ("2012-01-02", {}, "day"),
        ("2012-06-01", {"date_precision": "day"}, "day"),
        ("2012-01-01", {"date_precision": "day"}, "day"),
        ("2012-06-20", {"date_precision": "month"}, "month"),
        ("2012-06-20", {"date_precision": "year"}, "year"),
        ("2012-06-01", {"date_precision": "unknown"}, "month"),
        (None, {}, "none"),
        ("", {}, "none"),
        ("2012", {}, "none"),
        ("2012-06", {}, "none"),
        ("2012-02-30", {}, "none"),
        ("2012-06-20T10:00:00", {}, "none"),
        (20120620, {}, "none"),
    ],
)
def test_date_precision(date, extra, precision):
    assert mki.date_precision(row("d", date=date, **extra)) == precision


def test_dates_in_words():
    import datetime

    day = datetime.date(2012, 6, 20)
    first = datetime.date(2024, 3, 1)
    assert mki.date_in_words(day, "day", "en") == "20 June 2012"
    assert mki.date_in_words(day, "day", "fr") == "20 juin 2012"
    assert mki.date_in_words(day, "month", "en") == "June 2012"
    assert mki.date_in_words(day, "month", "fr") == "juin 2012"
    assert mki.date_in_words(first, "day", "en") == "1 March 2024"
    assert mki.date_in_words(first, "day", "fr") == "1er mars 2024"
    assert mki.date_in_words(datetime.date(2001, 8, 21), "day", "fr") == "21 août 2001"


@pytest.mark.parametrize(
    ("title", "date", "problem"),
    [
        ("FOMC minutes 2012-06-20", "2012-06-20", None),
        ("", "2012-06-20", "no title"),
        ("   ", "2012-06-20", "no title"),
        (None, "2012-06-20", "no title"),
        ("may-2018.pdf", "2018-01-01", "the title is a file name"),
        ("Report.PDF", "2018-01-01", "the title is a file name"),
        ("Minutes of 9 May 2018.pdf", "2018-01-01", None),
        (
            "RBA Statement on Monetary Policy 2015-05-01",
            "2015-05-01",
            "the title spells out a placeholder date",
        ),
        ("Report 2015-01-01", "2015-01-01", "the title spells out a placeholder date"),
        ("BoC Monetary Policy Report 2017-01", "2017-01-01", None),
        ("[PDF 108KB]", "2017-03-02", "no title"),
    ],
)
def test_title_problem(title, date, problem):
    assert mki.title_problem(row("d", title=title, date=date)) == problem


def test_a_title_dated_the_first_is_kept_when_the_manifest_says_the_day_is_known():
    known = row("d", title="Decision 2015-05-01", date="2015-05-01", date_precision="day")
    assert mki.title_problem(known) is None


def test_clean_title_drops_the_file_note_and_odd_spaces():
    noisy = row("d", title="Meeting on  April 26 and 27, 2017 [PDF 108KB]")
    assert mki.clean_title(noisy) == "Meeting on April 26 and 27, 2017"


# ---------------------------------------------------------------------------
# Real titles and catalogue labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("bank", "title", "label"),
    [
        ("us", "FOMC minutes 1997-11-12", True),
        ("us", "FOMC Monetary Policy Report 2009-02-24", True),
        ("us", "Summary of Economic Projections 2010-06-23", True),
        ("ca", "BoC Monetary Policy Report 2022-04", True),
        ("ca", "BoC governing council deliberations 2025-10", True),
        ("ecb", "ECB E1 2018-06-01", True),
        ("ecb", "ECB press conference 2022-06-09", True),
        ("au", "RBA Statement on Monetary Policy 2015-05-01", True),
        # A bank without patterns: a title that ends with an ISO date is a label all the same.
        ("zz", "Policy rate decision 2020-03-04", True),
        ("gb", "Something 2020-03", True),
        ("gb", "Minutes of the MPC Meeting held on 9 and 10 April 2008", False),
        ("gb", "Monetary Policy Report - November 2020", False),
        ("jp", "Meeting on April 26 and 27, 2017", False),
        ("us", "Minutes of the Federal Open Market Committee, June 19-20, 2012", False),
        ("us", "FOMC minutes of 2012", False),
        ("us", "The outlook for 2020-2021", False),
        ("us", "", False),
    ],
)
def test_catalogue_labels_are_told_from_real_titles(bank, title, label):
    assert mki.is_catalogue_label(bank, title) is label


def test_every_label_pattern_ends_with_an_iso_date_so_the_general_rule_covers_it():
    for bank, prefixes in mki.LABEL_PREFIXES.items():
        for prefix in prefixes:
            assert mki.is_catalogue_label(bank, f"{prefix} 2012-06-20")
            assert mki.is_catalogue_label(bank, f"{prefix} 2012-06")
            assert not mki.is_catalogue_label(bank, prefix)


def test_a_label_and_a_real_title_give_two_kinds():
    rows = [
        row("lab", title="FOMC minutes 2012-06-20"),
        row("real", date="2013-01-30", title="Minutes of the Federal Open Market Committee, January 2013"),
    ]
    found = by_id(build(rows)[0])
    assert found["ki-label-lab"]["kind"] == "label"
    assert found["ki-label-lab"]["question"] == "FOMC minutes 2012-06-20"
    assert found["ki-title-real"]["kind"] == "title"
    assert "ki-title-lab" not in found and "ki-label-real" not in found
    # A label spells the stored date, whose meaning is the one of its cell; a real title is kept as it is.
    assert "date-meaning:meeting" in found["ki-label-lab"]["tags"]
    assert "date-meaning:not-stated" in found["ki-title-real"]["tags"]


# ---------------------------------------------------------------------------
# Wording: one table per (bank, doc type), a default per doc type
# ---------------------------------------------------------------------------
def test_the_wording_tables_are_complete_and_consistent():
    for cell, (english, french, agreement) in mki.CELL_WORDING.items():
        assert cell[0] in mki.BANKS and cell[1] in mki.TYPE_WORDING, cell
        assert english and french and agreement in ("", "e", "s", "es"), cell
        assert "{" not in english and "{" not in french, cell
        assert cell in mki.DATE_MEANING, cell
    for doc_type, (english, french, agreement) in mki.TYPE_WORDING.items():
        assert "{bank}" in english and "{de_bank}" in french, doc_type
        assert agreement in ("", "e", "s", "es")
    assert set(mki.DATE_MEANING.values()) <= set(mki.DATE_MEANINGS)
    assert mki.DATE_MEANINGS == ("meeting", "publication", "period", "unknown")


@pytest.mark.parametrize(
    ("bank", "doc_type", "english", "french"),
    [
        (
            "ecb",
            "B1",
            "European Central Bank combined monetary policy decisions and statement of 9 June 2022",
            (
                "Document combiné des décisions et de la déclaration de politique monétaire de la Banque "
                "centrale européenne du 9 juin 2022"
            ),
        ),
        (
            "us",
            "B1",
            "Transcript of the Federal Reserve FOMC press conference of 9 June 2022",
            "Transcription de la conférence de presse du FOMC de la Réserve fédérale du 9 juin 2022",
        ),
        (
            "us",
            "E1",
            "Federal Reserve Monetary Policy Report to the Congress, published on 9 June 2022",
            "Rapport sur la politique monétaire de la Réserve fédérale au Congrès, publié le 9 juin 2022",
        ),
        (
            "ecb",
            "A3",
            "European Central Bank account of the monetary policy meeting, published on 9 June 2022",
            (
                "Compte rendu de la réunion de politique monétaire de la Banque centrale européenne, "
                "publié le 9 juin 2022"
            ),
        ),
        (
            "ecb",
            "A1",
            "European Central Bank monetary policy decisions of 9 June 2022",
            "Décisions de politique monétaire de la Banque centrale européenne du 9 juin 2022",
        ),
        (
            "au",
            "A1",
            "Reserve Bank of Australia monetary policy decision media release, published on 9 June 2022",
            (
                "Communiqué de décision de politique monétaire de la Banque de réserve d'Australie, "
                "publié le 9 juin 2022"
            ),
        ),
        # A cell the table does not know: the default of the doc type, and a bare date.
        (
            "jp",
            "A1",
            "Bank of Japan rate decision press release, 9 June 2022",
            "Communiqué de décision de taux de la Banque du Japon, 9 juin 2022",
        ),
        (
            "gb",
            "B1",
            "Bank of England press conference transcript, 9 June 2022",
            "Transcription de la conférence de presse de la Banque d'Angleterre, 9 juin 2022",
        ),
    ],
)
def test_the_words_of_a_doc_type_depend_on_the_bank(bank, doc_type, english, french):
    found = by_id(build([row("d1", bank=bank, doc_type=doc_type, date="2022-06-09")])[0])
    assert found["ki-described-en-d1"]["question"] == english
    assert found["ki-described-fr-d1"]["question"] == french
    assert found["ki-described-filters-en-d1"]["question"] == english


def test_ecb_projections_and_the_report_to_congress_are_not_called_fomc_or_policy_reports():
    ecb = mki.CELL_WORDING[("ecb", "E1")]
    assert "staff macroeconomic projections" in ecb[0] and "report" not in ecb[0].lower()
    assert "press conference" not in mki.CELL_WORDING[("ecb", "B1")][0]
    assert "FOMC" not in mki.CELL_WORDING[("us", "E1")][0]
    assert "FOMC" not in mki.CELL_WORDING[("us", "E1")][1]


@pytest.mark.parametrize(
    ("meaning", "agreement", "precision", "english", "french"),
    [
        ("meeting", "", "day", "X of 1 April 2012", "Y du 1er avril 2012"),
        ("meeting", "", "month", "X of April 2012", "Y d'avril 2012"),
        ("publication", "", "day", "X, published on 1 April 2012", "Y, publié le 1er avril 2012"),
        ("publication", "e", "month", "X, published in April 2012", "Y, publiée en avril 2012"),
        ("publication", "es", "month", "X, published in April 2012", "Y, publiées en avril 2012"),
        ("period", "", "month", "X, April 2012", "Y, avril 2012"),
        ("period", "", "day", "X, 1 April 2012", "Y, 1er avril 2012"),
        ("unknown", "e", "day", "X, 1 April 2012", "Y, 1er avril 2012"),
    ],
)
def test_the_date_is_worded_after_its_meaning(meaning, agreement, precision, english, french):
    import datetime

    date = datetime.date(2012, 4, 1)
    assert mki.dated("X", "Y", agreement, date, precision, meaning) == (english, french)


def test_french_months_are_elided_after_de():
    import datetime

    def month(number):
        return mki.dated("X", "Y", "", datetime.date(2012, number, 15), "month", "meeting")[1]

    assert month(4) == "Y d'avril 2012"
    assert month(8) == "Y d'août 2012"
    assert month(10) == "Y d'octobre 2012"
    assert month(6) == "Y de juin 2012"


def test_a_publication_date_is_never_presented_as_a_meeting_date():
    # The Bank of Canada summary dated October 2025 is the one of the September meeting.
    rows = [
        row(
            "d1",
            bank="ca",
            doc_type="A3",
            date="2025-10-01",
            title="BoC governing council deliberations 2025-10",
        )
    ]
    found = by_id(build(rows)[0])
    english, french = found["ki-described-en-d1"], found["ki-described-fr-d1"]
    assert english["question"] == (
        "Bank of Canada summary of Governing Council deliberations, published in October 2025"
    )
    assert french["question"] == (
        "Résumé des délibérations du Conseil de direction de la Banque du Canada, publié en octobre 2025"
    )
    assert "date-meaning:publication" in english["tags"] and "date-meaning:publication" in french["tags"]
    assert "meeting" not in english["question"].split(",")[1]
    assert mki.DATE_MEANING[("ca", "A3")] == mki.DATE_MEANING[("ecb", "A3")] == "publication"
    assert mki.DATE_MEANING[("us", "A3")] == mki.DATE_MEANING[("jp", "A3")] == "meeting"


# ---------------------------------------------------------------------------
# The questions of one document
# ---------------------------------------------------------------------------
def test_a_document_dated_to_the_day_gives_the_four_questions():
    questions, summary = build([row("d1", title="FOMC minutes 2012-06-20")])
    assert [question["id"] for question in questions] == [
        "ki-label-d1",
        "ki-described-en-d1",
        "ki-described-fr-d1",
        "ki-described-filters-en-d1",
    ]
    label, english, french, filtered = questions
    assert label == {
        "id": "ki-label-d1",
        "question": "FOMC minutes 2012-06-20",
        "expected_doc_ids": ["d1"],
        "kind": "label",
        "tags": ["bank:us", "doc_type:A3", "date:day", "date-meaning:meeting"],
    }
    assert english["question"] == "Minutes of the Federal Reserve FOMC meeting of 20 June 2012"
    assert (english["kind"], english["language"]) == ("described", "en")
    assert french["question"] == "Compte rendu de la réunion du FOMC de la Réserve fédérale du 20 juin 2012"
    assert (french["kind"], french["language"]) == ("described", "fr")
    assert filtered["question"] == english["question"]
    assert filtered["kind"] == "described+filters"
    assert filtered["filters"] == {
        "source_codes": ["us"],
        "doc_types": ["A3"],
        "date_from": "2012-05-06",
        "date_to": "2012-08-04",
    }
    assert filtered["tags"] == [
        "bank:us",
        "doc_type:A3",
        "date:day",
        "date-meaning:meeting",
        "filter-candidates:1",
    ]
    assert "filters" not in english and "filters" not in label
    assert list(filtered) == ["id", "question", "expected_doc_ids", "filters", "kind", "language", "tags"]
    assert summary["by_kind"] == {"title": 0, "label": 1, "described": 2, "described+filters": 1}
    assert summary["by_date_meaning"] == {"meeting": 4}
    assert summary["by_filter_candidates"] == {"1": 1}


def test_the_date_window_is_a_parameter():
    questions, _ = build([row("d1")], date_window_days=3)
    filters = by_id(questions)["ki-described-filters-en-d1"]["filters"]
    assert (filters["date_from"], filters["date_to"]) == ("2012-06-17", "2012-06-23")


def test_a_first_of_month_is_described_by_its_month_never_by_a_day():
    questions, _ = build(
        [row("d1", bank="au", doc_type="E1", date="2015-05-01", title="Statement, May 2015")]
    )
    found = by_id(questions)
    assert (
        found["ki-described-en-d1"]["question"]
        == "Reserve Bank of Australia Statement on Monetary Policy, May 2015"
    )
    assert found["ki-described-fr-d1"]["question"].endswith("de la Banque de réserve d'Australie, mai 2015")
    # The window covers the whole month, widened on both sides.
    filters = found["ki-described-filters-en-d1"]["filters"]
    assert (filters["date_from"], filters["date_to"]) == ("2015-03-17", "2015-07-15")
    assert found["ki-title-d1"]["tags"] == [
        "bank:au",
        "doc_type:E1",
        "date:month",
        "date-meaning:not-stated",
    ]
    assert "date-meaning:period" in found["ki-described-en-d1"]["tags"]
    assert not any(re.search(r"\b1(er)? (May|mai)\b", question["question"]) for question in questions)


def test_a_first_of_january_is_never_stated_and_the_title_is_used_instead():
    title = "Minutes of the MPC Meeting held on 9 and 10 April 2008"
    questions, summary = build([row("d1", bank="gb", date="2008-01-01", title=title)])
    found = by_id(questions)
    assert found["ki-title-d1"]["question"] == title
    assert (
        found["ki-described-en-d1"]["question"]
        == f"Bank of England Monetary Policy Committee minutes: {title}"
    )
    # No date is known: the filters carry the bank and the type only.
    assert found["ki-described-filters-en-d1"]["filters"] == {"source_codes": ["gb"], "doc_types": ["A3"]}
    for question in questions:
        assert "January" not in question["question"] and "janvier" not in question["question"]
        assert "2008-01-01" not in json.dumps(question)
        assert "date:year" in question["tags"] and "date-meaning:not-stated" in question["tags"]
    assert summary["by_date_meaning"] == {"not-stated": 3}


def test_no_french_question_is_a_french_prefix_glued_to_an_english_title():
    rows = [
        row("year", bank="gb", date="2008-01-01", title="Minutes of the MPC Meeting held on 9 April 2008"),
        row("none", bank="jp", date=None, title="Meeting on April 27 and 28, 2026 [PDF 336KB]"),
        row("day", bank="jp", date="2017-04-27", title="Meeting on April 26 and 27, 2017"),
    ]
    questions, summary = build(rows)
    found = by_id(questions)
    assert "ki-described-fr-year" not in found and "ki-described-fr-none" not in found
    assert "ki-described-en-year" in found and "ki-described-en-none" in found
    assert found["ki-described-fr-day"]["question"] == (
        "Compte rendu de la réunion de politique monétaire de la Banque du Japon du 27 avril 2017"
    )
    assert summary["skipped"]["described (French): only an English title describes the document"] == 2
    assert summary["by_language"] == {"(title)": 3, "en": 6, "fr": 1}
    assert "only an English title" in mki.format_summary(summary)


def test_a_label_is_not_used_to_describe_a_document_without_a_date_known_to_the_month():
    rows = [
        row("d1", bank="ca", doc_type="E1", date="2014-01-01", title="BoC Monetary Policy Report 2014-01")
    ]
    questions, summary = build(rows)
    assert [question["id"] for question in questions] == ["ki-label-d1"]
    assert summary["skipped"] == {
        "described: no date known to the month, and the title is a catalogue label": 1,
        "described+filters: no date known to the month, and the title is a catalogue label": 1,
    }


def test_a_document_without_a_date_uses_its_title_too():
    title = "Meeting on April 27 and 28, 2026 [PDF 336KB]"
    questions, _ = build([row("d1", bank="jp", date=None, title=title)])
    found = by_id(questions)
    assert found["ki-title-d1"]["question"] == "Meeting on April 27 and 28, 2026"
    assert found["ki-described-en-d1"]["question"] == (
        "Minutes of the Bank of Japan Monetary Policy Meeting: Meeting on April 27 and 28, 2026"
    )
    assert "date_from" not in found["ki-described-filters-en-d1"]["filters"]
    assert "date:none" in found["ki-title-d1"]["tags"]


def test_a_label_that_spells_out_a_placeholder_date_is_not_a_question():
    questions, summary = build(
        [
            row(
                "d1",
                bank="au",
                doc_type="E1",
                date="2015-05-01",
                title="RBA Statement on Monetary Policy 2015-05-01",
            )
        ]
    )
    assert [question["kind"] for question in questions] == ["described", "described", "described+filters"]
    assert summary["skipped"] == {"label: the title spells out a placeholder date": 1}
    assert not any("2015-05-01" in question["question"] for question in questions)


def test_what_cannot_be_said_honestly_is_skipped_and_counted():
    rows = [
        row("no-title", date="2012-06-20", title=""),
        row("file-name", bank="gb", date="2018-01-01", title="may-2018.pdf"),
        row("nothing", bank="ca", doc_type="E1", date=None, title=""),
        row("unknown-bank", bank="zz", date="2012-06-20", title="A statement"),
    ]
    questions, summary = build(rows)
    assert sorted(question["id"] for question in questions) == [
        "ki-described-en-no-title",
        "ki-described-filters-en-no-title",
        "ki-described-fr-no-title",
        "ki-title-unknown-bank",
    ]
    assert summary["skipped"] == {
        "described: no date known to the month, and no title": 1,
        "described: no date known to the month, and the title is a file name": 1,
        "described: no name for the bank": 1,
        "described+filters: no date known to the month, and no title": 1,
        "described+filters: no date known to the month, and the title is a file name": 1,
        "described+filters: no name for the bank": 1,
        "title: no title": 2,
        "title: the title is a file name": 1,
    }
    # A document that gives nothing is looked at, not sampled.
    cells = {(cell["bank"], cell["doc_type"]): cell for cell in summary["cells"]}
    assert (cells[("gb", "A3")]["visited"], cells[("gb", "A3")]["sampled"]) == (1, 0)
    assert (summary["documents_visited"], summary["documents_sampled"]) == (4, 2)


def test_a_doc_type_without_words_gives_its_title_only():
    questions, summary = build([row("d1", doc_type="C1")], doc_types=["C1"])
    assert [question["id"] for question in questions] == ["ki-title-d1"]
    assert summary["skipped"]["described: no words for the doc type"] == 1


# ---------------------------------------------------------------------------
# Ambiguity
# ---------------------------------------------------------------------------
def test_documents_that_share_a_date_are_all_expected():
    rows = [
        row("b", bank="ecb", doc_type="A1", date="2022-07-21", title="Monetary policy decision 2022-07-21"),
        row("a", bank="ecb", doc_type="A1", date="2022-07-21", title="Monetary policy decision 2022-07-21"),
        row("c", bank="ecb", doc_type="A1", date="2022-09-08", title="Another decision"),
    ]
    questions, summary = build(rows)
    found = by_id(questions)
    # One question for the pair, whichever was drawn, named after the first id.
    assert sorted(found) == [
        "ki-described-en-a",
        "ki-described-en-c",
        "ki-described-filters-en-a",
        "ki-described-filters-en-c",
        "ki-described-fr-a",
        "ki-described-fr-c",
        "ki-label-a",
        "ki-title-c",
    ]
    for identifier in ("ki-label-a", "ki-described-en-a", "ki-described-fr-a", "ki-described-filters-en-a"):
        assert found[identifier]["expected_doc_ids"] == ["a", "b"]
        assert found[identifier]["tags"][-1] == "several-expected"
        assert "equivalent-documents" not in found[identifier]["tags"]
    assert found["ki-title-c"]["expected_doc_ids"] == ["c"]
    assert "several-expected" not in found["ki-title-c"]["tags"]
    assert summary["several_expected"] == {"described": 2, "described+filters": 1, "label": 1}
    assert summary["too_ambiguous"] == {}
    assert summary["documents_sampled"] == 2  # the second of the pair adds no question
    assert summary["scope_shared_day"] == {"groups": 1, "documents": 2}
    assert summary["scope_shared_title"] == {"groups": 1, "documents": 2}
    assert "2 documents share their bank, type and day" in mki.format_summary(summary)


def test_too_many_documents_for_one_question_and_it_is_not_written():
    rows = [row(f"d{n}", bank="ca", doc_type="E1", date="2025-10-29", title="") for n in range(5)]
    rows.append(row("alone", bank="ca", doc_type="E1", date="2025-07-30", title="Report of July"))
    questions, summary = build(rows)
    assert {question["expected_doc_ids"][0] for question in questions} == {"alone"}
    assert summary["too_ambiguous"] == {"described": 10, "described+filters": 5}
    assert summary["skipped"]["described: ambiguous: more than 3 documents fit"] == 10
    questions, summary = build(rows, max_expected=5)
    assert by_id(questions)["ki-described-en-d0"]["expected_doc_ids"] == ["d0", "d1", "d2", "d3", "d4"]
    assert summary["too_ambiguous"] == {}


def test_a_month_question_expects_every_document_of_that_month():
    rows = [
        row("first", bank="ecb", doc_type="A1", date="2012-03-01", title="Decision A"),
        row("later", bank="ecb", doc_type="A1", date="2012-03-15", title="Decision B"),
        row("other-bank", bank="us", doc_type="A1", date="2012-03-20", title="Decision C"),
        row("other-year", bank="ecb", doc_type="A1", date="2013-03-20", title="Decision D"),
    ]
    found = by_id(build(rows)[0])
    month = found["ki-described-en-first"]
    assert month["question"] == "European Central Bank monetary policy decisions of March 2012"
    assert month["expected_doc_ids"] == ["first", "later"]
    # The document dated to the day is alone on its day.
    assert found["ki-described-en-later"]["expected_doc_ids"] == ["later"]
    assert found["ki-described-en-later"]["question"].endswith(" of 15 March 2012")
    assert found["ki-described-en-other-bank"]["question"] == (
        "Federal Reserve rate decision press release, 20 March 2012"
    )
    assert "date-meaning:unknown" in found["ki-described-en-other-bank"]["tags"]


def test_the_same_title_in_two_cells_is_ambiguous_for_the_title_only():
    rows = [
        row("a", bank="ecb", doc_type="A1", date="2012-03-15", title="Statement"),
        row("b", bank="us", doc_type="A2", date="2014-05-20", title="Statement"),
    ]
    found = by_id(build(rows)[0])
    assert found["ki-title-a"]["expected_doc_ids"] == ["a", "b"]
    assert "ki-title-b" not in found
    assert found["ki-described-en-a"]["expected_doc_ids"] == ["a"]
    assert found["ki-described-en-b"]["expected_doc_ids"] == ["b"]


def test_ambiguity_is_judged_on_the_whole_scope_not_on_the_sample():
    rows = [row(f"d{n}", date="2012-06-20", title=f"T{n}") for n in range(2)]
    rows += [row(f"e{n}", date=f"2013-06-{10 + n}", title=f"U{n}") for n in range(20)]
    questions, _ = build(rows, per_cell=1)
    for question in questions:
        if question["expected_doc_ids"][0] in ("d0", "d1") and question["kind"] != "title":
            assert question["expected_doc_ids"] == ["d0", "d1"]


# ---------------------------------------------------------------------------
# Equivalent documents: the same text under two doc types
# ---------------------------------------------------------------------------
def ecb_day(date="2022-06-09", suffix=""):
    return [
        row(f"dec{suffix}", bank="ecb", doc_type="A1", date=date, title=f"Monetary policy decision {date}"),
        row(f"sta{suffix}", bank="ecb", doc_type="A2", date=date, title=f"Monetary policy statement {date}"),
        row(f"com{suffix}", bank="ecb", doc_type="B1", date=date, title=f"ECB press conference {date}"),
    ]


def test_the_combined_document_declares_its_parts():
    assert mki.COMBINED_DOCUMENTS == {("ecb", "B1"): ("A1", "A2")}
    assert mki.equivalent_types("ecb", "B1") == ("A1", "A2")
    assert mki.equivalent_types("ecb", "A1") == ("B1",)
    assert mki.equivalent_types("ecb", "A2") == ("B1",)
    assert mki.equivalent_types("ecb", "A3") == ()
    assert mki.equivalent_types("us", "B1") == ()


def test_a_part_also_expects_the_combined_document_and_the_combined_document_its_parts():
    questions, summary = build(ecb_day())
    found = by_id(questions)
    kinds = ("ki-label-{}", "ki-described-en-{}", "ki-described-fr-{}", "ki-described-filters-en-{}")
    for pattern in kinds:
        # A part: itself, then the combined document; never the other part, which is another text.
        assert found[pattern.format("dec")]["expected_doc_ids"] == ["dec", "com"]
        assert found[pattern.format("sta")]["expected_doc_ids"] == ["sta", "com"]
        # The combined document: itself, then its parts.
        assert found[pattern.format("com")]["expected_doc_ids"] == ["com", "dec", "sta"]
        for name in ("dec", "sta", "com"):
            tags = found[pattern.format(name)]["tags"]
            assert tags[-2:] == ["several-expected", "equivalent-documents"]
    assert len(questions) == 12
    assert summary["equivalents_added"] == {"described": 6, "described+filters": 3, "label": 3}
    assert "equivalent" in mki.format_summary(summary)


def test_the_filters_admit_the_types_of_every_expected_document():
    found = by_id(build(ecb_day())[0])
    assert found["ki-described-filters-en-dec"]["filters"]["doc_types"] == ["A1", "B1"]
    assert found["ki-described-filters-en-sta"]["filters"]["doc_types"] == ["A2", "B1"]
    assert found["ki-described-filters-en-com"]["filters"]["doc_types"] == ["A1", "A2", "B1"]
    # The candidates are counted with the widened filter.
    assert "filter-candidates:2-3" in found["ki-described-filters-en-dec"]["tags"]
    assert "filter-candidates:2-3" in found["ki-described-filters-en-com"]["tags"]


def test_documents_are_equivalent_on_the_same_day_only_and_in_the_declared_cells_only():
    rows = [
        *ecb_day("2022-06-09"),
        row(
            "dec-later",
            bank="ecb",
            doc_type="A1",
            date="2022-07-21",
            title="Monetary policy decision 2022-07-21",
        ),
        row(
            "sta-later",
            bank="ecb",
            doc_type="A2",
            date="2022-07-21",
            title="Monetary policy statement 2022-07-21",
        ),
        row("us-sta", bank="us", doc_type="A2", date="2022-06-09", title="FOMC statement 2022-06-09"),
        row(
            "us-b1",
            bank="us",
            doc_type="B1",
            date="2022-06-09",
            title="FOMC press conference transcript 2022-06-09",
        ),
        row(
            "account",
            bank="ecb",
            doc_type="A3",
            date="2022-06-09",
            title="Monetary policy account 2022-06-09",
        ),
    ]
    found = by_id(build(rows)[0])
    # Without a combined document that day, the decision and the statement stand alone.
    assert found["ki-described-en-dec-later"]["expected_doc_ids"] == ["dec-later"]
    assert found["ki-described-en-sta-later"]["expected_doc_ids"] == ["sta-later"]
    assert "equivalent-documents" not in found["ki-described-en-dec-later"]["tags"]
    assert found["ki-described-filters-en-dec-later"]["filters"]["doc_types"] == ["A1"]
    assert found["ki-described-en-us-sta"]["expected_doc_ids"] == ["us-sta"]
    assert found["ki-described-en-us-b1"]["expected_doc_ids"] == ["us-b1"]
    assert found["ki-described-en-account"]["expected_doc_ids"] == ["account"]
    assert found["ki-described-en-com"]["expected_doc_ids"] == ["com", "dec", "sta"]


def test_equivalent_documents_do_not_count_against_max_expected():
    rows = [*ecb_day(), row("dec2", bank="ecb", doc_type="A1", date="2022-06-09", title="Another release")]
    questions, summary = build(rows, max_expected=2)
    found = by_id(questions)
    assert found["ki-described-en-dec"]["expected_doc_ids"] == ["dec", "dec2", "com"]
    assert found["ki-described-en-com"]["expected_doc_ids"] == ["com", "dec", "dec2", "sta"]
    assert summary["too_ambiguous"] == {}


def test_a_month_question_expects_the_equivalent_documents_of_that_month():
    rows = [
        row("dec", bank="ecb", doc_type="A1", date="2022-09-01", title="Decision of September"),
        row("com", bank="ecb", doc_type="B1", date="2022-09-08", title="Combined of September"),
        # Known to be of the first of the month; the decision is only known to be of September.
        row("first", bank="ecb", doc_type="B1", date="2022-09-01", title="Combined", date_precision="day"),
    ]
    found = by_id(build(rows)[0])
    assert found["ki-described-en-dec"]["question"].endswith(" of September 2022")
    assert found["ki-described-en-dec"]["expected_doc_ids"] == ["dec", "com", "first"]
    # The title of a document known to the month only has no equivalent: its day is not known,
    # even when the stored placeholder happens to be the day of a combined document.
    assert found["ki-title-dec"]["expected_doc_ids"] == ["dec"]
    # The combined documents are known to the day, and no part is known to be of their day.
    assert found["ki-described-en-com"]["expected_doc_ids"] == ["com"]
    assert found["ki-described-en-first"]["expected_doc_ids"] == ["first"]
    assert found["ki-title-first"]["expected_doc_ids"] == ["first"]


# ---------------------------------------------------------------------------
# How many documents pass the filters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("count", "bucket"),
    [(1, "1"), (2, "2-3"), (3, "2-3"), (4, "4-10"), (10, "4-10"), (11, ">10"), (500, ">10")],
)
def test_candidate_buckets(count, bucket):
    assert mki.candidates_bucket(count) == bucket


def test_filter_candidates_are_counted_on_the_whole_scope():
    rows = [row("d0", date="2012-06-20", title="T0")]
    rows += [row(f"near{n}", date=f"2012-07-{10 + n}", title=f"N{n}") for n in range(4)]
    rows += [row("far", date="2013-06-20", title="Far"), row("undated", date=None, title="Undated")]
    rows += [row("other-type", doc_type="A2", date="2012-06-21", title="O")]
    rows += [row("other-bank", bank="ecb", date="2012-06-21", title="E")]
    rows += [row(f"gb{n}", bank="gb", date="2008-01-01", title=f"Minutes {n}") for n in range(12)]
    questions, summary = build(rows, per_cell=1, seed=2)
    found = by_id(questions)
    drawn = next(
        q
        for q in questions
        if q["kind"] == "described+filters" and "bank:us" in q["tags"] and "doc_type:A3" in q["tags"]
    )
    document = drawn["expected_doc_ids"][0]
    passing = {
        "d0": 5,  # itself and the four within 45 days; never the undated document
        "near0": 5,
        "near1": 5,
        "near2": 5,
        "near3": 5,
        "far": 1,
        "undated": 7,  # no date window: the whole cell
    }[document]
    assert f"filter-candidates:{mki.candidates_bucket(passing)}" in drawn["tags"]
    # A year-only date gives no window: every document of the cell passes.
    british = next(q for q in questions if q["kind"] == "described+filters" and "bank:gb" in q["tags"])
    assert "filter-candidates:>10" in british["tags"]
    assert sum(summary["by_filter_candidates"].values()) == summary["by_kind"]["described+filters"]
    assert all(
        any(tag.startswith("filter-candidates:") for tag in q["tags"]) is (q["kind"] == "described+filters")
        for q in found.values()
    )


# ---------------------------------------------------------------------------
# Scope and sampling
# ---------------------------------------------------------------------------
def cells_of(count_by_cell):
    rows = []
    for (bank, doc_type), count in count_by_cell.items():
        for n in range(count):
            date = f"20{10 + n // 28:02d}-{1 + n % 12:02d}-{2 + n % 27:02d}"
            rows.append(row(f"{bank}-{doc_type}-{n:03d}", bank=bank, doc_type=doc_type, date=date))
    return rows


def test_at_most_per_cell_documents_per_bank_and_doc_type():
    rows = cells_of({("us", "A3"): 40, ("us", "A2"): 3, ("ecb", "A1"): 12, ("jp", "A3"): 1})
    questions, summary = build(rows, per_cell=5)
    sampled = {(cell["bank"], cell["doc_type"]): cell["sampled"] for cell in summary["cells"]}
    assert sampled == {("ecb", "A1"): 5, ("jp", "A3"): 1, ("us", "A2"): 3, ("us", "A3"): 5}
    assert len(questions) == 14 * 4
    assert summary["by_bank"] == {"ecb": 20, "jp": 4, "us": 32}
    assert summary["by_doc_type"] == {"A1": 20, "A2": 12, "A3": 24}
    assert summary["scope_documents"] == 56
    assert len({question["id"] for question in questions}) == len(questions)


def test_the_scope_is_the_doc_types_the_mime_type_and_the_banks_asked_for():
    rows = [
        row("in"),
        row("speech", doc_type="C1"),
        row("html", mime_type="text/html"),
        row("no-mime", mime_type=None),
        row("other-bank", bank="ecb"),
    ]

    def ids(questions):
        return {question["expected_doc_ids"][0] for question in questions}

    assert ids(build(rows)[0]) == {"in", "other-bank"}
    assert ids(build(rows, mime=None)[0]) == {"in", "html", "no-mime", "other-bank"}
    assert ids(build(rows, banks=["us"])[0]) == {"in"}
    assert ids(build(rows, doc_types=["C1"])[0]) == {"speech"}


def test_a_repeated_doc_id_is_counted_and_used_once():
    questions, summary = build([row("d1"), row("d1")])
    assert len(questions) == 4
    assert (summary["scope_documents"], summary["repeated_doc_ids"]) == (1, 1)


def test_the_same_seed_gives_the_same_questions_whatever_the_order_of_the_rows():
    rows = cells_of({("us", "A3"): 40, ("ecb", "A1"): 30})
    first, _ = build(rows, per_cell=6, seed=7)
    again, _ = build(list(reversed(rows)), per_cell=6, seed=7)
    other, _ = build(rows, per_cell=6, seed=8)
    assert first == again
    assert {q["id"] for q in first} != {q["id"] for q in other}
    assert len(first) == len(other) == 48


def test_the_sampling_order_is_pinned():
    # A hash of the seed and of the doc_id: no random generator, the same on every Python.
    import hashlib

    assert mki.sampling_key(1, "abc") == hashlib.sha256(b"1\x00abc").hexdigest()
    assert mki.sampling_key(1, "abc") != mki.sampling_key(2, "abc")


def test_the_dates_table_counts_first_of_january_per_cell():
    rows = [
        row("a", bank="gb", date="2008-01-01"),
        row("b", bank="gb", date="2009-01-01"),
        row("c", bank="us", date="2012-06-20"),
        row("d", bank="us", date="2012-06-01"),
        row("e", bank="us", date="2012-01-01", date_precision="day"),
        row("f", bank="us", date=None),
    ]
    _, summary = build(rows)
    cells = {cell["bank"]: cell for cell in summary["cells"]}
    assert (cells["gb"]["documents"], cells["gb"]["january_first"], cells["gb"]["first_of_month"]) == (
        2,
        2,
        2,
    )
    assert cells["gb"]["precision"] == {"day": 0, "month": 0, "year": 2, "none": 0}
    assert (cells["us"]["documents"], cells["us"]["january_first"], cells["us"]["first_of_month"]) == (
        4,
        1,
        2,
    )
    assert cells["us"]["precision"] == {"day": 2, "month": 1, "year": 0, "none": 1}
    text = mki.format_summary(summary)
    assert "100.0%" in text and "25.0%" in text


@pytest.mark.parametrize("kwargs", [{"per_cell": 0}, {"max_expected": 0}, {"date_window_days": -1}])
def test_impossible_parameters_are_refused(kwargs):
    with pytest.raises(ValueError):
        build([row("d1")], **kwargs)


# ---------------------------------------------------------------------------
# Files: the manifests in, the question file out
# ---------------------------------------------------------------------------
def write_manifest(path, rows):
    path.write_text("".join(json.dumps(entry) + "\n" for entry in rows), encoding="utf-8")


def test_the_command_writes_a_file_the_harness_reads(tmp_path, capsys):
    write_manifest(tmp_path / "us.jsonl", cells_of({("us", "A3"): 6}) + [row("speech", doc_type="C1")])
    write_manifest(
        tmp_path / "gb.jsonl", [row("g1", bank="gb", date="2008-01-01", title="Minutes of April 2008")]
    )
    (tmp_path / "us.jsonl").write_text((tmp_path / "us.jsonl").read_text() + "\n\n")
    output = tmp_path / "out.txt"
    arguments = ["--manifests", str(tmp_path / "*.jsonl"), "--doc-types", "A3", "--per-cell", "4"]
    assert mki.main([*arguments, "--seed", "3", "--output", str(output)]) == 0
    printed = capsys.readouterr().out
    assert "Questions: 19 from 5 documents" in printed and "gb" in printed and "100.0%" in printed
    questions = load_questions(output)
    assert len(questions) == 19  # the British document, described by its title, has no French question
    assert {question.kind for question in questions} == {"title", "described", "described+filters"}
    assert {tag for question in questions for tag in question.tags} >= {"bank:us", "bank:gb", "doc_type:A3"}
    assert all(question.filters is not None for question in questions if question.kind == "described+filters")
    first = output.read_bytes()
    assert mki.main([*arguments, "--seed", "3", "--output", str(output)]) == 0
    assert output.read_bytes() == first
    assert b"\r" not in first and first.endswith(b"\n")
    assert "é" in first.decode("utf-8")  # written as it is, not as an escape


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{not json\n", "line 1: not valid JSON"),
        ('{"doc_id": "a", "bank_code": "us", "doc_type": "A3"}\n[1]\n', "line 2: not a JSON object"),
        ('{"doc_id": "a", "bank_code": "us"}\n', "line 1: 'doc_type'"),
        ('{"doc_id": "", "bank_code": "us", "doc_type": "A3"}\n', "line 1: 'doc_id'"),
    ],
)
def test_a_bad_manifest_is_refused_with_its_line(tmp_path, capsys, content, match):
    (tmp_path / "bad.jsonl").write_text(content, encoding="utf-8")
    with pytest.raises(mki.ManifestError, match=match):
        mki.read_manifests([str(tmp_path / "bad.jsonl")])
    code = mki.main(
        ["--manifests", str(tmp_path / "bad.jsonl"), "--doc-types", "A3", "--output", str(tmp_path / "o")]
    )
    assert code == 2
    assert "make_known_item: error:" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_no_manifest_and_no_question_are_errors(tmp_path, capsys):
    output = tmp_path / "o.txt"
    assert (
        mki.main(["--manifests", str(tmp_path / "*.jsonl"), "--doc-types", "A3", "--output", str(output)])
        == 2
    )
    assert "no manifest file matches" in capsys.readouterr().err
    write_manifest(tmp_path / "us.jsonl", [row("speech", doc_type="C1")])
    assert (
        mki.main(["--manifests", str(tmp_path / "*.jsonl"), "--doc-types", "A3", "--output", str(output)])
        == 2
    )
    assert "no question could be built" in capsys.readouterr().err
    assert not output.exists()


def test_the_script_needs_the_standard_library_only():
    source = SCRIPT.read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "argparse",
        "collections",
        "datetime",
        "glob",
        "hashlib",
        "json",
        "re",
        "sys",
        "typing",
    }


# ---------------------------------------------------------------------------
# The committed batch
# ---------------------------------------------------------------------------
def tag_of(question, prefix):
    values = [tag[len(prefix) :] for tag in question.tags if tag.startswith(prefix)]
    assert len(values) == 1, (question.id, prefix)
    return values[0]


def test_the_committed_batch_is_a_valid_question_file():
    questions = load_questions(KNOWN_ITEM)
    assert len(questions) > 500
    assert {question.kind for question in questions} == {"title", "label", "described", "described+filters"}
    assert {question.language for question in questions} == {None, "en", "fr"}
    banks = {tag for question in questions for tag in question.tags if tag.startswith("bank:")}
    assert banks == {"bank:au", "bank:ca", "bank:ecb", "bank:gb", "bank:jp", "bank:us"}
    for question in questions:
        assert question.id.startswith("ki-")
        equivalents = "equivalent-documents" in question.tags
        # At most three documents of the cell of the question, and the equivalent ones on top.
        assert 1 <= len(question.expected_doc_ids) <= (5 if equivalents else 3), question.id
        assert (question.filters is not None) is (question.kind == "described+filters")
        assert (len(question.expected_doc_ids) > 1) is ("several-expected" in question.tags)
        assert not equivalents or "several-expected" in question.tags
        assert tag_of(question, "date-meaning:") in (*mki.DATE_MEANINGS, "not-stated")
        assert tag_of(question, "date:") in ("day", "month", "year", "none")


def test_the_committed_batch_is_the_one_the_readme_describes():
    import hashlib

    readme = (EVAL / "README.md").read_text(encoding="utf-8")
    questions = load_questions(KNOWN_ITEM)
    assert hashlib.sha256(KNOWN_ITEM.read_bytes()).hexdigest() in readme
    assert f"{len(questions)} automatic questions" in readme
    for kind in ("title", "label", "described", "described+filters"):
        count = sum(1 for question in questions if question.kind == kind)
        assert f"`{kind}` {count}" in readme, kind
    # The wording and the meaning of the date of every cell are listed.
    for cell, (english, _french, _agreement) in mki.CELL_WORDING.items():
        assert english in readme, cell
        assert f"| `{cell[0]}` | {cell[1]} |" in readme, cell


def test_the_committed_batch_tells_labels_from_titles():
    for question in load_questions(KNOWN_ITEM):
        bank = tag_of(question, "bank:")
        if question.kind == "label":
            assert mki.has_label_prefix(bank, question.question), question.id
        elif question.kind == "title":
            assert not mki.is_catalogue_label(bank, question.question), question.id
            assert not re.search(r"\d{4}-\d{2}", question.question), question.id


def test_the_committed_batch_has_no_french_prefix_on_an_english_title():
    for question in load_questions(KNOWN_ITEM):
        if question.language == "fr":
            # A French question always carries a date in French words; it never quotes a title.
            assert tag_of(question, "date-meaning:") != "not-stated", question.id
            assert " : " not in question.question and ": " not in question.question, question.id
            assert re.search(rf"({'|'.join(mki.MONTHS_FR)}) \d{{4}}$", question.question), question.id


def test_the_committed_batch_counts_the_candidates_of_every_filtered_question():
    for question in load_questions(KNOWN_ITEM):
        candidates = [tag for tag in question.tags if tag.startswith("filter-candidates:")]
        if question.kind != "described+filters":
            assert candidates == [], question.id
            continue
        bucket = tag_of(question, "filter-candidates:")
        assert bucket in ("1", "2-3", "4-10", ">10"), question.id
        # At least the expected documents pass the filters.
        expected = len(question.expected_doc_ids)
        assert mki.candidates_bucket(expected) in ("1", "2-3", "4-10", ">10")
        order = ("1", "2-3", "4-10", ">10")
        assert order.index(bucket) >= order.index(mki.candidates_bucket(expected)), question.id
        assert question.filters.source_codes == (tag_of(question, "bank:"),)
        assert tag_of(question, "doc_type:") in question.filters.doc_types
        assert (len(question.filters.doc_types) > 1) is ("equivalent-documents" in question.tags), question.id


def test_the_committed_batch_expects_equivalent_documents_for_the_declared_cells_only():
    declared = {
        (bank, doc_type)
        for (bank, combined), parts in mki.COMBINED_DOCUMENTS.items()
        for doc_type in (combined, *parts)
    }
    seen = set()
    for question in load_questions(KNOWN_ITEM):
        if "equivalent-documents" in question.tags:
            cell = (tag_of(question, "bank:"), tag_of(question, "doc_type:"))
            assert cell in declared, question.id
            seen.add(cell)
    assert seen == declared
    # Every question about a combined document expects its parts.
    for question in load_questions(KNOWN_ITEM):
        if (tag_of(question, "bank:"), tag_of(question, "doc_type:")) in mki.COMBINED_DOCUMENTS:
            assert "equivalent-documents" in question.tags, question.id


def test_the_committed_batch_words_every_date_after_its_meaning():
    months = "|".join((*mki.MONTHS_EN, *mki.MONTHS_FR))
    day = rf"\d{{1,2}}(er)? ({months}) \d{{4}}$"
    month = rf"({months}) \d{{4}}$"
    patterns = {
        ("meeting", "day"): rf" (of|du) {day}",
        ("meeting", "month"): rf" (of |de |d')({months}) \d{{4}}$",
        ("publication", "day"): rf", (published on|publié(e|s|es)? le) {day}",
        ("publication", "month"): rf", (published in|publié(e|s|es)? en) {month}",
        ("period", "day"): rf", {day}",
        ("period", "month"): rf", {month}",
        ("unknown", "day"): rf", {day}",
        ("unknown", "month"): rf", {month}",
    }
    for question in load_questions(KNOWN_ITEM):
        if question.kind not in ("described", "described+filters"):
            continue
        cell = (tag_of(question, "bank:"), tag_of(question, "doc_type:"))
        precision, meaning = tag_of(question, "date:"), tag_of(question, "date-meaning:")
        text = question.question
        if precision in ("year", "none"):
            assert meaning == "not-stated" and ": " in text, question.id
            continue
        assert meaning == mki.date_meaning(*cell), question.id
        assert re.search(patterns[(meaning, precision)], text), question.id
        if meaning != "meeting":
            # A date that is not a meeting date is never attached to the meeting by "of" or "du".
            assert not re.search(rf" (of|du|de) (\d{{1,2}}(er)? )?({months}) \d{{4}}$", text), question.id
        if meaning == "publication":
            assert ("published" in text) or ("publié" in text), question.id


def test_the_committed_batch_never_states_a_placeholder_date():
    months = "|".join((*mki.MONTHS_EN, *mki.MONTHS_FR))
    for question in load_questions(KNOWN_ITEM):
        precision = tag_of(question, "date:")
        text = question.question
        if question.kind in ("title", "label"):
            # A title is the manifest's own wording, kept unless it spells out a stored placeholder.
            assert not (precision != "day" and re.search(r"\d{4}-\d{2}-01\b", text)), question.id
            continue
        # A description never names 1 January, and names a day only for a date known to the day.
        assert not re.search(r"\b1(er)? (January|janvier)\b", text), question.id
        described_day = re.search(rf"[ ,] ?\d{{1,2}}(er)? ({months}) \d{{4}}$", text)
        described_month = re.search(rf"({months}) \d{{4}}$", text)
        if precision == "day":
            assert described_day, question.id
        elif precision == "month":
            assert described_month and not described_day, question.id
        else:
            # The title of the document follows the words, and no date of the generator.
            assert ": " in text, question.id
            assert question.filters is None or (
                question.filters.date_from is None and question.filters.date_to is None
            ), question.id
