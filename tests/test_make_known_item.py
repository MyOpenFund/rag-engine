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
# The questions of one document
# ---------------------------------------------------------------------------
def test_a_document_dated_to_the_day_gives_the_four_questions():
    questions, summary = build([row("d1", title="FOMC minutes 2012-06-20")])
    assert [question["id"] for question in questions] == [
        "ki-title-d1",
        "ki-described-en-d1",
        "ki-described-fr-d1",
        "ki-described-filters-en-d1",
    ]
    title, english, french, filtered = questions
    assert title == {
        "id": "ki-title-d1",
        "question": "FOMC minutes 2012-06-20",
        "expected_doc_ids": ["d1"],
        "kind": "title",
        "tags": ["bank:us", "doc_type:A3", "date:day"],
    }
    assert english["question"] == "Federal Reserve FOMC minutes, 20 June 2012"
    assert (english["kind"], english["language"]) == ("described", "en")
    assert french["question"] == (
        "Compte rendu de la réunion de politique monétaire du FOMC de la Réserve fédérale, 20 juin 2012"
    )
    assert (french["kind"], french["language"]) == ("described", "fr")
    assert filtered["question"] == english["question"]
    assert filtered["kind"] == "described+filters"
    assert filtered["filters"] == {
        "source_codes": ["us"],
        "doc_types": ["A3"],
        "date_from": "2012-05-06",
        "date_to": "2012-08-04",
    }
    assert "filters" not in english and "filters" not in title
    assert list(filtered) == ["id", "question", "expected_doc_ids", "filters", "kind", "language", "tags"]
    assert summary["by_kind"] == {"title": 1, "described": 2, "described+filters": 1}


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
        == "Reserve Bank of Australia monetary policy report, May 2015"
    )
    assert found["ki-described-fr-d1"]["question"] == (
        "Rapport sur la politique monétaire de la Banque de réserve d'Australie, mai 2015"
    )
    # The window covers the whole month, widened on both sides.
    filters = found["ki-described-filters-en-d1"]["filters"]
    assert (filters["date_from"], filters["date_to"]) == ("2015-03-17", "2015-07-15")
    assert found["ki-title-d1"]["tags"] == ["bank:au", "doc_type:E1", "date:month"]
    assert not any(re.search(r"\b1(er)? (May|mai)\b", question["question"]) for question in questions)


def test_a_first_of_january_is_never_stated_and_the_title_is_used_instead():
    title = "Minutes of the MPC Meeting held on 9 and 10 April 2008"
    questions, _ = build([row("d1", bank="gb", date="2008-01-01", title=title)])
    found = by_id(questions)
    assert found["ki-title-d1"]["question"] == title
    assert found["ki-described-en-d1"]["question"] == f"Bank of England MPC minutes: {title}"
    assert found["ki-described-fr-d1"]["question"] == (
        "Compte rendu de la réunion de politique monétaire du Comité de politique monétaire "
        f"de la Banque d'Angleterre : {title}"
    )
    # No date is known: the filters carry the bank and the type only.
    assert found["ki-described-filters-en-d1"]["filters"] == {"source_codes": ["gb"], "doc_types": ["A3"]}
    for question in questions:
        assert "January" not in question["question"] and "janvier" not in question["question"]
        assert "2008-01-01" not in json.dumps(question)
        assert question["tags"][-1] == "date:year"


def test_a_document_without_a_date_uses_its_title_too():
    title = "Meeting on April 27 and 28, 2026 [PDF 336KB]"
    questions, _ = build([row("d1", bank="jp", date=None, title=title)])
    found = by_id(questions)
    assert found["ki-title-d1"]["question"] == "Meeting on April 27 and 28, 2026"
    assert (
        found["ki-described-en-d1"]["question"] == "Bank of Japan minutes: Meeting on April 27 and 28, 2026"
    )
    assert "date_from" not in found["ki-described-filters-en-d1"]["filters"]
    assert found["ki-title-d1"]["tags"][-1] == "date:none"


def test_a_title_that_spells_out_a_placeholder_date_is_not_a_question():
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
    assert summary["skipped"] == {"title: the title spells out a placeholder date": 1}
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
        "ki-title-a",
        "ki-title-c",
    ]
    for identifier in ("ki-title-a", "ki-described-en-a", "ki-described-fr-a", "ki-described-filters-en-a"):
        assert found[identifier]["expected_doc_ids"] == ["a", "b"]
        assert found[identifier]["tags"][-1] == "several-expected"
    assert found["ki-title-c"]["expected_doc_ids"] == ["c"]
    assert "several-expected" not in found["ki-title-c"]["tags"]
    assert summary["several_expected"] == {"described": 2, "described+filters": 1, "title": 1}
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
    assert month["question"] == "European Central Bank rate decision press release, March 2012"
    assert month["expected_doc_ids"] == ["first", "later"]
    # The document dated to the day is alone on its day.
    assert found["ki-described-en-later"]["expected_doc_ids"] == ["later"]
    assert found["ki-described-en-later"]["question"].endswith(", 15 March 2012")


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
    assert "Questions: 20 from 5 documents" in printed and "gb" in printed and "100.0%" in printed
    questions = load_questions(output)
    assert len(questions) == 20
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
def test_the_committed_batch_is_a_valid_question_file():
    questions = load_questions(KNOWN_ITEM)
    assert len(questions) > 500
    assert {question.kind for question in questions} == {"title", "described", "described+filters"}
    assert {question.language for question in questions} == {None, "en", "fr"}
    banks = {tag for question in questions for tag in question.tags if tag.startswith("bank:")}
    assert banks == {"bank:au", "bank:ca", "bank:ecb", "bank:gb", "bank:jp", "bank:us"}
    for question in questions:
        assert question.id.startswith("ki-")
        assert 1 <= len(question.expected_doc_ids) <= 3
        assert (question.filters is not None) is (question.kind == "described+filters")
        assert (len(question.expected_doc_ids) > 1) is ("several-expected" in question.tags)


def test_the_committed_batch_never_states_a_placeholder_date():
    months = "|".join((*mki.MONTHS_EN, *mki.MONTHS_FR))
    for question in load_questions(KNOWN_ITEM):
        precision = next(tag for tag in question.tags if tag.startswith("date:"))[5:]
        text = question.question
        if question.kind == "title":
            # A title is the manifest's own wording, kept unless it spells out a stored placeholder.
            assert not (precision != "day" and re.search(r"\d{4}-\d{2}-01\b", text)), question.id
            continue
        # A description never names 1 January, and names a day only for a date known to the day.
        assert not re.search(r"\b1(er)? (January|janvier)\b", text), question.id
        described_day = re.search(rf", \d{{1,2}}(er)? ({months}) \d{{4}}$", text)
        described_month = re.search(rf", ({months}) \d{{4}}$", text)
        if precision == "day":
            assert described_day, question.id
        elif precision == "month":
            assert described_month and not described_day, question.id
        else:
            assert not described_day and not described_month, question.id
            if question.filters is not None:
                assert question.filters.date_from is None and question.filters.date_to is None, question.id
