"""``rag-engine eval-check``: a question file checked without any collection.

The command reads the file and, when asked, the manifests of a corpus; it builds no store and
no embedder, and reads no setting.
"""

from __future__ import annotations

import json

import pytest
from eval_fixture import QUESTIONS, QUESTIONS_V2, fixture_corpus

from rag_engine import cli
from rag_engine.eval import EvalError, Question, load_questions
from rag_engine.eval_check import (
    CatalogueEntry,
    DocumentReference,
    QuestionCheck,
    check_questions,
    format_check,
    load_catalogue,
)
from rag_engine.search import SearchFilters


def write(path, *lines) -> str:
    text = "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines)
    path.write_text(text + "\n", encoding="utf-8")
    return str(path)


def q(**changes):
    base = {"id": "q1", "question": "What was decided?", "expected_doc_ids": ["d1"]}
    base.update(changes)
    return {key: value for key, value in base.items() if value is not ...}


@pytest.fixture
def manifests(tmp_path):
    """The corpus of the fixture as two manifest files, one per half, with fields the check does not read."""
    rows = [
        {
            "doc_id": document["doc_id"],
            "bank_code": document["metadata"]["source_code"],
            "doc_type": document["metadata"]["doc_type"],
            "mime_type": "application/pdf",
            "title": document["metadata"]["title"],
        }
        for document in fixture_corpus()["documents"]
    ]
    folder = tmp_path / "manifest"
    folder.mkdir()
    write(folder / "a.jsonl", *rows[:6], "")
    write(folder / "b.jsonl", *rows[6:])
    return folder


@pytest.fixture
def no_engine(monkeypatch):
    """The command may not build a store or an embedder, nor read the settings."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("eval-check must not connect to anything")

    monkeypatch.setattr(cli, "_store", refuse)
    monkeypatch.setattr(cli, "_embedder", refuse)
    monkeypatch.setattr(cli.Settings, "from_env", refuse)


def run(*arguments: str) -> int:
    return cli.main(["eval-check", *arguments])


# ---------------------------------------------------------------------------
# check_questions
# ---------------------------------------------------------------------------
def test_counts_of_the_second_fixture():
    check = check_questions(load_questions(QUESTIONS_V2))
    assert isinstance(check, QuestionCheck)
    assert (check.questions, check.k) == (11, 10)
    assert check.by_kind == {"graded": 5, "graded+filters": 4, "abstention": 2}
    assert list(check.by_kind) == ["graded", "graded+filters", "abstention"]  # most frequent first
    assert check.by_tag == {
        "bank:ecb": 4,
        "factual": 4,
        "out-of-corpus": 3,
        "bank:gb": 2,
        "chain": 2,
        "false-premise": 2,
        "aggregation": 1,
        "bank:au": 1,
        "bank:us": 1,
        "short-query": 1,
    }
    assert check.by_behavior == {"answer": 6, "correct_premise": 2, "abstain": 3}
    assert check.by_language == {"en": 10, "(none)": 1}
    assert (check.with_filters, check.run_unfiltered) == (5, 2)
    # fx2-q03 and fx2-q11 do not say which documents are essential; fx2-q10 says there is none.
    assert check.essential_not_stated == 2
    assert check.essential_counts == {0: 1, 1: 5, 2: 2, 13: 1}
    assert check.expected_counts == {0: 2, 1: 3, 2: 4, 3: 1, 13: 1}
    assert check.without_documents == 2
    assert check.more_essential_than_k == ("fx2-q05",)
    assert check.duplicate_texts == (("fx2-q01", "fx2-q07"),)
    assert check.mixed_kinds == ()
    assert check.documents == 14  # the thirteen of the corpus and one that is not in it
    assert (check.catalogue_documents, check.missing, check.out_of_scope) == (None, (), ())
    assert (check.by_doc_type, check.by_mime_type) == (None, None)
    assert check.ok is True


def test_counts_of_the_first_fixture():
    check = check_questions(load_questions(QUESTIONS))
    assert check.questions == 13
    assert check.by_kind == {"filtered": 6, "verbatim": 6, "second": 1}
    assert check.by_behavior == {"answer": 13}
    assert check.by_language == {"(none)": 12, "fr": 1}
    assert (check.with_filters, check.run_unfiltered) == (6, 0)
    assert (check.essential_not_stated, check.essential_counts) == (13, {})
    assert check.expected_counts == {1: 10, 2: 3}
    assert check.duplicate_texts == (("fx-q12", "fx-q13"),)
    # fx-q09 is of the kind "filtered" and carries no filter.
    assert check.mixed_kinds == ("filtered",)
    assert (check.without_documents, check.more_essential_than_k) == (0, ())


def test_what_deserves_a_look():
    filters = SearchFilters(doc_types=["A1"])
    questions = [
        Question(id="q1", question="The  same text", expected_doc_ids=("a",), kind="mixed"),
        Question(id="q2", question="the same TEXT ", expected_doc_ids=("a",), kind="mixed", filters=filters),
        Question(id="q3", question="Another", expected_doc_ids=("a",), filters=filters),
        Question(id="q4", question="the same text", expected_doc_ids=("b",)),
        Question(id="q5", question="Many", essential_ids=("a", "b", "c"), kind="plain"),
        Question(
            id="q6", question="Exactly k", essential_ids=("a", "b"), acceptable_ids=("c",), kind="plain"
        ),
        Question(
            id="q7",
            question="None",
            expected_doc_ids=(),
            expected_behavior="abstain",
            kind="apart",
            filters=filters,
        ),
        Question(id="q8", question="Also apart", expected_doc_ids=("a",), kind="apart", filters=filters),
    ]
    check = check_questions(questions, k=2)
    # Compared without regard to case and spacing; in the order of the file.
    assert check.duplicate_texts == (("q1", "q2", "q4"),)
    # "(none)" mixes q3 (filters) and q4. "apart" does not: its question without filters is not scored.
    assert check.mixed_kinds == ("(none)", "mixed")
    assert check.more_essential_than_k == ("q5",)  # three essential documents; q6 has two, and one acceptable
    assert check_questions(questions, k=3).more_essential_than_k == ()
    assert check.by_kind == {"mixed": 2, "(none)": 2, "apart": 2, "plain": 2}
    assert check.essential_counts == {2: 1, 3: 1} and check.essential_not_stated == 6


@pytest.mark.parametrize("k", [0, -1, True, 2.0, "10"])
def test_k_must_be_a_count(k):
    with pytest.raises(EvalError, match="k must be an integer >= 1"):
        check_questions(load_questions(QUESTIONS_V2), k=k)


def test_a_scope_needs_a_catalogue():
    with pytest.raises(EvalError, match="needs a catalogue"):
        check_questions(load_questions(QUESTIONS_V2), doc_types=["A1"])
    with pytest.raises(EvalError, match="needs a catalogue"):
        check_questions(load_questions(QUESTIONS_V2), mime_types=["application/pdf"])
    with pytest.raises(EvalError, match="doc_types must be a list of strings"):
        check_questions(load_questions(QUESTIONS_V2), catalogue={}, doc_types="A1")
    with pytest.raises(EvalError, match="mime_types must be a non-empty list"):
        check_questions(load_questions(QUESTIONS_V2), catalogue={}, mime_types=[])


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------
def test_load_catalogue(manifests):
    catalogue = load_catalogue([str(manifests / "*.jsonl")])
    assert len(catalogue) == 13
    assert catalogue["fx-jp-minutes"] == CatalogueEntry("fx-jp-minutes", "application/pdf", "A3")
    assert load_catalogue([manifests / "a.jsonl"]).keys() < catalogue.keys()
    # The same file through two patterns is read once; a doc_id met again keeps its first row.
    write(manifests / "c.jsonl", {"doc_id": "fx-jp-minutes", "doc_type": "Z9"}, {"doc_id": "bare"})
    both = load_catalogue([str(manifests / "*.jsonl"), str(manifests / "c.jsonl")])
    assert len(both) == 14
    assert both["fx-jp-minutes"].doc_type == "A3"
    assert both["bare"] == CatalogueEntry("bare", None, None)


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        (["{not json"], "bad.jsonl, line 1: not valid JSON"),
        ([{"doc_id": "a"}, "[1]"], "bad.jsonl, line 2: a line must hold one JSON object"),
        ([{"doc_type": "A1"}], "bad.jsonl, line 1: 'doc_id' is missing"),
        ([{"doc_id": "a"}, "", {"doc_id": " "}], "bad.jsonl, line 3: 'doc_id'"),
        ([{"doc_id": 7}], "line 1: 'doc_id'"),
        ([{"doc_id": "a", "mime_type": 3}], "line 1: 'mime_type' must be a string"),
        ([{"doc_id": "a", "doc_type": ["A1"]}], "line 1: 'doc_type' must be a string"),
    ],
)
def test_a_bad_manifest_is_refused_with_its_line(tmp_path, lines, match):
    path = write(tmp_path / "bad.jsonl", *lines)
    with pytest.raises(EvalError, match=match):
        load_catalogue([path])


def test_a_pattern_without_a_file_is_refused(tmp_path, manifests):
    with pytest.raises(EvalError, match="no manifest file matches"):
        load_catalogue([str(tmp_path / "nowhere" / "*.jsonl")])
    with pytest.raises(EvalError, match="no manifest file matches"):
        load_catalogue([str(manifests / "a.jsonl"), str(manifests / "z.jsonl")])
    with pytest.raises(EvalError, match="no manifest file was given"):
        load_catalogue([])
    (tmp_path / "latin.jsonl").write_bytes(b'{"doc_id": "caf\xe9"}\n')
    with pytest.raises(EvalError, match="not UTF-8"):
        load_catalogue([str(tmp_path / "latin.jsonl")])


def test_documents_that_are_not_in_the_catalogue_are_reported(manifests):
    catalogue = load_catalogue([str(manifests / "*.jsonl")])
    check = check_questions(load_questions(QUESTIONS_V2), catalogue=catalogue)
    assert check.catalogue_documents == 13
    # The one document of the file that is not in the corpus, with who names it and how.
    assert check.missing == (DocumentReference("fx-us-statement-2016-01", (("fx2-q02", 2, "acceptable"),)),)
    assert check.out_of_scope == () and check.ok is False
    assert check.by_mime_type == {"application/pdf": 13}
    assert check.by_doc_type == {"A3": 4, "A1": 2, "A2": 2, "B1": 2, "E1": 2, "F1": 1}
    old = check_questions(load_questions(QUESTIONS), catalogue=catalogue)
    assert [(reference.doc_id, reference.questions) for reference in old.missing] == [
        ("fx-ch-statement-2016-06", (("fx-q09", 9, "expected"),)),
        ("fx-us-presser-2016-03", (("fx-q10", 10, "expected"),)),
    ]
    assert old.missing[0].to_dict() == {
        "doc_id": "fx-ch-statement-2016-06",
        "mime_type": None,
        "doc_type": None,
        "questions": [{"id": "fx-q09", "line": 9, "as": "expected"}],
    }


def test_documents_outside_the_scope_are_reported(manifests):
    catalogue = dict(load_catalogue([str(manifests / "*.jsonl")]))
    catalogue["fx-us-statement-2016-01"] = CatalogueEntry("fx-us-statement-2016-01", "text/html", "A2")
    questions = load_questions(QUESTIONS_V2)
    assert (
        check_questions(questions, catalogue=catalogue).ok is True
    )  # in the catalogue: no scope, no complaint
    by_mime = check_questions(questions, catalogue=catalogue, mime_types=["application/pdf"])
    assert by_mime.missing == () and by_mime.ok is False
    assert by_mime.out_of_scope == (
        DocumentReference("fx-us-statement-2016-01", (("fx2-q02", 2, "acceptable"),), "text/html", "A2"),
    )
    assert check_questions(questions, catalogue=catalogue, mime_types=["application/pdf", "text/html"]).ok
    by_type = check_questions(questions, catalogue=catalogue, doc_types=["A1", "A2", "A3", "B1", "E1"])
    # The projections (F1) are essential to fx2-q05 only.
    assert [(reference.doc_id, reference.doc_type) for reference in by_type.out_of_scope] == [
        ("fx-us-projections-2015-12", "F1")
    ]
    assert by_type.out_of_scope[0].questions == (("fx2-q05", 5, "essential"),)
    assert by_type.scope_doc_types == ("A1", "A2", "A3", "B1", "E1") and by_type.scope_mime_types is None
    both = check_questions(
        questions,
        catalogue=catalogue,
        doc_types=["A1", "A2", "A3", "B1", "E1"],
        mime_types=["application/pdf"],
    )
    assert len(both.out_of_scope) == 2
    # A document without a doc type or a MIME type in the catalogue is in no scope.
    catalogue["fx-jp-minutes"] = CatalogueEntry("fx-jp-minutes")
    bare = check_questions(questions, catalogue=catalogue, mime_types=["application/pdf", "text/html"])
    assert [reference.doc_id for reference in bare.out_of_scope] == ["fx-jp-minutes"]
    assert bare.by_mime_type["(none)"] == 1


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------
def test_a_valid_file_is_described_and_exits_with_0(capsys, no_engine):
    assert run("--questions", str(QUESTIONS_V2)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    questions = load_questions(QUESTIONS_V2)
    expected = format_check(check_questions(questions), name=str(QUESTIONS_V2), sha256=questions.sha256)
    assert captured.out == expected + "\n"
    lines = captured.out.splitlines()
    assert lines[0].endswith(f"questions_v2.jsonl: 11 questions (sha256 {questions.sha256[:12]})")
    assert "kind: graded 5, graded+filters 4, abstention 2" in lines
    assert "expected_behavior: answer 6, correct_premise 2, abstain 3" in lines
    assert "language: en 10, (none) 1" in lines
    assert "filters: 5 with filters, 2 of them scored again without (run_unfiltered)" in lines
    assert (
        "essential documents per question (documents: questions): not stated: 2, 0: 1, 1: 5, 2: 2, 13: 1"
        in lines
    )
    assert "expected documents per question (documents: questions): 0: 2, 1: 3, 2: 4, 3: 1, 13: 1" in lines
    assert "distinct documents named: 14" in lines
    assert any(line.startswith("without any document: 2 ") for line in lines)
    assert "duplicate question texts: 1" in lines and "  the same text: fx2-q01, fx2-q07" in lines
    assert "more essential documents than k=10: fx2-q05" in lines
    assert "tag: bank:ecb 4, factual 4, out-of-corpus 3" in captured.out
    assert lines[-1] == "The file is valid."
    assert "catalogue" not in captured.out
    assert all(len(line) <= 120 for line in lines)


def test_k_is_an_option_of_the_check(capsys, no_engine):
    assert run("--questions", str(QUESTIONS_V2), "--k", "13") == 0
    assert "more essential documents" not in capsys.readouterr().out
    assert run("--questions", str(QUESTIONS_V2), "--k", "1") == 0
    assert "more essential documents than k=1: fx2-q05, fx2-q06, fx2-q08" in capsys.readouterr().out
    assert run("--questions", str(QUESTIONS_V2), "--k", "0") == 2
    assert "--k must be at least 1" in capsys.readouterr().err


def test_a_kind_that_mixes_filters_is_pointed_at(capsys, no_engine):
    assert run("--questions", str(QUESTIONS)) == 0
    out = capsys.readouterr().out
    assert "kinds with and without filters: filtered" in out
    assert "the whole kind is kept out of the" in out
    assert "essential documents per question (documents: questions): not stated: 13" in out


@pytest.mark.parametrize(
    ("line", "fragment"),
    [
        ("{not json", "not valid JSON"),
        (q(expected_doc_ids=[]), "'expected_doc_ids' is empty"),
        (q(expected_behavior="refuse"), "'expected_behavior'"),
        (q(essential_ids=["d1"], acceptable_ids=["d1"]), "both name 'd1'"),
        (q(essential_ids=["d2"]), "it lacks 'd2'"),
        (q(must_include="5-4"), "'must_include'"),
        (q(must_not_include=[""]), "'must_not_include'"),
        (q(run_unfiltered=True), "'run_unfiltered' is true on a question without filters"),
        (q(filters={"source": ["ecb"]}), "unknown filter 'source'"),
        (q(essential=["d1"]), "unknown field 'essential'"),
        (q(id="q0"), "duplicate id 'q0'"),
    ],
)
def test_a_file_that_breaks_a_rule_exits_with_2_and_names_the_line(
    tmp_path, capsys, no_engine, line, fragment
):
    path = write(tmp_path / "q.jsonl", q(id="q0"), line)
    assert run("--questions", path) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("rag-engine: error: ")
    assert "q.jsonl, line 2" in captured.err and fragment in captured.err


def test_a_missing_or_empty_file_exits_with_2(tmp_path, capsys, no_engine):
    assert run("--questions", str(tmp_path / "absent.jsonl")) == 2
    assert "absent.jsonl" in capsys.readouterr().err
    assert run("--questions", write(tmp_path / "empty.jsonl", "")) == 2
    assert "holds no question" in capsys.readouterr().err
    assert cli.main(["eval-check"]) == 2  # --questions is required
    assert "--questions" in capsys.readouterr().err


def test_every_document_found_in_the_manifests_exits_with_0(tmp_path, capsys, no_engine, manifests):
    write(
        manifests / "c.jsonl",
        {"doc_id": "fx-us-statement-2016-01", "mime_type": "application/pdf", "doc_type": "A2"},
    )
    pattern = str(manifests / "*.jsonl")
    assert run("--questions", str(QUESTIONS_V2), "--manifests", pattern) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert "catalogue: 14 documents" in lines
    assert "documents named, per doc type: A3 4, A2 3, A1 2, B1 2, E1 2, F1 1" in lines
    assert "documents named, per MIME type: application/pdf 14" in lines
    assert "not in the catalogue: 0" in lines
    assert not any(line.startswith("in the catalogue, outside the scope") for line in lines)
    assert lines[-1] == "The file is valid, and every document it names is in the catalogue."
    # With the scope of the collection.
    scope = ["--doc-types", "A1", "A2", "A3", "B1", "E1", "F1", "--mime", "application/pdf"]
    assert run("--questions", str(QUESTIONS_V2), "--manifests", pattern, *scope) == 0
    out = capsys.readouterr().out
    assert "scope: doc types A1, A2, A3, B1, E1, F1; MIME types application/pdf" in out
    assert "in the catalogue, outside the scope: 0" in out


def test_a_document_missing_from_the_manifests_exits_with_2(capsys, no_engine, manifests):
    files = [str(manifests / "a.jsonl"), str(manifests / "b.jsonl")]
    assert run("--questions", str(QUESTIONS_V2), "--manifests", *files) == 2
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert "not in the catalogue: 1" in lines
    assert "  fx-us-statement-2016-01: fx2-q02 (acceptable, line 2)" in lines
    assert (
        lines[-1] == "The file is valid, but 1 document it names is not in the catalogue or not in the scope."
    )
    assert captured.err.startswith("rag-engine: error: 1 document named by")
    # Half of the catalogue: every document of the other half is listed, with every question that names it.
    assert run("--questions", str(QUESTIONS_V2), "--manifests", files[0]) == 2
    captured = capsys.readouterr()
    assert "not in the catalogue: 8" in captured.out
    assert "  fx-gb-report-2016-08: fx2-q04 (essential, line 4), fx2-q05 (essential, line 5)," in captured.out
    assert "8 documents named by" in captured.err and "are not in the catalogue" in captured.err


def test_a_document_outside_the_scope_exits_with_2(capsys, no_engine, manifests):
    write(
        manifests / "c.jsonl",
        {"doc_id": "fx-us-statement-2016-01", "mime_type": "text/html", "doc_type": "A2"},
    )
    pattern = str(manifests / "*.jsonl")
    assert run("--questions", str(QUESTIONS_V2), "--manifests", pattern) == 0
    capsys.readouterr()
    assert run("--questions", str(QUESTIONS_V2), "--manifests", pattern, "--mime", "application/pdf") == 2
    captured = capsys.readouterr()
    assert "in the catalogue, outside the scope: 1" in captured.out
    assert "  fx-us-statement-2016-01 [A2, text/html]: fx2-q02 (acceptable, line 2)" in captured.out
    assert "documents named, per MIME type: application/pdf 13, text/html 1" in captured.out
    assert run("--questions", str(QUESTIONS_V2), "--manifests", pattern, "--doc-types", "A1", "A2") == 2
    assert "in the catalogue, outside the scope: 9" in capsys.readouterr().out


def test_unusable_manifests_exit_with_2(tmp_path, capsys, no_engine, manifests):
    assert run("--questions", str(QUESTIONS_V2), "--manifests", str(tmp_path / "none-*.jsonl")) == 2
    assert "no manifest file matches" in capsys.readouterr().err
    bad = write(tmp_path / "bad.jsonl", {"doc_id": "a"}, {"title": "no id"})
    assert run("--questions", str(QUESTIONS_V2), "--manifests", bad) == 2
    captured = capsys.readouterr()
    assert "bad.jsonl, line 2: 'doc_id' is missing" in captured.err and captured.out == ""
    assert run("--questions", str(QUESTIONS_V2), "--doc-types", "A1") == 2
    assert "need --manifests" in capsys.readouterr().err
    assert run("--questions", str(QUESTIONS_V2), "--mime", "application/pdf") == 2
    assert "need --manifests" in capsys.readouterr().err


def test_the_committed_question_files_are_valid(capsys, no_engine):
    from pathlib import Path

    known = Path(__file__).resolve().parent.parent / "eval" / "known_item_v1.jsonl"
    assert run("--questions", str(known)) == 0
    out = capsys.readouterr().out
    assert "797 questions" in out and "expected_behavior: answer 797" in out
    assert "kind: described 393, described+filters 210, label 158, title 36" in out
    assert "filters: 210 with filters, 0 of them scored again without" in out
    assert out.splitlines()[-1] == "The file is valid."
