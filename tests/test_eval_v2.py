"""The second version of the question format: essential documents, behaviours, second runs, abstention.

``eval/fixtures/questions_v2.jsonl`` is asked of the corpus of the fixture; every value below
was worked out by hand from the file. The questions of the first version, and their values,
are in ``test_eval.py`` and are not touched by any of this.
"""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction

import pytest
from eval_fixture import QUESTIONS, QUESTIONS_V2, fixture_corpus, fixture_embedder, load_fixture
from qdrant_client import QdrantClient, models

from rag_engine.config import Settings
from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.eval import (
    BEHAVIORS,
    DEFAULT_SCORE_LEVELS,
    QUESTION_FIELDS,
    UNFILTERED_SUFFIX,
    UNFILTERED_TAG,
    Abstention,
    Aggregate,
    EssentialAggregate,
    EssentialMeans,
    EvalError,
    EvalReport,
    FilterEffect,
    Question,
    QuestionResult,
    RankedDocument,
    TopScores,
    evaluate,
    format_report,
    load_questions,
)
from rag_engine.search import Hit, SearchFilters
from rag_engine.store import QdrantStore

COLLECTION = "eval-fixture"
F = Fraction
ALL_DOCUMENTS = [document["doc_id"] for document in fixture_corpus()["documents"]]


@pytest.fixture(scope="module")
def store():
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client, settings=Settings(batch_size=16))
    assert load_fixture(store, COLLECTION) == 32
    yield store
    client.close()


@pytest.fixture(scope="module")
def questions():
    return load_questions(QUESTIONS_V2)


@pytest.fixture(scope="module")
def report(store, questions):
    # Two score levels with a known count: every cosine is at least -1, and only a chunk equal to
    # the question reaches 0.999.
    return evaluate(
        questions, collection=COLLECTION, store=store, embedder=fixture_embedder(), score_levels=(0.999, -1)
    )


def write(tmp_path, *lines) -> str:
    path = tmp_path / "questions.jsonl"
    text = "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines)
    path.write_text(text + "\n", encoding="utf-8")
    return str(path)


def q(**changes):
    base = {"id": "q1", "question": "What was decided?", "expected_doc_ids": ["d1"]}
    base.update(changes)
    return {key: value for key, value in base.items() if value is not ...}


def graded(**changes):
    """A question that states its essential and acceptable documents, and leaves ``expected_doc_ids`` out."""
    return q(**{"expected_doc_ids": ..., "essential_ids": ["d1"], "acceptable_ids": ["d2"], **changes})


# ---------------------------------------------------------------------------
# The question file
# ---------------------------------------------------------------------------
def test_the_fields_of_the_first_version_keep_their_place():
    assert QUESTION_FIELDS[:8] == (
        "id",
        "question",
        "expected_doc_ids",
        "filters",
        "kind",
        "tags",
        "language",
        "rationale",
    )
    assert QUESTION_FIELDS[8:] == (
        "expected_behavior",
        "essential_ids",
        "acceptable_ids",
        "must_include",
        "must_not_include",
        "run_unfiltered",
        "essential_any_of",
        "expected_spans",  # the passage-level fields, documented in eval/filings_README.md
        "span_match",
        "difficulty",
    )
    assert BEHAVIORS == ("answer", "correct_premise", "abstain")


def test_load_the_second_fixture(questions):
    assert [question.id for question in questions] == [
        f"fx2-q{n:02d}" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 10, 11)
    ]
    first, second, third, fourth, fifth, sixth, seventh, twelfth, thirteenth, tenth, eleventh = (
        questions[index] for index in (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12)
    )
    assert first.expected_behavior == "answer"
    assert first.essential_ids == ("fx-ecb-decision-2016-03",)
    assert first.acceptable_ids == ("fx-ecb-combined-2016-03",)
    assert first.expected_doc_ids == ("fx-ecb-decision-2016-03", "fx-ecb-combined-2016-03")
    assert first.must_include == ("80 billion euro", "April")
    assert first.must_not_include == ("60 billion euro",)
    assert first.run_unfiltered is False
    # expected_doc_ids left out: the essential documents, then the acceptable ones.
    assert second.expected_behavior == "correct_premise"
    assert second.expected_doc_ids == ("fx-us-statement-2015-12", "fx-us-statement-2016-01")
    # The first version of a line: which documents are essential is not known.
    assert (third.expected_behavior, third.essential_ids, third.acceptable_ids) == ("answer", None, None)
    assert (third.must_include, third.must_not_include) == ((), ())
    assert len(fifth.essential_ids) == 13 and fifth.acceptable_ids == ()
    # Given in another order than essential + acceptable: accepted, and kept as given.
    assert sixth.expected_doc_ids == (
        "fx-us-minutes-2015-12",
        "fx-gb-report-2016-08",
        "fx-gb-minutes-2016-08",
    )
    assert sixth.essential_ids == ("fx-gb-minutes-2016-08", "fx-gb-report-2016-08")
    assert (seventh.run_unfiltered, seventh.filters) == (True, SearchFilters(doc_types=["A1"]))
    assert (tenth.expected_behavior, tenth.expected_doc_ids) == ("abstain", ())
    assert (tenth.essential_ids, tenth.acceptable_ids) == ((), ())
    assert (eleventh.expected_doc_ids, eleventh.essential_ids, eleventh.run_unfiltered) == ((), None, False)
    # An abstention with a context document: nothing essential.
    assert (fourth.expected_behavior, fourth.essential_ids) == ("abstain", ())
    assert fourth.acceptable_ids == fourth.expected_doc_ids == ("fx-gb-report-2016-08",)
    # Either-of groups: alone, and next to an essential document.
    group = ("fx-ecb-statement-2016-03", "fx-ecb-combined-2016-03")
    assert (twelfth.essential_ids, twelfth.essential_any_of, twelfth.expected_doc_ids) == (
        (),
        (group,),
        group,
    )
    assert thirteenth.requirements == (("fx-ecb-decision-2016-03",), group)
    assert thirteenth.expected_doc_ids == ("fx-ecb-decision-2016-03", *group)


def test_a_question_is_written_back_as_it_was_read(questions, tmp_path):
    lines = [question.to_dict() for question in questions]
    again = load_questions(write(tmp_path, *lines))
    assert list(again) == list(questions)  # the lines are the same ones, so the line numbers too
    first = lines[0]
    assert list(first)[:5] == ["id", "question", "essential_ids", "acceptable_ids", "expected_doc_ids"]
    assert "expected_behavior" not in first and "run_unfiltered" not in first  # the defaults are not written
    assert lines[1]["expected_behavior"] == "correct_premise"
    assert lines[6]["run_unfiltered"] is True
    assert "essential_ids" not in lines[2] and "must_include" not in lines[2]
    assert lines[11]["expected_doc_ids"] == [] and lines[11]["must_not_include"] == ["minus 0.1 percent"]
    assert lines[9]["essential_any_of"] == [["fx-ecb-statement-2016-03", "fx-ecb-combined-2016-03"]]


def test_the_document_lists_of_a_question(tmp_path):
    lines = [
        graded(),
        graded(id="q2", acceptable_ids=...),
        graded(id="q3", essential_ids=...),
        graded(id="q4", expected_doc_ids=["d2", "d1"]),
        graded(id="q5", essential_ids=None, acceptable_ids=None, expected_doc_ids=["d9"]),
        graded(id="q6", essential_ids=["d1"], acceptable_ids=[], expected_doc_ids=None),
    ]
    one, two, three, four, five, six = load_questions(write(tmp_path, *lines))
    assert (one.essential_ids, one.acceptable_ids, one.expected_doc_ids) == (("d1",), ("d2",), ("d1", "d2"))
    assert (two.essential_ids, two.acceptable_ids, two.expected_doc_ids) == (("d1",), (), ("d1",))
    assert (three.essential_ids, three.acceptable_ids, three.expected_doc_ids) == ((), ("d2",), ("d2",))
    assert four.expected_doc_ids == ("d2", "d1")
    assert (five.essential_ids, five.acceptable_ids, five.expected_doc_ids) == (None, None, ("d9",))
    assert six.expected_doc_ids == ("d1",)


def test_only_a_question_that_abstains_may_expect_no_document(tmp_path):
    lines = [
        q(expected_doc_ids=[], expected_behavior="abstain"),
        graded(id="q2", essential_ids=[], acceptable_ids=[], expected_behavior="abstain"),
        graded(
            id="q3", essential_ids=[], acceptable_ids=[], expected_doc_ids=[], expected_behavior="abstain"
        ),
        q(id="q4", expected_behavior="abstain"),
    ]
    one, two, three, four = load_questions(write(tmp_path, *lines))
    assert (one.expected_doc_ids, one.essential_ids) == ((), None)
    assert (two.expected_doc_ids, two.essential_ids, two.acceptable_ids) == ((), (), ())
    assert three.expected_doc_ids == ()
    assert four.expected_doc_ids == ("d1",)  # an abstention may list documents: its context documents


def test_optional_fields_may_be_null_or_left_out(tmp_path):
    nulls = dict.fromkeys(
        ("expected_behavior", "essential_ids", "acceptable_ids", "must_include", "must_not_include"), None
    )
    one, two, three = load_questions(
        write(
            tmp_path,
            q(**nulls, run_unfiltered=None),
            q(id="q2", run_unfiltered=False),
            q(
                id="q3",
                must_include=[],
                must_not_include=[],
                filters={"doc_types": ["A1"]},
                run_unfiltered=True,
            ),
        )
    )
    assert one == Question(id="q1", question="What was decided?", expected_doc_ids=("d1",), line=1)
    assert one.to_dict() == {"id": "q1", "question": "What was decided?", "expected_doc_ids": ["d1"]}
    assert two.run_unfiltered is False  # false asks for nothing, with or without filters
    assert (three.must_include, three.must_not_include, three.run_unfiltered) == ((), (), True)


@pytest.mark.parametrize(
    ("lines", "line", "match"),
    [
        (
            [q(expected_behavior="refuse")],
            1,
            "'expected_behavior' must be one of answer, correct_premise, abstain",
        ),
        ([q(expected_behavior="Abstain")], 1, "'expected_behavior'"),
        ([q(expected_behavior="")], 1, "'expected_behavior'"),
        ([q(), q(id="q2", expected_behavior=3)], 2, "'expected_behavior' must be a string"),
        ([q(expected_behavior=["abstain"])], 1, "'expected_behavior'"),
        # No expected document, on a question that does not abstain.
        ([q(expected_doc_ids=[])], 1, "'expected_doc_ids' is empty.*'abstain'"),
        ([q(expected_doc_ids=[], expected_behavior="answer")], 1, "'expected_doc_ids' is empty"),
        ([q(expected_doc_ids=[], expected_behavior="correct_premise")], 1, "'expected_doc_ids' is empty"),
        ([graded(essential_ids=[], acceptable_ids=[])], 1, "'essential_ids' and 'acceptable_ids' are empty"),
        (
            [q(), graded(id="q2", essential_ids=[], acceptable_ids=[], expected_behavior="correct_premise")],
            2,
            "are empty",
        ),
        # The lists themselves.
        ([graded(essential_ids="d1")], 1, "'essential_ids' must be a list"),
        ([graded(essential_ids=["d1", ""])], 1, "'essential_ids'"),
        ([graded(essential_ids=["d1", 7])], 1, "'essential_ids'"),
        ([graded(essential_ids=["d1", "d1"])], 1, "'essential_ids' names 'd1' twice"),
        ([graded(acceptable_ids={"d2": 1})], 1, "'acceptable_ids' must be a list"),
        ([graded(acceptable_ids=["d2", " "])], 1, "'acceptable_ids'"),
        ([graded(acceptable_ids=["d2", "d2"])], 1, "'acceptable_ids' names 'd2' twice"),
        ([graded(acceptable_ids=["d2", "d1"])], 1, "both name 'd1'"),
        ([q(), q(id="q2"), graded(id="q3", essential_ids=["d1", "d2"])], 3, "both name 'd2'"),
        # expected_doc_ids, when given with them, is their union.
        ([graded(expected_doc_ids=["d1"])], 1, "'expected_doc_ids' must name.*it lacks 'd2'"),
        ([graded(expected_doc_ids=["d1", "d2", "d3"])], 1, "'expected_doc_ids' must name.*it adds 'd3'"),
        ([graded(expected_doc_ids=["d1", "d3"])], 1, "it lacks 'd2' and it adds 'd3'"),
        ([graded(expected_doc_ids=[])], 1, "it lacks 'd1', 'd2'"),
        ([graded(acceptable_ids=..., expected_doc_ids=["d1", "d2"])], 1, "it adds 'd2'"),
        ([graded(essential_ids=..., expected_doc_ids=["d1", "d2"])], 1, "it adds 'd1'"),
        (
            [q(), graded(id="q2", expected_doc_ids=["d2", "d2", "d1"])],
            2,
            "'expected_doc_ids' names 'd2' twice",
        ),
        ([graded(expected_doc_ids="d1")], 1, "'expected_doc_ids' must be a list"),
        # Without them it stays required.
        ([q(expected_doc_ids=...)], 1, "'expected_doc_ids' is missing"),
        ([q(expected_doc_ids=..., essential_ids=None)], 1, "'expected_doc_ids' is missing"),
        ([q(expected_doc_ids=None)], 1, "'expected_doc_ids' must be a list"),
        # What a right answer holds.
        ([q(must_include="5-4")], 1, "'must_include' must be a list"),
        ([q(must_include=["5-4", ""])], 1, "'must_include' must be a list of non-blank strings"),
        ([q(must_include=["5-4", 4])], 1, "'must_include'"),
        ([q(must_include=["5-4", "5-4"])], 1, "'must_include' names '5-4' twice"),
        ([q(must_not_include="unanimous")], 1, "'must_not_include' must be a list"),
        ([q(), q(id="q2", must_not_include=["  "])], 2, "'must_not_include'"),
        ([q(must_not_include=[None])], 1, "'must_not_include'"),
        # A second run without filters needs filters.
        ([q(run_unfiltered="yes")], 1, "'run_unfiltered' must be true or false"),
        ([q(run_unfiltered=1)], 1, "'run_unfiltered' must be true or false"),
        ([q(run_unfiltered=True)], 1, "'run_unfiltered' is true on a question without filters"),
        ([q(), q(id="q2", run_unfiltered=True, filters={})], 2, "without filters"),
        ([q(run_unfiltered=True, filters={"doc_types": None})], 1, "without filters"),
        ([q(run_unfiltered=True, filters=None)], 1, "without filters"),
        # Fields that do not exist are still refused.
        ([q(essential=["d1"])], 1, "unknown field 'essential'"),
        ([q(must_includes=["x"])], 1, "unknown field 'must_includes'"),
        ([q(expected_behaviour="abstain")], 1, "unknown field 'expected_behaviour'"),
        ([q(), q(id="q2", run_unfiltred=True)], 2, "unknown field 'run_unfiltred'"),
        ([q(answer="5-4")], 1, "unknown field 'answer'"),
    ],
)
def test_a_bad_line_of_the_second_version_is_refused_with_its_line(tmp_path, lines, line, match):
    path = write(tmp_path, *lines)
    with pytest.raises(EvalError, match=match) as info:
        load_questions(path)
    assert f"questions.jsonl, line {line}:" in str(info.value)


def test_an_id_may_not_be_the_one_of_a_second_run(tmp_path):
    again = q(id="q2", filters={"doc_types": ["A1"]}, run_unfiltered=True)
    with pytest.raises(EvalError, match=r"line 1: the id 'q2#unfiltered'.*'q2' \(line 3"):
        load_questions(write(tmp_path, q(id="q2#unfiltered"), q(), again))
    # Without a second run the id is an id like another.
    assert len(load_questions(write(tmp_path, q(id="q2#unfiltered"), q(id="q2")))) == 2
    with pytest.raises(EvalError, match="'q1#unfiltered' is the one the second run of 'q1'"):
        evaluate(
            [
                Question(
                    id="q1",
                    question="alpha",
                    expected_doc_ids=("a",),
                    filters=SearchFilters(doc_ids=["a"]),
                    run_unfiltered=True,
                ),
                Question(id="q1#unfiltered", question="alpha", expected_doc_ids=("a",)),
            ],
            collection="c",
            store=small_store({"a": ["alpha"]}),
            embedder=FakeEmbedder(SPEC),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"expected_behavior": "maybe"}, "expected_behavior"),
        ({"expected_behavior": None}, "expected_behavior"),
        ({"expected_doc_ids": None}, "'expected_doc_ids' is missing"),
        ({"expected_doc_ids": ()}, "empty"),
        ({"expected_doc_ids": None, "essential_ids": ("a",), "acceptable_ids": ("a",)}, "both name 'a'"),
        ({"essential_ids": ("b",)}, "it lacks 'b' and it adds 'a'"),
        ({"essential_ids": ("a",), "acceptable_ids": ("b",)}, "it lacks 'b'"),
        ({"essential_ids": "a"}, "essential_ids"),
        ({"must_include": "x"}, "must_include"),
        ({"must_not_include": ("",)}, "must_not_include"),
        ({"run_unfiltered": True}, "without filters"),
        ({"run_unfiltered": "true", "filters": SearchFilters(doc_ids=["a"])}, "run_unfiltered"),
        ({"run_unfiltered": True, "filters": SearchFilters()}, "without filters"),
    ],
)
def test_a_question_built_in_code_follows_the_same_rules(kwargs, match):
    values = {"id": "q1", "question": "alpha", "expected_doc_ids": ("a",)}
    values.update(kwargs)
    with pytest.raises(EvalError, match=match):
        Question(**values)


def test_a_question_built_in_code_keeps_lists_as_tuples():
    built = Question(
        id="q1", question="alpha", essential_ids=["a"], acceptable_ids=["b"], must_include=["x"], tags=["t"]
    )
    assert built.expected_doc_ids == ("a", "b")
    assert (built.essential_ids, built.acceptable_ids, built.must_include) == (("a",), ("b",), ("x",))
    empty = Question(id="q2", question="alpha", expected_doc_ids=[], expected_behavior="abstain")
    assert (empty.expected_doc_ids, empty.essential_ids) == ((), None)


# ---------------------------------------------------------------------------
# The second fixture: exact values
# ---------------------------------------------------------------------------
def exactly(group: Aggregate, questions, documents, per_question, per_document):
    assert (group.questions, group.documents) == (questions, documents)
    got = (group.hit_at_1, group.hit_at_k, group.mrr, group.recall_at_k)
    assert got == pytest.approx([float(value) for value in per_question], abs=1e-12)
    means = group.per_document
    got = (means.hit_at_1, means.hit_at_k, means.mrr, means.recall_at_k)
    assert got == pytest.approx([float(value) for value in per_document], abs=1e-12)


def essential_exactly(group: Aggregate, questions, documents, unreachable, complete, mean, satisfied, needed):
    """``complete`` questions have all they need; ``satisfied`` of ``needed`` requirements; ``mean`` share."""
    essential = group.essential
    assert isinstance(essential, EssentialAggregate)
    assert (essential.questions, essential.documents, essential.unreachable) == (
        questions,
        documents,
        unreachable,
    )
    assert essential.all_essential_questions == complete
    assert essential.all_essential_at_k == pytest.approx(complete / questions, abs=1e-12)
    assert essential.essential_recall_at_k == pytest.approx(float(mean), abs=1e-12)
    assert (essential.requirements_satisfied, essential.requirements) == (satisfied, needed)
    assert essential.pooled_essential_recall_at_k == pytest.approx(satisfied / needed, abs=1e-12)
    assert isinstance(essential.per_document, EssentialMeans)


SCORED = (1, 2, 3, 5, 6, 7, 8, 9, 12, 13)


def test_what_is_scored_and_what_is_not(report):
    assert [result.id for result in report.results] == [f"fx2-q{n:02d}" for n in SCORED]
    assert [result.id for result in report.unfiltered_reruns] == ["fx2-q07#unfiltered", "fx2-q08#unfiltered"]
    # Every question that abstains, with a context document (fx2-q04) or without.
    assert [entry.id for entry in report.abstentions] == ["fx2-q04", "fx2-q10", "fx2-q11"]
    assert report.not_in_collection == ()
    assert [result.id for result in report.searches] == [
        "fx2-q01",
        "fx2-q02",
        "fx2-q03",
        "fx2-q05",
        "fx2-q06",
        "fx2-q07",
        "fx2-q07#unfiltered",
        "fx2-q08",
        "fx2-q08#unfiltered",
        "fx2-q09",
        "fx2-q12",
        "fx2-q13",
    ]


def test_per_question_values_of_the_second_fixture(report):
    by_id = {result.id: result for result in report.searches}

    def values(id_):
        result = by_id[id_]
        return (
            result.rank,
            result.hit_at_1,
            result.reciprocal_rank,
            result.recall_at_k,
            result.essential_present,
            result.essential_recall_at_k,
            result.all_essential_at_k,
            result.essential_unreachable,
        )

    # The acceptable document is first (equal scores, ordered by doc_id) and the essential one second:
    # a hit at rank 1, and the essential document is among the first ten.
    assert values("fx2-q01") == (1, True, 1.0, 1.0, 1, 1.0, True, False)
    assert [document.doc_id for document in by_id["fx2-q01"].top_documents[:2]] == [
        "fx-ecb-combined-2016-03",
        "fx-ecb-decision-2016-03",
    ]
    assert by_id["fx2-q01"].tied_at_decision is False
    # The acceptable document is not in the corpus: recall counts the documents that are.
    assert values("fx2-q02") == (1, True, 1.0, 1.0, 1, 1.0, True, False)
    assert by_id["fx2-q02"].absent_doc_ids == ("fx-us-statement-2016-01",)
    # Essential documents not stated: the two metrics are not computed.
    assert values("fx2-q03") == (1, True, 1.0, 1.0, None, None, None, False)
    assert "fx2-q04" not in by_id  # it abstains
    # Thirteen essential documents and k = 10: ten of them are among the first ten, whichever.
    assert values("fx2-q05") == (
        1,
        True,
        1.0,
        pytest.approx(10 / 13),
        13,
        pytest.approx(10 / 13),
        False,
        True,
    )
    assert by_id["fx2-q05"].documents_ranked == 13
    # Minutes only: one essential document of two, and the acceptable one.
    assert values("fx2-q06") == (1, True, 1.0, pytest.approx(2 / 3), 2, 0.5, False, False)
    assert by_id["fx2-q06"].documents_ranked == 4
    assert values("fx2-q07") == (1, True, 1.0, 1.0, 1, 1.0, True, False)
    assert values("fx2-q07#unfiltered") == (2, False, 0.5, 1.0, 1, 1.0, True, False)
    assert values("fx2-q08") == (1, True, 1.0, 0.5, 2, 0.5, False, False)
    assert values("fx2-q08#unfiltered") == (1, True, 1.0, 1.0, 2, 1.0, True, False)
    # The acceptable document is a hit; the essential one is left out by the filter.
    assert values("fx2-q09") == (1, True, 1.0, 0.5, 1, 0.0, False, False)
    assert by_id["fx2-q09"].top_documents[0].doc_id == "fx-ecb-account-2016-04"
    # One requirement, two documents that hold the chunk: both first, no essential_ids.
    assert values("fx2-q12") == (1, True, 1.0, 1.0, 0, 1.0, True, False)
    assert (by_id["fx2-q12"].requirements_present, by_id["fx2-q12"].requirements_satisfied) == (1, 1)
    assert by_id["fx2-q12"].tied_at_decision is False
    # The filters leave the combined document alone: it satisfies the group; the decision is left out.
    assert values("fx2-q13") == (1, True, 1.0, pytest.approx(1 / 3), 1, 0.5, False, False)
    assert (by_id["fx2-q13"].requirements_present, by_id["fx2-q13"].requirements_satisfied) == (2, 1)
    assert [document.doc_id for document in by_id["fx2-q13"].top_documents] == ["fx-ecb-combined-2016-03"]
    assert [result.expected_behavior for result in report.results] == [
        "answer",
        "correct_premise",
        "answer",
        "answer",
        "answer",
        "answer",
        "answer",
        "correct_premise",
        "answer",
        "answer",
    ]
    assert [result.id for result in report.results if result.expected_beyond_k] == ["fx2-q05"]


def test_a_second_run_is_the_question_without_its_filters(report, questions):
    written = {result.id: result for result in report.results}
    for rerun in report.unfiltered_reruns:
        original = written[rerun.rerun_of]
        assert rerun.id == original.id + UNFILTERED_SUFFIX == original.id + "#unfiltered"
        assert rerun.kind == "graded+filters#unfiltered" and original.kind == "graded+filters"
        assert rerun.tags == (*original.tags, UNFILTERED_TAG) and UNFILTERED_TAG == "unfiltered-rerun"
        assert rerun.filters is None and original.filters is not None
        assert (rerun.question, rerun.expected_doc_ids) == (original.question, original.expected_doc_ids)
        assert (rerun.essential_ids, rerun.must_include) == (original.essential_ids, original.must_include)
        assert rerun.documents_ranked == 13 and original.documents_ranked == 2
        assert original.rerun_of is None
    # Without the filter the combined document, not expected by fx2-q07, ties with the decision.
    assert [result.tied_at_decision for result in report.unfiltered_reruns] == [True, False]
    assert report.tied_at_decision == 1
    # The questions that did not ask for it are run once.
    assert {rerun.rerun_of for rerun in report.unfiltered_reruns} == {
        question.id for question in questions if question.run_unfiltered
    }


def test_kinds_and_headline_of_the_second_fixture(report):
    assert list(report.by_kind) == ["graded", "graded+filters", "graded+filters#unfiltered"]
    # The second runs carry no filters: their kind is asked without filters and is in the headline.
    assert report.filtered_kinds == ("graded+filters",)
    assert report.headline_kinds == ("graded", "graded+filters#unfiltered")
    exactly(
        report.by_kind["graded"],
        5,
        13,
        per_question=(1, 1, 1, F(62, 65)),
        per_document=(1, 1, 1, F(138, 169)),
    )
    # The five questions as written: the second runs (one of them second) are not here.
    exactly(
        report.by_kind["graded+filters"],
        5,
        7,
        per_question=(1, 1, 1, F(3, 5)),
        per_document=(1, 1, 1, F(71, 126)),
    )
    exactly(
        report.by_kind["graded+filters#unfiltered"],
        2,
        3,
        per_question=(F(1, 2), 1, F(3, 4), 1),
        per_document=(F(2, 3), 1, F(5, 6), 1),
    )
    # Five questions and two second runs. The decision has three (first, second without the filter,
    # first), the twelve other documents are always first.
    exactly(
        report.headline,
        7,
        13,
        per_question=(F(6, 7), 1, F(13, 14), F(88, 91)),
        per_document=(F(38, 39), 1, F(77, 78), F(557, 676)),
    )
    # Pooled: each question once, as written. No second run, no question that abstains.
    overall = report.overall
    assert (overall.questions, overall.documents) == (10, 13)
    assert (overall.hit_at_1, overall.hit_at_k, overall.mrr) == (1.0, 1.0, 1.0)
    assert overall.recall_at_k == pytest.approx(float(F(101, 130)), abs=1e-12)


def test_essential_metrics_of_the_second_fixture(report):
    # graded: fx2-q01, 02 and 12 have their one requirement; fx2-q05 ten of thirteen; fx2-q03 does not
    # state any and is left out. Three questions of four; 13 requirements of 16.
    graded = report.by_kind["graded"]
    essential_exactly(graded, 4, 13, 1, complete=3, mean=F(49, 52), satisfied=13, needed=16)
    # graded+filters: 1 of 2, 1 of 1, 1 of 2, 0 of 1 and 1 of 2 (the group, by the combined document).
    essential_exactly(
        report.by_kind["graded+filters"], 5, 5, 0, complete=1, mean=F(1, 2), satisfied=4, needed=8
    )
    essential_exactly(
        report.by_kind["graded+filters#unfiltered"], 2, 3, 0, complete=2, mean=1, satisfied=3, needed=3
    )
    essential_exactly(report.headline, 6, 13, 1, complete=5, mean=F(25, 26), satisfied=16, needed=19)
    essential_exactly(report.overall, 9, 13, 1, complete=4, mean=F(163, 234), satisfied=17, needed=24)
    # The share of questions is not the per-document mean kept in the JSON report: there fx2-q05
    # counts for each of its thirteen documents.
    assert graded.essential.all_essential_at_k == 0.75
    assert graded.essential.per_document == EssentialMeans(
        pytest.approx(float(F(136, 169))), pytest.approx(float(F(2, 13)))
    )
    assert report.headline.essential.per_document == EssentialMeans(
        pytest.approx(float(F(275, 338))), pytest.approx(float(F(5, 26)))
    )
    assert report.essential_unreachable == 1
    assert [result.id for result in report.results if result.essential_unreachable] == ["fx2-q05"]
    # No interval on six questions, and none asked for the pooled line.
    essential = report.headline.essential
    assert (essential.question_intervals, essential.few_questions) == (None, True)
    assert (essential.intervals, essential.few_documents) == (None, True)
    assert (report.overall.essential.question_intervals, report.overall.essential.few_questions) == (
        None,
        False,
    )
    # The essential metrics are not the expected-document ones under another name.
    filtered = report.by_kind["graded+filters"]
    assert filtered.essential.essential_recall_at_k != pytest.approx(filtered.recall_at_k)
    assert report.headline.essential.essential_recall_at_k != pytest.approx(report.headline.recall_at_k)


def test_breakdown_by_expected_behavior(report):
    # The three questions that abstain are scored nowhere: there is no line for them.
    assert list(report.by_behavior) == ["answer", "correct_premise"]
    answer, premise = report.by_behavior.values()
    assert (answer.questions, answer.documents) == (8, 13)
    assert (answer.hit_at_1, answer.mrr) == (1.0, 1.0)
    assert answer.recall_at_k == pytest.approx(float(F(163, 208)), abs=1e-12)
    essential_exactly(answer, 7, 13, 1, complete=3, mean=F(137, 182), satisfied=16, needed=22)
    # fx2-q02 and fx2-q09: a premise to correct is scored on its documents like any question.
    exactly(premise, 2, 3, per_question=(1, 1, 1, F(3, 4)), per_document=(1, 1, 1, F(2, 3)))
    essential_exactly(premise, 2, 2, 0, complete=1, mean=F(1, 2), satisfied=1, needed=2)
    assert sum(group.questions for group in report.by_behavior.values()) == len(report.results) == 10
    assert all(group.intervals is None and not group.few_documents for group in report.by_behavior.values())


def test_breakdown_by_tag_counts_the_second_runs_under_their_tags(report):
    exactly(
        report.by_tag["unfiltered-rerun"],
        2,
        3,
        per_question=(F(1, 2), 1, F(3, 4), 1),
        per_document=(F(2, 3), 1, F(5, 6), 1),
    )
    exactly(
        report.by_tag["false-premise"], 2, 3, per_question=(1, 1, 1, F(3, 4)), per_document=(1, 1, 1, F(2, 3))
    )
    # fx2-q05, fx2-q08 and the second run of fx2-q08.
    assert (report.by_tag["chain"].questions, report.by_tag["chain"].documents) == (3, 13)
    # The questions that abstain give no line: nothing of theirs is scored.
    assert "out-of-corpus" not in report.by_tag and "bank:gb" in report.by_tag
    assert report.by_tag["bank:gb"].questions == 1  # fx2-q06; fx2-q04 abstains
    assert "short-query" not in report.by_tag and "abstention" not in report.by_kind
    assert report.by_tag["either-of"].questions == 2


def test_what_the_filter_brings(report):
    effect = report.filter_effect
    assert isinstance(effect, FilterEffect)
    assert (effect.questions, effect.essential_questions) == (2, 2)
    assert [(written.id, rerun.id) for written, rerun in effect.pairs] == [
        ("fx2-q07", "fx2-q07#unfiltered"),
        ("fx2-q08", "fx2-q08#unfiltered"),
    ]
    # With the filters both are first, and one loses an essential document; without, one is second.
    assert effect.filtered == {"hit_at_1": 1.0, "hit_at_k": 1.0, "mrr": 1.0, "essential_recall_at_k": 0.75}
    assert effect.unfiltered == {"hit_at_1": 0.5, "hit_at_k": 1.0, "mrr": 0.75, "essential_recall_at_k": 1.0}
    data = effect.to_dict()
    assert data["questions"] == 2 and data["filtered"] == effect.filtered
    assert data["pairs"][0] == {
        "id": "fx2-q07",
        "filtered": {
            "rank": 1,
            "hit_at_1": True,
            "hit_at_k": True,
            "reciprocal_rank": 1.0,
            "essential_recall_at_k": 1.0,
        },
        "unfiltered": {
            "rank": 2,
            "hit_at_1": False,
            "hit_at_k": True,
            "reciprocal_rank": 0.5,
            "essential_recall_at_k": 1.0,
        },
    }


def test_abstention_questions_are_searched_and_not_scored(report):
    fourth, tenth, eleventh = report.abstentions
    assert isinstance(tenth, Abstention)
    # fx2-q04 lists a context document, which holds the text of the question: it comes first, and
    # that is not a hit. One context document, among the first ten.
    assert (fourth.top_document.doc_id, fourth.kind) == ("fx-gb-report-2016-08", "graded")
    assert (fourth.context_doc_ids, fourth.absent_context_doc_ids, fourth.context_at_k) == (
        ("fx-gb-report-2016-08",),
        (),
        1,
    )
    assert (tenth.context_doc_ids, tenth.context_at_k) == ((), None)
    # The text is a chunk of the minutes of the Bank of Japan: that document is first, with a score of 1.
    assert tenth.top_document.doc_id == "fx-jp-minutes"
    assert tenth.top_document.title == "Minutes of the Monetary Policy Meeting on January 28 and 29, 2016"
    assert tenth.top_score == pytest.approx(1.0, abs=1e-5)
    assert (tenth.chunks_returned, tenth.documents_ranked) == (32, 13)
    # The levels in increasing order: every document is at or above -1, one only reaches 0.999.
    assert tenth.documents_at_or_above == ((-1.0, 13), (0.999, 1))
    assert (tenth.kind, tenth.tags, tenth.filters, tenth.rerun_of) == (
        "abstention",
        ("out-of-corpus",),
        None,
        None,
    )
    assert tenth.must_not_include == ("minus 0.1 percent",)
    assert len(tenth.top_documents) == 10
    # The filter of fx2-q11 leaves nothing.
    assert (eleventh.top_document, eleventh.top_score, eleventh.documents_ranked) == (None, None, 0)
    assert eleventh.documents_at_or_above == ((-1.0, 0), (0.999, 0))
    assert eleventh.filters == SearchFilters(source_codes=["ch"])
    # They are in no result, no kind, no tag, no miss, no count of the scored questions.
    scored = {result.id for result in report.searches}
    assert not scored & {"fx2-q04", "fx2-q10", "fx2-q11"}
    assert not {result.id for result in report.misses} & {"fx2-q04", "fx2-q10", "fx2-q11"}
    assert report.overall.questions == 10 and report.fewer_than_k_documents == 5
    # For comparison: nine scored questions are the text of a chunk they can reach, so their first
    # document scores 1; the filters of fx2-q13 leave a document that does not hold its text.
    top = report.top_scores
    assert isinstance(top, TopScores) and top.questions == 10
    assert (top.median, top.max) == pytest.approx((1.0, 1.0), abs=1e-5) and top.min < 0.999


def test_the_first_fixture_has_none_of_this(store):
    old = evaluate(load_questions(QUESTIONS), collection=COLLECTION, store=store, embedder=fixture_embedder())
    assert (old.unfiltered_reruns, old.abstentions, old.filter_effect) == ((), (), None)
    assert list(old.by_behavior) == ["answer"] and old.by_behavior["answer"].questions == 12
    assert old.by_behavior["answer"].per_document == old.overall.per_document
    assert all(group.essential is None for group in (old.overall, old.headline, *old.by_kind.values()))
    assert all(result.essential_recall_at_k is None for result in old.results)
    assert all(result.essential_ids is None and result.rerun_of is None for result in old.results)
    assert old.essential_unreachable == 0
    assert old.searches == old.results
    text = format_report(old)
    for word in (
        "essential",
        "expected_behavior",
        "Abstention",
        "What the filter brings",
        "#unfiltered",
        "Per question",
        "More expected documents",
        "requirement",
        "context@",
    ):
        assert word not in text
    assert all(result.requirements_present is None and not result.essential_any_of for result in old.results)


def test_a_larger_k_reaches_every_essential_document(store, questions):
    report = evaluate(questions, collection=COLLECTION, store=store, embedder=fixture_embedder(), k=13)
    fifth = {result.id: result for result in report.results}["fx2-q05"]
    assert (fifth.essential_recall_at_k, fifth.all_essential_at_k, fifth.essential_unreachable) == (
        1.0,
        True,
        False,
    )
    assert report.essential_unreachable == 0 and report.headline.essential.unreachable == 0
    assert "More requirements" not in format_report(report)
    assert "More expected documents" not in format_report(report) and report.expected_beyond_k == 0
    one = evaluate(questions, collection=COLLECTION, store=store, embedder=fixture_embedder(), k=1)
    by_id = {result.id: result for result in one.searches}
    # At k = 1: one of thirteen; the acceptable document of fx2-q01 takes the only place.
    assert by_id["fx2-q05"].essential_recall_at_k == pytest.approx(1 / 13)
    assert (by_id["fx2-q01"].hit_at_k, by_id["fx2-q01"].essential_recall_at_k) == (True, 0.0)
    assert by_id["fx2-q01"].all_essential_at_k is False
    assert (by_id["fx2-q08"].essential_unreachable, by_id["fx2-q07"].essential_unreachable) == (True, False)
    # A group is one requirement, whatever the number of its documents: fx2-q12 can be satisfied.
    assert (by_id["fx2-q12"].essential_unreachable, by_id["fx2-q12"].all_essential_at_k) == (False, True)
    assert one.essential_unreachable == 4  # fx2-q05, 06, 08 and 13: each question once


def test_intervals_cover_the_essential_metrics_on_the_kinds_and_the_headline(store, questions):
    report = evaluate(
        questions,
        collection=COLLECTION,
        store=store,
        embedder=fixture_embedder(),
        interval_min_documents=4,
        interval_min_questions=4,
        bootstrap_resamples=200,
    )
    assert report.parameters["interval_min_questions"] == 4
    essential = report.headline.essential
    assert essential.few_questions is False
    assert set(essential.question_intervals) == {
        "essential_recall_at_k",
        "all_essential_at_k",
        "pooled_essential_recall_at_k",
    }
    # Six questions drawn again: five have all they need, and one has ten of thirteen.
    low, high = essential.question_intervals["all_essential_at_k"]
    assert 0.0 <= low <= 5 / 6 <= high <= 1.0 and low < high
    low, high = essential.question_intervals["essential_recall_at_k"]
    assert 10 / 13 <= low <= 25 / 26 <= high <= 1.0 and low < high
    low, high = essential.question_intervals["pooled_essential_recall_at_k"]
    assert 10 / 13 <= low <= 16 / 19 <= high <= 1.0 and low < high
    assert set(report.headline.intervals) == {"hit_at_1", "hit_at_k", "mrr"}  # as before
    assert report.by_kind["graded+filters"].essential.question_intervals is not None
    assert report.by_kind["graded+filters#unfiltered"].essential.few_questions is True
    assert report.overall.essential.question_intervals is None
    assert report.by_tag["chain"].essential.question_intervals is None
    # The interval of the first form, over the documents, is still in the JSON report.
    assert set(essential.intervals) == {"essential_recall_at_k", "all_essential_at_k"}
    # The same results give the same intervals.
    again = evaluate(
        questions,
        collection=COLLECTION,
        store=store,
        embedder=fixture_embedder(),
        interval_min_documents=4,
        interval_min_questions=4,
        bootstrap_resamples=200,
    )
    assert again.headline.essential.question_intervals == essential.question_intervals
    text = format_report(report)
    assert "95% intervals, from 200 resamples of the questions (seed 1):" in text
    lines = text.splitlines()
    start = lines.index("95% intervals, from 200 resamples of the questions (seed 1):")
    low, high = essential.question_intervals["all_essential_at_k"]
    (headline,) = [line for line in lines[start:] if line.startswith("headline")]
    assert headline.split()[1:3] == f"[{low:.2f}, {high:.2f}]".split()  # the first column: all_essential
    assert "kind: graded+filters#unfiltered: fewer than 4 questions: no interval" in lines[start:]
    with pytest.raises(EvalError, match="interval_min_questions must be an integer >= 1"):
        evaluate(
            questions,
            collection=COLLECTION,
            store=store,
            embedder=fixture_embedder(),
            interval_min_questions=0,
        )


# ---------------------------------------------------------------------------
# The report of the second fixture
# ---------------------------------------------------------------------------
def test_json_report_of_the_second_fixture(report, questions):
    data = json.loads(report.to_json())
    assert list(data) == [
        "parameters",
        "summary",
        "metrics",
        "misses",
        "not_in_collection",
        "results",
        "unfiltered_reruns",
        "filter_effect",
        "abstentions",
    ]
    assert data["parameters"]["score_levels"] == [-1.0, 0.999]
    assert data["summary"] == {
        "questions": 13,
        "scored": 10,
        "not_in_collection": 0,
        "partially_absent": 1,
        "fewer_than_k_documents": 5,
        "chunk_budget_exhausted": 0,
        "tied_at_decision": 1,
        "unfiltered_reruns": 2,
        "abstentions": 3,
        "essential_unreachable": 1,
        "expected_beyond_k": 1,
    }
    assert len(data["results"]) == 10 and len(data["unfiltered_reruns"]) == 2
    first = data["results"][0]
    assert first["expected_behavior"] == "answer"
    assert first["essential_ids"] == ["fx-ecb-decision-2016-03"]
    assert first["acceptable_ids"] == ["fx-ecb-combined-2016-03"]
    # Carried for a scorer of answers, as the question gives them.
    assert first["must_include"] == ["80 billion euro", "April"]
    assert first["must_not_include"] == ["60 billion euro"]
    assert (first["essential_recall_at_k"], first["all_essential_at_k"]) == (1.0, True)
    assert (first["essential_present"], first["essential_unreachable"], first["rerun_of"]) == (1, False, None)
    third = data["results"][2]
    assert (third["essential_ids"], third["acceptable_ids"]) == (None, None)
    assert (third["essential_recall_at_k"], third["all_essential_at_k"]) == (None, None)
    assert (third["must_include"], third["must_not_include"]) == ([], [])
    assert (first["requirements_present"], first["requirements_satisfied"]) == (1, 1)
    assert (first["essential_any_of"], first["expected_beyond_k"]) == ([], False)
    assert (third["requirements_present"], third["requirements_satisfied"]) == (None, None)
    fifth = data["results"][3]
    assert (fifth["id"], fifth["essential_unreachable"], fifth["expected_beyond_k"]) == (
        "fx2-q05",
        True,
        True,
    )
    thirteenth = data["results"][9]
    assert thirteenth["essential_any_of"] == [["fx-ecb-statement-2016-03", "fx-ecb-combined-2016-03"]]
    assert (thirteenth["requirements_present"], thirteenth["requirements_satisfied"]) == (2, 1)
    assert (thirteenth["essential_present"], thirteenth["essential_recall_at_k"]) == (1, 0.5)
    rerun = data["unfiltered_reruns"][0]
    assert (rerun["id"], rerun["rerun_of"], rerun["kind"]) == (
        "fx2-q07#unfiltered",
        "fx2-q07",
        "graded+filters#unfiltered",
    )
    assert "filters" not in rerun and rerun["tags"][-1] == "unfiltered-rerun"
    metrics = data["metrics"]
    assert list(metrics) == ["headline", "overall", "by_kind", "by_tag", "by_expected_behavior"]
    assert metrics["headline"]["kinds"] == ["graded", "graded+filters#unfiltered"]
    assert metrics["by_kind"]["graded+filters#unfiltered"]["filtered"] is False
    assert metrics["by_kind"]["graded+filters"]["filtered"] is True
    assert metrics["headline"]["essential"] == {
        "questions": 6,
        "documents": 13,
        "unreachable": 1,
        "essential_recall_at_k": pytest.approx(25 / 26),
        "all_essential_at_k": pytest.approx(5 / 6),
        "per_document": {
            "essential_recall_at_k": pytest.approx(275 / 338),
            "all_essential_at_k": pytest.approx(5 / 26),
        },
        "intervals": None,
        "few_documents": True,
        "all_essential_questions": 5,
        "requirements": 19,
        "requirements_satisfied": 16,
        "pooled_essential_recall_at_k": pytest.approx(16 / 19),
        "question_intervals": None,
        "few_questions": True,
    }
    # No question that abstains is scored: there is no aggregate for that behaviour.
    assert list(metrics["by_expected_behavior"]) == ["answer", "correct_premise"]
    assert metrics["by_expected_behavior"]["correct_premise"]["questions"] == 2
    assert data["filter_effect"]["unfiltered"]["mrr"] == 0.75
    abstentions = data["abstentions"]
    assert abstentions["score_levels"] == [-1.0, 0.999]
    assert abstentions["top_score_of_scored_questions"] == {
        "questions": 10,
        "min": pytest.approx(report.top_scores.min),
        "median": pytest.approx(1.0, abs=1e-5),
        "max": pytest.approx(1.0, abs=1e-5),
    }
    fourth, tenth, eleventh = abstentions["questions"]
    assert (fourth["id"], fourth["context_doc_ids"]) == ("fx2-q04", ["fx-gb-report-2016-08"])
    assert (fourth["absent_context_doc_ids"], fourth["context_documents_at_k"]) == ([], 1)
    assert (tenth["context_doc_ids"], tenth["context_documents_at_k"]) == ([], None)
    assert tenth["id"] == "fx2-q10" and tenth["top_document"]["doc_id"] == "fx-jp-minutes"
    assert tenth["top_score"] == tenth["top_document"]["score"] == pytest.approx(1.0, abs=1e-5)
    assert tenth["documents_at_or_above"] == [
        {"score": -1.0, "documents": 13},
        {"score": 0.999, "documents": 1},
    ]
    assert tenth["must_not_include"] == ["minus 0.1 percent"] and tenth["rerun_of"] is None
    assert (eleventh["top_score"], eleventh["top_document"], eleventh["top_documents"]) == (None, None, [])
    assert eleventh["filters"] == {"source_codes": ["ch"]}
    # Nothing in the report judges an abstention.
    assert not {"pass", "passed", "abstained", "threshold"} & set(tenth)


def rows_of(text: str, section: str) -> dict[str, list[str]]:
    """The lines of the table that follows the line starting with ``section``, cut into cells, by name."""
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(section))
    rows = {}
    for line in lines[start + 2 :]:
        cells = [cell.strip() for cell in line.split("  ") if cell.strip()]
        if not cells or len(cells) < 3:
            break
        rows[cells[0]] = cells[1:]
    return rows


def test_text_report_of_the_second_fixture(report):
    text = format_report(report)
    lines = text.splitlines()
    assert all(len(line) <= 120 for line in lines)
    assert not text.endswith("\n")
    assert "10 questions scored (13 read, 0 not in the collection)" in lines[0]
    assert lines[1] == "and 3 questions that abstain (searched, not scored), 2 second runs without filters"
    main = rows_of(text, "questions: ")
    assert main["kind: graded"] == ["5", "13", "1.00", "1.00", "1.00", "0.82", "(few documents)"]
    assert main["kind: graded+filters *"] == ["5", "7", "1.00", "1.00", "1.00", "0.56", "(few documents)"]
    assert main["kind: graded+filters#unfiltered"] == [
        "2",
        "3",
        "0.67",
        "1.00",
        "0.83",
        "1.00",
        "(few documents)",
    ]
    assert main["headline"] == ["7", "13", "0.97", "1.00", "0.99", "0.82", "(few documents)"]
    assert main["overall (pooled)"] == ["10", "13", "1.00", "1.00", "1.00", "0.78"]
    assert "headline: the kinds asked without filters (graded, graded+filters#unfiltered)." in text
    assert "#unfiltered: the questions of a kind that ask for it (run_unfiltered)" in text

    # The same lines per question: fx2-q05 and its thirteen documents count once.
    per_question = rows_of(text, "Per question")
    assert per_question == {
        "kind: graded": ["5", "1.00", "1.00", "1.00", "0.95"],
        "kind: graded+filters *": ["5", "1.00", "1.00", "1.00", "0.60"],
        "kind: graded+filters#unfiltered": ["2", "0.50", "1.00", "0.75", "1.00"],
        "headline": ["7", "0.86", "1.00", "0.93", "0.97"],
    }
    assert "Which to read. Per document (the first table) when several questions ask for one document" in text
    assert "More expected documents than k=10 (1 question: recall@10 cannot reach 1 for it): fx2-q05" in lines
    assert all(len(line) <= 110 for line in lines)

    behaviors = rows_of(text, "Per expected behaviour")
    assert list(behaviors) == [
        "expected_behavior: answer",
        "expected_behavior: correct_premise",
        "expected_behavior: abstain",
    ]
    assert behaviors["expected_behavior: correct_premise"] == ["2", "3", "1.00", "1.00", "1.00", "0.67"]
    # The questions that abstain have a count and no metric.
    assert behaviors["expected_behavior: abstain"] == ["3", "see the Abstention section"]
    assert "correct_premise: scored on the documents that show the premise is false" in text
    assert "abstain: never scored, whether the question lists documents or not" in text

    essential = rows_of(text, "Essential documents")
    # all_essential is a share of questions, with its counts; the recall both ways.
    assert essential == {
        "kind: graded": ["0.75 (3/4)", "0.94", "0.81 (13/16)", "1"],
        "kind: graded+filters *": ["0.20 (1/5)", "0.50", "0.50 (4/8)", "0"],
        "kind: graded+filters#unfiltered": ["1.00 (2/2)", "1.00", "1.00 (3/3)", "0"],
        "headline": ["0.83 (5/6)", "0.96", "0.84 (16/19)", "1"],
        "overall (pooled)": ["0.44 (4/9)", "0.70", "0.71 (17/24)", "1"],
        "expected_behavior: answer": ["0.43 (3/7)", "0.75", "0.73 (16/22)", "1"],
        "expected_behavior: correct_premise": ["0.50 (1/2)", "0.50", "0.50 (1/2)", "0"],
    }
    for legend in (
        "requirement: one document of essential_ids, or one group of essential_any_of",
        "all_essential@10: share of the questions with every requirement satisfied",
        "essential_recall@10: mean over these questions of the share of their requirements satisfied.",
        "pooled: requirements satisfied / requirements, these questions together",
        "unreachable: questions with more requirements than k=10",
        "raise --k (at most --chunk-limit",
        "hit@1, hit@10, MRR and recall@10 look at every expected document",
        "No interval: every line has fewer than 20 questions that state requirements.",
    ):
        assert legend in text
    assert "per-document" not in text[text.index("Essential documents") : text.index("What the filter")]
    assert "More requirements than k=10 (1):\n  fx2-q05: 13 requirements" in text

    effect = rows_of(text, "What the filter brings (2 questions")
    assert effect == {
        "with their filters": ["2", "1.00", "1.00", "1.00", "0.75"],
        "without filters": ["2", "0.50", "1.00", "0.75", "1.00"],
    }

    assert "Abstention (3 questions whose expected behaviour is to abstain" in text
    abstentions = rows_of(text, "Abstention (")
    assert abstentions["fx2-q04"][:4] == ["1.000", "13", "1", "1/1"]
    assert abstentions["fx2-q04"][4] == "fx-gb-report-2016-08"
    assert abstentions["fx2-q10"][:5] == ["1.000", "13", "1", "-", "fx-jp-minutes"]
    assert abstentions["fx2-q11"] == ["n/a", "0", "0", "-", "nothing was returned"]
    assert ">=-1" in text and ">=0.999" in text and "context@10" in text
    assert "top score of the 10 scored questions: min " in text
    assert "No threshold is applied" in text
    assert "tied_at_decision concerns hit@1 and hit@10 only" in text
    assert "Misses (0)" in text


def test_the_default_score_levels_are_reported(store, questions):
    report = evaluate(questions, collection=COLLECTION, store=store, embedder=fixture_embedder())
    assert DEFAULT_SCORE_LEVELS == (0.7, 0.75, 0.8, 0.85, 0.9)
    assert report.parameters["score_levels"] == [0.7, 0.75, 0.8, 0.85, 0.9]
    tenth = report.abstentions[1]
    assert [level for level, _count in tenth.documents_at_or_above] == [0.7, 0.75, 0.8, 0.85, 0.9]
    counts = [count for _level, count in tenth.documents_at_or_above]
    assert counts == sorted(counts, reverse=True) and counts[-1] == 1
    assert ">=0.7  >=0.75  >=0.8  >=0.85  >=0.9" in format_report(report)


# ---------------------------------------------------------------------------
# Small collections: the rules one at a time
# ---------------------------------------------------------------------------
SPEC = ModelSpec(name="tiny", dim=8, passage_prefix="", query_prefix="")


def small_store(documents: dict[str, list[str]]):
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client)
    store.ensure_collection("c", SPEC.dim)
    embedder = FakeEmbedder(SPEC)
    points = []
    for doc_id, texts in documents.items():
        for index, vector in enumerate(embedder.encode_passage(texts)):
            payload = {"doc_id": doc_id, "text": texts[index], "chunk_number": index}
            points.append(models.PointStruct(id=len(points) + 1, vector=vector.tolist(), payload=payload))
    if points:
        store.upsert("c", points)
    return store


def ask(store, questions, **kwargs):
    return evaluate(questions, collection="c", store=store, embedder=FakeEmbedder(SPEC), **kwargs)


def hit(doc_id, score, chunk_number=0) -> Hit:
    return Hit(
        text="t",
        score=score,
        doc_id=doc_id,
        title=f"Title of {doc_id}",
        date=None,
        source_code=None,
        doc_type=None,
        language=None,
        page=None,
        chunk_number=chunk_number,
        corpus=None,
        point_id=f"{doc_id}-{chunk_number}",
        payload={},
    )


def answering(monkeypatch, answers, documents=("a", "b", "c", "d", "e")):
    """Make the search return ``answers[(query, filtered)]``: a dict of doc_id -> score."""
    calls = []

    def search(query, *, filters=None, **_kwargs):
        calls.append((query, filters))
        scores = answers[(query, filters is not None)]
        return [hit(doc_id, score) for doc_id, score in scores.items()]

    monkeypatch.setattr("rag_engine.eval.search", search)
    return small_store({name: ["x"] for name in documents}), calls


FILTER = SearchFilters(doc_types=["A1"])


def test_essential_and_acceptable_documents_are_not_confused(monkeypatch):
    # The ranking is a, b, c, d and k = 2. "c" and "d" are beyond the first two.
    ranking = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}
    store, _calls = answering(monkeypatch, {("x", False): ranking})
    questions = [
        Question(id="acceptable-found", question="x", essential_ids=("c",), acceptable_ids=("a",)),
        Question(id="essential-found", question="x", essential_ids=("a",), acceptable_ids=("c",)),
        Question(id="half", question="x", essential_ids=("b", "d"), acceptable_ids=("a", "c")),
        Question(id="none", question="x", essential_ids=("c", "d"), acceptable_ids=()),
        Question(id="all", question="x", essential_ids=("b", "a"), acceptable_ids=("d",)),
    ]
    by_id = {result.id: result for result in ask(store, questions, k=2).results}

    def values(id_):
        result = by_id[id_]
        return (
            result.rank,
            result.hit_at_1,
            result.recall_at_k,
            result.essential_recall_at_k,
            result.all_essential_at_k,
        )

    # An acceptable document first is a hit, and says nothing of the essential one.
    assert values("acceptable-found") == (1, True, 0.5, 0.0, False)
    # The essential one first: the acceptable one that is missed costs recall, not essential recall.
    assert values("essential-found") == (1, True, 0.5, 1.0, True)
    assert values("half") == (1, True, 0.5, 0.5, False)
    assert values("none") == (3, False, 0.0, 0.0, False)
    assert values("all") == (1, True, pytest.approx(2 / 3), 1.0, True)


def test_essential_metrics_count_the_documents_of_the_collection_only(monkeypatch):
    store, _calls = answering(monkeypatch, {("x", False): {"a": 0.9, "b": 0.8, "c": 0.7}})
    questions = [
        Question(id="one-absent", question="x", essential_ids=("a", "ghost"), acceptable_ids=()),
        Question(id="all-absent", question="x", essential_ids=("ghost",), acceptable_ids=("a",)),
        Question(id="no-essential", question="x", essential_ids=(), acceptable_ids=("a",)),
        Question(id="three", question="x", essential_ids=("a", "b", "ghost"), acceptable_ids=()),
        Question(id="four", question="x", essential_ids=("a", "b", "c", "ghost"), acceptable_ids=()),
    ]
    report = ask(store, questions, k=2)
    by_id = {result.id: result for result in report.results}

    def values(id_):
        result = by_id[id_]
        return (
            result.essential_present,
            result.essential_recall_at_k,
            result.all_essential_at_k,
            result.essential_unreachable,
        )

    assert values("one-absent") == (1, 1.0, True, False)
    # No essential document in the collection: scored on the acceptable one, and no essential metric.
    assert values("all-absent") == (0, None, None, False)
    assert (by_id["all-absent"].rank, by_id["all-absent"].hit_at_1) == (1, True)
    assert values("no-essential") == (0, None, None, False)
    # Three essential documents named, two in the collection: both fit in the first two.
    assert values("three") == (2, 1.0, True, False)
    # Three in the collection and k = 2: out of reach, whatever is returned.
    assert values("four") == (3, pytest.approx(2 / 3), False, True)
    essential = report.overall.essential
    assert (essential.questions, essential.unreachable) == (3, 1)  # the two without any are left out
    assert essential.essential_recall_at_k == pytest.approx((1 + 1 + 2 / 3) / 3)
    assert essential.all_essential_at_k == pytest.approx(2 / 3)
    assert essential.documents == 3  # a, b and c; "ghost" is not a document of the collection
    assert report.essential_unreachable == 1


def test_a_second_run_is_counted_in_a_kind_of_its_own(monkeypatch):
    # With the filter "a" is alone; without it "b" comes first.
    answers = {("x", True): {"a": 0.9}, ("x", False): {"b": 0.95, "a": 0.9}, ("y", False): {"c": 0.9}}
    store, calls = answering(monkeypatch, answers)
    questions = [
        Question(
            id="f",
            question="x",
            expected_doc_ids=("a",),
            filters=FILTER,
            kind="k",
            tags=("t",),
            run_unfiltered=True,
        ),
        Question(id="g", question="x", expected_doc_ids=("a",), filters=FILTER, kind="k"),
        Question(id="u", question="y", expected_doc_ids=("c",), kind="plain"),
        Question(id="n", question="x", expected_doc_ids=("a",), filters=FILTER, run_unfiltered=True),
    ]
    report = ask(store, questions)
    assert calls == [("x", FILTER), ("x", None), ("x", FILTER), ("y", None), ("x", FILTER), ("x", None)]
    assert [result.id for result in report.results] == ["f", "g", "u", "n"]
    first, second = report.unfiltered_reruns
    assert (first.id, first.kind, first.tags, first.rank) == (
        "f#unfiltered",
        "k#unfiltered",
        ("t", "unfiltered-rerun"),
        2,
    )
    # A question without a kind: its second run is still apart from "(none)", which carries filters.
    assert (second.id, second.kind, second.tags) == (
        "n#unfiltered",
        "(none)#unfiltered",
        ("unfiltered-rerun",),
    )
    assert report.filtered_kinds == ("(none)", "k")
    assert report.headline_kinds == ("(none)#unfiltered", "k#unfiltered", "plain")
    # The filtered kind holds the two questions as written, both first: the second run is not in it.
    kind = report.by_kind["k"]
    assert (kind.questions, kind.hit_at_1, kind.mrr) == (2, 1.0, 1.0)
    rerun = report.by_kind["k#unfiltered"]
    assert (rerun.questions, rerun.hit_at_1, rerun.mrr) == (1, 0.0, 0.5)
    # The headline: the two second runs (second) and the plain question (first).
    headline = report.headline
    assert (headline.questions, headline.documents) == (3, 2)
    assert (headline.hit_at_1, headline.mrr) == (pytest.approx(1 / 3), pytest.approx(2 / 3))
    assert headline.per_document.hit_at_1 == 0.5  # "a": 0 twice; "c": 1
    # Pooled: the four questions as written, all first. No second run.
    assert (report.overall.questions, report.overall.hit_at_1) == (4, 1.0)
    assert report.by_tag["t"].questions == 2 and report.by_tag["unfiltered-rerun"].questions == 2
    assert report.by_behavior["answer"].questions == 4
    effect = report.filter_effect
    assert effect.questions == 2 and effect.essential_questions == 0
    assert effect.filtered == {"hit_at_1": 1.0, "hit_at_k": 1.0, "mrr": 1.0, "essential_recall_at_k": None}
    assert effect.unfiltered == {"hit_at_1": 0.0, "hit_at_k": 1.0, "mrr": 0.5, "essential_recall_at_k": None}
    data = report.to_dict()
    assert data["summary"]["questions"] == 4 and data["summary"]["scored"] == 4
    assert data["summary"]["unfiltered_reruns"] == 2
    text = format_report(report)
    assert "What the filter brings (2 questions" in text and "n/a" in text
    assert "kind: k#unfiltered" in text and "kind: k *" in text and "kind: k#unfiltered *" not in text


def test_a_missed_second_run_is_a_miss_under_its_own_id(monkeypatch):
    answers = {("x", True): {"a": 0.9}, ("x", False): {"b": 0.9, "c": 0.8}}
    store, _calls = answering(monkeypatch, answers)
    asked = Question(id="f", question="x", expected_doc_ids=("a",), filters=FILTER, run_unfiltered=True)
    report = ask(store, [asked])
    assert [result.id for result in report.misses] == ["f#unfiltered"]
    assert report.to_dict()["misses"][0]["id"] == "f#unfiltered"
    assert "Misses (1)" in format_report(report) and "  f#unfiltered  x" in format_report(report)


def test_a_question_whose_documents_are_absent_is_left_out_once(monkeypatch):
    store, calls = answering(monkeypatch, {})
    asked = Question(id="f", question="x", expected_doc_ids=("ghost",), filters=FILTER, run_unfiltered=True)
    report = ask(store, [asked])
    assert [skipped.id for skipped in report.not_in_collection] == ["f"]
    assert (report.results, report.unfiltered_reruns, calls) == ((), (), [])
    assert report.filter_effect is None and report.top_scores is None


def test_abstentions_do_not_leak_into_any_mean(monkeypatch):
    answers = {
        ("found", False): {"a": 0.9, "b": 0.5},
        ("missed", False): {"b": 0.7, "c": 0.6},
        ("third", False): {"a": 0.5},
        ("nothing", False): {"a": 0.95, "b": 0.85, "c": 0.6, "d": 0.6},
        ("filtered", True): {"e": 0.4},
        ("filtered", False): {"a": 0.8, "e": 0.4},
    }
    store, calls = answering(monkeypatch, answers)
    answerable = [
        Question(id="q1", question="found", expected_doc_ids=("a",), kind="k", tags=("t",)),
        Question(id="q2", question="missed", expected_doc_ids=("a",), kind="k", tags=("t",)),
        Question(id="q3", question="third", essential_ids=("a",), kind="k", tags=("t",)),
    ]
    abstaining = [
        Question(
            id="a1",
            question="nothing",
            expected_doc_ids=(),
            expected_behavior="abstain",
            kind="k",
            tags=("t",),
            must_not_include=("invented",),
        ),
        Question(
            id="a2",
            question="filtered",
            essential_ids=(),
            acceptable_ids=(),
            expected_behavior="abstain",
            kind="k",
            tags=("t",),
            filters=FILTER,
            run_unfiltered=True,
        ),
    ]
    levels = (0.6, 0.85, 0.9)
    alone = ask(store, answerable, score_levels=levels)
    mixed = ask(store, [abstaining[0], *answerable, abstaining[1]], score_levels=levels)
    # The same scored questions, the same numbers: the abstentions change no mean and no count.
    assert mixed.results == alone.results
    for name in (
        "overall",
        "headline",
        "by_kind",
        "by_tag",
        "by_behavior",
        "headline_kinds",
        "filtered_kinds",
    ):
        assert getattr(mixed, name) == getattr(alone, name), name
    assert (mixed.overall.questions, mixed.overall.hit_at_1, mixed.overall.hit_at_k) == (
        3,
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
    )
    assert mixed.misses == alone.misses and [result.id for result in mixed.misses] == ["q2"]
    assert list(mixed.by_behavior) == ["answer"]  # an abstention without documents is in no line
    assert mixed.filtered_kinds == ()  # nor does its filter make its kind a filtered one
    assert mixed.top_scores == alone.top_scores == TopScores(3, 0.5, 0.7, 0.9)
    # Each abstention is searched as written, and again without filters when it asks for that.
    assert [entry.id for entry in mixed.abstentions] == ["a1", "a2", "a2#unfiltered"]
    assert [call for call in calls[-6:] if call[0] in ("nothing", "filtered")] == [
        ("nothing", None),
        ("filtered", FILTER),
        ("filtered", None),
    ]
    first, second, third = mixed.abstentions
    assert (first.top_document.doc_id, first.top_score, first.documents_ranked) == ("a", 0.95, 4)
    assert first.top_document.title == "Title of a"
    assert first.documents_at_or_above == ((0.6, 4), (0.85, 2), (0.9, 1))  # at or above: 0.6 counts
    assert first.must_not_include == ("invented",)
    assert (second.top_score, second.documents_at_or_above, second.rerun_of) == (
        0.4,
        ((0.6, 0), (0.85, 0), (0.9, 0)),
        None,
    )
    assert (third.top_score, third.filters, third.rerun_of) == (0.8, None, "a2")
    assert third.documents_at_or_above == ((0.6, 1), (0.85, 0), (0.9, 0))
    data = mixed.to_dict()
    assert data["summary"]["questions"] == 5 and data["summary"]["scored"] == 3
    assert data["summary"]["abstentions"] == 2 and data["summary"]["unfiltered_reruns"] == 0
    assert data["unfiltered_reruns"] == [] and data["filter_effect"] is None
    assert data["abstentions"]["top_score_of_scored_questions"] == {
        "questions": 3,
        "min": 0.5,
        "median": 0.7,
        "max": 0.9,
    }
    text = format_report(mixed)
    assert "3 questions scored (5 read, 0 not in the collection)" in text
    assert "and 2 questions that abstain (searched, not scored)" in text
    assert "top score of the 3 scored questions: min 0.500, median 0.700, max 0.900." in text
    assert "expected_behavior: answer" in text and "see the Abstention section" in text
    assert "a2#unfiltered" in text and ">=0.6  >=0.85  >=0.9" in text


def test_a_file_of_abstentions_only_is_searched_and_scores_nothing(monkeypatch):
    store, _calls = answering(monkeypatch, {("nothing", False): {"a": 0.5}})
    asked = Question(id="a1", question="nothing", expected_doc_ids=(), expected_behavior="abstain")
    seen = []
    store.existing_doc_ids = lambda *args, **kwargs: seen.append(args) or set()
    report = ask(store, [asked])
    assert seen == []  # there is no document to look up
    assert (report.results, report.not_in_collection) == ((), ())
    assert report.overall == Aggregate(0, 0, None, None, None, None)
    assert report.headline.per_document is None and report.top_scores is None
    assert [entry.top_score for entry in report.abstentions] == [0.5]
    text = format_report(report)
    assert "0 questions scored (1 read, 0 not in the collection)" in text
    assert "and 1 question that abstain (searched, not scored)" in text
    assert "expected_behavior: abstain" in text and "Per question" in text
    assert "No question was scored: there is no top score to compare with." in text
    assert json.loads(report.to_json())["abstentions"]["top_score_of_scored_questions"] is None


def test_a_scored_question_that_was_returned_nothing_has_no_top_score(monkeypatch):
    answers = {("x", False): {"a": 0.9}, ("y", True): {}, ("nothing", False): {"b": 0.4}}
    store, _calls = answering(monkeypatch, answers)
    questions = [
        Question(id="q1", question="x", expected_doc_ids=("a",)),
        Question(id="q2", question="y", expected_doc_ids=("a",), filters=FILTER),
        Question(id="a1", question="nothing", expected_doc_ids=(), expected_behavior="abstain"),
    ]
    report = ask(store, questions)
    assert len(report.results) == 2 and report.top_scores == TopScores(1, 0.9, 0.9, 0.9)
    # The line says why it compares with one question when two were scored.
    text = format_report(report)
    assert (
        "For comparison, top score of 1 of the 2 scored questions (1 returned no document):\n  min 0.900"
        in text
    )


@pytest.mark.parametrize(
    ("levels", "match"),
    [
        ((), "score_levels is empty"),
        ((0.8, 0.8), "twice"),
        ((0.8, "0.9"), "finite numbers"),
        ((0.8, float("nan")), "finite numbers"),
        ((float("inf"),), "finite numbers"),
        ((True,), "finite numbers"),
        ("0.8", "list of numbers"),
        (0.8, "list of numbers"),
        (None, "list of numbers"),
    ],
)
def test_bad_score_levels_are_refused_before_anything_is_asked(levels, match):
    store = small_store({"a": ["alpha"]})
    embedder = FakeEmbedder(SPEC)
    asked = Question(id="q1", question="alpha", expected_doc_ids=("a",))
    with pytest.raises(EvalError, match=match):
        evaluate([asked], collection="c", store=store, embedder=embedder, score_levels=levels)
    assert embedder.calls == []


def test_score_levels_are_put_in_order():
    store = small_store({"a": ["alpha"]})
    asked = Question(id="q1", question="alpha", expected_doc_ids=(), expected_behavior="abstain")
    report = ask(store, [asked], score_levels=[0.9, 0, 0.5])
    assert report.parameters["score_levels"] == [0.0, 0.5, 0.9]
    assert report.abstentions[0].documents_at_or_above == ((0.0, 1), (0.5, 1), (0.9, 1))


# ---------------------------------------------------------------------------
# Aggregates built from results
# ---------------------------------------------------------------------------
def scored(
    id, essential, found, *, acceptable=(), absent=(), unreachable=False, kind=None, behavior="answer"
):
    """A result at rank 1 whose essential documents are ``essential``, ``found`` of them among the first k."""
    present = [doc_id for doc_id in essential if doc_id not in absent]
    return QuestionResult(
        id=id,
        question="q",
        kind=kind,
        tags=(),
        expected_doc_ids=(*essential, *acceptable),
        absent_doc_ids=tuple(absent),
        filters=None,
        rank=1,
        hit_at_1=True,
        hit_at_k=True,
        reciprocal_rank=1.0,
        recall_at_k=1.0,
        chunks_returned=10,
        documents_ranked=10,
        chunk_budget_exhausted=False,
        top_documents=(RankedDocument("x", None, 1.0, 1),),
        expected_behavior=behavior,
        essential_ids=tuple(essential),
        acceptable_ids=tuple(acceptable),
        essential_present=len(present),
        essential_recall_at_k=found / len(present) if present else None,
        all_essential_at_k=found == len(present) if present else None,
        essential_unreachable=unreachable,
    )


def test_the_questions_of_an_essential_document_are_averaged_before_the_documents():
    # "a" is essential to three questions that find it, "b" to one that does not.
    results = [scored(f"a{n}", ["a"], 1) for n in range(3)] + [scored("b", ["b"], 0)]
    essential = Aggregate.of(results).essential
    assert (essential.questions, essential.documents, essential.unreachable) == (4, 2, 0)
    assert (essential.essential_recall_at_k, essential.all_essential_at_k) == (0.75, 0.75)
    assert essential.per_document == EssentialMeans(0.5, 0.5)
    assert essential.to_dict() == {
        "questions": 4,
        "documents": 2,
        "unreachable": 0,
        "essential_recall_at_k": 0.75,
        "all_essential_at_k": 0.75,
        "per_document": {"essential_recall_at_k": 0.5, "all_essential_at_k": 0.5},
        "intervals": None,
        "few_documents": False,
        "all_essential_questions": 3,
        "requirements": 4,
        "requirements_satisfied": 3,
        "pooled_essential_recall_at_k": 0.75,
        "question_intervals": None,
        "few_questions": False,
    }


def test_only_essential_documents_stand_behind_the_essential_means():
    results = [
        scored("q1", ["a", "b"], 1, acceptable=["c"]),  # 1/2, not all
        scored("q2", ["a"], 1, acceptable=["b"]),  # 1, all
        scored("q3", ["b", "ghost"], 1, absent=["ghost"], unreachable=True),  # 1 of the one present
    ]
    group = Aggregate.of(results)
    assert group.documents == 3  # a, b and c are expected
    essential = group.essential
    # "c" is acceptable only, and "ghost" is not in the collection: two essential documents.
    assert (essential.questions, essential.documents, essential.unreachable) == (3, 2, 1)
    assert essential.essential_recall_at_k == pytest.approx((0.5 + 1 + 1) / 3)
    assert essential.all_essential_at_k == pytest.approx(2 / 3)
    # a: q1 and q2 -> (3/4, 1/2); b: q1 and q3 -> (3/4, 1/2). "b" is acceptable to q2: not counted there.
    assert essential.per_document == EssentialMeans(0.75, 0.5)


def test_a_group_without_essential_documents_has_no_essential_metrics():
    old = QuestionResult(
        id="old",
        question="q",
        kind=None,
        tags=(),
        expected_doc_ids=("a",),
        absent_doc_ids=(),
        filters=None,
        rank=1,
        hit_at_1=True,
        hit_at_k=True,
        reciprocal_rank=1.0,
        recall_at_k=1.0,
        chunks_returned=1,
        documents_ranked=1,
        chunk_budget_exhausted=False,
        top_documents=(),
    )
    assert Aggregate.of([old]).essential is None
    assert Aggregate.of([old]).to_dict()["essential"] is None
    assert Aggregate.of([]).essential is None
    assert EssentialAggregate.of([old]) is None
    mixed = Aggregate.of([old, scored("new", ["a"], 0)])
    assert mixed.questions == 2 and mixed.essential.questions == 1  # left out, not counted as zero
    assert mixed.essential.essential_recall_at_k == 0.0


def test_essential_intervals_come_from_resampling_the_essential_documents():
    # Every essential document is found by one question and missed by another: each has a mean of 1/2.
    results = []
    for n in range(30):
        results += [scored(f"hit-{n}", [f"d{n}"], 1), scored(f"miss-{n}", [f"d{n}"], 0)]
    group = Aggregate.of(results, interval="x")
    assert group.essential.documents == 30 and group.essential.few_documents is False
    assert group.essential.intervals == {
        "essential_recall_at_k": (0.5, 0.5),
        "all_essential_at_k": (0.5, 0.5),
    }
    assert Aggregate.of(results).essential.intervals is None
    few = Aggregate.of(results[:38], interval="x").essential
    assert (few.documents, few.intervals, few.few_documents) == (19, None, True)
    assert Aggregate.of(results[:38], interval="x", min_documents=19).essential.intervals is not None
    spread = [scored(f"q{n}", [f"d{n}"], n % 2) for n in range(40)]
    low, high = Aggregate.of(spread, interval="x").essential.intervals["essential_recall_at_k"]
    assert 0.0 < low < 0.5 < high < 1.0
    assert (
        Aggregate.of(spread, interval="y").essential.intervals
        != Aggregate.of(spread, interval="x").essential.intervals
    )


def test_a_report_built_from_results_groups_by_behavior_in_a_fixed_order():
    results = [
        scored("q1", ["a"], 1, behavior="abstain"),
        scored("q2", ["b"], 0, behavior="correct_premise"),
        scored("q3", ["c"], 1, behavior="correct_premise"),
    ]
    report = EvalReport(parameters={"collection": "c", "k": 10}, results=tuple(results))
    assert list(report.by_behavior) == ["correct_premise", "abstain"]
    assert report.by_behavior["correct_premise"].essential.essential_recall_at_k == 0.5
    assert (report.unfiltered_reruns, report.abstentions, report.filter_effect) == ((), (), None)
    data = report.to_dict()
    assert data["abstentions"] == {
        "score_levels": [0.7, 0.75, 0.8, 0.85, 0.9],
        "top_score_of_scored_questions": {"questions": 3, "min": 1.0, "median": 1.0, "max": 1.0},
        "questions": [],
    }


# ---------------------------------------------------------------------------
# The documentation
# ---------------------------------------------------------------------------
def test_the_tables_shown_in_the_documentation_are_those_of_the_second_fixture(store, questions):
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    readme = (root / "eval" / "README.md").read_text(encoding="utf-8")
    report = evaluate(questions, collection=COLLECTION, store=store, embedder=fixture_embedder())
    parts = format_report(report).split("\n\n")
    shown = [parts[1]]
    for start in (
        "Per question",
        "Per expected behaviour",
        "Essential documents",
        "More requirements than",
        "What the filter brings",
        "Abstention (",
    ):
        shown += [part for part in parts if part.startswith(start)]
    assert len(shown) == 7
    for part in shown:
        assert part in readme, part.splitlines()[0]


def test_the_example_line_of_the_documentation_is_a_valid_question(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    readme = (root / "eval" / "README.md").read_text(encoding="utf-8")
    assert '"id": "an-' not in readme  # an example is not a question of the analytical batch
    (line,) = [line for line in readme.splitlines() if line.startswith('{"id": "ex-01"')]
    # complete: every field of the document-level format (the passage-level ones aside)
    assert set(json.loads(line)) == set(QUESTION_FIELDS) - {"expected_spans", "span_match", "difficulty"}
    (question,) = load_questions(write(tmp_path, line))
    assert (question.expected_behavior, question.run_unfiltered) == ("answer", True)
    members = tuple(doc_id for group in question.essential_any_of for doc_id in group)
    assert len(question.essential_any_of) == 1 and len(members) == 2
    assert question.expected_doc_ids == question.essential_ids + members + question.acceptable_ids
    assert question.must_include == ("unanimously", "0.25 percent") and question.must_not_include == ("6-3",)


# ---------------------------------------------------------------------------
# Either-of requirements
# ---------------------------------------------------------------------------
def test_either_of_groups_are_read_and_written_back(tmp_path):
    line = graded(essential_ids=["d1"], essential_any_of=[["d3", "d4"], ["d5"]], acceptable_ids=["d2"])
    given = graded(
        id="q2", essential_any_of=[["d3", "d4"]], acceptable_ids=[], expected_doc_ids=["d4", "d1", "d3"]
    )
    one, two, three = load_questions(write(tmp_path, line, given, graded(id="q3", essential_any_of=None)))
    assert one.essential_any_of == (("d3", "d4"), ("d5",))
    # One requirement per essential document, then one per group.
    assert one.requirements == (("d1",), ("d3", "d4"), ("d5",))
    # Left out, expected_doc_ids is the essential documents, the members of the groups, the acceptable ones.
    assert one.expected_doc_ids == ("d1", "d3", "d4", "d5", "d2")
    written = one.to_dict()
    assert written["essential_any_of"] == [["d3", "d4"], ["d5"]]
    assert list(written)[:6] == [
        "id",
        "question",
        "essential_ids",
        "essential_any_of",
        "acceptable_ids",
        "expected_doc_ids",
    ]
    assert two.expected_doc_ids == ("d4", "d1", "d3")  # given in another order: kept
    assert three.essential_any_of == () and "essential_any_of" not in three.to_dict()
    assert three.requirements == (("d1",),)
    again = load_questions(write(tmp_path, one.to_dict(), two.to_dict(), three.to_dict()))
    assert list(again) == [one, two, three]


def test_a_question_may_state_groups_only():
    built = Question(id="q", question="alpha", essential_any_of=[["a", "b"]])
    assert (built.essential_ids, built.acceptable_ids, built.expected_doc_ids) == ((), (), ("a", "b"))
    assert built.requirements == (("a", "b"),)
    old = Question(id="q", question="alpha", expected_doc_ids=["a"])
    assert old.essential_any_of == () and old.requirements is None
    assert Question(id="q", question="alpha", essential_ids=[], acceptable_ids=["a"]).requirements == ()


def test_an_empty_list_of_groups_states_the_essential_documents_like_an_empty_essential_list(tmp_path):
    """A field that is given, even empty, makes the question one that states what it needs."""
    for name in ("essential_ids", "essential_any_of"):
        only = {name: []}
        # On a question that abstains, nothing more is needed: no document, no essential one.
        abstains = q(expected_doc_ids=..., expected_behavior="abstain", **only)
        (loaded,) = load_questions(write(tmp_path, abstains))
        assert (loaded.essential_ids, loaded.essential_any_of, loaded.acceptable_ids) == ((), (), ()), name
        assert (loaded.expected_doc_ids, loaded.requirements) == ((), ()), name
        assert loaded.to_dict()["essential_ids"] == [] and "essential_any_of" not in loaded.to_dict()
        assert list(load_questions(write(tmp_path, loaded.to_dict()))) == [loaded]
        # Next to acceptable documents: no requirement, and the acceptable ones are the expected ones.
        (loaded,) = load_questions(write(tmp_path, q(expected_doc_ids=..., acceptable_ids=["d2"], **only)))
        assert (loaded.essential_ids, loaded.requirements, loaded.expected_doc_ids) == ((), (), ("d2",)), name
        # The expected documents must then be those of the three fields: none here.
        with pytest.raises(EvalError, match="'expected_doc_ids' must name the documents of .* it adds 'd1'"):
            load_questions(write(tmp_path, q(**only)))
        with pytest.raises(EvalError, match="'essential_ids' and 'acceptable_ids' are empty"):
            load_questions(write(tmp_path, q(expected_doc_ids=..., **only)))
        built = Question(id="q", question="alpha", expected_behavior="abstain", **only)
        assert (built.essential_ids, built.essential_any_of, built.requirements) == ((), (), ()), name
    # Null is a field left out, for the groups as for the lists.
    (old,) = load_questions(write(tmp_path, q(essential_any_of=None)))
    assert (old.essential_ids, old.essential_any_of, old.requirements) == (None, (), None)
    built = Question(id="q", question="alpha", expected_doc_ids=["a"], essential_any_of=None)
    assert built.requirements is None


@pytest.mark.parametrize(
    ("line", "match"),
    [
        (graded(essential_any_of="d3"), "'essential_any_of' must be a list of groups"),
        (graded(essential_any_of=["d3"]), "'essential_any_of' must be a list of groups"),
        (graded(essential_any_of=[{"d3": 1}]), "'essential_any_of' must be a list of groups"),
        (graded(essential_any_of=[[]]), "'essential_any_of' holds an empty group"),
        (graded(essential_any_of=[["d3", ""]]), r"'essential_any_of\[1\]'"),
        (graded(essential_any_of=[["d3", 4]]), r"'essential_any_of\[1\]'"),
        (graded(essential_any_of=[["d3"], ["d4", "d4"]]), r"'essential_any_of\[2\]' names 'd4' twice"),
        (graded(essential_any_of=[["d3", "d4"], ["d5", "d3"]]), "names 'd3' in two groups"),
        (graded(essential_any_of=[["d3", "d1"]]), "'essential_ids' and 'essential_any_of' both name 'd1'"),
        (graded(essential_any_of=[["d3", "d2"]]), "'acceptable_ids' and 'essential_any_of' both name 'd2'"),
        (
            graded(essential_any_of=[["d3", "d4"]], expected_doc_ids=["d1", "d2", "d3"]),
            "'expected_doc_ids' must name.*it lacks 'd4'",
        ),
        (
            graded(essential_any_of=[["d3"]], expected_doc_ids=["d1", "d2", "d3", "d9"]),
            "'expected_doc_ids' must name.*it adds 'd9'",
        ),
        (
            q(expected_doc_ids=..., essential_any_of=[["d3"]], expected_behavior="abstain"),
            "'essential_any_of'.*'abstain'",
        ),
        (graded(essential_any_of=[[" d3"]]), "leading or trailing whitespace"),
    ],
)
def test_a_bad_group_is_refused_with_its_line(tmp_path, line, match):
    with pytest.raises(EvalError, match=match) as info:
        load_questions(write(tmp_path, q(id="q0"), line))
    assert "questions.jsonl, line 2:" in str(info.value)


def test_a_group_is_satisfied_by_any_of_its_documents(monkeypatch):
    # The ranking is a, b, c, d and k = 2: "a" and "b" are among the first two.
    ranking = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}
    store, _calls = answering(monkeypatch, {("x", False): ranking})
    questions = [
        Question(id="sibling", question="x", essential_any_of=[("c", "a")]),
        Question(id="mixed", question="x", essential_ids=("d",), essential_any_of=[("b", "c")]),
        Question(id="none", question="x", essential_any_of=[("c", "d")], acceptable_ids=("a",)),
        Question(id="absent-group", question="x", essential_ids=("a",), essential_any_of=[("ghost", "g2")]),
        Question(id="half-absent", question="x", essential_any_of=[("ghost", "b")]),
        Question(id="three", question="x", essential_ids=("a",), essential_any_of=[("b", "c"), ("d", "e")]),
        Question(id="large-group", question="x", essential_any_of=[("d", "c", "b", "a")]),
        Question(id="no-group-there", question="x", essential_any_of=[("ghost",)], acceptable_ids=("a",)),
    ]
    report = ask(store, questions, k=2)
    by_id = {result.id: result for result in report.results}

    def values(id_):
        result = by_id[id_]
        return (
            result.requirements_present,
            result.requirements_satisfied,
            result.essential_recall_at_k,
            result.all_essential_at_k,
            result.essential_unreachable,
        )

    # One document of the group among the first two is enough, whichever it is.
    assert values("sibling") == (1, 1, 1.0, True, False)
    assert values("mixed") == (2, 1, 0.5, False, False)
    assert values("none") == (1, 0, 0.0, False, False)
    # A group none of whose documents is in the collection is left out, like an absent essential document.
    assert values("absent-group") == (1, 1, 1.0, True, False)
    assert values("half-absent") == (1, 1, 1.0, True, False)
    # Three requirements for two places: out of reach, although two of them are satisfied.
    assert values("three") == (3, 2, pytest.approx(2 / 3), False, True)
    # Four documents and one requirement: nothing is out of reach.
    assert values("large-group") == (1, 1, 1.0, True, False)
    assert values("no-group-there") == (0, None, None, None, False)
    # hit@1, MRR and recall@k look at every expected document, the members of the groups among them.
    sibling = by_id["sibling"]
    assert (sibling.rank, sibling.hit_at_1, sibling.recall_at_k) == (1, True, 0.5)
    assert sibling.expected_doc_ids == ("c", "a") and sibling.essential_any_of == (("c", "a"),)
    assert (by_id["none"].rank, by_id["none"].recall_at_k) == (1, pytest.approx(1 / 3))
    # essential_present keeps its meaning: the documents of essential_ids that are in the collection.
    assert [by_id[id_].essential_present for id_ in ("sibling", "mixed", "three")] == [0, 1, 1]
    data = sibling.to_dict()
    assert data["essential_any_of"] == [["c", "a"]]
    assert (data["requirements_present"], data["requirements_satisfied"]) == (1, 1)

    essential = report.overall.essential
    assert (essential.questions, essential.unreachable) == (7, 1)
    assert (essential.requirements, essential.requirements_satisfied) == (10, 7)
    assert essential.pooled_essential_recall_at_k == pytest.approx(0.7)
    assert essential.essential_recall_at_k == pytest.approx((1 + 0.5 + 0 + 1 + 1 + 2 / 3 + 1) / 7)
    assert (essential.all_essential_questions, essential.all_essential_at_k) == (4, pytest.approx(4 / 7))
    assert essential.documents == 5  # a, b, c, d and e; the ghosts are not documents of the collection
    assert report.essential_unreachable == 1
    text = format_report(report)
    assert "More requirements than k=2 (1):\n  three: 3 requirements" in text


# ---------------------------------------------------------------------------
# The essential metrics as they are shown
# ---------------------------------------------------------------------------
PARAMETERS = {
    "collection": "c",
    "k": 10,
    "chunk_limit": 50,
    "model": "fake",
    "backend": "fake",
    "query_normalization": None,
    "engine_version": "0",
}


def test_the_essential_table_counts_questions_and_requirements():
    results = [
        scored("q1", ["a", "b", "c", "d"], 1, kind="chain"),  # one of four, not all
        scored("q2", ["e"], 1, kind="chain"),
        scored("q3", ["f"], 1, kind="chain"),
    ]
    report = EvalReport(parameters=PARAMETERS, results=tuple(results))
    essential = report.by_kind["chain"].essential
    # Two questions of three have all they need; the mean share is (1/4 + 1 + 1) / 3; three of six pooled.
    assert (essential.all_essential_questions, essential.all_essential_at_k) == (2, pytest.approx(2 / 3))
    assert essential.essential_recall_at_k == pytest.approx(0.75)
    assert (essential.requirements, essential.requirements_satisfied) == (6, 3)
    assert essential.pooled_essential_recall_at_k == 0.5
    # The per-document mean of before is kept in the JSON report: 1/6 of the documents are those of a
    # question that has them all... it is not a share of questions, and the table does not print it.
    assert essential.per_document.all_essential_at_k == pytest.approx(2 / 6)
    data = essential.to_dict()
    assert list(data) == [
        "questions",
        "documents",
        "unreachable",
        "essential_recall_at_k",
        "all_essential_at_k",
        "per_document",
        "intervals",
        "few_documents",
        "all_essential_questions",
        "requirements",
        "requirements_satisfied",
        "pooled_essential_recall_at_k",
        "question_intervals",
        "few_questions",
    ]
    rows = rows_of(format_report(report), "Essential documents")
    assert rows["kind: chain"] == ["0.67 (2/3)", "0.75", "0.50 (3/6)", "0"]
    assert rows["headline"] == rows["kind: chain"] == rows["overall (pooled)"]


def test_essential_intervals_come_from_resampling_the_questions():
    # Forty questions, every other one with what it needs: one requirement each.
    spread = [scored(f"q{n}", [f"d{n}"], n % 2) for n in range(40)]
    essential = Aggregate.of(spread, interval="x").essential
    assert essential.few_questions is False
    assert set(essential.question_intervals) == {
        "essential_recall_at_k",
        "all_essential_at_k",
        "pooled_essential_recall_at_k",
    }
    for low, high in essential.question_intervals.values():
        assert 0.0 < low < 0.5 < high < 1.0
    assert Aggregate.of(spread, interval="x").essential.question_intervals == essential.question_intervals
    # The name of the group seeds its draws: another group, other draws.
    many = [scored(f"q{n}", [f"d{n}", f"e{n}", f"f{n}"], n % 4) for n in range(203)]
    assert (
        Aggregate.of(many, interval="y").essential.question_intervals
        != Aggregate.of(many, interval="x").essential.question_intervals
    )
    assert Aggregate.of(spread).essential.question_intervals is None
    # One question with forty essential documents: forty documents and one question. The questions
    # are what is drawn again, so there is no interval, however many documents there are.
    one = Aggregate.of([scored("q", [f"d{n}" for n in range(40)], 20)], interval="x").essential
    assert (one.documents, one.questions) == (40, 1)
    assert (one.question_intervals, one.few_questions) == (None, True)
    assert one.intervals is not None  # the interval of before, over the documents, is still computed
    few = Aggregate.of(spread[:19], interval="x").essential
    assert (few.question_intervals, few.few_questions) == (None, True)
    assert Aggregate.of(spread[:19], interval="x", min_questions=19).essential.question_intervals is not None
    # A question with many requirements weighs more in the pooled share than in the mean over questions.
    uneven = [scored(f"q{n}", [f"d{n}"], 1) for n in range(30)]
    uneven.append(scored("long", [f"l{n}" for n in range(30)], 0))
    essential = Aggregate.of(uneven, interval="x").essential
    assert essential.essential_recall_at_k == pytest.approx(30 / 31)
    assert essential.pooled_essential_recall_at_k == 0.5
    low, high = essential.question_intervals["pooled_essential_recall_at_k"]
    assert low <= 0.5 <= high and (low, high) != essential.question_intervals["essential_recall_at_k"]
    report = EvalReport(
        parameters=PARAMETERS,
        results=tuple(scored(f"q{n}", [f"d{n}"], n % 2, kind="one") for n in range(40)),
    )
    text = format_report(report)
    assert "95% intervals, from 1000 resamples of the questions (seed 1):" in text
    low, high = report.by_kind["one"].essential.question_intervals["all_essential_at_k"]
    assert f"[{low:.2f}, {high:.2f}]" in text


# ---------------------------------------------------------------------------
# Abstention: every question that abstains
# ---------------------------------------------------------------------------
def test_an_abstention_that_names_documents_is_not_scored(monkeypatch):
    answers = {
        ("found", False): {"a": 0.9, "b": 0.5},
        ("bait", False): {"b": 0.95, "a": 0.85, "c": 0.6, "d": 0.4},
        ("bait", True): {"c": 0.6},
    }
    store, _calls = answering(monkeypatch, answers)
    answerable = [Question(id="q1", question="found", expected_doc_ids=("a",), kind="k", tags=("t",))]
    abstaining = [
        Question(
            id="a1",
            question="bait",
            acceptable_ids=("a", "c", "ghost"),
            expected_behavior="abstain",
            kind="k",
            tags=("t", "apart"),
        ),
        Question(
            id="a2",
            question="bait",
            expected_doc_ids=("d",),
            expected_behavior="abstain",
            filters=FILTER,
            run_unfiltered=True,
        ),
    ]
    alone = ask(store, answerable, k=2, score_levels=(0.5, 0.9))
    mixed = ask(store, [*abstaining, *answerable], k=2, score_levels=(0.5, 0.9))
    # The context documents returned first are no hit: the question is in no result and no mean.
    assert mixed.results == alone.results
    for name in ("overall", "headline", "by_kind", "by_tag", "by_behavior", "filtered_kinds", "top_scores"):
        assert getattr(mixed, name) == getattr(alone, name), name
    assert list(mixed.by_behavior) == ["answer"] and "apart" not in mixed.by_tag
    assert mixed.misses == () and mixed.not_in_collection == () and mixed.unfiltered_reruns == ()
    first, second, third = mixed.abstentions
    assert [entry.id for entry in mixed.abstentions] == ["a1", "a2", "a2#unfiltered"]
    assert first.context_doc_ids == ("a", "c", "ghost") and first.absent_context_doc_ids == ("ghost",)
    # "a" is second, "c" third: one of the two context documents of the collection among the first two.
    assert (first.context_at_k, first.top_document.doc_id, first.top_score) == (1, "b", 0.95)
    assert first.documents_at_or_above == ((0.5, 3), (0.9, 1))
    assert (second.context_doc_ids, second.context_at_k, second.top_document.doc_id) == (("d",), 0, "c")
    assert (third.context_at_k, third.rerun_of) == (0, "a2")
    plain = Question(id="a3", question="bait", expected_doc_ids=(), expected_behavior="abstain")
    assert ask(store, [plain]).abstentions[0].context_at_k is None
    data = mixed.to_dict()
    assert data["summary"]["questions"] == 3 and data["summary"]["scored"] == 1
    assert data["summary"]["abstentions"] == 2
    entry = data["abstentions"]["questions"][0]
    assert entry["context_doc_ids"] == ["a", "c", "ghost"]
    assert entry["absent_context_doc_ids"] == ["ghost"]
    assert entry["context_documents_at_k"] == 1
    assert "abstain" not in data["metrics"]["by_expected_behavior"]
    text = format_report(mixed)
    lines = text.splitlines()
    assert "1 questions scored (3 read, 0 not in the collection)" in lines[0]
    assert lines[1] == "and 2 questions that abstain (searched, not scored)"
    assert "Abstention (2 questions whose expected behaviour is to abstain" in text
    rows = rows_of(text, "Abstention (")
    assert rows["a1"][:4] == ["0.950", "3", "1", "1/2"]
    assert rows["a2"][:4] == ["0.600", "1", "0", "0/1"]
    assert "context@2: the context documents of the question" in text
    behaviors = [line for line in lines if line.startswith("expected_behavior: abstain")]
    assert [" ".join(line.split()) for line in behaviors] == [
        "expected_behavior: abstain 2 see the Abstention section"
    ]


def test_an_essential_document_on_a_question_that_abstains_is_refused(tmp_path):
    bad = q(
        id="q2", expected_doc_ids=..., essential_ids=["d1"], acceptable_ids=[], expected_behavior="abstain"
    )
    with pytest.raises(EvalError, match="questions.jsonl, line 2: 'essential_ids' names.*'abstain'"):
        load_questions(write(tmp_path, q(), bad))
    with pytest.raises(EvalError, match="'essential_ids' names 'a' on a question"):
        Question(id="q", question="alpha", essential_ids=("a",), expected_behavior="abstain")
    # The documents such a question lists are its context documents.
    kept = q(
        id="q2", expected_doc_ids=..., essential_ids=[], acceptable_ids=["d1"], expected_behavior="abstain"
    )
    old = q(id="q3", expected_behavior="abstain")
    two, three = load_questions(write(tmp_path, kept, old))
    assert (two.essential_ids, two.acceptable_ids, two.expected_doc_ids) == ((), ("d1",), ("d1",))
    assert (three.essential_ids, three.expected_doc_ids) == (None, ("d1",))


# ---------------------------------------------------------------------------
# Per question, and long lists of expected documents
# ---------------------------------------------------------------------------
def test_the_report_gives_the_means_per_question_next_to_those_per_document(monkeypatch):
    ranking = {name: 1.0 - place / 100 for place, name in enumerate("abcdefgh")}
    store, _calls = answering(
        monkeypatch, {("long", False): ranking, ("short", False): {"z": 0.9}}, documents="abcdefghz"
    )
    questions = [
        # Six expected documents, the first of them first: found, and it weighs six times per document.
        Question(id="long", question="long", essential_ids=("a",), acceptable_ids=tuple("defgh"), kind="k"),
        Question(id="s1", question="short", essential_ids=("a",), kind="k"),
        Question(id="s2", question="short", essential_ids=("b",), kind="k"),
    ]
    report = ask(store, questions, k=3)
    kind = report.by_kind["k"]
    assert (kind.hit_at_1, kind.per_document.hit_at_1) == (pytest.approx(1 / 3), pytest.approx(5.5 / 7))
    by_id = {result.id: result for result in report.results}
    # Six expected documents and three places: recall@3 is 1/6 here and could not be above 1/2.
    assert (by_id["long"].expected_beyond_k, by_id["long"].recall_at_k) == (True, pytest.approx(1 / 6))
    assert by_id["s1"].expected_beyond_k is False
    assert report.expected_beyond_k == 1
    data = report.to_dict()
    assert data["summary"]["expected_beyond_k"] == 1 and data["results"][0]["expected_beyond_k"] is True
    text = format_report(report)
    assert rows_of(text, "k=3, chunk_limit")["kind: k"][:3] == ["3", "7", "0.79"]
    per_question = rows_of(text, "Per question")
    assert per_question["kind: k"] == ["3", "0.33", "0.33", "0.33", "0.06"]
    assert per_question["headline"] == per_question["kind: k"]
    assert "More expected documents than k=3 (1 question: recall@3 cannot reach 1 for it): long" in text


def test_a_wide_table_puts_its_intervals_on_a_line_of_their_own():
    name = "analytical+filters#unfiltered"
    results = [scored(f"q{n}", [f"d{n}"], 1, kind=name) for n in range(25)]
    report = EvalReport(parameters=PARAMETERS, results=tuple(results))
    lines = format_report(report).splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(f"kind: {name}"))
    assert lines[start].split()[2:] == ["25", "25", "1.00", "1.00", "1.00", "1.00"]
    assert lines[start + 1].split() == [
        "95%",
        "interval",
        "[1.00,",
        "1.00]",
        "[1.00,",
        "1.00]",
        "[1.00,",
        "1.00]",
    ]
    assert all(len(line) <= 110 for line in lines)


# ---------------------------------------------------------------------------
# What the filter brings: the essential mean
# ---------------------------------------------------------------------------
def test_the_essential_mean_of_the_filter_effect_is_over_the_pairs_that_state_requirements():
    def pair(id_, essential, with_filters, without):
        if essential:
            written = scored(id_, ["a"], with_filters)
            rerun = scored(id_ + "#unfiltered", ["a"], without)
        else:
            written = dataclasses.replace(
                scored(id_, [], 0), essential_ids=None, acceptable_ids=None, essential_present=None
            )
            rerun = dataclasses.replace(written, id=id_ + "#unfiltered")
        return (written, dataclasses.replace(rerun, rerun_of=id_))

    effect = FilterEffect((pair("q1", True, 1, 0), pair("q2", False, 0, 0), pair("q3", False, 0, 0)))
    # One pair of three states requirements: the mean is its value, not a third of it.
    assert (effect.questions, effect.essential_questions) == (3, 1)
    assert effect.filtered["essential_recall_at_k"] == 1.0
    assert effect.unfiltered["essential_recall_at_k"] == 0.0
    assert effect.filtered["hit_at_1"] == 1.0  # the three pairs
    both = FilterEffect((pair("q1", True, 1, 0), pair("q2", True, 1, 1), pair("q3", False, 0, 0)))
    assert (both.filtered["essential_recall_at_k"], both.unfiltered["essential_recall_at_k"]) == (1.0, 0.5)


# ---------------------------------------------------------------------------
# The loader: what is refused besides
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("line", "match"),
    [
        (
            '{"id": "q2", "question": "x", "expected_doc_ids": ["d1"], "id": "q3"}',
            "the key 'id' is given twice",
        ),
        (
            '{"id": "q2", "question": "x", "expected_doc_ids": ["d1"], "filters": {"corpus": "a", "corpus": "a"}}',  # noqa: E501
            "the key 'corpus' is given twice",
        ),
        (
            q(id="q2", expected_doc_ids=["d1 "]),
            "'expected_doc_ids' holds a document id with leading or trailing",
        ),
        (
            graded(id="q2", essential_ids=[" d1"]),
            "'essential_ids' holds a document id with leading or trailing",
        ),
        (
            graded(id="q2", acceptable_ids=["d2\t"]),
            "'acceptable_ids' holds a document id with leading or trailing",
        ),
        (
            q(id="q2", must_include=["5-4", "Shirai"], must_not_include=["unanimous", "5-4"]),
            "'must_include' and 'must_not_include' both hold '5-4'",
        ),
        (q(id="q2", kind="analytical#unfiltered"), "'kind' may not end with '#unfiltered'"),
    ],
)
def test_what_the_loader_refuses_besides(tmp_path, line, match):
    with pytest.raises(EvalError, match=match) as info:
        load_questions(write(tmp_path, q(), line))
    assert "questions.jsonl, line 2:" in str(info.value)


def test_a_kind_may_hold_the_suffix_elsewhere(tmp_path):
    (one,) = load_questions(write(tmp_path, q(kind="#unfiltered-first")))
    assert one.kind == "#unfiltered-first"
    with pytest.raises(EvalError, match="may not end with"):
        Question(id="q", question="alpha", expected_doc_ids=("a",), kind="k#unfiltered")


# ---------------------------------------------------------------------------
# The widths, on long names and long lists
# ---------------------------------------------------------------------------
def test_long_lists_are_wrapped_to_the_width(monkeypatch):
    names = [f"document-{n:02d}" for n in range(12)]
    store, _calls = answering(
        monkeypatch, {("x", False): {name: 1.0 - n / 100 for n, name in enumerate(names)}}, documents=names
    )
    questions = [
        Question(
            id=f"a-question-with-a-long-identifier-{n:02d}",
            question="x",
            essential_ids=names[:1],
            acceptable_ids=names[1:],
            kind="analytical+filters",
        )
        for n in range(8)
    ]
    questions.append(
        Question(
            id="abstains",
            question="x",
            acceptable_ids=names,
            expected_behavior="abstain",
            kind="analytical+filters",
        )
    )
    lines = format_report(ask(store, questions, k=3)).splitlines()
    assert all(len(line) <= 110 for line in lines)
    start = next(index for index, line in enumerate(lines) if line.startswith("More expected documents"))
    listed = " ".join(lines[start : start + 6])
    assert all(question.id in listed for question in questions[:8])  # none is cut away
    (row,) = [line for line in lines if line.startswith("abstains")]
    assert row.split()[1:3] == ["1.000", "12"] and row.split()[7] == "3/12"
