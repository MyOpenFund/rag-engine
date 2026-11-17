"""The second version of the question format: essential documents, behaviours, second runs, abstention.

``eval/fixtures/questions_v2.jsonl`` is asked of the corpus of the fixture; every value below
was worked out by hand from the file. The questions of the first version, and their values,
are in ``test_eval.py`` and are not touched by any of this.
"""

from __future__ import annotations

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
    )
    assert BEHAVIORS == ("answer", "correct_premise", "abstain")


def test_load_the_second_fixture(questions):
    assert [question.id for question in questions] == [f"fx2-q{n:02d}" for n in range(1, 12)]
    first, second, third, fifth, sixth, seventh, tenth, eleventh = (
        questions[index] for index in (0, 1, 2, 4, 5, 6, 9, 10)
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
    assert lines[9]["expected_doc_ids"] == [] and lines[9]["must_not_include"] == ["minus 0.1 percent"]


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
    assert four.expected_doc_ids == ("d1",)  # an abstention may name documents


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


def essential_exactly(group: Aggregate, questions, documents, unreachable, per_question, per_document):
    essential = group.essential
    assert isinstance(essential, EssentialAggregate)
    assert (essential.questions, essential.documents, essential.unreachable) == (
        questions,
        documents,
        unreachable,
    )
    got = (essential.essential_recall_at_k, essential.all_essential_at_k)
    assert got == pytest.approx([float(value) for value in per_question], abs=1e-12)
    assert isinstance(essential.per_document, EssentialMeans)
    got = (essential.per_document.essential_recall_at_k, essential.per_document.all_essential_at_k)
    assert got == pytest.approx([float(value) for value in per_document], abs=1e-12)


def test_what_is_scored_and_what_is_not(report):
    assert [result.id for result in report.results] == [f"fx2-q{n:02d}" for n in range(1, 10)]
    assert [result.id for result in report.unfiltered_reruns] == ["fx2-q07#unfiltered", "fx2-q08#unfiltered"]
    assert [entry.id for entry in report.abstentions] == ["fx2-q10", "fx2-q11"]
    assert report.not_in_collection == ()
    assert [result.id for result in report.searches] == [
        "fx2-q01",
        "fx2-q02",
        "fx2-q03",
        "fx2-q04",
        "fx2-q05",
        "fx2-q06",
        "fx2-q07",
        "fx2-q07#unfiltered",
        "fx2-q08",
        "fx2-q08#unfiltered",
        "fx2-q09",
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
    assert values("fx2-q04") == (1, True, 1.0, 1.0, 1, 1.0, True, False)
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
    assert [result.expected_behavior for result in report.results] == [
        "answer",
        "correct_premise",
        "answer",
        "abstain",
        "answer",
        "answer",
        "answer",
        "answer",
        "correct_premise",
    ]


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
        per_document=(1, 1, 1, F(275, 338)),
    )
    # The four questions as written: the second runs (one of them second) are not here.
    exactly(
        report.by_kind["graded+filters"],
        4,
        7,
        per_question=(1, 1, 1, F(2, 3)),
        per_document=(1, 1, 1, F(17, 28)),
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
        per_document=(F(38, 39), 1, F(77, 78), F(140, 169)),
    )
    # Pooled: each question once, as written. No second run, no abstention without documents.
    overall = report.overall
    assert (overall.questions, overall.documents) == (9, 13)
    assert (overall.hit_at_1, overall.hit_at_k, overall.mrr) == (1.0, 1.0, 1.0)
    assert overall.recall_at_k == pytest.approx(float(F(290, 351)), abs=1e-12)


def test_essential_metrics_of_the_second_fixture(report):
    # graded: fx2-q01, 02 and 04 have their one essential document; fx2-q05 ten of thirteen; fx2-q03
    # does not state any and is left out.
    essential_exactly(
        report.by_kind["graded"],
        4,
        13,
        1,
        per_question=(F(49, 52), F(3, 4)),
        per_document=(F(269, 338), F(3, 26)),
    )
    # graded+filters: 1/2, 1, 1/2 and 0. The decision is essential to two of them (1 and 0).
    essential_exactly(
        report.by_kind["graded+filters"],
        4,
        5,
        0,
        per_question=(F(1, 2), F(1, 4)),
        per_document=(F(1, 2), F(1, 10)),
    )
    essential_exactly(
        report.by_kind["graded+filters#unfiltered"], 2, 3, 0, per_question=(1, 1), per_document=(1, 1)
    )
    essential_exactly(
        report.headline,
        6,
        13,
        1,
        per_question=(F(25, 26), F(5, 6)),
        per_document=(F(138, 169), F(8, 39)),
    )
    essential_exactly(
        report.overall,
        8,
        13,
        1,
        per_question=(F(75, 104), F(1, 2)),
        # The decision is essential to four questions (1, 10/13, 1, 0), the report to three, ...
        per_document=(F(1501, 2028), F(4, 39)),
    )
    assert report.essential_unreachable == 1
    assert [result.id for result in report.results if result.essential_unreachable] == ["fx2-q05"]
    # No interval on thirteen documents, and none asked for the pooled line.
    assert (report.headline.essential.intervals, report.headline.essential.few_documents) == (None, True)
    assert (report.overall.essential.intervals, report.overall.essential.few_documents) == (None, False)
    # The essential metrics are not the expected-document ones under another name.
    filtered = report.by_kind["graded+filters"]
    assert filtered.essential.essential_recall_at_k != pytest.approx(filtered.recall_at_k)
    assert report.headline.essential.essential_recall_at_k != pytest.approx(report.headline.recall_at_k)


def test_breakdown_by_expected_behavior(report):
    assert list(report.by_behavior) == ["answer", "correct_premise", "abstain"]
    answer, premise, abstain = report.by_behavior.values()
    assert (answer.questions, answer.documents) == (6, 13)
    assert (answer.hit_at_1, answer.mrr) == (1.0, 1.0)
    assert answer.recall_at_k == pytest.approx(float(F(385, 468)), abs=1e-12)
    essential_exactly(
        answer, 5, 13, 1, per_question=(F(49, 65), F(2, 5)), per_document=(F(125, 169), F(2, 39))
    )
    # fx2-q02 and fx2-q09: a premise to correct is scored on its documents like any question.
    exactly(premise, 2, 3, per_question=(1, 1, 1, F(3, 4)), per_document=(1, 1, 1, F(2, 3)))
    essential_exactly(premise, 2, 2, 0, per_question=(F(1, 2), F(1, 2)), per_document=(F(1, 2), F(1, 2)))
    # fx2-q04 abstains and names a document: it is scored. fx2-q10 and fx2-q11 name none: they are not.
    exactly(abstain, 1, 1, per_question=(1, 1, 1, 1), per_document=(1, 1, 1, 1))
    essential_exactly(abstain, 1, 1, 0, per_question=(1, 1), per_document=(1, 1))
    assert sum(group.questions for group in report.by_behavior.values()) == len(report.results) == 9
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
    # The abstention questions without documents give no line: nothing of theirs is scored.
    assert report.by_tag["out-of-corpus"].questions == 1
    assert "short-query" not in report.by_tag and "abstention" not in report.by_kind


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
    tenth, eleventh = report.abstentions
    assert isinstance(tenth, Abstention)
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
    assert not scored & {"fx2-q10", "fx2-q11"}
    assert not {result.id for result in report.misses} & {"fx2-q10", "fx2-q11"}
    assert report.overall.questions == 9 and report.fewer_than_k_documents == 4
    # For comparison: every scored question is the text of a chunk, so its first document scores 1.
    top = report.top_scores
    assert isinstance(top, TopScores) and top.questions == 9
    assert (top.min, top.median, top.max) == pytest.approx((1.0, 1.0, 1.0), abs=1e-5)


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
    for word in ("essential", "expected_behavior", "Abstention", "What the filter brings", "#unfiltered"):
        assert word not in text


def test_a_larger_k_reaches_every_essential_document(store, questions):
    report = evaluate(questions, collection=COLLECTION, store=store, embedder=fixture_embedder(), k=13)
    fifth = {result.id: result for result in report.results}["fx2-q05"]
    assert (fifth.essential_recall_at_k, fifth.all_essential_at_k, fifth.essential_unreachable) == (
        1.0,
        True,
        False,
    )
    assert report.essential_unreachable == 0 and report.headline.essential.unreachable == 0
    assert "More essential documents" not in format_report(report)
    one = evaluate(questions, collection=COLLECTION, store=store, embedder=fixture_embedder(), k=1)
    by_id = {result.id: result for result in one.searches}
    # At k = 1: one of thirteen; the acceptable document of fx2-q01 takes the only place.
    assert by_id["fx2-q05"].essential_recall_at_k == pytest.approx(1 / 13)
    assert (by_id["fx2-q01"].hit_at_k, by_id["fx2-q01"].essential_recall_at_k) == (True, 0.0)
    assert by_id["fx2-q01"].all_essential_at_k is False
    assert (by_id["fx2-q08"].essential_unreachable, by_id["fx2-q07"].essential_unreachable) == (True, False)
    assert one.essential_unreachable == 3  # fx2-q05, fx2-q06 and fx2-q08: each question once


def test_intervals_cover_the_essential_metrics_on_the_kinds_and_the_headline(store, questions):
    report = evaluate(
        questions,
        collection=COLLECTION,
        store=store,
        embedder=fixture_embedder(),
        interval_min_documents=4,
        bootstrap_resamples=200,
    )
    essential = report.headline.essential
    assert essential.few_documents is False
    assert set(essential.intervals) == {"essential_recall_at_k", "all_essential_at_k"}
    low, high = essential.intervals["essential_recall_at_k"]
    # Thirteen documents whose means lie between 10/13 and 12/13.
    assert 10 / 13 <= low <= 138 / 169 <= high <= 12 / 13 and low < high
    low, high = essential.intervals["all_essential_at_k"]
    assert 0.0 <= low <= 8 / 39 <= high <= 2 / 3 and low < high
    assert set(report.headline.intervals) == {"hit_at_1", "hit_at_k", "mrr"}  # as before
    assert report.by_kind["graded+filters"].essential.intervals is not None
    assert report.by_kind["graded+filters#unfiltered"].essential.few_documents is True
    assert report.overall.essential.intervals is None and report.by_tag["chain"].essential.intervals is None
    # The same results give the same intervals.
    again = evaluate(
        questions,
        collection=COLLECTION,
        store=store,
        embedder=fixture_embedder(),
        interval_min_documents=4,
        bootstrap_resamples=200,
    )
    assert again.headline.essential.intervals == essential.intervals
    text = format_report(report)
    assert f"{138 / 169:.2f} [{essential.intervals['essential_recall_at_k'][0]:.2f}, " in text
    assert "(few): fewer than 4 essential documents" in text


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
        "questions": 11,
        "scored": 9,
        "not_in_collection": 0,
        "partially_absent": 1,
        "fewer_than_k_documents": 4,
        "chunk_budget_exhausted": 0,
        "tied_at_decision": 1,
        "unfiltered_reruns": 2,
        "abstentions": 2,
        "essential_unreachable": 1,
    }
    assert len(data["results"]) == 9 and len(data["unfiltered_reruns"]) == 2
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
    assert data["results"][4]["essential_unreachable"] is True
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
            "essential_recall_at_k": pytest.approx(138 / 169),
            "all_essential_at_k": pytest.approx(8 / 39),
        },
        "intervals": None,
        "few_documents": True,
    }
    assert list(metrics["by_expected_behavior"]) == ["answer", "correct_premise", "abstain"]
    assert metrics["by_expected_behavior"]["correct_premise"]["questions"] == 2
    assert data["filter_effect"]["unfiltered"]["mrr"] == 0.75
    abstentions = data["abstentions"]
    assert abstentions["score_levels"] == [-1.0, 0.999]
    assert abstentions["top_score_of_scored_questions"] == {
        "questions": 9,
        "min": pytest.approx(1.0, abs=1e-5),
        "median": pytest.approx(1.0, abs=1e-5),
        "max": pytest.approx(1.0, abs=1e-5),
    }
    tenth, eleventh = abstentions["questions"]
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
    assert "9 questions scored (11 read, 0 not in the collection)" in lines[0]
    assert lines[1] == "and 2 questions that expect no document (not scored), 2 second runs without filters"
    main = rows_of(text, "questions: ")
    assert main["kind: graded"] == ["5", "13", "1.00", "1.00", "1.00", "0.81", "(few documents)"]
    assert main["kind: graded+filters *"] == ["4", "7", "1.00", "1.00", "1.00", "0.61", "(few documents)"]
    assert main["kind: graded+filters#unfiltered"] == [
        "2",
        "3",
        "0.67",
        "1.00",
        "0.83",
        "1.00",
        "(few documents)",
    ]
    assert main["headline"] == ["7", "13", "0.97", "1.00", "0.99", "0.83", "(few documents)"]
    assert main["overall (pooled)"] == ["9", "13", "1.00", "1.00", "1.00", "0.83"]
    assert "headline: the kinds asked without filters (graded, graded+filters#unfiltered)." in text
    assert "#unfiltered: the questions of a kind that ask for it (run_unfiltered)" in text

    behaviors = rows_of(text, "Per expected behaviour")
    assert list(behaviors) == [
        "expected_behavior: answer",
        "expected_behavior: correct_premise",
        "expected_behavior: abstain",
    ]
    assert behaviors["expected_behavior: correct_premise"] == ["2", "3", "1.00", "1.00", "1.00", "0.67"]

    essential = rows_of(text, "Essential documents")
    assert essential["kind: graded"] == ["4", "13", "0.80", "0.12", "1", "(few)"]
    assert essential["kind: graded+filters *"] == ["4", "5", "0.50", "0.10", "0", "(few)"]
    assert essential["headline"] == ["6", "13", "0.82", "0.21", "1", "(few)"]
    assert essential["overall (pooled)"] == ["8", "13", "0.72", "0.50", "1"]  # over the questions
    assert essential["expected_behavior: correct_premise"] == ["2", "2", "0.50", "0.50", "0"]
    for legend in (
        "essential_recall@10: for a question, the share of its essential documents among the first 10.",
        "all_essential@10: for a question, 1 when all of them are among the first 10, else 0.",
        "In a per-document mean a question counts once for each of its essential documents.",
        "unreachable: questions with more essential documents than k=10",
        "raise --k (at most --chunk-limit",
        "hit@1, hit@10 and MRR look at the first expected document, essential or acceptable.",
    ):
        assert legend in text
    assert "More essential documents than k=10 (1):\n  fx2-q05: 13 essential documents" in text

    effect = rows_of(text, "What the filter brings (2 questions")
    assert effect == {
        "with their filters": ["2", "1.00", "1.00", "1.00", "0.75"],
        "without filters": ["2", "0.50", "1.00", "0.75", "1.00"],
    }

    assert "Abstention (2 questions that expect no document" in text
    (tenth,) = [line for line in lines if line.startswith("fx2-q10")]
    assert tenth.split()[:5] == ["fx2-q10", "1.000", "13", "1", "fx-jp-minutes"]
    (eleventh,) = [line for line in lines if line.startswith("fx2-q11")]
    assert eleventh.split()[:4] == ["fx2-q11", "n/a", "0", "0"] and "nothing was returned" in eleventh
    assert ">=-1" in text and ">=0.999" in text
    assert "top score of the 9 scored questions: min 1.000, median 1.000, max 1.000." in text
    assert "No threshold is applied" in text
    assert "Misses (0)" in text


def test_the_default_score_levels_are_reported(store, questions):
    report = evaluate(questions, collection=COLLECTION, store=store, embedder=fixture_embedder())
    assert DEFAULT_SCORE_LEVELS == (0.7, 0.75, 0.8, 0.85, 0.9)
    assert report.parameters["score_levels"] == [0.7, 0.75, 0.8, 0.85, 0.9]
    tenth = report.abstentions[0]
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
    assert "and 2 questions that expect no document (not scored)" in text
    assert "top score of the 3 scored questions: min 0.500, median 0.700, max 0.900." in text
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
    assert "and 1 question that expect no document (not scored)" in text
    assert "No question was scored: there is no top score to compare with." in text
    assert json.loads(report.to_json())["abstentions"]["top_score_of_scored_questions"] is None


def test_an_abstention_that_names_documents_is_scored(monkeypatch):
    store, _calls = answering(monkeypatch, {("x", False): {"b": 0.9, "a": 0.8}})
    asked = Question(id="q1", question="x", essential_ids=("a",), expected_behavior="abstain")
    report = ask(store, [asked])
    assert report.abstentions == ()
    (result,) = report.results
    assert (result.rank, result.expected_behavior, result.essential_recall_at_k) == (2, "abstain", 1.0)
    assert list(report.by_behavior) == ["abstain"] and report.overall.questions == 1
    assert "expected_behavior: abstain" in format_report(report)
    assert "Abstention (" not in format_report(report)


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
    for start in ("Per expected behaviour", "Essential documents", "What the filter brings", "Abstention ("):
        shown += [part for part in parts if part.startswith(start)]
    assert len(shown) == 5
    for part in shown:
        assert part in readme, part.splitlines()[0]


def test_the_example_line_of_the_documentation_is_a_valid_question(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    readme = (root / "eval" / "README.md").read_text(encoding="utf-8")
    (line,) = [line for line in readme.splitlines() if line.startswith('{"id": "an-21"')]
    assert set(json.loads(line)) == set(QUESTION_FIELDS)  # complete: every field of the format
    (question,) = load_questions(write(tmp_path, line))
    assert (question.expected_behavior, question.run_unfiltered) == ("answer", True)
    assert question.expected_doc_ids == question.essential_ids + question.acceptable_ids
    assert question.must_include == ("unanimously", "0.25 percent") and question.must_not_include == ("6-3",)
