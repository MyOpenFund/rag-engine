"""The evaluation harness, end to end, against the real Qdrant given by RAG_ENGINE_IT_QDRANT_URL.

The fixture of ``eval/fixtures/`` is written to a real collection with the payload indexes
of the engine, and evaluated: the metrics must be the ones the unit tests get in
qdrant-client's local mode. What only a real server shows is checked too: the lookup of the
expected documents through the keyword index on ``doc_id``, in several batches and pages.
"""

from __future__ import annotations

import os
import uuid

import pytest
from eval_fixture import QUESTIONS, QUESTIONS_V2, fixture_corpus, fixture_embedder, load_fixture

from rag_engine.config import Settings
from rag_engine.eval import evaluate, format_report, load_questions
from rag_engine.schema import PAYLOAD_INDEXES
from rag_engine.store import QdrantStore

QDRANT_URL = os.environ.get("RAG_ENGINE_IT_QDRANT_URL", "").strip()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not QDRANT_URL, reason="RAG_ENGINE_IT_QDRANT_URL is not set"),
]

DOC_IDS = sorted(document["doc_id"] for document in fixture_corpus()["documents"])


@pytest.fixture(scope="module")
def store():
    api_key = os.environ.get("RAG_ENGINE_IT_QDRANT_API_KEY", "").strip() or None
    store = QdrantStore.from_url(QDRANT_URL, api_key, settings=Settings(batch_size=7))
    yield store
    store.client.close()


@pytest.fixture(scope="module", params=["plain", "indexed"])
def collection(store, request):
    """The fixture corpus in a collection of its own, without and with the payload indexes."""
    name = f"rag-engine-it-{uuid.uuid4().hex}"
    assert load_fixture(store, name) == 32
    if request.param == "indexed":
        assert store.ensure_payload_indexes(name) == [field for field, _kind in PAYLOAD_INDEXES]
    yield name
    store.client.delete_collection(collection_name=name)


def test_the_fixture_is_evaluated_end_to_end(store, collection):
    questions = load_questions(QUESTIONS)
    report = evaluate(questions, collection=collection, store=store, embedder=fixture_embedder())
    ranks = {result.id: result.rank for result in report.results}
    assert ranks == {
        "fx-q01": 1,
        "fx-q02": 1,
        "fx-q03": 1,
        "fx-q04": 1,
        "fx-q05": 1,
        "fx-q06": 1,
        "fx-q07": 2,
        "fx-q08": None,
        "fx-q10": 1,
        "fx-q11": 1,
        "fx-q12": 2,
        "fx-q13": 1,
    }
    overall = report.overall
    assert (overall.questions, overall.documents) == (12, 11)
    assert overall.hit_at_1 == pytest.approx(9 / 12)
    assert overall.hit_at_k == pytest.approx(11 / 12)
    assert overall.mrr == pytest.approx(10 / 12)
    assert overall.recall_at_k == pytest.approx(10.5 / 12)
    headline = report.headline
    assert report.headline_kinds == ("verbatim",)
    assert (headline.questions, headline.documents) == (6, 4)
    assert headline.per_document.hit_at_1 == pytest.approx(11 / 12)
    assert headline.per_document.mrr == pytest.approx(23 / 24)
    assert [skipped.id for skipped in report.not_in_collection] == ["fx-q09"]
    assert [result.id for result in report.misses] == ["fx-q08"]
    tenth = {result.id: result for result in report.results}["fx-q10"]
    assert tenth.absent_doc_ids == ("fx-us-presser-2016-03",)
    assert (report.fewer_than_k_documents, report.chunk_budget_exhausted) == (6, 0)
    assert report.parameters["collection"] == collection
    assert "12 questions scored" in format_report(report)


def test_the_second_fixture_is_evaluated_end_to_end(store, collection):
    """Requirements, either-of groups, second runs without filters and abstentions, on a real server."""
    questions = load_questions(QUESTIONS_V2)
    report = evaluate(
        questions, collection=collection, store=store, embedder=fixture_embedder(), score_levels=(-1, 0.999)
    )
    assert [result.id for result in report.results] == [
        f"fx2-q{n:02d}" for n in (1, 2, 3, 5, 6, 7, 8, 9, 12, 13)
    ]
    assert {result.id: result.rank for result in report.searches} == {
        "fx2-q01": 1,
        "fx2-q02": 1,
        "fx2-q03": 1,
        "fx2-q05": 1,
        "fx2-q06": 1,
        "fx2-q07": 1,
        "fx2-q07#unfiltered": 2,
        "fx2-q08": 1,
        "fx2-q08#unfiltered": 1,
        "fx2-q09": 1,
        "fx2-q12": 1,
        "fx2-q13": 1,
    }
    by_id = {result.id: result for result in report.searches}
    assert by_id["fx2-q05"].essential_recall_at_k == pytest.approx(10 / 13)
    assert (by_id["fx2-q05"].all_essential_at_k, by_id["fx2-q05"].essential_unreachable) == (False, True)
    assert (by_id["fx2-q06"].essential_recall_at_k, by_id["fx2-q06"].recall_at_k) == (
        0.5,
        pytest.approx(2 / 3),
    )
    assert (by_id["fx2-q09"].hit_at_1, by_id["fx2-q09"].essential_recall_at_k) == (True, 0.0)
    assert by_id["fx2-q03"].essential_recall_at_k is None
    assert by_id["fx2-q07#unfiltered"].tied_at_decision is True
    # Either-of groups: one requirement satisfied by either document; a group satisfied by the
    # only document the filters of the server leave.
    assert (by_id["fx2-q12"].requirements_present, by_id["fx2-q12"].all_essential_at_k) == (1, True)
    assert (by_id["fx2-q13"].requirements_satisfied, by_id["fx2-q13"].essential_recall_at_k) == (1, 0.5)
    assert [document.doc_id for document in by_id["fx2-q13"].top_documents] == ["fx-ecb-combined-2016-03"]
    assert report.headline_kinds == ("graded", "graded+filters#unfiltered")
    assert report.filtered_kinds == ("graded+filters",)
    headline = report.headline
    assert (headline.questions, headline.documents) == (7, 13)
    assert headline.per_document.hit_at_1 == pytest.approx(38 / 39)
    assert headline.per_document.mrr == pytest.approx(77 / 78)
    assert headline.per_document.recall_at_k == pytest.approx(557 / 676)
    essential = headline.essential
    assert (essential.questions, essential.documents, essential.unreachable) == (6, 13, 1)
    assert (essential.all_essential_questions, essential.all_essential_at_k) == (5, pytest.approx(5 / 6))
    assert essential.essential_recall_at_k == pytest.approx(25 / 26)
    assert (essential.requirements_satisfied, essential.requirements) == (16, 19)
    assert (report.overall.questions, report.overall.recall_at_k) == (10, pytest.approx(101 / 130))
    assert {name: group.questions for name, group in report.by_behavior.items()} == {
        "answer": 8,
        "correct_premise": 2,
    }
    effect = report.filter_effect
    assert effect.filtered == {"hit_at_1": 1.0, "hit_at_k": 1.0, "mrr": 1.0, "essential_recall_at_k": 0.75}
    assert effect.unfiltered == {"hit_at_1": 0.5, "hit_at_k": 1.0, "mrr": 0.75, "essential_recall_at_k": 1.0}
    # The three questions that abstain: one lists a context document, which the server returns first.
    fourth, tenth, eleventh = report.abstentions
    assert (fourth.id, fourth.top_document.doc_id, fourth.context_at_k) == (
        "fx2-q04",
        "fx-gb-report-2016-08",
        1,
    )
    assert (tenth.id, tenth.top_document.doc_id) == ("fx2-q10", "fx-jp-minutes")
    assert tenth.top_score == pytest.approx(1.0, abs=1e-5)
    assert tenth.documents_at_or_above == ((-1.0, 13), (0.999, 1))
    assert (eleventh.top_document, eleventh.documents_ranked) == (None, 0)
    top = report.top_scores
    assert (top.questions, top.max) == (10, pytest.approx(1.0, abs=1e-5))
    text = format_report(report)
    assert "10 questions scored (13 read, 0 not in the collection)" in text
    assert "What the filter brings (2 questions" in text and "Abstention (3 questions" in text


def test_equal_scores_are_ordered_the_same_way_on_a_real_server(store, collection):
    """The same chunk stored in two documents: a real server gives both exactly the same score."""
    questions = [question for question in load_questions(QUESTIONS) if question.id in ("fx-q12", "fx-q13")]
    for chunk_limit in (50, 13, 2):
        report = evaluate(
            questions,
            collection=collection,
            store=store,
            embedder=fixture_embedder(),
            k=2,
            chunk_limit=chunk_limit,
        )
        twelfth, thirteenth = report.results
        first, second = twelfth.top_documents[:2]
        assert (first.doc_id, second.doc_id) == ("fx-ecb-combined-2016-03", "fx-ecb-decision-2016-03")
        assert first.score == second.score
        assert (twelfth.rank, twelfth.tied_at_decision) == (2, True)
        assert (thirteenth.rank, thirteenth.tied_at_decision) == (1, False)
        assert report.tied_at_decision == 1


def test_a_small_chunk_budget_is_reported_on_a_real_server(store, collection):
    questions = load_questions(QUESTIONS)
    report = evaluate(
        questions, collection=collection, store=store, embedder=fixture_embedder(), k=10, chunk_limit=10
    )
    assert report.overall.hit_at_1 == pytest.approx(9 / 12)
    exhausted = {result.id for result in report.results if result.chunk_budget_exhausted}
    assert exhausted == {"fx-q01", "fx-q02", "fx-q03", "fx-q08", "fx-q10", "fx-q11", "fx-q12", "fx-q13"}


def test_existing_doc_ids_on_a_real_server(store, collection):
    wanted = [*DOC_IDS, "fx-ch-statement-2016-06", "FX-JP-MINUTES", "fx-jp"]
    assert store.existing_doc_ids(collection, wanted) == set(DOC_IDS)
    assert store.existing_doc_ids(collection, ["nowhere"]) == set()
    assert store.existing_doc_ids(collection, []) == set()


@pytest.mark.parametrize(("batch_size", "page_size"), [(1, 1), (5, 2), (12, 1), (100, 3), (3, 256)])
def test_existing_doc_ids_in_small_batches_and_pages(store, collection, batch_size, page_size):
    wanted = [*DOC_IDS, "absent-1", "absent-2"]
    found = store.existing_doc_ids(collection, wanted, batch_size=batch_size, page_size=page_size)
    assert found == set(DOC_IDS)


def test_existing_doc_ids_on_a_missing_collection_raises(store):
    from qdrant_client.http.exceptions import UnexpectedResponse

    with pytest.raises(UnexpectedResponse) as info:
        store.existing_doc_ids(f"rag-engine-it-{uuid.uuid4().hex}", ["a"])
    assert info.value.status_code == 404
