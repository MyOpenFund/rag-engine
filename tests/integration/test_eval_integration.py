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
from eval_fixture import QUESTIONS, fixture_corpus, fixture_embedder, load_fixture

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
    assert load_fixture(store, name) == 30
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
    }
    overall = report.overall
    assert overall.questions == 9
    assert overall.hit_at_1 == pytest.approx(7 / 9)
    assert overall.hit_at_k == pytest.approx(8 / 9)
    assert overall.mrr == pytest.approx(7.5 / 9)
    assert overall.recall_at_k == pytest.approx(7.5 / 9)
    assert [skipped.id for skipped in report.not_in_collection] == ["fx-q09"]
    assert [result.id for result in report.misses] == ["fx-q08"]
    tenth = report.results[-1]
    assert tenth.absent_doc_ids == ("fx-us-presser-2016-03",)
    assert (report.fewer_than_k_documents, report.chunk_budget_exhausted) == (6, 0)
    assert report.parameters["collection"] == collection
    assert "9 questions scored" in format_report(report)


def test_a_small_chunk_budget_is_reported_on_a_real_server(store, collection):
    questions = load_questions(QUESTIONS)
    report = evaluate(
        questions, collection=collection, store=store, embedder=fixture_embedder(), k=10, chunk_limit=10
    )
    assert report.overall.hit_at_1 == pytest.approx(7 / 9)
    exhausted = {result.id for result in report.results if result.chunk_budget_exhausted}
    assert exhausted == {"fx-q01", "fx-q02", "fx-q03", "fx-q08", "fx-q10"}


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
