"""``ingest_document`` against the real Qdrant given by RAG_ENGINE_IT_QDRANT_URL.

The two documents of ``tests/fixtures/golden/`` are parsed with the real libraries, written to
the server, and read back: what the server holds must be what the previous pipeline wrote
(``expected.json``, see the README of that folder). ``tests/test_ingest.py`` checks the same
points as they leave the engine; here they have gone through the server.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from rag_engine.config import Settings
from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.ingest import IngestResult, ingest_document
from rag_engine.store import CollectionMismatchError, QdrantStore

# The documents are parsed for real: without the ``ingest`` extra there is nothing to test.
pytest.importorskip("chunknorris")

QDRANT_URL = os.environ.get("RAG_ENGINE_IT_QDRANT_URL", "").strip()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not QDRANT_URL, reason="RAG_ENGINE_IT_QDRANT_URL is not set"),
]

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
GOLDEN = json.loads((GOLDEN_DIR / "expected.json").read_text(encoding="utf-8"))
DOCUMENTS = GOLDEN["documents"]
MARKDOWN = next(document for document in DOCUMENTS if document["file"].endswith(".md"))
EXPECTED_POINTS = [point for document in DOCUMENTS for point in document["points"]]
# A cosine collection may store the vectors normalised. Those of the FakeEmbedder already have
# unit length, so what comes back differs from what was sent by float32 rounding at most.
VECTOR_TOLERANCE = 1e-6


@pytest.fixture
def store():
    api_key = os.environ.get("RAG_ENGINE_IT_QDRANT_API_KEY", "").strip() or None
    store = QdrantStore.from_url(QDRANT_URL, api_key, settings=Settings())
    yield store
    store.client.close()


@pytest.fixture
def collection(store):
    """A collection name of this test only, dropped afterwards (the test creates it or not)."""
    name = f"rag-engine-it-{uuid.uuid4().hex}"
    yield name
    store.client.delete_collection(collection_name=name)


def golden_embedder() -> FakeEmbedder:
    return FakeEmbedder(ModelSpec(**GOLDEN["embedder"]["model"]), seed=GOLDEN["embedder"]["seed"])


def ingest_golden(store, collection: str, document: dict, **kwargs) -> IngestResult:
    """Ingest one golden document as it was captured, into ``collection``."""
    return ingest_document(
        GOLDEN_DIR / document["file"],
        document["doc_id"],
        document["metadata"],
        collection=collection,
        store=store,
        embedder=golden_embedder(),
        ocr=GOLDEN["ocr"],
        ingestion_date=document["ingestion_date"],
        **kwargs,
    )


def stored_points(store, collection: str) -> dict:
    """Every point of ``collection`` as the server holds it, by id, with payload and vector."""
    points: dict = {}
    offset = None
    while True:
        records, offset = store.client.scroll(
            collection_name=collection,
            limit=3,  # smaller than the collection: the following pages are read too
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for record in records:
            assert record.id not in points
            points[record.id] = record
        if offset is None:
            return points


def assert_holds(store, collection: str, expected: list[dict]) -> None:
    """``collection`` holds the ``expected`` points and nothing else."""
    assert store.count(collection) == len(expected)
    stored = stored_points(store, collection)
    assert sorted(stored) == sorted(point["id"] for point in expected)
    for point in expected:
        record = stored[point["id"]]
        assert record.payload == point["payload"]
        assert len(record.vector) == len(point["vector"])
        assert record.vector == pytest.approx(point["vector"], abs=VECTOR_TOLERANCE)


def test_the_fixture_is_ingested_as_captured():
    assert GOLDEN["ocr"] == "never"
    assert len(DOCUMENTS) == 2 and all(document["points"] for document in DOCUMENTS)
    assert len({point["id"] for point in EXPECTED_POINTS}) == len(EXPECTED_POINTS)


def test_the_server_holds_the_points_of_the_previous_pipeline(store, collection):
    embedder = golden_embedder()
    store.ensure_collection(collection, embedder.dim)
    assert store.count(collection) == 0

    for document in DOCUMENTS:
        result = ingest_golden(store, collection, document)
        assert (result.chunks, result.empty) == (len(document["points"]), False)

    assert_holds(store, collection, EXPECTED_POINTS)


def test_ingesting_the_same_documents_again_adds_no_point(store, collection):
    store.ensure_collection(collection, golden_embedder().dim)
    for _ in range(2):
        for document in DOCUMENTS:
            ingest_golden(store, collection, document)
        assert store.count(collection) == len(EXPECTED_POINTS)
    assert_holds(store, collection, EXPECTED_POINTS)


def test_a_document_written_in_small_batches_is_complete(store, collection):
    expected = MARKDOWN["points"]
    assert len(expected) > 2 and len(expected) % 2 == 1  # several batches, the last one not full
    store.ensure_collection(collection, golden_embedder().dim)

    result = ingest_golden(store, collection, MARKDOWN, batch_size=2)

    assert result.chunks == len(expected)
    assert_holds(store, collection, expected)


def test_a_collection_of_another_dimension_is_refused_and_keeps_its_points(store, collection):
    small = golden_embedder()
    spec = {**GOLDEN["embedder"]["model"], "dim": small.dim * 2}
    large = FakeEmbedder(ModelSpec(**spec), seed=GOLDEN["embedder"]["seed"])
    assert (small.dim, large.dim) == (8, 16)
    store.ensure_collection(collection, small.dim)
    ingest_golden(store, collection, MARKDOWN)

    with pytest.raises(CollectionMismatchError, match=f"{large.dim}-d"):
        store.ensure_collection(collection, large.dim)

    vectors = store.client.get_collection(collection_name=collection).config.params.vectors
    assert vectors.size == small.dim
    assert_holds(store, collection, MARKDOWN["points"])
