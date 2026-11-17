"""The store against the real Qdrant given by RAG_ENGINE_IT_QDRANT_URL."""

from __future__ import annotations

import os
import uuid

import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_engine.config import Settings
from rag_engine.store import CollectionMismatchError, CollectionNotFoundError, QdrantStore

QDRANT_URL = os.environ.get("RAG_ENGINE_IT_QDRANT_URL", "").strip()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not QDRANT_URL, reason="RAG_ENGINE_IT_QDRANT_URL is not set"),
]

DIM = 8


@pytest.fixture
def store():
    api_key = os.environ.get("RAG_ENGINE_IT_QDRANT_API_KEY", "").strip() or None
    store = QdrantStore.from_url(QDRANT_URL, api_key, settings=Settings(batch_size=3))
    yield store
    store.client.close()


@pytest.fixture
def collection(store):
    """A collection name of this test only, dropped afterwards (the test creates it or not)."""
    name = f"rag-engine-it-{uuid.uuid4().hex}"
    yield name
    store.client.delete_collection(collection_name=name)


def _points(n: int) -> list[models.PointStruct]:
    return [models.PointStruct(id=i, vector=[0.1 * (i + 1)] * DIM, payload={"n": i}) for i in range(n)]


def test_the_server_is_reachable(store):
    assert store.is_reachable() is True


def test_ensure_collection_creates_then_is_idempotent(store, collection):
    assert not store.client.collection_exists(collection_name=collection)
    store.ensure_collection(collection, DIM)
    vectors = store.client.get_collection(collection_name=collection).config.params.vectors
    assert (vectors.size, vectors.distance) == (DIM, models.Distance.COSINE)
    store.upsert(collection, _points(2))
    store.ensure_collection(collection, DIM)  # second call: no error, the points are kept
    assert store.count(collection) == 2


def test_a_collection_with_another_dimension_is_refused(store, collection):
    store.ensure_collection(collection, DIM * 2)
    with pytest.raises(CollectionMismatchError, match=f"{DIM * 2}-d"):
        store.ensure_collection(collection, DIM)


def test_a_collection_with_named_vectors_is_refused(store, collection):
    store.client.create_collection(
        collection_name=collection,
        vectors_config={"text": models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
    )
    with pytest.raises(CollectionMismatchError, match="named"):
        store.ensure_collection(collection, DIM)


def test_a_collection_with_another_distance_is_refused(store, collection):
    store.client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
    )
    with pytest.raises(CollectionMismatchError, match="Dot"):
        store.ensure_collection(collection, DIM)


# --- check_collection: the read-only check -----------------------------------------------
def _names(store) -> list[str]:
    return sorted(entry.name for entry in store.client.get_collections().collections)


def test_check_collection_of_a_missing_collection_creates_nothing(store, collection):
    before = _names(store)
    with pytest.raises(CollectionNotFoundError, match=collection):
        store.check_collection(collection, DIM)
    assert not store.client.collection_exists(collection_name=collection)
    assert _names(store) == before  # as many collections as before


def test_check_collection_accepts_the_collection_of_a_writer_and_changes_nothing(store, collection):
    store.ensure_collection(collection, DIM)
    store.upsert(collection, _points(3))
    before, names = store.client.get_collection(collection_name=collection), _names(store)
    assert store.check_collection(collection, DIM) is None
    after = store.client.get_collection(collection_name=collection)
    assert after.config == before.config and after.payload_schema == before.payload_schema
    assert store.count(collection) == 3
    assert _names(store) == names


def test_check_collection_refuses_another_dimension(store, collection):
    store.ensure_collection(collection, DIM * 2)
    with pytest.raises(CollectionMismatchError, match=f"found {DIM * 2}-d"):
        store.check_collection(collection, DIM)


@pytest.mark.parametrize("distance", [models.Distance.DOT, models.Distance.EUCLID, models.Distance.MANHATTAN])
def test_check_collection_refuses_another_distance(store, collection, distance):
    store.client.create_collection(
        collection_name=collection, vectors_config=models.VectorParams(size=DIM, distance=distance)
    )
    with pytest.raises(CollectionMismatchError, match=f"found {distance.value} distance"):
        store.check_collection(collection, DIM)


def test_check_collection_refuses_named_vectors(store, collection):
    store.client.create_collection(
        collection_name=collection,
        vectors_config={"text": models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
    )
    with pytest.raises(CollectionMismatchError, match="named vectors \\('text'\\)"):
        store.check_collection(collection, DIM)


def test_check_collection_refuses_a_collection_without_a_dense_vector(store, collection):
    store.client.create_collection(
        collection_name=collection,
        vectors_config={},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    with pytest.raises(CollectionMismatchError, match="no dense vector"):
        store.check_collection(collection, DIM)


def test_a_refused_check_leaves_the_collection_as_it_was(store, collection):
    store.ensure_collection(collection, DIM * 2)
    store.upsert(collection, [models.PointStruct(id=1, vector=[0.5] * (DIM * 2), payload={"n": 1})])
    before, names = store.client.get_collection(collection_name=collection).config, _names(store)
    with pytest.raises(CollectionMismatchError):
        store.check_collection(collection, DIM)
    assert store.client.get_collection(collection_name=collection).config == before
    assert store.count(collection) == 1 and _names(store) == names


@pytest.mark.parametrize("name", ["with/a-slash", "x" * 300], ids=["slash", "too-long"])
def test_a_name_the_server_does_not_accept_is_raised_as_the_client_raises_it(store, name):
    # Not a "collection not found": the server refuses the request before it looks anything up.
    with pytest.raises(UnexpectedResponse) as info:
        store.check_collection(name, DIM)
    assert info.value.status_code == 422
    with pytest.raises(UnexpectedResponse) as info:
        store.query(name, [0.1] * DIM, limit=1)
    assert info.value.status_code == 422


def test_an_unreachable_server_is_raised_as_the_client_raises_it(monkeypatch):
    from qdrant_client.http.exceptions import ResponseHandlingException
    from qdrant_client.qdrant_remote import QdrantRemote

    # The client looks at the version of the server in a thread, and would warn during a later test.
    monkeypatch.setattr(QdrantRemote, "_check_compatibility", staticmethod(lambda *args: None))
    dead = QdrantStore("127.0.0.1", 1, timeout=2, settings=Settings())  # nothing listens on port 1
    with pytest.raises(ResponseHandlingException):
        dead.check_collection("anything", DIM)
    assert dead.is_reachable() is False
    dead.client.close()


def test_creating_a_collection_that_exists_is_survived_as_a_lost_race(store, collection):
    """Pins how the real server refuses a second create: the store must survive it."""

    class Raced:
        def __init__(self, inner):
            self._inner = inner
            self._hidden = True

        def collection_exists(self, collection_name):
            if self._hidden:  # the first look happens before the other process creates
                self._hidden = False
                return False
            return self._inner.collection_exists(collection_name=collection_name)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    store.ensure_collection(collection, DIM)
    raced = QdrantStore(client=Raced(store.client), settings=Settings())
    raced.ensure_collection(collection, DIM)
    with pytest.raises(CollectionMismatchError):
        QdrantStore(client=Raced(store.client), settings=Settings()).ensure_collection(collection, DIM * 2)


@pytest.mark.parametrize("n", [0, 1, 3, 4, 7])
def test_upsert_then_count(store, collection, n):
    store.ensure_collection(collection, DIM)
    assert store.upsert(collection, _points(n)) == n  # batches of 3
    assert store.count(collection) == n
    assert store.upsert(collection, _points(n)) == n  # same ids: overwritten, not added
    assert store.count(collection) == n


def test_count_of_a_missing_collection_raises_and_creates_nothing(store, collection):
    with pytest.raises(UnexpectedResponse) as info:
        store.count(collection)
    assert info.value.status_code == 404
    assert not store.client.collection_exists(collection_name=collection)
