"""Search against the real Qdrant given by RAG_ENGINE_IT_QDRANT_URL.

The matrix of ``tests/search_corpus.py`` — the one the unit tests run in qdrant-client's
local mode — is asked to a real server, on a collection without payload indexes and on a
collection with the indexes of ``ensure_payload_indexes``: a filter must select the same
chunks either way. The date bounds are the point of it: only a real server tells how a
datetime index reads the stored dates.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
from search_corpus import (
    ALL_DOCUMENTS,
    SEARCH_CASES,
    SEARCH_POINT_COUNT,
    SEARCH_SPEC,
    search_points,
)

from rag_engine.config import Settings
from rag_engine.embedding import FakeEmbedder
from rag_engine.schema import PAYLOAD_INDEXES
from rag_engine.search import SearchFilters, search
from rag_engine.store import QdrantStore

QDRANT_URL = os.environ.get("RAG_ENGINE_IT_QDRANT_URL", "").strip()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not QDRANT_URL, reason="RAG_ENGINE_IT_QDRANT_URL is not set"),
]

QUERY = "what did the governing council decide?"
FIELDS = [field for field, _kind in PAYLOAD_INDEXES]


@pytest.fixture(scope="module")
def store():
    api_key = os.environ.get("RAG_ENGINE_IT_QDRANT_API_KEY", "").strip() or None
    store = QdrantStore.from_url(QDRANT_URL, api_key, settings=Settings(batch_size=4))
    yield store
    store.client.close()


def _populated(store) -> str:
    name = f"rag-engine-it-{uuid.uuid4().hex}"
    store.ensure_collection(name, SEARCH_SPEC.dim)
    assert store.upsert(name, search_points()) == SEARCH_POINT_COUNT
    return name


@pytest.fixture(scope="module")
def plain(store):
    """The corpus in a collection without any payload index (read only)."""
    name = _populated(store)
    yield name
    store.client.delete_collection(collection_name=name)


@pytest.fixture(scope="module")
def indexed(store):
    """The corpus in a collection with the payload indexes of the engine (read only)."""
    name = _populated(store)
    assert store.ensure_payload_indexes(name) == FIELDS
    yield name
    store.client.delete_collection(collection_name=name)


@pytest.fixture
def collection(store):
    """A collection name of one test, dropped afterwards (the test creates it or not)."""
    name = f"rag-engine-it-{uuid.uuid4().hex}"
    yield name
    store.client.delete_collection(collection_name=name)


def run(store, collection, query=QUERY, **kwargs):
    kwargs.setdefault("limit", 50)
    return search(query, collection=collection, store=store, embedder=FakeEmbedder(SEARCH_SPEC), **kwargs)


def expected_ranking(query: str = QUERY) -> list[tuple[int, float]]:
    """``(point id, cosine)`` of every chunk of the corpus for ``query``, best first."""
    reference = FakeEmbedder(SEARCH_SPEC)
    vector = reference.encode_query(query).astype(np.float64)
    points = search_points(reference)
    scores = [float(np.dot(vector, np.asarray(point.vector, dtype=np.float64))) for point in points]
    return sorted(zip([point.id for point in points], scores), key=lambda pair: -pair[1])


def _answer(hits) -> list[tuple]:
    """What must not depend on the indexes: the points, their order, their payloads."""
    return [(hit.point_id, hit.to_dict(include_payload=True) | {"score": None}) for hit in hits]


def test_the_two_collections_differ_by_their_indexes_only(store, plain, indexed):
    assert store.client.get_collection(collection_name=plain).payload_schema == {}
    schema = store.client.get_collection(collection_name=indexed).payload_schema
    assert {field: info.data_type.value for field, info in schema.items()} == dict(PAYLOAD_INDEXES)
    assert store.count(plain) == store.count(indexed) == SEARCH_POINT_COUNT


def test_the_filters_select_the_expected_chunks_with_and_without_indexes(store, plain, indexed, search_case):
    filters = SearchFilters(**search_case.filters)
    without = run(store, plain, filters=filters)
    with_indexes = run(store, indexed, filters=filters)

    assert {hit.doc_id for hit in with_indexes} == search_case.documents
    document_of = {point.id: point.payload["doc_id"] for point in search_points()}
    ranking = [point for point, _score in expected_ranking() if document_of[point] in search_case.documents]
    assert [hit.point_id for hit in with_indexes] == ranking
    assert _answer(without) == _answer(with_indexes)
    assert [hit.score for hit in without] == pytest.approx([hit.score for hit in with_indexes], abs=1e-6)


def test_results_are_identical_before_and_after_index_creation(store, collection):
    store.ensure_collection(collection, SEARCH_SPEC.dim)
    store.upsert(collection, search_points())

    def answers() -> dict[str, list]:
        return {
            case.label: [
                (hit.point_id, hit.score, hit.to_dict(include_payload=True))
                for hit in run(store, collection, filters=SearchFilters(**case.filters))
            ]
            for case in SEARCH_CASES
        }

    before = answers()
    assert store.ensure_payload_indexes(collection) == FIELDS
    after = answers()

    assert after == before
    assert {label for label, hits in before.items() if hits} and {
        label for label, hits in before.items() if not hits
    }
    assert store.ensure_payload_indexes(collection) == []
    assert answers() == before


@pytest.mark.parametrize("which", ["plain", "indexed"])
def test_the_upper_date_bound_includes_the_whole_day(store, which, request):
    name = request.getfixturevalue(which)
    # ecb-a is stored as "2016-03-10", ecb-b as "2016-03-10T15:30:00Z", fed-b as "2016-03-11".
    found = {hit.doc_id: hit.date for hit in run(store, name, filters=SearchFilters(date_to="2016-03-10"))}
    assert found == {
        "ecb-a": "2016-03-10",
        "ecb-b": "2016-03-10T15:30:00Z",
        "fed-a": "2016-03-09",
        "snb-a": "2015-06-18",
    }
    one_day = SearchFilters(date_from="2016-03-10", date_to="2016-03-10")
    assert {hit.doc_id for hit in run(store, name, filters=one_day)} == {"ecb-a", "ecb-b"}


# How a real server reads the stored values a day can be written as, around 2016-03-10.
STORED_DATES = {
    1: "2016-03-09",
    2: "2016-03-09T23:59:59Z",
    3: "2016-03-10",
    4: "2016-03-10T00:00:00Z",
    5: "2016-03-10T15:30:00Z",
    6: "2016-03-10T23:59:59.999999Z",
    7: "2016-03-11",
    8: "2016-03-11T00:00:00Z",
    9: "2016-03-11T00:30:00+02:00",  # 2016-03-10T22:30:00Z: a UTC day, not a local one
    10: "2016-03-10T23:30:00-05:00",  # 2016-03-11T04:30:00Z
    11: "2016-03",
    12: "2016",
    13: "",
    14: None,
}
ON_THE_DAY = [3, 4, 5, 6, 9]


@pytest.mark.parametrize("with_indexes", [False, True], ids=["without indexes", "with indexes"])
def test_a_day_is_a_utc_day_whatever_the_time_stored(store, collection, with_indexes):
    store.ensure_collection(collection, SEARCH_SPEC.dim)
    vectors = FakeEmbedder(SEARCH_SPEC).encode_passage([f"text {i}" for i in STORED_DATES])
    store.upsert(
        collection,
        [
            models.PointStruct(id=i, vector=vector.tolist(), payload={"doc_id": f"d{i}", "date": stored})
            for (i, stored), vector in zip(STORED_DATES.items(), vectors)
        ]
        + [models.PointStruct(id=15, vector=vectors[0].tolist(), payload={"doc_id": "d15"})],
    )
    if with_indexes:
        store.ensure_payload_indexes(collection)

    def ids(**bounds) -> list[int]:
        return sorted(hit.point_id for hit in run(store, collection, filters=SearchFilters(**bounds)))

    assert ids(date_from="2016-03-10", date_to="2016-03-10") == ON_THE_DAY
    assert ids(date_to="2016-03-10") == [1, 2, *ON_THE_DAY]
    assert ids(date_from="2016-03-10") == [*ON_THE_DAY[:-1], 7, 8, 9, 10]
    assert ids(date_from="2016-03-11", date_to="2016-03-11") == [7, 8, 10]
    assert ids(date_from="0001-01-01") == list(range(1, 11))
    assert ids(date_to="9999-12-31") == list(range(1, 11))
    assert len(run(store, collection)) == 15


def test_a_document_without_a_full_date_is_excluded_once_a_date_bound_is_set(store, indexed):
    undated = {"boj-a", "boj-b", "boe-a"}
    assert {hit.doc_id for hit in run(store, indexed)} == ALL_DOCUMENTS
    for bounds in ({"date_from": "0001-01-01"}, {"date_to": "9999-12-31"}):
        found = {hit.doc_id for hit in run(store, indexed, filters=SearchFilters(**bounds))}
        assert found == ALL_DOCUMENTS - undated, bounds


def test_scores_are_the_cosines_in_decreasing_order(store, indexed):
    hits = run(store, indexed)
    ranking = expected_ranking()
    assert [hit.point_id for hit in hits] == [point for point, _score in ranking]
    assert [hit.score for hit in hits] == pytest.approx([score for _point, score in ranking], abs=1e-5)


def test_limit_keeps_the_best_hits(store, indexed):
    best = [point for point, _score in expected_ranking()]
    assert [hit.point_id for hit in run(store, indexed, limit=3)] == best[:3]
    assert len(search(QUERY, collection=indexed, store=store, embedder=FakeEmbedder(SEARCH_SPEC))) == 10


def test_a_missing_collection_is_the_error_of_the_client(store):
    with pytest.raises(UnexpectedResponse) as raised:
        run(store, f"rag-engine-it-absent-{uuid.uuid4().hex}")
    assert raised.value.status_code == 404


def test_a_collection_of_another_dimension_is_the_error_of_the_client(store, collection):
    store.ensure_collection(collection, SEARCH_SPEC.dim + 1)
    with pytest.raises(UnexpectedResponse) as raised:
        run(store, collection)
    assert raised.value.status_code == 400
