"""Payload indexes against the real Qdrant given by RAG_ENGINE_IT_QDRANT_URL.

Only a real server keeps payload indexes: qdrant-client's local mode accepts their creation
and forgets it. What a second call does, what an index of another kind does, and what an
index does with a date it cannot read are therefore pinned here.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_engine.config import Settings
from rag_engine.schema import PAYLOAD_INDEXES
from rag_engine.store import CollectionMismatchError, QdrantStore

QDRANT_URL = os.environ.get("RAG_ENGINE_IT_QDRANT_URL", "").strip()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not QDRANT_URL, reason="RAG_ENGINE_IT_QDRANT_URL is not set"),
]

DIM = 4
FIELDS = [field for field, _kind in PAYLOAD_INDEXES]

# Points 1, 2 and 8 have a date Qdrant reads; the others are the shapes it does not read.
PAYLOADS = {
    1: {"doc_id": "a", "source_code": "fr", "date": "2016-03-10", "year": 2016},
    2: {"doc_id": "b", "source_code": "fr", "date": "2016-12-31", "year": 2016},
    3: {"doc_id": "c", "source_code": "fr", "date": "2016-03", "year": 2016},
    4: {"doc_id": "d", "source_code": "de", "date": "2016", "year": 2016},
    5: {"doc_id": "e", "source_code": "de", "date": None, "year": None},
    6: {"doc_id": "f", "source_code": "de"},
    7: {"doc_id": "g", "source_code": "de", "date": "", "year": 2016},
    8: {"doc_id": "h", "source_code": "de", "date": "2017-01-01", "year": 2017},
}
READABLE_DATES = 3
WITH_A_YEAR = 6


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


@pytest.fixture
def populated(store, collection):
    """A collection that already holds the points, without any payload index."""
    store.ensure_collection(collection, DIM)
    store.upsert(collection, _points())
    assert store.client.get_collection(collection_name=collection).payload_schema == {}
    return collection


def _points() -> list[models.PointStruct]:
    return [
        models.PointStruct(id=i, vector=[1.0, 0.1 * i, 0.0, 0.0], payload=payload)
        for i, payload in PAYLOADS.items()
    ]


def _kinds(store, collection) -> dict[str, str]:
    schema = store.client.get_collection(collection_name=collection).payload_schema
    return {field: info.data_type.value for field, info in schema.items()}


def _ids(store, collection, *conditions) -> list[int]:
    found, _ = store.client.scroll(
        collection_name=collection, scroll_filter=models.Filter(must=list(conditions)), limit=100
    )
    return sorted(point.id for point in found)


def _date_range(**bounds) -> models.FieldCondition:
    return models.FieldCondition(key="date", range=models.DatetimeRange(**bounds))


def _date_queries(store, collection) -> list[list[int]]:
    return [
        _ids(store, collection, _date_range(gte="2016-01-01", lte="2016-12-31")),
        _ids(store, collection, _date_range(gte="2016-03-10", lte="2016-03-10")),
        _ids(store, collection, _date_range(gt="2016-03-10")),
        _ids(store, collection, _date_range(lt="2016-12-31")),
        _ids(store, collection, _date_range(gte="1900-01-01")),
    ]


# The bounds are included with gte / lte; a date that is not a full date, an empty one, a null
# and a missing key match no range, not even the widest one.
EXPECTED_DATE_ANSWERS = [[1, 2], [1], [2, 8], [1], [1, 2, 8]]


def test_indexes_are_created_on_a_collection_that_holds_points(store, populated):
    assert store.ensure_payload_indexes(populated) == FIELDS
    assert _kinds(store, populated) == dict(PAYLOAD_INDEXES)
    # No point was rewritten or lost, the ones with an unreadable date included.
    assert store.count(populated) == len(PAYLOADS)
    stored = store.client.retrieve(collection_name=populated, ids=list(PAYLOADS), with_payload=True)
    assert {point.id: point.payload for point in stored} == PAYLOADS


def test_the_second_call_is_a_no_op(store, populated):
    store.ensure_payload_indexes(populated)
    before = store.client.get_collection(collection_name=populated).payload_schema
    assert store.ensure_payload_indexes(populated) == []
    assert store.client.get_collection(collection_name=populated).payload_schema == before
    assert store.count(populated) == len(PAYLOADS)


def test_indexes_on_an_empty_collection_then_points_with_odd_dates_are_accepted(store, collection):
    store.ensure_collection(collection, DIM)
    assert store.ensure_payload_indexes(collection) == FIELDS
    assert store.upsert(collection, _points()) == len(PAYLOADS)  # no value is refused by an index
    assert store.count(collection) == len(PAYLOADS)
    assert _date_queries(store, collection) == EXPECTED_DATE_ANSWERS


def test_only_the_missing_indexes_are_created(store, populated):
    store.client.create_payload_index(
        collection_name=populated, field_name="doc_id", field_schema="keyword", wait=True
    )
    store.client.create_payload_index(
        collection_name=populated, field_name="title", field_schema="text", wait=True
    )
    assert store.ensure_payload_indexes(populated) == FIELDS[1:]
    assert _kinds(store, populated) == {**dict(PAYLOAD_INDEXES), "title": "text"}


@pytest.mark.parametrize(
    ("field", "found", "expected"), [("date", "keyword", "datetime"), ("year", "keyword", "integer")]
)
def test_an_index_of_another_kind_is_refused_and_left_as_it_is(store, populated, field, found, expected):
    store.client.create_payload_index(
        collection_name=populated, field_name=field, field_schema=found, wait=True
    )
    with pytest.raises(CollectionMismatchError) as info:
        store.ensure_payload_indexes(populated)
    message = str(info.value)
    assert repr(field) in message
    assert f"expected a {expected} index" in message
    assert f"found a {found} index" in message
    assert _kinds(store, populated) == {field: found}  # not replaced, and nothing else created


def test_indexes_of_a_missing_collection_raise_and_create_nothing(store, collection):
    with pytest.raises(UnexpectedResponse) as info:
        store.ensure_payload_indexes(collection)
    assert info.value.status_code == 404
    assert not store.client.collection_exists(collection_name=collection)


def _count(store, collection, condition) -> int:
    count_filter = models.Filter(must=[condition])
    return store.client.count(collection_name=collection, count_filter=count_filter, exact=True).count


def _index_sizes(store, collection) -> dict[str, int]:
    schema = store.client.get_collection(collection_name=collection).payload_schema
    return {field: info.points for field, info in schema.items()}


def test_the_indexes_hold_the_values_they_can_read(store, populated):
    """The points each index took in: an unreadable date is left out, not refused."""
    store.ensure_payload_indexes(populated)
    # Through filters, which answer as soon as the indexes exist.
    for field in ("doc_id", "source_code"):
        values = sorted({payload[field] for payload in PAYLOADS.values()})
        any_value = models.FieldCondition(key=field, match=models.MatchAny(any=values))
        assert _count(store, populated, any_value) == len(PAYLOADS)
    any_year = models.FieldCondition(key="year", range=models.Range(gte=-(10**6), lte=10**6))
    assert _count(store, populated, any_year) == WITH_A_YEAR
    any_date = _date_range(gte="0001-01-01", lte="9999-12-31")
    assert _count(store, populated, any_date) == READABLE_DATES
    # The size each index reports. On a large collection this count can lag behind for a
    # few seconds while the index is built: it is polled, never read once.
    expected = {
        "doc_id": len(PAYLOADS),
        "source_code": len(PAYLOADS),
        "year": WITH_A_YEAR,
        "date": READABLE_DATES,
    }
    deadline = time.monotonic() + 15
    while True:
        sizes = _index_sizes(store, populated)
        reported = {field: sizes[field] for field in expected}
        if reported == expected or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    assert reported == expected


def test_keyword_match_on_source_code(store, populated):
    store.ensure_payload_indexes(populated)
    match = models.FieldCondition(key="source_code", match=models.MatchValue(value="fr"))
    assert _ids(store, populated, match) == [1, 2, 3]
    either = models.FieldCondition(key="source_code", match=models.MatchAny(any=["fr", "de"]))
    assert _ids(store, populated, either) == sorted(PAYLOADS)
    unknown = models.FieldCondition(key="source_code", match=models.MatchValue(value="zz"))
    assert _ids(store, populated, unknown) == []


def test_integer_range_on_year(store, populated):
    store.ensure_payload_indexes(populated)
    in_2016 = models.FieldCondition(key="year", range=models.Range(gte=2016, lte=2016))
    assert _ids(store, populated, in_2016) == [1, 2, 3, 4, 7]  # whatever the shape of the date
    since_2017 = models.FieldCondition(key="year", range=models.Range(gte=2017))
    assert _ids(store, populated, since_2017) == [8]
    exactly = models.FieldCondition(key="year", match=models.MatchValue(value=2017))
    assert _ids(store, populated, exactly) == [8]


def test_date_range_on_date_answers_the_same_before_and_after_the_index(store, populated):
    assert _date_queries(store, populated) == EXPECTED_DATE_ANSWERS  # no index yet
    store.ensure_payload_indexes(populated)
    assert _date_queries(store, populated) == EXPECTED_DATE_ANSWERS


def test_filters_combined_in_a_vector_search(store, populated):
    store.ensure_payload_indexes(populated)
    hits = store.client.query_points(
        collection_name=populated,
        query=[1.0, 0.1, 0.0, 0.0],
        query_filter=models.Filter(
            must=[
                models.FieldCondition(key="source_code", match=models.MatchValue(value="fr")),
                models.FieldCondition(key="year", range=models.Range(gte=2016, lte=2016)),
                _date_range(gte="2016-06-01"),
            ]
        ),
        limit=10,
    ).points
    assert [hit.id for hit in hits] == [2]
