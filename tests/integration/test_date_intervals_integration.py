"""Date bounds against the real Qdrant given by RAG_ENGINE_IT_QDRANT_URL.

The corpus, the ranges and the oracle of ``tests/date_corpus.py`` — those the unit tests use
in qdrant-client's local mode — on a real server, where the payload indexes exist and
``date_filter_mode="auto"`` really chooses. Four collections hold the same documents:

``legacy``
    Points as they were written before payload schema v2, with the seven indexes of that
    time: the production collection before its upgrade.
``interval``
    Points of schema v2 with every index of the engine.
``half``
    Every index, and the interval on every other document only: an upgrade under way.
``unindexed``
    Points of schema v2 without any index.

Several hundred ranges, written by hand at the edges and drawn at random, are asked of each
one in each mode, with and without the undated documents, and compared with the oracle. A
last collection is upgraded step by step with ``set_payload`` while a store that was built
before the upgrade keeps searching it.
"""

from __future__ import annotations

import logging
import os
import uuid

import numpy as np
import pytest
from date_corpus import (
    DATE_DOCUMENTS,
    DATE_QUERY,
    DATE_SPEC,
    SERVER_ONLY_DOCUMENTS,
    all_ranges,
    date_points,
    expected_documents,
    has_interval,
)

from rag_engine.config import Settings
from rag_engine.embedding import FakeEmbedder
from rag_engine.schema import DERIVED_KEYS, PAYLOAD_INDEXES, date_interval_fields
from rag_engine.search import SearchFilters, search
from rag_engine.store import QdrantStore, ReservedKeyError

QDRANT_URL = os.environ.get("RAG_ENGINE_IT_QDRANT_URL", "").strip()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not QDRANT_URL, reason="RAG_ENGINE_IT_QDRANT_URL is not set"),
]

DOCUMENTS = DATE_DOCUMENTS + SERVER_ONLY_DOCUMENTS
COUNT = len(DOCUMENTS)
RANGES = all_ranges(300, seed=20261118)
#: The indexes a collection had before payload schema v2.
V1_INDEXES = PAYLOAD_INDEXES[:7]
V2_FIELDS = [field for field, _kind in PAYLOAD_INDEXES[7:]]

#: collection -> (how its points were written, its indexes, the filter "auto" must choose)
COLLECTIONS = {
    "legacy": ("v1", V1_INDEXES, "legacy"),
    "interval": ("v2", PAYLOAD_INDEXES, "interval"),
    "half": ("half", PAYLOAD_INDEXES, "interval"),
    "unindexed": ("v2", (), "legacy"),
}


def connect() -> QdrantStore:
    api_key = os.environ.get("RAG_ENGINE_IT_QDRANT_API_KEY", "").strip() or None
    return QdrantStore.from_url(QDRANT_URL, api_key, settings=Settings(batch_size=32))


@pytest.fixture(scope="module")
def store():
    store = connect()
    yield store
    store.client.close()


def create(store, state: str, indexes) -> str:
    name = f"rag-engine-it-{uuid.uuid4().hex}"
    store.ensure_collection(name, DATE_SPEC.dim)
    assert store.upsert(name, date_points(state, DOCUMENTS)) == COUNT
    if indexes:
        assert store.ensure_payload_indexes(name, indexes) == [field for field, _kind in indexes]
    return name


@pytest.fixture(scope="module")
def collections(store):
    """The four collections, by their name in ``COLLECTIONS`` (read only)."""
    names = {label: create(store, state, indexes) for label, (state, indexes, _mode) in COLLECTIONS.items()}
    yield names
    for name in names.values():
        store.client.delete_collection(collection_name=name)


def found(store, collection: str, date_from, date_to, *, include_undated=False, **kwargs) -> set[str]:
    filters = SearchFilters(date_from=date_from, date_to=date_to, include_undated=include_undated)
    hits = search(
        DATE_QUERY,
        collection=collection,
        store=store,
        embedder=FakeEmbedder(DATE_SPEC),
        filters=filters,
        limit=COUNT + 10,
        max_limit=COUNT + 10,
        **kwargs,
    )
    documents = [hit.doc_id for hit in hits]
    assert len(documents) == len(set(documents))
    return set(documents)


def expected(state: str, mode: str, date_from, date_to, include_undated=False) -> set[str]:
    return expected_documents(state, mode, date_from, date_to, include_undated, DOCUMENTS)


def test_the_ranges_are_several_hundred_and_cover_the_edges():
    assert len(RANGES) >= 330
    assert sum(a == b for a, b in RANGES) > 40  # one-day ranges
    assert any(a is None for a, _b in RANGES) and any(b is None for _a, b in RANGES)
    # The oracle is not trivially "everything" or "nothing" in either mode.
    for mode in ("legacy", "interval"):
        sizes = {len(expected("v2", mode, a, b)) for a, b in RANGES}
        assert 0 in sizes and len(sizes) > 20
    more = sum(expected("v2", "interval", a, b) > expected("v2", "legacy", a, b) for a, b in RANGES)
    assert more > 100  # ranges where only the interval finds a month or a year document


def test_the_four_collections_are_what_they_are_named_after(store, collections):
    for label, (state, indexes, _mode) in COLLECTIONS.items():
        name = collections[label]
        schema = store.client.get_collection(collection_name=name).payload_schema
        assert {field: info.data_type.value for field, info in schema.items()} == dict(indexes)
        records, _ = store.client.scroll(name, limit=1000, with_payload=True)
        assert len(records) == COUNT
        by_id = {record.payload["doc_id"]: record.payload for record in records}
        for index, document in enumerate(DOCUMENTS):
            payload = by_id[document.doc_id]
            carries = has_interval(state, index) and document.read_by_engine is not None
            assert all((key in payload) == carries for key in DERIVED_KEYS), (label, document.doc_id)


@pytest.mark.parametrize(("label", "mode"), [(label, spec[2]) for label, spec in COLLECTIONS.items()])
def test_auto_reads_the_indexes_of_the_collection(collections, label, mode):
    fresh = connect()  # a store that knows nothing of the collection yet
    try:
        assert fresh.has_date_intervals(collections[label]) is (mode == "interval")
    finally:
        fresh.client.close()


@pytest.mark.parametrize("include_undated", [False, True], ids=["dated only", "with undated"])
@pytest.mark.parametrize("asked", ["auto", "interval", "legacy"])
@pytest.mark.parametrize("label", list(COLLECTIONS))
def test_every_range_returns_what_the_oracle_says(store, collections, label, asked, include_undated):
    state, _indexes, chosen = COLLECTIONS[label]
    mode = chosen if asked == "auto" else asked
    wrong = []
    for date_from, date_to in RANGES:
        got = found(
            store,
            collections[label],
            date_from,
            date_to,
            include_undated=include_undated,
            date_filter_mode=asked,
        )
        want = expected(state, mode, date_from, date_to, include_undated)
        if got != want:
            wrong.append((date_from, date_to, sorted(got - want), sorted(want - got)))
    assert not wrong, f"{len(wrong)} of {len(RANGES)} ranges differ; first: {wrong[:3]}"


@pytest.mark.parametrize("include_undated", [False, True], ids=["dated only", "with undated"])
def test_a_half_upgraded_collection_loses_nothing_the_legacy_filter_returned(
    store, collections, include_undated
):
    """legacy collection <= half-upgraded collection <= upgraded collection, in auto mode, for every range."""
    gained = 0
    for date_from, date_to in RANGES:
        kwargs = {"include_undated": include_undated}
        before = found(store, collections["legacy"], date_from, date_to, **kwargs)
        half = found(store, collections["half"], date_from, date_to, **kwargs)
        full = found(store, collections["interval"], date_from, date_to, **kwargs)
        # With include_undated, "engine-only" (a date the index does not read) is undated until
        # it is given its interval, and dated afterwards: the one document that may leave.
        allowed = {"engine-only"} if include_undated else set()
        assert before - half <= allowed, (date_from, date_to)
        assert half - full <= allowed, (date_from, date_to)
        gained += len(half) > len(before)
    assert gained > 50


def test_the_adversarial_documents_on_a_real_server(store, collections):
    def docs(label, date_from, date_to, **kwargs) -> set[str]:
        return found(store, collections[label], date_from, date_to, **kwargs)

    # A month-precision document and a one-day range inside its month.
    assert "month-july" in docs("interval", "2016-07-14", "2016-07-14")
    assert "month-july" not in docs("legacy", "2016-07-14", "2016-07-14")
    # A date with a precision that is not one: a day, in both collections.
    assert "day-garbage" not in docs("interval", "2016-07-14", "2016-07-14")
    assert "day-garbage" in docs("interval", "2016-07-01", "2016-07-01")
    # The last day of a month, of a leap February, of a year: one-day ranges.
    assert "month-thirty-days" in docs("interval", "2016-04-30", "2016-04-30")
    assert "month-thirty-days" not in docs("interval", "2016-05-01", "2016-05-01")
    assert {"month-leap-february", "day-leap"} <= docs("interval", "2016-02-29", "2016-02-29")
    assert "month-plain-february" in docs("interval", "2015-02-28", "2015-02-28")
    assert "month-plain-february" not in docs("interval", "2015-03-01", "2015-03-01")
    assert "month-leap-february" not in docs("interval", "2016-03-01", "2016-03-01")
    new_year_eve = docs("interval", "2016-12-31", "2016-12-31")
    assert {"year-2016", "month-december", "day-new-year-eve"} <= new_year_eve
    assert "day-new-year" not in new_year_eve
    # A year-precision document at the edge of a range, on each side.
    assert "year-2016" in docs("interval", None, "2016-01-01")
    assert "year-2016" not in docs("interval", None, "2015-12-31")
    assert "year-2016" in docs("interval", "2016-12-31", None)
    assert "year-2016" not in docs("interval", "2017-01-01", None)
    # Datetime strings with offsets: the UTC day, as before.
    for label in ("legacy", "interval", "half"):
        on_the_tenth = docs(label, "2016-03-10", "2016-03-10")
        assert {"time-utc", "time-plus", "time-last-instant", "day-stated"} <= on_the_tenth, label
        assert "time-minus" not in on_the_tenth, label
        assert "time-minus" in docs(label, "2016-03-11", "2016-03-11"), label
    # A date the index reads and the engine does not has no interval, and is found by its date.
    assert {"index-only-space", "index-only-list"} <= docs("interval", "2016-03-10", "2016-03-10")
    assert "index-only-space" not in docs("interval", "2016-03-11", "2016-03-31")  # no month for it


def test_include_undated_adds_exactly_the_undated_documents(store, collections):
    undated = {document.doc_id for document in DOCUMENTS if document.day is None}
    for label in COLLECTIONS:
        without = found(store, collections[label], "2016-03-10", "2016-03-10")
        with_undated = found(store, collections[label], "2016-03-10", "2016-03-10", include_undated=True)
        assert without <= with_undated
        assert with_undated - without == undated - without, label
        assert with_undated - without >= undated - {"engine-only"}, label
    everything = found(store, collections["interval"], "0001-01-01", "9999-12-31", include_undated=True)
    assert everything == {document.doc_id for document in DOCUMENTS}
    # Without a bound, the option is not a filter at all.
    hits = search(
        DATE_QUERY,
        collection=collections["interval"],
        store=store,
        embedder=FakeEmbedder(DATE_SPEC),
        filters=SearchFilters(include_undated=True),
        limit=COUNT + 10,
        max_limit=COUNT + 10,
    )
    assert len(hits) == COUNT


# --------------------------------------------------------------------------------------
# set_payload, and an upgrade while a search service runs
# --------------------------------------------------------------------------------------


@pytest.fixture
def production(store):
    """A collection as production holds it today: schema v1 points and the seven indexes of that time."""
    name = create(store, "v1", V1_INDEXES)
    yield name
    store.client.delete_collection(collection_name=name)


def snapshot(store, collection: str) -> dict:
    """Every point as the server holds it, by id: ``(payload, vector)``."""
    points: dict = {}
    offset = None
    while True:
        records, offset = store.client.scroll(
            collection_name=collection, limit=64, offset=offset, with_payload=True, with_vectors=True
        )
        for record in records:
            points[record.id] = (record.payload, record.vector)
        if offset is None:
            return points


def vector_bytes(points: dict) -> dict:
    return {
        point: np.asarray(vector, dtype=np.float32).tobytes() for point, (_payload, vector) in points.items()
    }


def upgrade(store, collection: str, documents) -> int:
    """Give each of ``documents`` its interval, from the date and the precision its points store."""
    written = 0
    for document in documents:
        metadata = document.metadata
        fields = date_interval_fields(metadata.get("date"), metadata.get("date_precision"))
        written += store.set_payload(collection, document.doc_id, fields)
    return written


def test_set_payload_touches_the_payload_of_one_document_and_no_vector(store, production):
    before = snapshot(store, production)
    fields = {
        "date": "2016-07-01",
        "date_precision": "month",
        "date_kind": "period",
        **date_interval_fields("2016-07-01", "month"),
    }

    assert store.set_payload(production, "day-stated", fields) == 1
    assert store.set_payload(production, "no-such-document", fields) == 0
    with pytest.raises(ReservedKeyError, match="text"):
        store.set_payload(production, "day-stated", {"text": "another text"})

    after = snapshot(store, production)
    assert sorted(after) == sorted(before) and store.count(production) == COUNT
    assert vector_bytes(after) == vector_bytes(before)
    assert {point: vector for point, (_payload, vector) in after.items()} == {
        point: vector for point, (_payload, vector) in before.items()
    }
    changed = 0
    for point, (payload, _vector) in after.items():
        old = before[point][0]
        if old["doc_id"] == "day-stated":
            assert payload == {**old, **fields}
            assert payload["date_end"] == "2016-07-31T00:00:00Z"
            changed += 1
        else:
            assert payload == old
    assert changed == 1
    # The corrected document is now found where it belongs.
    assert "day-stated" in found(store, production, "2016-07-20", "2016-07-20", date_filter_mode="interval")
    assert "day-stated" not in found(
        store, production, "2016-03-10", "2016-03-10", date_filter_mode="interval"
    )


def test_a_null_given_to_set_payload_removes_the_key(store, production):
    before = snapshot(store, production)
    assert store.set_payload(production, "day-stated", date_interval_fields("2016-03-10", "day")) == 1
    fields = {"date": None, **date_interval_fields(None)}
    assert fields == {"date": None, "date_start": None, "date_end": None}
    assert store.set_payload(production, "day-stated", fields) == 1
    after = snapshot(store, production)
    (point,) = [point for point, (payload, _vector) in after.items() if payload["doc_id"] == "day-stated"]
    # Qdrant 1.20 merges the payload and drops a key set to null: the document has no date any more.
    assert after[point][0] == {key: value for key, value in before[point][0].items() if key != "date"}
    assert vector_bytes(after) == vector_bytes(before)
    for mode in ("legacy", "interval"):
        kwargs = {"date_filter_mode": mode}
        assert "day-stated" not in found(store, production, "0001-01-01", "9999-12-31", **kwargs)
        assert "day-stated" in found(
            store, production, "2016-03-10", "2016-03-10", include_undated=True, **kwargs
        )


def test_a_collection_upgraded_while_a_service_runs_switches_without_a_restart(store, production, caplog):
    """Indexes first, then the points in two halves: nothing is lost at any step, and no vector moves."""
    reader = connect()  # the long-lived search service
    writer = connect()  # the tool that upgrades the collection
    ranges = RANGES[:120]
    try:

        def answers(**kwargs) -> list[set[str]]:
            return [found(reader, production, date_from, date_to, **kwargs) for date_from, date_to in ranges]

        def oracle(state: str, mode: str) -> list[set[str]]:
            return [expected(state, mode, date_from, date_to) for date_from, date_to in ranges]

        vectors = vector_bytes(snapshot(store, production))
        with caplog.at_level(logging.INFO, logger="rag_engine.store"):
            # 1. Today: no interval index. The service answers as it always did.
            start = answers(date_filter_refresh=3600)
            assert start == oracle("v1", "legacy")

            # 2. The indexes are created by another process, before any point is upgraded.
            assert writer.ensure_payload_indexes(production) == V2_FIELDS
            # Within the refresh time the service has not looked again: same filter, same answers.
            assert answers(date_filter_refresh=3600) == start
            # Past it (here: 0 seconds), it reads the new schema and uses the intervals. No
            # point carries one yet, and every document is still found by its date.
            assert answers(date_filter_refresh=0) == start == oracle("v1", "interval")

            # 3. Half of the documents are given their interval.
            even = [document for index, document in enumerate(DOCUMENTS) if index % 2 == 0]
            assert upgrade(writer, production, even) == len(even)
            half = answers(date_filter_refresh=3600)  # the answer of step 2 is remembered: intervals
            assert half == oracle("half", "interval")
            assert all(before <= now for before, now in zip(start, half))

            # 4. The other half.
            odd = [document for index, document in enumerate(DOCUMENTS) if index % 2 == 1]
            assert upgrade(writer, production, odd) == len(odd)
            full = answers(date_filter_refresh=3600)
            assert full == oracle("v2", "interval")
            assert all(before <= now for before, now in zip(half, full))
            assert sum(len(now) > len(before) for before, now in zip(start, full)) > 20

        # The service said once which filter it uses, and once more when that changed.
        messages = [record.getMessage() for record in caplog.records if production in record.getMessage()]
        said = [message for message in messages if "date filters in auto mode" in message]
        assert len(said) == 2 and "legacy" in said[0] and "use the intervals" in said[1]

        # The upgrade wrote payload keys and nothing else: same points, same vectors, byte for byte.
        after = snapshot(store, production)
        assert vector_bytes(after) == vectors and store.count(production) == COUNT
        originals = {point.id: point.payload for point in date_points("v1", DOCUMENTS)}
        for point, (payload, _vector) in after.items():
            assert {key: value for key, value in payload.items() if key not in DERIVED_KEYS} == originals[
                point
            ]
        # And the upgraded collection holds exactly what an ingestion under schema v2 writes: an
        # undated document was given nulls, which add no key.
        ingested = {point.id: point.payload for point in date_points("v2", DOCUMENTS)}
        assert {point: payload for point, (payload, _vector) in after.items()} == ingested
    finally:
        reader.client.close()
        writer.client.close()
