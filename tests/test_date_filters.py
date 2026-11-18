"""Date bounds: the interval filter, the legacy filter, the choice between them, undated chunks.

The corpus, the ranges and the oracle are those of ``date_corpus.py``; the same questions are
asked of a real server in ``tests/integration/test_date_intervals_integration.py``. Here the
collections live in qdrant-client's local mode, which keeps no payload index: ``auto`` can
only answer "legacy" there, so the two filters are asked for by name and the detection is
tested against a client that reports the payload schema it is told to.
"""

from __future__ import annotations

import datetime
import json
import logging
import types

import pytest
from date_corpus import (
    DATE_DOCUMENTS,
    DATE_QUERY,
    DATE_SPEC,
    DOCUMENT_COUNT,
    STATES,
    D,
    all_ranges,
    date_points,
    expected_documents,
)
from qdrant_client import QdrantClient, models

import rag_engine.store as store_module
from rag_engine.config import DATE_FILTER_MODES, Settings
from rag_engine.embedding import FakeEmbedder
from rag_engine.errors import RagEngineError
from rag_engine.eval import _filters_dict
from rag_engine.schema import PAYLOAD_INDEXES
from rag_engine.search import SearchError, SearchFilters, _qdrant_filter, search
from rag_engine.store import QdrantAuthError, QdrantStore

UTC = datetime.timezone.utc
# The edges written by hand and a few dozen random ranges; the real server is asked hundreds.
RANGES = all_ranges(45, seed=7)
FIRST = datetime.datetime(1, 1, 1, tzinfo=UTC)


def at(text: str) -> datetime.datetime:
    day = D(text)
    return datetime.datetime(day.year, day.month, day.day, tzinfo=UTC)


def date_range(key: str, **bounds) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.DatetimeRange(**bounds))


# --------------------------------------------------------------------------------------
# The filter sent to Qdrant, in each mode
# --------------------------------------------------------------------------------------

BOTH = SearchFilters(date_from="2016-03-01", date_to="2016-03-31")
LEGACY_BOTH = date_range("date", gte=at("2016-03-01"), lt=at("2016-04-01"))
INTERVAL_BOTH = models.Filter(
    must=[date_range("date_start", lte=at("2016-03-31")), date_range("date_end", gte=at("2016-03-01"))]
)
NO_DATE = models.Filter(must_not=[date_range("date", gte=FIRST)])
NO_DATE_NO_INTERVAL = models.Filter(
    must_not=[date_range("date_start", gte=FIRST), date_range("date", gte=FIRST)]
)


def test_the_legacy_filter_is_the_filter_of_schema_v1_unchanged():
    assert _qdrant_filter(BOTH) == models.Filter(must=[LEGACY_BOTH])
    assert _qdrant_filter(BOTH, intervals=False) == _qdrant_filter(BOTH)


def test_the_interval_filter_is_the_overlap_or_the_legacy_condition():
    assert _qdrant_filter(BOTH, intervals=True) == models.Filter(
        must=[models.Filter(should=[INTERVAL_BOTH, LEGACY_BOTH])]
    )


def test_the_interval_bounds_are_both_inclusive_midnights_and_one_bound_gives_one_condition():
    lower = _qdrant_filter(SearchFilters(date_from="2016-12-31"), intervals=True)
    assert lower == models.Filter(
        must=[
            models.Filter(
                should=[
                    models.Filter(must=[date_range("date_end", gte=at("2016-12-31"))]),
                    date_range("date", gte=at("2016-12-31"), lt=None),
                ]
            )
        ]
    )
    upper = _qdrant_filter(SearchFilters(date_to="2016-02-29"), intervals=True)
    assert upper == models.Filter(
        must=[
            models.Filter(
                should=[
                    models.Filter(must=[date_range("date_start", lte=at("2016-02-29"))]),
                    date_range("date", gte=None, lt=at("2016-03-01")),
                ]
            )
        ]
    )


def test_the_last_day_of_the_calendar_is_a_bound_like_another_for_the_intervals():
    (condition,) = _qdrant_filter(SearchFilters(date_to="9999-12-31"), intervals=True).must
    assert condition.should == [
        models.Filter(must=[date_range("date_start", lte=at("9999-12-31"))]),
        date_range("date", gte=FIRST),  # the legacy half: any readable date
    ]


def test_include_undated_adds_the_points_without_a_readable_date_in_both_modes():
    undated = SearchFilters(date_from="2016-03-01", date_to="2016-03-31", include_undated=True)
    assert _qdrant_filter(undated) == models.Filter(must=[models.Filter(should=[LEGACY_BOTH, NO_DATE])])
    assert _qdrant_filter(undated, intervals=True) == models.Filter(
        must=[models.Filter(should=[INTERVAL_BOTH, LEGACY_BOTH, NO_DATE_NO_INTERVAL])]
    )


@pytest.mark.parametrize("intervals", [False, True])
def test_include_undated_changes_nothing_without_a_date_bound(intervals):
    assert _qdrant_filter(SearchFilters(include_undated=True), intervals=intervals) is None
    with_source = SearchFilters(source_codes=["ecb"], include_undated=True)
    assert _qdrant_filter(with_source, intervals=intervals) == _qdrant_filter(
        SearchFilters(source_codes=["ecb"]), intervals=intervals
    )


@pytest.mark.parametrize("intervals", [False, True])
def test_the_date_condition_is_one_more_must_next_to_the_other_fields(intervals):
    filters = SearchFilters(
        corpus="dates", source_codes=["day"], date_from="2016-03-10", include_undated=True
    )
    built = _qdrant_filter(filters, intervals=intervals)
    assert [type(condition) for condition in built.must] == [
        models.FieldCondition,
        models.FieldCondition,
        models.Filter,
    ]
    assert built.should is None and built.must_not is None


# --------------------------------------------------------------------------------------
# include_undated as a filter field
# --------------------------------------------------------------------------------------


def test_include_undated_defaults_to_false_and_is_the_last_field():
    assert SearchFilters().include_undated is False
    assert SearchFilters(date_from="2016-03-10").include_undated is False
    assert SearchFilters(include_undated=True).include_undated is True
    assert SearchFilters(include_undated=False) == SearchFilters()
    assert SearchFilters(include_undated=True) != SearchFilters()
    assert hash(SearchFilters(include_undated=True)) == hash(SearchFilters(include_undated=True))


@pytest.mark.parametrize("bad", [None, 0, 1, "true", "false", "", [], [True], 1.0], ids=repr)
def test_include_undated_must_be_a_boolean(bad):
    with pytest.raises(SearchError, match="include_undated must be True or False") as raised:
        SearchFilters(date_from="2016-03-10", include_undated=bad)
    assert isinstance(raised.value, ValueError) and isinstance(raised.value, RagEngineError)


def test_from_mapping_reads_include_undated_as_a_json_boolean():
    given = json.loads('{"date_from": "2016-03-01", "include_undated": true}')
    assert SearchFilters.from_mapping(given) == SearchFilters(date_from="2016-03-01", include_undated=True)
    assert SearchFilters.from_mapping({"include_undated": False}) == SearchFilters()
    # A null is "not set", as for every other field: the default.
    assert SearchFilters.from_mapping({"date_to": "2016-03-01", "include_undated": None}) == SearchFilters(
        date_to="2016-03-01"
    )


@pytest.mark.parametrize("bad", [0, 1, "true", "false", "yes", [True]], ids=repr)
def test_from_mapping_refuses_an_include_undated_that_is_not_a_boolean(bad):
    with pytest.raises(SearchError, match="include_undated must be True or False"):
        SearchFilters.from_mapping({"date_from": "2016-03-01", "include_undated": bad})


def test_the_evaluation_writes_include_undated_only_when_it_is_set():
    # The reports of the question files that never use it are unchanged, byte for byte.
    assert _filters_dict(SearchFilters()) is None
    assert _filters_dict(SearchFilters(date_from="2016-03-01")) == {"date_from": "2016-03-01"}
    assert _filters_dict(SearchFilters(date_from="2016-03-01", include_undated=True)) == {
        "date_from": "2016-03-01",
        "include_undated": True,
    }
    assert json.dumps(_filters_dict(SearchFilters(date_to="2016-03-01", include_undated=True)))


# --------------------------------------------------------------------------------------
# The documents that come back, against the oracle (local mode)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stores():
    """One local-mode collection per state of the corpus (read only), named after the state."""
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client, settings=Settings(batch_size=16))
    for state in STATES:
        store.ensure_collection(state, DATE_SPEC.dim)
        assert store.upsert(state, date_points(state)) == DOCUMENT_COUNT
    yield store
    client.close()


def found(store, collection: str, date_from, date_to, **kwargs) -> set[str]:
    include_undated = kwargs.pop("include_undated", False)
    filters = SearchFilters(date_from=date_from, date_to=date_to, include_undated=include_undated)
    hits = search(
        DATE_QUERY,
        collection=collection,
        store=store,
        embedder=FakeEmbedder(DATE_SPEC),
        filters=filters,
        limit=DOCUMENT_COUNT + 10,
        max_limit=DOCUMENT_COUNT + 10,
        **kwargs,
    )
    documents = [hit.doc_id for hit in hits]
    assert len(documents) == len(set(documents))
    return set(documents)


def test_the_corpus_holds_every_kind_of_document_in_both_halves():
    precisions = {document.precision for document in DATE_DOCUMENTS if document.day is not None}
    assert {"day", "month", "year", None, "fortnight"} <= precisions
    assert sum(document.day is None for document in DATE_DOCUMENTS) >= 8
    halves = {
        (index % 2, document.precision) for index, document in enumerate(DATE_DOCUMENTS) if document.day
    }
    for precision in ("day", "month", "year"):
        assert (0, precision) in halves and (1, precision) in halves
    assert len(RANGES) >= 80
    assert any(a is None for a, _b in RANGES) and any(b is None for _a, b in RANGES)
    assert any(a == b for a, b in RANGES)


def test_the_points_of_each_state_carry_the_interval_or_not(stores):
    def carrying(state: str) -> int:
        records, _ = stores.client.scroll(state, limit=1000, with_payload=True)
        assert len(records) == DOCUMENT_COUNT
        for record in records:
            assert ("date_start" in record.payload) == ("date_end" in record.payload)
        return sum("date_start" in record.payload for record in records)

    dated_by_the_engine = sum(document.read_by_engine is not None for document in DATE_DOCUMENTS)
    assert carrying("v1") == 0
    assert carrying("v2") == dated_by_the_engine
    assert 0 < carrying("half") < dated_by_the_engine


@pytest.mark.parametrize("include_undated", [False, True], ids=["dated only", "with undated"])
@pytest.mark.parametrize("mode", ["legacy", "interval"])
@pytest.mark.parametrize("state", STATES)
def test_every_range_returns_what_the_oracle_says(stores, state, mode, include_undated):
    wrong = []
    for date_from, date_to in RANGES:
        got = found(stores, state, date_from, date_to, date_filter_mode=mode, include_undated=include_undated)
        want = expected_documents(state, mode, date_from, date_to, include_undated)
        if got != want:
            wrong.append((date_from, date_to, sorted(got - want), sorted(want - got)))
    assert not wrong, f"{len(wrong)} of {len(RANGES)} ranges differ; first: {wrong[:3]}"


@pytest.mark.parametrize("include_undated", [False, True], ids=["dated only", "with undated"])
def test_nothing_the_legacy_filter_returned_is_lost_at_any_stage_of_the_upgrade(stores, include_undated):
    """v1 searched as before <= half upgraded <= fully upgraded, for every range."""
    grew = 0
    for date_from, date_to in RANGES:
        kwargs = {"include_undated": include_undated}
        before = found(stores, "v1", date_from, date_to, date_filter_mode="legacy", **kwargs)
        # The indexes exist, no point upgraded yet: the interval filter answers as the legacy one.
        assert found(stores, "v1", date_from, date_to, date_filter_mode="interval", **kwargs) == before
        half = found(stores, "half", date_from, date_to, date_filter_mode="interval", **kwargs)
        full = found(stores, "v2", date_from, date_to, date_filter_mode="interval", **kwargs)
        # "engine-only" has no date the index reads: undated until it is given its interval.
        lost = (before - half) | (half - full)
        assert lost <= ({"engine-only"} if include_undated else set()), (date_from, date_to, lost)
        grew += len(full) > len(before)
    assert grew > 10  # the intervals do find more: month and year documents inside their period


def test_a_month_precision_document_is_found_by_a_day_inside_its_month(stores):
    one_day = ("2016-07-14", "2016-07-14")
    assert found(stores, "v2", *one_day, date_filter_mode="interval") >= {"month-july"}
    assert "month-july" not in found(stores, "v2", *one_day, date_filter_mode="legacy")
    assert "month-july" not in found(stores, "v1", *one_day, date_filter_mode="interval")
    # The same date with a precision the engine does not know stays a day.
    assert "day-garbage" not in found(stores, "v2", *one_day, date_filter_mode="interval")
    first = found(stores, "v2", "2016-07-01", "2016-07-01", date_filter_mode="interval")
    assert {"month-july", "day-garbage", "day-wrong-case"} <= first


def test_a_year_precision_document_is_found_at_both_edges_of_its_year_and_not_beyond(stores):
    def year_2016(date_from, date_to) -> bool:
        return "year-2016" in found(stores, "v2", date_from, date_to, date_filter_mode="interval")

    assert year_2016("2016-12-31", None) and year_2016("2016-12-31", "2016-12-31")
    assert year_2016(None, "2016-01-01") and year_2016("2016-01-01", "2016-01-01")
    assert not year_2016("2017-01-01", None) and not year_2016(None, "2015-12-31")
    assert not year_2016("2017-01-01", "2017-01-01") and not year_2016("2015-12-31", "2015-12-31")


def test_include_undated_adds_exactly_the_undated_documents(stores):
    undated = {document.doc_id for document in DATE_DOCUMENTS if document.day is None}
    for mode in ("legacy", "interval"):
        without = found(stores, "v2", "2016-03-10", "2016-03-10", date_filter_mode=mode)
        with_undated = found(
            stores, "v2", "2016-03-10", "2016-03-10", date_filter_mode=mode, include_undated=True
        )
        added = with_undated - without
        # In interval mode "engine-only" is dated by its interval, and is in both answers.
        assert added == undated - without
        assert added and not (without - with_undated)
    everything = found(
        stores, "v2", "0001-01-01", "9999-12-31", date_filter_mode="interval", include_undated=True
    )
    assert everything == {document.doc_id for document in DATE_DOCUMENTS}


# --------------------------------------------------------------------------------------
# auto: reading the payload schema of the collection
# --------------------------------------------------------------------------------------


def index(kind):
    return types.SimpleNamespace(data_type=kind, points=0)


V1_SCHEMA = {field: index(models.PayloadSchemaType(kind)) for field, kind in PAYLOAD_INDEXES[:7]}
V2_SCHEMA = {field: index(models.PayloadSchemaType(kind)) for field, kind in PAYLOAD_INDEXES}


class SchemaClient:
    """A client that answers ``get_collection`` with the payload schema it holds, and counts the calls."""

    def __init__(self, schema=None) -> None:
        self.schema = schema
        self.asked: list[str] = []
        self.requests: list[dict] = []
        self.error: Exception | None = None

    def get_collection(self, collection_name):
        self.asked.append(collection_name)
        if self.error is not None:
            raise self.error
        schema = (
            self.schema[collection_name] if isinstance(self.schema, dict) and self.by_name else self.schema
        )
        return types.SimpleNamespace(payload_schema=schema)

    by_name = False

    def query_points(self, **kwargs):
        self.requests.append(kwargs)
        return types.SimpleNamespace(points=[])

    def create_payload_index(self, collection_name, field_name, field_schema, wait):
        self.schema = {**(self.schema or {}), field_name: index(field_schema)}


class Clock:
    def __init__(self, monkeypatch) -> None:
        self.now = 1000.0
        monkeypatch.setattr(store_module, "_now", lambda: self.now)


@pytest.fixture
def clock(monkeypatch):
    return Clock(monkeypatch)


def schema_store(schema=None) -> tuple[QdrantStore, SchemaClient]:
    client = SchemaClient(schema)
    return QdrantStore(client=client), client


@pytest.mark.parametrize(
    ("schema", "answer"),
    [
        (None, False),
        ({}, False),
        (V1_SCHEMA, False),  # the production collection before its upgrade
        (V2_SCHEMA, True),
        ({"date_start": index("datetime"), "date_end": index("datetime")}, True),  # kinds as plain strings
        ({**V1_SCHEMA, "date_start": V2_SCHEMA["date_start"]}, False),  # one of the two only
        ({**V1_SCHEMA, "date_end": V2_SCHEMA["date_end"]}, False),
        ({**V2_SCHEMA, "date_end": index(models.PayloadSchemaType.KEYWORD)}, False),  # another kind
        ({**V2_SCHEMA, "date_start": index(None)}, False),
    ],
    ids=["none", "empty", "v1", "v2", "strings", "start only", "end only", "keyword", "no kind"],
)
def test_has_date_intervals_needs_a_datetime_index_on_both_fields(schema, answer):
    store, client = schema_store(schema)
    assert store.has_date_intervals("c") is answer
    assert client.asked == ["c"]


def test_the_answer_is_kept_for_max_age_seconds_then_asked_again(clock):
    store, client = schema_store(V1_SCHEMA)
    assert store.has_date_intervals("c", max_age=60) is False
    client.schema = V2_SCHEMA  # the collection is upgraded by another process
    clock.now += 59.9
    assert store.has_date_intervals("c", max_age=60) is False
    assert client.asked == ["c"]
    clock.now += 0.1  # 60 seconds old: no longer fresh
    assert store.has_date_intervals("c", max_age=60) is True
    assert client.asked == ["c", "c"]
    clock.now += 30
    assert store.has_date_intervals("c", max_age=60) is True
    assert client.asked == ["c", "c"]


def test_the_default_max_age_is_the_default_of_the_settings(clock):
    assert Settings.date_filter_refresh == 300
    store, client = schema_store(V1_SCHEMA)
    store.has_date_intervals("c")
    clock.now += 299
    store.has_date_intervals("c")
    assert len(client.asked) == 1
    clock.now += 1
    store.has_date_intervals("c")
    assert len(client.asked) == 2


def test_max_age_zero_asks_every_time_and_a_shorter_max_age_wins_over_a_stored_answer(clock):
    store, client = schema_store(V1_SCHEMA)
    for _ in range(3):
        store.has_date_intervals("c", max_age=0)
    assert len(client.asked) == 3
    clock.now += 10
    store.has_date_intervals("c", max_age=3600)  # ten seconds old: fresh enough for an hour
    assert len(client.asked) == 3
    store.has_date_intervals("c", max_age=5)  # not fresh enough for five seconds
    assert len(client.asked) == 4


def test_a_clock_that_went_back_does_not_keep_an_answer_for_ever(clock):
    store, client = schema_store(V1_SCHEMA)
    store.has_date_intervals("c", max_age=60)
    clock.now -= 500
    store.has_date_intervals("c", max_age=60)
    assert len(client.asked) == 2


def test_the_answer_is_kept_per_collection_and_per_store(clock):
    client = SchemaClient({"old": V1_SCHEMA, "new": V2_SCHEMA})
    client.by_name = True
    store = QdrantStore(client=client)
    assert [store.has_date_intervals(name) for name in ("old", "new", "old", "new")] == [False, True] * 2
    assert client.asked == ["old", "new"]
    other = QdrantStore(client=client)
    assert other.has_date_intervals("new") is True
    assert client.asked == ["old", "new", "new"]


@pytest.mark.parametrize("bad", [-1, -0.001, float("nan"), float("inf"), True, "60", [60], 10**400], ids=repr)
def test_a_max_age_that_is_not_zero_or_a_positive_number_is_refused(bad):
    store, client = schema_store(V1_SCHEMA)
    with pytest.raises(ValueError, match="max_age must be zero or a positive number of seconds"):
        store.has_date_intervals("c", max_age=bad)
    assert client.asked == []


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Unexpected Response: {status_code}")
        self.status_code = status_code


def test_a_failed_read_is_raised_as_it_is_and_nothing_is_remembered(clock):
    store, client = schema_store(V2_SCHEMA)
    client.error = StatusError(500)
    with pytest.raises(StatusError) as raised:
        store.has_date_intervals("c")
    assert raised.value is client.error
    client.error = StatusError(401)
    with pytest.raises(QdrantAuthError, match="HTTP 401"):
        store.has_date_intervals("c")
    client.error = None
    assert store.has_date_intervals("c") is True
    assert len(client.asked) == 3
    # A failure after a good answer does not erase it either: it is simply raised.
    clock.now += 1000
    client.error = StatusError(503)
    with pytest.raises(StatusError):
        store.has_date_intervals("c")


def test_the_mode_is_logged_once_and_again_only_when_it_changes(clock, caplog):
    store, client = schema_store(V1_SCHEMA)
    with caplog.at_level(logging.INFO, logger="rag_engine.store"):
        for _ in range(4):
            store.has_date_intervals("c", max_age=0)
        legacy = (
            "Qdrant collection 'c' has no datetime index on both date_start and date_end: "
            "date filters in auto mode use the date field (legacy)"
        )
        assert [record.getMessage() for record in caplog.records] == [legacy]
        client.schema = V2_SCHEMA
        for _ in range(4):
            store.has_date_intervals("c", max_age=0)
        assert len(caplog.records) == 2
        assert caplog.records[1].getMessage() == (
            "Qdrant collection 'c' has datetime indexes on date_start and date_end: "
            "date filters in auto mode use the intervals"
        )
        store.has_date_intervals("other", max_age=0)
        assert len(caplog.records) == 3  # another collection: its own first line
        client.schema = V1_SCHEMA  # an index was dropped: said too
        store.has_date_intervals("c", max_age=0)
        assert len(caplog.records) == 4
    assert all(record.levelno == logging.INFO for record in caplog.records)


def test_creating_the_indexes_through_the_store_drops_the_remembered_answer(clock):
    store, client = schema_store(dict(V1_SCHEMA))
    assert store.has_date_intervals("c", max_age=3600) is False
    assert store.ensure_payload_indexes("c") == ["date_start", "date_end"]
    assert store.has_date_intervals("c", max_age=3600) is True  # no wait: asked again at once
    # Indexes that have nothing to do with the dates leave the answer alone.
    asked = len(client.asked)
    store.ensure_payload_indexes("c", [("title", "keyword")])
    assert store.has_date_intervals("c", max_age=3600) is True
    assert len(client.asked) == asked + 1  # the one read of ensure_payload_indexes itself


def test_local_mode_keeps_no_index_and_always_answers_legacy(stores):
    assert stores.has_date_intervals("v2", max_age=0) is False


# --------------------------------------------------------------------------------------
# search: which filter is sent, and when the collection is asked
# --------------------------------------------------------------------------------------


def run(store, filters=None, **kwargs):
    return search(
        DATE_QUERY, collection="c", store=store, embedder=FakeEmbedder(DATE_SPEC), filters=filters, **kwargs
    )


def test_auto_is_the_default_and_follows_the_indexes_of_the_collection():
    assert DATE_FILTER_MODES == ("auto", "interval", "legacy") and Settings.date_filter_mode == "auto"
    for schema, intervals in ((V1_SCHEMA, False), (V2_SCHEMA, True)):
        store, client = schema_store(schema)
        run(store, BOTH)
        run(store, BOTH, date_filter_mode="auto")
        run(store, BOTH, date_filter_mode=None)
        assert [request["query_filter"] for request in client.requests] == [
            _qdrant_filter(BOTH, intervals=intervals)
        ] * 3
        assert client.asked == ["c"]  # asked once, then remembered


@pytest.mark.parametrize("schema", [V1_SCHEMA, V2_SCHEMA], ids=["v1", "v2"])
@pytest.mark.parametrize(("mode", "intervals"), [("interval", True), ("legacy", False)])
def test_a_mode_given_by_name_is_applied_without_asking_the_collection(schema, mode, intervals):
    store, client = schema_store(schema)
    run(store, BOTH, date_filter_mode=mode)
    assert client.requests[0]["query_filter"] == _qdrant_filter(BOTH, intervals=intervals)
    assert client.asked == []


@pytest.mark.parametrize(
    "filters",
    [None, SearchFilters(), SearchFilters(source_codes=["ecb"]), SearchFilters(include_undated=True)],
    ids=["none", "empty", "a source", "undated alone"],
)
def test_a_search_without_a_date_bound_never_asks_the_collection(filters):
    store, client = schema_store(V2_SCHEMA)
    run(store, filters)
    assert client.asked == [] and len(client.requests) == 1


def test_a_running_service_follows_an_upgrade_within_the_refresh_time(clock):
    store, client = schema_store(V1_SCHEMA)
    run(store, BOTH, date_filter_refresh=120)
    client.schema = V2_SCHEMA  # another process creates the two indexes
    clock.now += 119
    run(store, BOTH, date_filter_refresh=120)
    clock.now += 1
    run(store, BOTH, date_filter_refresh=120)
    assert [request["query_filter"] for request in client.requests] == [
        _qdrant_filter(BOTH),
        _qdrant_filter(BOTH),
        _qdrant_filter(BOTH, intervals=True),
    ]
    assert len(client.asked) == 2


def test_refresh_zero_asks_before_every_search_with_a_date_bound():
    store, client = schema_store(V1_SCHEMA)
    for _ in range(3):
        run(store, BOTH, date_filter_refresh=0)
    assert len(client.asked) == 3


def test_the_settings_are_passed_by_the_caller(monkeypatch):
    monkeypatch.setenv("RAG_ENGINE_DATE_FILTER_MODE", "legacy")
    store, client = schema_store(V2_SCHEMA)
    run(store, BOTH)  # the environment is not read: auto
    assert client.requests[-1]["query_filter"] == _qdrant_filter(BOTH, intervals=True)
    settings = Settings.from_env()
    run(
        store,
        BOTH,
        date_filter_mode=settings.date_filter_mode,
        date_filter_refresh=settings.date_filter_refresh,
    )
    assert client.requests[-1]["query_filter"] == _qdrant_filter(BOTH)


@pytest.mark.parametrize("bad", ["", "Auto", "intervals", "v2", 1, True, ["auto"]], ids=repr)
def test_an_unknown_date_filter_mode_is_refused_before_anything_is_sent(bad):
    store, client = schema_store(V2_SCHEMA)
    with pytest.raises(SearchError, match="date_filter_mode must be one of auto, interval, legacy"):
        run(store, None, date_filter_mode=bad)
    assert client.asked == [] and client.requests == []


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "300", [1], 10**400], ids=repr)
def test_a_date_filter_refresh_that_is_not_zero_or_positive_is_refused(bad):
    store, client = schema_store(V2_SCHEMA)
    with pytest.raises(SearchError, match="date_filter_refresh must be zero or a positive number of seconds"):
        run(store, BOTH, date_filter_refresh=bad)
    assert client.asked == [] and client.requests == []


# --------------------------------------------------------------------------------------
# auto when the collection cannot be read: never a new way for a search to fail
# --------------------------------------------------------------------------------------

UNREADABLE = "could not be read"


def warnings_of(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]


@pytest.mark.parametrize(
    "error",
    [
        StatusError(500),
        StatusError(503),
        TimeoutError("timed out"),
        ConnectionError("refused"),
        StatusError(403),
    ],
    ids=["500", "503", "timeout", "connection", "403"],
)
def test_a_cold_store_that_cannot_read_the_collection_serves_the_legacy_filter(clock, caplog, error):
    store, client = schema_store(V2_SCHEMA)
    client.error = error
    with caplog.at_level(logging.INFO, logger="rag_engine.store"):
        assert run(store, BOTH) == []
    assert [request["query_filter"] for request in client.requests] == [_qdrant_filter(BOTH)]
    (warning,) = warnings_of(caplog)
    assert "'c'" in warning and UNREADABLE in warning and "legacy" in warning
    # A refused key is named as the engine names it; the search that follows raises it itself.
    assert ("QdrantAuthError" if error.args == StatusError(403).args else type(error).__name__) in warning


def test_a_warm_store_keeps_its_last_answer_when_the_collection_cannot_be_read(clock, caplog):
    store, client = schema_store(V2_SCHEMA)
    run(store, BOTH, date_filter_refresh=60)  # known: intervals
    client.error = StatusError(503)
    clock.now += 61  # the answer has expired, and the collection cannot be read
    with caplog.at_level(logging.WARNING, logger="rag_engine.store"):
        run(store, BOTH, date_filter_refresh=60)
    assert client.requests[-1]["query_filter"] == _qdrant_filter(BOTH, intervals=True)
    (warning,) = warnings_of(caplog)
    assert "last known answer (the intervals)" in warning


def test_no_request_and_no_warning_per_search_while_the_collection_cannot_be_read(clock, caplog):
    store, client = schema_store(V2_SCHEMA)
    client.error = StatusError(500)
    with caplog.at_level(logging.INFO, logger="rag_engine.store"):
        for _ in range(5):
            run(store, BOTH, date_filter_refresh=0, date_filter_retry=30)
            clock.now += 5  # 25 seconds in all: inside the retry delay
        assert len(client.asked) == 1 and len(client.requests) == 5
        clock.now += 5  # 30 seconds after the failure: asked again, fails again
        run(store, BOTH, date_filter_refresh=0, date_filter_retry=30)
        assert len(client.asked) == 2
        clock.now += 29
        run(store, BOTH, date_filter_refresh=0, date_filter_retry=30)
        assert len(client.asked) == 2
        assert len(warnings_of(caplog)) == 1  # said once, not once per search nor per retry
    assert all(request["query_filter"] == _qdrant_filter(BOTH) for request in client.requests)


def test_the_detection_recovers_after_the_retry_delay_and_says_so_once(clock, caplog):
    store, client = schema_store(V2_SCHEMA)
    client.error = StatusError(500)
    with caplog.at_level(logging.INFO, logger="rag_engine.store"):
        run(store, BOTH, date_filter_retry=10)
        client.error = None  # the server is back
        clock.now += 9
        run(store, BOTH, date_filter_retry=10)  # not asked yet
        assert client.requests[-1]["query_filter"] == _qdrant_filter(BOTH)
        clock.now += 1
        run(store, BOTH, date_filter_retry=10)
        run(store, BOTH, date_filter_retry=10)
        assert client.requests[-1]["query_filter"] == _qdrant_filter(BOTH, intervals=True)
        assert len(client.asked) == 2
        messages = [record.getMessage() for record in caplog.records]
        assert len(warnings_of(caplog)) == 1
        assert sum("can be read again" in message for message in messages) == 1
        # A second outage is a new change of state: said again, once.
        client.error = StatusError(500)
        clock.now += 1000
        run(store, BOTH, date_filter_retry=10)
        run(store, BOTH, date_filter_retry=10)
        assert len(warnings_of(caplog)) == 2
    assert client.requests[-1]["query_filter"] == _qdrant_filter(BOTH, intervals=True)  # the last answer


def test_one_read_at_a_time_and_one_warning_when_many_threads_search_at_once(clock, caplog):
    """Cold, 16 threads: one read of the schema, the others wait for it, one warning. Then, the
    retry delay over and a read hanging, the others answer at once with the legacy filter."""
    import threading

    store, _client = schema_store(V2_SCHEMA)
    reads, entered, release = [], threading.Event(), threading.Event()

    def hanging(collection, *, max_age=None):
        reads.append(collection)
        entered.set()
        assert release.wait(10)
        raise TimeoutError("timed out")

    store.has_date_intervals = hanging
    answers: list[bool] = []
    with caplog.at_level(logging.INFO, logger="rag_engine.store"):
        threads = [
            threading.Thread(target=lambda: answers.append(store.uses_date_intervals("c"))) for _ in range(16)
        ]
        for thread in threads:
            thread.start()
        assert entered.wait(10)
        release.set()
        for thread in threads:
            thread.join(10)
        assert answers == [False] * 16 and reads == ["c"]
        assert len(warnings_of(caplog)) == 1
        # After the delay one thread reads again and hangs: nobody else waits behind it.
        clock.now += 31
        entered.clear()
        release.clear()
        reader = threading.Thread(target=lambda: store.uses_date_intervals("c"))
        reader.start()
        assert entered.wait(10)
        assert [store.uses_date_intervals("c") for _ in range(8)] == [False] * 8  # at once, no read
        release.set()
        reader.join(10)
        assert reads == ["c", "c"] and len(warnings_of(caplog)) == 1


def test_date_fields_of_a_datetime_whose_offset_has_seconds_agrees_with_the_ingestion():
    import datetime
    from zoneinfo import ZoneInfo

    from rag_engine.schema import build_payload, date_fields

    moment = datetime.datetime(1890, 3, 10, 12, 0, tzinfo=ZoneInfo("Europe/Paris"))  # +00:09:21
    fields = date_fields(moment, "day")
    ingested = build_payload(
        "d", {"date": moment}, filename="f", chunk_number=0, ingestion_date="2026-01-01", text="t", page=None
    )
    assert fields["date_start"] == ingested["date_start"] == "1890-03-10T00:00:00Z"
    assert fields["date_end"] == ingested["date_end"]


def test_retry_zero_asks_before_every_search_while_failing(clock):
    store, client = schema_store(V1_SCHEMA)
    client.error = StatusError(500)
    for _ in range(3):
        run(store, BOTH, date_filter_retry=0)
    assert len(client.asked) == 3


def test_the_failure_of_one_collection_does_not_touch_another(clock):
    client = SchemaClient({"c": V2_SCHEMA, "other": V2_SCHEMA})
    client.by_name = True
    store = QdrantStore(client=client)
    assert store.uses_date_intervals("other") is True
    client.error = StatusError(500)
    assert store.uses_date_intervals("c") is False  # never read: legacy
    assert store.uses_date_intervals("other") is True  # still fresh: not even asked
    assert client.asked == ["other", "c"]


def test_has_date_intervals_itself_still_raises(clock):
    store, client = schema_store(V2_SCHEMA)
    client.error = StatusError(500)
    assert store.uses_date_intervals("c") is False
    with pytest.raises(StatusError):
        store.has_date_intervals("c")


def test_a_refused_key_is_reported_by_the_search_itself_not_by_the_detection():
    class Refusing(SchemaClient):
        def query_points(self, **kwargs):
            raise StatusError(401)

    client = Refusing(V2_SCHEMA)
    client.error = StatusError(401)
    with pytest.raises(QdrantAuthError, match="HTTP 401"):
        run(QdrantStore(client=client), BOTH)


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "30", [1], 10**400], ids=repr)
def test_a_date_filter_retry_that_is_not_zero_or_positive_is_refused(bad):
    store, client = schema_store(V2_SCHEMA)
    with pytest.raises(SearchError, match="date_filter_retry must be zero or a positive number of seconds"):
        run(store, BOTH, date_filter_retry=bad)
    with pytest.raises(ValueError, match="retry_after must be zero or a positive number of seconds"):
        store.uses_date_intervals("c", retry_after=bad)
    assert client.asked == [] and client.requests == []
