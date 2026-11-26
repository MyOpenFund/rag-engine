"""The on-disk local index (Qdrant local mode + SQLite FTS5) and the hybrid search over it.

Offline: a ``FakeEmbedder`` and temporary folders. The date filters of the FTS side are checked
against the oracle of ``date_corpus.py``, the one the dense filters are checked against.
"""

from __future__ import annotations

import datetime
import sqlite3
import threading

import numpy as np
import pytest
from date_corpus import DATE_DOCUMENTS, all_ranges, expected_documents
from qdrant_client import models

from rag_engine.chunking import Chunk
from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.local import DEFAULT_MAX_TERMS, LocalIndex, collection_name, fts_query, hybrid_search
from rag_engine.schema import page_for_id, point_id
from rag_engine.search import SearchError, SearchFilters, search
from rag_engine.store import IndexLocked, QdrantStore, StoreError

SPEC = ModelSpec(name="fake", dim=8, passage_prefix="passage: ", query_prefix="query: ")
COLLECTION = "test_local"


@pytest.fixture
def embedder():
    return FakeEmbedder(SPEC)


@pytest.fixture
def index(tmp_path):
    opened = LocalIndex(tmp_path / "index", collection=COLLECTION, dim=SPEC.dim)
    yield opened
    opened.close()


def _add(index, embedder, doc_id, texts, *, first_index=0, contexts=None, **payload):
    chunks = [
        Chunk(text, page=i, context=None if contexts is None else contexts[i]) for i, text in enumerate(texts)
    ]
    vectors = embedder.encode_passage([chunk.embedded_text for chunk in chunks])
    meta = {"doc_type": "10-K", "date": "2024-02-01", "title": f"title {doc_id}", **payload}
    return index.add(doc_id, chunks, vectors, meta, first_index=first_index)


def _unit(vector):
    return (vector / np.linalg.norm(vector)).astype(np.float32)


# --- writing --------------------------------------------------------------------------------------


def test_add_then_dense_finds_doc(index, embedder):
    assert _add(index, embedder, "a", ["capital expenditure rose", "revenue fell"]) == 2
    hits = hybrid_search("capital expenditure", index=index, embedder=embedder, filters=None, limit=5)
    assert {hit.doc_id for hit in hits} == {"a"}
    assert len(hits) == 2
    first = hits[0]
    assert first.title == "title a" and first.doc_type == "10-K" and first.date == "2024-02-01"
    assert first.payload["date_start"] == "2024-02-01T00:00:00Z"  # the interval is derived


def test_add_returns_count_and_ids_are_deterministic_per_doc_and_index(index, embedder):
    _add(index, embedder, "a", ["one", "two"])
    _add(index, embedder, "a", ["three"], first_index=2)
    assert index.count("a") == 3
    hits = hybrid_search("one", index=index, embedder=embedder, filters=None, limit=3)
    ids = {hit.point_id for hit in hits}
    assert ids == {point_id("a", page_for_id(0), 0), point_id("a", page_for_id(1), 1), point_id("a", 0, 2)}
    assert sorted(hit.chunk_number for hit in hits) == [0, 1, 2]


def test_adding_the_same_batch_again_overwrites_it(index, embedder):
    _add(index, embedder, "a", ["one", "two"])
    _add(index, embedder, "a", ["one", "two"])
    assert index.count("a") == 2
    assert index.store.count(COLLECTION) == 2


def test_text_and_context_are_stored_apart_and_the_context_is_searchable(index, embedder):
    _add(index, embedder, "a", ["the body text"], contexts=["Item 7 › Liquidity"])
    hits = hybrid_search("liquidity", index=index, embedder=embedder, filters=None, limit=5)
    assert hits and hits[0].text == "the body text"
    assert hits[0].payload["context"] == "Item 7 › Liquidity"


def test_add_refuses_mismatched_vectors_and_engine_keys(index, embedder):
    chunks = [Chunk("x", page=None)]
    with pytest.raises(ValueError, match="vectors"):
        index.add("a", chunks, np.zeros((2, SPEC.dim), np.float32), {}, first_index=0)
    with pytest.raises(ValueError, match="vectors"):
        index.add("a", chunks, np.zeros((1, 3), np.float32), {}, first_index=0)
    with pytest.raises(ValueError, match="text"):
        index.add("a", chunks, np.ones((1, SPEC.dim), np.float32), {"text": "x"}, first_index=0)
    with pytest.raises(ValueError, match="doc_id"):
        index.add("a", chunks, np.ones((1, SPEC.dim), np.float32), {"doc_id": "b"}, first_index=0)
    with pytest.raises(ValueError, match="first_index"):
        index.add("a", chunks, np.ones((1, SPEC.dim), np.float32), {}, first_index=-1)
    assert index.doc_ids() == set()


def test_delete_removes_from_both_sides(index, embedder):
    _add(index, embedder, "a", ["capex grew"])
    _add(index, embedder, "b", ["capex shrank"])
    index.delete("a")
    assert index.doc_ids() == {"b"}
    assert index.count("a") == 0
    assert index.store.count(COLLECTION) == 1
    rows = index._fts_rows(fts_query("capex"), None, 10)
    assert {row.doc_id for row in rows} == {"b"}
    hits = hybrid_search("capex", index=index, embedder=embedder, filters=None, limit=10)
    assert {hit.doc_id for hit in hits} == {"b"}


def test_the_index_survives_a_reopen(tmp_path, embedder):
    with LocalIndex(tmp_path / "i", collection=COLLECTION, dim=SPEC.dim) as first:
        _add(first, embedder, "a", ["capex"])
    with LocalIndex(tmp_path / "i", collection=COLLECTION, dim=SPEC.dim) as again:
        assert again.doc_ids() == {"a"}
        assert again.store.count(COLLECTION) == 1


def test_sqlite_is_in_wal_mode(index):
    connection = sqlite3.connect(index.root / "fts.sqlite3")
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()


def test_second_open_of_locked_folder_raises_index_locked(tmp_path):
    first = LocalIndex(tmp_path / "i", collection=COLLECTION, dim=SPEC.dim)
    try:
        with pytest.raises(IndexLocked) as caught:
            LocalIndex(tmp_path / "i", collection=COLLECTION, dim=SPEC.dim)
        assert isinstance(caught.value, StoreError)
        with pytest.raises(IndexLocked):
            QdrantStore.local(tmp_path / "i" / "qdrant")
    finally:
        first.close()
    LocalIndex(tmp_path / "i", collection=COLLECTION, dim=SPEC.dim).close()  # released by close()


def test_store_local_opens_a_folder(tmp_path):
    store = QdrantStore.local(tmp_path / "q")
    try:
        store.ensure_collection("c", 4)
        assert store.count("c") == 0
    finally:
        store.client.close()


# --- names and FTS query --------------------------------------------------------------------------


def test_collection_name_is_stable_safe_and_distinct():
    name = collection_name("intfloat/multilingual-e5-small", "abc123", "v2", "cpu")
    assert name == collection_name("intfloat/multilingual-e5-small", "abc123", "v2", "cpu")
    assert all(c.isalnum() or c in "_-" for c in name)
    others = {
        collection_name("intfloat/multilingual-e5-small", "abc124", "v2", "cpu"),
        collection_name("intfloat/multilingual-e5-small", "abc123", "v1", "cpu"),
        collection_name("intfloat/multilingual-e5-small", "abc123", "v2", "coreml"),
        collection_name("intfloat/multilingual-e5-base", "abc123", "v2", "cpu"),
    }
    assert name not in others and len(others) == 4
    with pytest.raises(ValueError):
        collection_name("", "r", "v2", "cpu")


@pytest.mark.parametrize("text", ["AND", "capex*", "-debt", "( capex", "🚀 capex", 'a "b" OR c NEAR(d)'])
def test_fts_query_neutralises_operators(text):
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    connection.execute("INSERT INTO t VALUES ('capex and debt')")
    match = fts_query(text)
    connection.execute("SELECT * FROM t WHERE t MATCH ?", (match,)).fetchall()  # no syntax error
    assert match.startswith('"') and match.endswith('"')


def test_fts_query_quotes_and_ors_words():
    assert fts_query("capex* -debt") == '"capex" OR "debt"'
    assert fts_query("AND") == '"AND"'
    assert fts_query("🚀 capex") == '"capex"'


# A lone '"' or '(' holds no word either: refused like '"()' (the brief listed them as valid).
@pytest.mark.parametrize("text", ["  ", '"()', "🚀", "", '"', "("])
def test_fts_query_without_words_is_refused(text):
    with pytest.raises(ValueError):
        fts_query(text)


# --- fusion and filters ---------------------------------------------------------------------------


def test_rrf_fusion_order(index, embedder):
    query = embedder.encode_query("capex")
    rng = np.random.default_rng(0)
    other = rng.normal(size=SPEC.dim)
    other -= other.dot(query) * query  # orthogonal to the query
    other = _unit(other)
    vectors = {"c": _unit(query), "b": _unit(query + 0.5 * other), "a": _unit(query + 1.0 * other)}
    texts = {"a": "capex rose sharply", "b": "revenue was flat", "c": "margins improved"}
    for doc_id in ("a", "b", "c"):
        index.add(doc_id, [Chunk(texts[doc_id], page=None)], vectors[doc_id][None, :], {}, first_index=0)
    hits = hybrid_search("capex", index=index, embedder=embedder, filters=None, limit=3, rrf_k=60)
    order = [hit.doc_id for hit in hits]
    assert order == ["a", "c", "b"]  # a: 1st by FTS and 3rd dense; b: 2nd dense only
    scores = {hit.doc_id: hit.score for hit in hits}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 61)


def test_fts_side_ranks_by_bm25(index, embedder):
    _add(index, embedder, "x", ["alpha alpha alpha beta"])
    _add(index, embedder, "y", ["beta beta beta alpha"])
    assert index._fts_rows(fts_query("alpha"), None, 2)[0].doc_id == "x"
    assert index._fts_rows(fts_query("beta"), None, 2)[0].doc_id == "y"


def test_fts_only_hits_carry_their_payload(index, embedder):
    _add(index, embedder, "a", ["capex rose"], source_code="sec")
    _add(index, embedder, "b", ["revenue"], source_code="sec")
    hits = hybrid_search("capex", index=index, embedder=embedder, filters=None, limit=2, candidates=1)
    # candidates=1: the one dense hit and the one FTS hit may differ; both come back whole.
    by_doc = {hit.doc_id: hit for hit in hits}
    assert by_doc["a"].text == "capex rose" and by_doc["a"].source_code == "sec"
    assert by_doc["a"].point_id == point_id("a", 0, 0)


def test_filters_apply_to_both_sides(index, embedder):
    _add(index, embedder, "a", ["capex in a"], doc_type="10-K")
    _add(index, embedder, "b", ["capex in b"], doc_type="10-Q")
    only_a = SearchFilters(doc_ids=["a"])
    assert {row.doc_id for row in index._fts_rows(fts_query("capex"), only_a, 10)} == {"a"}
    hits = hybrid_search("capex", index=index, embedder=embedder, filters=only_a, limit=10)
    assert {hit.doc_id for hit in hits} == {"a"}
    tenq = SearchFilters(doc_types=["10-Q"])
    hits = hybrid_search("capex", index=index, embedder=embedder, filters=tenq, limit=10)
    assert {hit.doc_id for hit in hits} == {"b"}


def _spy():
    seen = {}

    def selector(hits, vectors, limit):
        seen["hits"], seen["vectors"], seen["limit"] = hits, vectors, limit
        return hits[::-1][:limit]

    return seen, selector


def test_selector_receives_one_stored_vector_per_candidate_in_order(index, embedder):
    texts = ["capex one", "capex capex two", "capex three", "capex four"]
    _add(index, embedder, "a", texts)
    stored = dict(zip(texts, embedder.encode_passage(texts)))
    # candidates=1: the dense side gives "capex four", FTS gives "capex capex two" (FTS only).
    dense = search(
        "capex",
        collection=COLLECTION,
        store=index.store,
        embedder=embedder,
        filters=None,
        limit=1,
        date_filter_mode="interval",
    )
    fts = index._fts_rows(fts_query("capex"), None, 1)
    assert dense[0].point_id != fts[0].chunk_id
    seen, selector = _spy()
    calls_before = len(embedder.calls)
    hits = hybrid_search(
        "capex", index=index, embedder=embedder, filters=None, limit=1, candidates=1, selector=selector
    )
    assert len(embedder.calls) == calls_before + 1  # the query is embedded once
    assert [hit.point_id for hit in seen["hits"]] == [dense[0].point_id, fts[0].chunk_id]
    assert seen["limit"] == 1
    assert seen["vectors"].shape == (2, SPEC.dim)
    for hit, vector in zip(seen["hits"], seen["vectors"]):
        np.testing.assert_allclose(vector, stored[hit.text], atol=1e-6)
    assert hits == seen["hits"][::-1][:1]


def test_selector_input_leaves_out_a_candidate_without_its_point(index, embedder, caplog):
    _add(index, embedder, "a", ["capex one", "capex two", "capex three"])
    gone = index._fts_rows(fts_query("two"), None, 1)[0].chunk_id
    # A write cut half-way: the SQLite row is there, the point is not.
    index.store.client.delete(
        collection_name=COLLECTION, points_selector=models.PointIdsList(points=[gone]), wait=True
    )
    plain = hybrid_search("capex two", index=index, embedder=embedder, filters=None, limit=5)
    assert gone in {hit.point_id for hit in plain}  # found by FTS
    seen, selector = _spy()
    with caplog.at_level("WARNING", logger="rag_engine.local"):
        hits = hybrid_search(
            "capex two", index=index, embedder=embedder, filters=None, limit=2, selector=selector
        )
    assert gone not in {hit.point_id for hit in seen["hits"]}
    assert gone not in {hit.point_id for hit in hits}
    assert seen["vectors"].shape == (len(seen["hits"]), SPEC.dim) == (2, SPEC.dim)
    assert f"1 of 3 candidates have no point in {COLLECTION}" in caplog.text
    assert f"[{gone}]" in caplog.text


def test_selector_is_not_called_when_limit_holds_every_candidate(index, embedder, monkeypatch):
    _add(index, embedder, "a", ["capex one", "capex two", "capex three"])

    def selector(hits, vectors, limit):
        raise AssertionError("not called")

    def no_retrieve(*args, **kwargs):
        raise AssertionError("no vector is read")

    monkeypatch.setattr(index.store.client, "retrieve", no_retrieve)
    plain = hybrid_search("capex", index=index, embedder=embedder, filters=None, limit=3)
    hits = hybrid_search("capex", index=index, embedder=embedder, filters=None, limit=3, selector=selector)
    assert hits == plain and len(hits) == 3


def test_hybrid_search_with_select_diverse_returns_limit_hits(index, embedder):
    pytest.importorskip("eigenmind")
    from rag_engine.select import select_diverse

    _add(index, embedder, "a", [f"capex note {i}" for i in range(8)])
    _add(index, embedder, "b", [f"capex memo {i}" for i in range(8)])
    arguments = {"index": index, "embedder": embedder, "filters": None, "limit": 4}
    arguments["selector"] = select_diverse
    first = hybrid_search("capex", **arguments)
    again = hybrid_search("capex", **arguments)
    assert len(first) == 4
    assert [hit.point_id for hit in first] == [hit.point_id for hit in again]
    scores = [hit.score for hit in first]
    assert scores == sorted(scores, reverse=True)  # fused order, best first


def test_a_query_without_words_searches_the_dense_side_alone(index, embedder):
    _add(index, embedder, "a", ["capex"])
    hits = hybrid_search("???", index=index, embedder=embedder, filters=None, limit=1)
    assert len(hits) == 1 and hits[0].score == pytest.approx(1 / 61)


@pytest.mark.parametrize(
    "kwargs",
    [{"limit": 0}, {"candidates": 0}, {"rrf_k": 0}, {"limit": True}, {"rrf_k": 1.5}],
)
def test_hybrid_search_refuses_bad_numbers(index, embedder, kwargs):
    arguments = {"limit": 5, **kwargs}
    with pytest.raises(SearchError):
        hybrid_search("capex", index=index, embedder=embedder, filters=None, **arguments)


def test_hybrid_search_refuses_a_blank_query(index, embedder):
    with pytest.raises(SearchError):
        hybrid_search("   ", index=index, embedder=embedder, filters=None, limit=5)


# --- date parity with the dense side --------------------------------------------------------------


@pytest.fixture(scope="module")
def dated_index(tmp_path_factory):
    embedder = FakeEmbedder(SPEC)
    opened = LocalIndex(tmp_path_factory.mktemp("dated"), collection=COLLECTION, dim=SPEC.dim)
    for document in DATE_DOCUMENTS:
        chunks = [Chunk(f"the text of {document.doc_id}", page=0)]
        vectors = embedder.encode_passage([chunk.embedded_text for chunk in chunks])
        opened.add(document.doc_id, chunks, vectors, document.metadata, first_index=0)
    yield opened, embedder
    opened.close()


@pytest.mark.parametrize("include_undated", [False, True])
def test_date_bounds_match_the_oracle_on_both_sides(dated_index, include_undated):
    index, embedder = dated_index
    deep = len(DATE_DOCUMENTS) + 10
    for date_from, date_to in all_ranges(40, seed=20261125):
        filters = SearchFilters(date_from=date_from, date_to=date_to, include_undated=include_undated)
        expected = expected_documents("v2", "interval", date_from, date_to, include_undated)
        fts = {row.doc_id for row in index._fts_rows(fts_query("text"), filters, deep)}
        assert fts == expected, (date_from, date_to)
        hits = hybrid_search(
            "the text", index=index, embedder=embedder, filters=filters, limit=deep, candidates=deep
        )
        assert {hit.doc_id for hit in hits} == expected, (date_from, date_to)


def test_other_list_filters_match_on_the_fts_side(dated_index):
    index, _embedder = dated_index
    filters = SearchFilters(source_codes=["day"], corpus="dates", date_to=datetime.date(2016, 3, 10))
    rows = index._fts_rows(fts_query("text"), filters, 500)
    assert {row.doc_id for row in rows} == {"day-stated", "day-no-key", "day-leap"}


def test_an_empty_batch_writes_nothing_and_close_twice_is_harmless(tmp_path):
    opened = LocalIndex(tmp_path / "i", collection=COLLECTION, dim=SPEC.dim)
    assert opened.add("a", [], np.empty((0,)), {}, first_index=0) == 0
    assert opened.doc_ids() == set()
    opened.close()
    opened.close()


# --- threads, locking and partial failures (fix round 1) ------------------------------------------


def test_add_from_another_thread_works_where_sqlite_checks_the_thread(tmp_path, embedder, monkeypatch):
    # macOS builds SQLite THREADSAFE=2, and qdrant-client then opens its storage with
    # check_same_thread=True. Simulate that on any platform: the flag must override it.
    from qdrant_client.local.persistence import CollectionPersistence

    monkeypatch.setattr(CollectionPersistence, "CHECK_SAME_THREAD", True)
    with LocalIndex(tmp_path / "i", collection=COLLECTION, dim=SPEC.dim) as opened:
        assert opened.store.client._client.force_disable_check_same_thread is True
        assert CollectionPersistence.CHECK_SAME_THREAD is False
        errors: list[BaseException] = []

        def work():
            try:
                _add(opened, embedder, "a", ["capex from a worker"])
                opened.delete("a")
                _add(opened, embedder, "b", ["capex again"])
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        worker = threading.Thread(target=work)
        worker.start()
        worker.join()
        assert errors == []
        assert opened.doc_ids() == {"b"}


def test_hybrid_search_embeds_first_then_searches_both_sides_under_the_index_lock(index, embedder):
    _add(index, embedder, "a", ["capex"])
    calls = len(embedder.calls)
    done = threading.Event()
    result: list = []

    def run():
        result.extend(hybrid_search("capex", index=index, embedder=embedder, filters=None, limit=5))
        done.set()

    with index._lock:
        worker = threading.Thread(target=run)
        worker.start()
        assert not done.wait(0.5)  # blocked on the lock...
        assert len(embedder.calls) == calls + 1  # ...after embedding the query
    worker.join()
    assert [hit.doc_id for hit in result] == ["a"]


def test_a_large_doc_ids_filter_needs_one_sql_variable(index, embedder):
    _add(index, embedder, "a", ["capex"])
    _add(index, embedder, "b", ["capex"])
    many = SearchFilters(doc_ids=[f"doc-{n}" for n in range(40_000)] + ["a"])
    assert {row.doc_id for row in index._fts_rows(fts_query("capex"), many, 10)} == {"a"}
    hits = hybrid_search("capex", index=index, embedder=embedder, filters=many, limit=10)
    assert {hit.doc_id for hit in hits} == {"a"}


class _FailingDb:
    """A SQLite connection whose ``executemany`` fails; everything else is the real one."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def executemany(self, *args):
        raise sqlite3.OperationalError("disk I/O error")


def test_a_failed_sqlite_write_is_a_store_error_and_leaves_no_points(index, embedder, monkeypatch):
    monkeypatch.setattr(index, "_db", _FailingDb(index._db))
    with pytest.raises(StoreError, match="disk I/O error"):
        _add(index, embedder, "a", ["capex", "debt"])
    monkeypatch.undo()
    assert index.store.count(COLLECTION) == 0
    assert index.doc_ids() == set()


def test_a_payload_that_is_not_json_writes_nothing(index):
    vectors = np.ones((1, SPEC.dim), np.float32)
    with pytest.raises((TypeError, ValueError)):
        index.add("a", [Chunk("x", page=None)], vectors, {"tags": {"set"}}, first_index=0)
    with pytest.raises(ValueError):
        index.add("a", [Chunk("x", page=None)], vectors, {"score": float("nan")}, first_index=0)
    assert index.store.count(COLLECTION) == 0


def test_a_failed_qdrant_delete_leaves_nothing_served_and_a_retry_cleans_up(index, embedder, monkeypatch):
    _add(index, embedder, "a", ["capex"])
    _add(index, embedder, "b", ["capex"])

    def broken(**_kwargs):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(index.store.client, "delete", broken)
    with pytest.raises(RuntimeError):
        index.delete("a")
    monkeypatch.undo()
    assert index.doc_ids() == {"b"}
    assert index.store.count(COLLECTION) == 2  # the point of "a" is left behind...
    hits = hybrid_search("capex", index=index, embedder=embedder, filters=None, limit=10)
    assert {hit.doc_id for hit in hits} == {"b"}  # ...and not served
    index.delete("a")
    assert index.store.count(COLLECTION) == 1


def test_sqlite_errors_are_store_errors(tmp_path):
    opened = LocalIndex(tmp_path / "i", collection=COLLECTION, dim=SPEC.dim)
    opened.close()
    with pytest.raises(StoreError):
        opened.doc_ids()


def test_the_dense_search_runs_under_the_index_lock(index, embedder, monkeypatch):
    import rag_engine.local as local_module

    _add(index, embedder, "a", ["capex"])
    owned: list[bool] = []
    real = local_module.search

    def spy(*args, **kwargs):
        owned.append(index._lock._is_owned())
        return real(*args, **kwargs)

    monkeypatch.setattr(local_module, "search", spy)
    hybrid_search("capex", index=index, embedder=embedder, filters=None, limit=5)
    assert owned == [True]


# --- final review: query size, unencodable text ----------------------------------------------------


def test_fts_query_keeps_the_first_max_terms_distinct_words():
    assert fts_query("a b a c d", max_terms=3) == '"a" OR "b" OR "c"'
    assert fts_query(" ".join(f"w{i}" for i in range(100))).count(" OR ") == DEFAULT_MAX_TERMS - 1
    for bad in (0, -1, True, 2.0, None):
        with pytest.raises(ValueError, match="max_terms"):
            fts_query("capex", max_terms=bad)


def test_the_default_max_terms_is_a_documented_parameter():
    import inspect

    assert DEFAULT_MAX_TERMS == 64
    assert inspect.signature(fts_query).parameters["max_terms"].default == DEFAULT_MAX_TERMS
    assert inspect.signature(hybrid_search).parameters["max_terms"].default == DEFAULT_MAX_TERMS


def test_a_200k_word_query_is_fast(index, embedder):
    import time

    _add(index, embedder, "a", ["capital expenditure rose", "revenue fell"])
    query = "capital " + " ".join(f"word{i}" for i in range(200_000))
    started = time.perf_counter()
    match = fts_query(query)
    assert match.count(" OR ") == DEFAULT_MAX_TERMS - 1 and match.startswith('"capital"')
    hits = hybrid_search(query, index=index, embedder=embedder, filters=None, limit=5)
    elapsed = time.perf_counter() - started
    assert {hit.doc_id for hit in hits} == {"a"}
    assert elapsed < 5.0, elapsed


def test_hybrid_search_passes_max_terms_to_the_fts_side(index, embedder, monkeypatch):
    seen = []
    original = index._fts_rows
    monkeypatch.setattr(index, "_fts_rows", lambda match, *a: seen.append(match) or original(match, *a))
    hybrid_search("alpha beta gamma", index=index, embedder=embedder, filters=None, limit=5, max_terms=2)
    assert seen == ['"alpha" OR "beta"']
    with pytest.raises(SearchError, match="max_terms"):
        hybrid_search("alpha", index=index, embedder=embedder, filters=None, limit=5, max_terms=0)


@pytest.mark.parametrize(
    "where",
    ["text", "context", "payload_value", "payload_key", "doc_id"],
)
def test_a_lone_surrogate_is_a_store_error_before_any_write(index, embedder, where):
    bad = "capex \ud800"
    chunk = Chunk(bad if where == "text" else "capex", page=None, context=bad if where == "context" else None)
    payload = {"title": bad} if where == "payload_value" else {bad: "x"} if where == "payload_key" else {}
    doc_id = bad if where == "doc_id" else "a"
    vectors = np.ones((1, SPEC.dim), np.float32)
    with pytest.raises(StoreError, match="UTF-8"):
        index.add(doc_id, [chunk], vectors, payload, first_index=0)
    assert index.store.count(COLLECTION) == 0
    assert index.doc_ids() == set()
