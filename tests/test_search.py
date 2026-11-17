"""Search: the filters, the hits, the checks.

Run against qdrant-client's local mode (``QdrantClient(":memory:")``) with a ``FakeEmbedder``,
on the corpus of ``search_corpus.py``. The same matrix runs against a real server in
``tests/integration/test_search_integration.py``.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
import types
import unicodedata

import numpy as np
import pytest
from qdrant_client import QdrantClient, models
from search_corpus import (
    ALL_DOCUMENTS,
    SEARCH_CASES,
    SEARCH_POINT_COUNT,
    SEARCH_SPEC,
    SEARCH_WHEN,
    search_chunk_text,
    search_points,
)

from rag_engine.config import Settings
from rag_engine.embedding import FakeEmbedder
from rag_engine.errors import RagEngineError
from rag_engine.schema import point_id
from rag_engine.search import (
    QUERY_NORMALIZATION,
    Hit,
    SearchError,
    SearchFilters,
    _qdrant_filter,
    is_searchable,
    search,
)
from rag_engine.store import QdrantAuthError, QdrantStore

COLLECTION = "search-test"
QUERY = "what did the governing council decide?"


@pytest.fixture(scope="module")
def store():
    """The corpus in a local-mode collection, shared by the tests of this module (read only)."""
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client, settings=Settings(batch_size=4))
    store.ensure_collection(COLLECTION, SEARCH_SPEC.dim)
    assert store.upsert(COLLECTION, search_points()) == SEARCH_POINT_COUNT
    yield store
    client.close()


@pytest.fixture
def embedder():
    return FakeEmbedder(SEARCH_SPEC)


def run(store, embedder, query=QUERY, **kwargs):
    kwargs.setdefault("limit", 50)
    return search(query, collection=COLLECTION, store=store, embedder=embedder, **kwargs)


SCORE_TOLERANCE = 1e-6


def assert_same_hits(first: list[Hit], second: list[Hit]) -> None:
    """The same chunks in the same order, with the same fields; the scores within rounding noise.

    Two identical searches are not compared with ``==``: in qdrant-client's local mode the
    first query of a fresh in-memory collection gives scores about 1e-8 away from those of
    the later ones, so an exact comparison depends on which test ran first.
    """
    assert [hit.point_id for hit in first] == [hit.point_id for hit in second]
    for one, other in zip(first, second, strict=True):
        assert dataclasses.replace(one, score=0.0) == dataclasses.replace(other, score=0.0)
        assert one.score == pytest.approx(other.score, abs=SCORE_TOLERANCE)


def expected_ranking(query: str = QUERY) -> list[tuple[int, float]]:
    """``(point id, cosine)`` of every chunk of the corpus for ``query``, best first."""
    reference = FakeEmbedder(SEARCH_SPEC)
    vector = reference.encode_query(query).astype(np.float64)
    points = search_points(reference)
    scores = [float(np.dot(vector, np.asarray(point.vector, dtype=np.float64))) for point in points]
    return sorted(zip([point.id for point in points], scores), key=lambda pair: -pair[1])


# --------------------------------------------------------------------------------------
# The filter matrix
# --------------------------------------------------------------------------------------


def test_the_filters_select_the_expected_chunks_in_the_order_of_the_scores(store, embedder, search_case):
    hits = run(store, embedder, filters=SearchFilters(**search_case.filters))

    assert {hit.doc_id for hit in hits} == search_case.documents
    # Every chunk of the selected documents, in the order of an unfiltered ranking.
    document_of = {point.id: point.payload["doc_id"] for point in search_points()}
    ranking = [point for point, _score in expected_ranking() if document_of[point] in search_case.documents]
    assert [hit.point_id for hit in hits] == ranking


def test_the_matrix_names_real_documents_and_has_cases_without_any_result():
    assert len({case.label for case in SEARCH_CASES}) == len(SEARCH_CASES)
    assert all(case.documents <= ALL_DOCUMENTS for case in SEARCH_CASES)
    assert any(not case.documents for case in SEARCH_CASES)
    assert any(case.documents == ALL_DOCUMENTS for case in SEARCH_CASES)


def test_no_filters_and_empty_filters_are_the_same_search(store, embedder):
    without = run(store, embedder)
    assert_same_hits(run(store, embedder, filters=None), without)
    assert_same_hits(run(store, embedder, filters=SearchFilters()), without)
    assert len(without) == SEARCH_POINT_COUNT


def test_a_document_without_a_full_date_is_excluded_once_a_date_bound_is_set(store, embedder):
    undated = {"boj-a", "boj-b", "boe-a"}  # no date key, a null date, "2016-03"

    assert undated <= {hit.doc_id for hit in run(store, embedder)}
    for bounds in ({"date_from": "0001-01-01"}, {"date_to": "9999-12-31"}, {"date_from": "1900-01-01"}):
        found = {hit.doc_id for hit in run(store, embedder, filters=SearchFilters(**bounds))}
        assert found == ALL_DOCUMENTS - undated, bounds


def test_the_upper_date_bound_includes_the_whole_day(store, embedder):
    # ecb-a is stored as "2016-03-10", ecb-b as "2016-03-10T15:30:00Z".
    found = {hit.doc_id for hit in run(store, embedder, filters=SearchFilters(date_to="2016-03-10"))}
    assert {"ecb-a", "ecb-b"} <= found
    assert "fed-b" not in found  # 2016-03-11


def test_the_lower_date_bound_includes_its_day(store, embedder):
    found = {hit.doc_id for hit in run(store, embedder, filters=SearchFilters(date_from="2016-03-10"))}
    assert {"ecb-a", "ecb-b"} <= found
    assert "fed-a" not in found  # 2016-03-09


# --------------------------------------------------------------------------------------
# The Qdrant filter that is built
# --------------------------------------------------------------------------------------


def test_no_filter_is_built_when_nothing_restricts():
    assert _qdrant_filter(None) is None
    assert _qdrant_filter(SearchFilters()) is None


def test_date_bounds_are_sent_as_utc_datetimes_with_an_exclusive_next_day():
    built = _qdrant_filter(SearchFilters(date_from="2016-03-10", date_to="2016-03-10"))
    (condition,) = built.must
    utc = datetime.timezone.utc

    assert condition.key == "date"
    assert isinstance(condition.range, models.DatetimeRange)
    assert condition.range.gte == datetime.datetime(2016, 3, 10, tzinfo=utc)
    assert condition.range.lt == datetime.datetime(2016, 3, 11, tzinfo=utc)
    assert condition.range.lte is None and condition.range.gt is None


def test_the_upper_bound_crosses_the_end_of_a_month_and_of_a_leap_february():
    utc = datetime.timezone.utc
    for day, following in (("2016-02-29", (2016, 3, 1)), ("2015-12-31", (2016, 1, 1))):
        (condition,) = _qdrant_filter(SearchFilters(date_to=day)).must
        assert condition.range.lt == datetime.datetime(*following, tzinfo=utc)
        assert condition.range.gte is None


def test_the_last_day_of_the_calendar_still_requires_a_date():
    (condition,) = _qdrant_filter(SearchFilters(date_to=datetime.date.max)).must
    assert condition.key == "date"
    assert condition.range.lt is None and condition.range.gte is not None


def test_lists_become_match_any_and_fields_become_separate_conditions():
    built = _qdrant_filter(SearchFilters(source_codes=["ecb", "fed"], doc_ids=["a"], corpus="central-bank"))

    assert built.should is None and built.must_not is None
    by_key = {condition.key: condition.match for condition in built.must}
    assert by_key == {
        "corpus": models.MatchValue(value="central-bank"),
        "source_code": models.MatchAny(any=["ecb", "fed"]),
        "doc_id": models.MatchAny(any=["a"]),
    }


# --------------------------------------------------------------------------------------
# SearchFilters
# --------------------------------------------------------------------------------------


def test_filters_are_stored_in_one_form_whatever_was_given():
    given = ["fed", "ecb", "fed"]
    filters = SearchFilters(
        source_codes=given,
        doc_types=("minutes",),
        languages={"fr", "en"},
        doc_ids=frozenset({"b", "a"}),
        date_from="2016-03-10",
        date_to=datetime.date(2016, 3, 11),
        corpus="central-bank",
    )

    assert filters.source_codes == ("fed", "ecb")  # order kept, duplicate dropped
    assert filters.doc_types == ("minutes",)
    assert filters.languages == ("en", "fr")  # a set has no order: sorted
    assert filters.doc_ids == ("a", "b")
    assert filters.date_from == datetime.date(2016, 3, 10)
    assert filters.date_to == datetime.date(2016, 3, 11)
    assert given == ["fed", "ecb", "fed"]
    given.append("boj")
    assert filters.source_codes == ("fed", "ecb")
    assert filters == SearchFilters(**dataclasses.asdict(filters))
    assert hash(filters) == hash(SearchFilters(**dataclasses.asdict(filters)))


# --- from_mapping: the filters of a JSON object -----------------------------------------


def test_from_mapping_builds_the_filters_the_constructor_builds():
    given = {
        "source_codes": ["ecb", "fed"],
        "doc_types": ["minutes"],
        "languages": ["en"],
        "doc_ids": ["a", "b"],
        "date_from": "2016-01-01",
        "date_to": "2016-03-10",
        "corpus": "central-bank",
    }
    filters = SearchFilters.from_mapping(json.loads(json.dumps(given)))
    assert isinstance(filters, SearchFilters)
    assert filters == SearchFilters(**given)
    assert filters.date_to == datetime.date(2016, 3, 10) and filters.source_codes == ("ecb", "fed")
    assert given["source_codes"] == ["ecb", "fed"]  # the mapping is not changed


def test_from_mapping_takes_any_mapping_and_does_not_keep_it():
    given = types.MappingProxyType({"corpus": "press", "doc_ids": ["a"]})
    assert SearchFilters.from_mapping(given) == SearchFilters(corpus="press", doc_ids=["a"])


@pytest.mark.parametrize(
    "given",
    [{}, {"source_codes": None}, {name: None for name in ("source_codes", "date_from", "date_to", "corpus")}],
    ids=["empty", "one-null", "nulls"],
)
def test_from_mapping_reads_a_missing_key_and_a_null_as_not_set(given):
    assert SearchFilters.from_mapping(given) == SearchFilters()
    assert _qdrant_filter(SearchFilters.from_mapping(given)) is None


def test_from_mapping_mixes_nulls_and_values():
    filters = SearchFilters.from_mapping({"source_codes": None, "doc_types": ["A1"], "date_to": None})
    assert filters == SearchFilters(doc_types=["A1"])


ALLOWED = "source_codes, doc_types, languages, doc_ids, date_from, date_to, corpus"


@pytest.mark.parametrize(
    ("given", "named"),
    [
        ({"source": ["ecb"]}, "unknown filter 'source'"),
        ({"source_codes": ["ecb"], "year": 2016}, "unknown filter 'year'"),
        ({"Source_Codes": ["ecb"]}, "unknown filter 'Source_Codes'"),
        (
            {"year": 2016, "source": None, "corpus": "press", "bank": "ecb"},
            "unknown filters 'year', 'source', 'bank'",
        ),
        ({"": ["ecb"]}, "unknown filter ''"),
        ({3: ["ecb"]}, "unknown filter 3"),
        ({None: ["ecb"]}, "unknown filter None"),
    ],
    ids=["one", "next-to-a-known", "case", "several-in-order", "empty-name", "number", "none"],
)
def test_from_mapping_refuses_unknown_keys_naming_them_and_the_allowed_ones(given, named):
    with pytest.raises(SearchError) as raised:
        SearchFilters.from_mapping(given)
    message = str(raised.value)
    assert message == f"{named} (the filters are: {ALLOWED})"
    assert isinstance(raised.value, ValueError) and isinstance(raised.value, RagEngineError)


def test_the_constructor_raises_a_bare_type_error_for_an_unknown_key_and_from_mapping_does_not():
    given = {"source": ["ecb"]}
    with pytest.raises(TypeError):
        SearchFilters(**given)
    with pytest.raises(SearchError):
        SearchFilters.from_mapping(given)


@pytest.mark.parametrize(
    "given", [None, ["source_codes"], "source_codes", 3, [("source_codes", ["ecb"])], SearchFilters()]
)
def test_from_mapping_refuses_what_is_not_a_mapping(given):
    with pytest.raises(SearchError, match="filters must be a mapping of filter names to values"):
        SearchFilters.from_mapping(given)


@pytest.mark.parametrize(
    ("given", "match"),
    [
        ({"source_codes": "ecb"}, "source_codes must be a list of strings, got the single string"),
        ({"doc_types": []}, "doc_types is an empty list"),
        ({"languages": ["en", ""]}, "languages must hold non-blank strings"),
        ({"doc_ids": {"a": 1}}, "doc_ids must be a list of strings or None, got dict"),
        ({"date_from": "2016"}, "date_from must be a datetime.date or a string written YYYY-MM-DD"),
        ({"date_to": "2016-02-30"}, "date_to is not a date of the calendar"),
        ({"date_from": 20160310}, "date_from must be"),
        ({"date_from": "2016-03-02", "date_to": "2016-03-01"}, "is after date_to"),
        ({"corpus": ""}, "corpus must be a non-blank string"),
        ({"corpus": ["press"]}, "corpus must be a non-blank string"),
        ({"corpus": "\ud800"}, "corpus is a string that cannot be encoded"),
        ({"source_codes": ["\ud800"]}, "source_codes holds a string that cannot be encoded"),
    ],
)
def test_from_mapping_applies_the_checks_of_the_constructor(given, match):
    with pytest.raises(SearchError, match=match) as raised:
        SearchFilters.from_mapping(given)
    with pytest.raises(SearchError) as direct:
        SearchFilters(**given)
    assert str(raised.value) == str(direct.value)


def test_from_mapping_names_the_unknown_keys_before_it_checks_the_values():
    with pytest.raises(SearchError, match="unknown filter 'bank'"):
        SearchFilters.from_mapping({"source_codes": "ecb", "bank": "ecb"})


def test_from_mapping_filters_search_like_the_others(store, embedder):
    given = json.loads('{"source_codes": ["ecb"], "date_from": "2016-03-10", "languages": null}')
    hits = run(store, embedder, filters=SearchFilters.from_mapping(given))
    assert_same_hits(
        hits, run(store, embedder, filters=SearchFilters(source_codes=["ecb"], date_from="2016-03-10"))
    )
    assert hits and {hit.source_code for hit in hits} == {"ecb"}


def test_filters_are_frozen():
    filters = SearchFilters(source_codes=["ecb"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        filters.source_codes = ("fed",)  # type: ignore[misc]


def test_every_field_defaults_to_no_restriction():
    assert all(value is None for value in dataclasses.asdict(SearchFilters()).values())
    assert [f.name for f in dataclasses.fields(SearchFilters)] == [
        "source_codes",
        "doc_types",
        "languages",
        "doc_ids",
        "date_from",
        "date_to",
        "corpus",
    ]


def test_values_are_taken_as_they_are():
    assert SearchFilters(source_codes=[" ecb ", "ECB"]).source_codes == (" ecb ", "ECB")


LIST_FIELDS = ["source_codes", "doc_types", "languages", "doc_ids"]


@pytest.mark.parametrize("name", LIST_FIELDS)
def test_a_bare_string_is_refused_for_a_list(name):
    with pytest.raises(SearchError, match=rf"{name} must be a list of strings.*\['ecb'\]"):
        SearchFilters(**{name: "ecb"})
    with pytest.raises(SearchError, match=name):
        SearchFilters(**{name: b"ecb"})


@pytest.mark.parametrize("name", LIST_FIELDS)
@pytest.mark.parametrize("empty", [[], (), set(), frozenset()], ids=["list", "tuple", "set", "frozenset"])
def test_an_empty_list_is_refused(name, empty):
    with pytest.raises(SearchError, match=rf"{name} is an empty list.*use None"):
        SearchFilters(**{name: empty})


@pytest.mark.parametrize("name", LIST_FIELDS)
@pytest.mark.parametrize("bad", ["", "  ", "\t\n", None, 3, ["ecb"], b"ecb"], ids=repr)
def test_a_blank_or_non_string_value_is_refused(name, bad):
    with pytest.raises(SearchError, match=rf"{name} must hold non-blank strings"):
        SearchFilters(**{name: ["ecb", bad]})


@pytest.mark.parametrize("name", LIST_FIELDS)
@pytest.mark.parametrize(
    "bad", [3, 1.5, True, {"ecb": 1}, iter(["ecb"]), object()], ids=lambda v: type(v).__name__
)
def test_anything_but_a_list_a_tuple_or_a_set_is_refused(name, bad):
    with pytest.raises(SearchError, match=rf"{name} must be a list of strings or None"):
        SearchFilters(**{name: bad})


BAD_DATES = [
    "2016",
    "2016-03",
    "10/03/2016",
    "2016/03/10",
    "20160310",
    "2016-3-10",
    "16-03-10",
    "2016-03-10T00:00:00",
    "2016-03-10T00:00:00Z",
    "2016-03-10 ",
    " 2016-03-10",
    "2016-03-10\n",
    "２０１６-03-10",  # full-width digits
    "2016-02-30",
    "2016-13-01",
    "0000-01-01",
    "",
    "today",
    2016,
    20160310,
    1457568000.0,
    True,
    datetime.datetime(2016, 3, 10),
    datetime.datetime(2016, 3, 10, tzinfo=datetime.timezone.utc),
    ["2016-03-10"],
]


@pytest.mark.parametrize("name", ["date_from", "date_to"])
@pytest.mark.parametrize("bad", BAD_DATES, ids=repr)
def test_a_date_bound_that_is_not_a_full_date_is_refused(name, bad):
    with pytest.raises(SearchError, match=name) as raised:
        SearchFilters(**{name: bad})
    assert "YYYY-MM-DD" in str(raised.value) or "not a date of the calendar" in str(raised.value)


def test_a_day_that_does_not_exist_is_named_as_such():
    with pytest.raises(SearchError, match=r"date_to is not a date of the calendar: '2016-02-30'"):
        SearchFilters(date_to="2016-02-30")


def test_date_from_after_date_to_is_refused():
    with pytest.raises(SearchError, match=r"date_from \(2016-03-11\) is after date_to \(2016-03-10\)"):
        SearchFilters(date_from="2016-03-11", date_to=datetime.date(2016, 3, 10))
    assert SearchFilters(date_from="2016-03-10", date_to="2016-03-10").date_from == datetime.date(2016, 3, 10)


@pytest.mark.parametrize("bad", ["", "   ", 3, ["press"], b"press"], ids=repr)
def test_a_blank_or_non_string_corpus_is_refused(bad):
    with pytest.raises(SearchError, match="corpus must be a non-blank string or None"):
        SearchFilters(corpus=bad)


# A lone surrogate is what json.loads('"\\ud800"') returns: a str that no UTF-8 encoder accepts.
LONE_SURROGATES = ["\ud800", "ec\udfffb", "\udc80 press"]


def test_json_can_deliver_a_lone_surrogate():
    assert json.loads('"\\ud800"') == "\ud800"
    with pytest.raises(UnicodeEncodeError):
        "\ud800".encode()


@pytest.mark.parametrize("name", LIST_FIELDS)
@pytest.mark.parametrize("bad", LONE_SURROGATES, ids=ascii)
def test_a_list_value_that_cannot_be_encoded_as_utf8_is_refused(name, bad):
    with pytest.raises(
        SearchError, match=rf"{name} holds a string that cannot be encoded as UTF-8"
    ) as raised:
        SearchFilters(**{name: ["ecb", bad]})
    assert ascii(bad) in str(raised.value)
    str(raised.value).encode()  # the message itself can be logged and sent
    with pytest.raises(SearchError, match=name):
        SearchFilters(**{name: {bad}})


@pytest.mark.parametrize("bad", LONE_SURROGATES, ids=ascii)
def test_a_corpus_that_cannot_be_encoded_as_utf8_is_refused(bad):
    with pytest.raises(SearchError, match="corpus is a string that cannot be encoded as UTF-8") as raised:
        SearchFilters(corpus=bad)
    assert ascii(bad) in str(raised.value)
    str(raised.value).encode()


def test_a_pair_of_surrogates_written_as_one_character_is_a_valid_value():
    # json.loads joins an escaped pair into the character itself: nothing to refuse.
    value = json.loads('"\\ud83d\\ude00"')
    assert value == "\U0001f600"
    assert SearchFilters(source_codes=[value], corpus=value).corpus == value


def test_search_error_is_an_engine_error_and_a_value_error():
    assert issubclass(SearchError, RagEngineError)
    assert issubclass(SearchError, ValueError)


# --------------------------------------------------------------------------------------
# Hits
# --------------------------------------------------------------------------------------


def test_a_hit_carries_every_field_of_its_payload(store, embedder):
    hits = run(store, embedder, filters=SearchFilters(doc_ids=["ecb-a"]))
    by_chunk = {hit.chunk_number: hit for hit in hits}
    assert sorted(by_chunk) == [0, 1]
    hit = by_chunk[1]

    assert hit.text == search_chunk_text("ecb-a", 1)
    assert hit.doc_id == "ecb-a"
    assert hit.title == "Monetary policy statement"
    assert hit.date == "2016-03-10"
    assert hit.source_code == "ecb"
    assert hit.doc_type == "statement"
    assert hit.language == "en"
    assert hit.page == 1
    assert hit.chunk_number == 1
    assert hit.corpus == "central-bank"
    assert hit.point_id == point_id("ecb-a", 1, 1)
    assert isinstance(hit.score, float)
    assert hit.payload == {
        "doc_id": "ecb-a",
        "corpus": "central-bank",
        "source_code": "ecb",
        "doc_type": "statement",
        "title": "Monetary policy statement",
        "date": "2016-03-10",
        "year": 2016,
        "language": "en",
        "filename": "ecb-a.pdf",
        "chunk_number": 1,
        "ingestion_date": SEARCH_WHEN,
        "text": search_chunk_text("ecb-a", 1),
        "page": 1,
    }


def test_page_is_none_for_an_unpaginated_document_and_missing_keys_are_none(store, embedder):
    (hit,) = run(store, embedder, filters=SearchFilters(doc_ids=["boj-a"]))

    assert hit.page is None and "page" not in hit.payload
    assert hit.title is None and hit.date is None
    assert (hit.doc_id, hit.source_code, hit.doc_type, hit.language, hit.corpus) == (
        "boj-a",
        "boj",
        "statement",
        "ja",
        "central-bank",
    )
    assert hit.chunk_number == 0
    assert hit.point_id == point_id("boj-a", 0, 0)


def test_page_zero_is_not_none_and_a_null_date_is_none(store, embedder):
    (hit,) = run(store, embedder, filters=SearchFilters(doc_ids=["boj-b"]))
    assert hit.page == 0 and hit.page is not None
    assert hit.date is None and hit.payload["date"] is None


def test_the_date_comes_back_as_it_was_stored(store, embedder):
    dates = {hit.doc_id: hit.date for hit in run(store, embedder)}
    assert dates["ecb-b"] == "2016-03-10T15:30:00Z"
    assert dates["boe-a"] == "2016-03"


def test_a_hit_is_frozen_and_owns_its_payload(store, embedder):
    first = run(store, embedder, limit=1)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.score = 0.0  # type: ignore[misc]
    first.payload["text"] = "changed by the caller"
    assert run(store, embedder, limit=1)[0].payload["text"] != "changed by the caller"


def test_to_dict_is_json_ready_and_leaves_the_payload_out_unless_asked(store, embedder):
    (hit,) = run(store, embedder, filters=SearchFilters(doc_ids=["boj-a"]))
    as_dict = hit.to_dict()

    assert list(as_dict) == [
        "text",
        "score",
        "doc_id",
        "title",
        "date",
        "source_code",
        "doc_type",
        "language",
        "page",
        "chunk_number",
        "corpus",
        "point_id",
    ]
    assert as_dict["page"] is None and as_dict["title"] is None
    assert as_dict["point_id"] == hit.point_id and as_dict["score"] == hit.score
    assert json.loads(json.dumps(as_dict)) == as_dict

    full = hit.to_dict(include_payload=True)
    assert full["payload"] == hit.payload and full["payload"] is not hit.payload
    assert {key: value for key, value in full.items() if key != "payload"} == as_dict
    assert json.loads(json.dumps(full)) == full


def test_hit_has_the_specified_fields_in_order():
    assert [f.name for f in dataclasses.fields(Hit)] == [
        "text",
        "score",
        "doc_id",
        "title",
        "date",
        "source_code",
        "doc_type",
        "language",
        "page",
        "chunk_number",
        "corpus",
        "point_id",
        "payload",
    ]


# --------------------------------------------------------------------------------------
# Ranking and limit
# --------------------------------------------------------------------------------------


def test_hits_come_best_first_with_the_cosine_as_score(store, embedder):
    hits = run(store, embedder)
    ranking = expected_ranking()

    assert [hit.point_id for hit in hits] == [point for point, _score in ranking]
    assert [hit.score for hit in hits] == pytest.approx([score for _point, score in ranking], abs=1e-5)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
    assert len({hit.score for hit in hits}) == len(hits)  # no tie hides a wrong order


def test_another_query_gives_another_ranking(store, embedder):
    other = "inflation outlook"
    hits = run(store, embedder, query=other)
    assert [hit.point_id for hit in hits] == [point for point, _score in expected_ranking(other)]
    assert [hit.point_id for hit in hits] != [hit.point_id for hit in run(store, embedder)]


def test_the_query_is_embedded_once_as_a_query(store, embedder):
    run(store, embedder)
    assert embedder.calls == [SEARCH_SPEC.query_prefix + QUERY]


@pytest.mark.parametrize("limit", [1, 3, SEARCH_POINT_COUNT, 50])
def test_limit_keeps_the_best_hits(store, embedder, limit):
    hits = run(store, embedder, limit=limit)
    best = [point for point, _score in expected_ranking()][:limit]
    assert [hit.point_id for hit in hits] == best


def test_limit_applies_after_the_filters(store, embedder):
    filters = SearchFilters(source_codes=["fed"])
    everything = run(store, embedder, filters=filters)
    assert len(everything) == 3
    assert run(store, embedder, filters=filters, limit=2) == everything[:2]


def test_the_default_limit_is_ten_and_the_default_maximum_fifty(store, embedder):
    assert Settings.search_limit == 10 and Settings.search_max_limit == 50
    hits = search(QUERY, collection=COLLECTION, store=store, embedder=embedder)
    assert len(hits) == 10 < SEARCH_POINT_COUNT
    assert len(run(store, embedder, limit=50)) == SEARCH_POINT_COUNT
    with pytest.raises(SearchError, match="limit 51 is above the maximum of 50"):
        run(store, embedder, limit=51)


def test_a_limit_above_the_maximum_is_refused_and_names_the_maximum(store, embedder):
    with pytest.raises(SearchError, match=r"limit 6 is above the maximum of 5: ask for at most 5 hits"):
        run(store, embedder, limit=6, max_limit=5)
    assert embedder.calls == []  # refused before anything is embedded
    assert len(run(store, embedder, limit=5, max_limit=5)) == 5
    assert len(run(store, embedder, limit=60, max_limit=100)) == SEARCH_POINT_COUNT


def test_the_default_limit_is_refused_too_under_a_lower_maximum(store, embedder):
    with pytest.raises(SearchError, match="limit 10 is above the maximum of 5"):
        search(QUERY, collection=COLLECTION, store=store, embedder=embedder, max_limit=5)


def test_the_limits_of_the_settings_are_passed_by_the_caller(store, embedder):
    settings = Settings(search_limit=2, search_max_limit=4)
    kwargs = {"collection": COLLECTION, "store": store, "embedder": embedder}

    assert len(search(QUERY, limit=settings.search_limit, max_limit=settings.search_max_limit, **kwargs)) == 2
    with pytest.raises(SearchError, match="maximum of 4"):
        search(QUERY, limit=5, max_limit=settings.search_max_limit, **kwargs)


def test_search_reads_no_environment_variable(store, embedder, monkeypatch):
    monkeypatch.setenv("RAG_ENGINE_SEARCH_LIMIT", "1")
    monkeypatch.setenv("RAG_ENGINE_SEARCH_MAX_LIMIT", "1")
    monkeypatch.setenv("QDRANT_PORT", "not a port")
    assert len(search(QUERY, collection=COLLECTION, store=store, embedder=embedder)) == 10


@pytest.mark.parametrize("name", ["limit", "max_limit"])
@pytest.mark.parametrize("bad", [0, -1, True, False, 2.0, "5", [5]], ids=repr)
def test_a_limit_that_is_not_a_positive_integer_is_refused(store, embedder, name, bad):
    with pytest.raises(SearchError, match=rf"{name} must be an integer >= 1"):
        run(store, embedder, **{name: bad})
    assert embedder.calls == []


# --------------------------------------------------------------------------------------
# Argument checks
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, 3, ["a query"], b"a query"], ids=repr)
def test_a_blank_or_non_string_query_is_refused(store, embedder, bad):
    with pytest.raises(SearchError, match="query must be a non-blank string"):
        run(store, embedder, query=bad)
    assert embedder.calls == []


@pytest.mark.parametrize("bad", ["", "  ", None, 3], ids=repr)
def test_a_blank_or_non_string_collection_is_refused(store, embedder, bad):
    with pytest.raises(SearchError, match="collection must be a non-blank string"):
        search(QUERY, collection=bad, store=store, embedder=embedder)
    assert embedder.calls == []


@pytest.mark.parametrize("bad", LONE_SURROGATES, ids=ascii)
def test_a_query_that_cannot_be_encoded_as_utf8_is_refused(store, embedder, bad):
    with pytest.raises(SearchError, match="query is a string that cannot be encoded as UTF-8") as raised:
        run(store, embedder, query="rates " + bad)
    assert ascii(bad)[1:-1] in str(raised.value)
    str(raised.value).encode()
    assert embedder.calls == []


@pytest.mark.parametrize("bad", LONE_SURROGATES, ids=ascii)
def test_a_collection_that_cannot_be_encoded_as_utf8_is_refused(store, embedder, bad):
    with pytest.raises(SearchError, match="collection is a string that cannot be encoded as UTF-8"):
        search(QUERY, collection=bad, store=store, embedder=embedder)
    assert embedder.calls == []


# Nothing to search for: combining marks alone, zero-width and other format characters,
# controls, separators, private-use characters and unassigned code points.
UNSEARCHABLE_QUERIES = [
    "\u0301\u0301",  # two combining acute accents
    "\u200b",  # zero width space
    "\u2060",  # word joiner
    "\ufeff",  # zero width no-break space (byte order mark)
    "\u200d\u200c",  # zero width joiner and non-joiner
    "\u00ad",  # soft hyphen
    "\u202e\u2066",  # bidirectional controls
    "\x00\x1b\x7f",  # control characters
    " \u0301 \u200b\n",  # the same between spaces
    "\u0301\u2028\u3000\u0301",  # marks around a line separator and an ideographic space
    "\ufe0f",  # variation selector
    "\ue000",  # private use
    "\U000e0001",  # language tag
]


@pytest.mark.parametrize("bad", UNSEARCHABLE_QUERIES, ids=ascii)
@pytest.mark.parametrize("normalize", [True, False])
def test_a_query_without_any_searchable_character_is_refused(store, embedder, bad, normalize):
    with pytest.raises(SearchError, match="the query has no searchable character") as raised:
        run(store, embedder, query=bad, normalize_query=normalize)
    assert ascii(bad) in str(raised.value)
    assert embedder.calls == []


# A letter, a number, a symbol or a punctuation mark is enough, in any script.
SEARCHABLE_QUERIES = [
    "a",
    "2016",
    "\u0663",  # an Arabic-Indic digit
    "\u00bd",  # one half
    "\u2167",  # a Roman numeral
    "\u20ac",  # the euro sign
    "$AAPL",
    "+",
    "\u00b0",  # the degree sign
    "\U0001f600",  # an emoji
    "???",
    "\u2026",  # an ellipsis
    "\u00bf",  # an inverted question mark
    "\u3002",  # an ideographic full stop
    "e\u0301",  # a letter and its combining accent
    "\u200bbce\u200b",  # letters between zero-width spaces
    "\ufeffinflation",  # a byte order mark left in front of a word
]


@pytest.mark.parametrize("query", SEARCHABLE_QUERIES, ids=ascii)
@pytest.mark.parametrize("normalize", [True, False])
def test_one_letter_number_symbol_or_punctuation_mark_makes_a_query(store, embedder, query, normalize):
    assert len(run(store, embedder, query=query, normalize_query=normalize)) == SEARCH_POINT_COUNT
    expected = unicodedata.normalize("NFKD", query) if normalize else query
    assert embedder.calls == [SEARCH_SPEC.query_prefix + expected]


def test_a_query_of_punctuation_alone_is_accepted(store, embedder):
    first = run(store, embedder, query="???", limit=3)
    assert_same_hits(first, run(store, embedder, query="???", limit=3))
    assert len(first) == 3
    assert embedder.calls == [SEARCH_SPEC.query_prefix + "???"] * 2


def searchable(text: str) -> bool:
    """The rule, written again: one character of general category L, N, S or P."""
    return any(unicodedata.category(character)[0] in "LNSP" for character in text)


# One query per script, as typed (NFC), and what a keyboard or a web form may add around it.
SCRIPT_QUERIES = {
    "latin": "d\u00e9cision de politique mon\u00e9taire",
    "greek": "\u03bd\u03bf\u03bc\u03b9\u03c3\u03bc\u03b1\u03c4\u03b9\u03ba\u03ae",
    "cyrillic": "\u043a\u043b\u044e\u0447\u0435\u0432\u0430\u044f \u0441\u0442\u0430\u0432\u043a\u0430",
    "arabic": "\u0633\u0639\u0631 \u0627\u0644\u0641\u0627\u0626\u062f\u0629",
    "hebrew": "\u05e8\u05d9\u05d1\u05d9\u05ea",
    "devanagari": "\u092c\u094d\u092f\u093e\u091c \u0926\u0930",
    "thai": "\u0e2d\u0e31\u0e15\u0e23\u0e32\u0e14\u0e2d\u0e01\u0e40\u0e1a\u0e35\u0e49\u0e22",
    "han": "\u8d27\u5e01\u653f\u7b56",
    "kana": "\u30c7\u30d5\u30ec\u8131\u5374",
    "halfwidth kana": "\uff83\uff9e\uff8c\uff9a",
    "hangul": "\ud55c\uad6d\uc740\ud589",
    "vietnamese": "l\u00e3i su\u1ea5t \u0111i\u1ec1u h\u00e0nh",
    "digits": "2016",
    "fullwidth digits": "\uff12\uff10\uff11\uff16",
    "ticker": "$AAPL",
    "euro": "\u20ac",
    "ligature": "\ufb01nance",
    "emoji": "\U0001f4c8",
    "punctuation": "???",
}


@pytest.mark.parametrize("script", SCRIPT_QUERIES)
def test_a_query_in_any_script_is_still_searchable_once_normalised(store, embedder, script):
    query = SCRIPT_QUERIES[script]
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        for text in (unicodedata.normalize(form, query), "\u200b " + unicodedata.normalize(form, query)):
            assert searchable(text)
            assert searchable(unicodedata.normalize("NFKD", text))
            embedder.calls.clear()
            assert run(store, embedder, query=text, limit=1)
            assert embedder.calls == [SEARCH_SPEC.query_prefix + unicodedata.normalize("NFKD", text)]


def test_nfkd_only_empties_a_query_made_of_isolated_spacing_marks():
    # Over every code point: a character that counts (L, N, S or P) keeps one that counts in
    # NFKD, except the spacing form of a diacritic ("\u00a8", "\u00b4", "\uff9e", the isolated
    # Arabic vowel marks...), which NFKD writes as a space and combining marks, or as the
    # combining mark alone. Those are the only queries normalisation can leave empty.
    emptied = [
        character
        for character in map(chr, range(0x110000))
        if searchable(character) and not searchable(unicodedata.normalize("NFKD", character))
    ]
    assert "\u00a8" in emptied and "\uff9e" in emptied
    assert len(emptied) < 100
    for character in emptied:
        decomposed = unicodedata.normalize("NFKD", character)
        assert {unicodedata.category(part) for part in decomposed} <= {"Zs", "Mn"}, ascii(character)
        assert any(unicodedata.combining(part) for part in decomposed), ascii(character)


@pytest.mark.parametrize(
    "mark", ["\u00a8", "\u00b4", "\u309b", "\uff9e", "\ufe7c", " \u00b8\u02dc "], ids=ascii
)
def test_a_query_that_normalisation_leaves_without_a_searchable_character_is_refused(store, embedder, mark):
    assert searchable(mark) and not searchable(unicodedata.normalize("NFKD", mark))
    with pytest.raises(SearchError, match="the query has no searchable character.*NFKD") as raised:
        run(store, embedder, query=mark)
    assert ascii(mark) in str(raised.value)
    assert embedder.calls == []

    # As given, it is a symbol or a letter: without normalisation it is embedded.
    assert run(store, embedder, query=mark, normalize_query=False)
    assert embedder.calls == [SEARCH_SPEC.query_prefix + mark]

    # With anything else that counts, it is a query like another.
    embedder.calls.clear()
    assert run(store, embedder, query="a" + mark)


@pytest.mark.parametrize("bad", [{"source_codes": ["ecb"]}, ["ecb"], "ecb", 3], ids=repr)
def test_filters_must_be_a_search_filters(store, embedder, bad):
    with pytest.raises(SearchError, match="filters must be a SearchFilters or None"):
        run(store, embedder, filters=bad)
    assert embedder.calls == []


@pytest.mark.parametrize("bad", [None, 1, 0, "true", "NFC"], ids=repr)
def test_normalize_query_must_be_a_boolean(store, embedder, bad):
    with pytest.raises(SearchError, match="normalize_query must be True or False"):
        run(store, embedder, normalize_query=bad)
    assert embedder.calls == []


def test_everything_but_the_query_is_keyword_only(store, embedder):
    with pytest.raises(TypeError):
        search(QUERY, COLLECTION, store, embedder)  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Query normalisation
# --------------------------------------------------------------------------------------

# Composed forms: an accent, a voiced kana, a Hangul syllable, a Vietnamese vowel, a ligature.
COMPOSED_QUERIES = [
    "décision de politique monétaire",
    "デフレ脱却",
    "한국은행",
    "lãi suất điều hành",
    "ﬁnance",
]


def test_the_normalisation_form_is_nfkd():
    assert QUERY_NORMALIZATION == "NFKD"


@pytest.mark.parametrize("query", COMPOSED_QUERIES)
def test_the_query_is_embedded_in_nfkd_by_default(store, embedder, query):
    composed = unicodedata.normalize("NFC", query)
    decomposed = unicodedata.normalize("NFKD", query)
    assert composed != decomposed

    run(store, embedder, query=composed)
    assert embedder.calls == [SEARCH_SPEC.query_prefix + decomposed]


@pytest.mark.parametrize("query", COMPOSED_QUERIES)
def test_the_query_is_embedded_as_given_when_normalisation_is_off(store, embedder, query):
    composed = unicodedata.normalize("NFC", query)

    run(store, embedder, query=composed, normalize_query=False)
    assert embedder.calls == [SEARCH_SPEC.query_prefix + composed]


def test_composed_and_decomposed_queries_give_the_same_hits_when_normalised(store, embedder):
    composed = unicodedata.normalize("NFC", "デフレ脱却")
    decomposed = unicodedata.normalize("NFKD", composed)

    assert_same_hits(run(store, embedder, query=composed), run(store, embedder, query=decomposed))
    raw = run(store, embedder, query=composed, normalize_query=False)
    assert [hit.point_id for hit in raw] != [hit.point_id for hit in run(store, embedder, query=composed)]


def test_the_query_is_not_stripped(store, embedder):
    run(store, embedder, query="  rates \n")
    assert embedder.calls == [SEARCH_SPEC.query_prefix + "  rates \n"]


# --------------------------------------------------------------------------------------
# Nothing is mutated, errors propagate
# --------------------------------------------------------------------------------------


def test_a_search_changes_neither_the_filters_nor_the_collection(store, embedder):
    filters = SearchFilters(source_codes=["ecb", "fed"], date_from="2016-03-10", corpus="central-bank")
    before_filters = copy.deepcopy(filters)
    before = store.client.scroll(COLLECTION, limit=100, with_payload=True, with_vectors=True)[0]

    first = run(store, embedder, filters=filters)
    second = run(store, embedder, filters=filters)

    assert_same_hits(first, second)
    assert first
    assert filters == before_filters
    after = store.client.scroll(COLLECTION, limit=100, with_payload=True, with_vectors=True)[0]
    assert [(record.id, record.payload) for record in after] == [
        (record.id, record.payload) for record in before
    ]
    # Local mode hands the stored vectors back about 1e-8 away once a first query has run.
    for now, then in zip(after, before, strict=True):
        assert now.vector == pytest.approx(then.vector, abs=SCORE_TOLERANCE)
    assert store.count(COLLECTION) == SEARCH_POINT_COUNT


def test_a_missing_collection_is_raised_as_the_client_raises_it(store, embedder):
    with pytest.raises(Exception, match="absent") as raised:
        search(QUERY, collection="absent", store=store, embedder=embedder)
    assert not isinstance(raised.value, RagEngineError)


def test_an_error_of_the_embedder_is_raised_unchanged(store, embedder):
    embedder.release()
    with pytest.raises(RuntimeError, match="released"):
        run(store, embedder)


class StatusError(Exception):
    """An error with an HTTP status, as qdrant-client raises for a non-2xx answer."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Unexpected Response: {status_code}")
        self.status_code = status_code


class RecordingClient:
    """A client that records the query it receives and answers with ``answer`` (or raises it)."""

    def __init__(self, answer=None) -> None:
        self.answer = answer
        self.requests: list[dict] = []

    def query_points(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.answer, Exception):
            raise self.answer
        return types.SimpleNamespace(points=self.answer or [])


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_api_key_is_an_auth_error(embedder, status):
    store = QdrantStore(client=RecordingClient(StatusError(status)))
    with pytest.raises(QdrantAuthError, match=f"HTTP {status}"):
        search(QUERY, collection=COLLECTION, store=store, embedder=embedder)


def test_any_other_store_error_is_raised_unchanged(embedder):
    failure = StatusError(500)
    store = QdrantStore(client=RecordingClient(failure))
    with pytest.raises(StatusError) as raised:
        search(QUERY, collection=COLLECTION, store=store, embedder=embedder)
    assert raised.value is failure


def test_one_request_with_the_payload_and_without_the_vectors(embedder):
    client = RecordingClient()
    store = QdrantStore(client=client)
    filters = SearchFilters(source_codes=["ecb"], date_to="2016-03-10")

    assert (
        search(QUERY, collection=COLLECTION, store=store, embedder=embedder, filters=filters, limit=7) == []
    )

    (request,) = client.requests  # one request, and no other call: the client has no other method
    vector = FakeEmbedder(SEARCH_SPEC).encode_query(QUERY)
    assert request["collection_name"] == COLLECTION
    assert request["query"] == vector.tolist() and all(type(x) is float for x in request["query"])
    assert request["query_filter"] == _qdrant_filter(filters)
    assert request["limit"] == 7
    assert request["with_payload"] is True and request["with_vectors"] is False
    assert set(request) == {
        "collection_name",
        "query",
        "query_filter",
        "limit",
        "with_payload",
        "with_vectors",
    }


def test_a_string_point_id_and_an_empty_payload_make_a_hit(embedder):
    uuid_id = "5c56c793-69f3-4fbf-87e6-c4bf54c28c26"
    answer = [
        models.ScoredPoint(id=uuid_id, version=0, score=0.5, payload=None),
        models.ScoredPoint(id=7, version=0, score=0.25, payload={"text": "t", "extra": {"k": 1}}),
    ]
    store = QdrantStore(client=RecordingClient(answer))

    first, second = search(QUERY, collection=COLLECTION, store=store, embedder=embedder)

    assert (first.point_id, first.score, first.text, first.doc_id, first.payload) == (
        uuid_id,
        0.5,
        None,
        None,
        {},
    )
    assert (second.point_id, second.text, second.payload) == (7, "t", {"text": "t", "extra": {"k": 1}})
    assert json.loads(json.dumps(first.to_dict(include_payload=True)))["point_id"] == uuid_id


# --------------------------------------------------------------------------------------
# QdrantStore.query
# --------------------------------------------------------------------------------------


def test_store_query_returns_scored_points_best_first(store):
    vector = FakeEmbedder(SEARCH_SPEC).encode_query(QUERY)
    points = store.query(COLLECTION, vector, limit=3)

    assert [point.id for point in points] == [point for point, _score in expected_ranking()][:3]
    assert all(point.vector is None and point.payload for point in points)


def test_store_query_applies_a_filter(store):
    vector = FakeEmbedder(SEARCH_SPEC).encode_query(QUERY).tolist()
    only = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="fed-b"))])
    points = store.query(COLLECTION, vector, query_filter=only, limit=10)
    assert {point.payload["doc_id"] for point in points} == {"fed-b"} and len(points) == 2


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "3", None], ids=repr)
def test_store_query_refuses_a_limit_that_is_not_a_positive_integer(bad):
    client = RecordingClient()
    with pytest.raises(ValueError, match="limit must be an integer >= 1"):
        QdrantStore(client=client).query(COLLECTION, [0.0], limit=bad)
    assert client.requests == []


# ---------------------------------------------------------------------------
# is_searchable: the rule of the query, for a caller that checks its queries beforehand
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["2016", "\u20ac", "$AAPL", "???", "caf\u00e9", "\u65e5\u672c"], ids=ascii)
def test_is_searchable_accepts_what_search_accepts(text):
    assert is_searchable(text) is True
    assert is_searchable(text, normalized=True) is True


@pytest.mark.parametrize("text", [*UNSEARCHABLE_QUERIES, "", "   "], ids=ascii)
def test_is_searchable_refuses_what_search_refuses(text):
    assert is_searchable(text) is False
    assert is_searchable(text, normalized=True) is False


@pytest.mark.parametrize("mark", ["\u00a8", "\u00b4", "\u309b", " \u00b8\u02dc "], ids=ascii)
def test_is_searchable_can_look_at_the_normalised_form(store, embedder, mark):
    # Searchable as given, not once put in NFKD: the search refuses it unless normalize_query is off.
    assert is_searchable(mark) is True
    assert is_searchable(mark, normalized=True) is False
    with pytest.raises(SearchError):
        run(store, embedder, query=mark)
    assert run(store, embedder, query=mark, normalize_query=False)
