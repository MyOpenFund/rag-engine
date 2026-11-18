"""Payload schema v2 and point ids: byte-stable with the points already in Qdrant."""

import datetime
import subprocess
import sys
import types

import pytest

from rag_engine import schema
from rag_engine.schema import (
    DATE_PRECISIONS,
    DERIVED_KEYS,
    PAYLOAD_INDEXES,
    PAYLOAD_SCHEMA_VERSION,
    RESERVED_KEYS,
    build_payload,
    date_interval,
    date_interval_fields,
    page_for_id,
    point_id,
)

# --- point ids -------------------------------------------------------------

# Ported unchanged from data-orchestrator/tests/test_point_id.py.
GOLDEN_1 = 5299910714168844021
GOLDEN_2 = 5572097760344244118
GOLDEN_3 = 6912429553972229415
GOLDEN_4 = 703449921473339669


def test_point_id_golden_values():
    # Pinned outputs of the data-orchestrator implementation. If these change, ids of
    # every existing collection would drift — never accept a change here.
    assert point_id("c184d44f298ff622", 1, 0) == GOLDEN_1
    assert point_id("c184d44f298ff622", 1, 1) == GOLDEN_2
    assert point_id("c184d44f298ff622", 2, 0) == GOLDEN_3
    assert point_id("0" * 16, 0, 0) == GOLDEN_4


def test_point_id_is_deterministic_and_positive():
    a = point_id("abc", 3, 7)
    assert a == point_id("abc", 3, 7)
    assert 0 <= a < 2**63


# Captured by running data-orchestrator's `core._point_id` itself on these inputs.
ORCHESTRATOR_VECTORS = [
    (("rapport-été-2024-日本語", 0, 0), 2603997400248582809),
    (("rapport-été-2024-日本語", 12, 3), 8907132167374451235),
    (("a3f1c9e2b7d40586", 0, 7), 8896207534468579224),
    (("a3f1c9e2b7d40586", 250, 1999), 3622071315152915808),
    (("", 0, 0), 5377612559776746822),
    (("doc::with::colons", 1, 11), 261376066613953929),
    (("abc", 3, 7), 2159512958636821857),
]


@pytest.mark.parametrize(("args", "expected"), ORCHESTRATOR_VECTORS)
def test_point_id_matches_orchestrator_vectors(args, expected):
    assert point_id(*args) == expected


def test_point_id_separates_page_and_chunk_index():
    assert point_id("d", 1, 0) != point_id("d", 0, 1)
    assert len({point_id("d", p, c) for p in range(20) for c in range(20)}) == 400


# --- page handling ---------------------------------------------------------


def test_page_for_id_maps_none_to_zero():
    assert page_for_id(None) == 0
    assert page_for_id(0) == 0
    assert page_for_id(7) == 7


def test_unpaginated_chunk_uses_page_zero_in_id_but_no_page_in_payload():
    payload = build_payload(
        "d1", {}, filename="a.md", chunk_number=0, ingestion_date="t", text="x", page=None
    )
    assert "page" not in payload
    assert point_id("d1", page_for_id(None), 0) == point_id("d1", 0, 0)


def test_page_zero_is_kept_in_payload():
    payload = build_payload("d1", {}, filename="a.pdf", chunk_number=0, ingestion_date="t", text="x", page=0)
    assert payload["page"] == 0


# --- payload ---------------------------------------------------------------

VAULT_METADATA = {
    "corpus": "central-bank",
    "source_code": "A1",
    "doc_type": "speech",
    "title": "Monetary policy in uncertain times",
    "date": "2024-03-07",
    "year": 2024,
    "language": "en",
    "sha256": "ab" * 32,
    "provenance": "https://example.org/speech.pdf",
}
VAULT_INTERVAL = {"date_start": "2024-03-07T00:00:00Z", "date_end": "2024-03-07T00:00:00Z"}
ENGINE_FIELDS = {
    "filename": "speech.pdf",
    "chunk_number": 3,
    "ingestion_date": "2026-11-17T10:11:12.131415",
    "text": "Inflation has eased.",
}


def test_reserved_keys():
    assert set(RESERVED_KEYS) == {"doc_id", "filename", "chunk_number", "ingestion_date", "text", "page"}
    assert len(RESERVED_KEYS) == 6


def test_build_payload_matches_orchestrator_payload_with_page():
    payload = build_payload("c184d44f298ff622", VAULT_METADATA, **ENGINE_FIELDS, page=2)
    # Same content and same key order as data-orchestrator's `ingest_item`.
    assert list(payload.items()) == [
        ("doc_id", "c184d44f298ff622"),
        ("corpus", "central-bank"),
        ("source_code", "A1"),
        ("doc_type", "speech"),
        ("title", "Monetary policy in uncertain times"),
        ("date", "2024-03-07"),
        ("year", 2024),
        ("language", "en"),
        ("sha256", "ab" * 32),
        ("provenance", "https://example.org/speech.pdf"),
        ("filename", "speech.pdf"),
        ("chunk_number", 3),
        ("ingestion_date", "2026-11-17T10:11:12.131415"),
        ("text", "Inflation has eased."),
        ("page", 2),
        ("date_start", "2024-03-07T00:00:00Z"),
        ("date_end", "2024-03-07T00:00:00Z"),
    ]


def test_build_payload_without_page_has_no_page_key():
    payload = build_payload("c184d44f298ff622", VAULT_METADATA, **ENGINE_FIELDS, page=None)
    assert payload == {"doc_id": "c184d44f298ff622", **VAULT_METADATA, **ENGINE_FIELDS, **VAULT_INTERVAL}
    assert "page" not in payload
    assert "chunk_index" not in payload


def test_build_payload_keeps_unknown_metadata_keys_and_partial_metadata():
    # The vault row's `extra` adds arbitrary keys; null columns (e.g. `date`) are simply absent.
    payload = build_payload(
        "d1", {"corpus": "c", "series": "WP", "tags": ["a", "b"]}, **ENGINE_FIELDS, page=1
    )
    assert payload["series"] == "WP"
    assert payload["tags"] == ["a", "b"]
    assert "date" not in payload


def test_build_payload_tolerates_same_doc_id_in_metadata():
    # The orchestrator's vault payload carries `doc_id` itself.
    metadata = {"doc_id": "d1", **VAULT_METADATA}
    payload = build_payload("d1", metadata, **ENGINE_FIELDS, page=None)
    assert payload == {"doc_id": "d1", **VAULT_METADATA, **ENGINE_FIELDS, **VAULT_INTERVAL}
    assert next(iter(payload)) == "doc_id"


def test_build_payload_rejects_different_doc_id_in_metadata():
    with pytest.raises(ValueError, match="doc_id"):
        build_payload("d1", {"doc_id": "d2"}, **ENGINE_FIELDS, page=None)


@pytest.mark.parametrize("key", ["filename", "chunk_number", "ingestion_date", "text", "page"])
def test_build_payload_rejects_reserved_key(key):
    with pytest.raises(ValueError, match=rf"\b{key}\b") as excinfo:
        build_payload("d1", {"corpus": "c", key: "clash"}, **ENGINE_FIELDS, page=None)
    assert "corpus" not in str(excinfo.value)


def test_build_payload_rejects_reserved_key_even_with_equal_value_or_none():
    with pytest.raises(ValueError, match="text"):
        build_payload("d1", {"text": ENGINE_FIELDS["text"]}, **ENGINE_FIELDS, page=None)
    with pytest.raises(ValueError, match="page"):
        build_payload("d1", {"page": None}, **ENGINE_FIELDS, page=None)


def test_build_payload_names_every_offending_key():
    with pytest.raises(ValueError) as excinfo:
        build_payload(
            "d1", {"text": "t", "doc_id": "other", "page": 9, "title": "ok"}, **ENGINE_FIELDS, page=1
        )
    message = str(excinfo.value)
    for key in ("text", "doc_id", "page"):
        assert key in message
    assert "title" not in message
    assert "filename" not in message


def test_build_payload_does_not_mutate_metadata_and_returns_a_new_dict():
    metadata = {"doc_id": "d1", **VAULT_METADATA}
    snapshot = dict(metadata)
    first = build_payload("d1", metadata, **ENGINE_FIELDS, page=4)
    second = build_payload("d1", metadata, **ENGINE_FIELDS, page=None)
    assert metadata == snapshot
    assert first is not metadata and first is not second
    assert "page" in first and "page" not in second


def test_build_payload_accepts_a_read_only_mapping():
    payload = build_payload("d1", types.MappingProxyType({"corpus": "c"}), **ENGINE_FIELDS, page=None)
    assert payload["corpus"] == "c"


def test_build_payload_engine_fields_are_keyword_only():
    with pytest.raises(TypeError):
        build_payload("d1", {}, "a.pdf", 0, "t", "x", None)


# --- date intervals (schema v2) --------------------------------------------


def test_the_schema_version_and_the_derived_keys():
    assert PAYLOAD_SCHEMA_VERSION == 2
    assert DERIVED_KEYS == ("date_start", "date_end")
    assert DATE_PRECISIONS == ("day", "month", "year")
    assert not set(DERIVED_KEYS) & set(RESERVED_KEYS)
    assert PAYLOAD_INDEXES[-2:] == (("date_start", "datetime"), ("date_end", "datetime"))


INTERVALS = [
    # (date, precision, first day, last day)
    ("2016-03-10", "day", "2016-03-10", "2016-03-10"),
    ("2016-03-10", None, "2016-03-10", "2016-03-10"),
    ("2016-03-01", "month", "2016-03-01", "2016-03-31"),
    ("2016-03-17", "month", "2016-03-01", "2016-03-31"),  # not the first day: the month it falls in
    ("2016-02-01", "month", "2016-02-01", "2016-02-29"),  # a leap February
    ("2015-02-01", "month", "2015-02-01", "2015-02-28"),
    ("1900-02-01", "month", "1900-02-01", "1900-02-28"),  # a century that is not a leap year
    ("2000-02-01", "month", "2000-02-01", "2000-02-29"),
    ("2016-12-01", "month", "2016-12-01", "2016-12-31"),  # the month that ends the year
    ("2016-04-30", "month", "2016-04-01", "2016-04-30"),
    ("2016-01-01", "year", "2016-01-01", "2016-12-31"),
    ("2016-12-31", "year", "2016-01-01", "2016-12-31"),
    ("2016-02-29", "day", "2016-02-29", "2016-02-29"),
    ("2016-12-31", "day", "2016-12-31", "2016-12-31"),
    ("2016-02-29", "year", "2016-01-01", "2016-12-31"),
    ("0001-01-01", "year", "0001-01-01", "0001-12-31"),
    ("9999-12-31", "month", "9999-12-01", "9999-12-31"),
    ("9999-12-31", "year", "9999-01-01", "9999-12-31"),
    ("9999-12-31", "day", "9999-12-31", "9999-12-31"),
]


@pytest.mark.parametrize(("date", "precision", "first", "last"), INTERVALS)
def test_date_interval_is_the_first_and_the_last_day_at_midnight_utc(date, precision, first, last):
    assert date_interval(date, precision) == (f"{first}T00:00:00Z", f"{last}T00:00:00Z")


@pytest.mark.parametrize(
    "precision", ["", "DAY", "Month", " month", "week", "quarter", 3, True, ["month"], {}]
)
def test_an_unknown_precision_is_read_as_a_day(precision):
    assert date_interval("2016-03-01", precision) == ("2016-03-01T00:00:00Z", "2016-03-01T00:00:00Z")


STORED_FORMS = [
    # A date with a time is the UTC day of that instant, as the datetime index reads it.
    ("2016-03-10T15:30:00Z", "2016-03-10"),
    ("2016-03-10T15:30:00z", "2016-03-10"),
    ("2016-03-10 15:30:00", "2016-03-10"),
    ("2016-03-10T15:30", "2016-03-10"),
    ("2016-03-10T15:30:00.123456789Z", "2016-03-10"),
    ("2016-03-10T23:59:59.999999Z", "2016-03-10"),
    ("2016-03-11T00:30:00+02:00", "2016-03-10"),  # 22:30 the day before, in UTC
    ("2016-03-10T23:30:00-05:00", "2016-03-11"),  # 04:30 the day after, in UTC
    ("2016-03-10T12:00:00+0200", "2016-03-10"),
    ("2016-03-10T12:00:00+02", "2016-03-10"),
    ("2016-12-31T23:30:00-01:00", "2017-01-01"),  # an offset that crosses the year
    ("2016-3-10", "2016-03-10"),  # the lenient form the index reads
    (datetime.date(2016, 3, 10), "2016-03-10"),
    (datetime.datetime(2016, 3, 10, 15, 30), "2016-03-10"),  # naive: UTC
    (
        datetime.datetime(2016, 3, 11, 0, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
        "2016-03-10",
    ),
]


@pytest.mark.parametrize(("stored", "day"), STORED_FORMS, ids=repr)
def test_date_interval_reads_the_utc_day_of_a_date_with_a_time(stored, day):
    assert date_interval(stored, "day") == (f"{day}T00:00:00Z", f"{day}T00:00:00Z")
    assert date_interval(stored) == date_interval(stored, "day")


def test_the_period_of_a_date_with_an_offset_is_the_one_of_its_utc_day():
    assert date_interval("2016-04-01T00:30:00+02:00", "month") == (
        "2016-03-01T00:00:00Z",
        "2016-03-31T00:00:00Z",
    )
    assert date_interval("2017-01-01T00:30:00+02:00", "year") == (
        "2016-01-01T00:00:00Z",
        "2016-12-31T00:00:00Z",
    )


NO_DATE = [
    None,
    "",
    "   ",
    "2016",
    "2016-03",
    "10/03/2016",
    "20160310",
    "2016-02-30",
    "2015-02-29",
    "2016-13-01",
    "0000-01-01",
    "2016-03-10T25:00:00Z",
    "2016-03-10T12:00:00+25:00",
    "0001-01-01T00:30:00+02:00",  # before the first day of the calendar, in UTC
    "9999-12-31T23:30:00-02:00",  # after the last one
    "2016-03-10 ",
    "March 2016",
    20160310,
    1457568000,
    2016.0,
    True,
    ["2016-03-10"],
    {"date": "2016-03-10"},
    b"2016-03-10",
]


@pytest.mark.parametrize("value", NO_DATE, ids=repr)
@pytest.mark.parametrize("precision", [None, "day", "month", "year"])
def test_no_date_or_an_unreadable_one_gives_no_interval(value, precision):
    assert date_interval(value, precision) is None
    assert date_interval_fields(value, precision) == {"date_start": None, "date_end": None}


def test_date_interval_fields_names_the_two_keys():
    assert date_interval_fields("2016-07-01", "month") == {
        "date_start": "2016-07-01T00:00:00Z",
        "date_end": "2016-07-31T00:00:00Z",
    }
    assert list(date_interval_fields("2016-07-01")) == list(DERIVED_KEYS)


@pytest.mark.parametrize(("date", "precision", "first", "last"), INTERVALS)
def test_build_payload_writes_the_interval_date_interval_gives(date, precision, first, last):
    metadata = {"corpus": "c", "date": date, "date_precision": precision, "date_kind": "meeting"}
    payload = build_payload("d1", metadata, **ENGINE_FIELDS, page=3)
    assert (payload["date_start"], payload["date_end"]) == date_interval(date, precision)
    assert (payload["date_start"], payload["date_end"]) == (f"{first}T00:00:00Z", f"{last}T00:00:00Z")
    # The keys of version 1 come first, in their order; the two derived keys are the last ones.
    assert list(payload) == [
        "doc_id",
        "corpus",
        "date",
        "date_precision",
        "date_kind",
        "filename",
        "chunk_number",
        "ingestion_date",
        "text",
        "page",
        "date_start",
        "date_end",
    ]
    # The metadata passes through as given: the date is not rewritten.
    assert payload["date"] == date
    assert payload["date_precision"] == precision
    assert payload["date_kind"] == "meeting"


def test_build_payload_without_a_precision_key_reads_a_day():
    payload = build_payload("d1", {"date": "2016-07-01"}, **ENGINE_FIELDS, page=None)
    assert (payload["date_start"], payload["date_end"]) == ("2016-07-01T00:00:00Z", "2016-07-01T00:00:00Z")
    assert "date_precision" not in payload


def test_build_payload_reads_a_garbage_precision_as_a_day_and_keeps_it():
    payload = build_payload(
        "d1", {"date": "2016-07-01", "date_precision": "fortnight"}, **ENGINE_FIELDS, page=None
    )
    assert (payload["date_start"], payload["date_end"]) == ("2016-07-01T00:00:00Z", "2016-07-01T00:00:00Z")
    assert payload["date_precision"] == "fortnight"


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"date": None},
        {"date": None, "date_precision": None},
        {"date": ""},
        {"date": "2016-03", "date_precision": "month"},
        {"date": "2016", "date_precision": "year"},
        {"date_precision": "month", "year": 2016},
    ],
    ids=repr,
)
def test_build_payload_writes_no_interval_without_a_readable_date(metadata):
    payload = build_payload("d1", metadata, **ENGINE_FIELDS, page=None)
    assert payload == {"doc_id": "d1", **metadata, **ENGINE_FIELDS}
    assert "date_start" not in payload and "date_end" not in payload


def test_date_precision_and_date_kind_are_ordinary_metadata():
    assert not {"date_precision", "date_kind"} & (set(RESERVED_KEYS) | set(DERIVED_KEYS))
    payload = build_payload(
        "d1", {"date_precision": "year", "date_kind": "period"}, **ENGINE_FIELDS, page=None
    )
    assert (payload["date_precision"], payload["date_kind"]) == ("year", "period")


@pytest.mark.parametrize("key", ["date_start", "date_end"])
@pytest.mark.parametrize("value", ["2016-03-01T00:00:00Z", None])
def test_build_payload_refuses_a_derived_key_in_the_metadata(key, value):
    with pytest.raises(ValueError, match=rf"derived payload key\(s\): {key}\b"):
        build_payload("d1", {"date": "2016-03-01", key: value}, **ENGINE_FIELDS, page=None)


# --- documentation and dependencies ----------------------------------------


def test_module_docstring_documents_schema_v2():
    doc = schema.__doc__
    assert f"schema v{PAYLOAD_SCHEMA_VERSION}" in doc
    for key in (*RESERVED_KEYS, *DERIVED_KEYS, "date_precision", "date_kind"):
        assert f"``{key}``" in doc
    for key in (
        "corpus",
        "source_code",
        "doc_type",
        "title",
        "date",
        "year",
        "language",
        "sha256",
        "provenance",
    ):
        assert f"``{key}``" in doc
    assert "publication" in doc


def test_schema_imports_only_the_standard_library():
    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import rag_engine.schema\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before} - {'rag_engine'}\n"
        "print(sorted(new - set(sys.stdlib_module_names)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
