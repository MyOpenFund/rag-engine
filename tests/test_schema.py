"""Payload schema v1 and point ids: byte-stable with the points already in Qdrant."""

import subprocess
import sys
import types

import pytest

from rag_engine import schema
from rag_engine.schema import RESERVED_KEYS, build_payload, page_for_id, point_id

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
    payload = build_payload("d1", {}, filename="a.md", chunk_number=0, ingestion_date="t", text="x", page=None)
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
    ]


def test_build_payload_without_page_has_no_page_key():
    payload = build_payload("c184d44f298ff622", VAULT_METADATA, **ENGINE_FIELDS, page=None)
    assert payload == {"doc_id": "c184d44f298ff622", **VAULT_METADATA, **ENGINE_FIELDS}
    assert "page" not in payload
    assert "chunk_index" not in payload


def test_build_payload_keeps_unknown_metadata_keys_and_partial_metadata():
    # The vault row's `extra` adds arbitrary keys; null columns (e.g. `date`) are simply absent.
    payload = build_payload("d1", {"corpus": "c", "series": "WP", "tags": ["a", "b"]}, **ENGINE_FIELDS, page=1)
    assert payload["series"] == "WP"
    assert payload["tags"] == ["a", "b"]
    assert "date" not in payload


def test_build_payload_tolerates_same_doc_id_in_metadata():
    # The orchestrator's vault payload carries `doc_id` itself.
    metadata = {"doc_id": "d1", **VAULT_METADATA}
    payload = build_payload("d1", metadata, **ENGINE_FIELDS, page=None)
    assert payload == {"doc_id": "d1", **VAULT_METADATA, **ENGINE_FIELDS}
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
        build_payload("d1", {"text": "t", "doc_id": "other", "page": 9, "title": "ok"}, **ENGINE_FIELDS, page=1)
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


# --- documentation and dependencies ----------------------------------------


def test_module_docstring_documents_schema_v1():
    doc = schema.__doc__
    for key in RESERVED_KEYS:
        assert f"``{key}``" in doc
    for key in ("corpus", "source_code", "doc_type", "title", "date", "year", "language", "sha256", "provenance"):
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
