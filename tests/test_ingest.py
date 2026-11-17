"""``ingest_document``: one document chunked, embedded and written, pinned against the previous pipeline.

Two layers. The first replaces the chunker (``rag_engine.ingest.chunk_file``) by a stub, so the
contract is tested without the ``ingest`` extra. The second (bottom of the file) parses the two
documents of ``tests/fixtures/golden/`` with the real libraries and compares the points with the
ones the previous pipeline wrote for them (``expected.json``, see the README of that folder).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from qdrant_client import QdrantClient

import rag_engine.ingest as ingest_mod
from rag_engine.chunking import Chunk, ChunkingOptions, UnsupportedFormatError
from rag_engine.config import Settings
from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.ingest import IngestResult, ingest_document
from rag_engine.schema import RESERVED_KEYS, point_id
from rag_engine.store import QdrantAuthError, QdrantStore

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
SPEC = ModelSpec(name="fake", dim=8, passage_prefix="passage: ", query_prefix="query: ")
COLLECTION = "docs-fake-v1"
DOC_ID = "ecb/press-release/2025-01-30"
METADATA = {"corpus": "central-bank", "source_code": "ecb", "date": "2025-01-30", "year": 2025}
WHEN = "2026-01-02T03:04:05.000006"


class HttpError(Exception):
    """An error with an HTTP status, as qdrant-client raises for a non-2xx answer."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class RecordingClient:
    """The part of a Qdrant client the store uses to write. ``fail_at`` maps a call number (from 1) to an error."""

    def __init__(self, fail_at: dict[int, Exception] | None = None) -> None:
        self.fail_at = fail_at or {}
        self.calls: list[tuple[str, list]] = []

    def upsert(self, collection_name, points, wait=None):
        batch = list(points)
        error = self.fail_at.get(len(self.calls) + 1)
        self.calls.append((collection_name, batch))
        if error is not None:
            raise error

    @property
    def written(self) -> list:
        """Every point of every batch that did not fail, in order."""
        return [
            point
            for number, (_, batch) in enumerate(self.calls, start=1)
            if number not in self.fail_at
            for point in batch
        ]


class CountingEmbedder(FakeEmbedder):
    """A ``FakeEmbedder`` that also counts the ``encode_passage`` calls."""

    def __init__(self, spec: ModelSpec = SPEC) -> None:
        super().__init__(spec)
        self.passage_calls: list[list[str]] = []

    def encode_passage(self, texts):
        self.passage_calls.append(list(texts))
        return super().encode_passage(texts)


class Setup:
    """A store over a recording client, an embedder, and what the stub chunker was asked."""

    def __init__(self, fail_at: dict[int, Exception] | None = None, batch_size: int = 100) -> None:
        self.client = RecordingClient(fail_at)
        self.store = QdrantStore(client=self.client, settings=Settings(batch_size=batch_size))
        self.embedder = CountingEmbedder()
        self.chunk_calls: list[tuple] = []

    def ingest(self, path="report.pdf", doc_id=DOC_ID, metadata=METADATA, **kwargs) -> IngestResult:
        kwargs.setdefault("collection", COLLECTION)
        return ingest_document(path, doc_id, metadata, store=self.store, embedder=self.embedder, **kwargs)

    @property
    def untouched(self) -> bool:
        return not self.embedder.passage_calls and not self.embedder.calls and not self.client.calls


@pytest.fixture
def setup():
    return Setup()


def stub_chunker(monkeypatch, chunks, calls: list | None = None):
    """Make ``ingest_document`` see ``chunks`` (a list, or an exception to raise) for any file."""

    def chunk_file(path, *, ocr="auto", ocr_languages="eng", options=None):
        if calls is not None:
            calls.append((path, ocr, ocr_languages, options))
        if isinstance(chunks, BaseException):
            raise chunks
        return list(chunks)

    monkeypatch.setattr(ingest_mod, "chunk_file", chunk_file)


THREE = [Chunk("first chunk", 0), Chunk("second chunk", 0), Chunk("third chunk", 2)]


# ---------------------------------------------------------------------------
# Layer 1: stub chunker
# ---------------------------------------------------------------------------


def test_one_point_per_chunk_with_the_schema_payload_and_the_passage_vector(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    result = setup.ingest("some/folder/report.pdf", ingestion_date=WHEN)

    assert result == IngestResult(chunks=3, pages=2, empty=False)
    points = setup.client.written
    assert [call[0] for call in setup.client.calls] == [COLLECTION]
    assert [point.id for point in points] == [
        point_id(DOC_ID, 0, 0),
        point_id(DOC_ID, 0, 1),
        point_id(DOC_ID, 2, 2),
    ]
    assert points[2].payload == {
        "doc_id": DOC_ID,
        **METADATA,
        "filename": "report.pdf",
        "chunk_number": 2,
        "ingestion_date": WHEN,
        "text": "third chunk",
        "page": 2,
    }
    assert list(points[2].payload) == [
        "doc_id",
        *METADATA,
        "filename",
        "chunk_number",
        "ingestion_date",
        "text",
        "page",
    ]
    expected = FakeEmbedder(SPEC).encode_passage([chunk.text for chunk in THREE])
    for point, vector in zip(points, expected):
        assert point.vector == vector.tolist()
        assert all(type(value) is float for value in point.vector)


def test_all_the_texts_are_embedded_in_one_call_in_document_order(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    setup.ingest()
    assert setup.embedder.passage_calls == [["first chunk", "second chunk", "third chunk"]]


def test_the_arguments_reach_the_chunker(setup, monkeypatch):
    options = ChunkingOptions(max_chunk_word_count=50, hard_max_chunk_word_count=80)
    stub_chunker(monkeypatch, THREE, setup.chunk_calls)
    path = Path("folder") / "report.pdf"
    setup.ingest(path, ocr="always", ocr_languages="eng+fra", chunking_options=options)
    assert setup.chunk_calls == [(path, "always", "eng+fra", options)]


def test_the_chunker_defaults(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE, setup.chunk_calls)
    setup.ingest()
    assert setup.chunk_calls == [("report.pdf", "auto", "eng", None)]


@pytest.mark.parametrize("chunks", [[], [Chunk("only one", None)]], ids=["none", "one"])
def test_result_counts(setup, monkeypatch, chunks):
    stub_chunker(monkeypatch, chunks)
    assert setup.ingest() == IngestResult(chunks=len(chunks), pages=0, empty=not chunks)


def test_a_document_without_chunks_is_empty_and_nothing_is_embedded_or_written(setup, monkeypatch):
    stub_chunker(monkeypatch, [])
    assert setup.ingest() == IngestResult(chunks=0, pages=0, empty=True)
    assert setup.untouched


def test_the_result_cannot_be_changed(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    result = setup.ingest()
    with pytest.raises(AttributeError):
        result.chunks = 0


def test_chunks_with_the_same_text_keep_distinct_ids_and_are_all_written(setup, monkeypatch):
    stub_chunker(monkeypatch, [Chunk("same text", 3), Chunk("same text", 3)])
    assert setup.ingest() == IngestResult(chunks=2, pages=1, empty=False)
    points = setup.client.written
    assert len({point.id for point in points}) == 2
    assert [point.payload["chunk_number"] for point in points] == [0, 1]
    assert [point.payload["text"] for point in points] == ["same text", "same text"]
    assert points[0].vector == points[1].vector


def test_a_chunk_without_page_uses_page_zero_in_the_id_and_has_no_page_in_the_payload(setup, monkeypatch):
    stub_chunker(monkeypatch, [Chunk("no page here", None), Chunk("nor here", None)])
    assert setup.ingest("notes.md").pages == 0
    points = setup.client.written
    assert [point.id for point in points] == [point_id(DOC_ID, 0, 0), point_id(DOC_ID, 0, 1)]
    assert all("page" not in point.payload for point in points)


def test_pages_counts_the_distinct_pages(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)  # pages 0, 0, 2
    assert setup.ingest().pages == 2


def test_pages_ignores_the_chunks_without_page(setup, monkeypatch):
    stub_chunker(monkeypatch, [Chunk("a", None), Chunk("b", 4), Chunk("c", None)])
    assert setup.ingest().pages == 1


def test_metadata_without_date_is_accepted(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    setup.ingest(metadata={"corpus": "central-bank"})
    assert all("date" not in point.payload for point in setup.client.written)
    assert setup.client.written[0].payload["corpus"] == "central-bank"


def test_empty_metadata_is_accepted(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    setup.ingest(metadata={})
    assert list(setup.client.written[0].payload) == [
        "doc_id",
        "filename",
        "chunk_number",
        "ingestion_date",
        "text",
        "page",
    ]


def test_metadata_repeating_the_same_doc_id_is_accepted(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    setup.ingest(metadata={"doc_id": DOC_ID, "corpus": "central-bank"})
    assert setup.client.written[0].payload["doc_id"] == DOC_ID


@pytest.mark.parametrize("key", [key for key in RESERVED_KEYS])
def test_a_reserved_key_in_the_metadata_is_refused_before_anything_is_read_embedded_or_written(
    setup, monkeypatch, key
):
    stub_chunker(monkeypatch, THREE, setup.chunk_calls)
    with pytest.raises(ValueError, match=key):
        setup.ingest(metadata={**METADATA, key: "other"})
    assert setup.chunk_calls == []
    assert setup.untouched


def test_a_reserved_key_is_refused_even_when_the_document_is_empty(setup, monkeypatch):
    stub_chunker(monkeypatch, [])
    with pytest.raises(ValueError, match="text"):
        setup.ingest(metadata={"text": "x"})


def test_the_metadata_mapping_is_not_changed(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    nested = {"tags": ["a", "b"]}
    metadata = {**METADATA, "extra": nested}
    before = json.dumps(metadata, sort_keys=True)
    setup.ingest(metadata=metadata)
    assert json.dumps(metadata, sort_keys=True) == before
    assert list(metadata) == [*METADATA, "extra"]


def test_a_read_only_mapping_is_accepted(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    setup.ingest(metadata=MappingProxyType(dict(METADATA)))
    assert setup.client.written[0].payload["source_code"] == "ecb"


def test_the_points_do_not_share_one_payload_dict(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    setup.ingest()
    first, second, _ = setup.client.written
    assert first.payload is not second.payload


@pytest.mark.parametrize("metadata", [None, [("corpus", "x")], "corpus"])
def test_metadata_that_is_not_a_mapping_is_refused(setup, monkeypatch, metadata):
    stub_chunker(monkeypatch, THREE, setup.chunk_calls)
    with pytest.raises(TypeError, match="metadata"):
        setup.ingest(metadata=metadata)
    assert setup.chunk_calls == [] and setup.untouched


@pytest.mark.parametrize("doc_id", ["", "   ", "\t\n", None, 12])
def test_a_blank_doc_id_is_refused_before_the_file_is_read(setup, monkeypatch, doc_id):
    stub_chunker(monkeypatch, THREE, setup.chunk_calls)
    with pytest.raises(ValueError, match="doc_id"):
        setup.ingest(doc_id=doc_id)
    assert setup.chunk_calls == [] and setup.untouched


@pytest.mark.parametrize("collection", ["", "  ", None])
def test_a_blank_collection_is_refused_before_the_file_is_read(setup, monkeypatch, collection):
    stub_chunker(monkeypatch, THREE, setup.chunk_calls)
    with pytest.raises(ValueError, match="collection"):
        setup.ingest(collection=collection)
    assert setup.chunk_calls == [] and setup.untouched


@pytest.mark.parametrize("batch_size", [0, -1, True, 2.0, "10"])
def test_a_batch_size_that_cannot_be_used_is_refused_before_the_file_is_read(setup, monkeypatch, batch_size):
    stub_chunker(monkeypatch, THREE, setup.chunk_calls)
    with pytest.raises(ValueError, match="batch_size"):
        setup.ingest(batch_size=batch_size)
    assert setup.chunk_calls == [] and setup.untouched


@pytest.mark.parametrize("when", ["", "  ", datetime.datetime(2026, 1, 2), 1767322800])
def test_an_ingestion_date_that_is_not_a_non_blank_string_is_refused(setup, monkeypatch, when):
    stub_chunker(monkeypatch, THREE, setup.chunk_calls)
    with pytest.raises(ValueError, match="ingestion_date"):
        setup.ingest(ingestion_date=when)
    assert setup.chunk_calls == [] and setup.untouched


def test_a_path_without_a_file_name_is_refused(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE, setup.chunk_calls)
    with pytest.raises(ValueError, match="path"):
        setup.ingest(path="")
    assert setup.chunk_calls == [] and setup.untouched


def test_the_filename_is_the_base_name_of_the_path(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    setup.ingest(Path("/data") / "ecb" / "2025" / "Décision.PDF")
    assert {point.payload["filename"] for point in setup.client.written} == {"Décision.PDF"}


def test_without_ingestion_date_every_point_carries_the_same_current_iso_timestamp(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    before = datetime.datetime.now()
    setup.ingest()
    after = datetime.datetime.now()
    dates = {point.payload["ingestion_date"] for point in setup.client.written}
    assert len(dates) == 1
    when = datetime.datetime.fromisoformat(dates.pop())
    assert when.tzinfo is None  # local naive time, as the previous pipeline wrote it
    assert before <= when <= after


def test_the_clock_is_read_once_per_document(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    reads = []

    class Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            reads.append(tz)
            return cls(2026, 5, 6, 7, 8, 9, 123456)

    monkeypatch.setattr(ingest_mod.datetime, "datetime", Clock)
    setup.ingest()
    assert reads == [None]
    assert {point.payload["ingestion_date"] for point in setup.client.written} == {
        "2026-05-06T07:08:09.123456"
    }


def test_the_batch_size_is_handed_to_the_store(monkeypatch):
    setup = Setup(batch_size=100)
    stub_chunker(monkeypatch, THREE)
    setup.ingest(batch_size=2)
    assert [len(batch) for _, batch in setup.client.calls] == [2, 1]


def test_without_batch_size_the_store_uses_its_own(monkeypatch):
    setup = Setup(batch_size=1)
    stub_chunker(monkeypatch, THREE)
    setup.ingest()
    assert [len(batch) for _, batch in setup.client.calls] == [1, 1, 1]


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_api_key_on_the_first_batch_propagates_with_nothing_written(monkeypatch, status):
    setup = Setup(fail_at={1: HttpError(status)})
    stub_chunker(monkeypatch, THREE)
    with pytest.raises(QdrantAuthError, match=str(status)) as raised:
        setup.ingest(batch_size=1)
    assert raised.value.points_written == 0
    assert len(setup.client.calls) == 1
    assert setup.client.written == []


def test_a_refused_api_key_on_a_later_batch_propagates_with_the_points_already_written(monkeypatch):
    setup = Setup(fail_at={3: HttpError(401)})
    stub_chunker(monkeypatch, [Chunk(f"chunk {i}", i) for i in range(7)])
    with pytest.raises(QdrantAuthError) as raised:
        setup.ingest(batch_size=2)
    assert raised.value.points_written == 4
    assert len(setup.client.calls) == 3  # the batches after the failed one are not sent
    assert [point.payload["chunk_number"] for point in setup.client.written] == [0, 1, 2, 3]


def test_another_store_error_propagates_unchanged(monkeypatch):
    error = ConnectionError("qdrant went away")
    setup = Setup(fail_at={2: error})
    stub_chunker(monkeypatch, THREE)
    with pytest.raises(ConnectionError) as raised:
        setup.ingest(batch_size=1)
    assert raised.value is error
    assert raised.value.points_written == 1


class WrongCountEmbedder(CountingEmbedder):
    """Returns ``delta`` vectors more (or fewer) than it was given texts."""

    def __init__(self, delta: int) -> None:
        super().__init__()
        self.delta = delta

    def encode_passage(self, texts):
        vectors = super().encode_passage(texts)
        if self.delta < 0:
            return vectors[: self.delta]
        return np.concatenate([vectors, vectors[: self.delta]])


@pytest.mark.parametrize("delta", [-1, 1])
def test_an_embedder_returning_another_number_of_vectors_is_an_error_and_nothing_is_written(
    setup, monkeypatch, delta
):
    stub_chunker(monkeypatch, THREE)
    setup.embedder = WrongCountEmbedder(delta)
    with pytest.raises(ValueError, match=rf"{3 + delta} vectors for 3 chunks") as raised:
        setup.ingest()
    assert DOC_ID in str(raised.value)
    assert setup.client.calls == []


def test_an_embedder_error_propagates_unchanged_and_nothing_is_written(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    setup.embedder.release()
    with pytest.raises(RuntimeError, match="released"):
        setup.ingest()
    assert setup.client.calls == []


def test_an_embedder_returning_plain_lists_is_accepted(setup, monkeypatch):
    stub_chunker(monkeypatch, THREE)

    class ListEmbedder:
        def encode_passage(self, texts):
            return [[float(index), 0.5] for index, _ in enumerate(texts)]

    setup.embedder = ListEmbedder()
    setup.ingest()
    assert [point.vector for point in setup.client.written] == [[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]]


def test_an_unsupported_format_propagates_unchanged(setup, monkeypatch):
    error = UnsupportedFormatError("format '.xyz' is not supported")
    stub_chunker(monkeypatch, error)
    with pytest.raises(UnsupportedFormatError) as raised:
        setup.ingest("archive.xyz")
    assert raised.value is error
    assert setup.untouched


def test_a_parser_error_propagates_unchanged(setup, monkeypatch):
    error = OSError("cannot open broken document")
    stub_chunker(monkeypatch, error)
    with pytest.raises(OSError) as raised:
        setup.ingest()
    assert raised.value is error
    assert setup.untouched


def test_the_collection_is_never_created_or_checked(setup, monkeypatch):
    # RecordingClient has nothing but ``upsert``: any other call to it would fail.
    stub_chunker(monkeypatch, THREE)
    setup.ingest()
    assert len(setup.client.calls) == 1


def test_the_real_chunker_refuses_an_unsupported_extension_without_the_parsing_libraries(setup, tmp_path):
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        setup.ingest(path)
    assert setup.untouched


def test_the_real_chunker_refuses_an_unknown_ocr_mode(setup, tmp_path):
    with pytest.raises(ValueError, match="ocr"):
        setup.ingest(tmp_path / "report.pdf", ocr="sometimes")
    assert setup.untouched


# ---------------------------------------------------------------------------
# Qdrant local mode: what a real engine stores
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_store():
    client = QdrantClient(":memory:")
    yield QdrantStore(client=client, settings=Settings(batch_size=2))
    client.close()


def test_end_to_end_in_qdrant_local_mode_and_a_second_ingestion_overwrites_in_place(
    memory_store, monkeypatch
):
    chunks = [Chunk(f"chunk number {i}", page) for i, page in enumerate([0, 0, 1, 1, 3])]
    stub_chunker(monkeypatch, chunks)
    embedder = FakeEmbedder(SPEC)
    memory_store.ensure_collection(COLLECTION, SPEC.dim)

    def run(when: str) -> IngestResult:
        return ingest_document(
            "report.pdf",
            DOC_ID,
            METADATA,
            collection=COLLECTION,
            store=memory_store,
            embedder=embedder,
            ingestion_date=when,
        )

    assert run(WHEN) == IngestResult(chunks=5, pages=3, empty=False)
    assert memory_store.count(COLLECTION) == 5

    later = "2026-02-03T04:05:06"
    assert run(later) == IngestResult(chunks=5, pages=3, empty=False)
    assert memory_store.count(COLLECTION) == 5

    stored, _ = memory_store.client.scroll(COLLECTION, limit=100, with_payload=True, with_vectors=True)
    assert {record.id for record in stored} == {
        point_id(DOC_ID, chunk.page, index) for index, chunk in enumerate(chunks)
    }
    by_number = {record.payload["chunk_number"]: record for record in stored}
    assert by_number[4].payload == {
        "doc_id": DOC_ID,
        **METADATA,
        "filename": "report.pdf",
        "chunk_number": 4,
        "ingestion_date": later,
        "text": "chunk number 4",
        "page": 3,
    }
    assert {record.payload["ingestion_date"] for record in stored} == {later}
    np.testing.assert_allclose(
        by_number[4].vector, embedder.encode_passage(["chunk number 4"])[0], rtol=0, atol=1e-6
    )


def test_another_document_adds_its_own_points(memory_store, monkeypatch):
    stub_chunker(monkeypatch, [Chunk("same text", None), Chunk("same text", None)])
    embedder = FakeEmbedder(SPEC)
    memory_store.ensure_collection(COLLECTION, SPEC.dim)
    for doc_id in ("doc-a", "doc-b"):
        ingest_document("notes.md", doc_id, {}, collection=COLLECTION, store=memory_store, embedder=embedder)
    assert memory_store.count(COLLECTION) == 4


def test_a_missing_collection_is_an_error_not_a_creation(memory_store, monkeypatch):
    stub_chunker(monkeypatch, THREE)
    with pytest.raises(Exception, match="not found"):
        ingest_document(
            "report.pdf", DOC_ID, {}, collection="absent", store=memory_store, embedder=FakeEmbedder(SPEC)
        )
    assert not memory_store.client.collection_exists("absent")


# ---------------------------------------------------------------------------
# Layer 2: the real libraries, against the points of the previous pipeline
# ---------------------------------------------------------------------------

GOLDEN = json.loads((GOLDEN_DIR / "expected.json").read_text(encoding="utf-8"))
GOLDEN_DOCUMENTS = {document["file"]: document for document in GOLDEN["documents"]}


def golden_run(document: dict, **kwargs) -> tuple[IngestResult, list]:
    client = RecordingClient()
    embedder = FakeEmbedder(ModelSpec(**GOLDEN["embedder"]["model"]), seed=GOLDEN["embedder"]["seed"])
    result = ingest_document(
        GOLDEN_DIR / document["file"],
        document["doc_id"],
        document["metadata"],
        collection=document["collection"],
        store=QdrantStore(client=client, settings=Settings()),
        embedder=embedder,
        ocr=GOLDEN["ocr"],
        **kwargs,
    )
    assert {name for name, _ in client.calls} == {document["collection"]}
    return result, client.written


def test_the_golden_fixture_is_the_one_described():
    assert set(GOLDEN_DOCUMENTS) == {"monetary-policy-statement.md", "inflation-report-extract.pdf"}
    markdown = GOLDEN_DOCUMENTS["monetary-policy-statement.md"]["points"]
    pdf = GOLDEN_DOCUMENTS["inflation-report-extract.pdf"]["points"]
    assert len(markdown) == 5 and all("page" not in point["payload"] for point in markdown)
    assert [point["payload"]["page"] for point in pdf] == [0, 1, 2]
    # Accented text: ChunkNorris decomposes it (NFKD), so the chunk holds "e" + U+0301 where
    # the file and the title of the metadata hold U+00E9. The points in Qdrant are like this.
    assert all("e\u0301" in point["payload"]["text"] for point in markdown)
    assert all("\u00e9" in point["payload"]["title"] for point in markdown)
    assert all("\u00e9" not in point["payload"]["text"] for point in markdown)
    for document in GOLDEN_DOCUMENTS.values():
        dates = {point["payload"]["ingestion_date"] for point in document["points"]}
        assert dates == {document["ingestion_date"]}


@pytest.mark.parametrize("name", sorted(GOLDEN_DOCUMENTS))
def test_the_points_are_exactly_the_ones_of_the_previous_pipeline(name):
    pytest.importorskip("chunknorris")
    document = GOLDEN_DOCUMENTS[name]
    expected = document["points"]

    result, points = golden_run(document, ingestion_date=document["ingestion_date"])

    assert len(points) == len(expected)
    assert [point.id for point in points] == [point["id"] for point in expected]
    for point, golden in zip(points, expected):
        assert point.payload == golden["payload"]
        assert list(point.payload) == list(golden["payload"])  # same key order too
        assert point.vector == golden["vector"]
    # Through JSON as well: what is sent to Qdrant is the same document, types included.
    assert json.loads(json.dumps([point.payload for point in points])) == [p["payload"] for p in expected]

    pages = {point["payload"]["page"] for point in expected if "page" in point["payload"]}
    assert result == IngestResult(chunks=len(expected), pages=len(pages), empty=False)


@pytest.mark.parametrize("name", sorted(GOLDEN_DOCUMENTS))
def test_every_id_is_the_point_id_of_its_document_page_and_chunk_number(name):
    pytest.importorskip("chunknorris")
    document = GOLDEN_DOCUMENTS[name]
    _, points = golden_run(document)
    assert points
    for index, point in enumerate(points):
        assert point.payload["chunk_number"] == index
        assert point.id == point_id(document["doc_id"], point.payload.get("page") or 0, index)
    # The same holds for the captured points, whatever the chunker did.
    for golden in document["points"]:
        payload = golden["payload"]
        assert golden["id"] == point_id(payload["doc_id"], payload.get("page") or 0, payload["chunk_number"])


@pytest.mark.parametrize("name", sorted(GOLDEN_DOCUMENTS))
def test_without_ingestion_date_a_real_document_gets_one_iso_date_for_all_its_points(name):
    pytest.importorskip("chunknorris")
    document = GOLDEN_DOCUMENTS[name]
    _, points = golden_run(document)
    dates = {point.payload["ingestion_date"] for point in points}
    assert len(dates) == 1
    when = dates.pop()
    assert when != document["ingestion_date"]
    assert isinstance(datetime.datetime.fromisoformat(when), datetime.datetime)
    # Everything but the date is still what the previous pipeline wrote.
    for point, golden in zip(points, document["points"]):
        assert {**point.payload, "ingestion_date": document["ingestion_date"]} == golden["payload"]


@pytest.mark.parametrize("content", ["", "   \n\n\t\n"], ids=["empty", "blank"])
@pytest.mark.parametrize("extension", [".md", ".txt"])
def test_an_empty_or_blank_file_is_an_empty_document(tmp_path, content, extension):
    pytest.importorskip("chunknorris")
    path = tmp_path / f"nothing{extension}"
    path.write_text(content, encoding="utf-8")
    setup = Setup()
    assert setup.ingest(path) == IngestResult(chunks=0, pages=0, empty=True)
    assert setup.untouched


def test_the_golden_default_ocr_mode_gives_the_same_points_on_a_host_without_ocr(monkeypatch):
    pytest.importorskip("chunknorris")
    monkeypatch.delenv("ENABLE_OCR", raising=False)
    document = GOLDEN_DOCUMENTS["inflation-report-extract.pdf"]
    client = RecordingClient()
    ingest_document(
        GOLDEN_DIR / document["file"],
        document["doc_id"],
        document["metadata"],
        collection=document["collection"],
        store=QdrantStore(client=client, settings=Settings()),
        embedder=FakeEmbedder(ModelSpec(**GOLDEN["embedder"]["model"]), seed=GOLDEN["embedder"]["seed"]),
        ingestion_date=document["ingestion_date"],
    )
    assert [point.id for point in client.written] == [point["id"] for point in document["points"]]
    assert [point.payload for point in client.written] == [point["payload"] for point in document["points"]]
