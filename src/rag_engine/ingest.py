"""Ingest one document into Qdrant: chunk the file, embed the chunks, write one point per chunk.

The points are the ones the MyOpenFund data-orchestrator wrote before this function existed:
same ids, same payloads, same vectors for the same chunks (``tests/fixtures/golden/`` pins it).
Ingesting a document again therefore overwrites its points in place.

Importing this module loads no document parser and no embedding backend: the parsing
libraries are imported by :func:`rag_engine.chunking.chunk_file` when a file is read.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import models

from .chunking import ChunkingOptions, check_ocr_languages, chunk_file
from .embedding import Embedder
from .schema import build_payload, page_for_id, point_id
from .store import QdrantStore

__all__ = ["IngestResult", "ingest_document"]


@dataclass(frozen=True)
class IngestResult:
    """What one call of :func:`ingest_document` did.

    ``chunks``
        Number of chunks of the document, which is the number of points written.
    ``pages``
        Number of distinct pages the chunks start on. 0 for a format without pages (markdown,
        plain text, ...). A page no chunk starts on is not counted, so this can be lower than
        the number of pages of the file.
    ``empty``
        True when the document gave no chunk: nothing was embedded and nothing was written.
    """

    chunks: int
    pages: int
    empty: bool


def ingest_document(
    path: str | os.PathLike[str],
    doc_id: str,
    metadata: Mapping[str, Any],
    *,
    collection: str,
    store: QdrantStore,
    embedder: Embedder,
    ocr: str = "auto",
    ocr_languages: str = "eng",
    batch_size: int | None = None,
    chunking_options: ChunkingOptions | None = None,
    ingestion_date: str | None = None,
) -> IngestResult:
    """Chunk, embed and write the document at ``path`` into ``collection``.

    One point is written per chunk, in document order:

    - id: ``point_id(doc_id, page, i)``, ``i`` being the 0-based position of the chunk and
      ``page`` the page it starts on (0 for a chunk without page). The ids depend on nothing
      else, so a second ingestion of the same document overwrites its points;
    - payload: schema v1 (:mod:`rag_engine.schema`): ``doc_id``, the ``metadata`` as given,
      then ``filename`` (base name of ``path``), ``chunk_number`` (``i``), ``ingestion_date``,
      ``text``, and ``page`` when the chunk has one;
    - vector: the embedding of the chunk text as a passage.

    All the points of the document have been written when the function returns. A document
    with no chunk (an empty file, a scanned PDF read without OCR) is not an error: the result
    is ``IngestResult(0, 0, True)``, and neither the embedder nor the store is called.

    **This function reads no settings and no environment variable.** Every option has a fixed
    default written in the signature; a caller that keeps its configuration in
    :class:`rag_engine.config.Settings` passes the values itself::

        ingest_document(path, doc_id, metadata, collection=..., store=store, embedder=embedder,
                        ocr_languages=settings.ocr_languages)

    In particular ``RAG_ENGINE_OCR_LANGUAGES`` has no effect unless the caller passes
    ``ocr_languages=settings.ocr_languages``: left out, the languages are ``"eng"``. The same
    goes for ``batch_size``: left out, the writes use the batch size of ``store`` (the one of
    the settings the store was built with); pass ``batch_size=...`` only for another one.

    Arguments:

    - ``metadata`` is copied into every payload and is not changed. It may not contain a
      reserved key (``rag_engine.schema.RESERVED_KEYS``), except a ``doc_id`` equal to the
      ``doc_id`` argument, which is accepted.
    - ``collection`` must exist with the dimension of the embedder. **This function does not
      create or check it**: the caller calls ``store.ensure_collection(collection,
      embedder.dim)`` once per run, not once per document.
    - ``ocr`` and ``ocr_languages`` (PDF only) and ``chunking_options`` are those of
      :func:`rag_engine.chunking.chunk_file`. ``ocr_languages`` follows the rule of the
      settings (language codes joined by ``+``), for every format.
    - ``batch_size`` is the number of points per write; ``None`` leaves the store's own. It
      sizes the writes only, not the embedding.
    - ``ingestion_date`` is the timestamp written on every point of the document. ``None``,
      the normal case, means the local time at which the document is ingested
      (``datetime.datetime.now().isoformat()``, read once). A caller gives it to replay an
      ingestion or to test; it is written as given, so it should be an ISO 8601 string.

    All the chunk texts are embedded in one ``encode_passage`` call, and the whole document
    is held in memory until it is written.

    Errors. The arguments are checked before the file is read: ``ValueError`` for a ``doc_id``
    or a ``collection`` that is not a non-blank string, a ``path`` without a file name, a
    ``batch_size`` that is not an integer >= 1, an ``ingestion_date`` that is not a non-blank
    string, an ``ocr_languages`` that is not a list of language codes joined by ``+``, a
    reserved key in ``metadata``; ``TypeError`` for a ``metadata`` that is not a
    mapping. Nothing else is caught: an error of the chunker (``UnsupportedFormatError``, a
    missing file, a parser error), of the embedder or of the store
    (:class:`rag_engine.store.QdrantAuthError` and others, with the number of points written
    before the failure in ``points_written``) is raised unchanged. An embedder that does not
    return one vector per chunk is a ``ValueError``, raised before anything is written.
    A failed write is not undone: the caller ingests the document again.
    """
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"doc_id must be a non-blank string, got {doc_id!r}")
    if not isinstance(collection, str) or not collection.strip():
        raise ValueError(f"collection must be a non-blank string, got {collection!r} (document {doc_id!r})")
    if batch_size is not None and (
        isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
    ):
        raise ValueError(f"batch_size must be an integer >= 1 or None, got {batch_size!r}")
    if ingestion_date is not None and (not isinstance(ingestion_date, str) or not ingestion_date.strip()):
        raise ValueError(
            f"ingestion_date must be a non-blank ISO 8601 string or None, got {ingestion_date!r}"
        )
    check_ocr_languages(ocr_languages)
    filename = Path(path).name
    if not filename:
        raise ValueError(f"path has no file name: {os.fspath(path)!r} (document {doc_id!r})")
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__} (document {doc_id!r})")
    # A reserved key in the metadata is an error of the caller, whatever the document holds:
    # it is found here, on a payload that is thrown away, before the file is read.
    build_payload(doc_id, metadata, filename=filename, chunk_number=0, ingestion_date="", text="", page=None)

    chunks = chunk_file(path, ocr=ocr, ocr_languages=ocr_languages, options=chunking_options)
    if not chunks:
        return IngestResult(chunks=0, pages=0, empty=True)

    vectors = embedder.encode_passage([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError(
            f"the embedder returned {len(vectors)} vectors for {len(chunks)} chunks "
            f"(document {doc_id!r}): nothing was written"
        )

    # One date for the whole document, not one per point or per batch.
    when = ingestion_date if ingestion_date is not None else datetime.datetime.now().isoformat()
    points = [
        models.PointStruct(
            id=point_id(doc_id, page_for_id(chunk.page), index),
            vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
            payload=build_payload(
                doc_id,
                metadata,
                filename=filename,
                chunk_number=index,
                ingestion_date=when,
                text=chunk.text,
                page=chunk.page,
            ),
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    store.upsert(collection, points, batch_size)

    pages = {chunk.page for chunk in chunks if chunk.page is not None}
    return IngestResult(chunks=len(points), pages=len(pages), empty=False)
