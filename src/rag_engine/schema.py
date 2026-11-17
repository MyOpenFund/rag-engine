"""Payload schema v1 and deterministic point ids.

Schema v1 is the payload already written to Qdrant by the MyOpenFund
data-orchestrator. Nothing is renamed or removed and ids are computed the same
way, so existing points stay valid and ingesting a document again overwrites
its points in place.

One point is one chunk of one document. Its payload holds:

Keys written by the engine (``RESERVED_KEYS``; caller metadata may not set them):

- ``doc_id``: identifier of the document, given by the caller.
- ``filename``: base name of the ingested file.
- ``chunk_number``: 0-based position of the chunk in the document. The key is
  ``chunk_number``, not ``chunk_index``.
- ``ingestion_date``: the engine's own timestamp, an ISO 8601 string, identical
  on every point of a document. It says when the point was written, nothing
  about the document itself.
- ``text``: the text of the chunk.
- ``page``: page the chunk starts on. Absent for formats without pages
  (markdown, plain text, ...); never written as null or 0 in that case.

Keys copied from the caller's metadata. The engine does not interpret them at
ingestion; the MyOpenFund vault source provides these today, each one left out
when the vault has no value for it:

- ``corpus``: corpus the document belongs to.
- ``source_code``: code of the source inside the corpus.
- ``doc_type``: kind of document.
- ``title``: title of the document.
- ``date``: publication date of the document, an ISO 8601 string. Not to be
  confused with ``ingestion_date``.
- ``year``: publication year, an integer.
- ``language``: language code.
- ``sha256``: hash of the source file.
- ``provenance``: where the file came from.

Any other metadata key is copied as is.

The point id depends on ``(doc_id, page, chunk_index)`` only, where
``chunk_index`` is the value stored as ``chunk_number`` and ``page`` is 0 when
the chunk has no page (see ``page_for_id``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

__all__ = ["RESERVED_KEYS", "build_payload", "page_for_id", "point_id"]

RESERVED_KEYS: tuple[str, ...] = ("doc_id", "filename", "chunk_number", "ingestion_date", "text", "page")


def point_id(doc_id: str, page: int, chunk_index: int) -> int:
    """Return the Qdrant point id of one chunk: a stable integer in ``[0, 2**63)``.

    The algorithm is frozen: changing it would move the id of every point
    already stored.
    """
    key = f"{doc_id}::{page}::{chunk_index}"
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) >> 1


def page_for_id(page: int | None) -> int:
    """Return the page to use in ``point_id``: 0 for a chunk without a page.

    The payload keeps the real absence instead (``build_payload`` writes no
    ``page`` key for ``None``).
    """
    return 0 if page is None else page


def build_payload(
    doc_id: str,
    metadata: Mapping[str, Any],
    *,
    filename: str,
    chunk_number: int,
    ingestion_date: str,
    text: str,
    page: int | None,
) -> dict[str, Any]:
    """Return the schema v1 payload of one chunk as a new dict.

    ``metadata`` is copied untouched between ``doc_id`` and the engine's own
    keys. It may not contain a reserved key: that raises ``ValueError`` naming
    every offending key. The one exception is ``doc_id`` equal to the
    ``doc_id`` argument, which is accepted.
    """
    clashes = [
        key for key in RESERVED_KEYS if key in metadata and not (key == "doc_id" and metadata[key] == doc_id)
    ]
    if clashes:
        raise ValueError(
            f"metadata may not set reserved payload key(s): {', '.join(clashes)} (document {doc_id!r})"
        )
    payload: dict[str, Any] = {
        "doc_id": doc_id,
        **metadata,
        "filename": filename,
        "chunk_number": chunk_number,
        "ingestion_date": ingestion_date,
        "text": text,
    }
    if page is not None:
        payload["page"] = page
    return payload
