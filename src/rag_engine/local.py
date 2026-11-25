"""An index on the user's own disk, without a Qdrant server, and a hybrid search over it.

:class:`LocalIndex` keeps every chunk twice, in one folder:

- ``<root>/qdrant``: a collection of qdrant-client's **local mode** (:meth:`QdrantStore.local`),
  for the dense side. It is searched with :func:`rag_engine.search.search`, the production
  search, unchanged;
- ``<root>/fts.sqlite3``: an SQLite **FTS5** table ``chunks(doc_id, chunk_id, text, context,
  payload_json)`` over the same chunks, for the lexical side, ranked with ``bm25()``. SQLite
  runs in WAL mode.

:func:`hybrid_search` asks both sides for ``candidates`` chunks under the same
:class:`~rag_engine.search.SearchFilters` and fuses the two ranked lists by Reciprocal Rank
Fusion.

The index embeds nothing: :meth:`LocalIndex.add` receives the vectors, computed by the caller
from ``chunk.embedded_text`` with ``encode_passage``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import models

from .chunking import Chunk
from .embedding import Embedder
from .schema import date_interval, page_for_id, point_id
from .search import _PAYLOAD_FIELDS, QUERY_NORMALIZATION, Hit, SearchError, SearchFilters, search
from .store import QdrantStore

__all__ = [
    "DATE_FILTER_MODE",
    "DEFAULT_CANDIDATES",
    "DEFAULT_RRF_K",
    "LOCAL_KEYS",
    "LocalIndex",
    "collection_name",
    "fts_query",
    "hybrid_search",
]

#: How the dense side applies date bounds. ``"interval"`` is explicit, so ``search`` reads no
#: payload schema (``"auto"`` would ask the store, and local mode keeps no payload index, so it
#: would always answer ``legacy``). ``"interval"`` needs no index in local mode (no strict
#: mode), and every chunk that passes in ``legacy`` passes in ``interval`` too: it is the
#: overlap of ``date_start``..``date_end`` with the range, **or** the legacy condition on
#: ``date``. :meth:`LocalIndex.add` derives the interval of every chunk, so month- and
#: year-precision documents are found by any day of their period. The FTS side applies the
#: same condition in SQL.
DATE_FILTER_MODE = "interval"
#: How many chunks each side returns before the fusion: the default of ``candidates``.
DEFAULT_CANDIDATES = 50
#: The ``k`` of Reciprocal Rank Fusion (``1 / (k + rank)``), the default of ``rrf_k``: 60, the
#: value of Cormack, Clarke and Büttcher (2009).
DEFAULT_RRF_K = 60

#: Payload keys :meth:`LocalIndex.add` writes itself; the given payload may not set them.
LOCAL_KEYS = ("text", "context", "chunk_number", "page", "date_start", "date_end")

_QDRANT_DIR = "qdrant"
_FTS_FILE = "fts.sqlite3"
_SLUG = re.compile(r"[^A-Za-z0-9]+")


# --- names ----------------------------------------------------------------------------------------


def collection_name(model_name: str, model_revision: str, chunker_version: str, provider: str) -> str:
    """The collection a local index uses for one embedding setup.

    Vectors from another model, revision, chunker or execution provider must never be mixed in
    one collection: each combination gets its own. The name is readable (a slug of the model
    name, the chunker version and the provider) and ends with 12 hex characters of a SHA-256
    of the four values, so two setups whose slugs coincide still differ. It holds only
    ``[A-Za-z0-9_-]``, safe as a folder name of local mode. Each value must be a non-blank
    string, else ``ValueError``.
    """
    values = {
        "model_name": model_name,
        "model_revision": model_revision,
        "chunker_version": chunker_version,
        "provider": provider,
    }
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-blank string, got {value!r}")
    digest = hashlib.sha256("\0".join(values.values()).encode("utf-8", "surrogatepass")).hexdigest()[:12]

    def slug(text: str, size: int) -> str:
        return _SLUG.sub("_", text).strip("_")[:size] or "x"

    return f"{slug(model_name, 48)}-{slug(chunker_version, 16)}-{slug(provider, 16)}-{digest}"


def _word_character(character: str) -> bool:
    # Letters, numbers and marks: what FTS5's unicode61 tokenizer keeps (marks are kept so a
    # decomposed letter stays one word; the tokenizer drops the diacritic itself).
    return unicodedata.category(character)[0] in "LNM" or unicodedata.category(character) == "Co"


def fts_query(text: str) -> str:
    """``text`` as an FTS5 MATCH string that cannot be read as FTS5 syntax.

    The words of ``text`` (runs of letters, numbers and combining marks; everything else
    separates them) are each put in double quotes and joined with ``OR``: ``capex* -debt``
    becomes ``"capex" OR "debt"``. Inside quotes, ``AND``, ``NEAR``, ``*``, ``-``, ``:``,
    ``^`` and parentheses are plain text, and a quote cannot occur (it is a separator). A word
    made only of combining marks is dropped; repeated words are kept once. ``OR`` lets bm25()
    rank a chunk by how many of the words it holds, rather than demanding all of them.

    A ``text`` with no word (blank, punctuation, emoji: ``'"()'``, ``'"'``) is a
    ``ValueError``: there is nothing to match.
    """
    if not isinstance(text, str):
        raise TypeError(f"the FTS query must be a string, got {type(text).__name__}")
    words: list[str] = []
    current: list[str] = []
    for character in [*text, " "]:
        if _word_character(character):
            current.append(character)
            continue
        if current:
            word = "".join(current)
            if any(unicodedata.category(c)[0] in "LN" or unicodedata.category(c) == "Co" for c in word):
                words.append(word)
            current = []
    words = list(dict.fromkeys(words))
    if not words:
        raise ValueError(f"the FTS query holds no word to search for: {text!a}")
    return " OR ".join(f'"{word}"' for word in words)


# --- filters as SQL -------------------------------------------------------------------------------

# Keys and paths are constants of this module, never user input: only values are parameters.
_FULL_DATE_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*"


def _json(key: str) -> str:
    return f"json_extract(payload_json, '$.{key}')"


def _instant(key: str) -> str:
    """The UTC instant ``payload[key]`` holds as ``'YYYY-MM-DD HH:MM:SS'``, or NULL when it holds none.

    A text that starts with a full date and that SQLite's ``datetime()`` reads (an offset is
    converted to UTC). The type and GLOB checks keep out what ``datetime()`` would otherwise
    read as a Julian day number (``2016``, a number) or as ``now``.
    """
    return (
        f"(CASE WHEN json_type(payload_json, '$.{key}') = 'text' AND {_json(key)} GLOB '{_FULL_DATE_GLOB}' "
        f"THEN datetime({_json(key)}) END)"
    )


def _midnight(day: datetime.date) -> str:
    return f"{day.isoformat()} 00:00:00"


def _date_sql(filters: SearchFilters) -> tuple[str, list[Any]]:
    """The date bounds of ``filters`` in interval mode: :func:`rag_engine.search._date_condition`, in SQL."""
    params: list[Any] = []
    # Legacy: first <= date < last + 1 day; a readable date when date_to is the last day there is.
    legacy = [f"{_instant('date')} IS NOT NULL"]
    if filters.date_from is not None:
        legacy.append(f"{_instant('date')} >= ?")
        params.append(_midnight(filters.date_from))
    if filters.date_to is not None and filters.date_to < datetime.date.max:
        legacy.append(f"{_instant('date')} < ?")
        params.append(_midnight(filters.date_to + datetime.timedelta(days=1)))
    # Interval: date_start <= last and date_end >= first, both at midnight UTC.
    interval: list[str] = []
    interval_params: list[Any] = []
    if filters.date_to is not None:
        interval.append(f"{_instant('date_start')} <= ?")
        interval_params.append(_midnight(filters.date_to))
    if filters.date_from is not None:
        interval.append(f"{_instant('date_end')} >= ?")
        interval_params.append(_midnight(filters.date_from))
    should = [f"({' AND '.join(interval)})", f"({' AND '.join(legacy)})"]
    params = interval_params + params
    if filters.include_undated:
        should.append(f"({_instant('date_start')} IS NULL AND {_instant('date')} IS NULL)")
    return f"({' OR '.join(should)})", params


def _filter_sql(filters: SearchFilters | None) -> tuple[str, list[Any]]:
    """``filters`` as a SQL condition on the ``chunks`` table, with its parameters."""
    if filters is None:
        return "1", []
    clauses: list[str] = []
    params: list[Any] = []
    if filters.corpus is not None:
        clauses.append(f"{_json('corpus')} = ?")
        params.append(filters.corpus)
    for values, key in (
        (filters.source_codes, "source_code"),
        (filters.doc_types, "doc_type"),
        (filters.languages, "language"),
    ):
        if values is not None:
            clauses.append(f"{_json(key)} IN ({', '.join('?' * len(values))})")
            params.extend(values)
    if filters.doc_ids is not None:
        clauses.append(f"doc_id IN ({', '.join('?' * len(filters.doc_ids))})")
        params.extend(filters.doc_ids)
    if filters.date_from is not None or filters.date_to is not None:
        clause, date_params = _date_sql(filters)
        clauses.append(clause)
        params.extend(date_params)
    return (" AND ".join(clauses) or "1"), params


# --- the index ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    """One chunk as the FTS side returns it, best first."""

    doc_id: str
    chunk_id: int
    text: str
    context: str | None
    payload: dict[str, Any]


class LocalIndex:
    """Qdrant local under ``root/qdrant`` and an FTS5 table in ``root/fts.sqlite3``, written together.

    ``collection`` is the Qdrant collection (see :func:`collection_name`), created with ``dim``
    when missing and checked otherwise (``CollectionMismatchError`` for another dimension).
    ``root`` is created when missing. The folder is locked while the index is open: a second
    open, from this process or another, raises :class:`rag_engine.store.IndexLocked`.
    :meth:`close` (or leaving a ``with`` block) releases it.

    Writes go to Qdrant first, then to SQLite in one transaction: SQLite is the record of what
    is complete (:meth:`doc_ids`, :meth:`count`). A write cut between the two leaves points in
    Qdrant that SQLite does not list; adding the same batch again, or deleting the document,
    repairs it. The methods may be called from several threads.
    """

    def __init__(self, root: Path, *, collection: str, dim: int) -> None:
        self.root = Path(root)
        self.collection = collection
        self.dim = dim
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.store = QdrantStore.local(self.root / _QDRANT_DIR)
        try:
            self.store.ensure_collection(collection, dim)
            self._db = sqlite3.connect(self.root / _FTS_FILE, check_same_thread=False, isolation_level=None)
            try:
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
                    "doc_id UNINDEXED, chunk_id UNINDEXED, text, context, payload_json UNINDEXED, "
                    "tokenize = 'unicode61 remove_diacritics 2')"
                )
            except Exception:
                self._db.close()
                raise
        except Exception:
            self.store.client.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close SQLite and the Qdrant client, which releases the folder. Calling it twice is harmless."""
        with self._lock:
            self._db.close()
            self.store.client.close()

    def add(
        self,
        doc_id: str,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        payload: Mapping[str, Any],
        *,
        first_index: int,
    ) -> int:
        """Write ``chunks`` of ``doc_id`` with their ``vectors``; returns the number written.

        Chunk ``i`` of the batch is chunk number ``first_index + i`` of the document: its point
        id is ``point_id(doc_id, page_for_id(chunk.page), first_index + i)``, the id an
        ingestion into a server gives, so a document added in several batches (priority
        sections first) keeps distinct ids as long as the batches do not overlap. Writing a
        batch again overwrites it, on both sides.

        Each chunk is stored with ``{"doc_id": doc_id, **payload}`` plus ``chunk_number``,
        ``text``, ``context`` (when the chunk has one), ``page`` (when it has one), and
        ``date_start`` / ``date_end`` derived from ``payload["date"]`` and
        ``payload["date_precision"]`` (:func:`rag_engine.schema.date_interval`). FTS indexes
        ``text`` and ``context``.

        ``ValueError``: ``doc_id`` not a non-blank string, ``first_index`` not an integer >= 0,
        ``vectors`` not of shape ``(len(chunks), dim)``, a ``payload`` that sets one of
        :data:`LOCAL_KEYS` or a ``doc_id`` other than ``doc_id``. Nothing is written then.
        """
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"doc_id must be a non-blank string, got {doc_id!r}")
        if type(first_index) is not int or first_index < 0:
            raise ValueError(f"first_index must be an integer >= 0, got {first_index!r}")
        clashes = [key for key in LOCAL_KEYS if key in payload]
        if clashes:
            raise ValueError(
                f"payload may not set {', '.join(clashes)}: the index writes them (document {doc_id!r})"
            )
        if "doc_id" in payload and payload["doc_id"] != doc_id:
            raise ValueError(f"payload doc_id {payload['doc_id']!r} differs from doc_id {doc_id!r}")
        if not chunks:
            return 0
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape != (len(chunks), self.dim):
            raise ValueError(
                f"vectors must have the shape ({len(chunks)}, {self.dim}), got {array.shape} "
                f"(document {doc_id!r})"
            )
        interval = date_interval(payload.get("date"), payload.get("date_precision"))
        points: list[models.PointStruct] = []
        rows: list[tuple[str, int, str, str | None, str]] = []
        for offset, (chunk, vector) in enumerate(zip(chunks, array)):
            number = first_index + offset
            common: dict[str, Any] = {"doc_id": doc_id, **payload, "chunk_number": number}
            if chunk.page is not None:
                common["page"] = chunk.page
            if interval is not None:
                common["date_start"], common["date_end"] = interval
            full = {**common, "text": chunk.text}
            if chunk.context is not None:
                full["context"] = chunk.context
            ident = point_id(doc_id, page_for_id(chunk.page), number)
            points.append(models.PointStruct(id=ident, vector=vector.tolist(), payload=full))
            rows.append((doc_id, ident, chunk.text, chunk.context, json.dumps(common, ensure_ascii=False)))
        with self._lock:
            self.store.upsert(self.collection, points)
            with self._db:
                self._db.execute("BEGIN")
                self._db.execute(
                    "DELETE FROM chunks WHERE doc_id = ? AND chunk_id IN (SELECT value FROM json_each(?))",
                    (doc_id, json.dumps([row[1] for row in rows])),
                )
                self._db.executemany(
                    "INSERT INTO chunks (doc_id, chunk_id, text, context, payload_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
        return len(points)

    def delete(self, doc_id: str) -> None:
        """Remove every chunk of ``doc_id`` from both sides (Qdrant first, then SQLite)."""
        condition = models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        selector = models.FilterSelector(filter=models.Filter(must=[condition]))
        with self._lock:
            self.store.client.delete(collection_name=self.collection, points_selector=selector, wait=True)
            with self._db:
                self._db.execute("BEGIN")
                self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def doc_ids(self) -> set[str]:
        """The documents that have at least one chunk in the index."""
        with self._lock:
            return {row[0] for row in self._db.execute("SELECT DISTINCT doc_id FROM chunks")}

    def count(self, doc_id: str) -> int:
        """The number of chunks of ``doc_id`` in the index (0 for an unknown document)."""
        with self._lock:
            found = self._db.execute("SELECT count(*) FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()
        return int(found[0])

    def _fts_rows(self, match: str, filters: SearchFilters | None, limit: int) -> list[_Row]:
        """The ``limit`` chunks that match ``match`` (an :func:`fts_query`) best by bm25, under filters."""
        where, params = _filter_sql(filters)
        sql = (
            "SELECT doc_id, chunk_id, text, context, payload_json FROM chunks "
            f"WHERE chunks MATCH ? AND {where} ORDER BY bm25(chunks), chunk_id LIMIT ?"
        )
        with self._lock:
            found = self._db.execute(sql, [match, *params, limit]).fetchall()
        return [_Row(row[0], int(row[1]), row[2], row[3], json.loads(row[4])) for row in found]


# --- the hybrid search ----------------------------------------------------------------------------


class _OnceQuery:
    """An embedder that embeds the query once and remembers the vector, for ``search`` and the selector."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self.vector: np.ndarray | None = None

    def encode_query(self, text: str) -> np.ndarray:
        if self.vector is None:
            self.vector = np.asarray(self._embedder.encode_query(text))
        return self.vector


def _count(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise SearchError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def _row_hit(row: _Row, score: float) -> Hit:
    payload = {**row.payload, "text": row.text}
    if row.context is not None:
        payload["context"] = row.context
    fields = {name: payload.get(name) for name in _PAYLOAD_FIELDS}
    return Hit(score=score, point_id=row.chunk_id, payload=payload, **fields)


def hybrid_search(
    query: str,
    *,
    index: LocalIndex,
    embedder: Embedder,
    filters: SearchFilters | None,
    limit: int,
    candidates: int = DEFAULT_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
    selector: Callable[[list[Hit], np.ndarray, int], list[Hit]] | None = None,
) -> list[Hit]:
    """The chunks of ``index`` that best answer ``query``: dense and BM25, fused by RRF.

    - Dense: :func:`rag_engine.search.search` on ``index.store``, ``candidates`` deep, with
      ``date_filter_mode`` set to :data:`DATE_FILTER_MODE` (no payload-schema read).
    - Lexical: the FTS5 table, ``candidates`` deep, ordered by ``bm25()``, on
      :func:`fts_query` of the query put in NFKD (the form chunks are stored in). A query
      without any word (``"???"``) is searched on the dense side alone.
    - ``filters`` (every field of :class:`SearchFilters`) restrict both sides alike.
    - Fusion: each chunk scores ``sum(1 / (rrf_k + rank))`` over the lists it is in (``rank``
      from 1); ``Hit.score`` is that fused score. Ties keep the dense order, then the FTS one.
      A chunk found by FTS alone is rebuilt from SQLite, which holds its whole payload.
    - ``selector``: when given, it receives every fused candidate (best first), the query
      vector and ``limit``, and returns the hits to give back (an MMR, say); otherwise the
      first ``limit`` are returned. The query is embedded once either way.

    ``limit``, ``candidates`` and ``rrf_k`` are integers >= 1, else :class:`SearchError`
    (``limit`` may exceed ``candidates``: the fusion holds up to ``2 * candidates`` chunks).
    ``search`` raises its own :class:`SearchError` for a query it refuses (blank, nothing
    searchable).
    """
    _count("limit", limit)
    _count("candidates", candidates)
    _count("rrf_k", rrf_k)
    once = _OnceQuery(embedder)
    dense = search(
        query,
        collection=index.collection,
        store=index.store,
        embedder=once,  # type: ignore[arg-type]
        filters=filters,
        limit=candidates,
        max_limit=candidates,
        date_filter_mode=DATE_FILTER_MODE,
    )
    try:
        match = fts_query(unicodedata.normalize(QUERY_NORMALIZATION, query))
    except ValueError:
        lexical: list[_Row] = []
    else:
        lexical = index._fts_rows(match, filters, candidates)

    scores: dict[int | str, float] = {}
    hits: dict[int | str, Hit] = {}
    for rank, hit in enumerate(dense, start=1):
        scores[hit.point_id] = scores.get(hit.point_id, 0.0) + 1.0 / (rrf_k + rank)
        hits.setdefault(hit.point_id, hit)
    rows: dict[int | str, _Row] = {}
    for rank, row in enumerate(lexical, start=1):
        scores[row.chunk_id] = scores.get(row.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        rows.setdefault(row.chunk_id, row)
    # dict order is first-seen: dense order, then FTS-only order; sorted() is stable.
    ordered = sorted(scores, key=lambda ident: -scores[ident])
    fused = [
        replace(hits[ident], score=scores[ident]) if ident in hits else _row_hit(rows[ident], scores[ident])
        for ident in ordered
    ]
    if selector is None:
        return fused[:limit]
    vector = once.vector if once.vector is not None else np.asarray(embedder.encode_query(query))
    return selector(fused, vector, limit)
