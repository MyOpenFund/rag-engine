"""``QdrantStore``: the connection to Qdrant, strict collection checks and batched writes.

The store never guesses. A collection is created only when Qdrant answers that it does not
exist; a refused API key is reported as such; every other failure (connection refused,
timeout, server error) is raised as the client raised it.

This module imports ``qdrant-client`` and nothing else outside the engine. qdrant-client
itself imports ``fastembed`` when that package is installed (the ``onnx`` extra), which loads
the ONNX runtime, ``huggingface_hub`` and ``tokenizers``: in such an install, importing the
store loads them, although the engine does not ask for them.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from qdrant_client import QdrantClient, models

from rag_engine.config import Settings
from rag_engine.errors import RagEngineError
from rag_engine.schema import DATE_KEYS, DERIVED_KEYS, PAYLOAD_INDEXES, RESERVED_KEYS, date_interval_fields

__all__ = [
    "AUTH_STATUSES",
    "DEFAULT_TIMEOUT",
    "DEFAULT_URL_PORT",
    "DISTANCE",
    "CollectionMismatchError",
    "CollectionNotFoundError",
    "DateFieldsError",
    "IndexLocked",
    "MissingIndexError",
    "QdrantAuthError",
    "QdrantStore",
    "ReservedKeyError",
    "StoreError",
]

logger = logging.getLogger(__name__)

AUTH_STATUSES = (401, 403)
DISTANCE = models.Distance.COSINE
#: Seconds a request may take when nothing else is said: the default of ``Settings.qdrant_timeout``.
DEFAULT_TIMEOUT = 60
DEFAULT_URL_PORT = 6333


class StoreError(RagEngineError):
    """The store cannot do what was asked, for a reason it can name."""


class QdrantAuthError(StoreError):
    """Qdrant refused the request for lack of a valid API key (HTTP 401 or 403)."""


class CollectionMismatchError(StoreError):
    """An existing collection is not what the caller needs.

    Its vector configuration differs, or one of its payload fields is indexed with another
    kind of index.
    """


class CollectionNotFoundError(StoreError):
    """The collection asked about does not exist on the server.

    Raised by :meth:`QdrantStore.check_collection`, the check that never creates. The other
    methods of the store do not ask first: on a missing collection they raise what the
    Qdrant client raises.
    """


class ReservedKeyError(StoreError, ValueError):
    """:meth:`QdrantStore.set_payload` was asked to set a payload key the engine owns.

    The keys of :data:`rag_engine.schema.RESERVED_KEYS` say which chunk a point is
    (``doc_id``, ``chunk_number``, ``page``), hold its text and say which file and which
    ingestion it comes from: they change by ingesting the document again, never by a payload
    update. A ``ValueError`` as well, like every other bad argument of the store.
    """


class DateFieldsError(StoreError, ValueError):
    """:meth:`QdrantStore.set_payload` was given date keys that do not come together or do not agree.

    ``date``, ``date_precision``, ``date_start`` and ``date_end``
    (:data:`rag_engine.schema.DATE_KEYS`) are all read by a search. Written apart, they leave
    a document found under a date it no longer has: a new ``date`` with the old interval, a
    new interval with the old ``date``, a withdrawn date whose interval stays. A call that
    sets one of them must set the four, as :func:`rag_engine.schema.date_fields` builds
    them. A ``ValueError`` as well, like every other bad argument of the store.
    """


def _check_date_fields(doc_id: str, fields: Mapping[str, Any]) -> None:
    """Refuse date keys that are not the four of ``date_fields``, consistent with each other."""
    given = [key for key in DATE_KEYS if key in fields]
    if not given:
        return
    remedy = (
        "build the four keys with rag_engine.schema.date_fields(date, date_precision) and pass "
        "them together: a search reads all of them"
    )
    missing = [key for key in DATE_KEYS if key not in fields]
    if missing:
        raise DateFieldsError(
            f"set_payload was asked to set {', '.join(given)} without {', '.join(missing)} "
            f"(document {doc_id!r}): the points would keep values that no longer agree; {remedy}"
        )
    expected = date_interval_fields(fields["date"], fields["date_precision"])
    wrong = [key for key in DERIVED_KEYS if fields[key] != expected[key]]
    if wrong:
        raise DateFieldsError(
            f"set_payload was given {', '.join(wrong)} that are not the interval of the date "
            f"{fields['date']!r} with the precision {fields['date_precision']!r} (document {doc_id!r}; "
            f"expected {', '.join(f'{key}={expected[key]!r}' for key in wrong)}); {remedy}"
        )


class IndexLocked(StoreError):
    """A Qdrant local folder is already open, by another process or another client of this one.

    qdrant-client's local mode locks its folder for the lifetime of the client that opened it:
    one index folder serves one client at a time. Close the other client (or stop the process
    that holds it), or use a Qdrant server for concurrent access.
    """


#: What qdrant-client's local mode says when its folder is locked (a ``RuntimeError``).
_LOCKED_MESSAGE = "already accessed by another instance"


class MissingIndexError(StoreError):
    """The collection refuses a filter because the filtered payload field has no index.

    A collection in Qdrant's strict mode (``unindexed_filtering_retrieve`` or
    ``unindexed_filtering_update`` off) answers HTTP 400 to a filter on a field without a
    payload index, where another collection reads every point. The message names the
    collection, the field and the kind of index, and the remedy:
    :meth:`QdrantStore.ensure_payload_indexes`. The error of the client is its ``__cause__``.
    """


# The answer of Qdrant (checked on 1.20), as it is in the JSON body: the quotes are escaped there.
_INDEX_REQUIRED = re.compile(
    r"Index required but not found for \\*\"(?P<field>[^\"\\]+)\\*\" of one of the following types: "
    r"\[(?P<kinds>[^\]]*)\]"
)


def _raise_missing_index(exc: BaseException, collection: str) -> None:
    """Raise :class:`MissingIndexError` from ``exc`` when it is the refusal of an unindexed filter."""
    if getattr(exc, "status_code", None) != 400:
        return
    content = getattr(exc, "content", b"")
    text = content.decode("utf-8", "replace") if isinstance(content, bytes) else str(content)
    found = _INDEX_REQUIRED.search(text)
    if found is None:
        return
    field, kinds = found["field"], found["kinds"].strip() or "payload"
    raise MissingIndexError(
        f"collection {collection!r} refuses a filter on {field!r}: the field has no {kinds} payload "
        f"index and the collection is in strict mode (HTTP 400). Create the indexes of the engine with "
        f"QdrantStore.ensure_payload_indexes({collection!r}), which needs a key allowed to write, then "
        "try again"
    ) from exc


def _raise_auth(exc: BaseException) -> None:
    """Raise :class:`QdrantAuthError` from ``exc`` when it is an authentication refusal.

    The HTTP client of qdrant-client raises ``UnexpectedResponse`` for every non-2xx answer,
    with the HTTP status in ``status_code``. The message never holds the key.
    """
    status = getattr(exc, "status_code", None)
    if status in AUTH_STATUSES:
        raise QdrantAuthError(f"Qdrant refused the request (HTTP {status}): check QDRANT_API_KEY") from exc


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value < 1:  # a bool is an int subclass, and is not a count
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def _now() -> float:
    """Seconds on a clock that never goes back: the age of a cached answer is measured on it."""
    return time.monotonic()


def _seconds(name: str, value: object) -> float:
    try:
        usable = type(value) in (int, float) and math.isfinite(value) and value >= 0  # type: ignore[arg-type]
    except OverflowError:  # an integer too large to be a float
        usable = False
    if not usable:
        raise ValueError(f"{name} must be zero or a positive number of seconds, got {value!r}")
    return float(value)  # type: ignore[arg-type]


def _index_kinds(indexes: Iterable[Any]) -> list[tuple[str, models.PayloadSchemaType]]:
    """``indexes`` as ``(field, kind)`` pairs with the kind as Qdrant's enum; ``ValueError`` otherwise."""
    kinds = ", ".join(kind.value for kind in models.PayloadSchemaType)
    wanted: list[tuple[str, models.PayloadSchemaType]] = []
    seen: set[str] = set()
    for entry in indexes:
        if isinstance(entry, str) or not isinstance(entry, Iterable) or len(pair := tuple(entry)) != 2:
            raise ValueError(f"a payload index must be a (field, kind) pair, got {entry!r}")
        field, kind = pair
        if not isinstance(field, str) or not field:
            raise ValueError(f"the field of a payload index must be a non-empty string, got {field!r}")
        if field in seen:
            raise ValueError(f"payload field {field!r} is listed more than once")
        seen.add(field)
        try:
            wanted.append((field, models.PayloadSchemaType(kind)))
        except ValueError:
            raise ValueError(
                f"unknown kind of payload index for field {field!r}: {kind!r} (one of: {kinds})"
            ) from None
    return wanted


def _port(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 65535:  # a bool is an int subclass, not a port
        raise ValueError(f"port must be an integer between 1 and 65535, got {value!r}")
    return value


def _bare_host(host: str) -> str:
    """``host`` without the square brackets of an IPv6 literal (``[::1]`` -> ``::1``)."""
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _client_host(host: str) -> str:
    """``host`` as qdrant-client needs it: an IPv6 literal in square brackets.

    qdrant-client pastes the host into ``scheme://host:port`` as it is; a bare ``::1`` gives
    a URL that cannot be parsed, and every later call fails.
    """
    return f"[{host}]" if ":" in host else host


class QdrantStore:
    """One Qdrant server (or one injected client).

    Connection values are taken, in this order, from the explicit arguments, from ``settings``,
    and from the environment (:meth:`Settings.from_env`, read when the store is built and
    only when neither ``settings`` nor ``client`` is given). A blank ``api_key`` counts as not
    given. A ``port`` given explicitly must be an integer between 1 and 65535, else
    ``ValueError``.

    ``https`` is always passed to the client: left undecided, qdrant-client turns TLS on as
    soon as an API key is set.

    ``timeout`` is the number of seconds a request may take. Given explicitly, it wins;
    left out (``None``), it is the ``qdrant_timeout`` of the settings
    (``RAG_ENGINE_QDRANT_TIMEOUT``, default :data:`DEFAULT_TIMEOUT`, 60). qdrant-client counts
    in whole seconds and rounds a fraction up. It is not used with an injected client, which
    has its own.

    ``client`` injects a ready client (tests, qdrant-client's local mode); no connection is
    then built, and ``host`` / ``port`` are the ones given explicitly, else ``None``: the store
    cannot know where an injected client points to. With an injected client and no
    ``settings``, the environment is not read at all: there is no connection to configure,
    and the batch size is the built-in default (``Settings.batch_size``).

    An IPv6 address is accepted bare or in square brackets (``::1``, ``[::1]``); ``host``
    shows it bare, and the client receives it in brackets.

    The API key is handed to the client and not kept by the store: the settings are read
    when the store is built and not retained, only the batch size is. ``repr()`` never shows
    the key. The client itself (:attr:`client`) does hold it.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        https: bool | None = None,
        timeout: float | None = None,
        *,
        settings: Settings | None = None,
        client: Any = None,
    ) -> None:
        if port is not None:
            _port(port)
        # What has_date_intervals last read, per collection: (answer, when on the clock of _now).
        self._date_intervals: dict[str, tuple[bool, float]] = {}
        # When uses_date_intervals last failed to read a collection (clock of _now); absent
        # while the last read worked.
        self._date_intervals_failed: dict[str, float] = {}
        self._date_intervals_lock = threading.Lock()
        # One read of a collection's schema at a time: the searches that arrive meanwhile wait
        # for it when nothing is known yet, and use the last answer otherwise.
        self._date_intervals_reading: dict[str, threading.Lock] = {}
        if client is not None:
            # Nothing to connect to: the environment is not read, so a variable that has
            # nothing to do with this store (a bad QDRANT_PORT, say) cannot make it fail.
            self._batch_size = settings.batch_size if settings is not None else Settings.batch_size
            self._host, self._port, self._https = host, port, https
            self._client = client
            return
        # The settings are not kept: they hold the API key.
        settings = settings if settings is not None else Settings.from_env()
        self._batch_size = settings.batch_size
        self._host = _bare_host(host or settings.qdrant_host)
        self._port = port if port is not None else settings.qdrant_port
        self._https = https if https is not None else settings.qdrant_https
        self._client = QdrantClient(
            host=_client_host(self._host),
            port=self._port,
            api_key=(api_key or "").strip() or settings.qdrant_api_key,
            https=self._https,
            timeout=timeout if timeout is not None else settings.qdrant_timeout,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        api_key: str | None = None,
        timeout: float | None = None,
        *,
        settings: Settings | None = None,
    ) -> QdrantStore:
        """Build a store from ``http://host[:port]`` or ``https://host[:port]``.

        The scheme decides TLS, whatever the settings say; the port defaults to 6333. An IPv6
        address is written in square brackets (``http://[::1]:6333``). A URL with a path, a
        query or credentials, with a whitespace or control character anywhere, or with port 0
        is refused with a ``ValueError`` that never echoes the URL. ``api_key``, ``timeout``
        and ``settings`` are as in the constructor.
        """
        problem = "a Qdrant URL must look like http://host:port or https://host:port"
        # Checked on the raw text: urlsplit silently drops tabs and line breaks.
        if any(char.isspace() or not char.isprintable() for char in url):
            raise ValueError(f"{problem}, without whitespace or control characters")
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError:
            raise ValueError(problem) from None
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(problem)
        if parts.path not in ("", "/") or parts.query or parts.fragment or parts.username or parts.password:
            # The URL is not echoed: it may hold credentials.
            raise ValueError(f"{problem}, without path, query or credentials")
        if port == 0:
            raise ValueError(f"{problem}, with a port between 1 and 65535")
        return cls(
            parts.hostname,
            port if port is not None else DEFAULT_URL_PORT,
            api_key,
            parts.scheme == "https",
            timeout,
            settings=settings,
        )

    @classmethod
    def local(cls, path: str | os.PathLike[str]) -> QdrantStore:
        """A store on qdrant-client's local mode, persisted in the folder ``path``.

        No server: the collections live in ``path`` (created when missing) and are read and
        written in this process. Local mode keeps no payload index, so
        :meth:`has_date_intervals` is always ``False`` there; a search over it states its
        ``date_filter_mode`` explicitly instead of ``"auto"``.

        The folder is locked while the client is open. A folder already opened by another
        process, or by another client of this process, raises :class:`IndexLocked`. The
        client is closed with ``store.client.close()``, which releases the lock. The
        environment is not read.
        """
        try:
            client = QdrantClient(path=os.fspath(path))
        except RuntimeError as exc:
            if _LOCKED_MESSAGE in str(exc):
                raise IndexLocked(
                    f"the local index folder {os.fspath(path)!r} is already open in another process or "
                    "client: close it there first (local mode allows one client per folder)"
                ) from exc
            raise
        return cls(client=client)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(host={self._host!r}, port={self._port!r}, https={self._https!r})"

    @property
    def client(self) -> Any:
        """The underlying Qdrant client, for what the store does not wrap."""
        return self._client

    @property
    def host(self) -> str | None:
        return self._host

    @property
    def port(self) -> int | None:
        return self._port

    def is_reachable(self) -> bool:
        """Whether the server answers an authenticated request. Never raises, logs nothing.

        ``False`` covers every failure, a refused API key included. It waits at most the
        timeout the store was built with.
        """
        try:
            self._client.get_collections()
        except Exception:  # noqa: BLE001 - a health check answers, whatever went wrong
            return False
        return True

    def _read_collection(self, name: str) -> Any:
        """What the server says of collection ``name``; ``None`` when it answers that it does not exist."""
        try:
            exists = self._client.collection_exists(collection_name=name)
            return self._client.get_collection(collection_name=name) if exists else None
        except Exception as exc:
            _raise_auth(exc)
            raise

    def check_collection(self, name: str, dim: int) -> None:
        """Check that ``name`` exists with one unnamed cosine vector of size ``dim``. Reads only.

        The check of a reader: a search service calls it once when it starts, with the
        dimension of its embedder, and then knows that its queries can be answered. It
        sends two read requests and **never creates or changes anything**, so it works with
        a read-only API key. :meth:`ensure_collection` is the method of a writer.

        * The collection matches: returns ``None``.
        * Qdrant answers that it does not exist: :class:`CollectionNotFoundError`.
        * It exists and is not a single unnamed vector of size ``dim`` with cosine distance
          (another dimension, another distance, named vectors, no dense vector):
          :class:`CollectionMismatchError`, with the words of :meth:`ensure_collection`.
        * HTTP 401 or 403: :class:`QdrantAuthError`.
        * Anything else (connection refused, timeout, server error, a name the server does
          not accept, a collection deleted between the two requests) is raised as the client
          raised it. No failure is ever read as "the collection is absent".

        ``dim`` must be an integer >= 1, else ``ValueError``, raised before any request.
        """
        _positive_int("dim", dim)
        info = self._read_collection(name)
        if info is None:
            raise CollectionNotFoundError(
                f"collection {name!r} does not exist on the Qdrant server (a writer creates it with "
                "ensure_collection; this check never does)"
            )
        _check_vectors(name, dim, info.config.params.vectors)

    def ensure_collection(self, name: str, dim: int) -> None:
        """Make sure ``name`` exists with one unnamed cosine vector of size ``dim``.

        The method of a writer: it creates the collection when it is missing, which a
        read-only API key is refused (:class:`QdrantAuthError`). A reader calls
        :meth:`check_collection` instead.

        * The collection exists: its vector configuration is checked, and
          :class:`CollectionMismatchError` is raised when it is not a single unnamed vector of
          size ``dim`` with cosine distance.
        * Qdrant answers that it does not exist (and only then): it is created.
        * The creation fails and the collection exists afterwards (another process created it
          meanwhile, or our own request was applied although it reported an error, e.g. a
          timeout): the collection is checked as above. If it still does not exist, or if
          that second look fails, the error of the creation is raised.
        * HTTP 401 or 403: :class:`QdrantAuthError`.
        * Anything else (connection refused, timeout, server error) is raised as the client
          raised it. No failure is ever read as "the collection is absent".

        Existence is asked with the client's ``collection_exists``, which answers a boolean;
        the store does not interpret an error or its message. Qdrant has that question since
        1.8: an older server answers 404, which is raised like any other error.
        """
        _positive_int("dim", dim)
        info = self._read_collection(name)
        if info is not None:
            _check_vectors(name, dim, info.config.params.vectors)
            return
        info = self._create(name, dim)
        if info is not None:
            _check_vectors(name, dim, info.config.params.vectors)
            logger.info(
                "Create of Qdrant collection %r reported an error, but the collection exists and matches",
                name,
            )

    def _create(self, name: str, dim: int) -> Any:
        """Create the collection.

        Returns ``None``, or the collection found after a failed creation: who created it
        (another process, or our own request despite its error) cannot be told.
        """
        logger.info("Creating Qdrant collection %r (dim=%d, %s)", name, dim, DISTANCE.value)
        try:
            self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=dim, distance=DISTANCE),
            )
            return None
        except Exception as exc:  # noqa: BLE001 - re-raised below unless the collection now exists
            _raise_auth(exc)
            created_error = exc
        # Lost race, or a create applied despite its error? Any failure while finding out
        # reports the creation error: it is the one the operator needs.
        try:
            exists = self._client.collection_exists(collection_name=name)
            info = self._client.get_collection(collection_name=name) if exists else None
        except Exception:  # noqa: BLE001 - the creation error is the one to report
            raise created_error from None
        if info is None:
            raise created_error
        return info

    def upsert(self, collection: str, points: Iterable[Any], batch_size: int | None = None) -> int:
        """Write ``points`` in batches of ``batch_size``; returns the number written.

        ``batch_size`` defaults to the ``batch_size`` of the settings the store was built
        with (the built-in default for an injected client without settings); a value that is not an
        integer >= 1 is a ``ValueError``, raised before anything is sent. Each batch waits for
        Qdrant to apply it (``wait=True``). Nothing is sent for no points. HTTP 401 or 403 is
        a :class:`QdrantAuthError`; any other error is raised as is, at the first failed batch.

        A failure does not undo anything: the batches sent before it stay written, and the
        following ones are not sent. The error raised is the one of the failed batch (or the
        one ``points`` raised while being read), with the number of points written before it
        in its ``points_written`` attribute. With deterministic point ids, running the same
        write again overwrites the points already there.
        """
        size = self._batch_size if batch_size is None else _positive_int("batch_size", batch_size)
        written = 0
        batch: list[Any] = []
        try:
            for point in points:
                batch.append(point)
                if len(batch) == size:
                    written += self._send(collection, batch)
                    batch = []
            if batch:
                written += self._send(collection, batch)
        except Exception as exc:
            with contextlib.suppress(AttributeError, TypeError):  # an exception that takes no attribute
                exc.points_written = written  # type: ignore[attr-defined]
            raise
        return written

    def _send(self, collection: str, batch: list[Any]) -> int:
        try:
            self._client.upsert(collection_name=collection, points=batch, wait=True)
        except Exception as exc:
            _raise_auth(exc)
            raise
        return len(batch)

    def ensure_payload_indexes(self, collection: str, indexes: Iterable[Any] | None = None) -> list[str]:
        """Create the payload indexes ``collection`` does not have yet; returns the fields created now.

        ``indexes`` is a list of ``(field, kind)`` pairs, ``kind`` being the name of a Qdrant
        payload index type (``"keyword"``, ``"integer"``, ``"datetime"``, ...) or the matching
        ``models.PayloadSchemaType``; it defaults to :data:`rag_engine.schema.PAYLOAD_INDEXES`.
        A list that is not made of such pairs, or that names a field twice, is a
        ``ValueError`` raised before anything is sent.

        * A field without an index: the index is created, and the call waits until Qdrant
          has built it (``wait=True``), so it is usable when the method returns. The field is
          in the returned list, in the order of ``indexes``.
        * A field already indexed with the same kind: nothing is done, whatever the options
          that index was created with. When every index exists the list is empty, so the
          call can be repeated at will.
        * A field indexed with another kind: :class:`CollectionMismatchError` naming the
          field, the expected kind and the one found, raised before any index is created.
          The existing index is never replaced nor deleted: asked to create it, Qdrant would
          drop it and build the other kind without a word. Removing it is the operator's
          decision.
        * HTTP 401 or 403: :class:`QdrantAuthError`.
        * A collection that does not exist, and any other error, is raised as the client
          raised it; the collection is never created here. Indexes created before a failure
          stay, and a second call creates the remaining ones.

        **The returned list** names the fields for which a create call succeeded during
        this call, and says no more than that. The existing indexes are read once, before
        creating: two callers that run at the same moment may each find the indexes missing,
        each create them, and each return every field (and two that ask for different kinds
        for the same field are not told apart). qdrant-client's local mode keeps no payload
        index: it accepts the creation, warns that it has no effect, and reports none
        afterwards, so there every call returns the whole list.

        Creating an index never rewrites a point, and works on a collection that already
        holds points. A payload value the index cannot read is not an error, neither here nor
        when a point is written: the point stays stored, and a search without a filter on
        that field still returns it, but **a filter on that field no longer sees it**. An
        integer index reads integers (and floats with an integral value): a ``year`` stored
        as the string ``"2016"`` or as ``2016.5`` matches no integer range once indexed. A
        keyword index reads strings: a ``source_code`` stored as a number is not matched. A
        datetime index reads the forms listed with :data:`rag_engine.schema.PAYLOAD_INDEXES`.
        Before creating indexes on a collection that already holds points, check the types
        of the stored values of ``year`` and of the keyword fields.

        To check the result afterwards, count through a filter
        (``client.count(collection, count_filter=..., exact=True)``). Do not rely on
        ``get_collection(...).payload_schema[field].points`` right after the creation on a
        large collection: that count has been seen to lag for a few seconds while the index
        is built.
        """
        wanted = _index_kinds(PAYLOAD_INDEXES if indexes is None else indexes)
        try:
            existing = self._client.get_collection(collection_name=collection).payload_schema or {}
        except Exception as exc:
            _raise_auth(exc)
            raise
        missing: list[tuple[str, models.PayloadSchemaType]] = []
        mismatches: list[str] = []
        for field, kind in wanted:
            if field not in existing:
                missing.append((field, kind))
                continue
            data_type = getattr(existing[field], "data_type", None)
            found = getattr(data_type, "value", data_type)
            if found != kind.value:
                mismatches.append(f"field {field!r}: expected a {kind.value} index, found a {found} index")
        if mismatches:
            raise CollectionMismatchError(
                f"collection {collection!r}: {'; '.join(mismatches)} (an existing index is never replaced)"
            )
        created: list[str] = []
        for field, kind in missing:
            if field in DERIVED_KEYS:
                # What has_date_intervals remembers of this collection is about to be out of date.
                with self._date_intervals_lock:
                    self._date_intervals.pop(collection, None)
            logger.info(
                "Creating %s payload index on %r in Qdrant collection %r", kind.value, field, collection
            )
            try:
                self._client.create_payload_index(
                    collection_name=collection, field_name=field, field_schema=kind, wait=True
                )
            except Exception as exc:
                _raise_auth(exc)
                raise
            created.append(field)
        return created

    def has_date_intervals(self, collection: str, *, max_age: float | None = None) -> bool:
        """Whether ``collection`` has a datetime index on both ``date_start`` and ``date_end``.

        This is how a search in ``date_filter_mode="auto"`` decides between the interval
        filter and the filter on ``date`` (:func:`rag_engine.search.search`). The two indexes
        are created by :meth:`ensure_payload_indexes` since payload schema v2; a collection
        written before it, and never upgraded, has neither.

        It reads the payload schema of the collection (one ``get_collection`` request, which
        a read-only API key may send) and **keeps the answer for** ``max_age`` **seconds**,
        per store and per collection. Within that time it answers from memory, without a
        request. ``max_age`` defaults to ``Settings.date_filter_refresh`` as the class
        defines it (300); ``0`` asks every time; anything but zero or a positive, finite
        number is a ``ValueError``. An answer older than the ``max_age`` of the call is
        asked again, whatever ``max_age`` it was stored under.

        A collection whose indexes are created while a long-lived process holds this store
        is therefore seen at most ``max_age`` seconds late, without a restart. When the same
        store creates them (:meth:`ensure_payload_indexes`), the remembered answer is
        dropped at once.

        The answer says that the **indexes** exist, not that every point carries the two
        fields: the indexes may be created before the points written under schema v1 are
        given their interval. The interval filter of :func:`rag_engine.search.search` is
        built for that case.

        Each time the answer for a collection is first known, or changes, one line is logged
        at INFO level. HTTP 401 or 403 is a :class:`QdrantAuthError`; a collection that does
        not exist, and any other error, is raised as the client raised it, and nothing is
        remembered. qdrant-client's local mode keeps no payload index: there the answer is
        always ``False``.
        """
        age = Settings.date_filter_refresh if max_age is None else _seconds("max_age", max_age)
        with self._date_intervals_lock:
            known = self._date_intervals.get(collection)
        if known is not None and 0 <= _now() - known[1] < age:
            return known[0]
        try:
            schema = self._client.get_collection(collection_name=collection).payload_schema or {}
        except Exception as exc:
            _raise_auth(exc)
            raise
        kinds = {field: getattr(schema.get(field), "data_type", None) for field in DERIVED_KEYS}
        answer = all(getattr(kind, "value", kind) == "datetime" for kind in kinds.values())
        with self._date_intervals_lock:
            self._date_intervals[collection] = (answer, _now())
        if known is None or known[0] != answer:
            if answer:
                logger.info(
                    "Qdrant collection %r has datetime indexes on date_start and date_end: "
                    "date filters in auto mode use the intervals",
                    collection,
                )
            else:
                logger.info(
                    "Qdrant collection %r has no datetime index on both date_start and date_end: "
                    "date filters in auto mode use the date field (legacy)",
                    collection,
                )
        return answer

    def uses_date_intervals(
        self, collection: str, *, max_age: float | None = None, retry_after: float | None = None
    ) -> bool:
        """:meth:`has_date_intervals` for a search: the same answer, and never a failure of its own.

        This is what a search in ``date_filter_mode="auto"`` calls. While the collection can
        be read it is :meth:`has_date_intervals`, with the same ``max_age``. **When the read
        fails or times out, it does not raise**: a search that could be answered must not
        fail because a question asked on its behalf did. It answers instead

        * the last answer this store had for the collection, however old, when it has one;
        * ``False``, the filter on ``date``, when it never read the collection: the filter
          every collection answers, and the one that was sent before the intervals existed.

        It then leaves the collection alone for ``retry_after`` seconds: within that time
        every call gives the same answer **without a request**, and the first call after it
        tries the read again. ``retry_after`` defaults to ``Settings.date_filter_retry`` as
        the class defines it (30); ``0`` tries at every call. Both durations must be zero or
        a positive, finite number, else ``ValueError``.

        One line is logged at WARNING level when the collection stops being readable, with
        the error and the answer given meanwhile, and one at INFO level when it can be read
        again: once per change, not once per search nor per retry.

        Every error of the read is treated this way, a refused API key (HTTP 401 or 403)
        included: the search that follows is sent with the same key and raises the
        :class:`QdrantAuthError` itself. A caller that wants the error calls
        :meth:`has_date_intervals`.
        """
        retry = Settings.date_filter_retry if retry_after is None else _seconds("retry_after", retry_after)
        if max_age is not None:
            _seconds("max_age", max_age)
        with self._date_intervals_lock:
            known = self._date_intervals.get(collection)
            failed_at = self._date_intervals_failed.get(collection)
            reading = self._date_intervals_reading.setdefault(collection, threading.Lock())
        fallback = known[0] if known is not None else False
        if failed_at is not None and 0 <= _now() - failed_at < retry:
            return fallback
        # One thread reads; with an answer or a failure already on record the others do not wait
        # behind a read that may hang for a whole request timeout.
        if not reading.acquire(blocking=known is None and failed_at is None):
            return fallback
        try:
            with self._date_intervals_lock:  # what the read before this one left
                known = self._date_intervals.get(collection)
                failed_at = self._date_intervals_failed.get(collection)
            fallback = known[0] if known is not None else False
            if failed_at is not None and 0 <= _now() - failed_at < retry:
                return fallback
            try:
                answer = self.has_date_intervals(collection, max_age=max_age)
            except Exception as exc:  # noqa: BLE001 - the search that follows reports what is really wrong
                with self._date_intervals_lock:
                    self._date_intervals_failed[collection] = _now()
                if failed_at is None:
                    logger.warning(
                        "The payload schema of Qdrant collection %r could not be read (%s: %s): date filters "
                        "in auto mode use %s until it can; next try in %g seconds at the earliest",
                        collection,
                        type(exc).__name__,
                        exc,
                        f"the last known answer ({'the intervals' if fallback else 'the date field, legacy'})"
                        if known is not None
                        else "the date field (legacy), no answer being known",
                        retry,
                    )
                return fallback
            if failed_at is not None:
                with self._date_intervals_lock:
                    self._date_intervals_failed.pop(collection, None)
                logger.info("The payload schema of Qdrant collection %r can be read again", collection)
            return answer
        finally:
            reading.release()

    def query(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        query_filter: models.Filter | None = None,
        limit: int,
    ) -> list[Any]:
        """The ``limit`` points of ``collection`` closest to ``vector``, best first.

        One request: a dense search on the unnamed vector, restricted to the points that pass
        ``query_filter`` when one is given. The points come with their payload and their
        score, without their vector. ``limit`` must be an integer >= 1, else ``ValueError``.

        HTTP 401 or 403 is a :class:`QdrantAuthError`. A collection in strict mode that
        refuses the filter because a filtered field has no index (HTTP 400) is a
        :class:`MissingIndexError` naming the field and :meth:`ensure_payload_indexes`. A
        collection that does not exist, a vector of another dimension, and any other error
        is raised as the client raised it.
        Nothing is checked beforehand: this is one request, not two.

        It uses the query API (``query_points``): qdrant-client 1.10 or later, and a server
        of 1.10 or later.
        """
        _positive_int("limit", limit)
        try:
            response = self._client.query_points(
                collection_name=collection,
                query=list(vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            _raise_auth(exc)
            _raise_missing_index(exc, collection)
            raise
        return list(response.points)

    def existing_doc_ids(
        self,
        collection: str,
        doc_ids: Iterable[str],
        *,
        batch_size: int = 100,
        page_size: int = 256,
    ) -> set[str]:
        """Which of ``doc_ids`` have at least one point in ``collection``.

        ``doc_ids`` is a list (or tuple, or set) of non-empty strings, compared exactly with
        the payload field ``doc_id``; a bare string or anything else is a ``ValueError``,
        raised before anything is sent. Duplicates are asked once. The answer is the subset
        found, as a new set; it is empty for an empty list, without any request.

        How it asks. The ids are taken ``batch_size`` at a time. For a batch, one request
        reads up to ``page_size`` points whose ``doc_id`` is one of the batch (a filter on the
        keyword field, which the payload index on ``doc_id`` makes fast), with the ``doc_id``
        of their payload only and without vectors. The ids seen are found; the request is
        repeated for the others until it brings nothing new. A batch therefore costs one
        request more than the number of pages that found something: two when every document
        shows up in the first page, and never more than one per document plus one. Points
        come in the order of their ids, which mixes the documents, so a first page of 256
        points usually holds every document of a batch of 100.

        Both sizes must be integers >= 1, else ``ValueError``. HTTP 401 or 403 is a
        :class:`QdrantAuthError`; a collection that does not exist, and any other error, is
        raised as the client raised it.
        """
        if isinstance(doc_ids, (str, bytes)) or not isinstance(doc_ids, (list, tuple, set, frozenset)):
            # A ValueError, as for every other bad argument of the store.
            raise ValueError(f"doc_ids must be a list of strings, got {type(doc_ids).__name__}")  # noqa: TRY004
        for doc_id in doc_ids:
            if not isinstance(doc_id, str):
                raise ValueError(f"doc_ids must hold strings, got {doc_id!r}")  # noqa: TRY004
            if not doc_id:
                raise ValueError("doc_ids must hold non-empty strings, got an empty one")
        _positive_int("batch_size", batch_size)
        _positive_int("page_size", page_size)
        wanted = sorted(set(doc_ids))
        found: set[str] = set()
        for start in range(0, len(wanted), batch_size):
            remaining = set(wanted[start : start + batch_size])
            while remaining:
                condition = models.FieldCondition(key="doc_id", match=models.MatchAny(any=sorted(remaining)))
                try:
                    points, _next = self._client.scroll(
                        collection_name=collection,
                        scroll_filter=models.Filter(must=[condition]),
                        limit=page_size,
                        with_payload=["doc_id"],
                        with_vectors=False,
                    )
                except Exception as exc:
                    _raise_auth(exc)
                    _raise_missing_index(exc, collection)
                    raise
                seen = {(point.payload or {}).get("doc_id") for point in points} & remaining
                if not seen:
                    break
                found |= seen
                remaining -= seen
        return found

    def set_payload(self, collection: str, doc_id: str, fields: Mapping[str, Any]) -> int:
        """Set the payload keys ``fields`` on every point of document ``doc_id``; returns how many points.

        For a correction that needs no new embedding: a date, its precision, a title. Every
        point whose payload field ``doc_id`` is exactly ``doc_id`` gets each key of
        ``fields`` with its value, added or replaced. **Nothing else moves**: the other keys
        of the payload stay, the vectors are not sent, read or rewritten, the ids do not
        change, and no point is created. **A value of** ``None`` **removes the key**: Qdrant
        merges ``fields`` into the payload and drops a key set to null (checked on Qdrant
        1.20; a null left in a payload would be read by every filter as a missing key
        anyway). The call waits until Qdrant has applied the write (``wait=True``);
        repeating it changes nothing more.

        It returns the number of points of the document, counted exactly just before the
        write. **A document without any point is not an error**: the answer is 0 and nothing
        is written.

        **A date is changed with** :func:`rag_engine.schema.date_fields`, and in no other
        way. A search reads four keys: ``date``, ``date_precision`` and the interval the
        engine derived from them at ingestion, ``date_start`` and ``date_end``. This method
        is given the keys to change, not the payload, so it cannot derive the interval
        again; it checks instead that a call which sets one of the four sets them all, and
        that the interval is the one of the date and the precision of the same call::

            from rag_engine.schema import date_fields

            store.set_payload(collection, doc_id, {
                **date_fields("2016-07-01", "month", year=2016), "date_kind": "meeting",
            })

        ``date_fields(None)`` withdraws the date: the four keys are removed and the document
        is undated. The same call, with the date and the precision the source already has,
        gives their interval to the points written before payload schema v2. ``year`` is
        not derived and not checked: left out, the points keep theirs.

        **Not while the same document is being ingested.** The count and the write are two
        requests, and an ingestion writes whole payloads: points of the document written
        between the two are counted or not (the returned number may be too low), and a batch
        of an ingestion that started earlier and lands after this write puts the old
        metadata back on its points. Correct a document when no ingestion of it runs.

        Refused, before anything is sent:

        * a key of :data:`rag_engine.schema.RESERVED_KEYS` (``doc_id``, ``filename``,
          ``chunk_number``, ``ingestion_date``, ``text``, ``page``), whatever its value:
          :class:`ReservedKeyError` naming every such key. They say which chunk a point is
          and where it comes from; only a new ingestion changes them;
        * one of ``date``, ``date_precision``, ``date_start``, ``date_end`` without the three
          others, or an interval that is not
          ``date_interval_fields(fields["date"], fields["date_precision"])``:
          :class:`DateFieldsError`, which names what is missing or wrong and
          ``date_fields``;
        * a ``doc_id`` that is not a non-empty string, a ``fields`` that is not a mapping, an
          empty one, a key that is not a non-empty string: ``ValueError``.

        Two requests: a count, then, when there is at least one point, the write. Both go
        through a filter on ``doc_id``, which its payload index makes fast. HTTP 401 or 403
        is a :class:`QdrantAuthError` (the write needs a key allowed to write). A collection
        in strict mode that has no index on ``doc_id`` refuses that filter (HTTP 400): a
        :class:`MissingIndexError` naming the field and :meth:`ensure_payload_indexes`. A
        collection that does not exist, and any other error, is raised as the client raised
        it. A
        document ingested between the two requests is written to as well, and counted or
        not.
        """
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"doc_id must be a non-empty string, got {doc_id!r}")
        if not isinstance(fields, Mapping):
            raise ValueError(  # noqa: TRY004 - a ValueError, as for every other bad argument of the store
                f"fields must be a mapping of payload keys to values, got {type(fields).__name__}"
            )
        if not fields:
            raise ValueError("fields is empty: there is nothing to set")
        for key in fields:
            if not isinstance(key, str) or not key:
                raise ValueError(f"the keys of fields must be non-empty strings, got {key!r}")
        reserved = [key for key in RESERVED_KEYS if key in fields]
        if reserved:
            raise ReservedKeyError(
                f"set_payload may not set reserved payload key(s): {', '.join(reserved)} "
                f"(document {doc_id!r}): they change by ingesting the document again"
            )
        _check_date_fields(doc_id, fields)
        selector = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        try:
            found = self._client.count(collection_name=collection, count_filter=selector, exact=True)
            points = int(found.count)
            if points:
                self._client.set_payload(
                    collection_name=collection, payload=dict(fields), points=selector, wait=True
                )
        except Exception as exc:
            _raise_auth(exc)
            _raise_missing_index(exc, collection)
            raise
        return points

    def count(self, collection: str) -> int:
        """Exact number of points in ``collection``."""
        try:
            return int(self._client.count(collection_name=collection, exact=True).count)
        except Exception as exc:
            _raise_auth(exc)
            raise


def _check_vectors(name: str, dim: int, vectors: Any) -> None:
    """Refuse anything but one unnamed vector of size ``dim`` with the expected distance."""
    expected = f"expected a single unnamed {dim}-d vector with {DISTANCE.value} distance"
    if isinstance(vectors, Mapping) and vectors:
        names = ", ".join(repr(key) for key in vectors)
        raise CollectionMismatchError(f"collection {name!r}: {expected}, found named vectors ({names})")
    size = getattr(vectors, "size", None)
    if size is None:
        raise CollectionMismatchError(f"collection {name!r}: {expected}, found no dense vector")
    if int(size) != dim:
        raise CollectionMismatchError(f"collection {name!r}: {expected}, found {int(size)}-d vectors")
    distance = getattr(vectors, "distance", None)
    if distance != DISTANCE:
        found = getattr(distance, "value", distance)
        raise CollectionMismatchError(f"collection {name!r}: {expected}, found {found} distance")
