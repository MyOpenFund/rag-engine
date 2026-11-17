"""``QdrantStore``: the connection to Qdrant, strict collection checks and batched writes.

The store never guesses. A collection is created only when Qdrant answers that it does not
exist; a refused API key is reported as such; every other failure (connection refused,
timeout, server error) is raised as the client raised it.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from qdrant_client import QdrantClient, models

from rag_engine.config import Settings
from rag_engine.errors import RagEngineError
from rag_engine.schema import PAYLOAD_INDEXES

__all__ = [
    "AUTH_STATUSES",
    "DEFAULT_TIMEOUT",
    "DEFAULT_URL_PORT",
    "DISTANCE",
    "CollectionMismatchError",
    "QdrantAuthError",
    "QdrantStore",
    "StoreError",
]

logger = logging.getLogger(__name__)

AUTH_STATUSES = (401, 403)
DISTANCE = models.Distance.COSINE
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
        timeout: int = DEFAULT_TIMEOUT,
        *,
        settings: Settings | None = None,
        client: Any = None,
    ) -> None:
        if port is not None:
            _port(port)
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
            timeout=timeout,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        api_key: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        *,
        settings: Settings | None = None,
    ) -> QdrantStore:
        """Build a store from ``http://host[:port]`` or ``https://host[:port]``.

        The scheme decides TLS, whatever the settings say; the port defaults to 6333. An IPv6
        address is written in square brackets (``http://[::1]:6333``). A URL with a path, a
        query or credentials, with a whitespace or control character anywhere, or with port 0
        is refused with a ``ValueError`` that never echoes the URL. ``api_key`` and
        ``settings`` are as in the constructor.
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

    def ensure_collection(self, name: str, dim: int) -> None:
        """Make sure ``name`` exists with one unnamed cosine vector of size ``dim``.

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
        try:
            exists = self._client.collection_exists(collection_name=name)
            info = self._client.get_collection(collection_name=name) if exists else None
        except Exception as exc:
            _raise_auth(exc)
            raise
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

        HTTP 401 or 403 is a :class:`QdrantAuthError`; a collection that does not exist, a
        vector of another dimension, and any other error is raised as the client raised it.
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
                    raise
                seen = {(point.payload or {}).get("doc_id") for point in points} & remaining
                if not seen:
                    break
                found |= seen
                remaining -= seen
        return found

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
