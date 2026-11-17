"""``QdrantStore``: the connection to Qdrant, strict collection checks and batched writes.

The store never guesses. A collection is created only when Qdrant answers that it does not
exist; a refused API key is reported as such; every other failure (connection refused,
timeout, server error) is raised as the client raised it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

from qdrant_client import QdrantClient, models

from rag_engine.config import Settings

logger = logging.getLogger(__name__)

AUTH_STATUSES = (401, 403)
DISTANCE = models.Distance.COSINE
DEFAULT_TIMEOUT = 60
DEFAULT_URL_PORT = 6333


class StoreError(Exception):
    """The store cannot do what was asked, for a reason it can name."""


class QdrantAuthError(StoreError):
    """Qdrant refused the request for lack of a valid API key (HTTP 401 or 403)."""


class CollectionMismatchError(StoreError):
    """An existing collection does not have the vector configuration the caller needs."""


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


class QdrantStore:
    """One Qdrant server (or one injected client).

    Connection values are taken, in this order, from the explicit arguments, from ``settings``,
    and from the environment (:meth:`Settings.from_env`, read when the store is built and
    only when no ``settings`` is given). A blank ``api_key`` counts as not given.

    ``https`` is always passed to the client: left undecided, qdrant-client turns TLS on as
    soon as an API key is set.

    ``client`` injects a ready client (tests, qdrant-client's local mode); no connection is
    then built, and ``host`` / ``port`` are the ones given explicitly, else ``None``: the store
    cannot know where an injected client points to.

    The API key is handed to the client and not kept by the store; ``repr()`` never shows it.
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
        self._settings = settings if settings is not None else Settings.from_env()
        if client is not None:
            self._host, self._port, self._https = host, port, https
            self._client = client
            return
        self._host = host or self._settings.qdrant_host
        self._port = port if port is not None else self._settings.qdrant_port
        self._https = https if https is not None else self._settings.qdrant_https
        self._client = QdrantClient(
            host=self._host,
            port=self._port,
            api_key=(api_key or "").strip() or self._settings.qdrant_api_key,
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

        The scheme decides TLS, whatever the settings say; the port defaults to 6333. A URL
        with a path, a query or credentials is refused. ``api_key`` and ``settings`` are as
        in the constructor.
        """
        problem = "a Qdrant URL must look like http://host:port or https://host:port"
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
          meanwhile): the collection is checked as above. If it still does not exist, or if
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
        if info is None:
            info = self._create(name, dim)
        if info is not None:
            _check_vectors(name, dim, info.config.params.vectors)

    def _create(self, name: str, dim: int) -> Any:
        """Create the collection. Returns ``None``, or the collection another process created."""
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
        # Lost race? Any failure while finding out reports the creation error: it is the one
        # the operator needs.
        try:
            exists = self._client.collection_exists(collection_name=name)
            info = self._client.get_collection(collection_name=name) if exists else None
        except Exception:  # noqa: BLE001 - the creation error is the one to report
            raise created_error from None
        if info is None:
            raise created_error
        logger.info("Qdrant collection %r was created by another process meanwhile", name)
        return info

    def upsert(self, collection: str, points: Iterable[Any], batch_size: int | None = None) -> int:
        """Write ``points`` in batches of ``batch_size``; returns the number written.

        ``batch_size`` defaults to the ``batch_size`` of the settings; a value that is not an
        integer >= 1 is a ``ValueError``, raised before anything is sent. Each batch waits for
        Qdrant to apply it (``wait=True``). Nothing is sent for no points. HTTP 401 or 403 is
        a :class:`QdrantAuthError`; any other error is raised as is, at the first failed batch.
        """
        size = self._settings.batch_size if batch_size is None else _positive_int("batch_size", batch_size)
        written = 0
        batch: list[Any] = []
        for point in points:
            batch.append(point)
            if len(batch) == size:
                written += self._send(collection, batch)
                batch = []
        if batch:
            written += self._send(collection, batch)
        return written

    def _send(self, collection: str, batch: list[Any]) -> int:
        try:
            self._client.upsert(collection_name=collection, points=batch, wait=True)
        except Exception as exc:
            _raise_auth(exc)
            raise
        return len(batch)

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
