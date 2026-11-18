"""The Qdrant store: connection settings, strict collection checks, batched writes.

Three kinds of client are used:

* a hand-written fake, to produce each failure exactly (refusals, races, broken servers);
* qdrant-client's local mode (``QdrantClient(":memory:")``), for the behaviour of a real engine;
* the real HTTP client against a stub server, to pin how the installed qdrant-client reports
  an HTTP status.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType, SimpleNamespace

import httpx
import numpy as np
import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

import rag_engine.store as store_mod
from rag_engine.config import Settings
from rag_engine.errors import RagEngineError
from rag_engine.schema import PAYLOAD_INDEXES, RESERVED_KEYS, date_fields, date_interval_fields
from rag_engine.store import (
    CollectionMismatchError,
    CollectionNotFoundError,
    DateFieldsError,
    QdrantAuthError,
    QdrantStore,
    ReservedKeyError,
    StoreError,
)

# Expected here: the tests send a key over plain HTTP on purpose, and talk to servers that are
# stubs or absent (qdrant-client asks the server for its version when a client is built).
pytestmark = [
    pytest.mark.filterwarnings("ignore:Api key is used with an insecure connection"),
    pytest.mark.filterwarnings("ignore:Failed to obtain server version"),
]

SECRET = "s3cr3t-key-value"
COSINE = models.Distance.COSINE


def http_error(status: int, body: bytes = b"{}") -> UnexpectedResponse:
    return UnexpectedResponse(status, "x", body, httpx.Headers())


def unnamed(size: int, distance: models.Distance = COSINE) -> models.VectorParams:
    return models.VectorParams(size=size, distance=distance)


def info_of(vectors) -> SimpleNamespace:
    """The part of ``CollectionInfo`` the store reads."""
    return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))


class FakeClient:
    """Scripted Qdrant client. An exception given for a call is raised by that call.

    ``exists`` may be a list: one answer per call, the last one repeated.
    """

    def __init__(self, *, exists=False, vectors=None, get=None, create=None, upsert=None, count=None):
        self._exists = list(exists) if isinstance(exists, list) else [exists]
        self._vectors = vectors
        self._get = get
        self._create = create
        self._upsert = upsert
        self._count = count
        self.calls: list[tuple] = []

    def named(self, name: str) -> list[tuple]:
        return [call for call in self.calls if call[0] == name]

    def collection_exists(self, collection_name):
        self.calls.append(("collection_exists", collection_name))
        answer = self._exists.pop(0) if len(self._exists) > 1 else self._exists[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def get_collection(self, collection_name):
        self.calls.append(("get_collection", collection_name))
        if self._get is not None:
            raise self._get
        return info_of(self._vectors)

    def create_collection(self, collection_name, vectors_config):
        self.calls.append(("create_collection", collection_name, vectors_config))
        if self._create is not None:
            raise self._create
        self._vectors = vectors_config
        self._exists = [True]
        return True

    def upsert(self, collection_name, points, wait=None):
        self.calls.append(("upsert", collection_name, list(points), wait))
        if self._upsert is not None:
            raise self._upsert

    def count(self, collection_name, exact=None):
        self.calls.append(("count", collection_name, exact))
        if self._count is not None:
            raise self._count
        return SimpleNamespace(count=7)

    def get_collections(self):
        self.calls.append(("get_collections",))
        return SimpleNamespace(collections=[])


def fake_store(**script) -> tuple[QdrantStore, FakeClient]:
    client = FakeClient(**script)
    return QdrantStore(client=client, settings=Settings()), client


def points(n: int, dim: int = 4) -> list[models.PointStruct]:
    return [models.PointStruct(id=i, vector=[0.1] * dim, payload={"n": i}) for i in range(n)]


@pytest.fixture
def memory_store():
    client = QdrantClient(":memory:")
    yield QdrantStore(client=client, settings=Settings(batch_size=3))
    client.close()


# ---------------------------------------------------------------------------
# Construction: explicit arguments, then settings, then the environment.
# ---------------------------------------------------------------------------
@pytest.fixture
def built(monkeypatch):
    """Record the keyword arguments every ``QdrantClient`` is built with."""
    calls: list[dict] = []

    def recorder(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(store_mod, "QdrantClient", recorder)
    return calls


def test_explicit_arguments_win_over_settings(built):
    settings = Settings(
        qdrant_host="from-settings", qdrant_port=1111, qdrant_api_key="settings-key", qdrant_https=False
    )
    store = QdrantStore("explicit-host", 2222, SECRET, True, 5, settings=settings)
    assert built == [{"host": "explicit-host", "port": 2222, "api_key": SECRET, "https": True, "timeout": 5}]
    assert (store.host, store.port) == ("explicit-host", 2222)


def test_settings_fill_what_is_not_given(built):
    settings = Settings(
        qdrant_host="from-settings", qdrant_port=1111, qdrant_api_key=SECRET, qdrant_https=True
    )
    store = QdrantStore(settings=settings)
    assert built == [{"host": "from-settings", "port": 1111, "api_key": SECRET, "https": True, "timeout": 60}]
    assert (store.host, store.port) == ("from-settings", 1111)


def test_an_injected_client_without_settings_does_not_read_the_environment(monkeypatch):
    # An unrelated bad variable must not break a store that connects to nothing.
    monkeypatch.setenv("QDRANT_PORT", "abc")
    monkeypatch.setenv("RAG_ENGINE_BATCH_SIZE", "7")
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls, environ=None: pytest.fail("read")))
    client = FakeClient()
    store = QdrantStore(client=client)
    assert (store.host, store.port) == (None, None)
    # The batch size is the built-in default, not the one of the environment.
    assert Settings.batch_size == 100
    store.upsert("c", points(Settings.batch_size + 1))
    assert [len(call[2]) for call in client.calls if call[0] == "upsert"] == [Settings.batch_size, 1]


def test_an_injected_client_with_settings_takes_their_batch_size():
    client = FakeClient()
    QdrantStore(client=client, settings=Settings(batch_size=2)).upsert("c", points(3))
    assert [len(call[2]) for call in client.calls if call[0] == "upsert"] == [2, 1]


@pytest.mark.parametrize("port", [0, -1, 65536, True, 6333.0, "6333"])
def test_a_port_outside_1_to_65535_is_refused(built, port):
    with pytest.raises(ValueError, match="port") as caught:
        QdrantStore("h", port, settings=Settings())
    assert "65535" in str(caught.value)
    assert built == []


# --- the timeout: explicit, then the settings ------------------------------------
def test_the_timeout_of_the_settings_reaches_the_client(built):
    QdrantStore(settings=Settings(qdrant_timeout=7))
    QdrantStore(settings=Settings(qdrant_timeout=2.5))
    QdrantStore.from_url("http://qdrant.example", settings=Settings(qdrant_timeout=9))
    assert [call["timeout"] for call in built] == [7, 2.5, 9]


def test_an_explicit_timeout_wins_over_the_settings(built):
    settings = Settings(qdrant_timeout=7)
    QdrantStore(timeout=3, settings=settings)
    QdrantStore("h", 6333, None, False, 4, settings=settings)
    QdrantStore.from_url("http://qdrant.example", timeout=5, settings=settings)
    QdrantStore.from_url("http://qdrant.example", None, 6, settings=settings)
    assert [call["timeout"] for call in built] == [3, 4, 5, 6]


def test_the_timeout_is_read_from_the_environment_when_no_settings_are_given(built, monkeypatch):
    monkeypatch.setenv("RAG_ENGINE_QDRANT_TIMEOUT", "12")
    QdrantStore()
    QdrantStore(timeout=3)
    QdrantStore.from_url("http://qdrant.example")
    assert [call["timeout"] for call in built] == [12, 3, 12]


def test_the_default_timeout_is_the_one_of_the_settings_and_of_the_constant(built):
    QdrantStore(settings=Settings())
    assert built[0]["timeout"] == store_mod.DEFAULT_TIMEOUT == Settings.qdrant_timeout == 60


def test_a_bad_timeout_in_the_environment_is_a_config_error(built, monkeypatch):
    from rag_engine.config import ConfigError

    monkeypatch.setenv("RAG_ENGINE_QDRANT_TIMEOUT", "never")
    with pytest.raises(ConfigError, match="RAG_ENGINE_QDRANT_TIMEOUT"):
        QdrantStore()
    assert built == []
    # An injected client has no connection to configure: the variable is not read.
    QdrantStore(client=FakeClient())


def test_the_timeout_of_the_settings_is_the_one_the_real_client_waits(monkeypatch):
    # The real client, without a server: it keeps the timeout it was built with, in whole seconds.
    # Its look at the version of the server runs in a thread and would warn during a later test.
    from qdrant_client.qdrant_remote import QdrantRemote

    monkeypatch.setattr(QdrantRemote, "_check_compatibility", staticmethod(lambda *args: None))
    store = QdrantStore("127.0.0.1", 1, settings=Settings(qdrant_timeout=7))
    assert store.client._client._timeout == 7
    store.client.close()
    rounded = QdrantStore("127.0.0.1", 1, settings=Settings(qdrant_timeout=2.5))
    assert rounded.client._client._timeout == 3  # qdrant-client rounds up to a whole second
    rounded.client.close()


def test_a_bad_port_is_refused_with_an_injected_client_too():
    with pytest.raises(ValueError, match="port"):
        QdrantStore("h", 0, client=FakeClient(), settings=Settings())


@pytest.mark.parametrize("port", [1, 6333, 65535])
def test_the_bounds_of_the_port_range_are_accepted(built, port):
    assert QdrantStore("h", port, settings=Settings()).port == port


def test_without_settings_the_environment_is_read_at_construction(built, monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "env-host")
    monkeypatch.setenv("QDRANT_PORT", "7777")
    monkeypatch.setenv("QDRANT_API_KEY", SECRET)
    monkeypatch.delenv("QDRANT_HTTPS", raising=False)
    store = QdrantStore()
    assert built == [{"host": "env-host", "port": 7777, "api_key": SECRET, "https": False, "timeout": 60}]
    assert (store.host, store.port) == ("env-host", 7777)


def test_https_is_passed_explicitly_even_with_an_api_key(built):
    """Left as ``None``, the client would turn TLS on because a key is set."""
    QdrantStore(settings=Settings(qdrant_api_key=SECRET))
    assert built[0]["https"] is False


def test_an_explicit_false_https_wins_over_a_true_setting(built):
    QdrantStore(https=False, settings=Settings(qdrant_https=True))
    assert built[0]["https"] is False


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_explicit_api_key_falls_back_to_the_settings(built, blank):
    QdrantStore(api_key=blank, settings=Settings(qdrant_api_key=SECRET))
    assert built[0]["api_key"] == SECRET


def test_no_key_anywhere_gives_none(built):
    QdrantStore(settings=Settings())
    assert built[0]["api_key"] is None


def test_the_real_client_does_not_use_tls_with_a_key_and_https_false():
    store = QdrantStore("127.0.0.1", 1, SECRET, False, 1, settings=Settings())  # nothing listens there
    assert store.client._client.rest_uri == "http://127.0.0.1:1"


def test_an_injected_client_is_used_as_is(built):
    client = FakeClient()
    store = QdrantStore(client=client, settings=Settings(qdrant_host="unused"))
    assert store.client is client
    assert built == []
    # The store cannot know where an injected client points to.
    assert (store.host, store.port) == (None, None)


def test_an_injected_client_keeps_the_host_and_port_it_is_told():
    store = QdrantStore("h", 1234, client=FakeClient(), settings=Settings())
    assert (store.host, store.port) == ("h", 1234)


def test_the_api_key_is_never_printed(built):
    store = QdrantStore(api_key=SECRET, settings=Settings(qdrant_host="h"))
    assert SECRET not in repr(store) and SECRET not in str(store)
    assert "h" in repr(store)
    assert SECRET not in repr(vars(store))


def test_a_key_from_the_settings_is_not_kept_by_the_store(built):
    store = QdrantStore(settings=Settings(qdrant_host="h", qdrant_api_key=SECRET, batch_size=5))
    assert built[0]["api_key"] == SECRET  # it did reach the client
    assert SECRET not in repr(store) and SECRET not in str(store)
    assert SECRET not in repr(vars(store))  # the client is a stand-in here: nothing else holds the key
    assert not any(isinstance(value, Settings) for value in vars(store).values())
    public = [name for name in dir(store) if not name.startswith("_") and name != "client"]
    assert {"host", "port"} <= set(public)
    for name in public:
        value = getattr(store, name)
        assert SECRET not in repr(value), name
        assert not isinstance(value, Settings), name
        assert SECRET not in repr(getattr(value, "__dict__", {})), name


def test_a_key_from_the_environment_is_not_kept_by_the_store(built, monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", SECRET)
    store = QdrantStore()
    assert built[0]["api_key"] == SECRET
    assert SECRET not in repr(vars(store))
    assert not any(isinstance(value, Settings) for value in vars(store).values())


def test_an_injected_client_store_keeps_no_settings():
    store = QdrantStore(client=FakeClient(), settings=Settings(qdrant_api_key=SECRET))
    assert SECRET not in repr(vars(store))
    assert not any(isinstance(value, Settings) for value in vars(store).values())


# --- from_url ---------------------------------------------------------------
def test_from_url_reads_host_port_and_scheme(built):
    store = QdrantStore.from_url("https://qdrant.example:8443", SECRET, settings=Settings())
    assert built == [
        {"host": "qdrant.example", "port": 8443, "api_key": SECRET, "https": True, "timeout": 60}
    ]
    assert (store.host, store.port) == ("qdrant.example", 8443)


def test_from_url_http_stays_plain_even_with_a_key_and_a_true_setting(built):
    QdrantStore.from_url("http://localhost:6333/", SECRET, settings=Settings(qdrant_https=True))
    assert built[0]["https"] is False and built[0]["host"] == "localhost"


def test_from_url_without_a_port_uses_the_qdrant_default(built):
    QdrantStore.from_url("http://qdrant.example", settings=Settings(qdrant_port=1111))
    assert built[0]["port"] == 6333


@pytest.mark.parametrize(
    "url",
    [
        "qdrant.example:6333",
        "ftp://qdrant.example",
        "http://",
        "http://h:6333/prefix",
        "http://h:notaport",
        "http://h:6333?x=1",
    ],
)
def test_from_url_refuses_what_it_cannot_honour(built, url):
    with pytest.raises(ValueError, match="URL"):
        QdrantStore.from_url(url, settings=Settings())
    assert built == []


def test_from_url_never_echoes_credentials_of_the_url(built):
    with pytest.raises(ValueError) as info:
        QdrantStore.from_url(f"http://user:{SECRET}@h:6333/prefix", settings=Settings())
    assert SECRET not in str(info.value)


@pytest.mark.parametrize(
    "host",
    ["h h", "h\th", "h\nh", "h\rh", "h\x00", "h\x1f", "h\x7f", "h\u00a0h", "h\u200b", "h\u2028h", " h", "h "],
    ids=lambda host: host.encode("unicode_escape").decode(),
)
def test_from_url_refuses_whitespace_and_control_characters_in_the_host(built, host):
    with pytest.raises(ValueError, match="URL") as info:
        QdrantStore.from_url(f"http://{host}.qdrant.example:6333", SECRET, settings=Settings())
    assert "qdrant.example" not in str(info.value) and SECRET not in str(info.value)
    assert built == []


@pytest.mark.parametrize(
    "url", ["http://qdrant.example:6333\n", "\thttp://qdrant.example:6333", "http://h:63\n33"]
)
def test_from_url_refuses_whitespace_anywhere_rather_than_dropping_it(built, url):
    with pytest.raises(ValueError, match="URL"):
        QdrantStore.from_url(url, settings=Settings())
    assert built == []


@pytest.mark.parametrize("url", ["http://qdrant.example:0", "https://qdrant.example:0/", "http://[::1]:0"])
def test_from_url_refuses_port_zero(built, url):
    with pytest.raises(ValueError, match="URL") as info:
        QdrantStore.from_url(url, settings=Settings())
    assert "qdrant.example" not in str(info.value) and "::1" not in str(info.value)
    assert built == []


@pytest.mark.parametrize(
    "url", ["http://[::1", "http://::1:6333", "http://[::1]x:6333", "http://[not-an-address]:6333"]
)
def test_from_url_refuses_a_malformed_ipv6_literal(built, url):
    with pytest.raises(ValueError, match="URL"):
        QdrantStore.from_url(url, settings=Settings())
    assert built == []


# --- IPv6 literals: the client needs the brackets, the store shows the bare address ---------
def test_from_url_hands_an_ipv6_literal_to_the_client_in_brackets(built):
    store = QdrantStore.from_url("http://[::1]:6333", settings=Settings())
    assert built[0]["host"] == "[::1]" and built[0]["port"] == 6333
    assert (store.host, store.port) == ("::1", 6333)


@pytest.mark.parametrize("host", ["::1", "[::1]"])
def test_an_ipv6_host_given_to_the_constructor_is_bracketed_once(built, host):
    store = QdrantStore(host, 6333, settings=Settings())
    assert built[0]["host"] == "[::1]"
    assert store.host == "::1"


def test_an_ipv6_host_from_the_settings_is_bracketed(built):
    QdrantStore(settings=Settings(qdrant_host="2001:db8::1"))
    assert built[0]["host"] == "[2001:db8::1]"


def test_a_name_or_ipv4_host_is_handed_over_unchanged(built):
    QdrantStore("127.0.0.1", 6333, settings=Settings())
    QdrantStore("qdrant.example", 6333, settings=Settings())
    assert [call["host"] for call in built] == ["127.0.0.1", "qdrant.example"]


@pytest.mark.parametrize(
    ("url", "rest_uri"),
    [("http://[::1]:6333", "http://[::1]:6333"), ("https://[2001:db8::1]", "https://[2001:db8::1]:6333")],
)
def test_the_real_client_builds_a_valid_url_for_an_ipv6_literal(url, rest_uri):
    store = QdrantStore.from_url(url, timeout=1, settings=Settings())
    assert store.client._client.rest_uri == rest_uri
    parsed = httpx.URL(rest_uri + "/collections")  # raises InvalidURL for a bare, unbracketed address
    assert parsed.port == 6333 and parsed.host == store.host


def test_the_real_client_reaches_a_server_on_an_ipv6_loopback():
    class V6Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"result": {"collections": []}, "status": "ok", "time": 0.0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    try:
        server = V6Server(("::1", 0), Handler)
    except OSError:
        pytest.skip("no IPv6 loopback on this machine")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        store = QdrantStore.from_url(
            f"http://[::1]:{server.server_address[1]}", timeout=5, settings=Settings()
        )
        assert store.is_reachable() is True
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# check_collection: the read-only check
# ---------------------------------------------------------------------------
READS = {"collection_exists", "get_collection"}


def test_check_collection_accepts_a_matching_collection_with_two_reads():
    store, client = fake_store(exists=True, vectors=unnamed(8))
    assert store.check_collection("c", 8) is None
    assert client.calls == [("collection_exists", "c"), ("get_collection", "c")]


def test_check_collection_reports_an_absent_collection_and_never_creates_it():
    store, client = fake_store(exists=False)
    with pytest.raises(CollectionNotFoundError) as info:
        store.check_collection("central-bank", 8)
    message = str(info.value)
    assert "'central-bank'" in message and "does not exist" in message
    assert "ensure_collection" in message  # who creates it
    assert client.calls == [("collection_exists", "central-bank")]


def test_a_missing_collection_is_a_store_error_and_not_a_mismatch():
    assert issubclass(CollectionNotFoundError, StoreError)
    assert not issubclass(CollectionNotFoundError, CollectionMismatchError)
    assert not issubclass(CollectionMismatchError, CollectionNotFoundError)


@pytest.mark.parametrize(
    ("vectors", "words"),
    [
        (unnamed(384), ["768-d", "384-d"]),
        (unnamed(768, models.Distance.DOT), ["Cosine", "Dot"]),
        (unnamed(768, models.Distance.EUCLID), ["Cosine", "Euclid"]),
        (unnamed(768, models.Distance.MANHATTAN), ["Cosine", "Manhattan"]),
        ({"text": unnamed(768)}, ["named", "'text'"]),
        ({"a": unnamed(768), "b": unnamed(768)}, ["named", "'a'", "'b'"]),
        (None, ["no dense vector"]),
        ({}, ["no dense vector"]),
    ],
    ids=["dimension", "dot", "euclid", "manhattan", "named", "two-named", "none", "sparse-only"],
)
def test_check_collection_refuses_anything_but_one_unnamed_cosine_vector_of_the_dimension(vectors, words):
    store, client = fake_store(exists=True, vectors=vectors)
    with pytest.raises(CollectionMismatchError) as info:
        store.check_collection("c", 768)
    message = str(info.value)
    assert "'c'" in message and all(word in message for word in words), message
    assert {call[0] for call in client.calls} == READS


def test_check_collection_and_ensure_collection_refuse_with_the_same_words():
    for vectors in (unnamed(384), unnamed(8, models.Distance.DOT), {"text": unnamed(8)}, None):
        messages = []
        for method in ("check_collection", "ensure_collection"):
            store, _client = fake_store(exists=True, vectors=vectors)
            with pytest.raises(CollectionMismatchError) as info:
                getattr(store, method)("c", 8)
            messages.append(str(info.value))
        assert messages[0] == messages[1]


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("where", ["exists", "get"])
def test_check_collection_reports_a_refused_api_key(status, where):
    error = http_error(status, body=b'{"status":{"error":"Invalid API key"}}')
    script = {"exists": error} if where == "exists" else {"exists": True, "get": error}
    store, client = fake_store(vectors=unnamed(8), **script)
    with pytest.raises(
        QdrantAuthError, match=rf"^Qdrant refused the request \(HTTP {status}\): check QDRANT_API_KEY$"
    ) as info:
        store.check_collection("c", 8)
    assert info.value.__cause__ is error
    assert {call[0] for call in client.calls} <= READS


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("connection refused"),
        TimeoutError("timed out"),
        ResponseHandlingException(ConnectionError("connection refused")),
        http_error(500),
        http_error(404),  # e.g. a server too old to answer the existence question
        ValueError("Collection c not found"),  # a message is never read as an answer
    ],
    ids=lambda e: f"{type(e).__name__}-{getattr(e, 'status_code', '')}",
)
@pytest.mark.parametrize("where", ["exists", "get"])
def test_check_collection_raises_any_other_failure_as_the_client_raised_it(error, where):
    script = {"exists": error} if where == "exists" else {"exists": True, "get": error}
    store, client = fake_store(vectors=unnamed(8), **script)
    with pytest.raises(type(error)) as info:
        store.check_collection("c", 8)
    assert info.value is error
    assert not isinstance(info.value, StoreError)
    assert {call[0] for call in client.calls} <= READS


@pytest.mark.parametrize("dim", [0, -1, True, 8.0, "8", None])
def test_check_collection_refuses_a_bad_dimension_before_any_request(dim):
    store, client = fake_store(exists=True, vectors=unnamed(8))
    with pytest.raises(ValueError, match="dim must be an integer >= 1"):
        store.check_collection("c", dim)
    assert client.calls == []


def test_check_collection_only_reads():
    """Whatever the outcome, the client is asked the two questions and nothing else."""

    class ReadOnly:
        """Stands for a read-only API key: every call but the two reads is refused."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name not in READS:
                raise http_error(403)
            return getattr(self._inner, name)

    for script in (
        {"exists": False},
        {"exists": True, "vectors": unnamed(8)},
        {"exists": True, "vectors": None},
    ):
        store = QdrantStore(client=ReadOnly(FakeClient(**script)), settings=Settings())
        try:
            store.check_collection("c", 8)
        except (CollectionNotFoundError, CollectionMismatchError):
            pass
    # The writer's method, with the same key: refused as soon as it has to create.
    store = QdrantStore(client=ReadOnly(FakeClient(exists=False)), settings=Settings())
    with pytest.raises(QdrantAuthError):
        store.ensure_collection("c", 8)


# --- local mode ---------------------------------------------------------------
def test_local_mode_check_of_a_missing_collection_creates_nothing(memory_store):
    with pytest.raises(CollectionNotFoundError, match="'absent'"):
        memory_store.check_collection("absent", 4)
    assert memory_store.client.get_collections().collections == []
    assert not memory_store.client.collection_exists("absent")


def test_local_mode_check_accepts_what_ensure_collection_created_and_changes_nothing(memory_store):
    memory_store.ensure_collection("c", 4)
    memory_store.upsert("c", points(2))
    before = memory_store.client.get_collection("c")
    assert memory_store.check_collection("c", 4) is None
    assert memory_store.client.get_collection("c") == before
    assert memory_store.count("c") == 2
    assert [c.name for c in memory_store.client.get_collections().collections] == ["c"]


@pytest.mark.parametrize(
    ("config", "word"),
    [
        ({"vectors_config": unnamed(384)}, "384-d"),
        ({"vectors_config": unnamed(4, models.Distance.DOT)}, "Dot"),
        ({"vectors_config": unnamed(4, models.Distance.EUCLID)}, "Euclid"),
        ({"vectors_config": {"text": unnamed(4)}}, "named"),
        (
            {"vectors_config": {}, "sparse_vectors_config": {"sparse": models.SparseVectorParams()}},
            "no dense vector",
        ),
    ],
    ids=["dimension", "dot", "euclid", "named", "sparse-only"],
)
def test_local_mode_check_refuses_a_collection_that_does_not_match(memory_store, config, word):
    memory_store.client.create_collection("c", **config)
    before = memory_store.client.get_collection("c")
    with pytest.raises(CollectionMismatchError, match=word):
        memory_store.check_collection("c", 4)
    assert memory_store.client.get_collection("c") == before


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------
def test_an_absent_collection_is_created_cosine_with_the_dimension():
    store, client = fake_store(exists=False)
    assert store.ensure_collection("c", 8) is None
    assert client.named("create_collection") == [("create_collection", "c", unnamed(8))]


def test_an_existing_matching_collection_is_left_alone():
    store, client = fake_store(exists=True, vectors=unnamed(8))
    store.ensure_collection("c", 8)
    assert client.named("create_collection") == []


def test_another_dimension_is_refused():
    store, client = fake_store(exists=True, vectors=unnamed(384))
    with pytest.raises(CollectionMismatchError) as info:
        store.ensure_collection("central-bank", 768)
    message = str(info.value)
    assert "'central-bank'" in message and "768" in message and "384" in message
    assert client.named("create_collection") == []


def test_named_vectors_are_refused():
    store, client = fake_store(exists=True, vectors={"text": unnamed(8)})
    with pytest.raises(CollectionMismatchError) as info:
        store.ensure_collection("c", 8)
    message = str(info.value)
    assert "'c'" in message and "named" in message and "'text'" in message and "8" in message
    assert client.named("create_collection") == []


@pytest.mark.parametrize("vectors", [None, {}])
def test_a_collection_without_a_dense_vector_is_refused(vectors):
    store, _client = fake_store(exists=True, vectors=vectors)
    with pytest.raises(CollectionMismatchError, match="'c'"):
        store.ensure_collection("c", 8)


def test_another_distance_is_refused():
    store, client = fake_store(exists=True, vectors=unnamed(8, models.Distance.DOT))
    with pytest.raises(CollectionMismatchError) as info:
        store.ensure_collection("c", 8)
    message = str(info.value)
    assert "'c'" in message and "Cosine" in message and "Dot" in message
    assert client.named("create_collection") == []


def test_a_mismatch_is_a_store_error():
    assert issubclass(CollectionMismatchError, StoreError)
    assert issubclass(QdrantAuthError, StoreError)
    assert issubclass(StoreError, Exception)


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("connection refused"),
        TimeoutError("timed out"),
        ResponseHandlingException(ConnectionError("connection refused")),
        http_error(500),
        http_error(503),
        http_error(404),  # e.g. a server too old to answer the existence question
        ValueError("Collection c not found"),  # a message is never read as an answer
    ],
    ids=lambda e: f"{type(e).__name__}-{getattr(e, 'status_code', '')}",
)
def test_a_failed_existence_check_is_raised_and_never_read_as_absent(error):
    store, client = fake_store(exists=error)
    with pytest.raises(type(error)) as info:
        store.ensure_collection("c", 8)
    assert info.value is error
    assert client.named("create_collection") == []


@pytest.mark.parametrize(
    "error", [ConnectionError("gone"), http_error(500), http_error(404)], ids=["connection", "500", "404"]
)
def test_a_failed_read_of_an_existing_collection_is_raised_and_nothing_is_created(error):
    store, client = fake_store(exists=True, get=error)
    with pytest.raises(type(error)) as info:
        store.ensure_collection("c", 8)
    assert info.value is error
    assert client.named("create_collection") == []


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("where", ["exists", "get", "create"])
def test_a_refused_api_key_is_a_clear_error(status, where):
    error = http_error(status, body=b'{"status":{"error":"Invalid API key"}}')
    script = {"exists": error} if where == "exists" else {"exists": where == "get", where: error}
    store, client = fake_store(vectors=unnamed(8), **script)
    with pytest.raises(
        QdrantAuthError, match=rf"^Qdrant refused the request \(HTTP {status}\): check QDRANT_API_KEY$"
    ) as info:
        store.ensure_collection("c", 8)
    assert info.value.__cause__ is error
    if where != "create":
        assert client.named("create_collection") == []
    else:
        # A refused create is not a lost race: the store does not ask again.
        assert len(client.named("collection_exists")) == 1


# --- the creation race --------------------------------------------------------
@pytest.mark.parametrize(
    "error",
    [http_error(409), http_error(400), ValueError("Collection c already exists")],
    ids=["409", "400", "local"],
)
def test_losing_the_creation_race_is_not_fatal(error):
    store, client = fake_store(exists=[False, True], vectors=unnamed(8), create=error)
    store.ensure_collection("c", 8)
    assert len(client.named("create_collection")) == 1
    assert client.named("get_collection") == [("get_collection", "c")]


@pytest.mark.parametrize(
    "error", [TimeoutError("timed out"), http_error(500), http_error(409)], ids=["timeout", "500", "409"]
)
def test_a_failed_create_that_left_a_matching_collection_is_logged_without_guessing_who_created_it(
    error, caplog
):
    store, _client = fake_store(exists=[False, True], vectors=unnamed(8), create=error)
    with caplog.at_level("INFO", logger="rag_engine.store"):
        store.ensure_collection("c", 8)
    text = caplog.text.lower()
    assert "another process" not in text and "meanwhile" not in text
    last = caplog.records[-1].getMessage()
    assert "'c'" in last and "error" in last.lower() and "exists" in last.lower()


def test_a_failed_create_that_left_a_mismatching_collection_does_not_log_a_match(caplog):
    store, _client = fake_store(exists=[False, True], vectors=unnamed(384), create=http_error(500))
    with caplog.at_level("INFO", logger="rag_engine.store"), pytest.raises(CollectionMismatchError):
        store.ensure_collection("c", 8)
    assert "match" not in caplog.text.lower() and "another process" not in caplog.text.lower()


def test_the_race_winner_with_another_dimension_is_refused():
    store, _client = fake_store(exists=[False, True], vectors=unnamed(384), create=http_error(409))
    with pytest.raises(CollectionMismatchError, match="384"):
        store.ensure_collection("c", 8)


def test_the_race_winner_with_named_vectors_is_refused():
    store, _client = fake_store(exists=[False, True], vectors={"text": unnamed(8)}, create=http_error(409))
    with pytest.raises(CollectionMismatchError, match="named"):
        store.ensure_collection("c", 8)


@pytest.mark.parametrize(
    "error", [http_error(409), http_error(500), ConnectionError("gone")], ids=["409", "500", "connection"]
)
def test_a_failed_create_that_left_no_collection_is_raised_unchanged(error):
    store, _client = fake_store(exists=[False, False], create=error)
    with pytest.raises(type(error)) as info:
        store.ensure_collection("c", 8)
    assert info.value is error


def test_a_failing_recheck_after_a_failed_create_reports_the_create_error():
    create_error = http_error(409)
    store, _client = fake_store(exists=[False, ConnectionError("qdrant down")], create=create_error)
    with pytest.raises(UnexpectedResponse) as info:
        store.ensure_collection("c", 8)
    assert info.value is create_error
    assert info.value.__suppress_context__  # the follow-up failure is noise


def test_a_failing_read_after_a_lost_race_reports_the_create_error():
    create_error = http_error(409)
    store, _client = fake_store(exists=[False, True], create=create_error, get=ConnectionError("went away"))
    with pytest.raises(UnexpectedResponse) as info:
        store.ensure_collection("c", 8)
    assert info.value is create_error
    assert info.value.__suppress_context__


@pytest.mark.parametrize("dim", [0, -1, True, 7.0, "8", None])
def test_a_dimension_that_is_not_a_positive_integer_is_refused_before_any_call(dim):
    store, client = fake_store()
    with pytest.raises(ValueError, match="dim"):
        store.ensure_collection("c", dim)
    assert client.calls == []


# --- local mode: a real engine --------------------------------------------------
def test_local_mode_creates_then_the_second_call_is_a_no_op(memory_store):
    memory_store.ensure_collection("c", 4)
    vectors = memory_store.client.get_collection("c").config.params.vectors
    assert (vectors.size, vectors.distance) == (4, COSINE)
    memory_store.upsert("c", points(2))
    memory_store.ensure_collection("c", 4)  # idempotent, and keeps the points
    assert memory_store.count("c") == 2


def test_local_mode_refuses_another_dimension(memory_store):
    memory_store.client.create_collection("c", vectors_config=unnamed(384))
    with pytest.raises(CollectionMismatchError, match="384"):
        memory_store.ensure_collection("c", 4)


def test_local_mode_refuses_named_vectors(memory_store):
    memory_store.client.create_collection("c", vectors_config={"text": unnamed(4)})
    with pytest.raises(CollectionMismatchError, match="named"):
        memory_store.ensure_collection("c", 4)


def test_local_mode_refuses_a_sparse_only_collection(memory_store):
    memory_store.client.create_collection(
        "c", vectors_config={}, sparse_vectors_config={"sparse": models.SparseVectorParams()}
    )
    with pytest.raises(CollectionMismatchError):
        memory_store.ensure_collection("c", 4)


def test_local_mode_refuses_another_distance(memory_store):
    memory_store.client.create_collection("c", vectors_config=unnamed(4, models.Distance.EUCLID))
    with pytest.raises(CollectionMismatchError, match="Euclid"):
        memory_store.ensure_collection("c", 4)


class Raced:
    """A local client in front of which another process creates the collection first."""

    def __init__(self, inner: QdrantClient, other_dim: int):
        self._inner, self._other_dim = inner, other_dim

    def create_collection(self, collection_name, vectors_config):
        self._inner.create_collection(collection_name, vectors_config=unnamed(self._other_dim))
        return self._inner.create_collection(collection_name, vectors_config=vectors_config)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_local_mode_losing_the_race_is_not_fatal():
    inner = QdrantClient(":memory:")
    QdrantStore(client=Raced(inner, 4), settings=Settings()).ensure_collection("c", 4)
    assert inner.collection_exists("c")


def test_local_mode_race_winner_with_another_dimension_is_refused():
    inner = QdrantClient(":memory:")
    with pytest.raises(CollectionMismatchError, match="384"):
        QdrantStore(client=Raced(inner, 384), settings=Settings()).ensure_collection("c", 4)


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("n", "sizes"),
    [(0, []), (1, [1]), (2, [2]), (3, [3]), (4, [3, 1]), (6, [3, 3]), (7, [3, 3, 1])],
)
def test_upsert_batches_at_the_boundaries(n, sizes):
    store, client = fake_store()
    assert store.upsert("c", points(n), batch_size=3) == n
    calls = client.named("upsert")
    assert [len(call[2]) for call in calls] == sizes
    assert all(call[1] == "c" and call[3] is True for call in calls)  # wait=True
    assert [p.id for call in calls for p in call[2]] == list(range(n))  # order kept, none lost


def test_upsert_of_nothing_calls_nothing():
    store, client = fake_store()
    assert store.upsert("c", []) == 0
    assert client.calls == []


def test_upsert_default_batch_size_comes_from_the_settings():
    client = FakeClient()
    store = QdrantStore(client=client, settings=Settings(batch_size=2))
    assert store.upsert("c", points(5)) == 5
    assert [len(call[2]) for call in client.named("upsert")] == [2, 2, 1]


def test_upsert_batch_size_one_is_one_point_per_call():
    store, client = fake_store()
    store.upsert("c", points(3), batch_size=1)
    assert [len(call[2]) for call in client.named("upsert")] == [1, 1, 1]


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "3"])
def test_upsert_refuses_a_batch_size_that_is_not_a_positive_integer(bad):
    store, client = fake_store()
    with pytest.raises(ValueError, match="batch_size"):
        store.upsert("c", points(2), batch_size=bad)
    with pytest.raises(ValueError, match="batch_size"):
        store.upsert("c", [], batch_size=bad)  # refused even with nothing to write
    assert client.calls == []


def test_upsert_accepts_any_iterable():
    store, client = fake_store()
    assert store.upsert("c", iter(points(4)), batch_size=3) == 4
    assert [len(call[2]) for call in client.named("upsert")] == [3, 1]


@pytest.mark.parametrize("status", [401, 403])
def test_upsert_refused_api_key_is_a_clear_error(status):
    error = http_error(status)
    store, _client = fake_store(upsert=error)
    with pytest.raises(QdrantAuthError, match=rf"HTTP {status}\): check QDRANT_API_KEY$") as info:
        store.upsert("c", points(1))
    assert info.value.__cause__ is error


@pytest.mark.parametrize(
    "error", [http_error(500), http_error(404), ConnectionError("gone")], ids=["500", "404", "connection"]
)
def test_upsert_other_errors_are_raised_unchanged(error):
    store, client = fake_store(upsert=error)
    with pytest.raises(type(error)) as info:
        store.upsert("c", points(5), batch_size=2)
    assert info.value is error
    assert len(client.named("upsert")) == 1  # stops at the first failed batch


class FailingFrom(FakeClient):
    """Accepts the first ``ok`` upsert calls, then raises ``error`` on every later one."""

    def __init__(self, ok: int, error: Exception):
        super().__init__()
        self._ok, self._error = ok, error

    def upsert(self, collection_name, points, wait=None):
        super().upsert(collection_name, points, wait)
        if len(self.named("upsert")) > self._ok:
            raise self._error


@pytest.mark.parametrize(
    "error",
    [http_error(500), ConnectionError("gone"), TimeoutError("timed out")],
    ids=["500", "connection", "timeout"],
)
def test_upsert_failure_on_the_second_batch_tells_how_many_points_were_written(error):
    client = FailingFrom(1, error)
    store = QdrantStore(client=client, settings=Settings())
    with pytest.raises(type(error)) as info:
        store.upsert("c", points(7), batch_size=3)
    assert info.value is error  # same exception, same type
    assert info.value.points_written == 3
    assert [len(call[2]) for call in client.named("upsert")] == [3, 3]  # the third batch is never sent


def test_upsert_failure_on_the_first_batch_reports_zero_points_written():
    error = http_error(500)
    store, _client = fake_store(upsert=error)
    with pytest.raises(UnexpectedResponse) as info:
        store.upsert("c", points(5), batch_size=2)
    assert info.value.points_written == 0


def test_upsert_refused_key_after_some_batches_tells_how_many_points_were_written():
    error = http_error(401)
    client = FailingFrom(2, error)
    store = QdrantStore(client=client, settings=Settings())
    with pytest.raises(QdrantAuthError) as info:
        store.upsert("c", points(5), batch_size=2)
    assert info.value.points_written == 4
    assert info.value.__cause__ is error


@pytest.mark.parametrize(("good", "written"), [(0, 0), (2, 0), (3, 3), (5, 3), (6, 6)])
def test_upsert_of_a_generator_that_raises_midway_tells_how_many_points_were_written(good, written):
    boom = RuntimeError("the source broke")

    def source():
        yield from points(good)
        raise boom

    store, client = fake_store()
    with pytest.raises(RuntimeError) as info:
        store.upsert("c", source(), batch_size=3)
    assert info.value is boom
    assert info.value.points_written == written
    # Full batches went out before the failure; the partial one is not sent.
    assert sum(len(call[2]) for call in client.named("upsert")) == written


def test_upsert_failure_with_an_exception_that_takes_no_attribute_is_still_raised():
    class Sealed(Exception):
        __slots__ = ()

        def __setattr__(self, name, value):
            raise AttributeError(name)

    error = Sealed("sealed")
    store, _client = fake_store(upsert=error)
    with pytest.raises(Sealed) as info:
        store.upsert("c", points(1))
    assert info.value is error


def test_local_mode_batches_written_before_a_failure_stay_and_a_rerun_overwrites_them(memory_store):
    memory_store.ensure_collection("c", 4)

    def source():
        yield from points(4)
        raise RuntimeError("the source broke")

    with pytest.raises(RuntimeError) as info:
        memory_store.upsert("c", source())  # batch size 3 from the settings
    assert info.value.points_written == 3
    assert memory_store.count("c") == 3
    assert memory_store.upsert("c", points(4)) == 4  # same ids: the re-run overwrites
    assert memory_store.count("c") == 4


@pytest.mark.parametrize("n", [0, 1, 3, 4])
def test_local_mode_upsert_and_count(memory_store, n):
    memory_store.ensure_collection("c", 4)
    assert memory_store.upsert("c", points(n)) == n  # batch size 3 from the settings
    assert memory_store.count("c") == n


def test_local_mode_upsert_twice_overwrites_the_same_ids(memory_store):
    memory_store.ensure_collection("c", 4)
    memory_store.upsert("c", points(4))
    assert memory_store.upsert("c", points(4)) == 4
    assert memory_store.count("c") == 4


def test_local_mode_upsert_into_a_missing_collection_raises(memory_store):
    with pytest.raises(Exception, match="not found"):
        memory_store.upsert("missing", points(1))


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------
def test_count_is_exact():
    store, client = fake_store()
    assert store.count("c") == 7
    assert client.named("count") == [("count", "c", True)]


@pytest.mark.parametrize("status", [401, 403])
def test_count_refused_api_key_is_a_clear_error(status):
    store, _client = fake_store(count=http_error(status))
    with pytest.raises(QdrantAuthError, match=f"HTTP {status}"):
        store.count("c")


def test_count_other_errors_are_raised_unchanged():
    error = http_error(500)
    store, _client = fake_store(count=error)
    with pytest.raises(UnexpectedResponse) as info:
        store.count("c")
    assert info.value is error


# ---------------------------------------------------------------------------
# is_reachable
# ---------------------------------------------------------------------------
def test_is_reachable_true_when_the_server_answers(memory_store):
    assert memory_store.is_reachable() is True


@pytest.mark.parametrize(
    "error",
    [ConnectionError("refused"), http_error(401), http_error(500), ValueError("x")],
    ids=["connection", "401", "500", "other"],
)
def test_is_reachable_false_and_silent_on_any_failure(error, caplog, capsys):
    class Down(FakeClient):
        def get_collections(self):
            raise error

    store = QdrantStore(api_key=SECRET, client=Down(), settings=Settings(qdrant_api_key=SECRET))
    with caplog.at_level("DEBUG"):
        assert store.is_reachable() is False
    out = capsys.readouterr()
    assert SECRET not in caplog.text and SECRET not in out.out and SECRET not in out.err


# ---------------------------------------------------------------------------
# The real HTTP client: how the installed qdrant-client reports a status.
# ---------------------------------------------------------------------------
class StubQdrant:
    """An HTTP server that answers every request with one status, and records the requests."""

    def __init__(self, status: int):
        self.requests: list[tuple[str, str, str | None]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                stub.requests.append((self.command, self.path, self.headers.get("api-key")))
                body = json.dumps({"status": {"error": "stub"}, "time": 0.0}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_PUT = do_POST = do_DELETE = _answer

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def store(self) -> QdrantStore:
        return QdrantStore("127.0.0.1", self.port, SECRET, False, 5, settings=Settings())

    def writes(self) -> list[tuple[str, str, str | None]]:
        return [request for request in self.requests if request[0] != "GET"]


@pytest.mark.parametrize("status", [401, 403])
def test_real_client_refusal_becomes_an_auth_error_without_the_key(status):
    with StubQdrant(status) as stub:
        store = stub.store()
        with pytest.raises(QdrantAuthError) as info:
            store.ensure_collection("c", 8)
        assert SECRET not in str(info.value) and SECRET not in repr(info.value)
        with pytest.raises(QdrantAuthError):
            store.upsert("c", points(1))
        with pytest.raises(QdrantAuthError):
            store.count("c")
        assert store.is_reachable() is False
        assert stub.requests[0][2] == SECRET  # the key did travel, over plain HTTP
        assert [r for r in stub.writes() if "points" not in r[1]] == []  # no create attempted


@pytest.mark.parametrize("status", [404, 500, 503])
def test_real_client_server_error_is_raised_and_creates_nothing(status):
    with StubQdrant(status) as stub:
        store = stub.store()
        with pytest.raises(UnexpectedResponse) as info:
            store.ensure_collection("c", 8)
        assert info.value.status_code == status
        assert not isinstance(info.value, StoreError)
        assert stub.writes() == []
        assert store.is_reachable() is False


def test_real_client_connection_refused_is_raised_and_not_read_as_absent():
    with StubQdrant(200) as stub:
        port = stub.port
    store = QdrantStore("127.0.0.1", port, None, False, 2, settings=Settings())  # nothing listens now
    with pytest.raises(ResponseHandlingException):
        store.ensure_collection("c", 8)
    assert store.is_reachable() is False


# ---------------------------------------------------------------------------
# Payload indexes.
# ---------------------------------------------------------------------------
INDEXED_FIELDS = [
    "doc_id",
    "source_code",
    "doc_type",
    "language",
    "corpus",
    "year",
    "date",
    "date_start",
    "date_end",
]


def index_info(kind: str) -> SimpleNamespace:
    """The part of ``PayloadIndexInfo`` the store reads, with the enum the client gives."""
    return SimpleNamespace(data_type=models.PayloadSchemaType(kind))


class IndexClient:
    """Scripted client that keeps payload indexes the way the server does.

    Like the server, it replaces an index of another kind without a word: only the store
    stands between a wrong call and a rebuilt index.
    """

    def __init__(self, existing=None, *, get=None, create=None, fail_on=None):
        self.schema = {field: index_info(kind) for field, kind in (existing or {}).items()}
        self._get = get
        self._create = create
        self._fail_on = fail_on
        self.calls: list[tuple] = []

    def created(self) -> list[tuple]:
        return [call[1:] for call in self.calls if call[0] == "create_payload_index"]

    def get_collection(self, collection_name):
        self.calls.append(("get_collection", collection_name))
        if self._get is not None:
            raise self._get
        return SimpleNamespace(payload_schema=dict(self.schema))

    def create_payload_index(self, collection_name, field_name, field_schema=None, wait=None):
        self.calls.append(("create_payload_index", collection_name, field_name, field_schema, wait))
        if self._create is not None and self._fail_on in (None, field_name):
            raise self._create
        self.schema[field_name] = index_info(getattr(field_schema, "value", field_schema))


def index_store(*args, **script) -> tuple[QdrantStore, IndexClient]:
    client = IndexClient(*args, **script)
    return QdrantStore(client=client, settings=Settings()), client


def test_the_documented_index_list():
    assert PAYLOAD_INDEXES == (
        ("doc_id", "keyword"),
        ("source_code", "keyword"),
        ("doc_type", "keyword"),
        ("language", "keyword"),
        ("corpus", "keyword"),
        ("year", "integer"),
        ("date", "datetime"),
        ("date_start", "datetime"),
        ("date_end", "datetime"),
    )
    for _field, kind in PAYLOAD_INDEXES:
        models.PayloadSchemaType(kind)  # every kind is one Qdrant knows


def test_indexes_missing_are_all_created_and_waited_for():
    store, client = index_store()
    assert store.ensure_payload_indexes("c") == INDEXED_FIELDS
    assert client.created() == [
        ("c", field, models.PayloadSchemaType(kind), True) for field, kind in PAYLOAD_INDEXES
    ]


def test_indexes_second_call_creates_nothing():
    store, client = index_store()
    store.ensure_payload_indexes("c")
    before = len(client.created())
    assert store.ensure_payload_indexes("c") == []
    assert len(client.created()) == before


def test_indexes_only_the_missing_ones_are_created():
    store, client = index_store({"doc_id": "keyword", "year": "integer", "title": "text"})
    missing = ["source_code", "doc_type", "language", "corpus", "date", "date_start", "date_end"]
    assert store.ensure_payload_indexes("c") == missing
    assert [call[1] for call in client.created()] == missing
    assert client.schema["title"].data_type == models.PayloadSchemaType.TEXT  # not ours: left alone


def test_indexes_existing_kind_is_read_as_an_enum_or_as_a_plain_string():
    store, client = index_store(dict(PAYLOAD_INDEXES))
    client.schema["year"] = SimpleNamespace(data_type="integer")
    assert store.ensure_payload_indexes("c") == []
    assert client.created() == []


@pytest.mark.parametrize(
    ("field", "found", "expected"),
    [("date", "keyword", "datetime"), ("year", "keyword", "integer"), ("doc_id", "text", "keyword")],
)
def test_an_index_of_another_kind_is_refused_and_nothing_is_created(field, found, expected):
    store, client = index_store({field: found})
    with pytest.raises(CollectionMismatchError) as info:
        store.ensure_payload_indexes("c")
    message = str(info.value)
    assert repr(field) in message
    assert f"expected a {expected} index" in message
    assert f"found a {found} index" in message
    assert "'c'" in message
    assert client.created() == []  # not replaced, and no other index created either
    assert client.schema[field].data_type == models.PayloadSchemaType(found)


def test_every_index_of_another_kind_is_named():
    store, _ = index_store({"date": "keyword", "year": "float"})
    with pytest.raises(CollectionMismatchError) as info:
        store.ensure_payload_indexes("c")
    assert "'date'" in str(info.value)
    assert "'year'" in str(info.value)


def test_indexes_custom_list():
    store, client = index_store({"doc_id": "keyword"})
    wanted = [("title", "text"), ("page", models.PayloadSchemaType.INTEGER), ("doc_id", "keyword")]
    assert store.ensure_payload_indexes("c", wanted) == ["title", "page"]
    assert client.created() == [
        ("c", "title", models.PayloadSchemaType.TEXT, True),
        ("c", "page", models.PayloadSchemaType.INTEGER, True),
    ]


def test_indexes_empty_list_creates_nothing_but_still_needs_the_collection():
    store, client = index_store()
    assert store.ensure_payload_indexes("c", []) == []
    assert client.created() == []
    store, _ = index_store(get=http_error(404))
    with pytest.raises(UnexpectedResponse):
        store.ensure_payload_indexes("c", [])


@pytest.mark.parametrize(
    ("indexes", "match"),
    [
        ([("year", "number")], "number"),
        ([("year", None)], "None"),
        ([("", "keyword")], "field"),
        ([(3, "keyword")], "field"),
        ([("year", "integer"), ("year", "keyword")], "more than once"),
        ([("year", "integer"), ("year", "integer")], "more than once"),
        (["year"], "pair"),
        ([("year", "integer", "extra")], "pair"),
    ],
)
def test_a_bad_index_list_is_refused_before_any_call(indexes, match):
    store, client = index_store()
    with pytest.raises(ValueError, match=match):
        store.ensure_payload_indexes("c", indexes)
    assert client.calls == []


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("where", ["get", "create"])
def test_indexes_refused_api_key_is_a_clear_error(status, where):
    store, _ = index_store(**{where: http_error(status, SECRET.encode())})
    with pytest.raises(QdrantAuthError) as info:
        store.ensure_payload_indexes("c")
    assert str(status) in str(info.value)
    assert SECRET not in str(info.value)


@pytest.mark.parametrize(
    "error", [http_error(404), http_error(500), ConnectionError("refused"), TimeoutError("slow")]
)
def test_indexes_a_failed_read_of_the_collection_is_raised_and_nothing_is_created(error):
    store, client = index_store(get=error)
    with pytest.raises(type(error)) as info:
        store.ensure_payload_indexes("c")
    assert info.value is error
    assert client.created() == []


def test_indexes_a_failed_creation_is_raised_and_a_rerun_finishes_the_work():
    error = http_error(500)
    store, client = index_store(create=error, fail_on="language")
    with pytest.raises(UnexpectedResponse) as info:
        store.ensure_payload_indexes("c")
    assert info.value is error
    assert sorted(client.schema) == ["doc_id", "doc_type", "source_code"]  # created before it: kept
    client._create = None
    assert store.ensure_payload_indexes("c") == [
        "language",
        "corpus",
        "year",
        "date",
        "date_start",
        "date_end",
    ]


# --- local mode ---------------------------------------------------------------
# Local mode accepts create_payload_index, warns that it has no effect, and reports no index
# in the collection info: a second call cannot see the first one. What is checked here is that
# nothing fails and nothing is lost; idempotence on a real engine is an integration test.
LOCAL_INDEX_WARNING = "ignore:Payload indexes have no effect in the local Qdrant"

ODD_DATES = {
    1: {"date": "2016-03-10", "year": 2016, "source_code": "fr"},
    2: {"date": "2016-03", "year": 2016, "source_code": "fr"},
    3: {"date": "2016", "year": 2016, "source_code": "de"},
    4: {"date": None, "year": None, "source_code": "de"},
    5: {"source_code": "de"},
    6: {"date": "2017-01-01", "year": 2017, "source_code": "fr"},
}


def odd_points() -> list[models.PointStruct]:
    return [
        models.PointStruct(id=i, vector=[1.0, 0.1 * i, 0.0, 0.0], payload=payload)
        for i, payload in ODD_DATES.items()
    ]


def ids_matching(client, condition) -> list[int]:
    found, _ = client.scroll("c", scroll_filter=models.Filter(must=[condition]), limit=100)
    return sorted(point.id for point in found)


@pytest.mark.filterwarnings(LOCAL_INDEX_WARNING)
def test_local_mode_indexes_on_an_empty_collection(memory_store):
    memory_store.ensure_collection("c", 4)
    assert memory_store.ensure_payload_indexes("c") == INDEXED_FIELDS


@pytest.mark.filterwarnings(LOCAL_INDEX_WARNING)
def test_local_mode_indexes_on_a_collection_with_points_and_odd_dates(memory_store):
    memory_store.ensure_collection("c", 4)
    memory_store.upsert("c", odd_points())
    assert memory_store.ensure_payload_indexes("c") == INDEXED_FIELDS
    assert memory_store.count("c") == len(ODD_DATES)
    client = memory_store.client
    stored = {point.id: point.payload for point in client.retrieve("c", ids=list(ODD_DATES))}
    assert stored == ODD_DATES  # no payload rewritten
    # Odd dates are still accepted once the indexes are asked for.
    assert memory_store.upsert("c", odd_points()) == len(ODD_DATES)


@pytest.mark.filterwarnings(LOCAL_INDEX_WARNING)
def test_local_mode_filters_on_the_indexed_fields(memory_store):
    memory_store.ensure_collection("c", 4)
    memory_store.upsert("c", odd_points())
    memory_store.ensure_payload_indexes("c")
    client = memory_store.client

    def date_range(**bounds):
        return models.FieldCondition(key="date", range=models.DatetimeRange(**bounds))

    source = models.FieldCondition(key="source_code", match=models.MatchValue(value="fr"))
    assert ids_matching(client, source) == [1, 2, 6]
    year = models.FieldCondition(key="year", range=models.Range(gte=2016, lte=2016))
    assert ids_matching(client, year) == [1, 2, 3]
    # A date that is not a full date, a null date and a missing date never match a date range.
    assert ids_matching(client, date_range(gte="2016-01-01", lte="2016-12-31")) == [1]
    assert ids_matching(client, date_range(gte="2016-03-10", lte="2016-03-10")) == [1]  # bounds included
    assert ids_matching(client, date_range(gt="2016-03-10")) == [6]
    assert ids_matching(client, date_range(gte="1900-01-01")) == [1, 6]


@pytest.mark.filterwarnings(LOCAL_INDEX_WARNING)
def test_local_mode_indexes_on_a_missing_collection_raise_and_create_nothing(memory_store):
    with pytest.raises(ValueError, match="not found"):
        memory_store.ensure_payload_indexes("absent")
    assert not memory_store.client.collection_exists("absent")


# ---------------------------------------------------------------------------
# existing_doc_ids
# ---------------------------------------------------------------------------
def doc_points(*chunks_per_doc: tuple[str, int]) -> list[models.PointStruct]:
    """Points of documents given as ``(doc_id, number of chunks)``, with ids in no useful order."""
    result = []
    for doc_id, chunks in chunks_per_doc:
        for index in range(chunks):
            result.append(
                models.PointStruct(
                    id=len(result) + 1, vector=[1.0, 0.0, 0.0, float(index)], payload={"doc_id": doc_id}
                )
            )
    return result


class ScrollClient:
    """Answers ``scroll`` from a list of points, a page at a time, and records every call."""

    def __init__(self, stored: list[models.PointStruct], error: Exception | None = None):
        self.stored, self.error, self.calls = stored, error, []

    def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
        if self.error is not None:
            raise self.error
        wanted = set(scroll_filter.must[0].match.any)
        assert scroll_filter.must[0].key == "doc_id"
        self.calls.append((collection_name, sorted(wanted), limit, with_payload, with_vectors))
        matching = [point for point in self.stored if point.payload.get("doc_id") in wanted]
        return matching[:limit], None


def test_existing_doc_ids_local_mode(memory_store):
    memory_store.ensure_collection("c", 4)
    memory_store.upsert("c", doc_points(("a", 3), ("b", 1), ("c", 2)))
    assert memory_store.existing_doc_ids("c", ["a", "c", "nowhere", "A"]) == {"a", "c"}
    assert memory_store.existing_doc_ids("c", ("b",)) == {"b"}
    assert memory_store.existing_doc_ids("c", {"nowhere"}) == set()


def test_existing_doc_ids_of_nothing_sends_nothing():
    client = ScrollClient([])
    assert QdrantStore(client=client).existing_doc_ids("c", []) == set()
    assert client.calls == []


def test_existing_doc_ids_asks_in_batches_and_reads_the_doc_id_only():
    client = ScrollClient(doc_points(*[(f"d{i}", 1) for i in range(5)]))
    store = QdrantStore(client=client)
    wanted = ["d0", "d1", "d2", "d3", "d4", "x1", "x2"]
    assert store.existing_doc_ids("c", wanted, batch_size=3) == {"d0", "d1", "d2", "d3", "d4"}
    # Each batch is asked until nothing new comes back: once for what it finds, once for the rest.
    asked = [call[1] for call in client.calls]
    assert asked == [["d0", "d1", "d2"], ["d3", "d4", "x1"], ["x1"], ["x2"]]
    assert {call[0] for call in client.calls} == {"c"}
    assert {tuple(call[3]) for call in client.calls} == {("doc_id",)}
    assert {call[4] for call in client.calls} == {False}


def test_existing_doc_ids_finds_a_document_hidden_behind_a_long_one():
    # A page may hold the chunks of one long document only: the others are asked again.
    client = ScrollClient(doc_points(("long", 7), ("other", 4), ("short", 1)))
    store = QdrantStore(client=client)
    found = store.existing_doc_ids("c", ["long", "short", "other", "absent"], page_size=3)
    assert found == {"long", "short", "other"}
    assert [call[1] for call in client.calls] == [
        ["absent", "long", "other", "short"],
        ["absent", "other", "short"],
        ["absent", "short"],
        ["absent"],
    ]


def test_existing_doc_ids_duplicates_are_asked_once():
    client = ScrollClient(doc_points(("a", 1)))
    assert QdrantStore(client=client).existing_doc_ids("c", ["a", "a", "a"]) == {"a"}
    assert [call[1] for call in client.calls] == [["a"]]


@pytest.mark.parametrize(
    ("doc_ids", "match"),
    [
        ("abc", "list of strings"),
        (b"abc", "list of strings"),
        (["a", 3], "strings"),
        (["a", ""], "non-empty"),
        (7, "list of strings"),
    ],
)
def test_existing_doc_ids_refuses_what_is_not_a_list_of_ids(doc_ids, match):
    client = ScrollClient([])
    with pytest.raises(ValueError, match=match):
        QdrantStore(client=client).existing_doc_ids("c", doc_ids)
    assert client.calls == []


@pytest.mark.parametrize("name", ["batch_size", "page_size"])
@pytest.mark.parametrize("value", [0, -1, True, 2.0, "3"])
def test_existing_doc_ids_refuses_a_bad_size(name, value):
    with pytest.raises(ValueError, match=name):
        QdrantStore(client=ScrollClient([])).existing_doc_ids("c", ["a"], **{name: value})


@pytest.mark.parametrize("status", [401, 403])
def test_existing_doc_ids_refused_api_key_is_a_clear_error(status):
    store = QdrantStore(client=ScrollClient([], http_error(status)))
    with pytest.raises(QdrantAuthError, match=f"HTTP {status}"):
        store.existing_doc_ids("c", ["a"])


def test_existing_doc_ids_other_errors_are_raised_unchanged():
    error = http_error(500)
    with pytest.raises(UnexpectedResponse) as info:
        QdrantStore(client=ScrollClient([], error)).existing_doc_ids("c", ["a"])
    assert info.value is error


def test_existing_doc_ids_on_a_missing_collection_raises(memory_store):
    with pytest.raises(ValueError, match="missing"):
        memory_store.existing_doc_ids("missing", ["a"])


# ---------------------------------------------------------------------------
# set_payload
# ---------------------------------------------------------------------------
def payload_points() -> list[models.PointStruct]:
    """Three documents; ``a`` has three chunks. Vectors with many digits, to see any rewrite."""
    rows = [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("c", 0)]
    return [
        models.PointStruct(
            id=10 * (1 + "abc".index(doc_id)) + chunk,
            vector=[0.123456789 * (index + 1), -0.987654321, 0.5, 1 / 3],
            payload={
                "doc_id": doc_id,
                "chunk_number": chunk,
                "text": f"chunk {chunk} of {doc_id}",
                "date": "2016-01-01",
                "year": 2016,
                "title": f"Document {doc_id}",
            },
        )
        for index, (doc_id, chunk) in enumerate(rows)
    ]


def stored(store, collection: str = "c") -> dict:
    records, _ = store.client.scroll(collection, limit=100, with_payload=True, with_vectors=True)
    return {record.id: record for record in records}


@pytest.fixture
def payload_store(memory_store):
    memory_store.ensure_collection("c", 4)
    memory_store.upsert("c", payload_points())
    return memory_store


def test_set_payload_sets_the_keys_on_every_point_of_the_document_and_on_no_other(payload_store):
    before = stored(payload_store)
    fields = date_fields("2016-07-01", "month")

    assert payload_store.set_payload("c", "a", fields) == 3

    after = stored(payload_store)
    assert sorted(after) == sorted(before)
    for point_id, record in after.items():
        old = before[point_id]
        if old.payload["doc_id"] == "a":
            assert record.payload == {
                **old.payload,
                "date": "2016-07-01",
                "date_precision": "month",
                "date_start": "2016-07-01T00:00:00Z",
                "date_end": "2016-07-31T00:00:00Z",
            }
        else:
            assert record.payload == old.payload
        # No vector touched: the very same floats.
        assert np.asarray(record.vector, dtype=np.float32).tobytes() == (
            np.asarray(old.vector, dtype=np.float32).tobytes()
        )
    assert payload_store.count("c") == 5


def test_set_payload_keeps_the_keys_it_is_not_given_and_adds_new_ones(payload_store):
    assert payload_store.set_payload("c", "b", {"date_kind": "meeting"}) == 1
    (record,) = [r for r in stored(payload_store).values() if r.payload["doc_id"] == "b"]
    assert record.payload == {
        "doc_id": "b",
        "chunk_number": 0,
        "text": "chunk 0 of b",
        "date": "2016-01-01",
        "year": 2016,
        "title": "Document b",
        "date_kind": "meeting",
    }


def test_set_payload_with_a_null_leaves_no_value_under_the_key(payload_store):
    fields = date_fields(None, year=None)
    assert fields == {
        "date": None,
        "date_precision": None,
        "date_start": None,
        "date_end": None,
        "year": None,
    }
    assert payload_store.set_payload("c", "c", fields) == 1
    (record,) = [r for r in stored(payload_store).values() if r.payload["doc_id"] == "c"]
    # Qdrant drops a key set to null (an integration test pins it on a real server).
    for key in fields:
        assert record.payload.get(key) is None
    assert record.payload["title"] == "Document c"


def test_set_payload_twice_is_the_same_as_once(payload_store):
    fields = {"title": "A better title"}
    assert payload_store.set_payload("c", "a", fields) == 3
    once = {point_id: record.payload for point_id, record in stored(payload_store).items()}
    assert payload_store.set_payload("c", "a", fields) == 3
    assert {point_id: record.payload for point_id, record in stored(payload_store).items()} == once


def test_set_payload_on_an_unknown_document_returns_zero_and_writes_nothing(payload_store):
    before = {point_id: record.payload for point_id, record in stored(payload_store).items()}
    assert payload_store.set_payload("c", "no-such-document", {"title": "x"}) == 0
    assert payload_store.set_payload("c", "A", {"title": "x"}) == 0  # doc_id is compared exactly
    assert {point_id: record.payload for point_id, record in stored(payload_store).items()} == before


class PayloadClient:
    """Records ``count`` and ``set_payload``; ``points`` is the number of points of the document."""

    def __init__(self, points: int = 2, count_error=None, set_error=None) -> None:
        self.points = points
        self.count_error, self.set_error = count_error, set_error
        self.calls: list[tuple[str, dict]] = []

    def count(self, **kwargs):
        self.calls.append(("count", kwargs))
        if self.count_error is not None:
            raise self.count_error
        return SimpleNamespace(count=self.points)

    def set_payload(self, **kwargs):
        self.calls.append(("set_payload", kwargs))
        if self.set_error is not None:
            raise self.set_error


def test_set_payload_counts_exactly_then_writes_once_through_a_filter_on_doc_id_and_waits():
    client = PayloadClient(points=7)
    fields = {"title": "A title", "date_kind": "meeting"}
    assert QdrantStore(client=client).set_payload("col", "doc-1", fields) == 7
    selector = models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="doc-1"))]
    )
    assert client.calls == [
        ("count", {"collection_name": "col", "count_filter": selector, "exact": True}),
        ("set_payload", {"collection_name": "col", "payload": fields, "points": selector, "wait": True}),
    ]
    assert client.calls[1][1]["payload"] is not fields  # a copy: the caller's mapping is not handed over
    assert fields == {"title": "A title", "date_kind": "meeting"}


def test_set_payload_sends_no_write_for_a_document_without_points():
    client = PayloadClient(points=0)
    assert QdrantStore(client=client).set_payload("col", "doc-1", {"title": "x"}) == 0
    assert [name for name, _ in client.calls] == ["count"]


@pytest.mark.parametrize("key", RESERVED_KEYS)
def test_set_payload_refuses_a_reserved_key(key):
    client = PayloadClient()
    with pytest.raises(ReservedKeyError, match=rf"reserved payload key\(s\): {key}\b") as raised:
        QdrantStore(client=client).set_payload("col", "doc-1", {"title": "ok", key: "x"})
    assert "title" not in str(raised.value).split(":", 1)[1].split("(")[0]
    assert isinstance(raised.value, StoreError) and isinstance(raised.value, RagEngineError)
    assert isinstance(raised.value, ValueError)  # a bad argument, like every other one of the store
    assert client.calls == []


def test_set_payload_names_every_reserved_key_and_accepts_the_derived_and_date_keys():
    assert RESERVED_KEYS == ("doc_id", "filename", "chunk_number", "ingestion_date", "text", "page")
    client = PayloadClient()
    store = QdrantStore(client=client)
    with pytest.raises(ReservedKeyError, match="doc_id, text, page") as raised:
        store.set_payload("col", "doc-1", {"page": 1, "text": "t", "doc_id": "doc-1", "title": "x"})
    assert "'doc-1'" in str(raised.value) and client.calls == []
    # Even the document's own doc_id: nothing to gain, and a typo would move the points.
    with pytest.raises(ReservedKeyError):
        store.set_payload("col", "doc-1", {"doc_id": "doc-1"})
    fields = {**date_fields("2016-01-01", "year", year=2016), "date_kind": "period"}
    assert store.set_payload("col", "doc-1", fields) == 2


@pytest.mark.parametrize(
    ("doc_id", "fields", "match"),
    [
        ("", {"a": 1}, "doc_id must be a non-empty string"),
        (None, {"a": 1}, "doc_id must be a non-empty string"),
        (3, {"a": 1}, "doc_id must be a non-empty string"),
        (["d"], {"a": 1}, "doc_id must be a non-empty string"),
        ("d", {}, "fields is empty"),
        ("d", None, "fields must be a mapping"),
        ("d", [("a", 1)], "fields must be a mapping"),
        ("d", "a", "fields must be a mapping"),
        ("d", {3: 1}, "keys of fields must be non-empty strings"),
        ("d", {"": 1}, "keys of fields must be non-empty strings"),
    ],
)
def test_set_payload_refuses_bad_arguments_before_anything_is_sent(doc_id, fields, match):
    client = PayloadClient()
    with pytest.raises(ValueError, match=match) as raised:
        QdrantStore(client=client).set_payload("col", doc_id, fields)
    assert not isinstance(raised.value, StoreError)
    assert client.calls == []


def test_set_payload_takes_any_mapping():
    client = PayloadClient()
    assert QdrantStore(client=client).set_payload("col", "d", MappingProxyType({"title": "x"})) == 2
    assert client.calls[1][1]["payload"] == {"title": "x"}


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("step", ["count_error", "set_error"])
def test_set_payload_refused_api_key_is_a_clear_error(status, step):
    store = QdrantStore(client=PayloadClient(**{step: http_error(status)}))
    with pytest.raises(QdrantAuthError, match=f"HTTP {status}"):
        store.set_payload("col", "d", {"title": "x"})


@pytest.mark.parametrize("step", ["count_error", "set_error"])
def test_set_payload_other_errors_are_raised_unchanged(step):
    error = http_error(500)
    with pytest.raises(UnexpectedResponse) as info:
        QdrantStore(client=PayloadClient(**{step: error})).set_payload("col", "d", {"title": "x"})
    assert info.value is error


def test_set_payload_on_a_missing_collection_raises_and_creates_nothing(memory_store):
    with pytest.raises(ValueError, match="missing"):
        memory_store.set_payload("missing", "a", {"title": "x"})
    assert not memory_store.client.collection_exists("missing")


# --- set_payload: the four date keys go together --------------------------------------
JULY = date_fields("2016-07-01", "month")

DATE_MISUSES = [
    # A date or a precision without the interval: the stored interval would go stale.
    ({"date": "2016-07-01"}, "without date_precision, date_start, date_end"),
    ({"date": "2016-07-01", "date_precision": "month"}, "without date_start, date_end"),
    ({"date_precision": "month"}, "without date, date_start, date_end"),
    ({"date": None}, "without date_precision, date_start, date_end"),  # the old interval would stay
    # The interval alone: the stored date would still pass by itself.
    (date_interval_fields("2016-07-01", "month"), "without date, date_precision"),
    ({"date_start": "2016-07-01T00:00:00Z"}, "without date, date_precision, date_end"),
    ({"date_end": None}, "without date, date_precision, date_start"),
    # A precision change without the date, and a date without its precision.
    ({"date_precision": "year", **date_interval_fields("2016-01-01", "year")}, "without date"),
    ({"date": "2016-07-01", **date_interval_fields("2016-07-01")}, "without date_precision"),
    # The four keys, but an interval that is not the one of the date and the precision.
    ({**JULY, "date_end": "2016-07-01T00:00:00Z"}, "date_end"),
    ({**JULY, "date_start": "2016-01-01T00:00:00Z"}, "date_start"),
    ({**JULY, "date_precision": "year"}, "date_start, date_end"),
    ({**JULY, "date": "2016-08-01"}, "date_start, date_end"),
    ({**JULY, "date_start": None, "date_end": None}, "date_start, date_end"),
    ({**JULY, "date": None}, "date_start, date_end"),  # no date, and an interval all the same
    ({**JULY, "date": "2016-07"}, "date_start, date_end"),  # an unreadable date has no interval
    ({**JULY, "date_end": "2016-07-31"}, "date_end"),  # not the stored form
]


@pytest.mark.parametrize(
    ("fields", "named"), DATE_MISUSES, ids=[repr(sorted(f)) + n for f, n in DATE_MISUSES]
)
def test_set_payload_refuses_date_keys_that_do_not_come_together_and_agree(fields, named):
    client = PayloadClient()
    with pytest.raises(DateFieldsError, match="date_fields") as raised:
        QdrantStore(client=client).set_payload("col", "doc-1", {"title": "kept out of it", **fields})
    assert named in str(raised.value) and "'doc-1'" in str(raised.value)
    assert isinstance(raised.value, StoreError) and isinstance(raised.value, ValueError)
    assert client.calls == []


@pytest.mark.parametrize(
    "fields",
    [
        date_fields("2016-07-01", "month"),
        date_fields("2016-07-01", "month", year=2016),
        date_fields("2016-07-01"),
        date_fields("2016-01-01", "year"),
        date_fields("2016-03-11T00:30:00+02:00", "day"),
        date_fields("2016-07-01", "fortnight"),  # an unknown precision: a day, and kept as given
        date_fields(None),  # the document becomes undated
        date_fields(None, "month", year=None),
        date_fields("2016-03", "month"),  # an unreadable date: no interval
        {**date_fields("2016-07-01", "month"), "date_kind": "meeting", "title": "Minutes"},
        {"date_kind": "meeting", "year": 2016, "title": "Minutes"},  # no date key: nothing to check
    ],
    ids=repr,
)
def test_set_payload_accepts_what_date_fields_builds(fields):
    client = PayloadClient()
    assert QdrantStore(client=client).set_payload("col", "doc-1", fields) == 2
    assert client.calls[1][1]["payload"] == fields


def test_clearing_a_date_removes_the_interval_and_leaves_the_rest(payload_store):
    assert payload_store.set_payload("c", "a", date_fields("2016-07-01", "month", year=2016)) == 3
    assert payload_store.set_payload("c", "a", date_fields(None, year=None)) == 3
    for record in stored(payload_store).values():
        if record.payload["doc_id"] == "a":
            for key in ("date", "date_precision", "date_start", "date_end", "year"):
                assert record.payload.get(key) is None
            assert record.payload["title"] == "Document a"
        else:
            assert record.payload["date"] == "2016-01-01"


def test_a_reserved_key_is_named_before_the_date_keys_are_looked_at():
    with pytest.raises(ReservedKeyError):
        QdrantStore(client=PayloadClient()).set_payload("col", "d", {"text": "t", "date": "2016-07-01"})
