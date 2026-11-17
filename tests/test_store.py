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
from types import SimpleNamespace

import httpx
import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

import rag_engine.store as store_mod
from rag_engine.config import Settings
from rag_engine.store import CollectionMismatchError, QdrantAuthError, QdrantStore, StoreError

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
    settings = Settings(qdrant_host="from-settings", qdrant_port=1111, qdrant_api_key="settings-key",
                        qdrant_https=False)
    store = QdrantStore("explicit-host", 2222, SECRET, True, 5, settings=settings)
    assert built == [{"host": "explicit-host", "port": 2222, "api_key": SECRET, "https": True, "timeout": 5}]
    assert (store.host, store.port) == ("explicit-host", 2222)


def test_settings_fill_what_is_not_given(built):
    settings = Settings(qdrant_host="from-settings", qdrant_port=1111, qdrant_api_key=SECRET,
                        qdrant_https=True)
    store = QdrantStore(settings=settings)
    assert built == [{"host": "from-settings", "port": 1111, "api_key": SECRET, "https": True, "timeout": 60}]
    assert (store.host, store.port) == ("from-settings", 1111)


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
    assert built == [{"host": "qdrant.example", "port": 8443, "api_key": SECRET, "https": True, "timeout": 60}]
    assert (store.host, store.port) == ("qdrant.example", 8443)


def test_from_url_http_stays_plain_even_with_a_key_and_a_true_setting(built):
    QdrantStore.from_url("http://localhost:6333/", SECRET, settings=Settings(qdrant_https=True))
    assert built[0]["https"] is False and built[0]["host"] == "localhost"


def test_from_url_without_a_port_uses_the_qdrant_default(built):
    QdrantStore.from_url("http://qdrant.example", settings=Settings(qdrant_port=1111))
    assert built[0]["port"] == 6333


@pytest.mark.parametrize(
    "url",
    ["qdrant.example:6333", "ftp://qdrant.example", "http://", "http://h:6333/prefix", "http://h:notaport",
     "http://h:6333?x=1"],
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


@pytest.mark.parametrize("url", ["http://qdrant.example:6333\n", "\thttp://qdrant.example:6333", "http://h:63\n33"])
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


@pytest.mark.parametrize("url", ["http://[::1", "http://::1:6333", "http://[::1]x:6333", "http://[not-an-address]:6333"])
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
        store = QdrantStore.from_url(f"http://[::1]:{server.server_address[1]}", timeout=5, settings=Settings())
        assert store.is_reachable() is True
    finally:
        server.shutdown()
        server.server_close()


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


@pytest.mark.parametrize("error", [ConnectionError("gone"), http_error(500), http_error(404)],
                         ids=["connection", "500", "404"])
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
    with pytest.raises(QdrantAuthError,
                       match=rf"^Qdrant refused the request \(HTTP {status}\): check QDRANT_API_KEY$") as info:
        store.ensure_collection("c", 8)
    assert info.value.__cause__ is error
    if where != "create":
        assert client.named("create_collection") == []
    else:
        # A refused create is not a lost race: the store does not ask again.
        assert len(client.named("collection_exists")) == 1


# --- the creation race --------------------------------------------------------
@pytest.mark.parametrize("error", [http_error(409), http_error(400), ValueError("Collection c already exists")],
                         ids=["409", "400", "local"])
def test_losing_the_creation_race_is_not_fatal(error):
    store, client = fake_store(exists=[False, True], vectors=unnamed(8), create=error)
    store.ensure_collection("c", 8)
    assert len(client.named("create_collection")) == 1
    assert client.named("get_collection") == [("get_collection", "c")]


@pytest.mark.parametrize("error", [TimeoutError("timed out"), http_error(500), http_error(409)],
                         ids=["timeout", "500", "409"])
def test_a_failed_create_that_left_a_matching_collection_is_logged_without_guessing_who_created_it(error, caplog):
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


@pytest.mark.parametrize("error", [http_error(409), http_error(500), ConnectionError("gone")],
                         ids=["409", "500", "connection"])
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


@pytest.mark.parametrize("error", [http_error(500), http_error(404), ConnectionError("gone")],
                         ids=["500", "404", "connection"])
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


@pytest.mark.parametrize("error", [http_error(500), ConnectionError("gone"), TimeoutError("timed out")],
                         ids=["500", "connection", "timeout"])
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


@pytest.mark.parametrize("error", [ConnectionError("refused"), http_error(401), http_error(500), ValueError("x")],
                         ids=["connection", "401", "500", "other"])
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
