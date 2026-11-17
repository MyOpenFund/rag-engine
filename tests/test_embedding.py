"""Embedders: the shared prefix rule, the deterministic fake, the torch and onnx backends.

The torch backend is tested without torch: stub ``torch`` and ``sentence_transformers`` modules
are placed in ``sys.modules``, so the constructor's lazy imports find them. The onnx backend is
tested the same way, with a stub ``fastembed`` and a stub ``huggingface_hub``. One test, marked
``models``, loads the real model; it runs only where the ``torch`` extra is installed. The
comparison of the two real backends, and the check of the pinned ONNX file, are in
``test_parity.py``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

import rag_engine.embedding as embedding_mod
from rag_engine.config import Settings
from rag_engine.embedding import (
    Embedder,
    FakeEmbedder,
    ModelDownloadError,
    ModelIntegrityError,
    ModelSpec,
    OnnxEmbedder,
    TorchEmbedder,
    cosine_parity,
    detect_device,
    make_embedder,
)

SPEC = ModelSpec(name="stub/model", dim=8, passage_prefix="passage: ", query_prefix="query: ")


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------
def test_model_spec_from_settings_takes_the_four_model_fields():
    settings = Settings(model_name="acme/m", model_dim=12, passage_prefix="P ", query_prefix="Q ")
    assert ModelSpec.from_settings(settings) == ModelSpec(
        name="acme/m", dim=12, passage_prefix="P ", query_prefix="Q "
    )


def test_model_spec_defaults_are_the_e5_scheme():
    spec = ModelSpec.from_settings(Settings())
    assert spec == ModelSpec(
        name="intfloat/multilingual-e5-base", dim=768, passage_prefix="passage: ", query_prefix="query: "
    )


def test_model_spec_is_frozen_and_hashable():
    with pytest.raises(AttributeError):
        SPEC.dim = 3  # type: ignore[misc]
    assert hash(SPEC) == hash(ModelSpec("stub/model", 8, "passage: ", "query: "))


@pytest.mark.parametrize("name", ["", "   ", "\t\n", None, 5, b"stub/model"])
def test_model_spec_refuses_a_name_that_is_not_a_non_blank_string(name):
    with pytest.raises(ValueError, match="name"):
        ModelSpec(name, 8, "passage: ", "query: ")


@pytest.mark.parametrize("dim", [0, -1, True, 8.0, "8", None, np.int64(8)])
def test_model_spec_refuses_a_dim_that_is_not_an_integer_of_at_least_one(dim):
    with pytest.raises(ValueError, match="dim"):
        ModelSpec("stub/model", dim, "passage: ", "query: ")


@pytest.mark.parametrize("field", ["passage_prefix", "query_prefix"])
@pytest.mark.parametrize("prefix", [None, 3, b"passage: ", ["passage: "]])
def test_model_spec_refuses_a_prefix_that_is_not_a_string(field, prefix):
    values = {"name": "stub/model", "dim": 8, "passage_prefix": "p ", "query_prefix": "q ", field: prefix}
    with pytest.raises(ValueError, match=field):
        ModelSpec(**values)


def test_model_spec_accepts_empty_and_whitespace_prefixes_and_a_dim_of_one():
    spec = ModelSpec("stub/model", 1, "", "  ")
    assert (spec.dim, spec.passage_prefix, spec.query_prefix) == (1, "", "  ")


# ---------------------------------------------------------------------------
# FakeEmbedder, and through it the rules shared by every backend
# ---------------------------------------------------------------------------
def test_fake_satisfies_the_protocol():
    fake = FakeEmbedder(SPEC)
    assert isinstance(fake, Embedder)
    assert fake.spec is SPEC
    assert fake.dim == 8
    assert fake.model_name == "stub/model"


def test_an_object_without_the_methods_is_not_an_embedder():
    assert not isinstance(object(), Embedder)


def test_fake_passage_shape_dtype_and_norm():
    vectors = FakeEmbedder(SPEC).encode_passage(["a", "b", "c"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 8)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_fake_query_shape_dtype_and_norm():
    vector = FakeEmbedder(SPEC).encode_query("a")
    assert vector.shape == (8,)
    assert vector.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, rtol=1e-5)


def test_rows_convert_to_lists_of_python_floats():
    # The ingestion code calls ``vectors[i].tolist()`` to build Qdrant points.
    row = FakeEmbedder(SPEC).encode_passage(["a"])[0].tolist()
    assert len(row) == 8
    assert all(type(value) is float for value in row)


def test_fake_is_deterministic_across_instances_and_calls():
    first = FakeEmbedder(SPEC).encode_passage(["a", "b"])
    second = FakeEmbedder(SPEC).encode_passage(["a", "b"])
    np.testing.assert_array_equal(first, second)
    # A text gets the same vector whatever the batch it is in.
    np.testing.assert_array_equal(FakeEmbedder(SPEC).encode_passage(["b"])[0], first[1])


def test_fake_is_deterministic_across_processes():
    code = (
        "from rag_engine.embedding import FakeEmbedder, ModelSpec; "
        "print(FakeEmbedder(ModelSpec('stub/model', 8, 'passage: ', 'query: ')).encode_query('a').tolist())"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(FakeEmbedder(SPEC).encode_query("a").tolist())


def test_fake_seed_changes_the_vectors():
    assert not np.array_equal(
        FakeEmbedder(SPEC, seed=0).encode_query("a"), FakeEmbedder(SPEC, seed=1).encode_query("a")
    )


def test_different_texts_give_different_vectors():
    vectors = FakeEmbedder(SPEC).encode_passage(["a", "b"])
    assert not np.array_equal(vectors[0], vectors[1])


def test_same_text_as_passage_and_as_query_gives_different_vectors():
    fake = FakeEmbedder(SPEC)
    assert not np.array_equal(fake.encode_passage(["x"])[0], fake.encode_query("x"))


def test_equal_prefixed_strings_give_equal_vectors():
    # "passage: " + "x" == "" + "passage: x": the vector depends on the prefixed text only.
    bare = FakeEmbedder(ModelSpec("stub/model", 8, "", ""))
    np.testing.assert_array_equal(
        FakeEmbedder(SPEC).encode_passage(["x"])[0], bare.encode_passage(["passage: x"])[0]
    )
    np.testing.assert_array_equal(FakeEmbedder(SPEC).encode_query("x"), bare.encode_query("query: x"))


def test_fake_records_the_prefixed_strings():
    fake = FakeEmbedder(SPEC)
    fake.encode_passage(["a", "b"])
    fake.encode_query("c")
    assert fake.calls == ["passage: a", "passage: b", "query: c"]


def test_prefix_is_applied_once_and_verbatim():
    spec = ModelSpec("stub/model", 8, "  Passage:\t", "query: ")
    fake = FakeEmbedder(spec)
    # A text that already starts with the prefix still gets it: the engine never guesses.
    fake.encode_passage(["  Passage:\talready", " keep my spaces "])
    fake.encode_query("query: again")
    assert fake.calls == [
        "  Passage:\t  Passage:\talready",
        "  Passage:\t keep my spaces ",
        "query: query: again",
    ]


def test_empty_prefix_changes_nothing():
    fake = FakeEmbedder(ModelSpec("stub/model", 8, "", ""))
    fake.encode_passage(["a"])
    fake.encode_query("b")
    assert fake.calls == ["a", "b"]


def test_empty_sequence_returns_zero_rows_without_calling_the_model():
    fake = FakeEmbedder(SPEC)
    for empty in ([], ()):
        vectors = fake.encode_passage(empty)
        assert vectors.shape == (0, 8)
        assert vectors.dtype == np.float32
    assert fake.calls == []


def test_tuple_of_texts_is_accepted():
    assert FakeEmbedder(SPEC).encode_passage(("a", "b")).shape == (2, 8)


@pytest.mark.parametrize("bad", ["a bare string", b"bytes", None, 3, {"a", "b"}, (t for t in ["a"])])
def test_encode_passage_refuses_anything_but_a_sequence_of_texts(bad):
    fake = FakeEmbedder(SPEC)
    with pytest.raises(TypeError, match="encode_passage"):
        fake.encode_passage(bad)
    assert fake.calls == []


@pytest.mark.parametrize("bad", [["a", None], ["a", 3], [["a"]], [b"a"]])
def test_encode_passage_refuses_a_non_text_item(bad):
    fake = FakeEmbedder(SPEC)
    with pytest.raises(TypeError, match="item 1|item 0"):
        fake.encode_passage(bad)
    assert fake.calls == []


@pytest.mark.parametrize("bad", [["a"], None, 3, b"a"])
def test_encode_query_refuses_anything_but_one_text(bad):
    fake = FakeEmbedder(SPEC)
    with pytest.raises(TypeError, match="encode_query"):
        fake.encode_query(bad)
    assert fake.calls == []


def test_fake_release_then_use_raises():
    fake = FakeEmbedder(SPEC)
    fake.release()
    fake.release()  # twice is fine
    with pytest.raises(RuntimeError, match="released"):
        fake.encode_query("a")
    with pytest.raises(RuntimeError, match="released"):
        fake.encode_passage(["a"])


def test_fake_as_context_manager_releases_on_exit():
    with FakeEmbedder(SPEC) as fake:
        assert fake.encode_query("a").shape == (8,)
    with pytest.raises(RuntimeError, match="released"):
        fake.encode_query("a")


def test_a_backend_returning_the_wrong_shape_is_caught_in_the_shared_code():
    class Broken(embedding_mod.BaseEmbedder):
        def _embed(self, prefixed):
            return np.zeros((len(prefixed), 5), dtype=np.float32)

    with pytest.raises(ValueError, match=r"\(2, 5\).*\(2, 8\)"):
        Broken(SPEC).encode_passage(["a", "b"])


def test_a_backend_yielding_its_rows_is_read_like_a_list():
    class Lazy(embedding_mod.BaseEmbedder):
        def _embed(self, prefixed):
            for index, _ in enumerate(prefixed):
                yield [float(index)] * 8

    vectors = Lazy(SPEC).encode_passage(["a", "b"])
    assert vectors.shape == (2, 8)
    assert vectors.dtype == np.float32
    assert vectors[:, 0].tolist() == [0.0, 1.0]


def test_a_backend_yielding_rows_of_the_wrong_size_gets_the_shape_message():
    class LazyBroken(embedding_mod.BaseEmbedder):
        def _embed(self, prefixed):
            return ([0.0] * 5 for _ in prefixed)

    with pytest.raises(ValueError, match=r"\(2, 5\).*\(2, 8\)"):
        LazyBroken(SPEC).encode_passage(["a", "b"])


def test_a_backend_only_implements_the_raw_encoding():
    seen = []

    class Minimal(embedding_mod.BaseEmbedder):
        def _embed(self, prefixed):
            seen.extend(prefixed)
            return [[1.0] * 8 for _ in prefixed]  # lists of float64: the shared code converts

    backend = Minimal(SPEC)
    assert backend.encode_passage(["a"]).dtype == np.float32
    assert backend.encode_query("b").shape == (8,)
    assert seen == ["passage: a", "query: b"]
    assert isinstance(backend, Embedder)


# ---------------------------------------------------------------------------
# TorchEmbedder against stub modules
# ---------------------------------------------------------------------------
class StubModel:
    """Stands in for ``SentenceTransformer``; records how it is built and called."""

    instances: ClassVar[list[StubModel]] = []
    dimension: int | None = 8
    hold = 0.0  # seconds spent inside encode

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.encode_calls: list[tuple[tuple, dict]] = []
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()
        StubModel.instances.append(self)

    def get_sentence_embedding_dimension(self):
        return type(self).dimension

    def encode(self, *args, **kwargs):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.encode_calls.append((args, kwargs))
        try:
            if type(self).hold:
                time.sleep(type(self).hold)
            sentences = args[0]
            out = np.zeros((len(sentences), 8), dtype=np.float32)
            for row, sentence in enumerate(sentences):
                out[row, 0] = len(sentence)
            return out
        finally:
            with self._guard:
                self.active -= 1


@pytest.fixture
def stubs(monkeypatch):
    """Fake ``torch`` and ``sentence_transformers`` modules; returns the knobs of the fake torch."""
    state = types.SimpleNamespace(cuda=False, mps=False, cache_cleared=0)

    def empty_cache():
        state.cache_cleared += 1

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: state.cuda, empty_cache=empty_cache)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: state.mps))
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = StubModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setattr(StubModel, "instances", [])
    monkeypatch.setattr(StubModel, "dimension", 8)
    monkeypatch.setattr(StubModel, "hold", 0.0)
    return state


def test_torch_embedder_satisfies_the_protocol(stubs):
    embedder = TorchEmbedder(SPEC)
    assert isinstance(embedder, Embedder)
    assert (embedder.spec, embedder.dim, embedder.model_name) == (SPEC, 8, "stub/model")


def test_torch_loads_the_model_named_by_the_spec_on_the_detected_device(stubs):
    embedder = TorchEmbedder(SPEC)
    (model,) = StubModel.instances
    assert model.init_args == ("stub/model",)
    assert model.init_kwargs == {"device": "cpu"}  # no cache argument when none is set
    assert embedder.device == "cpu"


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_detect_device_prefers_cuda_then_mps_then_cpu(stubs, cuda, mps, expected):
    stubs.cuda, stubs.mps = cuda, mps
    assert detect_device() == expected
    assert TorchEmbedder(SPEC).device == expected


def test_detect_device_without_an_mps_backend_falls_back_to_cpu(stubs):
    del sys.modules["torch"].backends.mps  # a torch build that has no MPS backend at all
    assert detect_device() == "cpu"


def test_explicit_device_wins_and_skips_detection(stubs, monkeypatch):
    monkeypatch.setattr(embedding_mod, "detect_device", lambda: pytest.fail("detection must not run"))
    embedder = TorchEmbedder(SPEC, device="cuda:1")
    assert embedder.device == "cuda:1"
    assert StubModel.instances[0].init_kwargs["device"] == "cuda:1"


def test_cache_dir_is_passed_to_the_model(stubs, tmp_path):
    TorchEmbedder(SPEC, cache_dir=str(tmp_path))
    assert StubModel.instances[0].init_kwargs == {"device": "cpu", "cache_folder": str(tmp_path)}


def test_cache_dir_accepts_a_path(stubs, tmp_path):
    TorchEmbedder(SPEC, cache_dir=tmp_path)
    assert StubModel.instances[0].init_kwargs["cache_folder"] == str(tmp_path)


def test_torch_passage_gets_the_passage_prefix(stubs):
    embedder = TorchEmbedder(SPEC)
    vectors = embedder.encode_passage(["ab", "abcd"])
    (model,) = StubModel.instances
    assert model.encode_calls == [((["passage: ab", "passage: abcd"],), {})]
    assert vectors.shape == (2, 8)
    assert vectors.dtype == np.float32
    # Rows come back in input order.
    assert vectors[:, 0].tolist() == [len("passage: ab"), len("passage: abcd")]


def test_torch_query_gets_the_query_prefix(stubs):
    embedder = TorchEmbedder(SPEC)
    vector = embedder.encode_query("ab")
    assert StubModel.instances[0].encode_calls == [((["query: ab"],), {})]
    assert vector.shape == (8,)
    assert vector.dtype == np.float32


def test_torch_prefixes_come_from_the_spec(stubs):
    embedder = TorchEmbedder(ModelSpec("stub/model", 8, "doc> ", ""))
    embedder.encode_passage(["a"])
    embedder.encode_query("b")
    assert [call[0][0] for call in StubModel.instances[0].encode_calls] == [["doc> a"], ["b"]]


def test_torch_encode_passes_no_option_to_the_model(stubs):
    # The engine this package replaces calls ``model.encode(texts)`` and nothing else: no
    # normalisation flag, no batch size. Vectors already stored were produced that way.
    embedder = TorchEmbedder(SPEC)
    embedder.encode_passage(["a"])
    embedder.encode_query("a")
    assert all(kwargs == {} and len(args) == 1 for args, kwargs in StubModel.instances[0].encode_calls)


def test_torch_empty_sequence_does_not_call_the_model(stubs):
    embedder = TorchEmbedder(SPEC)
    assert embedder.encode_passage([]).shape == (0, 8)
    assert StubModel.instances[0].encode_calls == []


def test_torch_bare_string_is_refused(stubs):
    embedder = TorchEmbedder(SPEC)
    with pytest.raises(TypeError, match="encode_passage"):
        embedder.encode_passage("a bare string")
    assert StubModel.instances[0].encode_calls == []


def test_torch_dimension_mismatch_raises_at_load_naming_both(stubs):
    StubModel.dimension = 384
    with pytest.raises(ValueError, match=r"stub/model.*384.*8") as caught:
        TorchEmbedder(SPEC)
    assert "RAG_ENGINE_DIM" in str(caught.value)


def test_torch_dimension_mismatch_clears_the_cuda_cache(stubs):
    stubs.cuda = True
    StubModel.dimension = 384
    with pytest.raises(ValueError):
        TorchEmbedder(SPEC)
    assert stubs.cache_cleared == 1


def test_torch_unknown_model_dimension_is_checked_on_the_first_vectors(stubs):
    StubModel.dimension = None  # some models do not declare their dimension
    spec = ModelSpec("stub/model", 16, "", "")
    embedder = TorchEmbedder(spec)
    with pytest.raises(ValueError, match=r"\(1, 8\).*\(1, 16\)"):
        embedder.encode_passage(["a"])


def test_torch_newer_dimension_accessor_is_used_when_present(stubs, monkeypatch):
    monkeypatch.setattr(StubModel, "get_embedding_dimension", lambda self: 384, raising=False)
    with pytest.raises(ValueError, match="384"):
        TorchEmbedder(SPEC)


def test_torch_release_frees_the_model_and_clears_the_cuda_cache(stubs):
    stubs.cuda = True
    embedder = TorchEmbedder(SPEC)
    embedder.release()
    assert stubs.cache_cleared == 1
    assert embedder._model is None
    embedder.release()  # a second release does nothing
    assert stubs.cache_cleared == 1


def test_torch_release_without_cuda_does_not_touch_the_cuda_cache(stubs):
    embedder = TorchEmbedder(SPEC)
    embedder.release()
    assert stubs.cache_cleared == 0


def test_torch_use_after_release_raises_a_clear_error(stubs):
    embedder = TorchEmbedder(SPEC)
    embedder.release()
    with pytest.raises(RuntimeError, match="released.*new"):
        embedder.encode_passage(["a"])
    with pytest.raises(RuntimeError, match="released"):
        embedder.encode_query("a")
    # The description of the model stays readable.
    assert (embedder.dim, embedder.model_name) == (8, "stub/model")


def test_torch_context_manager_releases_on_exit_even_on_error(stubs):
    with pytest.raises(KeyError), TorchEmbedder(SPEC) as embedder:
        assert embedder.encode_query("a").shape == (8,)
        raise KeyError("boom")
    with pytest.raises(RuntimeError, match="released"):
        embedder.encode_query("a")


def test_torch_inference_is_never_concurrent(stubs):
    StubModel.hold = 0.02
    embedder = TorchEmbedder(SPEC)
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(3):
                if index % 2:
                    embedder.encode_passage(["a", "b"])
                else:
                    embedder.encode_query("a")
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []
    (model,) = StubModel.instances
    assert len(model.encode_calls) == 12
    assert model.max_active == 1


def test_the_stub_does_detect_concurrent_calls():
    # Guards the test above: without a lock, the same workload overlaps.
    StubModel.hold = 0.05
    try:
        model = StubModel()
        barrier = threading.Barrier(2)

        def work() -> None:
            barrier.wait(timeout=5)
            model.encode(["a"])

        threads = [threading.Thread(target=work) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert model.max_active == 2
    finally:
        StubModel.hold = 0.0
        StubModel.instances.clear()


def test_release_waits_for_a_running_inference(stubs):
    stubs.cuda = True
    embedder = TorchEmbedder(SPEC)
    (model,) = StubModel.instances
    entered = threading.Event()  # set when encode has started
    gate = threading.Event()  # encode stays blocked until it is set
    released = threading.Event()  # set when release() has returned
    events: list[str] = []
    results: list[np.ndarray] = []

    def blocking_encode(sentences):
        entered.set()
        if not gate.wait(timeout=10):
            events.append("encode gave up waiting")
        events.append("encode ended")
        return np.zeros((len(sentences), 8), dtype=np.float32)

    def do_release() -> None:
        embedder.release()
        released.set()

    model.encode = blocking_encode
    sys.modules["torch"].cuda.empty_cache = lambda: events.append("cache cleared")
    encoder = threading.Thread(target=lambda: results.append(embedder.encode_query("a")), daemon=True)
    releaser = threading.Thread(target=do_release, daemon=True)
    try:
        encoder.start()
        assert entered.wait(timeout=5)
        releaser.start()
        # release() must still be waiting: it has not returned and has cleared nothing.
        assert not released.wait(timeout=0.2)
        assert events == []
    finally:
        gate.set()
    encoder.join(timeout=5)
    releaser.join(timeout=5)
    assert not encoder.is_alive()
    assert not releaser.is_alive()
    assert released.is_set()
    assert len(results) == 1  # the running call finished with its vector
    assert events == ["encode ended", "cache cleared"]
    with pytest.raises(RuntimeError, match="released"):
        embedder.encode_query("a")


@pytest.mark.parametrize("missing", ["torch", "sentence_transformers"])
def test_missing_dependency_tells_which_extra_to_install(stubs, monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)  # makes ``import <missing>`` fail
    with pytest.raises(ImportError, match=r"rag-engine\[torch\]") as caught:
        TorchEmbedder(SPEC)
    assert isinstance(caught.value.__cause__, ImportError)


def test_detect_device_without_torch_tells_which_extra_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"rag-engine\[torch\]"):
        detect_device()


def test_module_import_and_fake_do_not_load_torch():
    code = (
        "import sys; from rag_engine.embedding import FakeEmbedder, ModelSpec; "
        "FakeEmbedder(ModelSpec('m', 4, 'p ', 'q ')).encode_passage(['a']); "
        "print([m for m in ('torch', 'sentence_transformers', 'fastembed') if m in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# OnnxEmbedder against a stub fastembed
# ---------------------------------------------------------------------------
E5 = "intfloat/multilingual-e5-base"
E5_SPEC = ModelSpec(name=E5, dim=8, passage_prefix="passage: ", query_prefix="query: ")
LISTED_SPEC = ModelSpec(name="listed/model", dim=8, passage_prefix="passage: ", query_prefix="query: ")
CPU_ONLY = ["CPUExecutionProvider"]
REVISION = "0123456789abcdef0123456789abcdef01234567"
ONNX_BYTES = b"the bytes of a stub ONNX export"
SUPPORT_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pinned(**changes):
    """A table entry that pins the stub ONNX file; ``changes`` replace fields."""
    fields = {"dim": 8, "revision": REVISION, "sha256": sha256(ONNX_BYTES), "size": len(ONNX_BYTES)}
    return embedding_mod.OnnxModel(**{**fields, **changes})


class StubHub:
    """Stands in for ``huggingface_hub``: a Hub that holds ``content`` at every revision asked.

    ``snapshot_download`` writes into the directory layout of the real one,
    ``<cache_dir>/models--<owner>--<name>/snapshots/<revision>/``, and records its calls.
    """

    def __init__(self):
        self.content = {"onnx/model.onnx": ONNX_BYTES, **{name: b"{}" for name in SUPPORT_FILES}}
        self.online = True
        self.calls: list[dict] = []

    @property
    def downloads(self) -> list[dict]:
        return [call for call in self.calls if not call["local_files_only"]]

    def snapshot_download(self, repo_id, *, revision, allow_patterns, cache_dir, local_files_only=False):
        self.calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "allow_patterns": list(allow_patterns),
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
            }
        )
        directory = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
        if local_files_only:
            if not directory.is_dir():
                raise FileNotFoundError("Cannot find an appropriate cached snapshot folder")
            return str(directory)
        if not self.online:
            raise ConnectionError("the Hub is unreachable")
        for name in allow_patterns:
            if name in self.content:
                (directory / name).parent.mkdir(parents=True, exist_ok=True)
                (directory / name).write_bytes(self.content[name])
        return str(directory)


def snapshot(cache_dir, repo=E5, revision=REVISION) -> Path:
    return Path(cache_dir) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


class StubPooling:
    """Stands in for fastembed's ``PoolingType`` enumeration."""

    MEAN = "pooling:MEAN"
    CLS = "pooling:CLS"

    def __class_getitem__(cls, name):
        return getattr(cls, name)


class StubSource:
    def __init__(self, hf=None):
        self.hf = hf


class StubTextEmbedding:
    """Stands in for ``fastembed.TextEmbedding``; records how it is built and called."""

    listed: ClassVar[list[dict]] = []
    registrations: ClassVar[list[dict]] = []
    instances: ClassVar[list[StubTextEmbedding]] = []
    hold = 0.0  # seconds spent producing each batch
    width = 8  # size of the rows produced

    @classmethod
    def list_supported_models(cls):
        return [dict(description) for description in cls.listed]

    @classmethod
    def add_custom_model(cls, **kwargs):
        if any(entry["model"].lower() == kwargs["model"].lower() for entry in cls.listed):
            raise ValueError(f"Model {kwargs['model']} is already registered")
        cls.registrations.append(kwargs)
        cls.listed.append({"model": kwargs["model"], "dim": kwargs["dim"]})

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.embed_calls: list[tuple[list[str], dict]] = []
        self.other_calls: list[str] = []
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()
        StubTextEmbedding.instances.append(self)

    def embed(self, documents, **kwargs):
        # A generator, as in fastembed: nothing runs before it is read.
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.embed_calls.append((list(documents), kwargs))
        try:
            if type(self).hold:
                time.sleep(type(self).hold)
            for document in documents:
                row = np.zeros(type(self).width, dtype=np.float32)
                row[0] = len(document)
                yield row
        finally:
            with self._guard:
                self.active -= 1

    def query_embed(self, *args, **kwargs):
        self.other_calls.append("query_embed")
        raise AssertionError("the engine applies the prefixes itself and calls embed only")

    def passage_embed(self, *args, **kwargs):
        self.other_calls.append("passage_embed")
        raise AssertionError("the engine applies the prefixes itself and calls embed only")


@pytest.fixture
def fastembed_stub(monkeypatch, tmp_path):
    """A fake ``fastembed`` package that lists one model of its own, ``listed/model``.

    With it come a fake ``huggingface_hub`` (``fastembed_stub.hub``), a default model directory
    under ``tmp_path`` (``fastembed_stub.default_cache``), and a table entry for the real model
    name that pins the stub ONNX file.
    """
    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = StubTextEmbedding
    common = types.ModuleType("fastembed.common")
    description = types.ModuleType("fastembed.common.model_description")
    description.PoolingType = StubPooling
    description.ModelSource = StubSource
    fastembed.common = common
    common.model_description = description
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setitem(sys.modules, "fastembed.common", common)
    monkeypatch.setitem(sys.modules, "fastembed.common.model_description", description)
    monkeypatch.setattr(StubTextEmbedding, "listed", [{"model": "listed/model", "dim": 8}])
    monkeypatch.setattr(StubTextEmbedding, "registrations", [])
    monkeypatch.setattr(StubTextEmbedding, "instances", [])
    monkeypatch.setattr(StubTextEmbedding, "hold", 0.0)
    monkeypatch.setattr(StubTextEmbedding, "width", 8)
    hub = StubHub()
    hub_module = types.ModuleType("huggingface_hub")
    hub_module.snapshot_download = hub.snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setattr(StubTextEmbedding, "hub", hub, raising=False)
    default_cache = str(tmp_path / "default-model-cache")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", default_cache)
    monkeypatch.setattr(StubTextEmbedding, "default_cache", default_cache, raising=False)
    monkeypatch.setattr(embedding_mod, "_sha256_cache", {})
    # The table entry of the real model, with the small dimension of the stub rows and the
    # hash of the stub file.
    monkeypatch.setattr(embedding_mod, "ONNX_MODELS", {E5: pinned()})
    return StubTextEmbedding


def test_the_table_pins_the_export_of_e5_base_whose_parity_was_measured():
    known = embedding_mod.ONNX_MODELS[E5]
    # The full-precision export, mean pooling over the attention mask, unit-length vectors;
    # the revision, hash and size of the file the parity numbers of the README were measured on.
    assert known == embedding_mod.OnnxModel(
        dim=768,
        revision="d128750597153bb5987e10b1c3493a34e5a4502a",
        sha256="84a4d426f7e87a6bf5bf195f0bae2c4a7d15f675b23ca96f42fab8326d7a77aa",
        size=1_110_059_084,
        model_file="onnx/model.onnx",
        pooling="mean",
        normalization=True,
        hf=None,
    )
    with pytest.raises(TypeError):
        embedding_mod.ONNX_MODELS["other/model"] = known  # type: ignore[index]


def test_a_table_entry_cannot_be_written_without_its_pin():
    with pytest.raises(TypeError):
        embedding_mod.OnnxModel(dim=8)
    with pytest.raises(TypeError):
        embedding_mod.OnnxModel(dim=8, revision=REVISION)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "main"),
        ("revision", REVISION[:12]),
        ("revision", REVISION.upper()),
        ("revision", None),
        ("sha256", "84a4d426"),
        ("sha256", "g" * 64),
        ("sha256", "A" * 64),
        ("sha256", 5),
        ("size", 0),
        ("size", -1),
        ("size", True),
        ("size", 1.5),
        ("size", "1110059084"),
    ],
)
def test_a_table_entry_refuses_a_pin_of_another_form(field, value):
    with pytest.raises(ValueError, match=rf"OnnxModel: {field} "):
        pinned(**{field: value})


def test_onnx_embedder_satisfies_the_protocol(fastembed_stub):
    embedder = OnnxEmbedder(E5_SPEC)
    assert isinstance(embedder, Embedder)
    assert (embedder.spec, embedder.dim, embedder.model_name) == (E5_SPEC, 8, E5)


def test_onnx_registers_a_pinned_model_under_a_name_of_the_engine(fastembed_stub):
    OnnxEmbedder(E5_SPEC)
    (registration,) = fastembed_stub.registrations
    source = registration.pop("sources")
    assert isinstance(source, StubSource) and source.hf == E5
    name = registration.pop("model")
    assert name.startswith(f"rag-engine/{E5}@") and name != E5
    assert registration == {
        "pooling": StubPooling.MEAN,
        "normalization": True,
        "dim": 8,
        "model_file": "onnx/model.onnx",
    }
    assert fastembed_stub.instances[0].init_args == (name,)


def test_onnx_registration_follows_the_table(fastembed_stub, monkeypatch):
    entry = pinned(model_file="onnx/other.onnx", pooling="cls", normalization=False, hf="mirror/repo")
    monkeypatch.setattr(embedding_mod, "ONNX_MODELS", {"acme/model": entry})
    fastembed_stub.hub.content["onnx/other.onnx"] = ONNX_BYTES
    embedder = OnnxEmbedder(ModelSpec("acme/model", 8, "", ""))
    (registration,) = fastembed_stub.registrations
    assert registration["sources"].hf == "mirror/repo"
    assert (registration["model_file"], registration["pooling"], registration["normalization"]) == (
        "onnx/other.onnx",
        StubPooling.CLS,
        False,
    )
    # The files come from the repository of the entry, and the ONNX file is the one it names.
    assert {call["repo_id"] for call in fastembed_stub.hub.calls} == {"mirror/repo"}
    assert fastembed_stub.hub.calls[0]["allow_patterns"][0] == "onnx/other.onnx"
    assert embedder.model_path == snapshot(fastembed_stub.default_cache, "mirror/repo") / "onnx/other.onnx"


def test_onnx_registers_a_model_once_for_several_embedders(fastembed_stub):
    OnnxEmbedder(E5_SPEC)
    OnnxEmbedder(E5_SPEC)
    assert len(fastembed_stub.registrations) == 1
    assert len(fastembed_stub.instances) == 2
    assert fastembed_stub.instances[0].init_args == fastembed_stub.instances[1].init_args


def test_onnx_two_entries_for_one_name_never_share_a_registration(fastembed_stub, monkeypatch):
    # The registry of fastembed outlives a change of the table (a test, a reload): the entry
    # that runs is the one of the table now, not the one registered first under that name.
    OnnxEmbedder(E5_SPEC)
    monkeypatch.setattr(embedding_mod, "ONNX_MODELS", {E5: pinned(pooling="cls")})
    OnnxEmbedder(E5_SPEC)
    first, second = fastembed_stub.registrations
    assert first["model"] != second["model"]
    assert (first["pooling"], second["pooling"]) == (StubPooling.MEAN, StubPooling.CLS)
    assert fastembed_stub.instances[1].init_args == (second["model"],)


def test_onnx_a_pinned_model_is_loaded_from_the_table_even_when_fastembed_lists_the_name(
    fastembed_stub, caplog
):
    # A later fastembed could ship the same name with another file, pooling or dimension.
    fastembed_stub.listed.append({"model": E5, "dim": 999})
    with caplog.at_level(logging.WARNING, logger="rag_engine.embedding"):
        embedder = OnnxEmbedder(E5_SPEC)
    (registration,) = fastembed_stub.registrations
    (model,) = fastembed_stub.instances
    assert model.init_args == (registration["model"],) != (E5,)
    assert model.init_kwargs["specific_model_path"] == str(snapshot(fastembed_stub.default_cache))
    assert embedder.model_sha256 == sha256(ONNX_BYTES)
    assert caplog.records == []


def test_onnx_the_table_is_read_whatever_the_case_of_the_name(fastembed_stub):
    # fastembed compares names without their case: so does the table, or a name in capitals
    # would fall through to whatever fastembed lists.
    fastembed_stub.listed.append({"model": E5, "dim": 8})
    embedder = OnnxEmbedder(ModelSpec(E5.upper(), 8, "", ""))
    assert embedder.model_sha256 == sha256(ONNX_BYTES)
    assert fastembed_stub.instances[0].init_args == (fastembed_stub.registrations[0]["model"],)


def test_onnx_a_model_only_fastembed_lists_is_used_as_listed_with_a_warning(fastembed_stub, caplog):
    with caplog.at_level(logging.WARNING, logger="rag_engine.embedding"):
        embedder = OnnxEmbedder(LISTED_SPEC)
    assert fastembed_stub.registrations == []
    assert fastembed_stub.instances[0].init_args == ("listed/model",)
    assert fastembed_stub.instances[0].init_kwargs == {"providers": CPU_ONLY}
    assert fastembed_stub.hub.calls == []  # fastembed downloads it itself
    assert (embedder.model_path, embedder.model_sha256) == (None, None)
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "listed/model" in record.getMessage()
    assert "not pinned or verified" in record.getMessage()


# --- the pinned files: cache, download, hash ----------------------------------


def test_onnx_downloads_the_pinned_revision_when_the_cache_is_empty(fastembed_stub):
    embedder = OnnxEmbedder(E5_SPEC)
    cache = fastembed_stub.default_cache
    wanted = {
        "repo_id": E5,
        "revision": REVISION,
        "allow_patterns": ["onnx/model.onnx", *SUPPORT_FILES],
        "cache_dir": cache,
    }
    # The cache is asked first, without the network; then the Hub, for the same files.
    assert fastembed_stub.hub.calls == [
        {**wanted, "local_files_only": True},
        {**wanted, "local_files_only": False},
    ]
    assert embedder.model_path == snapshot(cache) / "onnx/model.onnx"
    assert embedder.model_path.read_bytes() == ONNX_BYTES
    assert embedder.model_sha256 == sha256(ONNX_BYTES)
    # fastembed is handed the directory, so it downloads nothing itself, and runs on CPU.
    assert fastembed_stub.instances[0].init_kwargs == {
        "providers": CPU_ONLY,
        "cache_dir": cache,
        "specific_model_path": str(snapshot(cache)),
    }


def test_onnx_a_filled_cache_is_used_without_any_request_to_the_hub(fastembed_stub):
    OnnxEmbedder(E5_SPEC)
    assert len(fastembed_stub.hub.downloads) == 1
    fastembed_stub.hub.online = False
    embedder = OnnxEmbedder(E5_SPEC)
    assert len(fastembed_stub.hub.downloads) == 1
    assert embedder.model_sha256 == sha256(ONNX_BYTES)


@pytest.mark.parametrize("lost", ["onnx/model.onnx", "tokenizer.json", "special_tokens_map.json"])
def test_onnx_a_cache_that_lacks_a_file_is_completed(fastembed_stub, lost):
    OnnxEmbedder(E5_SPEC)
    (snapshot(fastembed_stub.default_cache) / lost).unlink()
    OnnxEmbedder(E5_SPEC)
    assert len(fastembed_stub.hub.downloads) == 2
    assert (snapshot(fastembed_stub.default_cache) / lost).is_file()


def test_onnx_download_failure_says_what_from_where_and_how_to_do_without(fastembed_stub, tmp_path):
    fastembed_stub.hub.online = False
    with pytest.raises(ModelDownloadError) as caught:
        OnnxEmbedder(E5_SPEC, cache_dir=tmp_path / "models")
    message = str(caught.value)
    for part in (
        repr(E5),
        "onnx/model.onnx",
        "tokenizer.json",
        "Hugging Face",
        REVISION,
        f"https://huggingface.co/{E5}/tree/{REVISION}",
        str(tmp_path / "models"),
        "RAG_ENGINE_MODEL_CACHE",
        "ConnectionError: the Hub is unreachable",
    ):
        assert part in message
    assert isinstance(caught.value.__cause__, ConnectionError)
    assert fastembed_stub.instances == []
    assert fastembed_stub.registrations == []


def test_onnx_a_download_that_leaves_a_file_out_is_an_error(fastembed_stub):
    del fastembed_stub.hub.content["tokenizer_config.json"]
    with pytest.raises(ModelDownloadError, match=r"tokenizer_config\.json missing"):
        OnnxEmbedder(E5_SPEC)
    assert fastembed_stub.instances == []


def test_onnx_without_huggingface_hub_tells_which_extra_to_install(fastembed_stub, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r"rag-engine\[onnx\]"):
        OnnxEmbedder(E5_SPEC)


def test_onnx_a_file_with_another_hash_is_refused_before_the_model_is_loaded(fastembed_stub):
    altered = ONNX_BYTES[:-1] + b"?"  # same size, one byte changed
    fastembed_stub.hub.content["onnx/model.onnx"] = altered
    with pytest.raises(ModelIntegrityError) as caught:
        OnnxEmbedder(E5_SPEC)
    message = str(caught.value)
    assert repr(E5) in message
    assert f"Expected sha256 {sha256(ONNX_BYTES)}" in message
    assert f"found sha256 {sha256(altered)}" in message
    assert REVISION in message
    assert str(snapshot(fastembed_stub.default_cache) / "onnx/model.onnx") in message
    assert "parity must be re-measured" in message and "ONNX_MODELS" in message
    # Nothing was registered with fastembed, and no model was built: no inference is possible.
    assert fastembed_stub.registrations == []
    assert fastembed_stub.instances == []


def test_onnx_a_file_altered_in_the_cache_is_refused_on_the_next_load(fastembed_stub):
    OnnxEmbedder(E5_SPEC)
    path = snapshot(fastembed_stub.default_cache) / "onnx/model.onnx"
    path.write_bytes(b"X" + ONNX_BYTES[1:])
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ModelIntegrityError):
        OnnxEmbedder(E5_SPEC)
    assert len(fastembed_stub.instances) == 1


def test_onnx_a_file_of_another_size_is_refused(fastembed_stub):
    fastembed_stub.hub.content["onnx/model.onnx"] = ONNX_BYTES + b"\0"
    with pytest.raises(
        ModelIntegrityError, match=rf"\({len(ONNX_BYTES)} bytes\).*\({len(ONNX_BYTES) + 1} bytes\)"
    ):
        OnnxEmbedder(E5_SPEC)
    assert fastembed_stub.instances == []


def test_onnx_the_expected_hash_is_the_one_of_the_table(fastembed_stub, monkeypatch):
    monkeypatch.setattr(embedding_mod, "ONNX_MODELS", {E5: pinned(sha256="0" * 64)})
    with pytest.raises(ModelIntegrityError, match=f"Expected sha256 {'0' * 64}"):
        OnnxEmbedder(E5_SPEC)
    assert fastembed_stub.instances == []


def test_the_hash_of_a_file_is_computed_once_for_a_path_size_and_mtime(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_mod, "_sha256_cache", {})
    path = tmp_path / "model.onnx"
    path.write_bytes(b"first")
    before = path.stat()
    assert embedding_mod._file_sha256(path) == (sha256(b"first"), 5)
    # Same path, size and modification time: the file is not read again.
    path.write_bytes(b"other")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert embedding_mod._file_sha256(path) == (sha256(b"first"), 5)
    # A newer modification time, or another size: it is hashed again.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert embedding_mod._file_sha256(path) == (sha256(b"other"), 5)
    path.write_bytes(b"longer")
    assert embedding_mod._file_sha256(path) == (sha256(b"longer"), 6)


def test_the_hash_cache_follows_a_link_to_the_file_it_points_to(tmp_path, monkeypatch):
    # The Hugging Face cache stores a snapshot as links into a blob directory.
    monkeypatch.setattr(embedding_mod, "_sha256_cache", {})
    blob = tmp_path / "blob"
    blob.write_bytes(b"first")
    link = tmp_path / "model.onnx"
    link.symlink_to(blob)
    assert embedding_mod._file_sha256(link) == (sha256(b"first"), 5)
    assert [key[0] for key in embedding_mod._sha256_cache] == [os.path.realpath(blob)]


def test_onnx_unknown_model_names_the_models_it_can_register(fastembed_stub):
    with pytest.raises(ValueError, match=r"'nobody/unknown'.*intfloat/multilingual-e5-base") as caught:
        OnnxEmbedder(ModelSpec("nobody/unknown", 8, "", ""))
    assert "RAG_ENGINE_MODEL" in str(caught.value)
    assert fastembed_stub.registrations == []
    assert fastembed_stub.instances == []


def test_onnx_passage_and_query_get_their_prefix_once_through_plain_embed(fastembed_stub):
    embedder = OnnxEmbedder(E5_SPEC)
    vectors = embedder.encode_passage(["ab", "abcd"])
    vector = embedder.encode_query("ab")
    (model,) = fastembed_stub.instances
    assert [call[0] for call in model.embed_calls] == [["passage: ab", "passage: abcd"], ["query: ab"]]
    assert model.other_calls == []
    assert vectors.shape == (2, 8) and vectors.dtype == np.float32
    assert vectors[:, 0].tolist() == [len("passage: ab"), len("passage: abcd")]
    assert vector.shape == (8,) and vector.dtype == np.float32


def test_onnx_prefixes_come_from_the_spec(fastembed_stub):
    embedder = OnnxEmbedder(ModelSpec(E5, 8, "doc> ", ""))
    embedder.encode_passage(["a"])
    embedder.encode_query("b")
    assert [call[0] for call in fastembed_stub.instances[0].embed_calls] == [["doc> a"], ["b"]]


def test_onnx_generator_is_read_before_the_lock_is_left(fastembed_stub):
    embedder = OnnxEmbedder(E5_SPEC)
    raw = embedder._embed(["passage: a", "passage: b"])
    assert isinstance(raw, list) and len(raw) == 2
    assert fastembed_stub.instances[0].active == 0  # the generator has run to its end


def test_onnx_batch_size_is_passed_to_fastembed(fastembed_stub):
    OnnxEmbedder(E5_SPEC).encode_passage(["a"])
    OnnxEmbedder(E5_SPEC, batch_size=4).encode_passage(["a"])
    first, second = fastembed_stub.instances
    assert first.embed_calls[0][1] == {"batch_size": embedding_mod.ONNX_BATCH_SIZE} == {"batch_size": 32}
    assert second.embed_calls[0][1] == {"batch_size": 4}


@pytest.mark.parametrize("batch_size", [0, -1, True, 2.0, "8", None])
def test_onnx_refuses_a_batch_size_that_is_not_an_integer_of_at_least_one(fastembed_stub, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        OnnxEmbedder(E5_SPEC, batch_size=batch_size)
    assert fastembed_stub.instances == []


def test_onnx_runs_on_the_cpu_provider_only_and_leaves_the_threads_to_the_runtime(fastembed_stub):
    OnnxEmbedder(E5_SPEC)
    OnnxEmbedder(LISTED_SPEC)
    for model in fastembed_stub.instances:
        assert model.init_kwargs["providers"] == CPU_ONLY
        assert "cuda" not in model.init_kwargs and "device_ids" not in model.init_kwargs
        assert "threads" not in model.init_kwargs


def test_onnx_cache_dir_and_threads_are_passed_to_fastembed(fastembed_stub, tmp_path):
    OnnxEmbedder(E5_SPEC, cache_dir=tmp_path, threads=2)
    assert fastembed_stub.instances[0].init_kwargs == {
        "providers": CPU_ONLY,
        "cache_dir": str(tmp_path),
        "specific_model_path": str(snapshot(tmp_path)),
        "threads": 2,
    }
    assert fastembed_stub.hub.calls[0]["cache_dir"] == str(tmp_path)
    assert not Path(fastembed_stub.default_cache).exists()


def test_onnx_cache_dir_and_threads_reach_a_model_only_fastembed_lists(fastembed_stub, tmp_path):
    OnnxEmbedder(LISTED_SPEC, cache_dir=tmp_path, threads=2)
    assert fastembed_stub.instances[0].init_kwargs == {
        "providers": CPU_ONLY,
        "cache_dir": str(tmp_path),
        "threads": 2,
    }


@pytest.mark.parametrize("threads", [0, -1, True, 2.0, "2"])
def test_onnx_refuses_threads_that_are_not_an_integer_of_at_least_one(fastembed_stub, threads):
    with pytest.raises(ValueError, match="threads must be None or an integer of at least 1"):
        OnnxEmbedder(E5_SPEC, threads=threads)
    assert fastembed_stub.instances == []


def test_onnx_empty_sequence_does_not_call_the_model(fastembed_stub):
    embedder = OnnxEmbedder(E5_SPEC)
    assert embedder.encode_passage([]).shape == (0, 8)
    assert fastembed_stub.instances[0].embed_calls == []


@pytest.mark.parametrize("spec", [ModelSpec(E5, 16, "", ""), ModelSpec("listed/model", 16, "", "")])
def test_onnx_dimension_mismatch_raises_before_the_model_is_loaded(fastembed_stub, spec):
    with pytest.raises(ValueError, match=rf"{spec.name}.*8.*16") as caught:
        OnnxEmbedder(spec)
    assert "RAG_ENGINE_DIM" in str(caught.value)
    assert fastembed_stub.instances == []
    assert fastembed_stub.hub.calls == []  # refused before anything is looked up or downloaded


def test_onnx_model_without_a_declared_dimension_is_checked_on_the_first_vectors(fastembed_stub):
    fastembed_stub.listed.append({"model": "bare/model"})
    embedder = OnnxEmbedder(ModelSpec("bare/model", 16, "", ""))
    with pytest.raises(ValueError, match=r"\(1, 8\).*\(1, 16\)"):
        embedder.encode_passage(["a"])


def test_onnx_rows_of_the_wrong_size_get_the_shape_message(fastembed_stub):
    embedder = OnnxEmbedder(E5_SPEC)
    fastembed_stub.width = 5
    with pytest.raises(ValueError, match=r"\(2, 5\).*\(2, 8\)"):
        embedder.encode_passage(["a", "b"])


def test_onnx_release_then_use_raises_a_clear_error(fastembed_stub):
    embedder = OnnxEmbedder(E5_SPEC)
    embedder.release()
    assert embedder._model is None
    embedder.release()  # a second release does nothing
    with pytest.raises(RuntimeError, match="OnnxEmbedder.*released.*new"):
        embedder.encode_passage(["a"])
    with pytest.raises(RuntimeError, match="released"):
        embedder.encode_query("a")
    assert (embedder.dim, embedder.model_name) == (8, E5)


def test_onnx_context_manager_releases_on_exit_even_on_error(fastembed_stub):
    with pytest.raises(KeyError), OnnxEmbedder(E5_SPEC) as embedder:
        assert embedder.encode_query("a").shape == (8,)
        raise KeyError("boom")
    with pytest.raises(RuntimeError, match="released"):
        embedder.encode_query("a")


def test_onnx_inference_is_never_concurrent(fastembed_stub):
    fastembed_stub.hold = 0.02
    embedder = OnnxEmbedder(E5_SPEC)
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(3):
                if index % 2:
                    embedder.encode_passage(["a", "b"])
                else:
                    embedder.encode_query("a")
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []
    (model,) = fastembed_stub.instances
    assert len(model.embed_calls) == 12
    assert model.max_active == 1


def test_the_fastembed_stub_does_detect_concurrent_calls(fastembed_stub):
    # Guards the test above: without a lock, the same workload overlaps.
    fastembed_stub.hold = 0.05
    model = fastembed_stub()
    barrier = threading.Barrier(2)

    def work() -> None:
        barrier.wait(timeout=5)
        list(model.embed(["a"]))

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert model.max_active == 2


def test_onnx_release_waits_for_a_running_inference(fastembed_stub):
    embedder = OnnxEmbedder(E5_SPEC)
    (model,) = fastembed_stub.instances
    entered, gate, released = threading.Event(), threading.Event(), threading.Event()
    results: list[np.ndarray] = []

    def blocking_embed(documents, **_):
        entered.set()
        gate.wait(timeout=10)
        yield from (np.zeros(8, dtype=np.float32) for _ in documents)

    def do_release() -> None:
        embedder.release()
        released.set()

    model.embed = blocking_embed
    encoder = threading.Thread(target=lambda: results.append(embedder.encode_query("a")), daemon=True)
    releaser = threading.Thread(target=do_release, daemon=True)
    try:
        encoder.start()
        assert entered.wait(timeout=5)
        releaser.start()
        assert not released.wait(timeout=0.2)  # release() is still waiting for the inference
    finally:
        gate.set()
    encoder.join(timeout=5)
    releaser.join(timeout=5)
    assert released.is_set()
    assert len(results) == 1  # the running call finished with its vector


def test_onnx_missing_dependency_tells_which_extra_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # makes ``import fastembed`` fail
    with pytest.raises(ImportError, match=r"rag-engine\[onnx\]") as caught:
        OnnxEmbedder(E5_SPEC)
    assert isinstance(caught.value.__cause__, ImportError)


def test_onnx_backend_does_not_load_torch(fastembed_stub, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert OnnxEmbedder(E5_SPEC).encode_query("a").shape == (8,)


# ---------------------------------------------------------------------------
# cosine_parity
# ---------------------------------------------------------------------------
def test_cosine_parity_is_the_cosine_of_each_pair_of_rows():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[2.0, 0.0], [0.0, 5.0], [4.0, 3.0], [-1.0, 0.0]], dtype=np.float32)
    result = cosine_parity(a, b)
    assert result.shape == (4,) and result.dtype == np.float64
    np.testing.assert_allclose(result, [1.0, 0.0, 0.96, -1.0], atol=1e-12)


def test_cosine_parity_accepts_lists_and_does_not_need_unit_vectors():
    np.testing.assert_allclose(cosine_parity([[10.0, 0.0]], [[0.1, 0.0]]), [1.0])


def test_cosine_parity_of_a_zero_row_is_not_a_number():
    result = cosine_parity([[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]])
    assert np.isnan(result[0]) and result[1] == 1.0
    assert not (result >= 0.999).all()  # a comparison with a tolerance fails, it does not pass


@pytest.mark.parametrize(
    ("a", "b"),
    [([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]), ([1.0, 0.0], [1.0, 0.0]), ([[1.0, 0.0]], [[1.0, 0.0, 0.0]])],
)
def test_cosine_parity_refuses_arrays_that_are_not_two_matching_tables(a, b):
    with pytest.raises(ValueError, match="cosine_parity"):
        cosine_parity(a, b)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_factory_builds_the_torch_backend_from_settings(stubs, tmp_path):
    settings = Settings(
        model_name="stub/model",
        model_dim=8,
        passage_prefix="P ",
        query_prefix="Q ",
        model_cache=str(tmp_path),
    )
    embedder = make_embedder(settings)
    assert type(embedder) is TorchEmbedder
    assert embedder.spec == ModelSpec("stub/model", 8, "P ", "Q ")
    assert StubModel.instances[0].init_kwargs == {"device": "cpu", "cache_folder": str(tmp_path)}


def test_factory_without_a_model_cache_passes_none(stubs):
    make_embedder(Settings(model_name="stub/model", model_dim=8))
    assert StubModel.instances[0].init_kwargs == {"device": "cpu"}


def test_factory_keyword_arguments_reach_the_backend_and_win(stubs, tmp_path):
    settings = Settings(model_name="stub/model", model_dim=8, model_cache="/from/settings")
    embedder = make_embedder(settings, device="mps", cache_dir=str(tmp_path))
    assert embedder.device == "mps"
    assert StubModel.instances[0].init_kwargs == {"device": "mps", "cache_folder": str(tmp_path)}


def test_factory_builds_the_onnx_backend_from_settings(fastembed_stub, tmp_path):
    settings = Settings(
        backend="onnx",
        model_name=E5,
        model_dim=8,
        passage_prefix="P ",
        query_prefix="Q ",
        model_cache=str(tmp_path),
    )
    embedder = make_embedder(settings)
    assert type(embedder) is OnnxEmbedder
    assert embedder.spec == ModelSpec(E5, 8, "P ", "Q ")
    assert fastembed_stub.instances[0].init_kwargs == {
        "providers": CPU_ONLY,
        "cache_dir": str(tmp_path),
        "specific_model_path": str(snapshot(tmp_path)),
    }


def test_factory_onnx_without_a_model_cache_leaves_the_default_of_fastembed(fastembed_stub):
    make_embedder(Settings(backend="onnx", model_name=E5, model_dim=8))
    assert fastembed_stub.instances[0].init_kwargs["cache_dir"] == fastembed_stub.default_cache
    make_embedder(Settings(backend="onnx", model_name="listed/model", model_dim=8))
    assert fastembed_stub.instances[1].init_kwargs == {"providers": CPU_ONLY}


def test_factory_passes_the_onnx_threads_of_the_settings(fastembed_stub):
    make_embedder(Settings(backend="onnx", model_name=E5, model_dim=8, onnx_threads=2))
    assert fastembed_stub.instances[0].init_kwargs["threads"] == 2
    make_embedder(Settings(backend="onnx", model_name=E5, model_dim=8))
    assert "threads" not in fastembed_stub.instances[1].init_kwargs


def test_factory_does_not_give_the_onnx_threads_to_the_torch_backend(stubs):
    embedder = make_embedder(Settings(backend="torch", model_name="stub/model", model_dim=8, onnx_threads=2))
    assert type(embedder) is TorchEmbedder


def test_factory_keyword_arguments_reach_the_onnx_backend_and_win(fastembed_stub, tmp_path):
    settings = Settings(
        backend="onnx", model_name=E5, model_dim=8, model_cache="/from/settings", onnx_threads=8
    )
    embedder = make_embedder(settings, cache_dir=str(tmp_path), threads=3, batch_size=4)
    assert fastembed_stub.instances[0].init_kwargs == {
        "providers": CPU_ONLY,
        "cache_dir": str(tmp_path),
        "specific_model_path": str(snapshot(tmp_path)),
        "threads": 3,
    }
    assert embedder.batch_size == 4


def test_factory_refuses_an_unknown_backend():
    settings = Settings()
    object.__setattr__(settings, "backend", "tensorflow")  # cannot be built through Settings
    with pytest.raises(ValueError, match="tensorflow"):
        make_embedder(settings)


# ---------------------------------------------------------------------------
# The real model (needs the ``torch`` extra and the model files)
# ---------------------------------------------------------------------------
@pytest.mark.models
def test_real_model_gives_the_vectors_of_sentence_transformers_with_the_e5_prefixes():
    sentence_transformers = pytest.importorskip("sentence_transformers")
    settings = Settings()
    long_text = " ".join(f"word{index}" for index in range(600))  # beyond the 512 tokens of the model
    texts = ["", "Le chat dort sur le canapé près de la fenêtre.", "A short English sentence.", long_text]
    with make_embedder(settings) as embedder:
        assert embedder.dim == 768
        passages = embedder.encode_passage(texts)
        queries = [embedder.encode_query(text) for text in texts]
        device = embedder.device
    options = {"device": device}
    if settings.model_cache is not None:
        options["cache_folder"] = settings.model_cache
    reference = sentence_transformers.SentenceTransformer(settings.model_name, **options)

    assert passages.shape == (len(texts), 768)
    assert passages.dtype == np.float32
    assert np.array_equal(passages, reference.encode(["passage: " + text for text in texts]))
    assert np.abs(np.linalg.norm(passages, axis=1) - 1.0).max() < 1e-4
    for text, query in zip(texts, queries, strict=True):
        assert query.shape == (768,)
        assert query.dtype == np.float32
        assert np.array_equal(query, reference.encode("query: " + text))
        assert abs(float(np.linalg.norm(query)) - 1.0) < 1e-4
    # The two prefixes give different vectors, and the truncated text is not a degenerate row.
    assert not np.allclose(passages, np.stack(queries), atol=1e-4)
    assert not np.allclose(passages[3], passages[2], atol=1e-4)
