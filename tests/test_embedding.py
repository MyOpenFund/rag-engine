"""Embedders: the shared prefix rule, the deterministic fake, and the torch backend.

The torch backend is tested without torch: stub ``torch`` and ``sentence_transformers`` modules
are placed in ``sys.modules``, so the constructor's lazy imports find them. One test, marked
``models``, loads the real model; it runs only where the ``torch`` extra is installed.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import types
from typing import ClassVar

import numpy as np
import pytest

import rag_engine.embedding as embedding_mod
from rag_engine.config import Settings
from rag_engine.embedding import (
    Embedder,
    FakeEmbedder,
    ModelSpec,
    TorchEmbedder,
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
    StubModel.hold = 0.1
    embedder = TorchEmbedder(SPEC)
    results = []
    thread = threading.Thread(target=lambda: results.append(embedder.encode_query("a")))
    thread.start()
    model = StubModel.instances[0]
    deadline = time.monotonic() + 5
    while model.active == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    embedder.release()
    thread.join(timeout=10)
    assert len(results) == 1  # the running call finished with its vector


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


def test_factory_onnx_backend_is_not_available_yet():
    with pytest.raises(NotImplementedError, match="onnx.*RAG_ENGINE_BACKEND=torch"):
        make_embedder(Settings(backend="onnx"))


def test_factory_refuses_an_unknown_backend():
    settings = Settings()
    object.__setattr__(settings, "backend", "tensorflow")  # cannot be built through Settings
    with pytest.raises(ValueError, match="tensorflow"):
        make_embedder(settings)


# ---------------------------------------------------------------------------
# The real model (needs the ``torch`` extra and the model files)
# ---------------------------------------------------------------------------
@pytest.mark.models
def test_real_model_dimension_and_prefixes():
    pytest.importorskip("sentence_transformers")
    with make_embedder(Settings()) as embedder:
        assert embedder.dim == 768
        passages = embedder.encode_passage(["x"])
        query = embedder.encode_query("x")
        assert passages.shape == (1, 768)
        assert passages.dtype == np.float32
        assert query.shape == (768,)
        assert not np.allclose(passages[0], query, atol=1e-4)
