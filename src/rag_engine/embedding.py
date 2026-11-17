"""Embedders: turn texts into vectors, with the prefix scheme of the model.

A retrieval model such as E5 expects a marker in front of every text: one for the passages
that are indexed, another for the search queries. The markers belong to the model, so they
live in :class:`ModelSpec`, and they are applied in one place, :class:`BaseEmbedder`. A backend
only implements :meth:`BaseEmbedder._embed`, which receives texts that already carry their
prefix: a backend cannot forget the prefix, nor apply it twice.

Importing this module is light: it needs numpy only. ``torch`` and ``sentence_transformers``
are imported when a :class:`TorchEmbedder` is built (or :func:`detect_device` is called), and
``fastembed`` when an :class:`OnnxEmbedder` is built.

Backends:

* :class:`TorchEmbedder` — sentence-transformers, extra ``rag-engine[torch]``;
* :class:`OnnxEmbedder` — fastembed on the ONNX runtime, without PyTorch, extra
  ``rag-engine[onnx]``;
* :class:`FakeEmbedder` — deterministic vectors without any model, for tests.

:func:`make_embedder` builds the backend named by the settings. :func:`cosine_parity` compares
the vectors of two backends, text by text.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from rag_engine.config import Settings

__all__ = [
    "ONNX_BATCH_SIZE",
    "ONNX_MODELS",
    "BaseEmbedder",
    "Embedder",
    "FakeEmbedder",
    "ModelSpec",
    "OnnxEmbedder",
    "OnnxModel",
    "TorchEmbedder",
    "cosine_parity",
    "detect_device",
    "make_embedder",
]

logger = logging.getLogger(__name__)

_TORCH_EXTRA = (
    "the torch embedding backend needs the 'torch' and 'sentence-transformers' packages: "
    "install them with: pip install 'rag-engine[torch]'"
)
_ONNX_EXTRA = (
    "the onnx embedding backend needs the 'fastembed' package: "
    "install it with: pip install 'rag-engine[onnx]'"
)


@dataclass(frozen=True)
class ModelSpec:
    """What identifies an embedding model for the engine.

    ``name``
        Name of the model, as the backend knows it (a Hugging Face model id for torch).
    ``dim``
        Dimension of the vectors the model produces. A backend checks it against the model
        it loaded.
    ``passage_prefix``
        Text put in front of every passage before it is embedded; ``""`` for none.
    ``query_prefix``
        Text put in front of every query before it is embedded; ``""`` for none.

    The fields are checked when the specification is built, and nothing is converted:
    ``name`` must be a non-blank string, ``dim`` an ``int`` of at least 1 (not a ``bool``, a
    ``float``, a numeric string or a numpy integer), and each prefix a string, taken as it is.
    Anything else raises ``ValueError`` naming the field.
    """

    name: str
    dim: int
    passage_prefix: str
    query_prefix: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"ModelSpec: name must be a non-blank string, got {self.name!r}")
        if type(self.dim) is not int or self.dim < 1:  # a bool is an int subclass, and is not a size
            raise ValueError(f"ModelSpec: dim must be an integer of at least 1, got {self.dim!r}")
        # ValueError for every field, as for name and dim: one error type for a bad specification.
        not_text = [f for f in ("passage_prefix", "query_prefix") if not isinstance(getattr(self, f), str)]
        if not_text:
            field = not_text[0]
            raise ValueError(
                f"ModelSpec: {field} must be a string ('' for no prefix), got {getattr(self, field)!r}"
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelSpec:
        """The model described by ``settings`` (``model_name``, ``model_dim``, the two prefixes)."""
        return cls(
            name=settings.model_name,
            dim=settings.model_dim,
            passage_prefix=settings.passage_prefix,
            query_prefix=settings.query_prefix,
        )


@runtime_checkable
class Embedder(Protocol):
    """What ingestion and search need from an embedding backend.

    ``encode_passage`` takes a sequence of texts and returns a ``float32`` array of shape
    ``(len(texts), dim)``, rows in the order of the texts. ``encode_query`` takes one text and
    returns a ``float32`` array of shape ``(dim,)``. Both apply the prefix of ``spec``.
    ``release`` frees the model; the embedder cannot be used afterwards.
    """

    @property
    def spec(self) -> ModelSpec: ...

    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def encode_passage(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...

    def release(self) -> None: ...


class BaseEmbedder:
    """The part shared by every backend: argument checks, prefixes, shape and type of the result.

    A backend subclasses it and implements :meth:`_embed`; it overrides :meth:`release` when it
    holds something to free. It can be used as a context manager, which releases it on exit.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def dim(self) -> int:
        """Dimension of the vectors, from the model specification."""
        return self._spec.dim

    @property
    def model_name(self) -> str:
        return self._spec.name

    def encode_passage(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passages for indexing: ``float32`` array of shape ``(len(texts), dim)``.

        ``texts`` must be a sequence of strings (a list, a tuple). A bare string is refused
        with a ``TypeError``: it would be read as one text per character. An empty sequence
        returns an array of shape ``(0, dim)`` and the model is not called.
        """
        if isinstance(texts, (str, bytes, bytearray)) or not isinstance(texts, Sequence):
            raise TypeError(
                "encode_passage takes a sequence of texts (a list or a tuple of str), "
                f"not {type(texts).__name__}; wrap a single text in a list"
            )
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"encode_passage: item {index} is {type(text).__name__}, not str")
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self._spec.passage_prefix
        return self._vectors([prefix + text for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        """Embed one search query: ``float32`` array of shape ``(dim,)``."""
        if not isinstance(text, str):
            raise TypeError(f"encode_query takes one text (str), not {type(text).__name__}")
        return self._vectors([self._spec.query_prefix + text])[0]

    def _vectors(self, prefixed: list[str]) -> np.ndarray:
        raw = self._embed(prefixed)
        if not hasattr(raw, "__array__"):
            # A generator or another iterable of rows: numpy reads a list, not an iterator.
            raw = list(raw)
        vectors = np.asarray(raw, dtype=np.float32)
        expected = (len(prefixed), self.dim)
        if vectors.shape != expected:
            raise ValueError(
                f"embedding model {self.model_name!r} returned an array of shape {vectors.shape}, "
                f"expected {expected}: the model does not produce {self.dim}-dimensional vectors "
                "(check RAG_ENGINE_MODEL and RAG_ENGINE_DIM)"
            )
        return vectors

    def _embed(self, prefixed: list[str]) -> Any:
        """Embed a non-empty list of texts that already carry their prefix, as they are.

        Returns one vector per text, in order: an array of shape ``(len(prefixed), dim)``, or
        any iterable of rows (a list, a generator) that numpy reads as one.
        """
        raise NotImplementedError

    def release(self) -> None:
        """Free what the backend holds. Nothing to free here."""

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(_TORCH_EXTRA) from exc
    return torch


def detect_device() -> str:
    """Best device available to torch: ``"cuda"``, then ``"mps"``, then ``"cpu"``.

    Imports ``torch``; raises an ``ImportError`` naming the extra to install when it is missing.
    """
    torch = _import_torch()
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class TorchEmbedder(BaseEmbedder):
    """sentence-transformers backend (extra ``rag-engine[torch]``).

    ``device``
        Device given to the model; ``None`` picks the best one (:func:`detect_device`).
    ``cache_dir``
        Directory where the model files are stored; ``None`` leaves the default of
        sentence-transformers.

    The model is loaded by the constructor. Its dimension is compared with ``spec.dim``, and a
    difference raises a ``ValueError``. Inference is serialised by a lock, so one instance can
    be shared between threads. Texts are passed to ``SentenceTransformer.encode`` without any
    option, exactly as the engine this package replaces does: batching, and whether the vectors
    are normalised, are those of sentence-transformers and of the model's own configuration.

    Use it as a context manager, or call :meth:`release`, to free the memory of the model::

        with TorchEmbedder(spec) as embedder:
            vectors = embedder.encode_passage(texts)
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(spec)
        torch = _import_torch()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(_TORCH_EXTRA) from exc
        self._torch = torch
        self._lock = threading.Lock()
        self._model: Any = None
        self.device = device or detect_device()
        options: dict[str, Any] = {"device": self.device}
        if cache_dir is not None:
            options["cache_folder"] = os.fspath(cache_dir)
        gc.collect()
        logger.info("Loading embedding model %s on %s", spec.name, self.device)
        self._model = SentenceTransformer(spec.name, **options)
        actual = self._model_dimension()
        if actual is not None and actual != spec.dim:
            self.release()
            raise ValueError(
                f"embedding model {spec.name!r} produces {actual}-dimensional vectors, but the "
                f"configured dimension is {spec.dim} (check RAG_ENGINE_MODEL and RAG_ENGINE_DIM)"
            )
        logger.info("Embedding model ready (dim=%d)", spec.dim)

    def _model_dimension(self) -> int | None:
        """Dimension declared by the loaded model; ``None`` when it declares none."""
        # sentence-transformers renamed the accessor; the newer name is tried first.
        for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            accessor = getattr(self._model, name, None)
            if accessor is not None:
                return accessor()
        return None

    def _embed(self, prefixed: list[str]) -> Any:
        with self._lock:
            if self._model is None:
                raise RuntimeError(
                    "this TorchEmbedder has been released: create a new one instead of reusing it "
                    "after release() or after the end of its 'with' block"
                )
            return self._model.encode(prefixed)

    def release(self) -> None:
        """Free the model and clear the CUDA cache. A second call does nothing.

        Waits for a running inference to finish. Later calls to ``encode_passage`` or
        ``encode_query`` raise a ``RuntimeError``.
        """
        with self._lock:
            if self._model is None:
                return
            self._model = None
            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        logger.info("Embedding model released")


@dataclass(frozen=True)
class OnnxModel:
    """How a model that fastembed does not ship is registered with it.

    ``dim``
        Dimension of the vectors of the model.
    ``model_file``
        Path of the ONNX file inside the Hugging Face repository.
    ``pooling``
        How token vectors become one vector: ``"mean"`` (average over the attention mask, so
        padding does not count), ``"cls"``, ``"last_token"`` or ``"disabled"``.
    ``normalization``
        Whether the vectors are scaled to unit length.
    ``hf``
        Hugging Face repository holding the ONNX file; ``None`` for the model name itself.
    """

    dim: int
    model_file: str = "onnx/model.onnx"
    pooling: str = "mean"
    normalization: bool = True
    hf: str | None = None


ONNX_MODELS: Mapping[str, OnnxModel] = MappingProxyType(
    {
        # The full-precision export published in the model's own repository, with the pipeline
        # of sentence-transformers for this model: mean pooling, then normalisation. The
        # repository also holds quantised variants; they are not used.
        "intfloat/multilingual-e5-base": OnnxModel(dim=768),
    }
)
"""Models :class:`OnnxEmbedder` registers with fastembed when fastembed does not list them."""

ONNX_BATCH_SIZE = 32
"""Default number of texts per inference call of :class:`OnnxEmbedder`."""

_onnx_registration_lock = threading.Lock()


def _fastembed_description(text_embedding: Any, name: str) -> Mapping[str, Any] | None:
    """What fastembed lists for ``name`` (names compare without case); ``None`` when it does not."""
    for description in text_embedding.list_supported_models():
        if str(description.get("model", "")).lower() == name.lower():
            return description
    return None


class OnnxEmbedder(BaseEmbedder):
    """fastembed backend on the ONNX runtime, without PyTorch (extra ``rag-engine[onnx]``).

    ``cache_dir``
        Directory where the model files are stored; ``None`` leaves the default of fastembed
        (``FASTEMBED_CACHE_PATH``, or a directory under the system's temporary directory).
    ``threads``
        Number of threads of the ONNX runtime session; ``None`` leaves its default.
    ``batch_size``
        Number of texts per inference call, at least 1. Texts of one call are padded to the
        longest of them, and the padding is left out of the pooling: the batch size bounds the
        memory, and changes the vectors by rounding noise only.

    A model fastembed lists is used as fastembed defines it. A model it does not list is
    registered from :data:`ONNX_MODELS`; a name found in neither raises a ``ValueError``. The
    dimension fastembed declares is compared with ``spec.dim`` before anything is downloaded.

    Texts are given to fastembed's plain ``embed``, with the prefix of the specification and
    nothing else. Texts longer than the model's limit (512 tokens for E5) are cut by the
    tokenizer, as sentence-transformers does. Inference is serialised by a lock, so one
    instance can be shared between threads.

    The vectors are meant to be compared with those of :class:`TorchEmbedder` for the same
    model: passages indexed with torch can be searched with queries embedded here. They are
    close, not bit-identical; :func:`cosine_parity` measures how close.

    Use it as a context manager, or call :meth:`release`, to free the memory of the model.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        threads: int | None = None,
        batch_size: int = ONNX_BATCH_SIZE,
    ) -> None:
        super().__init__(spec)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError(f"OnnxEmbedder: batch_size must be an integer of at least 1, got {batch_size!r}")
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(_ONNX_EXTRA) from exc
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._model: Any = None
        actual = self._registered_dimension(TextEmbedding)
        if actual is not None and actual != spec.dim:
            raise ValueError(
                f"embedding model {spec.name!r} produces {actual}-dimensional vectors, but the "
                f"configured dimension is {spec.dim} (check RAG_ENGINE_MODEL and RAG_ENGINE_DIM)"
            )
        options: dict[str, Any] = {}
        if cache_dir is not None:
            options["cache_dir"] = os.fspath(cache_dir)
        if threads is not None:
            options["threads"] = threads
        logger.info("Loading ONNX embedding model %s", spec.name)
        self._model = TextEmbedding(spec.name, **options)
        logger.info("ONNX embedding model ready (dim=%d)", spec.dim)

    def _registered_dimension(self, text_embedding: Any) -> int | None:
        """Make sure fastembed knows the model; return the dimension it declares, if any."""
        name = self._spec.name
        # The registry of fastembed is global to the process: two embedders built at the same
        # time must not both register the model (the second registration is an error).
        with _onnx_registration_lock:
            listed = _fastembed_description(text_embedding, name)
            if listed is not None:
                return listed.get("dim")
            known = ONNX_MODELS.get(name)
            if known is None:
                raise ValueError(
                    f"embedding model {name!r} is not available to the onnx backend: fastembed does "
                    f"not list it, and the engine can only register {', '.join(sorted(ONNX_MODELS))} "
                    "(check RAG_ENGINE_MODEL, or set RAG_ENGINE_BACKEND=torch)"
                )
            from fastembed.common.model_description import ModelSource, PoolingType

            text_embedding.add_custom_model(
                model=name,
                pooling=PoolingType[known.pooling.upper()],
                normalization=known.normalization,
                sources=ModelSource(hf=known.hf or name),
                dim=known.dim,
                model_file=known.model_file,
            )
            return known.dim

    def _embed(self, prefixed: list[str]) -> Any:
        with self._lock:
            if self._model is None:
                raise RuntimeError(
                    "this OnnxEmbedder has been released: create a new one instead of reusing it "
                    "after release() or after the end of its 'with' block"
                )
            # fastembed returns a generator that runs the model as it is read: it is read
            # here, inside the lock.
            return list(self._model.embed(prefixed, batch_size=self.batch_size))

    def release(self) -> None:
        """Free the model. A second call does nothing.

        Waits for a running inference to finish. Later calls to ``encode_passage`` or
        ``encode_query`` raise a ``RuntimeError``.
        """
        with self._lock:
            if self._model is None:
                return
            self._model = None
            gc.collect()
        logger.info("ONNX embedding model released")


class FakeEmbedder(BaseEmbedder):
    """Deterministic vectors without any model, for tests. Needs numpy only.

    The vector of a text is derived from a SHA-256 hash of ``seed`` and of the text with its
    prefix, then scaled to unit length. So: the same prefixed text always gives the same
    vector, in any process on the same platform; the same text gives one vector as a passage
    and another as a query, unless the two prefixes are equal. The vectors carry no meaning:
    two similar texts are not close.

    ``calls`` lists every prefixed text received, in order.
    """

    def __init__(self, spec: ModelSpec, *, seed: int = 0) -> None:
        super().__init__(spec)
        self.seed = seed
        self.calls: list[str] = []
        self._released = False

    def _embed(self, prefixed: list[str]) -> np.ndarray:
        if self._released:
            raise RuntimeError("this FakeEmbedder has been released: create a new one")
        self.calls.extend(prefixed)
        return np.stack([self._vector(text) for text in prefixed])

    def _vector(self, text: str) -> np.ndarray:
        key = f"{self.seed}\0{text}".encode("utf-8", "surrogatepass")
        needed = self.dim * 4
        blocks = []
        for counter in range(-(-needed // hashlib.sha256().digest_size)):
            blocks.append(hashlib.sha256(counter.to_bytes(8, "little") + key).digest())
        integers = np.frombuffer(b"".join(blocks)[:needed], dtype="<u4")
        values = integers.astype(np.float64) / 2**31 - 1.0  # in [-1, 1)
        norm = float(np.linalg.norm(values))
        if norm == 0.0:  # every 32-bit word exactly 2**31: not reachable in practice
            values[0], norm = 1.0, 1.0
        return (values / norm).astype(np.float32)

    def release(self) -> None:
        self._released = True


def cosine_parity(a: Any, b: Any) -> np.ndarray:
    """Cosine similarity of each row of ``a`` with the same row of ``b``: ``float64``, shape ``(n,)``.

    ``a`` and ``b`` are two arrays of vectors of the same shape ``(n, dim)``, for example the
    vectors of the same texts from two backends; anything else raises a ``ValueError``. A row
    of zeros has no direction: its cosine is ``nan``.
    """
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError(
            f"cosine_parity takes two arrays of the same shape (n, dim), got {left.shape} and {right.shape}"
        )
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.einsum("ij,ij->i", left, right) / norms


def make_embedder(settings: Settings, **kwargs: Any) -> Embedder:
    """Build the embedding backend named by ``settings.backend``.

    The model comes from :meth:`ModelSpec.from_settings`, and ``settings.model_cache`` is the
    directory of the model files. ``kwargs`` go to the constructor of the backend and win over
    the settings: ``device`` and ``cache_dir`` for the torch backend; ``cache_dir``, ``threads``
    and ``batch_size`` for the onnx backend.
    """
    spec = ModelSpec.from_settings(settings)
    if settings.backend == "torch":
        kwargs.setdefault("cache_dir", settings.model_cache)
        return TorchEmbedder(spec, **kwargs)
    if settings.backend == "onnx":
        kwargs.setdefault("cache_dir", settings.model_cache)
        return OnnxEmbedder(spec, **kwargs)
    raise ValueError(f"unknown embedding backend {settings.backend!r} (expected torch or onnx)")
