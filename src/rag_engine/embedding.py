"""Embedders: turn texts into vectors, with the prefix scheme of the model.

A retrieval model such as E5 expects a marker in front of every text: one for the passages
that are indexed, another for the search queries. The markers belong to the model, so they
live in :class:`ModelSpec`, and they are applied in one place, :class:`BaseEmbedder`. A backend
only implements :meth:`BaseEmbedder._embed`, which receives texts that already carry their
prefix: a backend cannot forget the prefix, nor apply it twice.

Importing this module is light: it needs numpy only. ``torch`` and ``sentence_transformers``
are imported when a :class:`TorchEmbedder` is built (or :func:`detect_device` is called).

Backends:

* :class:`TorchEmbedder` — sentence-transformers, extra ``rag-engine[torch]``;
* :class:`FakeEmbedder` — deterministic vectors without any model, for tests.

:func:`make_embedder` builds the backend named by the settings.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from rag_engine.config import Settings

logger = logging.getLogger(__name__)

_TORCH_EXTRA = (
    "the torch embedding backend needs the 'torch' and 'sentence-transformers' packages: "
    "install them with: pip install 'rag-engine[torch]'"
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
    """

    name: str
    dim: int
    passage_prefix: str
    query_prefix: str

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
        vectors = np.asarray(self._embed(prefixed), dtype=np.float32)
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

        Returns one vector per text, in order: anything numpy reads as an array of shape
        ``(len(prefixed), dim)``.
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


class FakeEmbedder(BaseEmbedder):
    """Deterministic vectors without any model, for tests. Needs numpy only.

    The vector of a text is derived from a SHA-256 hash of ``seed`` and of the text with its
    prefix, then scaled to unit length. So: the same prefixed text always gives the same
    vector, in any process and on any machine; the same text gives one vector as a passage and
    another as a query, unless the two prefixes are equal. The vectors carry no meaning: two
    similar texts are not close.

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


def make_embedder(settings: Settings, **kwargs: Any) -> Embedder:
    """Build the embedding backend named by ``settings.backend``.

    The model comes from :meth:`ModelSpec.from_settings`, and ``settings.model_cache`` is the
    directory of the model files. ``kwargs`` go to the constructor of the backend and win over
    the settings: ``device`` and ``cache_dir`` for the torch backend.
    """
    spec = ModelSpec.from_settings(settings)
    if settings.backend == "torch":
        kwargs.setdefault("cache_dir", settings.model_cache)
        return TorchEmbedder(spec, **kwargs)
    if settings.backend == "onnx":
        raise NotImplementedError(
            "the onnx embedding backend is not available in this version of rag-engine: "
            "set RAG_ENGINE_BACKEND=torch"
        )
    raise ValueError(f"unknown embedding backend {settings.backend!r} (expected torch or onnx)")
