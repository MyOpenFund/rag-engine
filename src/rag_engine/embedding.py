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
* :class:`OnnxEmbedder` — fastembed on the ONNX runtime (CPU), without PyTorch, extra
  ``rag-engine[onnx]``. The files of a model of :data:`ONNX_MODELS` (the ONNX file, the
  tokenizer and the configuration) are pinned to one revision, and each is checked against
  its SHA-256 before the model is loaded;
* :class:`FakeEmbedder` — deterministic vectors without any model, for tests.

:func:`make_embedder` builds the backend named by the settings. :func:`cosine_parity` compares
the vectors of two backends, text by text.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from rag_engine.errors import RagEngineError

if TYPE_CHECKING:
    from rag_engine.config import Settings

__all__ = [
    "ONNX_BATCH_SIZE",
    "ONNX_MODELS",
    "BaseEmbedder",
    "Embedder",
    "FakeEmbedder",
    "ModelDownloadError",
    "ModelIntegrityError",
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


class ModelIntegrityError(RagEngineError):
    """A model file is not the one the engine pins: its size or its SHA-256 differs.

    Raised before the model is loaded, for the ONNX file as for a tokenizer or configuration
    file. The message names the model, the file, the hash expected and the hash found.
    """


class ModelDownloadError(RagEngineError):
    """The pinned files of a model are not in the model cache and could not be downloaded.

    The message names the files, the repository and the revision they come from, and the
    directory they were looked for in.
    """


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


_ONNX_SUPPORT_FILES = ("config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")
"""Files fastembed reads next to the ONNX file. They decide the tokens, so the vectors."""

_HEXADECIMAL = frozenset("0123456789abcdef")


def _is_hexadecimal(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and not set(value) - _HEXADECIMAL


def _is_repository_path(name: object) -> bool:
    """Whether ``name`` is the path of a file inside a repository: relative, ``/``-separated."""
    if not isinstance(name, str) or not name.strip() or "\\" in name:
        return False
    return all(part not in ("", ".", "..") for part in name.split("/"))


@dataclass(frozen=True)
class OnnxModel:
    """One ONNX export the engine vouches for: where it comes from, and how it is run.

    ``dim``
        Dimension of the vectors of the model.
    ``revision``
        Commit of the Hugging Face repository the files are taken from: the full hash of 40
        hexadecimal digits, never a branch or a tag, which can move.
    ``files``
        Every file the engine loads for the model: a mapping from the path of the file inside
        the Hugging Face repository to the pair ``(sha256, size)`` — its SHA-256, 64 lower-case
        hexadecimal digits, and its size in bytes. It must name ``model_file`` and the files
        fastembed reads next to it: ``config.json``, ``special_tokens_map.json``,
        ``tokenizer.json`` and ``tokenizer_config.json``. The tokenizer decides the vectors as
        much as the weights do, so no file is trusted on its name. Every file is hashed
        before the model is loaded, and a different hash or size is a
        :class:`ModelIntegrityError`. A file listed beyond the required ones is fetched and
        verified in the same way.
    ``model_file``
        Path of the ONNX file inside the Hugging Face repository.
    ``pooling``
        How token vectors become one vector: ``"mean"`` (average over the attention mask, so
        padding does not count), ``"cls"``, ``"last_token"`` or ``"disabled"``.
    ``normalization``
        Whether the vectors are scaled to unit length.
    ``hf``
        Hugging Face repository holding the files; ``None`` for the model name itself.

    ``revision`` and ``files`` have no default: an export cannot be listed without being
    pinned. A value of another form, or a required file left out of ``files``, raises a
    ``ValueError`` naming the field and the file.

    ``files`` is kept as a read-only mapping of the entry's own, the ONNX file first and the
    others by name, whatever the order they were given in. ``sha256`` and ``size`` are those
    of the ONNX file.

    **The revision and the files identify what the vectors compared with those of
    sentence-transformers were computed with.** Change them only after the parity test
    (``pytest -m models``) has passed on the new files.
    """

    dim: int
    revision: str
    files: Mapping[str, tuple[str, int]]
    model_file: str = "onnx/model.onnx"
    pooling: str = "mean"
    normalization: bool = True
    hf: str | None = None

    def __post_init__(self) -> None:
        if not _is_hexadecimal(self.revision, 40):
            raise ValueError(
                f"OnnxModel: revision must be 40 lower-case hexadecimal digits, got {self.revision!r}"
            )
        if not isinstance(self.files, Mapping) or not self.files:
            raise ValueError(
                "OnnxModel: files must be a non-empty mapping from the path of a file to its "
                f"(sha256, size), got {self.files!r}"
            )
        for name, value in self.files.items():
            if not _is_repository_path(name):
                raise ValueError(
                    f"OnnxModel: files: {name!r} is not the path of a file inside the repository "
                    "(relative, with '/' between its parts)"
                )
            if (
                not isinstance(value, (tuple, list))
                or len(value) != 2
                or not _is_hexadecimal(value[0], 64)
                or type(value[1]) is not int  # a bool is an int subclass, and is not a size
                or value[1] < 1
            ):
                raise ValueError(
                    f"OnnxModel: files[{name!r}] must be (sha256, size): 64 lower-case hexadecimal "
                    f"digits and a number of bytes of at least 1, got {value!r}"
                )
        for required in (self.model_file, *_ONNX_SUPPORT_FILES):
            if required not in self.files:
                raise ValueError(
                    f"OnnxModel: files has no entry for {required!r}: every file loaded for the model "
                    "is pinned by its SHA-256 and its size"
                )
        # One order whatever was written, so that equal entries have the same repr (the
        # registration name is derived from it): the ONNX file, then the others by name.
        names = [self.model_file, *sorted(name for name in self.files if name != self.model_file)]
        ordered = {name: (self.files[name][0], self.files[name][1]) for name in names}
        object.__setattr__(self, "files", MappingProxyType(ordered))

    def __hash__(self) -> int:
        # Written out: the mapping of the files is not hashable by itself.
        return hash(
            (
                self.dim,
                self.revision,
                tuple(self.files.items()),
                self.model_file,
                self.pooling,
                self.normalization,
                self.hf,
            )
        )

    @property
    def sha256(self) -> str:
        """SHA-256 of the ONNX file."""
        return self.files[self.model_file][0]

    @property
    def size(self) -> int:
        """Size of the ONNX file, in bytes."""
        return self.files[self.model_file][1]


ONNX_MODELS: Mapping[str, OnnxModel] = MappingProxyType(
    {
        # The full-precision export published in the model's own repository, with the pipeline
        # of sentence-transformers for this model: mean pooling, then normalisation. The
        # repository also holds quantised variants; they are not used.
        # Parity with sentence-transformers was measured with exactly these files (cosine 1.0,
        # see the README). Do not change the revision, a hash or a size before the parity test
        # (pytest -m models) has passed on the new files.
        "intfloat/multilingual-e5-base": OnnxModel(
            dim=768,
            revision="d128750597153bb5987e10b1c3493a34e5a4502a",
            files={
                "onnx/model.onnx": (
                    "84a4d426f7e87a6bf5bf195f0bae2c4a7d15f675b23ca96f42fab8326d7a77aa",
                    1_110_059_084,
                ),
                "config.json": (
                    "9dab198f24c8c0879e481cf7822005d5ecbceedbacb390ffafa594e28d31bac4",
                    694,
                ),
                "special_tokens_map.json": (
                    "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f",
                    280,
                ),
                "tokenizer.json": (
                    "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
                    17_082_660,
                ),
                "tokenizer_config.json": (
                    "efb5c0d09722e5fe59a462cd2a9976ee216d55b037597d997cd3fe833216da15",
                    418,
                ),
            },
        ),
    }
)
"""The models :class:`OnnxEmbedder` pins: for each name, the one ONNX export it loads.

A name of this table is always loaded from the table, whatever fastembed lists under that name.
"""

ONNX_BATCH_SIZE = 32
"""Default number of texts per inference call of :class:`OnnxEmbedder`."""

_CPU_PROVIDERS = ("CPUExecutionProvider",)

_onnx_registration_lock = threading.Lock()
_sha256_lock = threading.Lock()
_sha256_cache: dict[tuple[str, int, int], str] = {}


def _fastembed_description(text_embedding: Any, name: str) -> Mapping[str, Any] | None:
    """What fastembed lists for ``name`` (names compare without case); ``None`` when it does not."""
    for description in text_embedding.list_supported_models():
        if str(description.get("model", "")).lower() == name.lower():
            return description
    return None


def _pinned_model(name: str) -> OnnxModel | None:
    """The entry of :data:`ONNX_MODELS` for ``name``, compared without case as fastembed does."""
    for known_name, known in ONNX_MODELS.items():
        if known_name.lower() == name.lower():
            return known
    return None


def _onnx_cache_dir(cache_dir: str | os.PathLike[str] | None) -> str:
    """The model directory: ``cache_dir``, else the default of fastembed."""
    if cache_dir is not None:
        return os.fspath(cache_dir)
    return os.environ.get("FASTEMBED_CACHE_PATH") or os.path.join(tempfile.gettempdir(), "fastembed_cache")


def _pinned_model_dir(name: str, known: OnnxModel, cache_dir: str) -> Path:
    """Directory holding the files of ``known`` at its pinned revision, downloaded if needed.

    The files already in ``cache_dir`` are used without any request to the Hugging Face Hub.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(_ONNX_EXTRA) from exc
    repository = known.hf or name
    files = list(known.files)
    request: dict[str, Any] = {"revision": known.revision, "allow_patterns": files, "cache_dir": cache_dir}

    def missing_from(directory: Path) -> list[str]:
        return [file for file in files if not (directory / file).is_file()]

    try:
        cached = Path(snapshot_download(repository, local_files_only=True, **request))
    except Exception:  # noqa: BLE001 - not in the cache, whatever the form: the download reports
        cached = None
    if cached is not None and not missing_from(cached):
        return cached
    source = (
        f"{', '.join(files)} of the Hugging Face repository {repository!r} at revision {known.revision} "
        f"(https://huggingface.co/{repository}/tree/{known.revision})"
    )
    logger.info("Downloading %s into %s", source, cache_dir)
    try:
        downloaded = Path(snapshot_download(repository, **request))
    except Exception as exc:
        raise ModelDownloadError(
            f"embedding model {name!r}: could not download {source} into {cache_dir!r}: "
            f"{type(exc).__name__}: {exc}. On a host without access to huggingface.co, set "
            "RAG_ENGINE_MODEL_CACHE to a directory already filled with these files (the directory "
            "of a host where the model was loaded once can be copied as it is)"
        ) from exc
    missing = missing_from(downloaded)
    if missing:
        raise ModelDownloadError(
            f"embedding model {name!r}: {', '.join(missing)} missing from {os.fspath(downloaded)!r} after "
            f"the download of {source}"
        )
    return downloaded


def _file_sha256(path: Path) -> tuple[str, int]:
    """SHA-256 and size of a file. Computed once per process for a given path, size and mtime."""
    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    key = (resolved, stat.st_size, stat.st_mtime_ns)
    # Hashing inside the lock: embedders built at the same time hash the file once.
    with _sha256_lock:
        digest = _sha256_cache.get(key)
        if digest is None:
            hasher = hashlib.sha256()
            with open(resolved, "rb") as stream:
                while block := stream.read(1 << 20):
                    hasher.update(block)
            digest = _sha256_cache[key] = hasher.hexdigest()
    return digest, stat.st_size


def _verify_pinned_files(name: str, known: OnnxModel, directory: Path) -> str:
    """Check every file about to be loaded against the table; return the SHA-256 of the ONNX file.

    The first file that differs raises :class:`ModelIntegrityError`, which names it.
    """
    for file, (expected, expected_size) in known.files.items():
        path = directory / file
        found, size = _file_sha256(path)
        if found != expected or size != expected_size:
            raise ModelIntegrityError(
                f"embedding model {name!r}: the file {file!r} in the model cache, {os.fspath(path)!r}, is "
                f"not the pinned one. Expected sha256 {expected} ({expected_size} bytes), revision "
                f"{known.revision} of {known.hf or name!r}; found sha256 {found} ({size} bytes). The "
                "vectors this file gives were never compared with those of sentence-transformers, so "
                "the model is not loaded. If the file is damaged, delete "
                # The real file, not the link of the snapshot: the Hub client keeps a blob it finds.
                f"{os.path.realpath(path)!r} and it is downloaded again. If the model is meant to "
                "change, parity must be re-measured (pytest -m models) before "
                "rag_engine.embedding.ONNX_MODELS is updated"
            )
    return known.sha256


def _register_pinned(text_embedding: Any, name: str, known: OnnxModel) -> str:
    """Register ``known`` with fastembed; return the name it is registered under.

    The name is the engine's own, not the model's: fastembed looks a model up by its name, and
    would give its own definition for a name it lists itself. The name carries a fingerprint of
    the whole table entry, so two different entries never share a registration.
    """
    fingerprint = hashlib.sha256(repr(known).encode()).hexdigest()[:16]
    registered = f"rag-engine/{name}@{fingerprint}"
    # The registry of fastembed is global to the process: two embedders built at the same
    # time must not both register the model (the second registration is an error).
    with _onnx_registration_lock:
        if _fastembed_description(text_embedding, registered) is None:
            from fastembed.common.model_description import ModelSource, PoolingType

            text_embedding.add_custom_model(
                model=registered,
                pooling=PoolingType[known.pooling.upper()],
                normalization=known.normalization,
                sources=ModelSource(hf=known.hf or name),
                dim=known.dim,
                model_file=known.model_file,
            )
    return registered


class OnnxEmbedder(BaseEmbedder):
    """fastembed backend on the ONNX runtime, on CPU, without PyTorch (extra ``rag-engine[onnx]``).

    ``cache_dir``
        Directory where the model files are stored; ``None`` leaves the default of fastembed
        (``FASTEMBED_CACHE_PATH``, or a directory under the system's temporary directory).
    ``threads``
        Number of threads of the ONNX runtime session, at least 1; ``None`` leaves its default,
        one thread per physical core of the *host*. In a container with a CPU limit, give the
        limit: the runtime does not see it, and more threads than CPUs slow inference down.
    ``batch_size``
        Number of texts per inference call, at least 1. Texts of one call are padded to the
        longest of them, and the padding is left out of the pooling: the batch size bounds the
        memory, and changes the vectors by rounding noise only.

    **A model of** :data:`ONNX_MODELS` **is pinned and verified.** Its files are those of the
    revision of the table: found in ``cache_dir``, or else downloaded from the Hugging Face Hub
    (:class:`ModelDownloadError` when that fails). When they are all in the cache, no request is
    made to the Hub. Every one of them is then hashed — the ONNX file, and the tokenizer and
    configuration files, which decide the vectors just as much — and the model is built only
    if each has the SHA-256 and the size of the table (:class:`ModelIntegrityError` naming
    the file otherwise): what runs is what the vectors compared with those of
    sentence-transformers were computed with. The hash of a file is computed once per process
    (about a second per gigabyte) and kept for the same path, size and modification time.
    ``model_path`` and ``model_sha256`` give the ONNX file loaded and its hash.
    fastembed is handed that directory and does not download anything, and the name is
    registered with it under a name of the engine's own, so that a model fastembed would list
    under the same name with another source or pooling is never used instead.

    A model that is not in the table but that fastembed lists is used as fastembed defines and
    downloads it, and a warning is logged: the engine neither pins nor verifies its files
    (``model_path`` and ``model_sha256`` are ``None``). A name found in neither raises a
    ``ValueError``. The declared dimension is compared with ``spec.dim`` before anything is
    downloaded.

    The session runs on the CPU execution provider only, also when the GPU build of
    onnxruntime is installed: parity was measured on CPU.

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
        if threads is not None and (type(threads) is not int or threads < 1):
            raise ValueError(
                f"OnnxEmbedder: threads must be None or an integer of at least 1, got {threads!r}"
            )
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(_ONNX_EXTRA) from exc
        self.batch_size = batch_size
        self.model_path: Path | None = None
        self.model_sha256: str | None = None
        self._lock = threading.Lock()
        self._model: Any = None
        # The CPU provider is named: left to itself, fastembed picks CUDA when onnxruntime has it.
        options: dict[str, Any] = {"providers": list(_CPU_PROVIDERS)}
        if threads is not None:
            options["threads"] = threads
        known = _pinned_model(spec.name)
        if known is not None:
            self._check_dimension(known.dim)
            cache = _onnx_cache_dir(cache_dir)
            model_dir = _pinned_model_dir(spec.name, known, cache)
            self.model_path = model_dir / known.model_file
            self.model_sha256 = _verify_pinned_files(spec.name, known, model_dir)
            registered = _register_pinned(TextEmbedding, spec.name, known)
            # specific_model_path: fastembed loads this directory and downloads nothing.
            options.update(cache_dir=cache, specific_model_path=os.fspath(model_dir))
            logger.info(
                "Loading ONNX embedding model %s (revision %s, %d files verified, sha256 of %s: %s)",
                spec.name,
                known.revision,
                len(known.files),
                known.model_file,
                self.model_sha256,
            )
        else:
            listed = _fastembed_description(TextEmbedding, spec.name)
            if listed is None:
                raise ValueError(
                    f"embedding model {spec.name!r} is not available to the onnx backend: it is not "
                    f"in rag_engine.embedding.ONNX_MODELS ({', '.join(sorted(ONNX_MODELS))}) and "
                    "fastembed does not list it (check RAG_ENGINE_MODEL, or set RAG_ENGINE_BACKEND=torch)"
                )
            self._check_dimension(listed.get("dim"))
            registered = spec.name
            if cache_dir is not None:
                options["cache_dir"] = os.fspath(cache_dir)
            logger.warning(
                "ONNX embedding model %s is not in rag_engine.embedding.ONNX_MODELS: fastembed "
                "downloads and loads it as it defines it; its files are not pinned or verified by the "
                "engine, and its parity with the torch backend was not measured",
                spec.name,
            )
        self._model = TextEmbedding(registered, **options)
        logger.info("ONNX embedding model ready (dim=%d)", spec.dim)

    def _check_dimension(self, declared: int | None) -> None:
        if declared is not None and declared != self._spec.dim:
            raise ValueError(
                f"embedding model {self._spec.name!r} produces {declared}-dimensional vectors, but the "
                f"configured dimension is {self._spec.dim} (check RAG_ENGINE_MODEL and RAG_ENGINE_DIM)"
            )

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
    and ``batch_size`` for the onnx backend, whose ``threads`` default to ``settings.onnx_threads``.
    """
    spec = ModelSpec.from_settings(settings)
    if settings.backend == "torch":
        kwargs.setdefault("cache_dir", settings.model_cache)
        return TorchEmbedder(spec, **kwargs)
    if settings.backend == "onnx":
        kwargs.setdefault("cache_dir", settings.model_cache)
        kwargs.setdefault("threads", settings.onnx_threads)
        return OnnxEmbedder(spec, **kwargs)
    raise ValueError(f"unknown embedding backend {settings.backend!r} (expected torch or onnx)")
