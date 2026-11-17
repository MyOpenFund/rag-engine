"""Settings of the engine, read from the environment on request.

Importing this module has no side effect: it reads no environment variable and loads no
``.env`` file. A caller that keeps its configuration in a ``.env`` file loads it itself, then
calls :meth:`Settings.from_env`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

BACKENDS = ("torch", "onnx")

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})
_OCR_LANGUAGES = re.compile(r"[A-Za-z][A-Za-z0-9_/-]*(\+[A-Za-z][A-Za-z0-9_/-]*)*")


class ConfigError(ValueError):
    """A setting has an invalid value. The message names the environment variable and the value."""


def _text(environ: Mapping[str, str], name: str) -> str | None:
    """Stripped value of ``name``; ``None`` when the variable is unset or blank."""
    return environ.get(name, "").strip() or None


def _int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = _text(environ, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name}={environ[name]!r} is not an integer") from None


def _bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _text(environ, name)
    if raw is None:
        return default
    text = raw.lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{name}={environ[name]!r} is not a boolean (use true or false)")


@dataclass(frozen=True)
class Settings:
    """Every tunable of the engine, with its default.

    Build it from the environment with :meth:`from_env`, or directly with keyword arguments
    (tests, callers with their own configuration system). Both paths apply the same validation
    and raise :class:`ConfigError` on an invalid value.

    Qdrant connection (same variable names and defaults as the engine this package replaces,
    so existing ``.env`` files keep working):

    ``qdrant_host`` — ``QDRANT_HOST``, default ``"localhost"``
        Host name or address of the Qdrant server.
    ``qdrant_port`` — ``QDRANT_PORT``, default ``6333``
        HTTP port of the Qdrant server, 1 to 65535.
    ``qdrant_api_key`` — ``QDRANT_API_KEY``, default ``None``
        API key, stripped. ``None`` (unset or blank) for a server without authentication.
    ``qdrant_https`` — ``QDRANT_HTTPS``, default ``False``
        Whether to connect over TLS. True values: ``1``, ``true``, ``yes``, ``on``; false
        values: ``0``, ``false``, ``no``, ``off``, blank (case-insensitive). The store always
        passes it to the client explicitly: left undecided, the Qdrant client turns TLS on as
        soon as an API key is set.

    Embedding model:

    ``model_name`` — ``RAG_ENGINE_MODEL``, default ``"intfloat/multilingual-e5-base"``
        Name of the embedding model.
    ``model_dim`` — ``RAG_ENGINE_DIM``, default ``768``
        Dimension of the vectors the model produces, at least 1. Must match the collection.
    ``passage_prefix`` — ``RAG_ENGINE_PASSAGE_PREFIX``, default ``"passage: "``
        Text put in front of every chunk before it is embedded. Never stripped: the trailing
        space is part of the default. A variable set to the empty string means "no prefix";
        only an unset variable gives the default.
    ``query_prefix`` — ``RAG_ENGINE_QUERY_PREFIX``, default ``"query: "``
        Text put in front of every search query before it is embedded. Same rules as
        ``passage_prefix``.
    ``backend`` — ``RAG_ENGINE_BACKEND``, default ``"torch"``
        Embedding backend, ``torch`` or ``onnx`` (case-insensitive in the environment).
    ``model_cache`` — ``RAG_ENGINE_MODEL_CACHE``, default ``None``
        Directory where model files are stored. ``None`` leaves the backend's own default.

    Ingestion:

    ``batch_size`` — ``RAG_ENGINE_BATCH_SIZE``, default ``100``
        Number of chunks embedded and written per batch, at least 1. The default is the value
        of the engine this package replaces.
    ``ocr_languages`` — ``RAG_ENGINE_OCR_LANGUAGES``, default ``"eng"``
        Tesseract languages used when a page is read by OCR: language codes joined by ``+``,
        for example ``"eng+fra"``. This is the value format of the engine this package
        replaces. That engine had no variable for it: it listed every language file found
        in the ``TESSDATA_PREFIX`` directory, and fell back to ``"eng"``. There is therefore
        no older variable name to fall back to, and ``ENABLE_OCR`` / ``TESSDATA_PREFIX`` are
        not read here: neither holds a language list.

    Search:

    ``search_limit`` — ``RAG_ENGINE_SEARCH_LIMIT``, default ``10``
        Number of hits returned when the caller gives no limit, at least 1.
    ``search_max_limit`` — ``RAG_ENGINE_SEARCH_MAX_LIMIT``, default ``50``
        Largest limit a caller may ask for, at least 1 and not below ``search_limit``.

    In the environment, an unset or blank variable gives the default (a blank placeholder
    copied from a template is not an error); the two prefixes are the only exception.
    """

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_https: bool = False
    model_name: str = "intfloat/multilingual-e5-base"
    model_dim: int = 768
    passage_prefix: str = "passage: "
    query_prefix: str = "query: "
    backend: str = "torch"
    batch_size: int = 100
    ocr_languages: str = "eng"
    search_limit: int = 10
    search_max_limit: int = 50
    model_cache: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.qdrant_port <= 65535:
            raise ConfigError(f"QDRANT_PORT={self.qdrant_port!r} is not a port (1 to 65535)")
        for name, value in (
            ("RAG_ENGINE_DIM", self.model_dim),
            ("RAG_ENGINE_BATCH_SIZE", self.batch_size),
            ("RAG_ENGINE_SEARCH_LIMIT", self.search_limit),
            ("RAG_ENGINE_SEARCH_MAX_LIMIT", self.search_max_limit),
        ):
            if value < 1:
                raise ConfigError(f"{name}={value!r} must be at least 1")
        if self.backend not in BACKENDS:
            raise ConfigError(f"RAG_ENGINE_BACKEND={self.backend!r} is not one of {', '.join(BACKENDS)}")
        if not _OCR_LANGUAGES.fullmatch(self.ocr_languages):
            raise ConfigError(
                f"RAG_ENGINE_OCR_LANGUAGES={self.ocr_languages!r} is not a list of Tesseract "
                "language codes joined by '+' (for example 'eng+fra')"
            )
        if self.search_limit > self.search_max_limit:
            raise ConfigError(
                f"RAG_ENGINE_SEARCH_LIMIT={self.search_limit!r} is above "
                f"RAG_ENGINE_SEARCH_MAX_LIMIT={self.search_max_limit!r}"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Read the settings from ``environ`` (default: ``os.environ``, read at call time).

        Raises :class:`ConfigError` naming the variable and its value when one is invalid.
        """
        env = os.environ if environ is None else environ
        return cls(
            qdrant_host=_text(env, "QDRANT_HOST") or cls.qdrant_host,
            qdrant_port=_int(env, "QDRANT_PORT", cls.qdrant_port),
            qdrant_api_key=_text(env, "QDRANT_API_KEY"),
            qdrant_https=_bool(env, "QDRANT_HTTPS", cls.qdrant_https),
            model_name=_text(env, "RAG_ENGINE_MODEL") or cls.model_name,
            model_dim=_int(env, "RAG_ENGINE_DIM", cls.model_dim),
            passage_prefix=env.get("RAG_ENGINE_PASSAGE_PREFIX", cls.passage_prefix),
            query_prefix=env.get("RAG_ENGINE_QUERY_PREFIX", cls.query_prefix),
            backend=(_text(env, "RAG_ENGINE_BACKEND") or cls.backend).lower(),
            batch_size=_int(env, "RAG_ENGINE_BATCH_SIZE", cls.batch_size),
            ocr_languages=_text(env, "RAG_ENGINE_OCR_LANGUAGES") or cls.ocr_languages,
            search_limit=_int(env, "RAG_ENGINE_SEARCH_LIMIT", cls.search_limit),
            search_max_limit=_int(env, "RAG_ENGINE_SEARCH_MAX_LIMIT", cls.search_max_limit),
            model_cache=_text(env, "RAG_ENGINE_MODEL_CACHE"),
        )
