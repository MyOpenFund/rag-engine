"""Settings of the engine, read from the environment on request.

Importing this module has no side effect: it reads no environment variable and loads no
``.env`` file. A caller that keeps its configuration in a ``.env`` file loads it itself, then
calls :meth:`Settings.from_env`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields

BACKENDS = ("torch", "onnx")
NO_PREFIX = "none"

_ECHO_LIMIT = 80

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})
_OCR_LANGUAGES = re.compile(r"[A-Za-z][A-Za-z0-9_/-]*(\+[A-Za-z][A-Za-z0-9_/-]*)*")


class ConfigError(ValueError):
    """A setting has an invalid value.

    The message names the environment variable of the setting and echoes the value, cut to
    80 characters. The Qdrant API key is the exception: its value is never echoed.
    """


def _show(value: object) -> str:
    """``repr`` of a value for an error message, cut so a huge value cannot flood the logs."""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - e.g. an integer too large to print; never mask the ConfigError
        text = f"<{type(value).__name__}>"
    if len(text) > _ECHO_LIMIT:
        return text[:_ECHO_LIMIT] + "…"
    return text


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
        raise ConfigError(f"{name}={_show(environ[name])} is not an integer") from None


def _bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _text(environ, name)
    if raw is None:
        return default
    text = raw.lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{name}={_show(environ[name])} is not a boolean (use true or false)")


def _backend(environ: Mapping[str, str], name: str, default: str) -> str:
    raw = _text(environ, name)
    if raw is None:
        return default
    backend = raw.lower()
    if backend not in BACKENDS:
        raise ConfigError(f"{name}={_show(environ[name])} is not one of {', '.join(BACKENDS)}")
    return backend


def _prefix(environ: Mapping[str, str], name: str, default: str) -> str:
    """Unset or empty: the default. ``none``: no prefix. Anything else: the value, verbatim."""
    raw = environ.get(name, "")
    if raw == "":
        return default
    if raw.strip().lower() == NO_PREFIX:
        return ""
    return raw


@dataclass(frozen=True, repr=False)
class Settings:
    """Every tunable of the engine, with its default.

    Build it from the environment with :meth:`from_env`, or directly with keyword arguments
    (tests, callers with their own configuration system). Both paths raise
    :class:`ConfigError` on an invalid value, and apply the same range checks. They differ in
    how lenient they are about the form of a value:

    * :meth:`from_env` parses text: it strips surrounding whitespace, reads integers and
      booleans from their text form, lower-cases the backend, and replaces a blank value by
      the default. The two prefixes have their own rule, given below.
    * Direct construction takes values as they are and converts nothing: an integer must be an
      ``int`` (not a ``bool``, a ``float`` or a numeric string), ``qdrant_https`` must be a
      ``bool`` (the string ``"false"`` is refused, it would be truthy), every other field must
      be a ``str``, and ``backend`` must be exactly ``"torch"`` or ``"onnx"``. Strings are not
      stripped. An empty or whitespace-only ``qdrant_host`` or ``model_name`` is refused; an
      empty or whitespace-only ``qdrant_api_key`` or ``model_cache`` becomes ``None``.

    The API key is never printed: ``repr()`` and ``str()`` show ``qdrant_api_key=<set>`` or
    ``qdrant_api_key=<unset>``, and no error message echoes it. It is still a plain attribute,
    and ``dataclasses.asdict`` returns it.

    Qdrant connection (same variable names and defaults as the engine this package replaces,
    so existing ``.env`` files keep working):

    ``qdrant_host`` — ``QDRANT_HOST``, default ``"localhost"``
        Host name or address of the Qdrant server.
    ``qdrant_port`` — ``QDRANT_PORT``, default ``6333``
        HTTP port of the Qdrant server, 1 to 65535.
    ``qdrant_api_key`` — ``QDRANT_API_KEY``, default ``None``
        API key, stripped when read from the environment. ``None`` (unset or blank) for a
        server without authentication.
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
        Text put in front of every chunk before it is embedded; the trailing space is part
        of the default. In the environment: an unset or empty variable gives the default
        (a compose file that passes an unset variable through delivers the empty string, and
        must not silently remove the prefix); the value ``none`` (case-insensitive, surrounding
        whitespace ignored) means "no prefix" and gives the empty string; any other value is
        taken verbatim, whitespace included, so a whitespace-only value is a prefix made of
        that whitespace. In direct construction the string is the prefix as it is: ``""`` is
        "no prefix" and ``"none"`` is the literal text. A prefix that differs from the one the
        collection was built with produces vectors that do not match it.
    ``query_prefix`` — ``RAG_ENGINE_QUERY_PREFIX``, default ``"query: "``
        Text put in front of every search query before it is embedded. Same rules as
        ``passage_prefix``.
    ``backend`` — ``RAG_ENGINE_BACKEND``, default ``"torch"``
        Embedding backend, ``torch`` or ``onnx``. Case-insensitive and stripped in the
        environment; matched exactly in direct construction.
    ``model_cache`` — ``RAG_ENGINE_MODEL_CACHE``, default ``None``
        Directory where model files are stored. ``None`` (unset or blank) leaves the
        backend's own default.

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
    copied from a template is not an error). The two prefixes follow their own rule above:
    only an unset or empty variable gives the default, a whitespace-only one is a prefix.
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
        self._check_types()
        if not 1 <= self.qdrant_port <= 65535:
            raise ConfigError(f"QDRANT_PORT={_show(self.qdrant_port)} is not a port (1 to 65535)")
        for name, value in (
            ("RAG_ENGINE_DIM", self.model_dim),
            ("RAG_ENGINE_BATCH_SIZE", self.batch_size),
            ("RAG_ENGINE_SEARCH_LIMIT", self.search_limit),
            ("RAG_ENGINE_SEARCH_MAX_LIMIT", self.search_max_limit),
        ):
            if value < 1:
                raise ConfigError(f"{name}={_show(value)} must be at least 1")
        for name, value in (("QDRANT_HOST", self.qdrant_host), ("RAG_ENGINE_MODEL", self.model_name)):
            if not value.strip():
                raise ConfigError(f"{name}={_show(value)} must not be empty")
        if self.backend not in BACKENDS:
            raise ConfigError(f"RAG_ENGINE_BACKEND={_show(self.backend)} is not one of {', '.join(BACKENDS)}")
        if not _OCR_LANGUAGES.fullmatch(self.ocr_languages):
            raise ConfigError(
                f"RAG_ENGINE_OCR_LANGUAGES={_show(self.ocr_languages)} is not a list of Tesseract "
                "language codes joined by '+' (for example 'eng+fra')"
            )
        if self.search_limit > self.search_max_limit:
            raise ConfigError(
                f"RAG_ENGINE_SEARCH_LIMIT={_show(self.search_limit)} is above "
                f"RAG_ENGINE_SEARCH_MAX_LIMIT={_show(self.search_max_limit)}"
            )

    def _check_types(self) -> None:
        """Refuse a value of the wrong type; turn a blank optional string into ``None``."""
        for name, value in (
            ("QDRANT_HOST", self.qdrant_host),
            ("RAG_ENGINE_MODEL", self.model_name),
            ("RAG_ENGINE_PASSAGE_PREFIX", self.passage_prefix),
            ("RAG_ENGINE_QUERY_PREFIX", self.query_prefix),
            ("RAG_ENGINE_BACKEND", self.backend),
            ("RAG_ENGINE_OCR_LANGUAGES", self.ocr_languages),
        ):
            if not isinstance(value, str):
                raise ConfigError(f"{name}={_show(value)} must be a string, not {type(value).__name__}")
        for name, value in (
            ("QDRANT_PORT", self.qdrant_port),
            ("RAG_ENGINE_DIM", self.model_dim),
            ("RAG_ENGINE_BATCH_SIZE", self.batch_size),
            ("RAG_ENGINE_SEARCH_LIMIT", self.search_limit),
            ("RAG_ENGINE_SEARCH_MAX_LIMIT", self.search_max_limit),
        ):
            if type(value) is not int:  # a bool is an int subclass, and is not a count
                raise ConfigError(f"{name}={_show(value)} must be an integer, not {type(value).__name__}")
        if type(self.qdrant_https) is not bool:
            raise ConfigError(
                f"QDRANT_HTTPS={_show(self.qdrant_https)} must be a boolean (True or False), "
                f"not {type(self.qdrant_https).__name__}"
            )
        if self.qdrant_api_key is not None:
            if not isinstance(self.qdrant_api_key, str):
                # The value is deliberately left out: it may be the key in another form.
                raise ConfigError(
                    f"QDRANT_API_KEY must be a string or None, not {type(self.qdrant_api_key).__name__}"
                )
            if not self.qdrant_api_key.strip():
                object.__setattr__(self, "qdrant_api_key", None)
        if self.model_cache is not None:
            if not isinstance(self.model_cache, str):
                raise ConfigError(
                    f"RAG_ENGINE_MODEL_CACHE={_show(self.model_cache)} must be a string or None, "
                    f"not {type(self.model_cache).__name__}"
                )
            if not self.model_cache.strip():
                object.__setattr__(self, "model_cache", None)

    def __repr__(self) -> str:
        """Every field and its value, except the API key: only whether one is set."""
        parts = []
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "qdrant_api_key":
                parts.append(f"{field.name}={'<unset>' if value is None else '<set>'}")
            else:
                parts.append(f"{field.name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Read the settings from ``environ`` (default: ``os.environ``, read at call time).

        Raises :class:`ConfigError` naming the variable and its value when one is invalid.
        The value is echoed as it was set, cut to 80 characters; the API key is never echoed.
        """
        env = os.environ if environ is None else environ
        return cls(
            qdrant_host=_text(env, "QDRANT_HOST") or cls.qdrant_host,
            qdrant_port=_int(env, "QDRANT_PORT", cls.qdrant_port),
            qdrant_api_key=_text(env, "QDRANT_API_KEY"),
            qdrant_https=_bool(env, "QDRANT_HTTPS", cls.qdrant_https),
            model_name=_text(env, "RAG_ENGINE_MODEL") or cls.model_name,
            model_dim=_int(env, "RAG_ENGINE_DIM", cls.model_dim),
            passage_prefix=_prefix(env, "RAG_ENGINE_PASSAGE_PREFIX", cls.passage_prefix),
            query_prefix=_prefix(env, "RAG_ENGINE_QUERY_PREFIX", cls.query_prefix),
            backend=_backend(env, "RAG_ENGINE_BACKEND", cls.backend),
            batch_size=_int(env, "RAG_ENGINE_BATCH_SIZE", cls.batch_size),
            ocr_languages=_text(env, "RAG_ENGINE_OCR_LANGUAGES") or cls.ocr_languages,
            search_limit=_int(env, "RAG_ENGINE_SEARCH_LIMIT", cls.search_limit),
            search_max_limit=_int(env, "RAG_ENGINE_SEARCH_MAX_LIMIT", cls.search_max_limit),
            model_cache=_text(env, "RAG_ENGINE_MODEL_CACHE"),
        )
