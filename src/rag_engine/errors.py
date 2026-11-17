"""The common base of the errors the engine defines itself.

``except RagEngineError`` catches every error raised under a class of this package:
:class:`rag_engine.config.ConfigError`, :class:`rag_engine.store.StoreError` and its
subclasses, :class:`rag_engine.chunking.UnsupportedFormatError`. It does not catch what the
engine lets through unchanged (an error of a parser, of the embedding backend or of the
Qdrant client), nor the plain ``ValueError`` / ``TypeError`` of a bad argument.
"""

from __future__ import annotations

__all__ = ["RagEngineError"]


class RagEngineError(Exception):
    """Base class of every exception class defined by rag-engine."""
