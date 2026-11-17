"""rag-engine: ingest one document into Qdrant and search what was ingested.

Importing this package stays light: embedding backends and document parsers
live behind optional extras and are only imported when they are used.

The package root exposes ``__version__`` only. Everything else is imported from
its module: ``rag_engine.config``, ``rag_engine.schema``, ``rag_engine.store``,
``rag_engine.embedding``, ``rag_engine.chunking``, ``rag_engine.ingest`` and
``rag_engine.errors``.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
