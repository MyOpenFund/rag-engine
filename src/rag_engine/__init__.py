"""rag-engine: ingest one document into Qdrant and search what was ingested.

Importing this package stays light: embedding backends and document parsers
live behind optional extras and are only imported when they are used.

The package root exposes ``__version__`` only. Everything else is imported from
its module: ``rag_engine.config``, ``rag_engine.schema``, ``rag_engine.store``,
``rag_engine.embedding``, ``rag_engine.chunking``, ``rag_engine.ingest``,
``rag_engine.search``, ``rag_engine.eval`` and ``rag_engine.errors``. ``rag_engine.cli``
is the ``rag-engine`` command.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
