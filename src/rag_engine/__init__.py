"""rag-engine: ingest one document into Qdrant and search what was ingested.

Importing this package stays light: embedding backends and document parsers
live behind optional extras, and the modules of the engine import them only when
they are used. ``torch`` is never imported unless the torch backend is built. One
import is not the engine's: ``qdrant-client``, which the store imports, itself
loads ``fastembed`` (and with it the ONNX runtime) when that package is installed.

The package root exposes ``__version__`` only. Everything else is imported from
its module: ``rag_engine.config``, ``rag_engine.schema``, ``rag_engine.store``,
``rag_engine.embedding``, ``rag_engine.chunking``, ``rag_engine.ingest``,
``rag_engine.search``, ``rag_engine.eval`` and ``rag_engine.errors``. ``rag_engine.cli``
is the ``rag-engine`` command.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
