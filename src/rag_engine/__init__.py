"""rag-engine: ingest one document into Qdrant and search what was ingested.

Importing this package stays light: embedding backends and document parsers
live behind optional extras, and the modules of the engine import them only when
they are used. ``torch`` is never imported unless the torch backend is built. One
import is not the engine's: ``qdrant-client``, which the store imports, itself
loads ``fastembed`` (and with it the ONNX runtime) when that package is installed.

Importing the package sets ``ORT_DISABLE_TELEMETRY=1`` in the environment of the
process unless the variable is already set: the ONNX runtime sends telemetry by default
and reads the variable when it is imported, which the store may cause. Set the variable
yourself, to another value, before the import to decide otherwise. The default comes too
late for a process that imported ``onnxruntime`` first (directly, or through ``fastembed``
or ``qdrant_client``): the package then warns once, with a ``RuntimeWarning``, unless the
variable was set before the import.

The package root exposes ``__version__`` only. Everything else is imported from
its module: ``rag_engine.config``, ``rag_engine.schema``, ``rag_engine.store``,
``rag_engine.embedding``, ``rag_engine.chunking``, ``rag_engine.ingest``,
``rag_engine.search``, ``rag_engine.eval`` and ``rag_engine.errors``. ``rag_engine.cli``
is the ``rag-engine`` command.
"""

import os as _os  # under private names: the package root exposes ``__version__`` only
import sys as _sys
import warnings as _warnings

# The runtime reads the variable when it is imported. Loaded already, with the variable unset,
# it has decided without the default below: say so, once (a module is imported once), instead
# of letting the user believe the default applied. Nothing is imported to find out.
if "onnxruntime" in _sys.modules and "ORT_DISABLE_TELEMETRY" not in _os.environ:
    _warnings.warn(
        "onnxruntime was imported before rag_engine with ORT_DISABLE_TELEMETRY unset: the runtime "
        "read the variable at its own import, so its telemetry is on in this process and the "
        "default rag_engine sets comes too late. Export ORT_DISABLE_TELEMETRY=1 before starting "
        "the process, or import rag_engine before onnxruntime, fastembed and qdrant_client.",
        RuntimeWarning,
        stacklevel=2,
    )

# Before any module of the engine can import the ONNX runtime: it reads the variable at import.
# A default only: a value that is already there, whatever it is, is the user's and is kept.
_os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__all__ = ["__version__"]

__version__ = "0.1.0"
