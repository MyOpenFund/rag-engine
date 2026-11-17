"""Packaging guarantees: the base import stays light."""

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

HEAVY_MODULES = ("torch", "sentence_transformers", "fastembed", "chunknorris", "markitdown")


def test_version():
    import rag_engine

    assert rag_engine.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "modules",
    [
        "rag_engine",
        "rag_engine.config",
        "rag_engine.schema",
        "rag_engine.store",
        "rag_engine.embedding",
        "rag_engine.chunking",
        "rag_engine.ingest",
    ],
)
def test_import_pulls_no_heavy_dependency(modules):
    code = (
        f"import {modules}, sys; "
        f"heavy = [m for m in {HEAVY_MODULES!r} if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"import {modules} loaded: {result.stdout.strip()}"


PUBLIC_NAMES = {
    "rag_engine": {"__version__"},
    "rag_engine.errors": {"RagEngineError"},
    "rag_engine.config": {"BACKENDS", "NO_PREFIX", "ConfigError", "Settings", "valid_ocr_languages"},
    "rag_engine.schema": {"PAYLOAD_INDEXES", "RESERVED_KEYS", "build_payload", "page_for_id", "point_id"},
    "rag_engine.store": {
        "AUTH_STATUSES",
        "DEFAULT_TIMEOUT",
        "DEFAULT_URL_PORT",
        "DISTANCE",
        "CollectionMismatchError",
        "QdrantAuthError",
        "QdrantStore",
        "StoreError",
    },
    "rag_engine.embedding": {
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
    },
    "rag_engine.chunking": {
        "ENABLE_OCR_ENV",
        "HEADER_LEVELS",
        "OCR_MODES",
        "PDF_ADD_HEADERS",
        "PDF_BODY_LINE_SPACING",
        "PDF_ENABLE_ML_FEATURES",
        "PDF_EXTRACT_TABLES",
        "SUPPORTED_EXTENSIONS",
        "TESSDATA_PREFIX_ENV",
        "Chunk",
        "ChunkingOptions",
        "UnsupportedFormatError",
        "check_ocr_languages",
        "chunk_file",
        "ocr_available",
    },
    "rag_engine.ingest": {"IngestResult", "ingest_document"},
}


@pytest.mark.parametrize("module_name", sorted(PUBLIC_NAMES))
def test_every_public_module_declares_its_public_names(module_name):
    module = importlib.import_module(module_name)
    assert isinstance(module.__all__, list)
    assert len(module.__all__) == len(set(module.__all__))
    assert set(module.__all__) == PUBLIC_NAMES[module_name]
    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.__all__ names {name!r}, which does not exist"


def test_every_module_of_the_package_is_covered():
    import rag_engine

    found = {f"rag_engine.{info.name}" for info in pkgutil.iter_modules(rag_engine.__path__)}
    assert found | {"rag_engine"} == set(PUBLIC_NAMES)


def test_the_package_root_exposes_the_version_only():
    import rag_engine

    public = {name for name in vars(rag_engine) if not name.startswith("_")}
    # Submodules appear as attributes once something imported them; nothing else may.
    assert public <= {name.split(".")[1] for name in PUBLIC_NAMES if "." in name}


def test_the_package_ships_the_typing_marker():
    import rag_engine

    assert (Path(rag_engine.__file__).parent / "py.typed").is_file()


PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture
def pyproject():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11 or later
    if not PYPROJECT.is_file():
        pytest.skip("no pyproject.toml next to the tests")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_qdrant_client_has_a_lower_bound(pyproject):
    # 1.8.0 is the first release with QdrantClient.collection_exists, which the store calls.
    assert "qdrant-client>=1.8" in pyproject["project"]["dependencies"]


def test_the_package_is_declared_typed(pyproject):
    assert "Typing :: Typed" in pyproject["project"]["classifiers"]


def test_the_sdist_leaves_out_the_tools_and_the_ci_files_and_keeps_the_tests(pyproject):
    excluded = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert {"/tools", "/.github"} <= set(excluded)
    assert not any("tests" in entry for entry in excluded)
