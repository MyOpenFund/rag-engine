"""Packaging guarantees: the base import stays light."""

import importlib
import importlib.util
import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

# What the engine never imports at import time: the embedding backends and the document parsers.
BACKENDS_AND_PARSERS = ("torch", "sentence_transformers", "fastembed", "chunknorris", "markitdown")
# What no import of the engine may load at all, whoever imports it. fastembed is not in this
# list: qdrant-client, a dependency of the base install, imports fastembed itself when that
# package is installed (the onnx extra), and the engine cannot prevent it.
NEVER_LOADED = ("torch", "sentence_transformers", "chunknorris", "markitdown")
# Who may import fastembed while an engine module is imported: qdrant-client, and fastembed itself.
FASTEMBED_IMPORTERS = ("qdrant_client", "fastembed")

ENGINE_MODULES = [
    "rag_engine",
    "rag_engine.errors",
    "rag_engine.config",
    "rag_engine.schema",
    "rag_engine.store",
    "rag_engine.embedding",
    "rag_engine.chunking",
    "rag_engine.ingest",
    "rag_engine.search",
    "rag_engine.eval",
    "rag_engine.eval_check",
    "rag_engine.cli",
]

# Run in a fresh interpreter: imports one module and reports, for each watched package, which
# module asked for it (the ``import`` statement goes through ``builtins.__import__``, which is
# given the globals of the importing module), what is loaded at the end, and what is installed.
IMPORT_SPY = """
import builtins, importlib.util, json, sys
watched = {watched!r}
seen = []
real_import = builtins.__import__
def spy(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.partition(".")[0]
    if level == 0 and top in watched:
        seen.append([top, (globals or {{}}).get("__name__") or "?"])
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = spy
import {module}
builtins.__import__ = real_import
loaded = [name for name in watched if name in sys.modules]
installed = [name for name in watched if importlib.util.find_spec(name) is not None]
print(json.dumps({{"seen": seen, "loaded": loaded, "installed": installed}}))
"""


def import_report(module: str, *, first_on_path: Path | None = None) -> dict:
    """What importing ``module`` in a fresh interpreter loads, and who asked for it."""
    code = IMPORT_SPY.format(watched=BACKENDS_AND_PARSERS, module=module)
    # No bytecode files: a test rewrites a stub module within the same second.
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if first_on_path is not None:
        # The engine itself stays importable when it is found through the working directory.
        import rag_engine

        source = str(Path(rag_engine.__file__).resolve().parent.parent)
        paths = [str(first_on_path), source, environment.get("PYTHONPATH")]
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, paths))
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def import_problems(report: dict, engine: str = "rag_engine") -> list[str]:
    """What the report shows against the guarantee of the engine; empty when it holds.

    * no module of the engine imports an embedding backend or a parser;
    * torch, sentence-transformers and the parsers are not loaded at all;
    * fastembed is loaded only where it is installed, and only because qdrant-client (or
      fastembed itself) imported it.
    """
    problems = []
    for name, importer in report["seen"]:
        top = importer.partition(".")[0]
        if top == engine:
            problems.append(f"{importer} imports {name}")
        elif name == "fastembed" and top not in FASTEMBED_IMPORTERS:
            problems.append(f"fastembed is imported by {importer}, which the engine loaded")
    problems += [f"{name} is loaded" for name in report["loaded"] if name in NEVER_LOADED]
    if "fastembed" in report["loaded"] and "fastembed" not in report["installed"]:
        problems.append("fastembed is loaded although it is not installed")
    return sorted(set(problems))


def test_version():
    import rag_engine

    assert rag_engine.__version__ == "0.1.0"


@pytest.mark.parametrize("module", ENGINE_MODULES)
def test_no_engine_module_imports_a_backend_or_a_parser(module):
    report = import_report(module)
    assert import_problems(report) == [], f"import {module}: {report}"
    if "fastembed" not in report["installed"]:
        # The light install and the ingest one: nothing of the list is loaded at all.
        assert report["loaded"] == [], f"import {module} loaded {report['loaded']}"


def _stub(folder: Path, name: str, source: str = "") -> None:
    (folder / f"{name}.py").write_text(source, encoding="utf-8")


@pytest.fixture
def stub_fastembed(tmp_path):
    """A folder to put first on the import path, with an empty package named ``fastembed``.

    It stands for an install with the onnx extra in the environments that do not have it:
    qdrant-client finds a ``fastembed`` and imports it.
    """
    _stub(tmp_path, "fastembed", "LOADED = True\n")
    return tmp_path


@pytest.mark.parametrize(
    "module", ["rag_engine.store", "rag_engine.search", "rag_engine.eval", "rag_engine.cli"]
)
def test_a_fastembed_loaded_by_qdrant_client_is_not_an_import_of_the_engine(module, stub_fastembed):
    if importlib.util.find_spec("fastembed") is not None:
        report = import_report(module)  # the real package: this is the onnx install itself
    else:
        report = import_report(module, first_on_path=stub_fastembed)
    assert "fastembed" in report["loaded"], "qdrant-client no longer imports fastembed by itself"
    assert {importer.partition(".")[0] for name, importer in report["seen"] if name == "fastembed"} <= set(
        FASTEMBED_IMPORTERS
    )
    assert import_problems(report) == []


def test_an_engine_module_that_imports_fastembed_is_reported(stub_fastembed):
    _stub(stub_fastembed, "fake_engine", "import fastembed\n")
    report = import_report("fake_engine", first_on_path=stub_fastembed)
    assert import_problems(report, engine="fake_engine") == ["fake_engine imports fastembed"]


def test_a_fastembed_imported_through_another_package_is_reported(stub_fastembed):
    _stub(stub_fastembed, "helper", "from fastembed import LOADED\n")
    _stub(stub_fastembed, "fake_engine", "import helper\n")
    report = import_report("fake_engine", first_on_path=stub_fastembed)
    assert import_problems(report, engine="fake_engine") == [
        "fastembed is imported by helper, which the engine loaded"
    ]


@pytest.mark.parametrize("heavy", NEVER_LOADED)
def test_a_backend_or_a_parser_loaded_by_anyone_is_reported(tmp_path, heavy):
    # A stub under the name of the package: found first, also where the real one is installed.
    _stub(tmp_path, heavy)
    _stub(tmp_path, "helper", f"import {heavy}\n")
    _stub(tmp_path, "fake_engine", "import helper\n")
    report = import_report("fake_engine", first_on_path=tmp_path)
    assert import_problems(report, engine="fake_engine") == [f"{heavy} is loaded"]
    _stub(tmp_path, "fake_engine", f"import {heavy}\n")
    report = import_report("fake_engine", first_on_path=tmp_path)
    assert import_problems(report, engine="fake_engine") == sorted(
        [f"fake_engine imports {heavy}", f"{heavy} is loaded"]
    )


def test_fastembed_loaded_without_being_installed_is_reported():
    report = {"seen": [], "loaded": ["fastembed"], "installed": []}
    assert import_problems(report) == ["fastembed is loaded although it is not installed"]
    assert import_problems({"seen": [], "loaded": ["fastembed"], "installed": ["fastembed"]}) == []


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
        "CollectionNotFoundError",
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
        "ModelDownloadError",
        "ModelIntegrityError",
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
    "rag_engine.search": {
        "QUERY_NORMALIZATION",
        "Hit",
        "SearchError",
        "SearchFilters",
        "is_searchable",
        "search",
    },
    "rag_engine.eval": {
        "BEHAVIORS",
        "BOOTSTRAP_SEED",
        "DEFAULT_BEHAVIOR",
        "DEFAULT_BOOTSTRAP_RESAMPLES",
        "DEFAULT_CHUNK_LIMIT",
        "DEFAULT_INTERVAL_MIN_DOCUMENTS",
        "DEFAULT_INTERVAL_MIN_QUESTIONS",
        "DEFAULT_K",
        "DEFAULT_SCORE_LEVELS",
        "FILTER_FIELDS",
        "INTERVAL_LEVEL",
        "QUESTION_FIELDS",
        "UNFILTERED_SUFFIX",
        "UNFILTERED_TAG",
        "Abstention",
        "Aggregate",
        "EssentialAggregate",
        "EssentialMeans",
        "EvalError",
        "EvalReport",
        "FilterEffect",
        "Means",
        "NotInCollection",
        "Question",
        "QuestionResult",
        "QuestionSet",
        "RankedDocument",
        "TopScores",
        "evaluate",
        "format_report",
        "load_questions",
    },
    "rag_engine.eval_check": {
        "CatalogueEntry",
        "DocumentReference",
        "QuestionCheck",
        "check_questions",
        "format_check",
        "load_catalogue",
    },
    "rag_engine.cli": {"main"},
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
    # 1.10.0 is the first release with QdrantClient.query_points, which the search calls
    # (collection_exists, which the store calls, came with 1.8.0).
    assert "qdrant-client>=1.10" in pyproject["project"]["dependencies"]


def test_the_onnx_extra_declares_what_the_backend_imports(pyproject):
    onnx = pyproject["project"]["optional-dependencies"]["onnx"]
    # fastembed, bounded to the minor version parity was measured on; and huggingface_hub, which
    # the backend imports itself (snapshot_download) and must not receive by chance.
    assert onnx == ["fastembed>=0.9,<0.10", "huggingface_hub>=0.20"]


def test_the_installed_huggingface_hub_takes_the_arguments_the_backend_uses():
    # The arguments the backend passes (tests/test_embedding.py pins them with a stand-in of
    # this function); huggingface_hub 0.20.0, the declared lower bound, has them all.
    hub = pytest.importorskip("huggingface_hub")
    import inspect

    parameters = inspect.signature(hub.snapshot_download).parameters
    assert {"repo_id", "revision", "allow_patterns", "cache_dir", "local_files_only"} <= set(parameters)
    assert next(iter(parameters)) == "repo_id"


def test_sentence_transformers_is_bounded_to_the_major_version_parity_was_measured_on(pyproject):
    assert pyproject["project"]["optional-dependencies"]["torch"] == ["sentence-transformers>=6,<7"]


def test_the_package_is_declared_typed(pyproject):
    assert "Typing :: Typed" in pyproject["project"]["classifiers"]


def test_the_sdist_leaves_out_the_tools_and_the_ci_files_and_keeps_the_tests(pyproject):
    excluded = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert {"/tools", "/.github"} <= set(excluded)
    assert not any("tests" in entry for entry in excluded)


def test_the_sdist_keeps_the_evaluation_files_and_the_wheel_holds_the_package_only(pyproject):
    targets = pyproject["tool"]["hatch"]["build"]["targets"]
    # Nothing under eval/ is excluded: the tests read eval/fixtures/ from the source distribution.
    assert not any("eval" in entry for entry in targets["sdist"]["exclude"])
    assert "include" not in targets["sdist"] and "only-include" not in targets["sdist"]
    assert targets["wheel"]["packages"] == ["src/rag_engine"]
    root = PYPROJECT.parent
    for name in ("README.md", "make_known_item.py", "known_item_v1.jsonl"):
        assert (root / "eval" / name).is_file(), name
    for name in ("corpus.json", "questions.jsonl", "questions_v2.jsonl"):
        assert (root / "eval" / "fixtures" / name).is_file(), name


def test_the_console_script_is_declared(pyproject):
    assert pyproject["project"]["scripts"] == {"rag-engine": "rag_engine.cli:main"}
    from rag_engine.cli import main

    assert callable(main)


def test_the_package_does_not_need_the_evaluation_files():
    # eval.py and cli.py read the files they are given, never a file of the repository.
    import rag_engine.cli
    import rag_engine.eval

    for module in (rag_engine.eval, rag_engine.cli):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "__file__" not in source
