"""Packaging guarantees: the base import stays light."""

import importlib
import importlib.metadata
import importlib.util
import json
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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
    from packaging.version import Version

    import rag_engine

    assert str(Version(rag_engine.__version__)) == rag_engine.__version__


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
    "rag_engine.config": {
        "BACKENDS",
        "DATE_FILTER_MODES",
        "NO_PREFIX",
        "ConfigError",
        "Settings",
        "valid_ocr_languages",
    },
    "rag_engine.schema": {
        "DATE_KEYS",
        "DATE_PRECISIONS",
        "DERIVED_KEYS",
        "PAYLOAD_INDEXES",
        "PAYLOAD_SCHEMA_VERSION",
        "RESERVED_KEYS",
        "build_payload",
        "date_fields",
        "date_interval",
        "date_interval_fields",
        "page_for_id",
        "point_id",
    },
    "rag_engine.store": {
        "AUTH_STATUSES",
        "DEFAULT_TIMEOUT",
        "DEFAULT_URL_PORT",
        "DISTANCE",
        "CollectionMismatchError",
        "CollectionNotFoundError",
        "DateFieldsError",
        "MissingIndexError",
        "QdrantAuthError",
        "QdrantStore",
        "ReservedKeyError",
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
        "DATE_FILTER_MODES",
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


# Publication metadata: the distribution is published on PyPI as `myopenfund-rag-engine`.
# The import name (`rag_engine`), the command and the tag prefix are unchanged.
ROOT = PYPROJECT.parent
DIST = "myopenfund-rag-engine"
CREDIT = (
    "Created and maintained by [Marc Jeulin](https://github.com/jeulinmarc), "
    "part of [MyOpenFund](https://github.com/MyOpenFund)."
)
RELEASE = ROOT / ".github" / "workflows" / "release.yml"
CI = ROOT / ".github" / "workflows" / "tests.yml"
BLOB = "https://github.com/MyOpenFund/rag-engine/blob/main/"


def _project(pyproject):
    return pyproject["project"]


def _workflow(path):
    # PyYAML reads the bare key `on` as the boolean True.
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))
    return data


def _upload_index(steps):
    return [i for i, s in enumerate(steps) if s.get("uses", "").startswith("actions/upload-artifact@")]


def test_name_and_dynamic_version(pyproject):
    project = _project(pyproject)
    assert project["name"] == DIST
    assert "version" in project["dynamic"]
    assert "version" not in project
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/rag_engine/__init__.py"


def test_installed_version_matches_init():
    import rag_engine

    assert importlib.metadata.version(DIST) == rag_engine.__version__


def test_credit(pyproject):
    project = _project(pyproject)
    assert project["authors"] == [{"name": "Marc Jeulin"}]
    assert project["maintainers"] == [{"name": "Marc Jeulin"}]
    assert "myopenfund" in project["keywords"]
    urls = project["urls"]
    assert urls["Organization"] == "https://github.com/MyOpenFund"
    assert urls["Homepage"] == "https://github.com/MyOpenFund/rag-engine"
    assert urls["Source"] == "https://github.com/MyOpenFund/rag-engine"
    assert urls["Issues"] == "https://github.com/MyOpenFund/rag-engine/issues"
    assert urls["Changelog"] == f"{BLOB}CHANGELOG.md"
    assert project["readme"] == "README.md"


def test_license_spdx_without_classifier_and_python_classifiers(pyproject):
    project = _project(pyproject)
    assert project["license"] == "MIT"
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]
    assert "Development Status :: 3 - Alpha" in project["classifiers"]
    for version in ("3.10", "3.11", "3.12", "3.13"):
        assert f"Programming Language :: Python :: {version}" in project["classifiers"]


def test_license_copyright():
    assert "Copyright (c) 2026 Marc Jeulin" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_readme_credit_install_and_migration():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# rag-engine"
    assert next(line for line in lines[1:] if line.strip()) == CREDIT
    for extra in ("onnx", "torch", "ingest"):
        assert f'pip install "{DIST}[{extra}]"' in text
    assert "`rag-engine[" not in text and 'pip install "rag-engine[' not in text
    assert "pip uninstall -y rag-engine" in text
    assert "git+https://github.com/MyOpenFund/rag-engine" in text
    assert "**Working across repos**" in text


def test_readme_links_are_absolute():
    # README.md is the PyPI description: relative links would 404 there.
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "](#" not in text
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    assert targets
    relative = [t for t in targets if not t.startswith("https://")]
    assert not relative, relative
    assert f"{BLOB}CHANGELOG.md" in text
    assert f"{BLOB}eval/README.md" in text


def test_dev_extra_has_pyyaml(pyproject):
    assert any(d.startswith("pyyaml") for d in _project(pyproject)["optional-dependencies"]["dev"])


def test_changelog_has_version_section():
    lines = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("## 0.1.0") for line in lines)


def test_no_git_urls_in_metadata(pyproject):
    project = _project(pyproject)
    deps = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        deps += extra
    assert not [d for d in deps if "git+" in d or "@" in d]


def test_release_workflow_publishes_through_environment():
    workflow = _workflow(RELEASE)
    assert workflow["permissions"] == {}
    assert workflow["on"]["push"]["tags"] == ["v*"]
    assert list(workflow["on"]["workflow_dispatch"]["inputs"]) == ["target"]
    publish = workflow["jobs"]["publish"]
    assert publish["environment"] == (
        "${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi')"
        " && 'testpypi' || 'pypi' }}"
    )
    assert publish["permissions"] == {"id-token": "write"}
    with_id_token = [
        name for name, job in workflow["jobs"].items() if "id-token" in job.get("permissions", {})
    ]
    assert with_id_token == ["publish"]
    assert publish["needs"] in ("build", ["build"])


def test_actions_pinned_by_sha():
    ci = yaml.safe_load(CI.read_text(encoding="utf-8"))
    texts = [RELEASE.read_text(encoding="utf-8"), yaml.safe_dump(ci["jobs"]["package"])]
    for text in texts:
        uses = re.findall(r"^\s*(?:-\s+)?uses:\s*(\S+)", text, re.MULTILINE)
        assert uses
        for ref in uses:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


def test_tag_check_present_with_prefix_v():
    steps = _workflow(RELEASE)["jobs"]["build"]["steps"]
    checks = [i for i, s in enumerate(steps) if "tag_version" in s.get("run", "") and "exit 1" in s["run"]]
    assert len(checks) == 1
    assert steps[checks[0]]["if"] == "github.event_name == 'push'"
    assert "${GITHUB_REF_NAME#v}" in steps[checks[0]]["run"]
    assert DIST in steps[checks[0]]["run"]
    upload = _upload_index(steps)
    assert upload and checks[0] < upload[0]


def test_build_checks_tag_is_on_default_branch():
    steps = _workflow(RELEASE)["jobs"]["build"]["steps"]
    checks = [i for i, s in enumerate(steps) if "merge-base --is-ancestor" in s.get("run", "")]
    assert len(checks) == 1
    assert "origin/main" in steps[checks[0]]["run"]
    assert steps[checks[0]]["if"] == "github.event_name == 'push'"
    checkout = [s for s in steps if s.get("uses", "").startswith("actions/checkout@")]
    assert checkout[0]["with"]["fetch-depth"] == 0
    upload = _upload_index(steps)
    assert upload and checks[0] < upload[0]


def test_testpypi_publish_skips_existing_but_pypi_does_not():
    steps = {s["name"]: s for s in _workflow(RELEASE)["jobs"]["publish"]["steps"] if "name" in s}
    assert steps["Publish to TestPyPI"]["with"]["skip-existing"] is True
    assert "skip-existing" not in steps["Publish to PyPI"].get("with", {})


def test_concurrency_group():
    assert _workflow(RELEASE)["concurrency"]["group"] == "release-${{ github.ref }}"


def test_dispatch_inputs():
    inputs = _workflow(RELEASE)["on"]["workflow_dispatch"]["inputs"]
    assert list(inputs) == ["target"]
    assert inputs["target"]["type"] == "choice"
    assert inputs["target"]["options"] == ["pypi", "testpypi"]
    assert inputs["target"]["default"] == "pypi"
    assert inputs["target"]["required"] is True


def test_build_dispatch_runs_on_default_branch_only():
    steps = _workflow(RELEASE)["jobs"]["build"]["steps"]
    guard = [s for s in steps if "refs/heads/main" in s.get("run", "")]
    assert len(guard) == 1
    assert guard[0]["if"] == "github.event_name == 'workflow_dispatch'"
    assert "exit 1" in guard[0]["run"]
    assert steps.index(guard[0]) == 0


def test_build_computes_tag_and_refuses_an_existing_pypi_tag():
    job = _workflow(RELEASE)["jobs"]["build"]
    step = [s for s in job["steps"] if s.get("id") == "tag"]
    assert len(step) == 1
    run = step[0]["run"]
    assert step[0]["if"] == "github.event_name == 'workflow_dispatch'"
    assert "importlib.metadata" in run and DIST in run
    assert 'tag="v$version"' in run
    assert 'git ls-remote --tags origin "refs/tags/$tag"' in run
    assert '"$INPUT_TARGET" = "pypi"' in run
    assert "already exists" in run and "exit 1" in run
    upload = _upload_index(job["steps"])
    assert job["steps"].index(step[0]) < upload[0]
    assert job["outputs"]["tag"] == "${{ steps.tag.outputs.tag || github.ref_name }}"
    assert job["outputs"]["sha"] == "${{ github.sha }}"


def test_testpypi_only_for_testpypi_target():
    steps = {s["name"]: s for s in _workflow(RELEASE)["jobs"]["publish"]["steps"] if "name" in s}
    assert "inputs.target == 'testpypi'" in steps["Publish to TestPyPI"]["if"]
    assert "inputs.target == 'testpypi'" in steps["Publish to PyPI"]["if"]
    assert steps["Publish to PyPI"]["if"].startswith("${{ !(")


def _release_job():
    job = _workflow(RELEASE)["jobs"]["github-release"]
    assert job["if"] == (
        "github.event_name == 'push' || (github.event_name == 'workflow_dispatch' && inputs.target == 'pypi')"
    )
    assert job["permissions"] == {"contents": "write"}
    assert set(job["needs"]) == {"build", "publish"}
    return job


def _assert_tag_created_and_pushed(run):
    # no personal address in a public workflow: the tagger email is the actor's noreply one
    assert 'user.name="Marc Jeulin"' in run
    assert 'user.email="${GITHUB_ACTOR_ID}+${GITHUB_ACTOR}@users.noreply.github.com"' in run
    assert 'tag -a "$TAG"' in run and '"$SHA"' in run
    assert 'git push origin "refs/tags/$TAG"' in run


def test_github_release_creates_and_pushes_the_tag_on_dispatch_not_testpypi():
    job = _release_job()
    checkout = [s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")]
    assert len(checkout) == 1
    assert checkout[0]["with"]["ref"] == "${{ needs.build.outputs.sha }}"
    # credentials stay in the checkout only for a dispatch (the tag push)
    assert checkout[0]["with"]["persist-credentials"] == "${{ github.event_name == 'workflow_dispatch' }}"
    tag = [s for s in job["steps"] if "git push" in s.get("run", "")]
    assert len(tag) == 1
    assert tag[0]["if"] == "github.event_name == 'workflow_dispatch'"
    _assert_tag_created_and_pushed(tag[0]["run"])


def test_dispatch_tag_step_is_idempotent():
    run = next(s for s in _release_job()["steps"] if "git push" in s.get("run", ""))["run"]
    lookup = run.index('git ls-remote origin "refs/tags/$TAG" "refs/tags/$TAG^{}"')
    assert lookup < run.index("git push")
    assert '"$remote_sha" = "$SHA"' in run  # already there at the built commit: skip
    assert "already exists" in run and "exit 1" in run  # there, but elsewhere: fail
    assert '[ -z "$remote_sha" ]' in run  # absent: create


def test_github_release_is_created_or_completed_on_both_paths():
    job = _release_job()
    release = [s for s in job["steps"] if "gh release create" in s.get("run", "")]
    assert len(release) == 1
    assert "if" not in release[0]
    run = release[0]["run"]
    assert 'gh release view "$TAG"' in run
    assert 'gh release upload "$TAG" dist/* --repo "$GITHUB_REPOSITORY" --clobber' in run
    assert '--target "$SHA"' in run and '--title "$TAG"' in run
    assert "--verify-tag" in run
    assert "--notes-file" in run and "--generate-notes" in run and "CHANGELOG.md" in run
    assert 'version="${TAG#v}"' in run


def test_smoke_uses_cli_version_and_onnx_extra():
    for path in (RELEASE, CI):
        text = path.read_text(encoding="utf-8")
        assert '/bin/rag-engine" --version' in text, path
        assert "[onnx]" in text, path
        assert "ONNX_MODELS" in text, path


def test_dependabot_covers_github_actions_only():
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    assert [(u["package-ecosystem"], u["schedule"]["interval"]) for u in config["updates"]] == [
        ("github-actions", "weekly")
    ]


def test_ci_keeps_its_jobs_and_adds_package():
    jobs = _workflow(CI)["jobs"]
    assert {"tests", "integration", "parity", "package"} <= set(jobs)
    runs = " ".join(s.get("run", "") for s in jobs["package"]["steps"])
    assert "build==1.6.1 twine==7.0.0" in runs
    assert "twine check --strict" in runs


def test_install_hints_name_the_published_distribution():
    # The PyPI name `rag-engine` belongs to another project: the hints must name ours.
    from rag_engine import chunking, embedding

    for hint, extra in (
        (embedding._TORCH_EXTRA, "torch"),
        (embedding._ONNX_EXTRA, "onnx"),
        (chunking._INSTALL_HINT, "ingest"),
    ):
        assert f"'{DIST}[{extra}]'" in hint
        assert "'rag-engine[" not in hint


# ---- fail-closed approval gate (release hardening) -------------------------


def _gate_job():
    job = _workflow(RELEASE)["jobs"]["build"]
    gate = [i for i, s in enumerate(job["steps"]) if s.get("name") == "Check the approval gate"]
    assert len(gate) == 1
    return job, gate[0]


def test_release_checks_the_approval_gate_before_building():
    job, gate = _gate_job()
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    step = job["steps"][gate]
    run = step["run"]
    assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert "inputs.target == 'testpypi'" in step["env"]["ENV_NAME"]
    assert "'testpypi'" in step["env"]["ENV_NAME"] and "'pypi'" in step["env"]["ENV_NAME"]
    assert 'gh api "repos/$GITHUB_REPOSITORY/environments/$ENV_NAME"' in run
    assert "if ! " in run  # an API failure (non-200) is tested, not ignored
    assert "required_reviewers" in run and "exit 1" in run
    for i, other in enumerate(job["steps"]):
        uses = other.get("uses", "")
        building = uses.startswith(
            ("actions/checkout@", "actions/setup-python@", "actions/upload-artifact@")
        ) or "build" in other.get("run", "")
        if i != gate and (building or other.get("name", "").startswith("Build")):
            assert gate < i, other


def _run_gate(tmp_path, response, status):
    """Run the gate step against a stub `gh` that prints `response` and exits `status`."""
    import os
    import shutil
    import subprocess

    import pytest

    if shutil.which("jq") is None or shutil.which("bash") is None:
        pytest.skip("needs jq and bash")
    job, gate = _gate_job()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "gh"
    stub.write_text(f"#!/bin/sh\ncat <<'JSON'\n{response}\nJSON\nexit {status}\n")
    stub.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "MyOpenFund/example",
        "ENV_NAME": "pypi",
    }
    return subprocess.run(
        ["bash", "-e", "-c", job["steps"][gate]["run"]],
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )


def test_gate_passes_with_a_required_reviewer(tmp_path):
    rules = (
        '{"protection_rules": [{"type": "wait_timer"}, {"type": "required_reviewers",'
        ' "reviewers": [{"type": "User", "reviewer": {"login": "release-approver"}}]}]}'
    )
    assert _run_gate(tmp_path, rules, 0).returncode == 0


def test_gate_fails_without_a_required_reviewer(tmp_path):
    for rules in (
        '{"protection_rules": []}',
        '{"protection_rules": [{"type": "required_reviewers", "reviewers": []}]}',
        '{"protection_rules": [{"type": "branch_policy"}]}',
        "{}",
    ):
        result = _run_gate(tmp_path, rules, 0)
        assert result.returncode == 1, rules
        assert "required reviewer" in result.stdout + result.stderr


def test_gate_fails_closed_when_the_api_call_fails(tmp_path):
    result = _run_gate(tmp_path, "gh: Not Found (HTTP 404)", 1)
    assert result.returncode == 1
    assert "HTTP 404" in result.stdout + result.stderr


def test_how_to_release_names_the_version_source():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("**How to release**", 1)[1].split("\n\n", 3)[1]
    assert "`__version__` in `src/rag_engine/__init__.py`" in section
    assert "and `version`" not in section
    assert "pip install -e" in section
