"""The engine does not let the ONNX runtime send telemetry on behalf of its users.

onnxruntime reads ``ORT_DISABLE_TELEMETRY`` when it is imported, not when a session is made:
its import alone creates a device identifier and an event store under the home directory,
and starts the uploader. The engine therefore sets the default when the package is imported,
before any of its modules can import the runtime (``rag_engine.store`` does, through
``qdrant-client``, when ``fastembed`` is installed). Each check runs in a process of its own.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

VARIABLE = "ORT_DISABLE_TELEMETRY"
ROOT = Path(__file__).resolve().parent.parent


def run(code: str, home: Path, **variables: str) -> str:
    environment = {name: value for name, value in os.environ.items() if name != VARIABLE}
    environment.update(HOME=str(home), XDG_CACHE_HOME=str(home / ".cache"), **variables)
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source, environment.get("PYTHONPATH")]))
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def written(home: Path) -> list[str]:
    return sorted(str(path.relative_to(home)) for path in home.rglob("*") if path.is_file())


SHOW = f"import os; print(os.environ.get({VARIABLE!r}))"


def test_importing_the_package_turns_the_telemetry_of_the_runtime_off(tmp_path):
    assert run("import os; print(os.environ.get('ORT_DISABLE_TELEMETRY'))", tmp_path) == "None"
    assert run(f"import rag_engine; {SHOW}", tmp_path) == "1"
    # Any module of the engine: the package is imported first.
    assert run(f"import rag_engine.config; {SHOW}", tmp_path) == "1"
    assert run(f"from rag_engine.embedding import FakeEmbedder; {SHOW}", tmp_path) == "1"


@pytest.mark.parametrize("value", ["0", "1", "false", ""])
def test_a_value_the_user_set_is_kept(tmp_path, value):
    shown = run(
        f"import os, rag_engine; print(repr(os.environ[{VARIABLE!r}]))", tmp_path, **{VARIABLE: value}
    )
    assert shown == repr(value)


def test_the_default_is_set_without_importing_anything_heavy(tmp_path):
    loaded = run(
        "import sys, rag_engine; "
        "print(sorted(name for name in ('onnxruntime', 'fastembed', 'torch', 'qdrant_client', 'numpy') "
        "if name in sys.modules))",
        tmp_path,
    )
    assert loaded == "[]"


WARNED = (
    "import sys, warnings\n"
    "with warnings.catch_warnings(record=True) as caught:\n"
    "    warnings.simplefilter('always')\n"
    "    {imports}\n"
    "ours = [w for w in caught if 'ORT_DISABLE_TELEMETRY' in str(w.message)]\n"
    "print([(w.category.__name__, str(w.message)) for w in ours])\n"
    "print(sorted(name for name in ('fastembed', 'qdrant_client', 'numpy', 'torch') if name in sys.modules))"
)


def stub_runtime(tmp_path: Path) -> dict[str, str]:
    """A module named ``onnxruntime`` that does nothing, first on the path: no real runtime is needed."""
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "onnxruntime.py").write_text("", encoding="utf-8")
    return {"PYTHONPATH": str(stub)}


def test_the_package_warns_when_the_runtime_was_imported_first(tmp_path):
    path = stub_runtime(tmp_path)
    shown, loaded = run(
        WARNED.format(imports="import onnxruntime, rag_engine"), tmp_path, **path
    ).splitlines()
    ((category, message),) = ast.literal_eval(shown)
    assert category == "RuntimeWarning"
    assert "onnxruntime was imported before rag_engine" in message and "telemetry is on" in message
    assert "Export ORT_DISABLE_TELEMETRY=1 before starting the process" in message
    assert "import rag_engine before" in message
    assert loaded == "[]"  # found out from sys.modules: nothing heavy was imported to check
    # Once: importing the package again, or one of its modules, says nothing more.
    again = "import onnxruntime, rag_engine, rag_engine.config; import rag_engine"
    assert len(ast.literal_eval(run(WARNED.format(imports=again), tmp_path, **path).splitlines()[0])) == 1
    # The default is set all the same, for whatever reads the variable later.
    assert (
        run(
            f"import warnings; warnings.simplefilter('ignore'); import onnxruntime, rag_engine; {SHOW}",
            tmp_path,
            **path,
        )
        == "1"
    )


def test_the_package_does_not_warn_in_the_normal_order(tmp_path):
    path = stub_runtime(tmp_path)
    shown = run(WARNED.format(imports="import rag_engine, onnxruntime"), tmp_path, **path)
    assert shown.splitlines()[0] == "[]"
    assert run(WARNED.format(imports="import rag_engine"), tmp_path).splitlines()[0] == "[]"


@pytest.mark.parametrize("value", ["1", "0", ""])
def test_the_package_does_not_warn_when_the_user_set_the_variable(tmp_path, value):
    path = stub_runtime(tmp_path)
    code = WARNED.format(imports="import onnxruntime, rag_engine")
    assert run(code, tmp_path, **path, **{VARIABLE: value}).splitlines()[0] == "[]"


def test_the_onnx_backend_sets_the_default_before_its_own_imports(monkeypatch):
    from rag_engine import embedding

    monkeypatch.delenv(VARIABLE, raising=False)
    embedding._keep_the_runtime_quiet()
    assert os.environ[VARIABLE] == "1"
    monkeypatch.setenv(VARIABLE, "0")
    embedding._keep_the_runtime_quiet()
    assert os.environ[VARIABLE] == "0"


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None, reason="onnxruntime is not installed")
def test_with_the_engine_imported_first_the_runtime_writes_nothing_under_home(tmp_path):
    """The identifier and the event store the runtime creates when its telemetry is on are the witness."""
    quiet, loud = tmp_path / "quiet", tmp_path / "loud"
    quiet.mkdir()
    loud.mkdir()
    # Through the engine: the store imports qdrant-client, which imports fastembed and the runtime.
    loaded = run("import sys, rag_engine.store, onnxruntime; print('onnxruntime' in sys.modules)", quiet)
    assert loaded == "True"
    assert [name for name in written(quiet) if "onnxruntime" in name] == []
    # The same import with the telemetry asked for: the runtime leaves its files.
    # Nothing may leave the machine for this check: the proxy named here answers nothing.
    nowhere = dict.fromkeys(
        ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"), "http://127.0.0.1:9"
    )
    run("import rag_engine.store, onnxruntime", loud, **{VARIABLE: "0"}, **nowhere)
    if not any(".onnxruntime" in name for name in written(loud)):
        # Not every build of the runtime writes these files (it depends on its version and on
        # how it was built): without the witness, the absence checked above proves nothing.
        pytest.skip("this build of onnxruntime leaves no telemetry file even when asked to")
