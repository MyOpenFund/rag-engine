"""The engine does not let the ONNX runtime send telemetry on behalf of its users.

onnxruntime reads ``ORT_DISABLE_TELEMETRY`` when it is imported, not when a session is made:
its import alone creates a device identifier and an event store under the home directory,
and starts the uploader. The engine therefore sets the default when the package is imported,
before any of its modules can import the runtime (``rag_engine.store`` does, through
``qdrant-client``, when ``fastembed`` is installed). Each check runs in a process of its own.
"""

from __future__ import annotations

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
    assert any(".onnxruntime" in name for name in written(loud)), written(loud)
