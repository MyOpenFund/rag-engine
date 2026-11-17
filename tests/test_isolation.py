"""The tests do not read the settings of the host (``conftest.py``).

A developer may have ``QDRANT_PORT``, ``RAG_ENGINE_BACKEND`` or ``TESSDATA_PREFIX`` set for
other work; the suite must give the same result with them. The check is made on a second
pytest, started with an environment full of unusable values.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import ENGINE_VARIABLES, KEPT, KEPT_BY_MARKER, is_engine_variable

ROOT = Path(__file__).resolve().parent.parent
PROBE = "ISOLATION_PROBE"  # not a name the engine reads: it reaches the second pytest
HOSTILE = {
    "QDRANT_HOST": "host-of-the-developer.internal",
    "QDRANT_PORT": "abc",
    "QDRANT_API_KEY": "key-of-the-developer",
    "QDRANT_HTTPS": "maybe",
    "RAG_ENGINE_BACKEND": "nonsense",
    "RAG_ENGINE_DIM": "0",
    "RAG_ENGINE_BATCH_SIZE": "-1",
    "RAG_ENGINE_QDRANT_TIMEOUT": "never",
    "RAG_ENGINE_SOMETHING_NEW": "x",
    "ENABLE_OCR": "true",
    "TESSDATA_PREFIX": "/nowhere/tessdata",
    "RAG_ENGINE_IT_QDRANT_URL": "http://it.example:6333",
    "RAG_ENGINE_IT_QDRANT_API_KEY": "it-key",
    "RAG_ENGINE_MODEL_CACHE": "/models/of/the/host",
    "RAG_ENGINE_PARITY_MIN": "0.5",
}
probe = pytest.mark.skipif(PROBE not in os.environ, reason="run by the test below, in a second pytest")


def visible() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if is_engine_variable(name)}


def test_the_names_kept_for_marked_tests_are_engine_variables():
    assert set(KEPT) == {name for names in KEPT_BY_MARKER.values() for name in names}
    assert all(is_engine_variable(name) for name in KEPT)
    assert all(is_engine_variable(name) for name in ENGINE_VARIABLES)
    assert not is_engine_variable("HF_HOME") and not is_engine_variable("FASTEMBED_CACHE_PATH")


def test_a_unit_test_sees_no_setting_of_the_host():
    assert visible() == {}


@probe
@pytest.mark.integration
def test_an_integration_test_is_told_where_the_server_is_and_nothing_else():
    assert visible() == {name: HOSTILE[name] for name in KEPT_BY_MARKER["integration"]}


@probe
@pytest.mark.models
def test_a_models_test_keeps_the_model_directory_and_the_tolerance_and_nothing_else():
    assert visible() == {name: HOSTILE[name] for name in KEPT_BY_MARKER["models"]}
    assert os.environ["HF_HOME"] == "/hf/home" and os.environ["FASTEMBED_CACHE_PATH"] == "/fastembed"


def test_the_suite_does_not_read_the_settings_of_the_host(tmp_path):
    # A second pytest, on tests that read the environment, with every variable set to
    # something unusable. The engine is found through PYTHONPATH, wherever it is installed.
    import rag_engine

    source = str(Path(rag_engine.__file__).resolve().parent.parent)
    environment = {name: value for name, value in os.environ.items() if not is_engine_variable(name)}
    environment.update(HOSTILE, HF_HOME="/hf/home", FASTEMBED_CACHE_PATH="/fastembed")
    environment[PROBE] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source, environment.get("PYTHONPATH")]))
    selected = [
        "tests/test_isolation.py",
        "tests/test_config.py::test_from_env_defaults_to_os_environ",
        "tests/test_store.py::test_without_settings_the_environment_is_read_at_construction",
        "tests/test_store.py::test_a_key_from_the_environment_is_not_kept_by_the_store",
        "tests/test_cli.py::test_the_settings_come_from_the_environment",
    ]
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly"]
    # Every marker: the default selection leaves the two marked tests of this file out.
    command += ["-m", "integration or not integration", f"--basetemp={tmp_path / 'second'}"]
    for name in (
        "test_the_suite_does_not_read_the_settings_of_the_host",
        "test_the_environment_of_the_host_is_put_back_when_the_run_ends",
    ):
        command += ["--deselect", f"tests/test_isolation.py::{name}"]
    result = subprocess.run(
        command + selected, cwd=ROOT, env=environment, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "8 passed, 2 deselected" in result.stdout and "skipped" not in result.stdout, result.stdout


def test_the_environment_of_the_host_is_put_back_when_the_run_ends(tmp_path):
    # pytest may run inside a larger process (a tool that calls pytest.main): what was taken
    # out of the environment is there again afterwards.
    code = (
        "import os, sys, pytest\n"
        "code = pytest.main(['-q', '-p', 'no:cacheprovider', sys.argv[1],"
        " 'tests/test_isolation.py::test_a_unit_test_sees_no_setting_of_the_host'])\n"
        "assert code == 0, code\n"
        "assert os.environ['QDRANT_PORT'] == 'abc' and os.environ['TESSDATA_PREFIX'] == '/nowhere/tessdata'\n"
    )
    import rag_engine

    source = str(Path(rag_engine.__file__).resolve().parent.parent)
    environment = dict(os.environ, **HOSTILE)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source, environment.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", code, f"--basetemp={tmp_path / 'second'}"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
