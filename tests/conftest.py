"""Shared test configuration.

* The search tests run once per case of the filter matrix.
* No test reads the settings of the host. The variables the engine reads (``QDRANT_*``,
  ``RAG_ENGINE_*``, ``ENABLE_OCR``, ``TESSDATA_PREFIX``) are taken out of the environment for
  the whole run, and put back when it ends; a test that needs one sets it itself. The few
  variables that are meant for the tests are named in ``KEPT_BY_MARKER`` and reach the tests
  of that marker only.
* The test run sends no telemetry. The first statement of this file sets
  ``ORT_DISABLE_TELEMETRY=1`` unless the variable is set already: the ONNX runtime reads it when
  it is imported, and the test modules import ``qdrant_client`` (which loads the runtime when
  ``fastembed`` is installed) before the engine has set its own default. The variable is not one
  of the engine's settings, so the isolation above leaves it alone; the tests of
  ``test_telemetry.py`` choose its value themselves, in processes of their own.
"""

from __future__ import annotations

import os

# Before every other import: pytest and the helper modules below may load the runtime.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import pytest
from search_corpus import SEARCH_CASES

#: The environment as pytest was started with, for the helpers that must ask the host itself
#: (where Tesseract keeps its language data).
HOST_ENVIRONMENT: dict[str, str] = dict(os.environ)

#: What the engine reads: the settings (``QDRANT_*``, ``RAG_ENGINE_*``) and the two OCR switches.
ENGINE_PREFIXES = ("QDRANT_", "RAG_ENGINE_")
ENGINE_VARIABLES = ("ENABLE_OCR", "TESSDATA_PREFIX")

#: What the tests of a marker are given all the same. The integration tests are told where the
#: real server is; the tests of the real models keep the model directory and the tolerance of
#: the parity test. The caches of the model libraries (``HF_HOME``, ``HF_HUB_CACHE``,
#: ``FASTEMBED_CACHE_PATH``, ...) are not settings of the engine and are never touched.
KEPT_BY_MARKER = {
    "integration": ("RAG_ENGINE_IT_QDRANT_URL", "RAG_ENGINE_IT_QDRANT_API_KEY"),
    "models": ("RAG_ENGINE_MODEL_CACHE", "RAG_ENGINE_PARITY_MIN"),
}
KEPT = tuple(name for names in KEPT_BY_MARKER.values() for name in names)

_removed: dict[str, str] = {}


def is_engine_variable(name: str) -> bool:
    return name.startswith(ENGINE_PREFIXES) or name in ENGINE_VARIABLES


def pytest_configure(config: pytest.Config) -> None:
    """Take the settings of the host out of the environment, before any test module is imported.

    For the whole run and not per test: a module fixture is set up before the function
    fixtures of the test that asks for it, and must not see them either. The names of
    ``KEPT`` stay, for the modules and module fixtures of the marked tests (the integration
    modules read the URL of the server when they are imported); the fixture below hides
    them from every other test.
    """
    for name in sorted(os.environ):
        if is_engine_variable(name) and name not in KEPT:
            _removed[name] = os.environ.pop(name)


def pytest_unconfigure(config: pytest.Config) -> None:
    os.environ.update(_removed)
    _removed.clear()


@pytest.fixture(scope="session")
def host_environment() -> dict[str, str]:
    """The environment pytest was started with, settings of the host included."""
    return dict(HOST_ENVIRONMENT)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run a test that takes a ``search_case`` argument once per entry of ``SEARCH_CASES``."""
    if "search_case" in metafunc.fixturenames:
        metafunc.parametrize("search_case", SEARCH_CASES, ids=[case.label for case in SEARCH_CASES])


@pytest.fixture(autouse=True)
def _engine_environment(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide from a test the variables kept for the tests of another marker."""
    for marker, names in KEPT_BY_MARKER.items():
        if request.node.get_closest_marker(marker) is None:
            for name in names:
                monkeypatch.delenv(name, raising=False)
