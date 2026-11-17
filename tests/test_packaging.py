"""Packaging guarantees: the base import stays light."""

import subprocess
import sys

import pytest

HEAVY_MODULES = ("torch", "sentence_transformers", "fastembed", "chunknorris", "markitdown")


def test_version():
    import rag_engine

    assert rag_engine.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "modules",
    ["rag_engine", "rag_engine.config", "rag_engine.schema", "rag_engine.store"],
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
