"""``tools/capture_golden.py``: where it writes, and what it refuses to overwrite.

The capture itself needs the previous pipeline and is not run here: ``capture`` is replaced.
The tool is not part of the package nor of the source distribution; the tests skip without it.
"""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "capture_golden.py"
RESULT = {"documents": [{"file": "a.md", "points": [{"payload": {"text": "é"}}]}]}


@pytest.fixture
def tool(monkeypatch):
    if not TOOL.is_file():
        pytest.skip("tools/capture_golden.py is not in this tree")
    spec = importlib.util.spec_from_file_location("capture_golden_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.captures = 0

    def capture():
        module.captures += 1
        return RESULT

    monkeypatch.setattr(module, "capture", capture)
    return module


def test_output_chooses_the_file_written(tool, tmp_path, capsys):
    target = tmp_path / "captured.json"
    assert tool.main(["--output", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == RESULT
    assert "é" in target.read_text(encoding="utf-8")  # written as UTF-8, not escaped
    assert str(target) in capsys.readouterr().out


def test_the_default_output_is_the_fixture(tool, tmp_path, monkeypatch):
    assert tool.EXPECTED == TOOL.parent.parent / "tests" / "fixtures" / "golden" / "expected.json"
    monkeypatch.setattr(tool, "EXPECTED", tmp_path / "expected.json")
    # The default of --output is fixed when main() builds its parser, from EXPECTED.
    assert tool.main([]) == 0
    assert (tmp_path / "expected.json").is_file()


def test_an_existing_file_is_not_overwritten_and_nothing_is_captured(tool, tmp_path, capsys):
    target = tmp_path / "captured.json"
    target.write_text("kept", encoding="utf-8")
    assert tool.main(["--output", str(target)]) == 1
    assert target.read_text(encoding="utf-8") == "kept"
    assert tool.captures == 0
    assert "--force" in capsys.readouterr().err


def test_force_overwrites_an_existing_file(tool, tmp_path):
    target = tmp_path / "captured.json"
    target.write_text("old", encoding="utf-8")
    assert tool.main(["--output", str(target), "--force"]) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == RESULT


def test_the_committed_fixture_is_protected_by_default(tool):
    if not tool.EXPECTED.is_file():
        pytest.skip("no golden fixture in this tree")
    before = tool.EXPECTED.read_bytes()
    assert tool.main([]) == 1
    assert tool.EXPECTED.read_bytes() == before
    assert tool.captures == 0
