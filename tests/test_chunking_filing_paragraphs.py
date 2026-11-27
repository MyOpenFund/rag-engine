"""Every paragraph and table row of a filing-shaped Markdown ends up in some chunk.

A 10-K converted to Markdown by company-corpus has ``---`` page breaks, a table of contents link
and a one-row title table at the top of a page, short header-like lines ("Forward-Looking
Statements", "ITEM 1A.RISK FACTORS") and page numbers. The no-vault MCP and
``scripts/filings_eval.py --cuts section`` chunk each Item section on its own, so a piece starts on
a page break followed by a table. 0.2.0 lost the first page of such pieces (ChunkNorris read it as
YAML front matter, fixed in 0.2.1). This test checks, under the production profiles, that the
parser and the chunker lose no line of the fixture, cut by section as the MCP cuts it and whole.
Each section piece fits in one chunk; the whole document is split (2 chunks in v1, about 10 in v2),
so that run also checks that no line falls between two chunks.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from rag_engine.chunking import ChunkingOptions, chunk_markdown

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("chunknorris") is None, reason="needs the ingest extra"
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "filing_section_cuts.md"
TITLE = "CONTOSO CORP 10-K for the period ended 2025-12-31, filed 2026-02-01"
#: The production profiles, minimum chunk size included: what the MCP ships. ChunkNorris drops a
#: chunk under 15 words by design, so every piece of the fixture keeps more than 15 words of prose
#: (the cover has a long sentence for that): a piece made only of short lines would give no chunk.
PRODUCTION = {"v1": ChunkingOptions(), "v2": ChunkingOptions.v2()}
#: A line of each Item section; the section starts on the page break above it, as the
#: ``sections`` of company-corpus do.
SECTION_MARKERS = (("Item 1", "| Item 1. |"), ("Item 1A", "ITEM 1A.RISK FACTORS"), ("Item 7", "| Item 7. |"))


class FakeTokenizer:
    """One token per whitespace word, plus two special tokens."""

    def encode(self, text: str) -> list[int]:
        return [0] * (len(text.split()) + 2)


def _load_section_pieces():
    path = ROOT / "scripts" / "filings_eval.py"
    spec = importlib.util.spec_from_file_location("filings_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module.section_pieces


def _norm(text: str) -> str:
    """Lower-case words and numbers, one space apart, with a space at both ends (word bounds)."""
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def _sections(text: str) -> list[tuple[str, int, int]]:
    starts = []
    for name, marker in SECTION_MARKERS:
        assert marker in text, f"the fixture lost its {name} marker {marker!r}"
        starts.append((name, text.rindex("\n---\n", 0, text.index(marker)) + 1))
    ends = [start for _, start in starts[1:]] + [len(text)]
    return [(name, start, end) for (name, start), end in zip(starts, ends)]


def _lines(text: str) -> list[str]:
    """The lines of ``text`` with a letter: prose, titles, table rows; not rules or page numbers."""
    return [line.strip() for line in text.splitlines() if re.search(r"[A-Za-z]", line)]


def _missing(piece: str, profile: str) -> list[str]:
    """The lines of ``piece`` that occur fewer times in its chunks than in the piece.

    Counting occurrences, rather than testing containment, keeps a lost line from being masked by
    the same words elsewhere (a table of contents link and the table row repeating it). A chunk's
    text and its context are both searched: in v2 a heading lives in the context, and a line
    directly above a ``---`` page break is a setext heading (the fixture has one, as filings do).
    Each distinct context is counted once, since v2 repeats it on every chunk under the heading.
    """
    chunks = chunk_markdown(piece, options=PRODUCTION[profile], title=TITLE, tokenizer=FakeTokenizer())
    contexts = dict.fromkeys(chunk.context for chunk in chunks if chunk.context)
    seen = _norm("\n".join([chunk.text for chunk in chunks] + list(contexts)))
    source = _norm(piece)
    return [
        line for line in dict.fromkeys(_lines(piece)) if seen.count(_norm(line)) < source.count(_norm(line))
    ]


@pytest.fixture(scope="module")
def filing() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pieces(filing) -> list[str]:
    return [piece for _, piece in _load_section_pieces()(filing, _sections(filing))]


def test_the_fixture_has_the_shapes_that_lost_text(filing, pieces):
    assert len(pieces) == 4  # the cover, then Items 1, 1A and 7
    assert pieces[0].startswith("---UNITED STATES")
    assert all(piece.startswith("---\n") for piece in pieces[1:])
    # A second bare page break in Items 1A and 7: 0.2.0 dropped everything above it as front matter.
    assert [piece.split("\n").count("---") >= 2 for piece in pieces[1:]] == [False, True, True]
    assert any(line.startswith("| Item 7. |") for line in _lines(pieces[3]))
    assert len(_lines(filing)) == 38, "the fixture changed: check its shapes are still there, then update"


@pytest.mark.parametrize("profile", PRODUCTION)
def test_every_line_of_a_section_cut_is_in_some_chunk(pieces, profile):
    missing = {index: _missing(piece, profile) for index, piece in enumerate(pieces)}
    assert missing == {index: [] for index in range(len(pieces))}


@pytest.mark.parametrize("profile", PRODUCTION)
def test_every_line_of_the_whole_filing_is_in_some_chunk(filing, profile):
    assert _missing(filing, profile) == []
