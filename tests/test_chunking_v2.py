"""Chunking v2: token cap, table splits with the header repeated, context header.

Offline: a fake tokenizer counts one token per whitespace word plus two specials. The test of
the real e5 tokenizer is marked ``models`` and deselected by default.
"""

import importlib.util
import itertools
from pathlib import Path

import pytest

from rag_engine.chunking import Chunk, ChunkingOptions, chunk_file, chunk_markdown, load_tokenizer

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("chunknorris") is None, reason="needs the ingest extra"
)

FIXTURE = Path(__file__).parent / "fixtures" / "msft_10k_excerpt.md"
TITLE = "Contoso Corporation Annual Report on Form 10-K"


class FakeTokenizer:
    """One token per whitespace word, plus two special tokens."""

    def encode(self, text: str) -> list[int]:
        return [0] * (len(text.split()) + 2)


@pytest.fixture(scope="module")
def excerpt() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def v2_chunks(excerpt) -> list[Chunk]:
    return chunk_markdown(excerpt, options=ChunkingOptions.v2(), title=TITLE, tokenizer=FakeTokenizer())


def test_v1_defaults_unchanged():
    assert ChunkingOptions() == ChunkingOptions(
        max_headers_to_use="h4",
        max_chunk_word_count=200,
        hard_max_chunk_word_count=400,
        min_chunk_word_count=15,
    )
    options = ChunkingOptions()
    assert (options.max_chunk_token_count, options.table_rows_per_chunk) == (None, None)
    assert (options.context_header, options.overlap_tokens, options.version) == (False, 0, "v1")


def test_v1_chunk_file_output_unchanged(excerpt, tmp_path):
    from chunknorris.chunkers import MarkdownChunker
    from chunknorris.parsers import MarkdownParser

    path = tmp_path / "doc.md"
    path.write_text(excerpt, encoding="utf-8")
    expected = [
        Chunk(c.get_text(), c.start_page)
        for c in MarkdownChunker().chunk(MarkdownParser().parse_string(excerpt))
    ]
    assert chunk_file(path) == expected
    assert all(c.context is None for c in expected)
    assert all(c.embedded_text == c.text for c in expected)


def test_v2_profile_values():
    v2 = ChunkingOptions.v2()
    assert v2.max_chunk_token_count == 480
    assert v2.table_rows_per_chunk is None
    assert v2.context_header is True
    assert v2.overlap_tokens == 32
    assert v2.version == "v2"
    assert "480" in (ChunkingOptions.v2.__doc__ or "")


def test_embedded_text():
    assert Chunk("body", None, "A › B").embedded_text == "A › B\n\nbody"
    assert Chunk("body", None).embedded_text == "body"


def test_v2_no_chunk_over_cap(v2_chunks):
    tok = FakeTokenizer()
    assert len(v2_chunks) > 10
    for chunk in v2_chunks:
        assert len(tok.encode("passage: " + chunk.embedded_text)) <= 512
        assert len(tok.encode(chunk.embedded_text)) <= 480


def test_v2_table_split_repeats_header(v2_chunks):
    table_chunks = [c for c in v2_chunks if "| Segment 0 line" in c.text or "| Segment " in c.text]
    assert len(table_chunks) > 5
    header = "| Segment | Fiscal Year | Revenue | Operating income | Margin | Commentary |"
    separator = "|---|---|---|---|---|---|"
    for chunk in table_chunks:
        assert chunk.text.startswith(header + "\n" + separator + "\n")
    # every row of the table survives exactly once
    rows = [
        line
        for c in table_chunks
        for line in c.text.splitlines()
        if line.startswith("| Segment ") and "line" in line
    ]
    assert len(rows) == 300


def test_v2_table_rows_per_chunk(excerpt):
    options = ChunkingOptions(**{**ChunkingOptions.v2().__dict__, "table_rows_per_chunk": 10})
    chunks = chunk_markdown(excerpt, options=options, title=TITLE, tokenizer=FakeTokenizer())
    table_chunks = [c for c in chunks if "| Segment" in c.text]
    assert len(table_chunks) == 30
    assert all(c.text.count("\n|") == 11 for c in table_chunks)  # separator + 10 rows


def test_v2_context_header(v2_chunks):
    item7 = [c for c in v2_chunks if c.context and "Item 7" in c.context]
    assert item7
    for chunk in item7:
        assert chunk.context.startswith(TITLE)
        assert "Item 7" in chunk.context
        assert "#" not in chunk.context
    liquidity = [c for c in item7 if c.context.endswith("Liquidity and Capital Resources")]
    assert liquidity
    assert all(c.context.count(TITLE) == 1 for c in v2_chunks if c.context)
    # the context replaces the header lines at the top of the text
    assert not any(c.text.startswith("#") for c in item7)


def test_v2_long_paragraph_windows_overlap():
    words = [f"w{i}" for i in range(2000)]
    text = "# Doc\n\n## Part\n\n" + " ".join(words) + "\n"
    options = ChunkingOptions(
        **{**ChunkingOptions.v2().__dict__, "max_chunk_token_count": 100, "overlap_tokens": 10}
    )
    chunks = chunk_markdown(text, options=options, title="Doc", tokenizer=FakeTokenizer())
    tok = FakeTokenizer()
    assert len(chunks) > 15
    assert all(len(tok.encode(c.embedded_text)) <= 100 for c in chunks)
    for previous, current in itertools.pairwise(chunks):
        shared = set(previous.text.split()) & set(current.text.split())
        assert 0 < len(shared) <= 10
    assert set(" ".join(c.text for c in chunks).split()) >= set(words)


def test_cap_without_tokenizer_raises(excerpt):
    with pytest.raises(ValueError, match="tokenizer"):
        chunk_markdown(excerpt, options=ChunkingOptions.v2())


def test_options_validation():
    for bad in (
        {"max_chunk_token_count": 0},
        {"max_chunk_token_count": True},
        {"table_rows_per_chunk": 0},
        {"overlap_tokens": -1},
        {"max_chunk_token_count": 50, "overlap_tokens": 50},
        {"version": ""},
    ):
        with pytest.raises(ValueError):
            ChunkingOptions(**bad)


def test_v1_options_without_tokenizer_work(excerpt):
    chunks = chunk_markdown(excerpt)
    assert chunks and all(c.context is None for c in chunks)


@pytest.mark.models
def test_v2_cap_with_real_e5_tokenizer(excerpt, tmp_path):
    tok = load_tokenizer(tmp_path, "intfloat/multilingual-e5-base")
    chunks = chunk_markdown(excerpt, options=ChunkingOptions.v2(), title=TITLE, tokenizer=tok)
    assert chunks
    for chunk in chunks:
        assert len(tok.encode("passage: " + chunk.embedded_text)) <= 512


def test_load_tokenizer_verifies_hash(tmp_path, monkeypatch):
    from rag_engine import embedding

    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(embedding, "_pinned_model_dir", lambda *args, **kwargs: tmp_path)
    with pytest.raises(embedding.ModelIntegrityError, match="tokenizer.json"):
        load_tokenizer(tmp_path, "intfloat/multilingual-e5-base")


def test_load_tokenizer_unknown_model(tmp_path):
    with pytest.raises(ValueError, match="no pinned tokenizer"):
        load_tokenizer(tmp_path, "nobody/not-a-model")
