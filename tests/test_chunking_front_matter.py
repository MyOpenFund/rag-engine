"""A text that starts with ``---`` loses no words.

ChunkNorris's ``MarkdownParser.parse_string`` reads a string that starts with ``---`` (after its
surrounding whitespace is stripped) as YAML front matter and drops everything up to the next
``---`` line. Filings converted to markdown have ``---`` page breaks, and a section cut at an
offset often starts on one: the first page of the section was lost. rag-engine never reads front
matter: the text is kept, a real YAML block included.
"""

import dataclasses
import importlib.util
import random
import re

import pytest

from rag_engine.chunking import ChunkingOptions, chunk_file, chunk_markdown

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("chunknorris") is None, reason="needs the ingest extra"
)


class FakeTokenizer:
    """One token per whitespace word, plus two special tokens."""

    def encode(self, text: str) -> list[int]:
        return [0] * (len(text.split()) + 2)


#: The production profiles.
DEFAULTS = {"v1": ChunkingOptions(), "v2": ChunkingOptions.v2()}
#: The same without the minimum size: ChunkNorris drops a chunk under ``min_chunk_word_count``
#: words by design, which would hide what the parser loses in the short texts below.
PROFILES = {name: dataclasses.replace(options, min_chunk_word_count=0) for name, options in DEFAULTS.items()}

_WORD = re.compile(r"[A-Za-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text))


def _chunk(text: str, profile: str, profiles=PROFILES):
    return chunk_markdown(text, options=profiles[profile], title="Doc", tokenizer=FakeTokenizer())


def _missing_words(text: str, chunks) -> list[str]:
    """The words of ``text`` found in no chunk (text or context)."""
    seen: set[str] = set()
    for chunk in chunks:
        seen |= _words(chunk.text)
        if chunk.context is not None:
            seen |= _words(chunk.context)
    return sorted(_words(text) - seen)


# Each page is over the 15-word minimum of the production profiles.
PAGE_BREAK_CUT = (
    "---\n"
    "Alpha page one opens the section with a discussion of net sales growth in every segment.\n"
    "\n"
    "Bravo page one continues with operating income commentary and the effect of foreign exchange.\n"
    "---\n"
    "Charlie page two talks about liquidity and capital resources, cash flows and the debt.\n"
)


@pytest.mark.parametrize("profiles", [DEFAULTS, PROFILES], ids=["production", "no-minimum"])
@pytest.mark.parametrize("profile", PROFILES)
def test_a_leading_page_break_keeps_the_first_page(profile, profiles):
    chunks = _chunk(PAGE_BREAK_CUT, profile, profiles)
    joined = " ".join(c.text for c in chunks)
    assert "Alpha" in joined and "Bravo" in joined and "Charlie" in joined
    for line in PAGE_BREAK_CUT.splitlines():
        if line.strip() and line.strip() != "---":
            assert not _missing_words(line, chunks), line


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "prefix",
    ["", "\n\n", "   \n \t\n", "\ufeff", "\ufeff\n  \n", "----\n", "---\n\n---\n", "------\n---\n"],
    ids=["none", "blank-lines", "spaces-tabs", "bom", "bom-blank", "four-dashes", "two-breaks", "stacked"],
)
def test_leading_whitespace_bom_and_repeated_breaks_lose_nothing(profile, prefix):
    text = prefix + PAGE_BREAK_CUT
    assert _missing_words(text, _chunk(text, profile)) == []


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "first_line",
    ["---UNITED STATES", "---[Table of Contents](#toc_page)", "---ARPU:--$11.57---$9.54", "--- Page 3 ---"],
)
def test_a_first_line_glued_to_its_dashes_keeps_its_words(profile, first_line):
    # Lines of the converted filings: the dashes are a page break glued to the next line.
    text = f"{first_line}\nDelta body text of the first page.\n---\nEcho body text of the next page.\n"
    assert _missing_words(text, _chunk(text, profile)) == []


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("rule", ["---", "------", "===", "--- "])
def test_a_rule_glued_to_text_under_a_line_keeps_its_words(profile, rule):
    # ChunkNorris reads a line starting with --- or === under a non-blank line as a setext
    # underline and drops the whole line, the text glued to the rule with it.
    text = f"Juliet intro line.\n{rule}UNITED STATES Kilo glued words\nLima next line.\n"
    assert _missing_words(text, _chunk(text, profile)) == []


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("rules", ["---\n---", "---\n-----\n---", "===\n---", "---\n---Mike glued"])
def test_stacked_rules_under_a_line_keep_its_words(profile, rules):
    # The first rule makes a heading of the line above; ChunkNorris reads the second as the
    # underline of the first and replaces that heading with it.
    text = f"Nova intro line here.\n{rules}\nOscar next line.\n"
    assert _missing_words(text, _chunk(text, profile)) == []


def test_a_rule_glued_to_text_inside_a_code_block_is_left_alone():
    from rag_engine.chunking import _markdown_for_parser

    text = "Intro.\n```\ncode line\n---not split\n```\nAfter.\n"
    assert _markdown_for_parser(text) == text


@pytest.mark.parametrize("profile", PROFILES)
def test_real_yaml_front_matter_is_kept_as_text(profile):
    # Deliberate: documents are not Jekyll pages. A YAML header is content like any other.
    text = "---\ntitle: Foxtrot Annual Report\nauthor: Golf Hotel\n---\n\nIndia body paragraph here.\n"
    chunks = _chunk(text, profile)
    assert _missing_words(text, chunks) == []


@pytest.mark.parametrize("profile", PROFILES)
def test_a_text_without_leading_dashes_is_chunked_as_before(profile):
    from chunknorris.chunkers import MarkdownChunker
    from chunknorris.parsers import MarkdownParser

    from rag_engine.chunking import _markdown_chunker, _to_chunks

    text = "# Title\n\nKilo paragraph one.\n\n---\n\nLima paragraph two.\n"
    options = PROFILES[profile]
    raw = _markdown_chunker(MarkdownChunker, options).chunk(MarkdownParser().parse_string(text))
    assert _chunk(text, profile) == _to_chunks(raw, options, "Doc", FakeTokenizer())


@pytest.mark.parametrize("name", ["cut.md", "cut.txt"])
@pytest.mark.parametrize("profile", PROFILES)
def test_chunk_file_keeps_the_first_page_of_a_text_file(tmp_path, name, profile):
    path = tmp_path / name
    path.write_text(PAGE_BREAK_CUT, encoding="utf-8")
    chunks = chunk_file(path, options=DEFAULTS[profile], tokenizer=FakeTokenizer())
    assert _missing_words(PAGE_BREAK_CUT, chunks) == []


def _random_document(rng: random.Random) -> str:
    vocabulary = [f"w{i}" for i in range(400)]
    lines: list[str] = []
    if rng.random() < 0.7:
        lines.append(rng.choice(["---", "----", "---\n", "\n---", "  ---", "\ufeff---"]))
    for _ in range(rng.randint(1, 12)):
        kind = rng.random()
        if kind < 0.35:
            lines.append(rng.choice(["---", "-----", "", "===", " ---", "--- ---"]))
        elif kind < 0.45:
            rule = rng.choice(["---", "----", "===", "--- ", "--- ---", "---===", "=== --- ", "--- --- "])
            lines.append(rule + " ".join(rng.choices(vocabulary, k=rng.randint(1, 5))))
        else:
            lines.append(" ".join(rng.choices(vocabulary, k=rng.randint(1, 60))))
        if rng.random() < 0.5:
            lines.append("")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("seed", range(100))
def test_random_paragraphs_and_rules_lose_no_word(profile, seed):
    text = _random_document(random.Random(seed))
    assert _missing_words(text, _chunk(text, profile)) == [], text


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "line",
    ["--- ---foo bar", "---===foo bar", "=== --- foo bar", "=== === foo", "--- --- Page 3", "--- ---\tfoo"],
)
def test_a_rule_glued_to_another_rule_and_text_keeps_every_word(profile, line):
    # Split once, the text left ("---foo bar") is itself a rule under the rule split off: it
    # needs the same check, else the line above and the glued text are both lost.
    text = f"Intro alpha.\n{line}\nNext beta.\n"
    assert _missing_words(text, _chunk(text, profile)) == []


def _parsed(text: str) -> str:
    from chunknorris.parsers import MarkdownParser

    from rag_engine.chunking import _parse_markdown

    return _parse_markdown(MarkdownParser, text).to_string()


YAML_HEADER = "---\ntitle: Foxtrot Annual Report\ntags: alpha, bravo\n---\nIndia body paragraph here.\n"


def test_yaml_front_matter_does_not_turn_its_last_key_into_a_heading():
    # Decision: the header is plain text. Its closing --- is a thematic break, not a setext
    # underline under "tags: ...".
    parsed = _parsed(YAML_HEADER)
    assert not [line for line in parsed.splitlines() if line.startswith("#")], parsed
    assert "tags: alpha, bravo" in parsed and "India body paragraph here." in parsed


@pytest.mark.parametrize("profile", PROFILES)
def test_yaml_front_matter_keys_are_in_no_heading_context(profile):
    chunks = _chunk(YAML_HEADER, profile)
    assert _missing_words(YAML_HEADER, chunks) == []
    for chunk in chunks:
        assert "tags" not in (chunk.context or ""), chunk.context
        assert not [line for line in chunk.text.splitlines() if line.startswith("#")], chunk.text


def test_a_page_break_cut_does_not_make_a_heading_of_the_line_above_its_second_break():
    parsed = _parsed(PAGE_BREAK_CUT)
    assert not [line for line in parsed.splitlines() if line.startswith("#")], parsed


def test_a_setext_heading_stays_a_heading():
    assert _parsed("Real Heading\n---\nBody words alpha beta.\n").splitlines()[0] == "## Real Heading"
    assert _parsed("Real Heading\n===\nBody words alpha beta.\n").splitlines()[0] == "# Real Heading"


@pytest.mark.parametrize(
    "text",
    [
        "Intro alpha.\n\n| a | b |\n|---|---|\n| 1 | 2 |",
        "a | b\n--- | ---\none | two",
        "Intro.\n```\ncode line\n---not split\n```\nAfter.",
    ],
    ids=["pipe-table", "table-without-pipes", "code-fence"],
)
def test_tables_and_code_fences_are_left_alone(text):
    from rag_engine.chunking import _markdown_for_parser

    assert _markdown_for_parser(text) == text


@pytest.mark.parametrize("profile", PROFILES)
def test_a_glued_rule_inside_a_code_fence_keeps_its_words_through_chunk_markdown(profile):
    text = "Intro alpha.\n```\ncode line gamma\n---kept delta\n```\nAfter epsilon.\n"
    assert _missing_words(text, _chunk(text, profile)) == []


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "text",
    [
        "---\r\npage one alpha\r\n---\r\npage two beta\r\n",
        "Intro alpha\r\n---GLUED gamma\r\nbeta\r\n",
        "Intro alpha\r\n--- ---foo gamma\r\nbeta\r\n",
    ],
    ids=["leading", "glued", "glued-twice"],
)
def test_crlf_text_keeps_its_words(profile, text):
    assert _missing_words(text, _chunk(text, profile)) == []


@pytest.mark.parametrize("profile", PROFILES)
def test_a_rule_left_after_the_parser_removes_a_base64_image_keeps_the_first_page(profile):
    # ChunkNorris removes base64 images first: the text then starts with the --- glued to it.
    text = "data:image/png;base64,AAAA---\npage one alpha\n---\npage two beta\n"
    assert {"page", "one", "alpha", "two", "beta"} <= _words(" ".join(c.text for c in _chunk(text, profile)))


@pytest.mark.parametrize("profile", PROFILES)
def test_chunk_file_keeps_glued_rule_text_of_a_pptx(tmp_path, profile):
    pptx = pytest.importorskip("pptx")
    pytest.importorskip("markitdown")
    from pptx.util import Inches

    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Deck title"
    box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(5), Inches(2)).text_frame
    box.text = "Alpha slide words here\n--- ---UNITED STATES Kilo\nBravo more words on the slide"
    path = tmp_path / "deck.pptx"
    deck.save(path)
    chunks = chunk_file(path, options=PROFILES[profile], tokenizer=FakeTokenizer())
    assert _missing_words(box.text, chunks) == []


@pytest.mark.parametrize(
    "text",
    [
        "text\n" + "-" * 600_000 + "x\nmore\n",
        "text\n" + "-" * 600_000 + "\nmore\n",
        "text\n---" + "." * 600_000 + "\nmore\n",
        "text\n" + "--- " * 150_000 + "foo\n",
        "---\n" * 150_000 + "text\n",
        "---\n\n" * 120_000 + "text\n",
        "text\n" + "---\n" * 150_000,
        ("word " * 40 + "\n---GLUED text here\n") * 2_800,
    ],
    ids=[
        "dash-line-glued",
        "dash-line",
        "dash-punctuation",
        "rule-groups",
        "leading-rules",
        "leading-rules-blank",
        "stacked-rules",
        "glued-every-other-line",
    ],
)
def test_preprocessing_is_linear_on_pathological_input(text):
    import time

    from rag_engine.chunking import _markdown_for_parser

    assert len(text) >= 600_000
    start = time.perf_counter()
    _markdown_for_parser(text)
    assert time.perf_counter() - start < 1.0
