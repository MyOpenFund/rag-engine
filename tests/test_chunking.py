"""Chunking: extension dispatch, OCR mode, blank chunks, pages, error handling.

Two layers. The first runs everywhere: stub ``chunknorris`` / ``markitdown`` modules are put in
``sys.modules``, so the dispatch and the contract are tested without the ``ingest`` extra. The
second (bottom of the file) needs the real libraries and skips when they are not installed.
"""

import importlib.util
import inspect
import logging
import os
import random
import shutil
import subprocess
import sys
import types

import pytest

from rag_engine import chunking
from rag_engine.chunking import (
    SUPPORTED_EXTENSIONS,
    Chunk,
    ChunkingOptions,
    UnsupportedFormatError,
    chunk_file,
    ocr_available,
)

# ---------------------------------------------------------------------------
# Layer 1: stub libraries
# ---------------------------------------------------------------------------


class StubChunk:
    def __init__(self, text, start_page=None):
        self._text = text
        self.start_page = start_page

    def get_text(self):
        return self._text


class PagelessChunk:
    """A chunk object with no ``start_page`` attribute at all."""

    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text


class StubTextNotFound(Exception):
    pass


class Stubs:
    """What the stub libraries were asked to do, and what they answer."""

    def __init__(self):
        self.calls = []
        self.chunks = [StubChunk("some text", 0)]
        self.parse_error = None
        self.chunk_error = None
        self.pptx_markdown = "# Deck\n\nslide text"


def _install_stubs(monkeypatch, *, with_line_spacing=True):
    stubs = Stubs()

    def make_parser(name):
        class Parser:
            def __init__(self, **kwargs):
                stubs.calls.append((name, "init", kwargs))

            def parse_file(self, filepath):
                stubs.calls.append((name, "parse_file", filepath))
                if stubs.parse_error is not None:
                    raise stubs.parse_error
                return ("doc", name, filepath)

            def parse_string(self, string):
                stubs.calls.append((name, "parse_string", string))
                if stubs.parse_error is not None:
                    raise stubs.parse_error
                return ("doc", name, string)

        Parser.__name__ = name
        return Parser

    class MarkdownChunker:
        def __init__(self, **kwargs):
            stubs.calls.append(("MarkdownChunker", "init", kwargs))

        def chunk(self, doc):
            stubs.calls.append(("MarkdownChunker", "chunk", doc))
            if stubs.chunk_error is not None:
                raise stubs.chunk_error
            return list(stubs.chunks)

    class MarkItDown:
        def convert(self, filepath):
            stubs.calls.append(("MarkItDown", "convert", filepath))
            return types.SimpleNamespace(text_content=stubs.pptx_markdown)

    pdf_parser = make_parser("PdfParser")
    if with_line_spacing:
        pdf_parser._get_line_spacing = staticmethod(lambda lines: "original")

    root = types.ModuleType("chunknorris")
    root.__path__ = []
    chunkers = types.ModuleType("chunknorris.chunkers")
    chunkers.MarkdownChunker = MarkdownChunker
    parsers = types.ModuleType("chunknorris.parsers")
    parsers.PdfParser = pdf_parser
    for name in ("CSVParser", "DocxParser", "ExcelParser", "MarkdownParser"):
        setattr(parsers, name, make_parser(name))
    exceptions = types.ModuleType("chunknorris.exceptions")
    exceptions.TextNotFoundException = StubTextNotFound
    markitdown = types.ModuleType("markitdown")
    markitdown.MarkItDown = MarkItDown

    for module in (root, chunkers, parsers, exceptions, markitdown):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    stubs.pdf_parser = pdf_parser
    return stubs


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.delenv("ENABLE_OCR", raising=False)
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    return _install_stubs(monkeypatch)


def _file(tmp_path, name, data=b"content"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _inits(stubs):
    """The parsers built, with their arguments (the chunker is not a parser)."""
    return [(name, arg) for name, what, arg in stubs.calls if what == "init" and name != "MarkdownChunker"]


def _chunker_kwargs(stubs):
    (kwargs,) = [arg for name, what, arg in stubs.calls if (name, what) == ("MarkdownChunker", "init")]
    return kwargs


#: What PdfParser receives besides the OCR mode and languages.
PDF_FIXED_KWARGS = {
    "extract_tables": True,
    "add_headers": True,
    "body_line_spacing": None,
    "enable_ml_features": False,
}


def _pdf_kwargs(stubs):
    (kwargs,) = [arg for name, what, arg in stubs.calls if (name, what) == ("PdfParser", "init")]
    return kwargs


# -- public constants and the Chunk type --------------------------------------


def test_supported_extensions_are_the_ones_the_old_engine_read():
    assert isinstance(SUPPORTED_EXTENSIONS, tuple)
    assert set(SUPPORTED_EXTENSIONS) == {".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".txt", ".md"}
    assert len(set(SUPPORTED_EXTENSIONS)) == len(SUPPORTED_EXTENSIONS)
    assert all(ext == ext.lower() and ext.startswith(".") for ext in SUPPORTED_EXTENSIONS)


def test_chunk_is_a_frozen_value():
    chunk = Chunk(text="a", page=3)
    assert chunk == Chunk("a", 3)
    with pytest.raises(AttributeError):
        chunk.text = "b"


# -- dispatch -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "parser"),
    [("a.pdf", "PdfParser"), ("a.docx", "DocxParser"), ("a.xlsx", "ExcelParser"), ("a.csv", "CSVParser")],
)
def test_binary_and_tabular_formats_go_to_their_parser(stubs, tmp_path, name, parser):
    path = _file(tmp_path, name)
    assert chunk_file(path) == [Chunk("some text", 0)]
    assert [n for n, _ in _inits(stubs)] == [parser]
    assert (parser, "parse_file", str(path)) in stubs.calls
    assert stubs.calls[-1] == ("MarkdownChunker", "chunk", ("doc", parser, str(path)))


@pytest.mark.parametrize("name", ["notes.txt", "notes.md"])
def test_text_formats_are_read_here_and_parsed_as_markdown(stubs, tmp_path, name):
    path = _file(tmp_path, name, "# Titre\n\ncafé\n".encode())
    chunk_file(path)
    assert [n for n, _ in _inits(stubs)] == ["MarkdownParser"]
    assert ("MarkdownParser", "parse_string", "# Titre\n\ncafé\n") in stubs.calls
    assert not [c for c in stubs.calls if c[1] == "parse_file"]


def test_pptx_goes_through_markitdown_then_the_markdown_parser(stubs, tmp_path):
    path = _file(tmp_path, "deck.pptx")
    chunk_file(path)
    assert ("MarkItDown", "convert", str(path)) in stubs.calls
    assert ("MarkdownParser", "parse_string", "# Deck\n\nslide text") in stubs.calls
    assert [n for n, _ in _inits(stubs)] == ["MarkdownParser"]


@pytest.mark.parametrize(
    ("name", "parser"),
    [("A.PDF", "PdfParser"), ("b.Docx", "DocxParser"), ("c.CSV", "CSVParser"), ("d.TXT", "MarkdownParser")],
)
def test_dispatch_ignores_the_case_of_the_extension(stubs, tmp_path, name, parser):
    chunk_file(_file(tmp_path, name))
    assert [n for n, _ in _inits(stubs)] == [parser]


def test_only_the_last_suffix_counts(stubs, tmp_path):
    chunk_file(_file(tmp_path, "report.pdf.txt"))
    assert [n for n, _ in _inits(stubs)] == ["MarkdownParser"]


def test_a_string_path_is_accepted(stubs, tmp_path):
    path = _file(tmp_path, "a.csv")
    assert chunk_file(str(path)) == [Chunk("some text", 0)]


@pytest.mark.parametrize("name", ["page.html", "data.json", "old.doc", "noextension", "archive.pdf.zip"])
def test_unsupported_extension_is_refused_by_name(stubs, tmp_path, name):
    path = _file(tmp_path, name)
    with pytest.raises(UnsupportedFormatError) as error:
        chunk_file(path)
    assert isinstance(error.value, ValueError)
    suffix = path.suffix
    assert repr(suffix) in str(error.value)
    assert stubs.calls == []


def test_unsupported_extension_is_refused_without_the_ingest_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "chunknorris", None)
    with pytest.raises(ValueError, match="'.html'"):
        chunk_file(_file(tmp_path, "page.html"))


def test_bad_arguments_are_plain_value_errors_not_unsupported_format(stubs, tmp_path):
    assert issubclass(UnsupportedFormatError, ValueError)
    for kwargs in ({"ocr": "sometimes"}, {"ocr_languages": ""}, {"options": {"min_chunk_word_count": 1}}):
        with pytest.raises(ValueError) as error:
            chunk_file(_file(tmp_path, "a.pdf"), **kwargs)
        assert type(error.value) is ValueError


def test_a_value_error_of_the_parser_is_not_turned_into_unsupported_format(stubs, tmp_path):
    error = ValueError("Could not determine delimiter")
    stubs.parse_error = error
    with pytest.raises(ValueError) as raised:
        chunk_file(_file(tmp_path, "empty.csv"))
    assert raised.value is error
    assert not isinstance(raised.value, UnsupportedFormatError)


# -- chunking options ---------------------------------------------------------


def test_default_options_are_the_values_the_old_engine_ran_with():
    options = ChunkingOptions()
    assert (
        options.max_headers_to_use,
        options.max_chunk_word_count,
        options.hard_max_chunk_word_count,
        options.min_chunk_word_count,
    ) == ("h4", 200, 400, 15)
    with pytest.raises(AttributeError):
        options.min_chunk_word_count = 1


@pytest.mark.parametrize("name", ["a.pdf", "a.docx", "a.xlsx", "a.csv", "a.pptx", "a.txt", "a.md"])
@pytest.mark.parametrize("options", [None, ChunkingOptions()])
def test_the_chunker_receives_every_sizing_option_explicitly(stubs, tmp_path, name, options):
    chunk_file(_file(tmp_path, name), options=options)
    assert _chunker_kwargs(stubs) == {
        "max_headers_to_use": "h4",
        "max_chunk_word_count": 200,
        "hard_max_chunk_word_count": 400,
        "min_chunk_word_count": 15,
        "hard_max_chunk_token_count": None,
    }


def test_given_options_reach_the_chunker(stubs, tmp_path):
    options = ChunkingOptions(
        max_headers_to_use="h2", max_chunk_word_count=50, hard_max_chunk_word_count=80, min_chunk_word_count=3
    )
    chunk_file(_file(tmp_path, "a.md"), options=options)
    assert _chunker_kwargs(stubs) == {
        "max_headers_to_use": "h2",
        "max_chunk_word_count": 50,
        "hard_max_chunk_word_count": 80,
        "min_chunk_word_count": 3,
        "hard_max_chunk_token_count": None,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_headers_to_use": "h7"},
        {"max_headers_to_use": "H4"},
        {"max_headers_to_use": 4},
        {"min_chunk_word_count": -1},
        {"min_chunk_word_count": 1.5},
        {"min_chunk_word_count": True},
        {"max_chunk_word_count": 0},
        {"hard_max_chunk_word_count": 0},
        {"max_chunk_word_count": "200"},
        {"max_chunk_word_count": 300, "hard_max_chunk_word_count": 200},
    ],
)
def test_invalid_options_are_refused_when_built(kwargs):
    with pytest.raises(ValueError):
        ChunkingOptions(**kwargs)


def test_a_minimum_of_zero_words_and_equal_maxima_are_accepted():
    ChunkingOptions(min_chunk_word_count=0, max_chunk_word_count=100, hard_max_chunk_word_count=100)


@pytest.mark.parametrize("options", [{"min_chunk_word_count": 1}, "h4", 15])
def test_options_must_be_a_chunking_options_value(stubs, tmp_path, options):
    with pytest.raises(ValueError, match="options"):
        chunk_file(_file(tmp_path, "a.md"), options=options)
    assert stubs.calls == []


def test_the_pdf_parser_options_are_named_constants_passed_explicitly(stubs, tmp_path):
    assert {
        "extract_tables": chunking.PDF_EXTRACT_TABLES,
        "add_headers": chunking.PDF_ADD_HEADERS,
        "body_line_spacing": chunking.PDF_BODY_LINE_SPACING,
        "enable_ml_features": chunking.PDF_ENABLE_ML_FEATURES,
    } == PDF_FIXED_KWARGS
    chunk_file(_file(tmp_path, "a.pdf"))
    assert _pdf_kwargs(stubs) == {"use_ocr": "never", "ocr_language": "eng", **PDF_FIXED_KWARGS}


# -- OCR mode and languages ---------------------------------------------------


@pytest.mark.parametrize("ocr", ["Auto", "ALWAYS", "", "yes", None, True, "auto "])
def test_unknown_ocr_mode_is_refused(stubs, tmp_path, ocr):
    with pytest.raises(ValueError, match="ocr"):
        chunk_file(_file(tmp_path, "a.pdf"), ocr=ocr)
    assert stubs.calls == []


def test_ocr_mode_is_checked_for_every_format(stubs, tmp_path):
    with pytest.raises(ValueError, match="ocr"):
        chunk_file(_file(tmp_path, "a.md"), ocr="sometimes")


@pytest.mark.parametrize("languages", ["", "  ", None, 3, ["eng"]])
def test_ocr_languages_must_be_a_non_blank_string(stubs, tmp_path, languages):
    with pytest.raises(ValueError, match="ocr_languages"):
        chunk_file(_file(tmp_path, "a.pdf"), ocr_languages=languages)
    assert stubs.calls == []


def test_auto_is_never_when_ocr_is_not_available(stubs, tmp_path):
    chunk_file(_file(tmp_path, "a.pdf"))
    assert _pdf_kwargs(stubs) == {"use_ocr": "never", "ocr_language": "eng", **PDF_FIXED_KWARGS}


def test_auto_is_auto_when_ocr_is_available(stubs, tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_OCR", "true")
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    chunk_file(_file(tmp_path, "a.pdf"), ocr="auto")
    assert _pdf_kwargs(stubs)["use_ocr"] == "auto"


@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("ocr", ["always", "never"])
def test_always_and_never_are_forced_whatever_the_host(stubs, tmp_path, monkeypatch, ocr, available):
    if available:
        monkeypatch.setenv("ENABLE_OCR", "true")
        monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    chunk_file(_file(tmp_path, "a.pdf"), ocr=ocr)
    assert _pdf_kwargs(stubs)["use_ocr"] == ocr


@pytest.mark.parametrize("ocr", ["auto", "always", "never"])
def test_languages_are_passed_to_the_pdf_parser_as_given(stubs, tmp_path, ocr):
    chunk_file(_file(tmp_path, "a.pdf"), ocr=ocr, ocr_languages="eng+fra+deu")
    assert _pdf_kwargs(stubs)["ocr_language"] == "eng+fra+deu"


def test_languages_are_not_read_from_the_environment(stubs, tmp_path, monkeypatch):
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "fra.traineddata").write_bytes(b"")
    (tessdata / "deu.traineddata").write_bytes(b"")
    monkeypatch.setenv("ENABLE_OCR", "true")
    monkeypatch.setenv("TESSDATA_PREFIX", str(tessdata))
    monkeypatch.setenv("RAG_ENGINE_OCR_LANGUAGES", "spa")
    chunk_file(_file(tmp_path, "a.pdf"))
    assert _pdf_kwargs(stubs) == {"use_ocr": "auto", "ocr_language": "eng", **PDF_FIXED_KWARGS}


def test_only_the_pdf_parser_receives_ocr_arguments(stubs, tmp_path):
    for name in ("a.docx", "a.xlsx", "a.csv", "a.md", "a.txt", "a.pptx"):
        chunk_file(_file(tmp_path, name), ocr="always", ocr_languages="fra")
    assert all(kwargs == {} for _, kwargs in _inits(stubs))


# -- ocr_available ------------------------------------------------------------


@pytest.mark.parametrize("value", ["true", "TRUE", "True"])
def test_ocr_is_available_with_the_switch_on_and_tessdata_set(monkeypatch, tmp_path, value):
    monkeypatch.setenv("ENABLE_OCR", value)
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    assert ocr_available() is True


@pytest.mark.parametrize("value", [None, "", "false", "1", "yes", "on", " true"])
def test_ocr_is_not_available_unless_the_switch_is_exactly_true(monkeypatch, tmp_path, value):
    if value is None:
        monkeypatch.delenv("ENABLE_OCR", raising=False)
    else:
        monkeypatch.setenv("ENABLE_OCR", value)
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    assert ocr_available() is False


@pytest.mark.parametrize("prefix", [None, ""])
def test_ocr_is_not_available_without_tessdata_prefix(monkeypatch, prefix):
    monkeypatch.setenv("ENABLE_OCR", "true")
    if prefix is None:
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    else:
        monkeypatch.setenv("TESSDATA_PREFIX", prefix)
    assert ocr_available() is False


def test_the_switch_is_read_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    monkeypatch.setenv("ENABLE_OCR", "false")
    assert ocr_available() is False
    monkeypatch.setenv("ENABLE_OCR", "true")
    assert ocr_available() is True


@pytest.mark.parametrize("switch", [None, "false", "true"])
def test_ocr_available_never_changes_the_environment(monkeypatch, tmp_path, switch):
    (tmp_path / "tessdata").mkdir()
    if switch is None:
        monkeypatch.delenv("ENABLE_OCR", raising=False)
    else:
        monkeypatch.setenv("ENABLE_OCR", switch)
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    before = dict(os.environ)
    assert ocr_available() is (switch == "true")
    assert dict(os.environ) == before


# -- TESSDATA_PREFIX correction -----------------------------------------------

# For each OCR mode and each state of the ENABLE_OCR switch: the mode PdfParser receives, and
# whether a TESSDATA_PREFIX set to the parent of a ``tessdata`` folder is corrected.
PREFIX_CASES = [
    ("auto", None, "never", False),
    ("auto", "false", "never", False),
    ("auto", "true", "auto", True),
    ("always", None, "always", True),
    ("always", "false", "always", True),
    ("always", "true", "always", True),
    ("never", None, "never", False),
    ("never", "false", "never", False),
    ("never", "true", "never", False),
]


@pytest.mark.parametrize(("ocr", "switch", "use_ocr", "corrected"), PREFIX_CASES)
def test_tessdata_prefix_is_corrected_only_when_ocr_may_run(
    stubs, tmp_path, monkeypatch, ocr, switch, use_ocr, corrected
):
    parent = tmp_path / "share"
    (parent / "tessdata").mkdir(parents=True)
    if switch is not None:
        monkeypatch.setenv("ENABLE_OCR", switch)
    monkeypatch.setenv("TESSDATA_PREFIX", str(parent))
    before = dict(os.environ)
    chunk_file(_file(tmp_path, "a.pdf"), ocr=ocr)
    assert _pdf_kwargs(stubs)["use_ocr"] == use_ocr
    if corrected:
        assert os.environ["TESSDATA_PREFIX"] == str(parent / "tessdata")
        # Idempotent: a second file does not descend further.
        chunk_file(_file(tmp_path, "b.pdf"), ocr=ocr)
        assert os.environ["TESSDATA_PREFIX"] == str(parent / "tessdata")
    else:
        assert dict(os.environ) == before


@pytest.mark.parametrize("ocr", ["auto", "always"])
def test_tessdata_prefix_already_on_the_language_data_is_left_alone(stubs, tmp_path, monkeypatch, ocr):
    monkeypatch.setenv("ENABLE_OCR", "true")
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    before = dict(os.environ)
    chunk_file(_file(tmp_path, "a.pdf"), ocr=ocr)
    assert dict(os.environ) == before


@pytest.mark.parametrize("switch", [None, "true"])
def test_always_without_tessdata_prefix_sets_nothing(stubs, tmp_path, monkeypatch, switch):
    if switch is not None:
        monkeypatch.setenv("ENABLE_OCR", switch)
    chunk_file(_file(tmp_path, "a.pdf"), ocr="always")
    assert "TESSDATA_PREFIX" not in os.environ
    assert _pdf_kwargs(stubs)["use_ocr"] == "always"


@pytest.mark.parametrize("name", ["a.docx", "a.xlsx", "a.csv", "a.md", "a.txt", "a.pptx"])
def test_tessdata_prefix_is_never_touched_for_a_format_without_ocr(stubs, tmp_path, monkeypatch, name):
    (tmp_path / "tessdata").mkdir()
    monkeypatch.setenv("ENABLE_OCR", "true")
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    chunk_file(_file(tmp_path, name), ocr="always")
    assert os.environ["TESSDATA_PREFIX"] == str(tmp_path)


# -- chunks: order, blanks, pages ---------------------------------------------


def test_chunks_keep_document_order_and_blank_ones_are_dropped(stubs, tmp_path):
    stubs.chunks = [
        StubChunk("first", 0),
        StubChunk("", 0),
        StubChunk("   \n\t ", 1),
        StubChunk("second", 1),
        StubChunk("\u00a0", 2),
        StubChunk("third", 4),
    ]
    assert chunk_file(_file(tmp_path, "a.pdf")) == [Chunk("first", 0), Chunk("second", 1), Chunk("third", 4)]


def test_text_is_kept_as_the_chunker_gave_it(stubs, tmp_path):
    stubs.chunks = [StubChunk("  # Title\n\nbody \n", 0)]
    assert chunk_file(_file(tmp_path, "a.pdf")) == [Chunk("  # Title\n\nbody \n", 0)]


def test_duplicate_chunk_texts_are_all_kept(stubs, tmp_path):
    stubs.chunks = [StubChunk("same", 0), StubChunk("same", 0), StubChunk("same", 1)]
    assert chunk_file(_file(tmp_path, "a.pdf")) == [Chunk("same", 0), Chunk("same", 0), Chunk("same", 1)]


def test_page_is_none_when_the_chunker_gives_none_or_no_attribute(stubs, tmp_path):
    stubs.chunks = [StubChunk("a", None), PagelessChunk("b"), StubChunk("c", 0), StubChunk("d", 7)]
    assert chunk_file(_file(tmp_path, "a.md")) == [
        Chunk("a", None),
        Chunk("b", None),
        Chunk("c", 0),
        Chunk("d", 7),
    ]


@pytest.mark.parametrize("chunks", [[], [StubChunk("", 0), StubChunk(" \n", 1)]])
def test_no_chunk_or_only_blank_chunks_gives_an_empty_list(stubs, tmp_path, chunks):
    stubs.chunks = chunks
    assert chunk_file(_file(tmp_path, "a.pdf")) == []


# -- errors -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a.pdf", "a.docx", "a.md", "a.pptx"])
def test_no_extractable_text_gives_an_empty_list(stubs, tmp_path, name):
    stubs.parse_error = StubTextNotFound("No text content found in document.")
    assert chunk_file(_file(tmp_path, name)) == []


def test_text_not_found_raised_by_the_chunker_gives_an_empty_list(stubs, tmp_path):
    stubs.chunk_error = StubTextNotFound("nothing")
    assert chunk_file(_file(tmp_path, "a.pdf")) == []


@pytest.mark.parametrize("name", ["a.pdf", "a.csv", "a.txt"])
def test_any_other_parser_error_propagates_unchanged(stubs, tmp_path, name):
    error = RuntimeError("broken file")
    stubs.parse_error = error
    with pytest.raises(RuntimeError) as raised:
        chunk_file(_file(tmp_path, name))
    assert raised.value is error


def test_chunker_error_propagates_unchanged(stubs, tmp_path):
    error = KeyError("boom")
    stubs.chunk_error = error
    with pytest.raises(KeyError) as raised:
        chunk_file(_file(tmp_path, "a.docx"))
    assert raised.value is error


def test_missing_text_file_raises_file_not_found(stubs, tmp_path):
    with pytest.raises(FileNotFoundError):
        chunk_file(tmp_path / "absent.txt")


# -- text encoding ------------------------------------------------------------


@pytest.mark.parametrize("name", ["latin.txt", "latin.md"])
def test_invalid_utf8_is_replaced_with_a_warning(stubs, tmp_path, caplog, name):
    path = _file(tmp_path, name, "résumé de l'année".encode("latin-1"))
    with caplog.at_level(logging.WARNING, logger="rag_engine.chunking"):
        chunk_file(path)
    (parsed,) = [arg for n, what, arg in stubs.calls if what == "parse_string"]
    assert parsed == "r\ufffdsum\ufffd de l'ann\ufffde"
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert str(path) in record.getMessage()
    assert "UTF-8" in record.getMessage()


def test_the_invalid_utf8_warning_gives_no_count(stubs, tmp_path, caplog):
    """U+FFFD characters already in the file are not invalid bytes: nothing is counted."""
    data = "d\ufffdj\ufffd \ufffd ".encode() + "caf\xe9".encode("latin-1")
    path = _file(tmp_path, "mixed.txt", data)
    with caplog.at_level(logging.WARNING, logger="rag_engine.chunking"):
        chunk_file(path)
    assert ("MarkdownParser", "parse_string", "d\ufffdj\ufffd \ufffd caf\ufffd") in stubs.calls
    (record,) = caplog.records
    assert record.getMessage() == f"{path} is not valid UTF-8: its invalid bytes were replaced by U+FFFD"


def test_valid_utf8_logs_nothing(stubs, tmp_path, caplog):
    path = _file(tmp_path, "ok.txt", "résumé de l'année, 日本語".encode())
    with caplog.at_level(logging.WARNING, logger="rag_engine.chunking"):
        chunk_file(path)
    assert caplog.records == []


def test_a_replacement_character_already_in_a_valid_file_is_not_reported(stubs, tmp_path, caplog):
    path = _file(tmp_path, "ok.txt", "already \ufffd there".encode())
    with caplog.at_level(logging.WARNING, logger="rag_engine.chunking"):
        chunk_file(path)
    assert ("MarkdownParser", "parse_string", "already \ufffd there") in stubs.calls
    assert caplog.records == []


def test_text_is_read_as_the_old_engine_read_it(stubs, tmp_path):
    """Universal newlines, and a byte order mark is kept: plain ``utf-8``, as before."""
    path = _file(tmp_path, "win.txt", b"\xef\xbb\xbfline one\r\nline two\r\n")
    chunk_file(path)
    assert ("MarkdownParser", "parse_string", "\ufeffline one\nline two\n") in stubs.calls


# -- lazy import and the PdfParser patch --------------------------------------


def test_importing_the_module_loads_no_parser_library():
    code = (
        "import sys, rag_engine.chunking; "
        "print(','.join(m for m in sys.modules if m.split('.')[0] in ('chunknorris', 'markitdown', 'pymupdf')))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


@pytest.mark.parametrize("name", ["a.pdf", "a.md", "a.txt", "a.csv"])
def test_missing_chunknorris_tells_to_install_the_ingest_extra(monkeypatch, tmp_path, name):
    monkeypatch.setitem(sys.modules, "chunknorris", None)
    with pytest.raises(ImportError, match=r"rag-engine\[ingest\]") as error:
        chunk_file(_file(tmp_path, name))
    assert isinstance(error.value.__cause__, ImportError)
    # The module that could not be imported is named, not hidden behind the hint.
    assert "chunknorris" in str(error.value)


def test_missing_markitdown_tells_to_install_the_ingest_extra(stubs, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "markitdown", None)
    with pytest.raises(ImportError, match=r"rag-engine\[ingest\]"):
        chunk_file(_file(tmp_path, "deck.pptx"))


def test_markitdown_is_not_needed_for_other_formats(stubs, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "markitdown", None)
    assert chunk_file(_file(tmp_path, "a.pdf")) == [Chunk("some text", 0)]


class _Box:
    def __init__(self, y0, y1):
        self.y0, self.y1 = y0, y1


class _Line:
    def __init__(self, y0, y1):
        self.bbox = _Box(y0, y1)


def test_the_pdf_parser_line_spacing_is_patched_on_first_use(stubs, tmp_path):
    assert stubs.pdf_parser._get_line_spacing([]) == "original"
    chunk_file(_file(tmp_path, "a.pdf"))
    patched = stubs.pdf_parser._get_line_spacing
    # Fewer than two lines, nothing to count.
    assert patched([]) == 0.0
    assert patched([_Line(0, 10)]) == 0.0
    # Otherwise the most common gap between consecutive lines.
    lines = [_Line(0, 10), _Line(12, 22), _Line(24, 34), _Line(40, 50)]
    assert patched(lines) == 2.0
    # Still a static method: callable from an instance too.
    assert stubs.pdf_parser()._get_line_spacing([]) == 0.0


def test_the_patch_is_applied_once_and_for_non_pdf_formats_too(stubs, tmp_path):
    chunk_file(_file(tmp_path, "a.md"))
    first = stubs.pdf_parser.__dict__["_get_line_spacing"]
    chunk_file(_file(tmp_path, "a.pdf"))
    assert stubs.pdf_parser.__dict__["_get_line_spacing"] is first


def test_a_pdf_parser_without_the_private_method_gives_a_warning_not_a_crash(monkeypatch, tmp_path, caplog):
    stubs = _install_stubs(monkeypatch, with_line_spacing=False)
    monkeypatch.setattr(chunking, "_patch_warning_logged", False)
    with caplog.at_level(logging.WARNING, logger="rag_engine.chunking"):
        assert chunk_file(_file(tmp_path, "a.pdf")) == [Chunk("some text", 0)]
        chunk_file(_file(tmp_path, "b.pdf"))
    assert not hasattr(stubs.pdf_parser, "_get_line_spacing")
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "_get_line_spacing" in record.getMessage()


# ---------------------------------------------------------------------------
# Layer 2: the real libraries (extra ``ingest``)
# ---------------------------------------------------------------------------

needs_ingest = pytest.mark.skipif(
    importlib.util.find_spec("chunknorris") is None, reason="needs the ingest extra (chunknorris)"
)

MARKDOWN = """# Monetary policy

The committee kept the policy rate unchanged at its last meeting.

## Inflation

Inflation eased over the quarter, mostly because energy prices fell.

## Outlook

Growth is expected to stay moderate next year.
"""


@pytest.fixture
def real(monkeypatch):
    """The real libraries, with OCR switched off on the host unless a test turns it on."""
    pytest.importorskip("chunknorris")
    monkeypatch.delenv("ENABLE_OCR", raising=False)


def _pdf(path, pages):
    """A PDF with one page per entry; an entry is the page's lines, ``[]`` for a page with no text."""
    import pymupdf

    document = pymupdf.open()
    for lines in pages:
        page = document.new_page()
        for index, line in enumerate(lines):
            page.insert_text((72, 80 + 14 * index), line, fontsize=10)
    document.save(str(path))
    document.close()
    return path


WORDS = (
    "rate", "inflation", "growth", "committee", "reserves", "liquidity", "outlook", "employment", "credit",
    "energy", "prices", "wages", "demand", "supply", "balance", "sheet", "policy", "stance", "guidance", "risk",
)  # fmt: skip


def _page_lines(number, count=45):
    """Lines of varied text, each starting with a ``P<page>L<line>`` marker.

    Varied on purpose: ChunkNorris takes lines that repeat from page to page for headers and
    footers, and a document made only of those has "no text". 45 lines is about 400 words, more
    than one chunk holds, so every page gets a chunk of its own.
    """
    rng = random.Random(number)
    return [
        f"P{number}L{i} " + " ".join(rng.choice(WORDS) for _ in range(8)) + "." for i in range(1, count + 1)
    ]


@needs_ingest
def test_real_markdown_with_headings(real, tmp_path):
    path = tmp_path / "note.md"
    path.write_text(MARKDOWN, encoding="utf-8")
    chunks = chunk_file(path)
    assert chunks
    assert all(isinstance(c, Chunk) and c.text.strip() for c in chunks)
    assert all(c.page is None for c in chunks)
    text = "\n".join(c.text for c in chunks)
    for expected in ("Monetary policy", "policy rate unchanged", "energy prices fell", "moderate next year"):
        assert expected in text
    assert text.index("policy rate unchanged") < text.index("energy prices fell") < text.index("moderate")


@needs_ingest
def test_real_plain_text(real, tmp_path):
    path = tmp_path / "note.txt"
    path.write_text(
        "A plain paragraph about reserves, long enough for the chunker to keep it as a chunk.\n\n"
        "A second paragraph about rates, which says that they were left unchanged this quarter.\n",
        encoding="utf-8",
    )
    chunks = chunk_file(path)
    assert chunks
    assert all(c.page is None for c in chunks)
    text = "\n".join(c.text for c in chunks)
    assert "reserves" in text and "rates" in text


@needs_ingest
def test_real_non_utf8_text_ingests_with_a_warning(real, tmp_path, caplog):
    path = tmp_path / "latin.txt"
    path.write_bytes(
        "Le comité a relevé le taux directeur lors de sa réunion, puis a publié ses prévisions de croissance "
        "et d'inflation pour les deux prochaines années.\n".encode("latin-1")
    )
    with caplog.at_level(logging.WARNING, logger="rag_engine.chunking"):
        chunks = chunk_file(path)
    assert chunks
    text = "\n".join(c.text for c in chunks)
    assert "Le comit\ufffd a relev\ufffd le taux directeur" in text
    assert [r for r in caplog.records if r.name == "rag_engine.chunking" and str(path) in r.getMessage()]


@needs_ingest
def test_real_csv(real, tmp_path):
    path = tmp_path / "rates.csv"
    path.write_text("bank,rate,year\nECB,4.0,2024\nFED,5.25,2024\nBOE,5.0,2024\n", encoding="utf-8")
    chunks = chunk_file(path)
    assert chunks
    assert all(c.page is None for c in chunks)
    text = "\n".join(c.text for c in chunks)
    for expected in ("ECB", "FED", "BOE", "5.25"):
        assert expected in text


@needs_ingest
@pytest.mark.parametrize("name", ["empty.txt", "empty.md", "blank.txt"])
def test_real_empty_text_file_gives_no_chunk(real, tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b" \n\n\t\n" if name.startswith("blank") else b"")
    assert chunk_file(path) == []


@needs_ingest
def test_real_two_page_pdf_has_ordered_page_numbers(real, tmp_path):
    path = _pdf(tmp_path / "report.pdf", [_page_lines(1), _page_lines(2)])
    chunks = chunk_file(path, ocr="never")
    assert chunks
    assert all(c.text.strip() for c in chunks)
    pages = [c.page for c in chunks]
    assert all(isinstance(p, int) and not isinstance(p, bool) for p in pages)
    assert pages == sorted(pages)
    # ChunkNorris numbers pages from 0; chunk_file passes that through unchanged.
    assert set(pages) == {0, 1}
    by_page = {p: "\n".join(c.text for c in chunks if c.page == p) for p in set(pages)}
    assert "P1L1 " in by_page[0] and "P1L45 " in by_page[0] and "P2L" not in by_page[0]
    assert "P2L1 " in by_page[1] and "P2L45 " in by_page[1] and "P1L" not in by_page[1]


@needs_ingest
def test_real_pdf_page_is_the_page_a_chunk_starts_on(real, tmp_path):
    """Short pages share one chunk, which carries its first page; a page without text is skipped."""
    short = _pdf(tmp_path / "short.pdf", [_page_lines(1, 8), _page_lines(2, 8), _page_lines(3, 8)])
    (chunk,) = chunk_file(short, ocr="never")
    assert chunk.page == 0
    assert "P1L1 " in chunk.text and "P3L8 " in chunk.text
    gap = _pdf(tmp_path / "gap.pdf", [_page_lines(1), [], _page_lines(3)])
    assert [c.page for c in chunk_file(gap, ocr="never")] == [0, 2]


@needs_ingest
def test_real_pdf_in_auto_mode_without_ocr_on_the_host(real, tmp_path):
    path = _pdf(tmp_path / "report.pdf", [_page_lines(1), _page_lines(2)])
    assert chunk_file(path) == chunk_file(path, ocr="never")


@needs_ingest
def test_real_pdf_with_a_single_line_does_not_crash(real, tmp_path):
    """One line, no gap to measure: ChunkNorris 1.3.7's own line spacing fails on it."""
    line = "Only one line of text on this page, long enough to pass the fifteen word minimum of the chunker."
    path = _pdf(tmp_path / "one-line.pdf", [[line]])
    assert chunk_file(path, ocr="never") == [Chunk(line, 0)]


@needs_ingest
@pytest.mark.parametrize("name", ["short.md", "short.txt", "short.pdf"])
def test_real_document_under_the_chunker_minimum_gives_no_chunk(real, tmp_path, name):
    """ChunkNorris's default drops chunks of fewer than 15 words; the old engine did the same."""
    path = tmp_path / name
    if name.endswith(".pdf"):
        _pdf(path, [["Too short to keep."]])
    else:
        path.write_text("Too short to keep.\n", encoding="utf-8")
    assert chunk_file(path, ocr="never") == []


@needs_ingest
@pytest.mark.parametrize("pages", [[[]], [[], []]])
def test_real_pdf_without_text_and_ocr_off_gives_no_chunk(real, tmp_path, pages):
    path = _pdf(tmp_path / "scan.pdf", pages)
    assert chunk_file(path, ocr="never") == []
    assert chunk_file(path) == []


@needs_ingest
@pytest.mark.parametrize("data", [b"", b"this is not a pdf"])
def test_real_empty_or_corrupt_pdf_raises_the_parser_error(real, tmp_path, data):
    """Zero bytes is a broken PDF, not a document without text: the parser's error comes out."""
    import pymupdf

    path = tmp_path / "broken.pdf"
    path.write_bytes(data)
    with pytest.raises((pymupdf.EmptyFileError, pymupdf.FileDataError)):
        chunk_file(path, ocr="never")


@needs_ingest
def test_real_empty_csv_raises_the_parser_error(real, tmp_path):
    """The parser's own ValueError, which is not the one for an unsupported extension."""
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="delimiter") as error:
        chunk_file(path)
    assert not isinstance(error.value, UnsupportedFormatError)
    html = tmp_path / "page.html"
    html.write_text("<p>text</p>", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        chunk_file(html)


# -- the line-spacing pin -----------------------------------------------------


def _pristine_pdf_parser():
    """A second, untouched copy of ChunkNorris's ``PdfParser`` class.

    ``chunk_file`` replaces ``_get_line_spacing`` on the class everybody imports, for the whole
    process. To compare with the library's own method, its module is executed again under
    another name (and not registered): that copy's class has the original method, and its code
    looks the method up on that copy, so the engine's replacement never reaches it.
    """
    import chunknorris.parsers.pdf.pdf_parser as live

    spec = importlib.util.spec_from_file_location(
        "chunknorris.parsers.pdf._pristine_pdf_parser", live.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PdfParser


def _list_then_paragraphs_pdf(path, items=60, paragraphs=40):
    """Large-font lines, one per item and widely spaced, then two-line body paragraphs.

    The body font (10 pt) has the most lines, so ChunkNorris's own line spacing only looks at
    body lines and finds the gap inside a paragraph. The engine's line spacing looks at every
    line and finds the gap between two items, the most frequent of all, which is wider than the
    gap between two paragraphs: paragraphs then merge into one block.
    """
    import pymupdf

    rng = random.Random(7)
    document = pymupdf.open()
    state = {"page": document.new_page(), "y": 80}

    def put(text, fontsize, step):
        if state["y"] > 760:
            state["page"], state["y"] = document.new_page(), 80
        state["page"].insert_text((72, state["y"]), text, fontsize=fontsize)
        state["y"] += step

    for i in range(items):
        put(f"Item {i} " + " ".join(rng.choice(WORDS) for _ in range(4)), 14, 30)
    for i in range(paragraphs):
        put(f"B{i}a " + " ".join(rng.choice(WORDS) for _ in range(9)), 10, 12)
        put(f"B{i}b " + " ".join(rng.choice(WORDS) for _ in range(9)) + ".", 10, 20)
    document.save(str(path))
    document.close()
    return path


def _library_pdf_chunks(pdf_parser_class, path):
    from chunknorris.chunkers import MarkdownChunker

    document = pdf_parser_class(use_ocr="never", ocr_language="eng").parse_file(str(path))
    return [Chunk(c.get_text(), c.start_page) for c in MarkdownChunker().chunk(document)]


@needs_ingest
def test_real_pdf_chunks_depend_on_the_line_spacing_pin(real, tmp_path):
    """Fails if ``chunk_file`` stops installing its line spacing on ``PdfParser``.

    The replacement is what production ran, and it does not give the same chunks as the
    library's own method: this is a document on which they differ.
    """
    path = _list_then_paragraphs_pdf(tmp_path / "list.pdf")
    chunks = chunk_file(path, ocr="never")
    assert len(chunks) >= 3

    library = _library_pdf_chunks(_pristine_pdf_parser(), path)
    pinned_class = _pristine_pdf_parser()
    pinned_class._get_line_spacing = staticmethod(chunking._safe_line_spacing)
    pinned = _library_pdf_chunks(pinned_class, path)

    assert [c.text for c in pinned] != [c.text for c in library], "the fixture no longer separates the two"
    assert chunks == pinned
    assert chunks != library


@needs_ingest
def test_real_pdf_parser_carries_the_engine_line_spacing_after_a_pdf(real, tmp_path):
    from chunknorris.parsers import PdfParser

    chunk_file(_pdf(tmp_path / "report.pdf", [_page_lines(1)]), ocr="never")
    assert PdfParser._get_line_spacing is chunking._safe_line_spacing
    assert isinstance(PdfParser.__dict__["_get_line_spacing"], staticmethod)


# -- explicit options: same result as the library's defaults --------------------


@needs_ingest
def test_real_library_defaults_are_the_values_passed_explicitly(real):
    """A release that changes a default shows up here, before it changes a chunk."""
    from chunknorris.chunkers import MarkdownChunker
    from chunknorris.parsers import PdfParser

    def defaults(callable_):
        return {
            name: p.default for name, p in inspect.signature(callable_).parameters.items() if name != "self"
        }

    chunker = defaults(MarkdownChunker.__init__)
    options = ChunkingOptions()
    assert chunker.pop("tokenizer") is None
    assert chunker == {
        "max_headers_to_use": options.max_headers_to_use,
        "max_chunk_word_count": options.max_chunk_word_count,
        "hard_max_chunk_word_count": options.hard_max_chunk_word_count,
        "min_chunk_word_count": options.min_chunk_word_count,
        "hard_max_chunk_token_count": None,
    }
    parser = defaults(PdfParser.__init__)
    # Decided per call (use_ocr, ocr_language) or left to the library (table_finder, an object).
    for name in ("use_ocr", "ocr_language", "table_finder"):
        parser.pop(name)
    assert parser == PDF_FIXED_KWARGS


@needs_ingest
def test_real_markdown_default_chunks_are_those_of_the_library_defaults(real, tmp_path):
    from chunknorris.chunkers import MarkdownChunker
    from chunknorris.parsers import MarkdownParser

    sections = [f"## Section {i}\n\n" + " ".join(_page_lines(i, 12)) + "\n" for i in range(1, 9)]
    text = "# Report\n\nShort intro.\n\n" + "\n".join(sections) + "\n### Tail\n\nToo short to keep.\n"
    path = tmp_path / "report.md"
    path.write_text(text, encoding="utf-8")

    expected = [
        Chunk(c.get_text(), c.start_page)
        for c in MarkdownChunker().chunk(MarkdownParser().parse_string(text))
    ]
    assert len(expected) >= 3
    assert chunk_file(path) == expected
    assert chunk_file(path, options=ChunkingOptions()) == expected


@needs_ingest
def test_real_pdf_default_chunks_are_those_of_the_library_defaults(real, tmp_path):
    from chunknorris.parsers import PdfParser

    for path in (
        _pdf(tmp_path / "report.pdf", [_page_lines(1), _page_lines(2), _page_lines(3, 8)]),
        _list_then_paragraphs_pdf(tmp_path / "list.pdf"),
    ):
        chunks = chunk_file(path, ocr="never")
        assert len(chunks) >= 3
        # PdfParser has the engine's line spacing by now (chunk_file ran); everything else is
        # the library's defaults.
        assert chunks == _library_pdf_chunks(PdfParser, path)
        assert chunks == chunk_file(path, ocr="never", options=ChunkingOptions())


@needs_ingest
@pytest.mark.parametrize("name", ["short.md", "short.pdf"])
def test_real_lower_minimum_keeps_a_chunk_dropped_by_default(real, tmp_path, name):
    path = tmp_path / name
    if name.endswith(".pdf"):
        _pdf(path, [["Too short to keep."]])
    else:
        path.write_text("Too short to keep.\n", encoding="utf-8")
    assert chunk_file(path, ocr="never") == []
    kept = chunk_file(path, ocr="never", options=ChunkingOptions(min_chunk_word_count=1))
    assert [c.text.strip() for c in kept] == ["Too short to keep."]


@needs_ingest
def test_real_lower_maximum_gives_more_chunks(real, tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("\n\n".join(" ".join(_page_lines(i, 6)) for i in range(1, 9)) + "\n", encoding="utf-8")
    default = chunk_file(path)
    smaller = chunk_file(
        path, options=ChunkingOptions(max_chunk_word_count=60, hard_max_chunk_word_count=120)
    )
    assert len(smaller) > len(default)


@needs_ingest
def test_real_pptx(real, tmp_path):
    pptx = pytest.importorskip("pptx")
    pytest.importorskip("markitdown")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Quarterly review"
    slide.placeholders[
        1
    ].text = (
        "Reserves rose during the quarter, while the policy rate and the balance sheet were left unchanged."
    )
    path = tmp_path / "deck.pptx"
    presentation.save(str(path))
    chunks = chunk_file(path)
    assert chunks
    text = "\n".join(c.text for c in chunks)
    assert "Quarterly review" in text and "Reserves rose" in text


def _tessdata_dir():
    """The folder holding ``eng.traineddata``, or None when tesseract is not usable here."""
    if shutil.which("tesseract") is None:
        return None
    result = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode != 0 or "eng" not in output.split():
        return None
    for line in output.splitlines():
        if '"' in line:
            folder = line.split('"')[1]
            return folder.rstrip("/") or None
    return None


@needs_ingest
@pytest.mark.parametrize("switch", [None, "false", "true"])
@pytest.mark.parametrize("prefix", ["folder", "parent"])
def test_real_scanned_pdf_is_read_with_ocr_always(real, tmp_path, monkeypatch, prefix, switch):
    """``always`` reads a scan whatever ENABLE_OCR says, TESSDATA_PREFIX being the language
    data folder or its parent."""
    tessdata = _tessdata_dir()
    if tessdata is None:
        pytest.skip("needs tesseract with the eng language data")
    if prefix == "parent" and os.path.basename(tessdata) != "tessdata":
        pytest.skip("the language data is not in a folder named tessdata")
    import pymupdf

    source = _pdf(
        tmp_path / "source.pdf",
        [
            [
                "Scanned central bank statement about monetary policy",
                "Interest rates stay unchanged for the whole quarter",
                "The committee will meet again in the spring",
            ]
        ],
    )
    with pymupdf.open(str(source)) as document:
        pixmap = document[0].get_pixmap(dpi=200)
    scan = pymupdf.open()
    page = scan.new_page()
    page.insert_image(page.rect, pixmap=pixmap)
    path = tmp_path / "scan.pdf"
    scan.save(str(path))
    scan.close()

    given = tessdata if prefix == "folder" else os.path.dirname(tessdata)
    monkeypatch.setenv("TESSDATA_PREFIX", given)
    if switch is not None:
        monkeypatch.setenv("ENABLE_OCR", switch)
    assert chunk_file(path, ocr="never") == []
    assert os.environ["TESSDATA_PREFIX"] == given
    chunks = chunk_file(path, ocr="always", ocr_languages="eng")
    assert os.environ["TESSDATA_PREFIX"] == tessdata
    assert chunks
    assert [c.page for c in chunks] == [0] * len(chunks)
    text = " ".join(" ".join(c.text for c in chunks).lower().split())
    assert "central bank" in text and "unchanged" in text
