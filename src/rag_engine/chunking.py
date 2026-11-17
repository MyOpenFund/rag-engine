"""Chunking: one file in, its text chunks out.

Every supported format is first turned into a markdown document, then split by ChunkNorris's
``MarkdownChunker``, so chunks have the same granularity whatever the source format.

The parsing libraries (``chunknorris``, ``markitdown``) belong to the ``ingest`` extra and are
imported inside the functions: importing this module loads neither of them.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass

from rag_engine.config import valid_ocr_languages
from rag_engine.errors import RagEngineError

__all__ = [
    "ENABLE_OCR_ENV",
    "HEADER_LEVELS",
    "OCR_MODES",
    "PDF_ADD_HEADERS",
    "PDF_BODY_LINE_SPACING",
    "PDF_ENABLE_ML_FEATURES",
    "PDF_EXTRACT_TABLES",
    "SUPPORTED_EXTENSIONS",
    "TESSDATA_PREFIX_ENV",
    "Chunk",
    "ChunkingOptions",
    "UnsupportedFormatError",
    "check_ocr_languages",
    "chunk_file",
    "ocr_available",
]

logger = logging.getLogger(__name__)

#: Extensions :func:`chunk_file` reads, lower-case with the dot.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".txt", ".md")

#: Accepted values of the ``ocr`` argument of :func:`chunk_file`.
OCR_MODES: tuple[str, ...] = ("auto", "always", "never")

#: Host switch for OCR, read by :func:`ocr_available`. OCR counts as available only when it is
#: ``true`` (any case). It only decides what ``ocr="auto"`` does.
ENABLE_OCR_ENV = "ENABLE_OCR"
#: Where Tesseract's ``*.traineddata`` files are; PyMuPDF reads this variable itself.
TESSDATA_PREFIX_ENV = "TESSDATA_PREFIX"

#: Accepted values of :attr:`ChunkingOptions.max_headers_to_use`.
HEADER_LEVELS: tuple[str, ...] = ("h1", "h2", "h3", "h4", "h5", "h6")

# What ``PdfParser`` is built with, besides the OCR mode and languages. These are the defaults
# of ChunkNorris 1.3.7 and 1.5.0, which production documents were parsed with; they are passed
# by name so that a release changing a default does not change the chunks. They are constants,
# not arguments of chunk_file: changing one changes the text of PDF chunks, so the ids of their
# points. ``table_finder`` is the one PdfParser argument left to the library: its default is an
# object (``TableFinder()``), not a value that can be written here.
#: Tables are extracted and written as markdown tables.
PDF_EXTRACT_TABLES = True
#: Headers are looked for (table of contents of the document or of its metadata) and marked.
PDF_ADD_HEADERS = True
#: Line spacing of the body text; ``None``: measured per document (see :func:`_safe_line_spacing`).
PDF_BODY_LINE_SPACING = None
#: ChunkNorris's page classifier model is not loaded.
PDF_ENABLE_ML_FEATURES = False

# ``MarkdownChunker`` can also split on a token count, given a tokenizer. The engine counts in
# words only; passed by name for the same reason as above.
_CHUNKER_HARD_MAX_TOKEN_COUNT = None

_INSTALL_HINT = "document parsing needs the ingest extra: pip install 'rag-engine[ingest]'"

_patch_warning_logged = False


@dataclass(frozen=True)
class Chunk:
    """One piece of a document.

    ``text`` is the chunk as the chunker produced it. ``page`` is the page the chunk starts on,
    as numbered by ChunkNorris (from 0), or ``None`` for a format without pages.
    """

    text: str
    page: int | None


@dataclass(frozen=True)
class ChunkingOptions:
    """Sizes the chunker works with; the defaults are the ones production documents were chunked with.

    They are ChunkNorris's own ``MarkdownChunker`` arguments, same names, and its defaults in
    1.3.7 and 1.5.0. Sizes are counted in words.

    - ``max_headers_to_use``: deepest header level a document is split on, ``"h1"`` to ``"h6"``.
    - ``max_chunk_word_count``: soft maximum. A longer chunk is split only where it has deeper
      headers to split on.
    - ``hard_max_chunk_word_count``: hard maximum. A longer chunk is always split.
    - ``min_chunk_word_count``: a chunk with fewer words is dropped.

    Changing a value changes the text of the chunks, so the ids of the points made from them:
    a collection is to be ingested with one set of options. ``ValueError`` for a value that
    cannot be used.
    """

    max_headers_to_use: str = "h4"
    max_chunk_word_count: int = 200
    hard_max_chunk_word_count: int = 400
    min_chunk_word_count: int = 15

    def __post_init__(self) -> None:
        if self.max_headers_to_use not in HEADER_LEVELS:
            raise ValueError(
                f"max_headers_to_use must be one of {', '.join(HEADER_LEVELS)}, "
                f"got {self.max_headers_to_use!r}"
            )
        for name, minimum in (
            ("max_chunk_word_count", 1),
            ("hard_max_chunk_word_count", 1),
            ("min_chunk_word_count", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}, got {value!r}")
        if self.max_chunk_word_count > self.hard_max_chunk_word_count:
            raise ValueError(
                f"max_chunk_word_count ({self.max_chunk_word_count}) must not exceed "
                f"hard_max_chunk_word_count ({self.hard_max_chunk_word_count})"
            )


class UnsupportedFormatError(RagEngineError, ValueError):
    """The file's extension is not one of :data:`SUPPORTED_EXTENSIONS`.

    A ``ValueError``, so code written before this class existed still catches it, and a
    :class:`rag_engine.errors.RagEngineError`. The parsing
    libraries raise plain ``ValueError`` of their own (an empty CSV, for instance): catch this
    class to tell "not a format the engine reads" from "a file the parser could not read".
    """


def check_ocr_languages(ocr_languages: object) -> str:
    """Return ``ocr_languages`` when it is a usable Tesseract language string, else ``ValueError``.

    The rule is :func:`rag_engine.config.valid_ocr_languages`, the one :class:`Settings`
    applies: codes joined by ``+``, no whitespace (``"eng"``, ``"eng+fra"``). :func:`chunk_file`
    and ``ingest_document`` both call this, so a value refused by the settings is refused by
    them too, and the reverse.
    """
    if not valid_ocr_languages(ocr_languages):
        raise ValueError(
            "ocr_languages must be Tesseract language codes joined by '+', without whitespace "
            f"(for example 'eng+fra'), got {ocr_languages!r}"
        )
    return ocr_languages  # type: ignore[return-value]  # a str: valid_ocr_languages checked it


def ocr_available() -> bool:
    """Whether OCR can be used on this host, which is what ``ocr="auto"`` depends on.

    True when the environment variable ``ENABLE_OCR`` is ``true`` (any case; unset means off)
    and ``TESSDATA_PREFIX`` is set. Both are read at each call, and nothing is changed. No
    Tesseract binary is looked for: PyMuPDF has its own Tesseract engine and only needs the
    language data.
    """
    if os.environ.get(ENABLE_OCR_ENV, "false").lower() != "true":
        return False
    return bool(os.environ.get(TESSDATA_PREFIX_ENV))


def _point_tessdata_prefix_at_language_data() -> None:
    """Change ``TESSDATA_PREFIX``, for this process, to its ``tessdata`` sub-folder if it has one.

    PyMuPDF expects ``TESSDATA_PREFIX`` to be the folder holding the ``*.traineddata`` files.
    Some systems set it to the parent folder, and OCR then fails with "eng.traineddata file not
    found". Called by :func:`chunk_file` when OCR may run on a PDF, and only then. Nothing is
    done when the variable is unset or has no ``tessdata`` sub-folder.
    """
    prefix = os.environ.get(TESSDATA_PREFIX_ENV)
    if not prefix:
        return
    nested = os.path.join(prefix, "tessdata")
    if os.path.isdir(nested):
        os.environ[TESSDATA_PREFIX_ENV] = nested


def chunk_file(
    path: str | os.PathLike[str],
    *,
    ocr: str = "auto",
    ocr_languages: str = "eng",
    options: ChunkingOptions | None = None,
) -> list[Chunk]:
    """Parse the file at ``path`` and return its chunks, in document order.

    The format is chosen by the file extension, whatever its case; see
    :data:`SUPPORTED_EXTENSIONS`. PDF, DOCX, XLSX and CSV are parsed by ChunkNorris; PPTX is
    converted to markdown by MarkItDown; TXT and MD are read as UTF-8 text, invalid bytes
    being replaced by U+FFFD with a warning.

    ``ocr`` and ``ocr_languages`` only matter for PDF:

    - ``"auto"``: when OCR is available on the host (:func:`ocr_available`), ChunkNorris
      decides page by page whether to run it; when it is not, no OCR.
    - ``"always"``: OCR is run on every page, whatever :func:`ocr_available` says: the
      ``ENABLE_OCR`` switch is not looked at. The language data must still be found by PyMuPDF,
      through ``TESSDATA_PREFIX``.
    - ``"never"``: no OCR.

    When OCR may run (``"always"``, or ``"auto"`` on a host where OCR is available), a
    ``TESSDATA_PREFIX`` set to the parent of a ``tessdata`` folder is changed to that folder
    for the running process. The environment is never changed otherwise.

    ``ocr_languages`` is the Tesseract language string (``"eng"``, ``"eng+fra"``); it is given
    by the caller (``Settings.ocr_languages``), never read from the environment here. It is
    checked for every format, with the rule of the settings (:func:`check_ocr_languages`).

    ``options`` are the chunker's sizes, for every format; ``None`` means ``ChunkingOptions()``,
    the values production documents were chunked with.

    Chunks whose text is empty or whitespace are left out. A document with no extractable text
    (an empty file, a scanned PDF read without OCR) gives ``[]``.

    Raises :class:`UnsupportedFormatError` (a ``ValueError``) for an unsupported extension,
    plain ``ValueError`` for an unknown ``ocr`` mode, an ``ocr_languages`` that is not a list
    of language codes joined by ``+``, or an
    ``options`` that is not a :class:`ChunkingOptions`, and ``ImportError`` when the ``ingest``
    extra is not installed. Any error of the parsing libraries, other than "no text found", is
    raised unchanged, whatever its type (an empty CSV gives the parser's own ``ValueError``).
    """
    if not isinstance(ocr, str) or ocr not in OCR_MODES:
        raise ValueError(f"ocr must be one of {', '.join(OCR_MODES)}, got {ocr!r}")
    check_ocr_languages(ocr_languages)
    if options is None:
        options = ChunkingOptions()
    elif not isinstance(options, ChunkingOptions):
        raise ValueError(f"options must be a ChunkingOptions or None, got {options!r}")
    filepath = os.fspath(path)
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"format {ext!r} is not supported: {filepath!r} (supported: {', '.join(SUPPORTED_EXTENSIONS)})"
        )

    try:
        from chunknorris.chunkers import MarkdownChunker
        from chunknorris.exceptions import TextNotFoundException
        from chunknorris.parsers import CSVParser, DocxParser, ExcelParser, MarkdownParser, PdfParser
    except ImportError as error:
        raise _missing_dependency(error) from error
    _patch_line_spacing(PdfParser)

    try:
        if ext == ".pptx":
            document = MarkdownParser().parse_string(_pptx_to_markdown(filepath))
        elif ext in (".txt", ".md"):
            # MarkdownParser.parse_file is strict about the encoding; the text is read here.
            document = MarkdownParser().parse_string(_read_text(filepath))
        elif ext == ".pdf":
            use_ocr = ("auto" if ocr_available() else "never") if ocr == "auto" else ocr
            if use_ocr != "never":
                # OCR may run: a forced "always" needs the language data as much as "auto",
                # whatever the ENABLE_OCR switch says.
                _point_tessdata_prefix_at_language_data()
            document = PdfParser(
                use_ocr=use_ocr,
                ocr_language=ocr_languages,
                extract_tables=PDF_EXTRACT_TABLES,
                add_headers=PDF_ADD_HEADERS,
                body_line_spacing=PDF_BODY_LINE_SPACING,
                enable_ml_features=PDF_ENABLE_ML_FEATURES,
            ).parse_file(filepath)
        elif ext == ".docx":
            document = DocxParser().parse_file(filepath)
        elif ext == ".xlsx":
            document = ExcelParser().parse_file(filepath)
        else:
            # .csv: the delimiter is detected by the parser.
            document = CSVParser().parse_file(filepath)
        raw_chunks = MarkdownChunker(
            max_headers_to_use=options.max_headers_to_use,
            max_chunk_word_count=options.max_chunk_word_count,
            hard_max_chunk_word_count=options.hard_max_chunk_word_count,
            min_chunk_word_count=options.min_chunk_word_count,
            hard_max_chunk_token_count=_CHUNKER_HARD_MAX_TOKEN_COUNT,
        ).chunk(document)
    except TextNotFoundException:
        # ChunkNorris raises instead of returning nothing when a document has no text at all
        # (a PDF without a text layer read with OCR off, for instance). Not an error here.
        return []

    chunks: list[Chunk] = []
    for raw in raw_chunks:
        text = raw.get_text()
        if not text.strip():
            continue
        chunks.append(Chunk(text=text, page=getattr(raw, "start_page", None)))
    return chunks


def _missing_dependency(error: ImportError) -> ImportError:
    """The ImportError to raise when a parsing library cannot be imported; names what failed."""
    return ImportError(f"{_INSTALL_HINT} (import failed: {error})", name=error.name, path=error.path)


def _read_text(filepath: str) -> str:
    """The file as UTF-8 text; invalid bytes become U+FFFD and are reported in the log.

    The warning gives no count: the U+FFFD characters of the decoded text may include some that
    were in the file already.
    """
    try:
        with open(filepath, encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        with open(filepath, encoding="utf-8", errors="replace") as file:
            text = file.read()
    logger.warning("%s is not valid UTF-8: its invalid bytes were replaced by U+FFFD", filepath)
    return text


def _pptx_to_markdown(filepath: str) -> str:
    """A .pptx deck as markdown, through MarkItDown (no image captioning)."""
    try:
        from markitdown import MarkItDown
    except ImportError as error:
        raise _missing_dependency(error) from error
    return MarkItDown().convert(filepath).text_content


def _safe_line_spacing(lines) -> float:
    """The most common vertical gap between consecutive lines; 0.0 when there is none to count.

    Replaces ``PdfParser._get_line_spacing``. This is a parity pin, not only a crash guard: the
    value decides where ChunkNorris cuts a page into text blocks, so where PDF chunks begin and
    end.

    ChunkNorris (1.3.7 and 1.5.0) ships another method: it counts only the gaps between lines
    of the body font and skips negative gaps, where this one counts every pair of consecutive
    lines. On most documents both give the same value; on others they do not (a list in a
    larger font above two-line paragraphs, for instance) and the chunk texts differ. Production
    documents were chunked with this function, not with the library's.

    Removing or changing it therefore moves the chunk boundaries of some PDFs. A chunk's text
    determines its point id: re-ingesting such a PDF would then write points under new ids and
    leave the old ones behind as stale points. Treat a change here as a change of chunking: it
    needs a comparison on real documents and a plan for the points already stored.

    It also returns 0.0 on a document with fewer than two lines, where the method of 1.3.7
    fails with ``ValueError: max() arg is an empty sequence``.
    """
    linespace_counts = Counter(
        round(curr_line.bbox.y0 - prev_line.bbox.y1, 1) for curr_line, prev_line in zip(lines[1:], lines[:-1])
    )
    if not linespace_counts:
        return 0.0
    return max(linespace_counts, key=linespace_counts.get)


_SAFE_LINE_SPACING = staticmethod(_safe_line_spacing)


def _patch_line_spacing(pdf_parser_class) -> None:
    """Install :func:`_safe_line_spacing` on ``PdfParser``, once, on first use.

    Needed for PDF chunks to be the ones production has: see :func:`_safe_line_spacing`.

    The method is private to ChunkNorris. If a release no longer has it, nothing is patched
    and a warning is logged once, instead of failing every ingestion; PDF chunk boundaries may
    then differ from those of the documents already ingested.
    """
    global _patch_warning_logged
    if not hasattr(pdf_parser_class, "_get_line_spacing"):
        if not _patch_warning_logged:
            _patch_warning_logged = True
            logger.warning(
                "chunknorris PdfParser has no _get_line_spacing to replace; PDF parsing runs with the "
                "library's own line spacing. PDF chunk boundaries may differ from those of documents "
                "ingested before: compare the chunks before re-ingesting PDFs with this chunknorris release."
            )
        return
    if pdf_parser_class.__dict__.get("_get_line_spacing") is not _SAFE_LINE_SPACING:
        pdf_parser_class._get_line_spacing = _SAFE_LINE_SPACING
