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

logger = logging.getLogger(__name__)

#: Extensions :func:`chunk_file` reads, lower-case with the dot.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".txt", ".md")

#: Accepted values of the ``ocr`` argument of :func:`chunk_file`.
OCR_MODES: tuple[str, ...] = ("auto", "always", "never")

#: Host switch for OCR, read by :func:`ocr_available`. OCR counts as available only when it is
#: ``true`` (any case).
ENABLE_OCR_ENV = "ENABLE_OCR"
#: Where Tesseract's ``*.traineddata`` files are; PyMuPDF reads this variable itself.
TESSDATA_PREFIX_ENV = "TESSDATA_PREFIX"

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


def ocr_available() -> bool:
    """Whether OCR can be used on this host, which is what ``ocr="auto"`` depends on.

    True when the environment variable ``ENABLE_OCR`` is ``true`` (any case; unset means off)
    and ``TESSDATA_PREFIX`` is set. Both are read at each call. No Tesseract binary is looked
    for: PyMuPDF has its own Tesseract engine and only needs the language data.

    PyMuPDF expects ``TESSDATA_PREFIX`` to be the folder holding the ``*.traineddata`` files.
    Some systems set it to the parent folder; when OCR is available and ``TESSDATA_PREFIX``
    has a ``tessdata`` sub-folder, the variable is changed, for this process, to that folder.
    """
    if os.environ.get(ENABLE_OCR_ENV, "false").lower() != "true":
        return False
    prefix = os.environ.get(TESSDATA_PREFIX_ENV)
    if not prefix:
        return False
    nested = os.path.join(prefix, "tessdata")
    if os.path.isdir(nested):
        os.environ[TESSDATA_PREFIX_ENV] = nested
    return True


def chunk_file(path: str | os.PathLike[str], *, ocr: str = "auto", ocr_languages: str = "eng") -> list[Chunk]:
    """Parse the file at ``path`` and return its chunks, in document order.

    The format is chosen by the file extension, whatever its case; see
    :data:`SUPPORTED_EXTENSIONS`. PDF, DOCX, XLSX and CSV are parsed by ChunkNorris; PPTX is
    converted to markdown by MarkItDown; TXT and MD are read as UTF-8 text, invalid bytes
    being replaced by U+FFFD with a warning.

    ``ocr`` and ``ocr_languages`` only matter for PDF:

    - ``"auto"``: when OCR is available on the host (:func:`ocr_available`), ChunkNorris
      decides page by page whether to run it; when it is not, no OCR.
    - ``"always"``: OCR is run on every page, whatever :func:`ocr_available` says.
    - ``"never"``: no OCR.

    ``ocr_languages`` is the Tesseract language string (``"eng"``, ``"eng+fra"``); it is given
    by the caller (``Settings.ocr_languages``), never read from the environment here.

    Chunks whose text is empty or whitespace are left out. A document with no extractable text
    (an empty file, a scanned PDF read without OCR) gives ``[]``.

    Raises ``ValueError`` for an unknown ``ocr`` mode, a blank ``ocr_languages`` or an
    unsupported extension, and ``ImportError`` when the ``ingest`` extra is not installed.
    Any error of the parsing libraries, other than "no text found", is raised unchanged.
    """
    if not isinstance(ocr, str) or ocr not in OCR_MODES:
        raise ValueError(f"ocr must be one of {', '.join(OCR_MODES)}, got {ocr!r}")
    if not isinstance(ocr_languages, str) or not ocr_languages.strip():
        raise ValueError(f"ocr_languages must be a non-empty string such as 'eng+fra', got {ocr_languages!r}")
    filepath = os.fspath(path)
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
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
            # ocr_available() is called for every mode: it also points TESSDATA_PREFIX at the
            # language data, which a forced "always" needs as much as "auto".
            available = ocr_available()
            use_ocr = ("auto" if available else "never") if ocr == "auto" else ocr
            document = PdfParser(use_ocr=use_ocr, ocr_language=ocr_languages).parse_file(filepath)
        elif ext == ".docx":
            document = DocxParser().parse_file(filepath)
        elif ext == ".xlsx":
            document = ExcelParser().parse_file(filepath)
        else:
            # .csv: the delimiter is detected by the parser.
            document = CSVParser().parse_file(filepath)
        raw_chunks = MarkdownChunker().chunk(document)
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
    """The file as UTF-8 text; invalid bytes become U+FFFD and are reported in the log."""
    try:
        with open(filepath, encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        with open(filepath, encoding="utf-8", errors="replace") as file:
            text = file.read()
    logger.warning(
        "%s is not valid UTF-8: %d invalid byte sequence(s) replaced by U+FFFD", filepath, text.count("\ufffd")
    )
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

    Replaces ``PdfParser._get_line_spacing``, which in some ChunkNorris releases fails with
    ``ValueError: max() arg is an empty sequence`` on a document with fewer than two lines.
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

    The method is private to ChunkNorris. If a release no longer has it, nothing is patched
    and a warning is logged once, instead of failing every ingestion.
    """
    global _patch_warning_logged
    if not hasattr(pdf_parser_class, "_get_line_spacing"):
        if not _patch_warning_logged:
            _patch_warning_logged = True
            logger.warning(
                "chunknorris PdfParser has no _get_line_spacing to patch; PDF parsing runs unpatched. "
                "Check that this chunknorris release handles documents with fewer than two lines."
            )
        return
    if pdf_parser_class.__dict__.get("_get_line_spacing") is not _SAFE_LINE_SPACING:
        pdf_parser_class._get_line_spacing = _SAFE_LINE_SPACING
