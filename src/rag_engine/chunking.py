"""Chunking: one file in, its text chunks out.

Every supported format is first turned into a markdown document, then split by ChunkNorris's
``MarkdownChunker``, so chunks have the same granularity whatever the source format.

Why the cut must stay stable. The id of a point does not depend on the text of its chunk: it
is derived from the ``doc_id``, the page the chunk starts on and the position of the chunk
(:func:`rag_engine.schema.point_id`). So when a document is ingested again after anything
changed where it is cut (a chunker option, a parser option, the line-spacing measure, a
release of ChunkNorris):

* the points that have the same (page, index) as before are overwritten, with another text
  and another vector;
* the (page, index) pairs of the old cut that the new cut no longer produces are not touched:
  they stay in the collection as stale points, with the old text.

Nothing detects this. A change of chunking therefore requires a clean re-ingestion of the
document set (its old points removed first, or a new collection), not a re-run over it.

The parsing libraries (``chunknorris``, ``markitdown``) belong to the ``ingest`` extra and are
imported inside the functions: importing this module loads neither of them.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_engine.config import valid_ocr_languages
from rag_engine.errors import RagEngineError

if TYPE_CHECKING:
    from chunknorris.chunkers.markdown_chunker import SupportsEncode

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
    "chunk_markdown",
    "load_tokenizer",
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
# not arguments of chunk_file: changing one changes where PDFs are cut, which requires a clean
# re-ingestion of the documents (see the module docstring). ``table_finder`` is the one
# PdfParser argument left to the library: its default is an object (``TableFinder()``), not a
# value that can be written here.
#: Tables are extracted and written as markdown tables.
PDF_EXTRACT_TABLES = True
#: Headers are looked for (table of contents of the document or of its metadata) and marked.
PDF_ADD_HEADERS = True
#: Line spacing of the body text; ``None``: measured per document (see :func:`_safe_line_spacing`).
PDF_BODY_LINE_SPACING = None
#: ChunkNorris's page classifier model is not loaded.
PDF_ENABLE_ML_FEATURES = False

# ``MarkdownChunker`` can also split on a token count, given a tokenizer. The engine counts in
# words only; passed by name for the same reason as above. The token cap of the v2 profile is
# enforced by :func:`_enforce_token_cap` instead: ChunkNorris's own token split cuts a table
# anywhere and does not repeat its header row.
_CHUNKER_HARD_MAX_TOKEN_COUNT = None

_INSTALL_HINT = "document parsing needs the ingest extra: pip install 'myopenfund-rag-engine[ingest]'"

_patch_warning_logged = False


@dataclass(frozen=True)
class Chunk:
    """One piece of a document.

    ``text`` is the chunk as the chunker produced it. ``page`` is the page the chunk starts on,
    as numbered by ChunkNorris (from 0), or ``None`` for a format without pages.

    ``context`` is the path of headings the chunk sits under, ``"<title> › <h1> › <h2> …"``;
    ``None`` unless the chunking profile has ``context_header`` (v2). A chunk is embedded as
    :attr:`embedded_text`, so that every consumer (indexing, token counting) agrees.
    """

    text: str
    page: int | None
    context: str | None = None

    @property
    def embedded_text(self) -> str:
        """The text to embed: ``context + "\\n\\n" + text``, ``text`` alone without a context."""
        if self.context is None:
            return self.text
        return f"{self.context}\n\n{self.text}"


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

    Changing a value changes where documents are cut. The ids do not follow the text (they
    come from ``doc_id``, page and chunk index), so re-ingesting a document with other options
    overwrites the points that keep their (page, index) with another text and vector, and
    leaves the surplus (page, index) pairs of the old cut behind as stale points. A collection
    is to be ingested with one set of options; changing them requires a clean re-ingestion of
    the document set. ``ValueError`` for a value that cannot be used.

    The four fields below are off by default: ``ChunkingOptions()`` is the v1 profile, the one
    production collections were ingested with, and is byte-for-byte what it was. They make the
    v2 profile (:meth:`v2`) and need a tokenizer (see :func:`chunk_markdown`):

    - ``max_chunk_token_count``: hard maximum of tokens of an embedded chunk (its context
      included), ``None`` for no token limit. It is checked with the tokenizer given by the
      caller, so it needs to be the one of the embedding model.
    - ``table_rows_per_chunk``: a markdown table longer than the cap is cut into chunks of this
      many rows, each starting with the header row and the separator line. ``None``: as many
      rows as fit under the cap.
    - ``context_header``: each chunk carries ``context``, its path of headings, and the
      heading lines at the top of its text are dropped (the context says the same).
    - ``overlap_tokens``: when a paragraph alone is longer than the cap and is cut in windows,
      the end of a window is repeated at the start of the next, up to this many tokens.
    - ``version``: a label of the profile, ``"v1"`` or ``"v2"``, for a collection to record
      what it was ingested with. It changes nothing in the cut.
    """

    max_headers_to_use: str = "h4"
    max_chunk_word_count: int = 200
    hard_max_chunk_word_count: int = 400
    min_chunk_word_count: int = 15
    max_chunk_token_count: int | None = None
    table_rows_per_chunk: int | None = None
    context_header: bool = False
    overlap_tokens: int = 0
    version: str = "v1"

    @classmethod
    def v2(cls) -> ChunkingOptions:
        """The v2 profile: chunks that fit the 512-token window of e5-base, with a context.

        - ``max_chunk_token_count=480``: 512 (the window of e5-base) minus 2 special tokens
          minus a margin for the ``"passage: "`` prefix and for differences between token counts.
        - ``table_rows_per_chunk=None``: a long table is split by tokens, the header row and the
          separator line repeated in every piece.
        - ``context_header=True``: the title and headings above a chunk are embedded with it.
        - ``overlap_tokens=32``: overlap between the windows of an over-long paragraph.

        A collection ingested with v1 needs a clean re-ingestion to move to v2.
        """
        return cls(
            max_chunk_token_count=480,
            table_rows_per_chunk=None,
            context_header=True,
            overlap_tokens=32,
            version="v2",
        )

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
        for name, minimum, nullable in (
            ("max_chunk_token_count", 8, True),
            ("table_rows_per_chunk", 1, True),
            ("overlap_tokens", 0, False),
        ):
            value = getattr(self, name)
            if value is None and nullable:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}, got {value!r}")
        if self.max_chunk_token_count is not None and self.overlap_tokens >= self.max_chunk_token_count:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be below max_chunk_token_count "
                f"({self.max_chunk_token_count})"
            )
        if not isinstance(self.context_header, bool):
            raise ValueError(f"context_header must be a bool, got {self.context_header!r}")  # noqa: TRY004
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError(f"version must be a non-empty string, got {self.version!r}")


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
    tokenizer: SupportsEncode | None = None,
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
    the values production documents were chunked with. ``tokenizer`` is needed when
    ``options.max_chunk_token_count`` is set (``ValueError`` without), and ignored otherwise;
    see :func:`chunk_markdown`. The context of a chunk then starts with the file's name, without
    its extension.

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
    _check_tokenizer(options, tokenizer)
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
        raw_chunks = _markdown_chunker(MarkdownChunker, options).chunk(document)
    except TextNotFoundException:
        # ChunkNorris raises instead of returning nothing when a document has no text at all
        # (a PDF without a text layer read with OCR off, for instance). Not an error here.
        return []

    title = os.path.splitext(os.path.basename(filepath))[0]
    return _to_chunks(raw_chunks, options, title, tokenizer)


def chunk_markdown(
    text: str,
    *,
    options: ChunkingOptions | None = None,
    title: str | None = None,
    tokenizer: SupportsEncode | None = None,
) -> list[Chunk]:
    """Chunk a markdown string, in document order; no file is involved.

    ``options`` as in :func:`chunk_file` (``None``: v1). ``tokenizer`` is any object with an
    ``encode(text) -> list`` method (see :func:`load_tokenizer`); it is required when
    ``options.max_chunk_token_count`` is set, and ``ValueError`` is raised without it.

    With a token cap, no chunk has more than ``max_chunk_token_count`` tokens once embedded
    (:attr:`Chunk.embedded_text`, the tokenizer's special tokens included). A chunk under the cap
    is left as ChunkNorris cut it. A longer one is split: a markdown table into pieces that each
    repeat its header row and separator line, other text by paragraphs, a paragraph longer than
    the cap by windows of words with ``overlap_tokens`` of overlap. Pieces keep the page of the
    chunk they come from.

    With ``context_header``, ``title`` is the first element of ``Chunk.context`` (the first
    heading is not repeated when it is the title); without a title the context is the headings.
    """
    if options is None:
        options = ChunkingOptions()
    elif not isinstance(options, ChunkingOptions):
        raise ValueError(f"options must be a ChunkingOptions or None, got {options!r}")
    _check_tokenizer(options, tokenizer)
    try:
        from chunknorris.chunkers import MarkdownChunker
        from chunknorris.exceptions import TextNotFoundException
        from chunknorris.parsers import MarkdownParser
    except ImportError as error:
        raise _missing_dependency(error) from error
    try:
        raw_chunks = _markdown_chunker(MarkdownChunker, options).chunk(MarkdownParser().parse_string(text))
    except TextNotFoundException:
        return []
    return _to_chunks(raw_chunks, options, title, tokenizer)


def _check_tokenizer(options: ChunkingOptions, tokenizer: object) -> None:
    if options.max_chunk_token_count is not None and tokenizer is None:
        raise ValueError(
            f"max_chunk_token_count={options.max_chunk_token_count} needs a tokenizer "
            "(see rag_engine.chunking.load_tokenizer); none was given"
        )


def _markdown_chunker(chunker_class, options: ChunkingOptions):
    return chunker_class(
        max_headers_to_use=options.max_headers_to_use,
        max_chunk_word_count=options.max_chunk_word_count,
        hard_max_chunk_word_count=options.hard_max_chunk_word_count,
        min_chunk_word_count=options.min_chunk_word_count,
        hard_max_chunk_token_count=_CHUNKER_HARD_MAX_TOKEN_COUNT,
    )


def _to_chunks(raw_chunks, options: ChunkingOptions, title: str | None, tokenizer) -> list[Chunk]:
    """ChunkNorris chunks to :class:`Chunk` objects, applying the v2 steps the options ask for."""
    chunks: list[Chunk] = []
    for raw in raw_chunks:
        text = raw.get_text()
        if not text.strip():
            continue
        page = getattr(raw, "start_page", None)
        if not options.context_header and options.max_chunk_token_count is None:
            chunks.append(Chunk(text=text, page=page))
            continue
        headers = [h.text for h in getattr(raw, "headers", None) or []]
        chunks.extend(_v2_chunks(text, page, headers, options, title, tokenizer))
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

    Removing or changing it therefore moves the chunk boundaries of some PDFs. Point ids come
    from ``doc_id``, page and chunk index, not from the text: re-ingesting such a PDF would
    overwrite the points that keep their (page, index) with another text and vector, and leave
    the surplus (page, index) pairs of the old cut behind as stale points. Treat a change here
    as a change of chunking: it needs a comparison on real documents and a clean re-ingestion
    of the document set.

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


# ---------------------------------------------------------------------------
# v2: context header and token cap
# ---------------------------------------------------------------------------

_CONTEXT_SEPARATOR = " › "
#: A context may take at most this share of the token cap; the heading levels between the title
#: and the last heading are dropped until it fits.
_CONTEXT_MAX_SHARE = 4
_HEADING_MARK = re.compile(r"^\s*#+\s*")


def _context_path(headers: Sequence[str], title: str | None, limit: int | None, count) -> str | None:
    """``"<title> › <h1> › <h2> …"`` for the headings above a chunk, ``None`` when there are none.

    A first heading equal to the title is not repeated. With ``limit`` (tokens), the middle
    levels are dropped, then the last label is cut, until the path fits.
    """
    parts = [title.strip()] if title and title.strip() else []
    for header in headers:
        label = _HEADING_MARK.sub("", header).strip()
        if label and not (not parts[1:] and parts and label == parts[0]):
            parts.append(label)
    if not parts:
        return None
    if limit is not None:
        while len(parts) > 2 and count(_CONTEXT_SEPARATOR.join(parts)) > limit:
            del parts[1]
        words = parts[-1].split()
        while len(words) > 1 and count(_CONTEXT_SEPARATOR.join([*parts[:-1], " ".join(words)])) > limit:
            words.pop()
        parts[-1] = " ".join(words)
    return _CONTEXT_SEPARATOR.join(parts)


def _v2_chunks(
    text: str,
    page: int | None,
    headers: list[str],
    options: ChunkingOptions,
    title: str | None,
    tokenizer,
) -> list[Chunk]:
    """The chunks of one ChunkNorris chunk under the v2 options (context and/or token cap)."""
    lead = "\n\n".join(headers)
    if headers and text.startswith(lead):
        body = text[len(lead) :].lstrip("\n")
    else:
        lead, body = "", text
    cap = options.max_chunk_token_count

    def count(value: str) -> int:
        return len(tokenizer.encode(value))

    context = None
    prefix = ""
    if options.context_header:
        limit = None if cap is None else cap // _CONTEXT_MAX_SHARE
        context = _context_path(headers, title, limit, count)
        if not body.strip():
            return []
    elif lead:
        prefix = lead + "\n\n"  # no context: the heading lines stay in the text, on every piece

    def make(piece: str) -> Chunk:
        return Chunk(text=prefix + piece, page=page, context=context)

    if cap is None:
        return [make(body)]

    def fits(piece: str) -> bool:
        return count(make(piece).embedded_text) <= cap

    if fits(body):
        return [make(body)]
    pieces: list[str] = []
    for kind, content in _blocks(body):
        if kind == "table":
            pieces.extend(_split_table(content[0], content[1], fits, options))
        else:
            pieces.extend(_split_prose(content, fits, count, options.overlap_tokens))
    return [make(piece) for piece in pieces if piece.strip()]


def _is_separator(line: str) -> bool:
    cells = line.strip().replace("|", "").replace(":", "").replace(" ", "")
    return "|" in line and bool(cells) and set(cells) == {"-"}


def _blocks(body: str) -> list[tuple[str, object]]:
    """The body as ``("prose", text)`` and ``("table", (header_lines, row_lines))`` blocks, in order."""
    lines = body.split("\n")
    blocks: list[tuple[str, object]] = []
    prose: list[str] = []

    def flush() -> None:
        if "".join(prose).strip():
            blocks.append(("prose", "\n".join(prose).strip("\n")))
        prose.clear()

    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            run = lines[i:j]
            if len(run) >= 2 and _is_separator(run[1]):
                flush()
                blocks.append(("table", (run[:2], run[2:])))
            else:
                prose.extend(run)
            i = j
        else:
            prose.append(lines[i])
            i += 1
    flush()
    return blocks


def _split_table(header: list[str], rows: list[str], fits, options: ChunkingOptions) -> list[str]:
    """Pieces of a table, each starting with its header row and separator line."""
    head = "\n".join(header)
    pieces: list[str] = []
    group: list[str] = []
    per_chunk = options.table_rows_per_chunk

    def emit() -> None:
        if group:
            pieces.append(head + "\n" + "\n".join(group))
            group.clear()

    for row in rows:
        if not fits(head + "\n" + row):
            # A row that does not fit even alone: cut as text, without the header.
            emit()
            pieces.extend(_windows(row, fits, None, 0))
            continue
        if (per_chunk is not None and len(group) >= per_chunk) or (
            group and not fits(head + "\n" + "\n".join([*group, row]))
        ):
            emit()
        group.append(row)
    emit()
    if not rows:
        pieces.append(head)
    return pieces


def _split_prose(text: str, fits, count, overlap_tokens: int) -> list[str]:
    """Pieces of prose: paragraphs packed up to the cap, an over-long paragraph cut in windows."""
    pieces: list[str] = []
    group: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        if not paragraph.strip():
            continue
        if not fits(paragraph):
            if group:
                pieces.append("\n\n".join(group))
                group = []
            pieces.extend(_windows(paragraph, fits, count, overlap_tokens))
        elif group and not fits("\n\n".join([*group, paragraph])):
            pieces.append("\n\n".join(group))
            group = [paragraph]
        else:
            group.append(paragraph)
    if group:
        pieces.append("\n\n".join(group))
    return pieces


def _windows(text: str, fits, count, overlap_tokens: int) -> list[str]:
    """Cut ``text`` into windows of words that fit, the end of one repeated at the start of the next."""
    specials = count("") if count is not None else 0
    pieces: list[str] = []
    current: list[str] = []

    def overlap_of(words: list[str]) -> list[str]:
        if count is None or overlap_tokens <= 0:
            return []
        taken: list[str] = []
        total = 0
        for word in reversed(words[1:]):  # never the whole window: progress is guaranteed
            total += max(count(word) - specials, 1)
            if total > overlap_tokens:
                break
            taken.append(word)
        return taken[::-1]

    for word in text.split():
        if fits(" ".join([*current, word])):
            current.append(word)
            continue
        if current:
            pieces.append(" ".join(current))
            carried = overlap_of(current)
            current = carried if fits(" ".join([*carried, word])) else []
            if fits(" ".join([*current, word])):
                current.append(word)
                continue
            current = []
        # A word that does not fit alone: cut it by characters.
        rest = word
        while rest and not fits(rest):
            low, high = 1, len(rest)
            while low < high:  # largest prefix that fits (at least one character)
                middle = (low + high + 1) // 2
                if fits(rest[:middle]):
                    low = middle
                else:
                    high = middle - 1
            pieces.append(rest[:low])
            rest = rest[low:]
        current = [rest] if rest else []
    if current:
        pieces.append(" ".join(current))
    return pieces


class _EncodingTokenizer:
    """``tokenizers.Tokenizer`` seen as ChunkNorris's ``SupportsEncode``: ``encode(text) -> list[int]``.

    Truncation and padding are switched off, so a count is the true length of the text; the
    special tokens are included, as when the model embeds it.
    """

    def __init__(self, tokenizer) -> None:
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids


def load_tokenizer(cache_dir: str | os.PathLike[str], model_name: str) -> SupportsEncode:
    """The tokenizer of a pinned embedding model, to count tokens the way the model does.

    ``model_name`` is a model of :data:`rag_engine.embedding.ONNX_MODELS`. Its ``tokenizer.json``
    is looked for in ``cache_dir`` (the model cache), downloaded from the pinned revision when
    it is not there, and checked against the pinned SHA-256 and size before it is read, as
    ``OnnxEmbedder`` does: :class:`rag_engine.embedding.ModelIntegrityError` when it differs.
    Only that file is fetched, not the ONNX weights. ``ValueError`` for a model that is not pinned;
    ``ImportError`` when the ``tokenizers`` package is missing.
    """
    from rag_engine import embedding

    known = embedding._pinned_model(model_name)
    if known is None:
        raise ValueError(
            f"no pinned tokenizer for {model_name!r}: models are {', '.join(sorted(embedding.ONNX_MODELS))}"
        )
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ImportError(
            f"loading a tokenizer needs the tokenizers package (import failed: {error})"
        ) from error
    only = ("tokenizer.json",)
    directory = embedding._pinned_model_dir(model_name, known, os.fspath(cache_dir), only)
    embedding._verify_pinned_files(model_name, known, directory, only)
    return _EncodingTokenizer(Tokenizer.from_file(os.fspath(directory / "tokenizer.json")))
