# rag-engine

**Status: under construction.** The package scaffold is in place; ingestion and search are being built.

rag-engine does two things:

1. **Ingestion** of one document into [Qdrant](https://qdrant.tech): chunk, embed, write.
2. **Search** over what was ingested.

It is used by two MyOpenFund components: the data-orchestrator (ingestion) and the MCP server (search).

## Install

Python 3.10 or later. The base install is light; each consumer adds only the extras it needs.

| Install | Adds | For |
|---|---|---|
| `rag-engine` | `qdrant-client`, `numpy` | payload schema, point ids, collections, search |
| `rag-engine[onnx]` | `fastembed` | ONNX embedding backend |
| `rag-engine[torch]` | `sentence-transformers` | PyTorch embedding backend |
| `rag-engine[ingest]` | `chunknorris`, `markitdown[pptx]`, `psutil` | document parsing and chunking |
| `rag-engine[dev]` | `pytest`, `ruff` | development |

`import rag_engine` never loads an embedding backend or a document parser.

## Configuration

`Settings.from_env()` (in `rag_engine.config`) reads these environment variables when it is called. Nothing is read at import, and no `.env` file is loaded: a caller that uses one loads it first.

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port (1 to 65535) |
| `QDRANT_API_KEY` | none | Qdrant API key; blank means no authentication |
| `QDRANT_HTTPS` | `false` | Connect over TLS (`1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`) |
| `RAG_ENGINE_MODEL` | `intfloat/multilingual-e5-base` | Embedding model |
| `RAG_ENGINE_DIM` | `768` | Vector dimension of the model |
| `RAG_ENGINE_PASSAGE_PREFIX` | `passage: ` (with the trailing space) | Text put before each chunk when embedding; `none` for no prefix |
| `RAG_ENGINE_QUERY_PREFIX` | `query: ` (with the trailing space) | Text put before each query when embedding; `none` for no prefix |
| `RAG_ENGINE_BACKEND` | `torch` | Embedding backend: `torch` or `onnx` (case-insensitive) |
| `RAG_ENGINE_MODEL_CACHE` | none | Directory for model files; unset leaves the backend default |
| `RAG_ENGINE_BATCH_SIZE` | `100` | Chunks embedded and written per batch (at least 1) |
| `RAG_ENGINE_OCR_LANGUAGES` | `eng` | Tesseract languages joined by `+`, e.g. `eng+fra` |
| `RAG_ENGINE_SEARCH_LIMIT` | `10` | Hits returned when no limit is given |
| `RAG_ENGINE_SEARCH_MAX_LIMIT` | `50` | Largest limit a caller may ask for; not below the default limit |

An unset or blank variable gives the default, and surrounding whitespace is ignored. An invalid value raises `ConfigError` (a `ValueError`) naming the variable and the value as it was set, cut to 80 characters.

The two prefixes follow their own rule:

- unset or empty: the default. A compose file that passes an unset variable through delivers an empty value, and that must not remove the prefix.
- `none` (any case, surrounding whitespace ignored): no prefix.
- any other value: taken verbatim, whitespace included. A whitespace-only value is a prefix made of that whitespace.

> **Warning.** A prefix value that ends with a space must be quoted in `.env` and compose files, for example `RAG_ENGINE_PASSAGE_PREFIX="passage: "`. Unquoted, the trailing space is lost, the prefix becomes `passage:`, and the vectors no longer match the ones already in the collection. The same holds for any change of prefix on an existing collection: re-ingest it.

The API key is never printed: `repr()` and `str()` of a `Settings` show `qdrant_api_key=<set>` or `<unset>`, and no error message echoes it. It stays a plain attribute, and `dataclasses.asdict()` returns it, so do not log that dictionary.

`Settings(...)` can also be built directly with keyword arguments. The same range checks apply, but nothing is parsed or converted: integers must be `int` (not `bool`, `float` or a numeric string), `qdrant_https` must be a `bool`, `backend` must be exactly `"torch"` or `"onnx"`, and a prefix string is the prefix as it is (`""` for no prefix). An empty `qdrant_host` or `model_name` is refused; a blank `qdrant_api_key` or `model_cache` becomes `None`. A wrong type raises `ConfigError`.

## Ingesting a document

`ingest_document` (in `rag_engine.ingest`, extra `ingest`) chunks one file, embeds its chunks and writes one point per chunk. The collection is ensured once per run, by the caller; `ingest_document` never creates or checks it.

```python
from rag_engine.config import Settings
from rag_engine.embedding import make_embedder
from rag_engine.ingest import ingest_document
from rag_engine.store import QdrantStore

settings = Settings.from_env()
store = QdrantStore(settings=settings)
embedder = make_embedder(settings)

store.ensure_collection("central-bank-e5b-v1", embedder.dim)  # once, not per document

result = ingest_document(
    "reports/2025-10-30-statement.pdf",
    "ecb/monetary-policy-statement/2025-10-30",  # doc_id
    {"corpus": "central-bank", "source_code": "ecb", "date": "2025-10-30", "year": 2025},
    collection="central-bank-e5b-v1",
    store=store,
    embedder=embedder,
    ocr_languages=settings.ocr_languages,
)
print(result.chunks, result.pages, result.empty)
```

What it guarantees:

- **Deterministic ids.** The id of a point depends on the `doc_id`, the page the chunk starts on and its position in the document, nothing else. Ingesting the same document again overwrites its points; it never adds duplicates.
- **One ingestion date per document.** Every point of a document carries the same `ingestion_date`, the local time of the ingestion as an ISO string (or the `ingestion_date` argument, for a replay or a test).
- **Nothing written for an empty document.** A file with no extractable text gives `IngestResult(chunks=0, pages=0, empty=True)`; the embedder and the store are not called. It is not an error.
- **Errors propagate.** Nothing is caught: an unsupported format, a parser error, an embedding error or a Qdrant error reaches the caller unchanged, and the caller decides what a failed document means. A store error carries the number of points written before it in `points_written`; a failed write is not undone, and ingesting the document again repairs it.
- **Checked before any work.** A blank `doc_id` or `collection`, an unusable `batch_size`, or metadata that sets a reserved payload key (`doc_id`, `filename`, `chunk_number`, `ingestion_date`, `text`, `page`) raises `ValueError` before the file is read.

`IngestResult.pages` counts the distinct pages chunks start on (0 for a format without pages). The other arguments are `ocr`, `ocr_languages` and `chunking_options` (see below), and `batch_size` (points per write; default `RAG_ENGINE_BATCH_SIZE`).

The points are the ones the previous pipeline wrote: `tests/fixtures/golden/` holds points captured from it for two documents, and the tests require the same ids, payloads and vectors.

## Supported formats and OCR

`chunk_file(path, *, ocr="auto", ocr_languages="eng", options=None)` (in `rag_engine.chunking`, extra `ingest`) reads one file and returns its chunks in document order, each with its `text` and its `page`. The format is chosen by the file extension, whatever its case; `SUPPORTED_EXTENSIONS` lists them.

| Extension | Read by | `page` |
|---|---|---|
| `.pdf` | ChunkNorris | page the chunk starts on, counted from 0 |
| `.docx`, `.xlsx`, `.csv` | ChunkNorris | `None` |
| `.pptx` | MarkItDown, then ChunkNorris | `None` |
| `.txt`, `.md` | read as UTF-8, then ChunkNorris | `None` |

Any other extension raises `UnsupportedFormatError`, a `ValueError`. A file the parser cannot read raises the parser's own error, unchanged: a zero-byte PDF, or an empty `.csv`, whose error is a plain `ValueError` and not an `UnsupportedFormatError`. Chunks with no text are left out, and a document with no extractable text (an empty text file, a scanned PDF read without OCR) gives an empty list, not an error.

Every format is turned into markdown and split by the same chunker. Its sizes, counted in words, are the `options` argument, a `ChunkingOptions`:

| Field | Default | Meaning |
|---|---|---|
| `max_headers_to_use` | `"h4"` | Deepest header level a document is split on. |
| `max_chunk_word_count` | `200` | Soft maximum: a longer chunk is split only where it has deeper headers. |
| `hard_max_chunk_word_count` | `400` | Hard maximum: a longer chunk is always split. |
| `min_chunk_word_count` | `15` | A chunk with fewer words is dropped. |

The defaults are the values the old engine ran with (ChunkNorris's own defaults), and `chunk_file` passes them to ChunkNorris by name, as it does for the PDF parser's options (tables extracted, headers detected; the `PDF_*` constants of the module). The text of a chunk determines the id of its point, so one collection should be ingested with one set of options, and for the same reason:

- the `ingest` extra accepts `chunknorris>=1.3.7,<1.6`: 1.3.7 and 1.5.0 give identical chunks, and a newer minor version must be compared before the bound is raised;
- `chunk_file` replaces ChunkNorris's private line-spacing measure for PDFs with the one the old engine used. The two do not cut every PDF into the same chunks, so this replacement must stay as long as points ingested with it are in use.

A `.txt` or `.md` file that is not valid UTF-8 is still read: each invalid byte sequence becomes `�` (U+FFFD) and a warning is logged.

OCR applies to PDF only, with one of three modes:

| `ocr` | Behaviour |
|---|---|
| `auto` (default) | If OCR is available on the host, ChunkNorris decides page by page whether to run it. If not, no OCR. |
| `always` | OCR on every page, whatever `ENABLE_OCR` says. The language data must still be found through `TESSDATA_PREFIX`. |
| `never` | No OCR. |

OCR is available on the host when `ENABLE_OCR=true` and `TESSDATA_PREFIX` is set (`ocr_available()`); these are the two variables the old engine used, read here at each call. `ocr_available()` only reads the environment. `ENABLE_OCR` only decides what `auto` does: `always` does not look at it.

`TESSDATA_PREFIX` is the folder that holds Tesseract's `*.traineddata` files. If it is set to the parent of a `tessdata` folder, `chunk_file` corrects it for the running process whenever OCR may run on the PDF it reads: with `always`, or with `auto` on a host where OCR is available. It is never changed with `never`, with `auto` on a host without OCR, or for another format.

The OCR languages are a setting: `RAG_ENGINE_OCR_LANGUAGES`, default `eng`, passed by the caller as `ocr_languages=settings.ocr_languages`. Each language listed must be installed.

Differences from the old engine:

- **OCR languages.** The old engine used every language found in the tessdata directory. To keep the old result on a host with several languages installed, list them, for example `RAG_ENGINE_OCR_LANGUAGES=eng+fra`.
- **Invalid UTF-8 in `.txt` and `.md` files.** Invalid bytes are now replaced by `�` (U+FFFD); the old engine dropped them silently. The text of such a file therefore differs slightly from what was ingested before, and so do the chunks that contain those bytes.
- **`ocr="always"` and `TESSDATA_PREFIX`.** The old engine corrected a `TESSDATA_PREFIX` set to the parent folder only when `ENABLE_OCR=true`; it is now also corrected for a forced `always`.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

The tests that parse real documents need the `ingest` extra (`pip install -e ".[dev,ingest]"`) and skip without it; the OCR test also needs Tesseract with the `eng` language data (`apt-get install tesseract-ocr`), which CI installs on the Python 3.12 job.

Two test markers are deselected by default:

- `integration`: needs a real Qdrant, given by `RAG_ENGINE_IT_QDRANT_URL`.
- `models`: downloads embedding models.

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:v1.20.0
RAG_ENGINE_IT_QDRANT_URL=http://localhost:6333 pytest -m integration -q
```

The integration tests of ingestion parse real documents, so they also need the `ingest` extra and skip without it; the CI integration job installs it and fails if the parsers cannot be imported.

## Licence

MIT, see [LICENSE](LICENSE). Portions are derived from eigenmind (MIT); see [NOTICE](NOTICE).
