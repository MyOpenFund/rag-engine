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

## Supported formats and OCR

`chunk_file(path, *, ocr="auto", ocr_languages="eng")` (in `rag_engine.chunking`, extra `ingest`) reads one file and returns its chunks in document order, each with its `text` and its `page`. The format is chosen by the file extension, whatever its case; `SUPPORTED_EXTENSIONS` lists them.

| Extension | Read by | `page` |
|---|---|---|
| `.pdf` | ChunkNorris | page the chunk starts on, counted from 0 |
| `.docx`, `.xlsx`, `.csv` | ChunkNorris | `None` |
| `.pptx` | MarkItDown, then ChunkNorris | `None` |
| `.txt`, `.md` | read as UTF-8, then ChunkNorris | `None` |

Any other extension raises `ValueError`. Every format is turned into markdown and split by the same chunker with its default sizes, which drop chunks of fewer than 15 words. Chunks with no text are left out, and a document with no extractable text (an empty text file, a scanned PDF read without OCR) gives an empty list, not an error. A file the parser cannot open, such as a zero-byte PDF, raises the parser's own error.

A `.txt` or `.md` file that is not valid UTF-8 is still read: each invalid byte sequence becomes `�` (U+FFFD) and a warning is logged.

OCR applies to PDF only, with one of three modes:

| `ocr` | Behaviour |
|---|---|
| `auto` (default) | If OCR is available on the host, ChunkNorris decides page by page whether to run it. If not, no OCR. |
| `always` | OCR on every page, whether or not the host is declared OCR-capable. |
| `never` | No OCR. |

OCR is available on the host when `ENABLE_OCR=true` and `TESSDATA_PREFIX` is set (`ocr_available()`); these are the two variables the old engine used, read here at each call. `TESSDATA_PREFIX` is the folder that holds Tesseract's `*.traineddata` files; if it is set to the parent of a `tessdata` folder, it is corrected for the running process.

The OCR languages are a setting: `RAG_ENGINE_OCR_LANGUAGES`, default `eng`, passed by the caller as `ocr_languages=settings.ocr_languages`. **This differs from the old engine**, which used every language found in the tessdata directory. To keep the old result on a host with several languages installed, list them, for example `RAG_ENGINE_OCR_LANGUAGES=eng+fra`. Each language listed must be installed.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

The tests that parse real documents need the `ingest` extra (`pip install -e ".[dev,ingest]"`) and skip without it; the OCR test also needs Tesseract with the `eng` language data.

Two test markers are deselected by default:

- `integration`: needs a real Qdrant, given by `RAG_ENGINE_IT_QDRANT_URL`.
- `models`: downloads embedding models.

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:v1.20.0
RAG_ENGINE_IT_QDRANT_URL=http://localhost:6333 pytest -m integration -q
```

## Licence

MIT, see [LICENSE](LICENSE). Portions are derived from eigenmind (MIT); see [NOTICE](NOTICE).
