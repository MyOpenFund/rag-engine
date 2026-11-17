# rag-engine

**Status: ingestion and both embedding backends (PyTorch, ONNX) are delivered; search arrives in the next release.**

rag-engine is built to do two things:

1. **Ingestion** of one document into [Qdrant](https://qdrant.tech): chunk, embed, write. Available.
2. **Search** over what was ingested. Not available yet: this release has no search function.

It is used by two MyOpenFund components: the data-orchestrator (ingestion) and the MCP server (search).

## Install

Python 3.10 or later. The base install is light; each consumer adds only the extras it needs.

| Install | Adds | For |
|---|---|---|
| `rag-engine` | `qdrant-client` (1.8 or later), `numpy` | settings, payload schema, point ids, the Qdrant store (collections, writes) |
| `rag-engine[torch]` | `sentence-transformers` | PyTorch embedding backend |
| `rag-engine[onnx]` | `fastembed` (ONNX runtime) | ONNX embedding backend, without PyTorch: for a service that embeds search queries |
| `rag-engine[ingest]` | `chunknorris`, `markitdown[pptx]`, `psutil` | document parsing and chunking |
| `rag-engine[dev]` | `pytest`, `ruff` | development |

`import rag_engine` never loads an embedding backend or a document parser. The package ships a `py.typed` marker: its annotations are usable by type checkers.

## Public API

The package root exposes only `__version__`. Everything else is imported from its submodule; each one lists its public names in `__all__`.

| Module | Main names |
|---|---|
| `rag_engine.config` | `Settings`, `ConfigError`, `valid_ocr_languages` |
| `rag_engine.schema` | `point_id`, `page_for_id`, `build_payload`, `RESERVED_KEYS`, `PAYLOAD_INDEXES` |
| `rag_engine.store` | `QdrantStore`, `StoreError`, `QdrantAuthError`, `CollectionMismatchError` |
| `rag_engine.embedding` | `ModelSpec`, `Embedder`, `make_embedder`, `TorchEmbedder`, `OnnxEmbedder`, `FakeEmbedder`, `BaseEmbedder`, `detect_device`, `cosine_parity`, `ONNX_MODELS`, `OnnxModel`, `ONNX_BATCH_SIZE` |
| `rag_engine.chunking` | `chunk_file`, `Chunk`, `ChunkingOptions`, `UnsupportedFormatError`, `ocr_available`, `check_ocr_languages`, `SUPPORTED_EXTENSIONS`, `OCR_MODES` |
| `rag_engine.ingest` | `ingest_document`, `IngestResult` |
| `rag_engine.errors` | `RagEngineError` |

Errors. Every exception class the engine defines derives from `rag_engine.errors.RagEngineError`: `ConfigError` (also a `ValueError`), `StoreError` with its subclasses `QdrantAuthError` and `CollectionMismatchError`, and `UnsupportedFormatError` (also a `ValueError`). `except RagEngineError` does not catch what the engine lets through unchanged (an error of a parser, of the embedding backend or of the Qdrant client), nor the plain `ValueError` / `TypeError` of a bad argument.

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
| `RAG_ENGINE_BACKEND` | `torch` | Embedding backend, `torch` or `onnx` (case-insensitive); see [Embedding backends](#embedding-backends) |
| `RAG_ENGINE_MODEL_CACHE` | none | Directory for model files. Unset leaves the backend default: the Hugging Face cache for `torch`; `FASTEMBED_CACHE_PATH`, or a folder under the system's temporary directory, for `onnx` |
| `RAG_ENGINE_PARITY_MIN` | `0.999` | Smallest cosine accepted between the `torch` and `onnx` vectors of the same text (above 0, at most 1). Read by the parity test only |
| `RAG_ENGINE_BATCH_SIZE` | `100` | Points sent to Qdrant per write (at least 1). It sizes the writes only: all the chunks of a document are embedded in one call |
| `RAG_ENGINE_OCR_LANGUAGES` | `eng` | Tesseract languages joined by `+`, without whitespace, e.g. `eng+fra`. The caller passes it to `ingest_document` (see below) |
| `RAG_ENGINE_SEARCH_LIMIT` | `10` | Hits returned when no limit is given. Read and checked, but not used yet: search is not available |
| `RAG_ENGINE_SEARCH_MAX_LIMIT` | `50` | Largest limit a caller may ask for; not below the default limit. Not used yet either |

An unset or blank variable gives the default, and surrounding whitespace is ignored. An invalid value raises `ConfigError` (a `ValueError`) naming the variable and the value as it was set, cut to 80 characters.

The two prefixes follow their own rule:

- unset or empty: the default. A compose file that passes an unset variable through delivers an empty value, and that must not remove the prefix.
- `none` (any case, surrounding whitespace ignored): no prefix.
- any other value: taken verbatim, whitespace included. A whitespace-only value is a prefix made of that whitespace.

> **Warning.** A prefix value that ends with a space must be quoted in `.env` and compose files, for example `RAG_ENGINE_PASSAGE_PREFIX="passage: "`. Unquoted, the trailing space is lost, the prefix becomes `passage:`, and the vectors no longer match the ones already in the collection. The same holds for any change of prefix on an existing collection: re-ingest it.

The API key is never printed: `repr()` and `str()` of a `Settings` show `qdrant_api_key=<set>` or `<unset>`, and no error message echoes it. It stays a plain attribute, and `dataclasses.asdict()` returns it, so do not log that dictionary.

`Settings(...)` can also be built directly with keyword arguments. The same range checks apply, but nothing is parsed or converted: integers must be `int` (not `bool`, `float` or a numeric string), `qdrant_https` must be a `bool`, `parity_min` must be a `float`, `backend` must be exactly `"torch"` or `"onnx"`, and a prefix string is the prefix as it is (`""` for no prefix). An empty `qdrant_host` or `model_name` is refused; a blank `qdrant_api_key` or `model_cache` becomes `None`. A wrong type raises `ConfigError`.

## Embedding backends

`make_embedder(settings)` builds the backend named by `RAG_ENGINE_BACKEND`. Both apply the same prefixes (`RAG_ENGINE_PASSAGE_PREFIX`, `RAG_ENGINE_QUERY_PREFIX`) and return `float32` vectors of `RAG_ENGINE_DIM` dimensions.

| Backend | Extra | Runs on | Meant for |
|---|---|---|---|
| `torch` | `rag-engine[torch]` | sentence-transformers, PyTorch (CUDA, MPS or CPU) | ingestion: the passages of a collection are embedded with it |
| `onnx` | `rag-engine[onnx]` | fastembed, ONNX runtime (CPU), no PyTorch | search: a service embeds its queries without installing PyTorch |

The intended setup is **passages embedded with `torch`, queries embedded with `onnx`**. It is valid only if both backends give the same vector for the same text, so this is measured, not assumed.

**Measured parity** for `intfloat/multilingual-e5-base` (repository revision `d1287505`, file `onnx/model.onnx`; fastembed 0.9.1, onnxruntime 1.32.0, torch 2.15.0 CPU, sentence-transformers 6.3.0), on 50 fixed texts in English, French, German, Spanish, Italian and Japanese — single words, an empty string, numbers and identifiers, composed and decomposed accents, texts of about 400 words and of about 900 words (cut at 512 tokens):

| Vectors | Minimum cosine, torch against onnx | Mean |
|---|---|---|
| passages | 1 − 1.3 × 10⁻¹¹ | 1.000000 |
| queries | 1 − 3.9 × 10⁻¹² | 1.000000 |

On a corpus of 20 chunks and 5 queries, the five best chunks of every query are the same, in the same order, with `onnx` queries as with `torch` queries. The test (`tests/test_parity.py`, marker `models`) fails below `RAG_ENGINE_PARITY_MIN` (0.999) or on any difference of ranking; CI runs it in the `parity` job. Run it again after changing the model or upgrading `fastembed`, `onnxruntime`, `torch` or `sentence-transformers`. `rag_engine.embedding.cosine_parity(a, b)` gives the cosine of each pair of rows of two arrays of vectors.

What the `onnx` backend does, and why it matches:

- **Model file.** fastembed does not ship `intfloat/multilingual-e5-base`; the engine registers it from the table `rag_engine.embedding.ONNX_MODELS`: the full-precision `onnx/model.onnx` of the model's own Hugging Face repository (not a quantised variant), mean pooling over the attention mask, then normalisation — the pipeline of sentence-transformers for this model. A model that fastembed lists itself is used as fastembed defines it. Any other name raises a `ValueError`.
- **Prefixes.** The engine adds the prefix once and calls fastembed's plain `embed`; fastembed adds none of its own.
- **Long texts.** The tokenizer cuts at the limit declared by the model (512 tokens), as sentence-transformers does: both backends see the same tokens.
- **Batches.** Texts are embedded 32 at a time (`OnnxEmbedder(..., batch_size=...)`, or `make_embedder(settings, batch_size=...)`). A batch is padded to its longest text and the padding is left out of the pooling, so the vector of a text does not depend on its neighbours.
- **Files.** The first use downloads about 1.1 GB into `RAG_ENGINE_MODEL_CACHE`. Without it, fastembed stores them under the system's temporary directory, which a reboot may clear: set the variable (or `FASTEMBED_CACHE_PATH`) on a server.

## Ingesting a document

`ingest_document` (in `rag_engine.ingest`, extra `ingest`) chunks one file, embeds its chunks and writes one point per chunk. The collection is ensured once per run, by the caller; `ingest_document` never creates or checks it.

`ingest_document` reads no settings and no environment variable. The caller builds everything from `Settings.from_env()` and passes it on: the store (which keeps the batch size), the embedder, and `ocr_languages=settings.ocr_languages`. Without that last argument the OCR languages are `eng`, whatever `RAG_ENGINE_OCR_LANGUAGES` says.

```python
from rag_engine.config import Settings
from rag_engine.embedding import make_embedder
from rag_engine.ingest import ingest_document
from rag_engine.store import QdrantStore

settings = Settings.from_env()  # the only place the environment is read
store = QdrantStore(settings=settings)  # connection, and RAG_ENGINE_BATCH_SIZE for the writes
embedder = make_embedder(settings)  # model, dimension, prefixes, backend, model cache

store.ensure_collection("central-bank-e5b-v1", embedder.dim)  # once, not per document

result = ingest_document(
    "reports/2025-10-30-statement.pdf",
    "ecb/monetary-policy-statement/2025-10-30",  # doc_id
    {"corpus": "central-bank", "source_code": "ecb", "date": "2025-10-30", "year": 2025},
    collection="central-bank-e5b-v1",
    store=store,
    embedder=embedder,
    ocr_languages=settings.ocr_languages,  # RAG_ENGINE_OCR_LANGUAGES; not read otherwise
    # batch_size=...  only for another write size than the store's (settings.batch_size)
)
print(result.chunks, result.pages, result.empty)
```

What it guarantees:

- **Deterministic ids.** The id of a point depends on the `doc_id`, the page the chunk starts on and its position in the document, nothing else. Ingesting the same document again overwrites its points; it never adds duplicates.
- **One ingestion date per document.** Every point of a document carries the same `ingestion_date`, the local time of the ingestion as an ISO string (or the `ingestion_date` argument, for a replay or a test).
- **Nothing written for an empty document.** A file with no extractable text gives `IngestResult(chunks=0, pages=0, empty=True)`; the embedder and the store are not called. It is not an error.
- **Errors propagate.** Nothing is caught: an unsupported format, a parser error, an embedding error or a Qdrant error reaches the caller unchanged, and the caller decides what a failed document means. A store error carries the number of points written before it in `points_written`; a failed write is not undone, and ingesting the document again repairs it.
- **Checked before any work.** A blank `doc_id` or `collection`, an unusable `batch_size`, an `ocr_languages` the settings would refuse, or metadata that sets a reserved payload key (`filename`, `chunk_number`, `ingestion_date`, `text`, `page`, or a `doc_id` different from the document's own) raises `ValueError` before the file is read. A `doc_id` equal to the document's id is accepted.

`IngestResult.pages` counts the distinct pages chunks start on (0 for a format without pages). The other arguments are `ocr`, `ocr_languages` and `chunking_options` (see below), and `batch_size` (points per write; left out, the batch size of the store, which is `RAG_ENGINE_BATCH_SIZE` when the store was built from the settings). The batch size does not bound memory or the embedding call: all the chunk texts of a document are embedded in one `encode_passage` call.

The points are the ones the previous pipeline wrote: `tests/fixtures/golden/` holds points captured from it for two documents, and the tests require the same ids, payloads and vectors.

## Payload indexes

Search filters need payload indexes: Qdrant answers a filter on an indexed field without reading every point. The engine indexes the fields listed in `rag_engine.schema.PAYLOAD_INDEXES`:

| Field | Index | Filter it serves |
|---|---|---|
| `doc_id`, `source_code`, `doc_type`, `language`, `corpus` | keyword | exact match, one value or several |
| `year` | integer | exact match and range |
| `date` | datetime | range |

`QdrantStore.ensure_payload_indexes` creates the ones a collection does not have yet and returns the fields it indexed:

```python
store = QdrantStore(settings=Settings.from_env())
store.ensure_payload_indexes("central-bank-e5b-v1")
# ['doc_id', 'source_code', 'doc_type', 'language', 'corpus', 'year', 'date'] the first time, [] afterwards
```

- **On an existing collection.** An index is built from the points already there; no point is rewritten, no document has to be ingested again. The call returns once every index is built and usable. Run it once per collection, before or after ingesting; running it again does nothing and returns `[]`.
- **Never replaces an index.** If a field is already indexed with another kind (a keyword index on `date`, say), the call raises `CollectionMismatchError` naming the field, the expected kind and the one found, and creates nothing. Qdrant itself would drop the existing index and build the other one; removing an index stays a decision of the operator (`store.client.delete_payload_index`).
- **Errors.** A refused API key is a `QdrantAuthError`. A collection that does not exist is not created: the error of the client is raised, like any other failure. Indexes created before a failure stay, and a second call creates the rest.
- **Another list.** `ensure_payload_indexes(collection, [("page", "integer")])` takes any list of `(field, kind)`, a kind being the name of a Qdrant payload index type.

**Dates.** `date` is the publication date exactly as the vault gives it; the engine does not rewrite it. Qdrant reads a full date (`2016-03-10`, taken as midnight UTC) or a full date and time (`2016-03-10T12:00:00Z`). Every other value is stored untouched and is never an error, neither when the index is built nor when a point is written, but **it matches no date range**: an imprecise date (`2016-03`, `2016`), an empty string, a null, a missing `date`. A document without a precise date is therefore left out as soon as a date bound is set, with or without the index; it can still be reached through `year`, which is filtered on its own. The bounds of a range are full dates too: Qdrant refuses `2016` or `2016-03` as a bound.

qdrant-client's local mode (`QdrantClient(":memory:")`) keeps no payload index: it accepts the creation with a warning and filters by reading every point, with the same answers.

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

The defaults are the values the old engine ran with (ChunkNorris's own defaults), and `chunk_file` passes them to ChunkNorris by name, as it does for the PDF parser's options (tables extracted, headers detected; the `PDF_*` constants of the module). One collection should be ingested with one way of cutting documents. The id of a point does not depend on the text of its chunk: it comes from the `doc_id`, the page and the position of the chunk. So when a document is ingested again after the cut changed, the points with the same (page, index) are overwritten with a different text and vector, and the (page, index) pairs that only the old cut produced remain as stale points. Nothing detects it: **a change of chunking requires a clean re-ingestion of the document set** (its old points removed first, or a new collection). This holds for the options above, and it is why:

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

The OCR languages are a setting: `RAG_ENGINE_OCR_LANGUAGES`, default `eng`, passed by the caller as `ocr_languages=settings.ocr_languages`; neither `chunk_file` nor `ingest_document` reads it. Each language listed must be installed. The value follows one rule everywhere (`Settings`, `chunk_file`, `ingest_document`): language codes joined by `+`, without whitespace, such as `eng` or `eng+fra`; anything else is refused (`ConfigError` by the settings, `ValueError` by the two functions).

Differences from the old engine:

- **OCR languages.** The old engine used every language found in the tessdata directory. To keep the old result on a host with several languages installed, list them, for example `RAG_ENGINE_OCR_LANGUAGES=eng+fra`.
- **Invalid UTF-8 in `.txt` and `.md` files.** Invalid bytes are now replaced by `�` (U+FFFD); the old engine dropped them silently. The text of such a file therefore differs slightly from what was ingested before, and so do the chunks that contain those bytes.
- **OCR switches read at each call.** `ENABLE_OCR` and `TESSDATA_PREFIX` are both read each time a file is chunked. The old engine read `ENABLE_OCR` once, when it was imported, so a value set afterwards was ignored; it now takes effect on the next document.
- **`ocr="always"` and `TESSDATA_PREFIX`.** The old engine corrected a `TESSDATA_PREFIX` set to the parent folder only when `ENABLE_OCR=true`; it is now also corrected for a forced `always`.
- **Reserved payload keys.** Metadata that sets a key the engine writes itself (`filename`, `chunk_number`, `ingestion_date`, `text`, `page`, or a `doc_id` other than the document's) is refused with a `ValueError`; the old pipeline accepted it silently.

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

The tests that parse real documents need the `ingest` extra (`pip install -e ".[dev,ingest]"`) and skip without it; the OCR test also needs Tesseract with the `eng` language data (`apt-get install tesseract-ocr`). CI installs both on the Python 3.12 and 3.13 jobs (3.13 is the version production runs on and the golden fixture was captured with); the 3.10 job keeps the light install.

Changes are recorded in [CHANGELOG.md](CHANGELOG.md).

Two test markers are deselected by default:

- `integration`: needs a real Qdrant, given by `RAG_ENGINE_IT_QDRANT_URL`.
- `models`: downloads embedding models. It holds the test of the real torch model and the parity test of the two backends (`tests/test_parity.py`); it needs `pip install -e ".[dev,torch,onnx]"`. CI runs it in its own job, `parity`, with the model files cached.

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:v1.20.0
RAG_ENGINE_IT_QDRANT_URL=http://localhost:6333 pytest -m integration -q
pytest -m models -q -s  # -s prints the measured parity
```

The integration tests of ingestion parse real documents, so they also need the `ingest` extra and skip without it; the CI integration job installs it and fails if the parsers cannot be imported.

## Licence

MIT, see [LICENSE](LICENSE). Portions are derived from eigenmind (MIT); see [NOTICE](NOTICE).
