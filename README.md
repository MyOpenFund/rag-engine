# rag-engine

**Status: ingestion, both embedding backends (PyTorch, ONNX), filtered search and the search evaluation harness are delivered.**

rag-engine does two things:

1. **Ingestion** of one document into [Qdrant](https://qdrant.tech): chunk, embed, write.
2. **Search** over what was ingested: one dense search, narrowed by filters on the source, the type, the language, the document, the corpus and the publication date.

It also measures that search: `rag-engine eval` runs a file of questions with known answers and reports how often the right documents come first (see [Evaluating search quality](#evaluating-search-quality)).

It is used by two MyOpenFund components: the data-orchestrator (ingestion) and the MCP server (search).

## Install

Python 3.10 or later. The base install is light; each consumer adds only the extras it needs.

| Install | Adds | For |
|---|---|---|
| `rag-engine` | `qdrant-client` (1.10 or later), `numpy` | settings, payload schema, point ids, the Qdrant store (collections, indexes, writes), search, the evaluation harness and the `rag-engine` command |
| `rag-engine[torch]` | `sentence-transformers` 6.x | PyTorch embedding backend |
| `rag-engine[onnx]` | `fastembed` 0.9.x (ONNX runtime), `huggingface_hub` (0.20 or later) | ONNX embedding backend, without PyTorch: for a service that embeds search queries |
| `rag-engine[ingest]` | `chunknorris`, `markitdown[pptx]`, `psutil` | document parsing and chunking |
| `rag-engine[dev]` | `pytest`, `ruff` | development |

The modules of the engine import no embedding backend and no document parser at import time: each is imported when it is used, and `torch` is never imported unless the torch backend is built. One import is not the engine's to decide: `qdrant-client`, which the store, the search, the evaluation and the command import, itself loads `fastembed` when that package is installed — so with the `onnx` extra, `import rag_engine.store` loads fastembed, the ONNX runtime, `huggingface_hub` and `tokenizers`. Without that extra none of them is loaded. Because of that import, `import rag_engine` itself sets `ORT_DISABLE_TELEMETRY=1` in the environment of the process, unless the variable is already set: the ONNX runtime reads it when it is imported (see [Embedding backends](#embedding-backends), *No telemetry*).

`sentence-transformers` is bounded to the major version parity was measured on (`>=6,<7`: 6.0.0 to 6.3.0 today, not 5.x), for the reason `fastembed` is bounded: the vectors depend on it. Raise a bound only after `pytest -m models` has passed on the new version.

The package ships a `py.typed` marker: its annotations are usable by type checkers.

## Public API

The package root exposes only `__version__`. Everything else is imported from its submodule; each one lists its public names in `__all__`.

| Module | Main names |
|---|---|
| `rag_engine.config` | `Settings`, `ConfigError`, `valid_ocr_languages` |
| `rag_engine.schema` | `point_id`, `page_for_id`, `build_payload`, `RESERVED_KEYS`, `PAYLOAD_INDEXES` |
| `rag_engine.store` | `QdrantStore`, `StoreError`, `QdrantAuthError`, `CollectionMismatchError`, `CollectionNotFoundError`, `DEFAULT_TIMEOUT` |
| `rag_engine.embedding` | `ModelSpec`, `Embedder`, `make_embedder`, `TorchEmbedder`, `OnnxEmbedder`, `FakeEmbedder`, `BaseEmbedder`, `detect_device`, `cosine_parity`, `ONNX_MODELS`, `OnnxModel`, `ONNX_BATCH_SIZE`, `ModelIntegrityError`, `ModelDownloadError` |
| `rag_engine.chunking` | `chunk_file`, `Chunk`, `ChunkingOptions`, `UnsupportedFormatError`, `ocr_available`, `check_ocr_languages`, `SUPPORTED_EXTENSIONS`, `OCR_MODES` |
| `rag_engine.ingest` | `ingest_document`, `IngestResult` |
| `rag_engine.search` | `search`, `SearchFilters`, `Hit`, `SearchError`, `is_searchable`, `QUERY_NORMALIZATION` |
| `rag_engine.eval` | `load_questions`, `evaluate`, `format_report`, `Question`, `QuestionSet`, `EvalReport`, `QuestionResult`, `RankedDocument`, `NotInCollection`, `Aggregate`, `Means`, `EssentialAggregate`, `EssentialMeans`, `FilterEffect`, `Abstention`, `TopScores`, `EvalError`, `DEFAULT_K`, `DEFAULT_CHUNK_LIMIT`, `DEFAULT_BOOTSTRAP_RESAMPLES`, `DEFAULT_INTERVAL_MIN_DOCUMENTS`, `DEFAULT_SCORE_LEVELS`, `BOOTSTRAP_SEED`, `INTERVAL_LEVEL`, `BEHAVIORS`, `DEFAULT_BEHAVIOR`, `UNFILTERED_SUFFIX`, `UNFILTERED_TAG`, `QUESTION_FIELDS`, `FILTER_FIELDS` |
| `rag_engine.eval_check` | `check_questions`, `load_catalogue`, `format_check`, `QuestionCheck`, `CatalogueEntry`, `DocumentReference` |
| `rag_engine.cli` | `main` (the `rag-engine` command) |
| `rag_engine.errors` | `RagEngineError` |

**Errors.** Every exception class the engine defines derives from `rag_engine.errors.RagEngineError`:

| Class | Module | Raised when |
|---|---|---|
| `ConfigError` (also a `ValueError`) | `rag_engine.config` | a setting has an invalid value |
| `StoreError` | `rag_engine.store` | base of the three below |
| `QdrantAuthError` | `rag_engine.store` | Qdrant refuses the API key (HTTP 401 or 403) |
| `CollectionMismatchError` | `rag_engine.store` | a collection is not one unnamed cosine vector of the expected dimension, or a payload field is indexed with another kind of index |
| `CollectionNotFoundError` | `rag_engine.store` | `check_collection` is asked about a collection that does not exist |
| `UnsupportedFormatError` (also a `ValueError`) | `rag_engine.chunking` | a file has an extension the engine does not read |
| `SearchError` (also a `ValueError`) | `rag_engine.search` | a search argument or a filter cannot be used as it is |
| `EvalError` (also a `ValueError`) | `rag_engine.eval` | a question file, a question or an evaluation parameter is unusable |
| `ModelIntegrityError` | `rag_engine.embedding` | a pinned model file has another size or SHA-256 than the table |
| `ModelDownloadError` | `rag_engine.embedding` | the pinned model files are not in the model cache and cannot be downloaded |

**Errors that are not `RagEngineError`.** `except RagEngineError` does not catch what the engine lets through unchanged. A service built on the engine meets these:

| Error | When |
|---|---|
| `qdrant_client.http.exceptions.UnexpectedResponse`, status 404 | a search, a write, a count or an index creation on a collection that does not exist; `check_collection` at start turns the case into a `CollectionNotFoundError` |
| `UnexpectedResponse`, status 400 | a query vector, or the vector of a point written, of another dimension than the collection (`Vector dimension error`); `check_collection` at start turns the case into a `CollectionMismatchError` |
| `UnexpectedResponse`, status 422 | a collection name the server does not accept (a `/` in it, more than 255 characters): the request is refused before anything is looked up, from `check_collection` as well |
| `UnexpectedResponse`, another status | any other answer of the server (500, 503, ...); 401 and 403 alone become `QdrantAuthError` |
| `qdrant_client.http.exceptions.ResponseHandlingException` | the server cannot be reached or does not answer in time (connection refused, DNS, TLS, the timeout of the store); it wraps the error of the HTTP client |
| `ImportError` | an embedder or a parser is built without its extra installed: the message names the extra (`rag-engine[torch]`, `rag-engine[onnx]`, `rag-engine[ingest]`) |
| `RuntimeError` | an embedder is used after `release()` or after the end of its `with` block |
| `ValueError`, `TypeError` | a bad argument of the store, of an embedder or of `ingest_document` (a `batch_size` of 0, a blank `doc_id`, a model of another dimension than announced, a model unknown to the backend); `SearchFilters(**mapping)` with an unknown key — `SearchFilters.from_mapping` raises a `SearchError` instead |
| the error of a parser | a file ChunkNorris or MarkItDown cannot read |

In qdrant-client's local mode (`QdrantClient(":memory:")`, tests) a missing collection and a vector of another dimension are plain `ValueError`s of the client. `store.is_reachable()` never raises.

## Configuration

`Settings.from_env()` (in `rag_engine.config`) reads these environment variables when it is called. Nothing is read at import, and no `.env` file is loaded: a caller that uses one loads it first.

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port (1 to 65535) |
| `QDRANT_API_KEY` | none | Qdrant API key; blank means no authentication |
| `QDRANT_HTTPS` | `false` | Connect over TLS (`1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`) |
| `RAG_ENGINE_QDRANT_TIMEOUT` | `60` | Seconds a request to Qdrant may take: a positive number. Read by `QdrantStore` when it is built from the settings; an explicit `QdrantStore(..., timeout=...)` wins. qdrant-client counts whole seconds and rounds a fraction up |
| `RAG_ENGINE_MODEL` | `intfloat/multilingual-e5-base` | Embedding model |
| `RAG_ENGINE_DIM` | `768` | Vector dimension of the model |
| `RAG_ENGINE_PASSAGE_PREFIX` | `passage: ` (with the trailing space) | Text put before each chunk when embedding; `none` for no prefix |
| `RAG_ENGINE_QUERY_PREFIX` | `query: ` (with the trailing space) | Text put before each query when embedding; `none` for no prefix |
| `RAG_ENGINE_BACKEND` | `torch` | Embedding backend, `torch` or `onnx` (case-insensitive); see [Embedding backends](#embedding-backends) |
| `RAG_ENGINE_MODEL_CACHE` | none | Directory for model files. Unset leaves the backend default: the Hugging Face cache for `torch`; `FASTEMBED_CACHE_PATH`, or a folder under the system's temporary directory, for `onnx`. **Set it on a server**, to a directory of its own, read-only for the service once filled (see [Embedding backends](#embedding-backends)) |
| `RAG_ENGINE_PARITY_MIN` | `0.999` | Smallest cosine accepted between the `torch` and `onnx` vectors of the same text (above 0, at most 1). Read by the parity test only |
| `ORT_DISABLE_TELEMETRY` | `1`, set by the engine | Not a setting of the engine: a variable of the ONNX runtime, which sends telemetry to its publisher unless it is `1`. `import rag_engine` sets it to `1` **when it is not set** (`os.environ.setdefault`); a value you set yourself, `0` included, is kept. It is read by the runtime when the runtime is imported; see [Embedding backends](#embedding-backends), *No telemetry* |
| `RAG_ENGINE_ONNX_THREADS` | none | Threads the `onnx` backend runs inference on (at least 1). Unset: the ONNX runtime decides, from the cores of the host. **In a container with a CPU limit, set it to the limit**; see [Embedding backends](#embedding-backends) |
| `RAG_ENGINE_BATCH_SIZE` | `100` | Points sent to Qdrant per write (at least 1). It sizes the writes only: all the chunks of a document are embedded in one call |
| `RAG_ENGINE_OCR_LANGUAGES` | `eng` | Tesseract languages joined by `+`, without whitespace, e.g. `eng+fra`. The caller passes it to `ingest_document` (see below) |
| `RAG_ENGINE_SEARCH_LIMIT` | `10` | Hits returned when no limit is given. The caller passes it to `search` (see [Searching](#searching)) |
| `RAG_ENGINE_SEARCH_MAX_LIMIT` | `50` | Largest limit a caller may ask for; not below the default limit. Passed to `search` by the caller too |

An unset or blank variable gives the default, and surrounding whitespace is ignored. An invalid value raises `ConfigError` (a `ValueError`) naming the variable and the value as it was set, cut to 80 characters.

The two prefixes follow their own rule:

- unset or empty: the default. A compose file that passes an unset variable through delivers an empty value, and that must not remove the prefix.
- `none` (any case, surrounding whitespace ignored): no prefix.
- any other value: taken verbatim, whitespace included. A whitespace-only value is a prefix made of that whitespace.

> **Warning.** A prefix value that ends with a space must be quoted in `.env` and compose files, for example `RAG_ENGINE_PASSAGE_PREFIX="passage: "`. Unquoted, the trailing space is lost, the prefix becomes `passage:`, and the vectors no longer match the ones already in the collection. The same holds for any change of prefix on an existing collection: re-ingest it.

The API key is never printed: `repr()` and `str()` of a `Settings` show `qdrant_api_key=<set>` or `<unset>`, and no error message echoes it. It stays a plain attribute, and `dataclasses.asdict()` returns it, so do not log that dictionary.

`Settings(...)` can also be built directly with keyword arguments. The same range checks apply, but nothing is parsed or converted: integers must be `int` (not `bool`, `float` or a numeric string), `qdrant_https` must be a `bool`, `parity_min` must be a `float`, `qdrant_timeout` an `int` or a `float`, `backend` must be exactly `"torch"` or `"onnx"`, and a prefix string is the prefix as it is (`""` for no prefix). An empty `qdrant_host` or `model_name` is refused; a blank `qdrant_api_key` or `model_cache` becomes `None`. A wrong type raises `ConfigError`.

## Embedding backends

`make_embedder(settings)` builds the backend named by `RAG_ENGINE_BACKEND`. Both apply the same prefixes (`RAG_ENGINE_PASSAGE_PREFIX`, `RAG_ENGINE_QUERY_PREFIX`) and return `float32` vectors of `RAG_ENGINE_DIM` dimensions.

| Backend | Extra | Runs on | Meant for |
|---|---|---|---|
| `torch` | `rag-engine[torch]` | sentence-transformers, PyTorch (CUDA, MPS or CPU) | ingestion: the passages of a collection are embedded with it |
| `onnx` | `rag-engine[onnx]` | fastembed, ONNX runtime (CPU), no PyTorch | search: a service embeds its queries without installing PyTorch |

The intended setup is **passages embedded with `torch`, queries embedded with `onnx`**. It is valid only if both backends give the same vector for the same text, so this is measured, not assumed.

**Measured parity** for `intfloat/multilingual-e5-base` (repository revision `d128750597153bb5987e10b1c3493a34e5a4502a`, file `onnx/model.onnx`, 1,110,059,084 bytes, SHA-256 `84a4d426f7e87a6bf5bf195f0bae2c4a7d15f675b23ca96f42fab8326d7a77aa`, with the tokenizer and configuration files listed below; fastembed 0.9.1, onnxruntime 1.32.0, torch 2.15.0 CPU, sentence-transformers 6.3.0), on 50 fixed texts in English, French, German, Spanish, Italian and Japanese — single words, an empty string, numbers and identifiers, composed and decomposed accents, texts of about 400 words and of about 900 words (cut at 512 tokens):

| Vectors | Minimum cosine, torch against onnx | Mean |
|---|---|---|
| passages | 1 − 1.3 × 10⁻¹¹ | 1.000000 |
| queries | 1 − 3.9 × 10⁻¹² | 1.000000 |

On a corpus of 20 chunks and 5 queries, the five best chunks of every query are the same, in the same order, with `onnx` queries as with `torch` queries. The test (`tests/test_parity.py`, marker `models`) fails below `RAG_ENGINE_PARITY_MIN` (0.999) or on any difference of ranking; CI runs it in the `parity` job. The test loads the model exactly as production does, checks that the vectors came from the pinned files on the CPU provider, and checks that a copy with one byte changed — in the ONNX file, then in `tokenizer.json` — is refused. Run it again after changing the model or upgrading `fastembed`, `onnxruntime`, `torch` or `sentence-transformers`; `fastembed` is bounded to the tested minor version (`>=0.9,<0.10`), and the bound is raised only after this test has passed on the new version. `rag_engine.embedding.cosine_parity(a, b)` gives the cosine of each pair of rows of two arrays of vectors.

What the `onnx` backend does, and why it matches:

- **Model files, pinned and verified.** The table `rag_engine.embedding.ONNX_MODELS` names the one ONNX export the engine loads for `intfloat/multilingual-e5-base`: the full-precision `onnx/model.onnx` of the model's own Hugging Face repository (not a quantised variant), at one revision — and, in `OnnxModel.files`, the size and SHA-256 of **every file loaded for the model**: the ONNX file and the four files the tokenizer is built from, which decide the vectors just as much as the weights do. They are the files the parity above was measured with:

  | File | Size (bytes) | SHA-256 |
  |---|---|---|
  | `onnx/model.onnx` | 1,110,059,084 | `84a4d426f7e87a6bf5bf195f0bae2c4a7d15f675b23ca96f42fab8326d7a77aa` |
  | `config.json` | 694 | `9dab198f24c8c0879e481cf7822005d5ecbceedbacb390ffafa594e28d31bac4` |
  | `special_tokens_map.json` | 280 | `06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f` |
  | `tokenizer.json` | 17,082,660 | `62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626` |
  | `tokenizer_config.json` | 418 | `efb5c0d09722e5fe59a462cd2a9976ee216d55b037597d997cd3fe833216da15` |

  The engine fetches these files at that revision itself (`huggingface_hub.snapshot_download`) and hands the directory to fastembed, which downloads nothing. **Before the model is built, every file is hashed and compared with the table**; a different size or hash raises `ModelIntegrityError`, naming the file and the expected and found hashes. A changed export or tokenizer can therefore not reach production unnoticed, whatever the head of the repository holds that day. The four small files (17 MB in all) are hashed every time an embedder is built. Hashing the 1.1 GB ONNX file takes about one second; it is done once per process, and the hash is kept for the same path, size and modification time. An `OnnxModel` cannot be written without a pin for the ONNX file and for each of the four other files (`files` maps a path in the repository to `(sha256, size)`; `sha256` and `size` are those of the ONNX file). To change a file, re-measure parity first, then update the table.

- **ONNX model: what the check protects against, and what it does not.** It catches a download that was changed or damaged, on the way or on disk, and an export or a tokenizer other than the one parity was measured with: a repository whose files moved, a cache filled by another tool or by hand, a file cut short. It does **not** protect against someone who can write to the model cache while the service runs. The files are verified when an embedder is built, not while it is used; the ONNX runtime opens them after the check; and within one process the kept hash of the ONNX file is trusted as long as its path, size and modification time are the same. So the model cache of a server should be a **dedicated directory, set with `RAG_ENGINE_MODEL_CACHE`, mounted read-only for the service once it is filled** (fill it once, with a first load or by copying the directory of another host). Left unset, the cache is a directory under the system's temporary directory (`<tmp>/fastembed_cache`, or `FASTEMBED_CACHE_PATH`), which other users of the host may be able to write to and which a reboot may clear.
- **Other models.** A model that is not in the table but that fastembed lists is used as fastembed defines and downloads it, with a warning in the log: its files are not pinned or verified by the engine, and its parity was not measured. Any other name raises a `ValueError`.
- **No telemetry, and no outbound connection once the cache is filled.** The official builds of the ONNX runtime send telemetry by default (onnxruntime 1.32.0, Linux wheel; its `Privacy.md` says so): importing `onnxruntime` creates a device identifier and an event store under `$HOME/.cache/Microsoft/DeveloperTools/.onnxruntime/` and starts an uploader that connects to `mobile.events.data.microsoft.com:443`. The runtime turns all of it off when the environment variable `ORT_DISABLE_TELEMETRY` is `1` **at the time it is imported**; setting it later, before a session is created, changes nothing (measured: the files are written by the import alone). The engine does not send telemetry on behalf of its users: `import rag_engine` sets `ORT_DISABLE_TELEMETRY=1` when the variable is not set, before any module of the engine can import the runtime (`rag_engine.store` does, through `qdrant-client`, when `fastembed` is installed), and the `onnx` backend sets the same default again before it imports fastembed. It is a default: set the variable yourself, to `0`, to allow the telemetry. Two limits. **The order of the imports**: a program that imports `qdrant_client` (which imports `fastembed` when it is installed), `fastembed` or `onnxruntime` **before** `rag_engine` loads the runtime before the default exists, and the telemetry is on for that process. It happens without anyone choosing it: tools that sort imports alphabetically put `qdrant_client` before `rag_engine`. `import rag_engine` then warns once (a `RuntimeWarning`, when `onnxruntime` is already loaded and the variable was not set), but cannot undo it. So a service should export `ORT_DISABLE_TELEMETRY=1` in its environment itself rather than rely on the order of its imports (the tests of this repository do: `tests/conftest.py` and the CI workflow set it). And the variable is the runtime's, so another version of it may read it differently (check its `Privacy.md` when upgrading).

  Measured behind a proxy that refuses and records every request, with a complete model cache, an `OnnxEmbedder` built, one query embedded and 75 seconds of waiting: with `ORT_DISABLE_TELEMETRY=0` forced, four `CONNECT mobile.events.data.microsoft.com:443` in that time, the first within a few seconds of the start, and the two files written under `$HOME/.cache/Microsoft/`; with the default of the engine, **no request at all and nothing written**. With a complete model cache the engine then makes no outbound connection: the model files are used without any request to the Hugging Face Hub. A service should still set **`HF_HUB_OFFLINE=1`**, so that a cache that is incomplete (a file removed, another model name) fails at once instead of reaching for the Hub.
- **CPU only.** The session is created with the CPU execution provider alone, also where the GPU build of onnxruntime is installed: parity was measured on CPU.
- **Threads.** By default the ONNX runtime sizes its thread pool from the cores of the host. It does not see the CPU limit of a container (`docker run --cpus=2`, a Kubernetes CPU limit): with more threads than CPUs, inference gets slower, not faster. In a CPU-limited container set `RAG_ENGINE_ONNX_THREADS` to the limit (`OnnxEmbedder(..., threads=...)` does the same).
- **Prefixes.** The engine adds the prefix once and calls fastembed's plain `embed`; fastembed adds none of its own.
- **Long texts.** The tokenizer cuts at the limit declared by the model (512 tokens), as sentence-transformers does: both backends see the same tokens.
- **Batches.** Texts are embedded 32 at a time (`OnnxEmbedder(..., batch_size=...)`, or `make_embedder(settings, batch_size=...)`). A batch is padded to its longest text and the padding is left out of the pooling, so the vector of a text does not depend on its neighbours.
- **Files.** The first use downloads about 1.1 GB into `RAG_ENGINE_MODEL_CACHE`. Without it, the files go where fastembed puts its own (`FASTEMBED_CACHE_PATH`, or a folder under the system's temporary directory, which a reboot may clear): set the variable on a server. Once the files are there, loading makes no request to the Hugging Face Hub. When they are missing and cannot be downloaded, a `ModelDownloadError` says which files, from which repository and revision, and into which directory; on a host without access to huggingface.co, point `RAG_ENGINE_MODEL_CACHE` at a directory filled beforehand (the directory of a host where the model was loaded once can be copied as it is).

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

`QdrantStore.ensure_payload_indexes` creates the ones a collection does not have yet and returns the fields it created an index for:

```python
store = QdrantStore(settings=Settings.from_env())
store.ensure_payload_indexes("central-bank-e5b-v1")
# ['doc_id', 'source_code', 'doc_type', 'language', 'corpus', 'year', 'date'] the first time, [] afterwards
```

- **On an existing collection.** An index is built from the points already there; no point is rewritten, no document has to be ingested again. The call returns once every index is built and usable. Run it once per collection, before or after ingesting; running it again does nothing and returns `[]`.
- **What the returned list says.** It lists the fields for which a create call succeeded in this call, no more. Two callers that run at the same moment may each find the indexes missing, each create them, and each report every field. In qdrant-client's local mode, which keeps no index, every call reports every field.
- **Checking afterwards.** Do not read `get_collection(...).payload_schema[field].points` right after the creation on a large collection: that count has been seen to lag for a few seconds while the index is built. Count through a filter instead: `store.client.count(collection, count_filter=..., exact=True)`.
- **Never replaces an index.** If a field is already indexed with another kind (a keyword index on `date`, say), the call raises `CollectionMismatchError` naming the field, the expected kind and the one found, and creates nothing. Qdrant itself would drop the existing index and build the other one; removing an index stays a decision of the operator (`store.client.delete_payload_index`).
- **Errors.** A refused API key is a `QdrantAuthError`. A collection that does not exist is not created: the error of the client is raised, like any other failure. Indexes created before a failure stay, and a second call creates the rest.
- **Another list.** `ensure_payload_indexes(collection, [("page", "integer")])` takes any list of `(field, kind)`, a kind being the name of a Qdrant payload index type.

**A value an index cannot read.** No value is ever refused, neither when an index is built nor when a point is written: the point stays stored, and a search without a filter on that field still returns it. But **a filter on that field no longer sees it**. Checked on Qdrant 1.20:

| Field | Values the index reads | Values it does not read: the point is invisible to a filter on the field |
|---|---|---|
| `year` (integer) | an integer (`2016`), a float with an integral value (`2016.0`) | a string (`"2016"`), a float such as `2016.5`, a boolean, a null, a missing key |
| keyword fields | a string, a list of strings | a number, a boolean, a null, a missing key: a filter, which matches strings, never finds them |
| `date` (datetime) | `2016-03-10` (taken as midnight UTC); with a time after `T` or a space (`2016-03-10T12:00:00`, `2016-03-10 12:00:00`, `2016-03-10T12:00`); with `Z` or an offset (`2016-03-10T12:00:00+02:00`); also the lenient `2016-3-10` | `2016-03`, `2016`, an empty string, a null, a missing key, an integer (`20160310`, a Unix timestamp), `10/03/2016` |

The index changes the answer in one case: without an integer index, a range on `year` still matches `2016.5`; once the index exists it does not. A `year` stored as the string `"2016"` matches no integer range, with or without the index. **Before creating the indexes on a collection that already holds points, check the types of the stored values** of `year` and of the keyword fields (read a sample with `store.client.scroll(collection, with_payload=["year", "source_code", "doc_type", "language", "corpus", "doc_id"])`): a field written with another type has to be rewritten before its filter can be trusted.

**Dates.** `date` is the publication date exactly as the vault gives it; the engine does not rewrite it. A document without a date the index reads is left out as soon as a date bound is set, with or without the index; its `year` is indexed on its own, but `search` has no filter on `year` in this version.

**Date bounds.** A bound must be a full date, and what happens to another value depends on how it reaches Qdrant:

- through the REST API, Qdrant refuses `"2016"` and `"2016-03"` alike with HTTP 400 (not an RFC 3339 date);
- through the Python client, `"2016-03"` never leaves the process: `models.DatetimeRange` raises a validation error;
- through the Python client, **`"2016"` is accepted**: the client reads it as a Unix timestamp, 2016 seconds after the epoch (`1970-01-01T00:33:36Z`), and sends that. Nothing fails: as a lower bound it matches almost every dated point, as an upper bound almost none.

This is why `rag_engine.search` accepts only a `datetime.date` or a `YYYY-MM-DD` string as a bound, and refuses everything else itself (see [Searching](#searching)).

qdrant-client's local mode (`QdrantClient(":memory:")`) keeps no payload index: it accepts the creation with a warning and filters by reading every point, with the same answers.

## Searching

`search` (in `rag_engine.search`, base install) embeds a query, asks Qdrant for the closest chunks that pass the filters, and returns them as `Hit` objects, best first. It is one dense cosine search; several chunks of one document may come back, nothing is grouped.

Like `ingest_document`, `search` reads no settings and no environment variable. The caller builds everything from `Settings.from_env()` and passes it on, the two limits included: without `limit=settings.search_limit` and `max_limit=settings.search_max_limit`, the limits are the built-in 10 and 50, whatever `RAG_ENGINE_SEARCH_LIMIT` and `RAG_ENGINE_SEARCH_MAX_LIMIT` say.

```python
from rag_engine.config import Settings
from rag_engine.embedding import make_embedder
from rag_engine.search import SearchFilters, search
from rag_engine.store import QdrantStore

settings = Settings.from_env()  # with RAG_ENGINE_BACKEND=onnx: queries are embedded without PyTorch
store = QdrantStore(settings=settings)
embedder = make_embedder(settings)  # same model, dimension and prefixes as the ingestion

collection = "central-bank-e5b-v1"
store.check_collection(collection, embedder.dim)  # once at start; reads only, creates nothing

hits = search(
    "What did the Governing Council decide on asset purchases?",
    collection=collection,
    store=store,
    embedder=embedder,
    filters=SearchFilters(
        source_codes=["ecb", "fed"],  # ecb OR fed
        doc_types=["monetary-policy-statement", "minutes"],  # AND one of these types
        date_from="2016-01-01",  # AND published from this day...
        date_to="2016-03-10",  # ...to this day, both included
    ),
    limit=settings.search_limit,  # RAG_ENGINE_SEARCH_LIMIT; not read otherwise
    max_limit=settings.search_max_limit,  # RAG_ENGINE_SEARCH_MAX_LIMIT; not read otherwise
)
for hit in hits:
    print(f"{hit.score:.3f}", hit.doc_id, hit.date, hit.page, hit.text[:80])
```

**Check the collection once, at start.** `search` itself sends one request and checks nothing: on a collection that does not exist, or that has another dimension, it raises the error of the Qdrant client, at the first query. A search service therefore calls `store.check_collection(collection, embedder.dim)` once when it starts. It sends two read requests and never creates or changes anything, so it works with a read-only API key, and it says what is wrong in the engine's own words:

| `check_collection` raises | When |
|---|---|
| `CollectionNotFoundError` | the collection does not exist (it is not created) |
| `CollectionMismatchError` | it is not one unnamed vector of that dimension compared by cosine: another dimension, another distance (Dot, Euclid, Manhattan), named vectors, no dense vector |
| `QdrantAuthError` | the API key is refused (HTTP 401 or 403) |
| the error of the Qdrant client | anything else: an unreachable server, a timeout, a server error |

**`ensure_collection` and `ensure_payload_indexes` are for writers only**: whoever owns the collection (the ingestion) calls them once per run. `ensure_collection` creates the collection when it is missing, and `ensure_payload_indexes` creates the indexes that make the filters fast; with a read-only API key both are refused with a `QdrantAuthError` as soon as they have something to create. The answers of a search are the same with and without the indexes.

**Hits.** A `Hit` is a frozen dataclass: `text`, `score` (the cosine, higher is closer), `doc_id`, `title`, `date`, `source_code`, `doc_type`, `language`, `page`, `chunk_number`, `corpus`, then `point_id` and `payload`, the whole payload for the keys that have no field (`year`, `filename`, `ingestion_date`, ...). A key the payload does not have gives `None`; `page` is `None` for a document without pages (and `0` is the first page). `date` is the publication date exactly as stored. `hit.to_dict()` returns the fields without `payload`, ready for `json.dumps`; `to_dict(include_payload=True)` adds it.

**Limits.** `limit` is the largest number of hits wanted, `max_limit` the largest limit accepted. A `limit` above `max_limit` is refused with a `SearchError` that names the maximum; it is never lowered silently.

**Filters.** A `SearchFilters` is built from keyword arguments, all optional, and checked when it is built. When the filters arrive as data — the arguments of a tool of an MCP server, a JSON body, the `filters` of a question file — build it with `SearchFilters.from_mapping`:

```python
import json

arguments = json.loads('{"source_codes": ["ecb"], "date_from": "2016-01-01", "languages": null}')
filters = SearchFilters.from_mapping(arguments)  # SearchFilters(source_codes=["ecb"], date_from="2016-01-01")

SearchFilters.from_mapping({"source": ["ecb"]})
# SearchError: unknown filter 'source' (the filters are: source_codes, doc_types, languages, doc_ids,
# date_from, date_to, corpus)
```

The keys are the field names below. A missing key and a `null` both mean "not set"; a key that is not a field is a `SearchError` naming it and listing the fields (`SearchFilters(**arguments)` would raise a bare `TypeError`); the values are checked exactly as the constructor checks them.

| Field | Payload field | Value |
|---|---|---|
| `source_codes`, `doc_types`, `languages`, `doc_ids` | `source_code`, `doc_type`, `language`, `doc_id` | a list (or tuple, or set) of non-blank strings, matched exactly |
| `date_from`, `date_to` | `date` | a `datetime.date`, or a string written exactly `YYYY-MM-DD` |
| `corpus` | `corpus` | one non-blank string |

- Fields are combined with **AND**; the values of one list with **OR**.
- `None` (the default) is the only way to write "no restriction". **An empty list is refused**: it could mean "match nothing" as well as "match everything", and the two give opposite results.
- A bare string where a list is expected (`source_codes="ecb"`) is refused: it would be read letter by letter. Write `["ecb"]`.
- A value no chunk carries is not an error: the search returns `[]`. Values are compared as they are, case included.
- A string that cannot be encoded as UTF-8 is refused, in a list and as `corpus`: a lone surrogate, which `json.loads('"\\ud800"')` returns and a client may therefore send. The error names the field, instead of the serialisation error the Qdrant client would raise later.
- There is no filter on `year` in this version.

**Dates.** Both bounds are inclusive, and a bound is a whole day in UTC: `date_to="2016-03-10"` keeps a document stored as `2016-03-10` and one stored as `2016-03-10T15:30:00Z`. (The upper bound is sent to Qdrant as "before the next day at 00:00 UTC", so it does not depend on the time of day a date was stored with.) Only a full date is a bound: `"2016"`, `"2016-03"`, `"10/03/2016"`, a date with a time, a `datetime.datetime` or a number raise a `SearchError`, and so does `date_from` after `date_to`. The check is the engine's own, on purpose: handed the string `"2016"`, qdrant-client reads a Unix timestamp and the filter matches almost everything. **As soon as one bound is set, a chunk without a full date is left out**: no `date`, a null, an empty string, or an imprecise date such as `2016-03` (see [Payload indexes](#payload-indexes)).

**Queries.** A query must hold something to search for: at least one letter, number, symbol or punctuation mark (Unicode general categories L, N, S, P), in any script. `"2016"`, `"€"` and `"$AAPL"` are queries, and so is punctuation alone (`"???"`): degenerate, but typed. A query made only of combining marks, zero-width or other invisible characters, control characters and spaces is refused with a `SearchError` ("the query has no searchable character") instead of returning a full ranked list for a question that asks nothing. The rule holds for the normalised query too: an isolated accent such as `"¨"` or `"´"`, which NFKD writes as a space and a combining mark, is refused when it is the whole query. A query or a collection name that cannot be encoded as UTF-8 (a lone surrogate) is refused as well.

**Query normalisation.** The chunker stores the text of the chunks in Unicode NFKD (accents and other marks as separate characters), and that text is what was embedded. `search` therefore puts the query in NFKD too before embedding it (`normalize_query=True`, the default). For French, German, Spanish or Greek the tokenizer of `intfloat/multilingual-e5-base` gives the same tokens for both forms, so nothing changes. For Japanese (voiced kana), Korean and Vietnamese it does not: typed as usual (NFC), `デフレ脱却` is cut into `デ`, `フレ`, `脱`, `却`, while the stored text was cut into `テ`, `゙`, `フレ`, `脱`, `却`; only the NFKD query gives the tokens of the passages. `normalize_query=False` embeds the query as typed; it is for a collection whose passages were embedded in another form. This is a parameter of `search`, not a setting.

**Errors.** `SearchError` (a `ValueError` and a `RagEngineError`) for a bad argument or filter, raised before anything is embedded or sent. A refused API key is a `QdrantAuthError`. Everything else — an error of the embedder, a missing collection, an unreachable server — reaches the caller unchanged: see "Errors that are not `RagEngineError`" under [Public API](#public-api). A request waits at most the timeout of the store (`RAG_ENGINE_QDRANT_TIMEOUT`, 60 seconds by default; a search service usually wants less).

**Versions.** Search uses Qdrant's query API: qdrant-client 1.10 or later, and a Qdrant server 1.10 or later.

## Evaluating search quality

`rag-engine eval` runs the search on a file of questions whose answers are known, and reports how well the expected documents are ranked.

```bash
export QDRANT_HOST=localhost QDRANT_PORT=6333   # the connection and the model come from the environment
export RAG_ENGINE_BACKEND=onnx                  # the model must be the one the collection was embedded with
export RAG_ENGINE_MODEL_CACHE=/var/cache/rag-engine-models   # where the model files are kept
rag-engine eval --questions eval/known_item_v1.jsonl --collection documents
rag-engine eval --questions eval/known_item_v1.jsonl --collection documents \
    --k 10 --chunk-limit 50 --backend torch --output report.json --fail-under-hit-at-k 0.9
```

| Option | Meaning |
|---|---|
| `--questions FILE` | The question file (JSON Lines). Required. |
| `--collection NAME` | The Qdrant collection to search. Required; it must exist, with one unnamed vector of the dimension of the model (`RAG_ENGINE_DIM`), compared by cosine. |
| `--k N` | Number of documents the metrics look at. Default 10; at most `--chunk-limit`. A question with more requirements (essential documents and either-of groups) than `k` can never have them all: raise it for chain and aggregation questions. |
| `--chunk-limit N` | Number of chunks asked to the search per question. Default 50; at most `RAG_ENGINE_SEARCH_MAX_LIMIT`. |
| `--backend torch\|onnx` | Embedding backend. Default: `RAG_ENGINE_BACKEND`. |
| `--output FILE` | Also write the full report as JSON: parameters, metrics, every question with its first documents. |
| `--fail-under-hit-at-k X` | Exit with code 1 when the **headline** hit@k is below `X` (between 0 and 1): the hit@k of the kinds of question asked without filters, averaged per document. Not the pooled mean of all questions. |
| `--bootstrap-resamples N` | Number of times the documents are drawn again for the 95% intervals. Default 1000. |
| `--score-levels X ...` | Scores the documents returned for a question that expects no document are counted against. Default `0.7 0.75 0.8 0.85 0.9`. Reported, never used as a threshold. |
| `--allow-empty` | Run even when the collection holds none of the expected documents. Without it, that is an error (exit code 2). |

The command is the one place of the engine that reads the settings by itself (`Settings.from_env()`; a `.env` file is not read). It prints the report on the standard output. Exit codes: **0** when the evaluation ran, **1** when `--fail-under-hit-at-k` is given and not met, **2** when the evaluation could not run to its end (bad arguments or settings, an unusable question file, an unreachable server or a refused key, a missing collection, a collection of another dimension, of another distance than cosine or with named vectors, a collection that holds none of the expected documents, a missing embedding backend, a failure midway). `rag-engine --version` prints the version.

**Model files.** The first run with the `onnx` backend downloads about 1.1 GB (the pinned ONNX export and its tokenizer) from the Hugging Face Hub; the `torch` backend downloads the weights of the model, about as much, into the Hugging Face cache. Set `RAG_ENGINE_MODEL_CACHE` to a directory that stays: unset, the ONNX files go under the system's temporary directory and are downloaded again after a reboot. The later runs make no request to the Hub, and none to anyone else: the engine turns the telemetry of the ONNX runtime off by default (`ORT_DISABLE_TELEMETRY=1`, see [Embedding backends](#embedding-backends)). For a service, set `HF_HUB_OFFLINE=1` once the cache is filled.

What is cheap is checked before the embedding model is loaded: the question file, the server, the collection (`QdrantStore.check_collection`, which only reads: the dimension of its vectors against `RAG_ENGINE_DIM`, and their distance, which must be cosine — a Dot or Euclid collection is refused, its scores are not the cosines the report is about), then the presence of the expected documents. A collection in which none of them is found is almost always the wrong collection, so the command stops there instead of printing an empty report and exiting with 0; `--allow-empty` runs it all the same, with a warning.

**The question file** is JSON Lines, one question per line: `id` (unique), `question`, `expected_doc_ids` (the `doc_id` of every document that answers it, at least one), and optionally `filters` (an object with the field names and the rules of `SearchFilters`), `kind` and `tags` (labels the report groups by), `language` and `rationale` (kept, not used). It is checked strictly when read: a duplicate id, an empty `expected_doc_ids`, an unknown field or filter (`SearchFilters.from_mapping` reads the filters), a bad date, a question that holds nothing to search for once put in NFKD raise an `EvalError` naming the line. One question file is kept under [`eval/`](eval/README.md):

- `eval/known_item_v1.jsonl`: 797 automatic questions that ask for a document by its title or its catalogue label, or by its bank, type and date in English and in French, with and without filters. They are built from the corpus manifests by `eval/make_known_item.py`. The words follow what a doc type is at each bank, a date is never stated more precisely than the manifest knows it nor presented as a meeting date when it is a publication date, and a question expects every document that holds the text asked for (the ECB publishes its decision and its statement a second time as one combined document). [`eval/README.md`](eval/README.md) has the tables and the evidence.

A second file, `eval/analytical_v2.jsonl`, holds 91 questions on the content of the documents, written by hand and checked against them: chains over several meetings, comparisons between banks, buried details, false premises and 15 questions the corpus cannot answer. It uses the second version of the question format (essential and acceptable documents, either-of groups, expected behaviour, a second run without filters). `eval/README.md` says how the questions were checked and what was not, which numbers to read for this batch, and its measured limits (87 of the 91 questions name the bank, 65 pairs of scored questions share a required document).

**Format v2.** A question may say more, and a file that says none of it is read and scored exactly as before:

| Field | Meaning |
|---|---|
| `expected_behavior` | `"answer"` (the default), `"correct_premise"` (the premise is false and the documents show it) or `"abstain"` (the collection does not hold the information). |
| `essential_ids`, `acceptable_ids` | The documents the answer needs, and those that are relevant without being needed: two lists without a common document. `expected_doc_ids` is then all of them together: it may be left out, and if given must name the same documents. It may be empty only for a question that abstains. |
| `essential_any_of` | Groups of documents that carry the same fact (the decision release and the combined document of one day): the answer needs any one document of each group. A *requirement* is one essential document or one group. |
| `must_include`, `must_not_include` | Short facts a right answer holds, and typical false statements. **Carried, not scored**: the engine returns documents and writes no answer; they are validated and written into the JSON report for a scorer of the answers a client generates. |
| `run_unfiltered` | `true`: score the question a second time without its filters. Only on a question with filters. |

What these add to the scoring, all on documents and cut at `k`:

- **`all_essential@k`**, always the **share of questions** with every requirement satisfied among the first `k`, printed with its counts (`0.75 (3/4)`), and **`essential_recall@k`**, the share of the requirements satisfied, printed as a mean over the questions and pooled over the requirements; intervals on the kinds and the headline come from resampling the questions. hit@1, hit@k and MRR are unchanged: they look at the first expected document, essential or acceptable, so an acceptable document at rank 1 is a hit. A question that does not state its essential documents is left out of the two new metrics. A question with more requirements than `k` can never reach 1: the report counts and lists them (`unreachable`); raise `--k`, which `--chunk-limit` bounds.
- **The first table again per question.** Its lines are per-document means, in which a question that expects 44 documents weighs 44 times a question that expects one; a second table gives the plain means over the questions, and the report counts the questions that expect more documents than `k`, whose recall@k cannot reach 1.
- **A second run without filters** is reported under `<id>#unfiltered`, in a kind of its own (`<kind>#unfiltered`) that belongs to the headline, with the tag `unfiltered-rerun`; a table, *what the filter brings*, compares the two runs of these questions.
- **A question that abstains** is searched and left out of every mean, whether it lists documents or not. The documents it lists are context documents, which a system may show while it says that the corpus does not answer; none is essential (an essential document on such a question is refused when the file is read). The report lists the first document returned, its score, the number of documents at or above a few score levels and how many of the context documents are among the first `k`, next to the lowest, median and highest top score of the scored questions: material to study a threshold for abstention. No threshold is applied and nothing passes or fails.
- One line **per expected behaviour**: `correct_premise` questions are scored on their documents like any question; the questions that abstain have a count and no metric.

`rag-engine eval-check --questions FILE [--manifests GLOB ...] [--doc-types ...] [--mime ...] [--k N]` checks a question file without any collection, model or setting: it prints the counts per kind, tag, expected behaviour and language, the questions with filters, the questions that abstain, the distribution of requirements and of expected documents, the texts used twice, the questions with more requirements or more expected documents than `k`, and, with the manifests of the corpus, the documents named that are not in the catalogue (or not of the doc types and MIME types ingested). Exit code 0, or 2 when the file breaks a rule or names a document that is not found. [`eval/README.md`](eval/README.md) has a complete example line, every rule, and how to read the new parts of the report.

How to read the numbers of the known-item batch: it is a stratified sample, 12 documents per (bank, doc type) cell, so its averages weigh the 18 cells equally and are not weighted like the corpus; and all the documents are English, so its French questions measure cross-lingual retrieval.

**Scoring is done on documents.** The search returns chunks, and several may come from one document, while an expected answer is a document. For each question the search is asked for `chunk_limit` chunks, which are collapsed into documents: a document takes the place and the score of its best chunk. The metrics look at the first `k` documents of that ranking:

| Metric | Meaning |
|---|---|
| hit@1 | Share of questions whose first document is an expected one. |
| hit@k | Share of questions with an expected document among the first `k`. |
| MRR | Mean of `1 / rank` of the best-placed expected document; a rank beyond `k` counts for 0. |
| recall@k | Mean share of the expected documents found among the first `k`. |

**The report.** One line per `kind`, then the headline, then all questions pooled, then one line per tag:

```
                  questions  documents  hit@1              hit@10             MRR                recall@10
kind: filtered *          5          6  0.83 [0.50, 1.00]  0.83 [0.50, 1.00]  0.83 [0.50, 1.00]  0.67
kind: second *            1          1  0.00               1.00               0.50               1.00       (few documents)
kind: verbatim            6          4  0.92 [0.75, 1.00]  1.00 [1.00, 1.00]  0.96 [0.88, 1.00]  1.00
headline                  6          4  0.92 [0.75, 1.00]  1.00 [1.00, 1.00]  0.96 [0.88, 1.00]  1.00
overall (pooled)         12         11  0.75               0.92               0.83               0.88
```

(The test fixture of `eval/fixtures/`, with the minimum number of documents for an interval lowered to 4. No measurement of the production collection is recorded here.)

- Every line gives the number of questions and of **distinct expected documents**. The lines of the kinds, of the headline and of the tags are **per-document means**: the questions that expect one document are averaged first, then the documents, because the questions of one document are correlated.
- `[low, high]` is a 95% interval of that mean, from 1000 resamples of the documents with a fixed seed (`--bootstrap-resamples`; standard library `random`, so the same results always give the same interval). A line with fewer than 20 documents gets none and is marked `(few documents)`. The kinds and the headline have intervals, the tags do not.
- **`headline`** is the kinds asked without filters, per document. It is what `--fail-under-hit-at-k` looks at.
- A kind marked `*` is asked with filters and kept out of the headline: where the filters leave fewer than `k` documents, hit@k and recall@k are true by construction, and only hit@1 and MRR are informative.
- **`overall (pooled)`** is the plain mean over all scored questions, every kind, each question counted once. It is kept, and labelled as pooled in the JSON as well; it is not the headline.
- Rates are printed to two decimals, which is what their uncertainty allows; the JSON report holds the exact values.

The questions without any expected document among the first `k` are listed as **misses**, with the three best documents returned and their titles.

**Equal scores.** Two chunks with the same text have exactly the same score, and Qdrant returns them in an order that can change from one request to the next. The chunks returned are put in one order (score, then `doc_id`, then `chunk_number`) before they become documents, so the same chunks always give the same ranking. A question whose hit@1 or hit@k hangs on the order of equal scores is marked `tied_at_decision`, and the report counts them. Ties at the edge of the chunk budget stay outside the harness: which of the tied chunks Qdrant returns is its choice.

`chunk_limit` chunks may hold fewer than `k` documents, when the best chunks come from a few long documents. The report does not hide it: it counts the questions for which fewer than `k` documents were ranked and, among them, those whose search returned the whole chunk budget, for which a larger `--chunk-limit` could have ranked more. The budget is bounded by the search maximum (`RAG_ENGINE_SEARCH_MAX_LIMIT`, default 50), so that the evaluation measures the search a caller gets; raise both to look deeper.

**Documents that are not in the collection.** Before any search, the expected documents of all the questions are looked up in the collection (`QdrantStore.existing_doc_ids`, a filter on `doc_id`). A question none of whose expected documents is there is reported under `not_in_collection` and left out of every metric: counting it as a miss would blame the search for a document that was never ingested. A question with only some of its documents absent is scored; the absent ones are listed, and its recall is the share of the documents that are there.

From Python, with a store and an embedder the caller already has:

```python
from rag_engine.eval import evaluate, format_report, load_questions

questions = load_questions("eval/known_item_v1.jsonl")
report = evaluate(questions, collection="documents", store=store, embedder=embedder, k=10, chunk_limit=50)
print(format_report(report))
report.headline.per_document.hit_at_k  # the headline; also .hit_at_1, .mrr, .recall_at_k
report.headline.intervals  # {"hit_at_1": (low, high), "hit_at_k": ..., "mrr": ...}, or None with few documents
report.by_kind, report.by_tag, report.overall  # an Aggregate each; overall is pooled over the questions
report.tied_at_decision  # questions whose hit@1 or hit@k hangs on the order of equal scores
report.misses  # the questions without an expected document among the first k
report.headline.essential  # all_essential_at_k (share of questions), essential_recall_at_k, pooled_..., question_intervals
report.unfiltered_reruns, report.filter_effect  # the second runs without filters, and the comparison
report.abstentions, report.top_scores  # the questions that abstain; the top scores to compare with
report.by_behavior  # an Aggregate per expected_behavior
report.to_json()  # everything, as JSON
```

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

The default suite also runs the evaluation harness end to end, in qdrant-client's local mode, on the fixture of `eval/fixtures/` (13 documents; 13 questions with known answers in the first version of the question format, 11 more that use every field of the second), and checks the committed question file `eval/known_item_v1.jsonl`.

The tests do not read the settings of the host: `tests/conftest.py` takes `QDRANT_*`, `RAG_ENGINE_*`, `ENABLE_OCR` and `TESSDATA_PREFIX` out of the environment for the run (and puts them back afterwards), so a `QDRANT_PORT` or a `RAG_ENGINE_BACKEND` set for other work changes nothing. Only the variables meant for the tests get through, to the tests of their marker: `RAG_ENGINE_IT_QDRANT_URL` and `RAG_ENGINE_IT_QDRANT_API_KEY` (`integration`), `RAG_ENGINE_MODEL_CACHE` and `RAG_ENGINE_PARITY_MIN` (`models`). The caches of the model libraries (`HF_HOME`, `FASTEMBED_CACHE_PATH`) are left alone. The tests do not depend on their order either (`pytest -p randomly` with `pytest-randomly` installed).

Two test markers are deselected by default:

- `integration`: needs a real Qdrant, given by `RAG_ENGINE_IT_QDRANT_URL`.
- `models`: downloads embedding models. It holds the test of the real torch model, the parity test of the two backends and the checks of the pinned model files (`tests/test_parity.py`); it needs `pip install -e ".[dev,torch,onnx]"`, and about 1.2 GB of free space in the temporary directory for the altered copy of the ONNX file, which the test removes when it ends. CI runs it in its own job, `parity`, with the model files cached under a key that carries the pinned revision.

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:v1.20.0
RAG_ENGINE_IT_QDRANT_URL=http://localhost:6333 pytest -m integration -q
pytest -m models -q -s  # -s prints the measured parity
```

The integration tests of ingestion parse real documents, so they also need the `ingest` extra and skip without it; the CI integration job installs it and fails if the parsers cannot be imported.

## Licence

MIT, see [LICENSE](LICENSE). Portions are derived from eigenmind (MIT); see [NOTICE](NOTICE).
