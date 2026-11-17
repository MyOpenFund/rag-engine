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
| `rag-engine[torch]` | `sentence-transformers` | PyTorch embedding backend |
| `rag-engine[onnx]` | `fastembed` 0.9.x (ONNX runtime, `huggingface_hub`) | ONNX embedding backend, without PyTorch: for a service that embeds search queries |
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
| `rag_engine.embedding` | `ModelSpec`, `Embedder`, `make_embedder`, `TorchEmbedder`, `OnnxEmbedder`, `FakeEmbedder`, `BaseEmbedder`, `detect_device`, `cosine_parity`, `ONNX_MODELS`, `OnnxModel`, `ONNX_BATCH_SIZE`, `ModelIntegrityError`, `ModelDownloadError` |
| `rag_engine.chunking` | `chunk_file`, `Chunk`, `ChunkingOptions`, `UnsupportedFormatError`, `ocr_available`, `check_ocr_languages`, `SUPPORTED_EXTENSIONS`, `OCR_MODES` |
| `rag_engine.ingest` | `ingest_document`, `IngestResult` |
| `rag_engine.search` | `search`, `SearchFilters`, `Hit`, `SearchError`, `QUERY_NORMALIZATION` |
| `rag_engine.eval` | `load_questions`, `evaluate`, `format_report`, `Question`, `QuestionSet`, `EvalReport`, `QuestionResult`, `RankedDocument`, `NotInCollection`, `Aggregate`, `EvalError` |
| `rag_engine.cli` | `main` (the `rag-engine` command) |
| `rag_engine.errors` | `RagEngineError` |

Errors. Every exception class the engine defines derives from `rag_engine.errors.RagEngineError`: `ConfigError` (also a `ValueError`), `StoreError` with its subclasses `QdrantAuthError` and `CollectionMismatchError`, `UnsupportedFormatError` (also a `ValueError`), `SearchError` (also a `ValueError`) and `EvalError` (also a `ValueError`). `except RagEngineError` does not catch what the engine lets through unchanged (an error of a parser, of the embedding backend or of the Qdrant client), nor the plain `ValueError` / `TypeError` of a bad argument.

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

`Settings(...)` can also be built directly with keyword arguments. The same range checks apply, but nothing is parsed or converted: integers must be `int` (not `bool`, `float` or a numeric string), `qdrant_https` must be a `bool`, `parity_min` must be a `float`, `backend` must be exactly `"torch"` or `"onnx"`, and a prefix string is the prefix as it is (`""` for no prefix). An empty `qdrant_host` or `model_name` is refused; a blank `qdrant_api_key` or `model_cache` becomes `None`. A wrong type raises `ConfigError`.

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

  The engine fetches these files at that revision itself (`huggingface_hub.snapshot_download`) and hands the directory to fastembed, which downloads nothing. **Before the model is built, every file is hashed and compared with the table**; a different size or hash raises `ModelIntegrityError`, naming the file and the expected and found hashes. A changed export or tokenizer can therefore not reach production unnoticed, whatever the head of the repository holds that day. Hashing 1.1 GB takes about one second; it is done once per process and file (kept for the same path, size and modification time). An `OnnxModel` cannot be written without a pin for the ONNX file and for each of the four other files (`files` maps a path in the repository to `(sha256, size)`; `sha256` and `size` are those of the ONNX file). To change a file, re-measure parity first, then update the table.

- **Other models.** A model that is not in the table but that fastembed lists is used as fastembed defines and downloads it, with a warning in the log: its files are not pinned or verified by the engine, and its parity was not measured. Any other name raises a `ValueError`.
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
store.ensure_collection(collection, embedder.dim)  # once at start: refuses another dimension
store.ensure_payload_indexes(collection)  # once: the filters are fast with the indexes

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

With an API key that only reads (a search service), leave out the two `ensure_` calls: they are for whoever owns the collection. `search` itself sends one request, and does not check the collection: a collection that does not exist or has another dimension is the error of the Qdrant client.

**Hits.** A `Hit` is a frozen dataclass: `text`, `score` (the cosine, higher is closer), `doc_id`, `title`, `date`, `source_code`, `doc_type`, `language`, `page`, `chunk_number`, `corpus`, then `point_id` and `payload`, the whole payload for the keys that have no field (`year`, `filename`, `ingestion_date`, ...). A key the payload does not have gives `None`; `page` is `None` for a document without pages (and `0` is the first page). `date` is the publication date exactly as stored. `hit.to_dict()` returns the fields without `payload`, ready for `json.dumps`; `to_dict(include_payload=True)` adds it.

**Limits.** `limit` is the largest number of hits wanted, `max_limit` the largest limit accepted. A `limit` above `max_limit` is refused with a `SearchError` that names the maximum; it is never lowered silently.

**Filters.** A `SearchFilters` is built from keyword arguments, all optional, and checked when it is built:

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

**Errors.** `SearchError` (a `ValueError` and a `RagEngineError`) for a bad argument or filter, raised before anything is embedded or sent. A refused API key is a `QdrantAuthError`. Everything else — an error of the embedder, a missing collection, an unreachable server — reaches the caller unchanged.

**Versions.** Search uses Qdrant's query API: qdrant-client 1.10 or later, and a Qdrant server 1.10 or later.

## Evaluating search quality

`rag-engine eval` runs the search on a file of questions whose answers are known, and reports how well the expected documents are ranked.

```bash
export QDRANT_HOST=localhost QDRANT_PORT=6333   # the connection and the model come from the environment
export RAG_ENGINE_BACKEND=onnx                  # the model must be the one the collection was embedded with
rag-engine eval --questions eval/known_item_v1.jsonl --collection documents
rag-engine eval --questions eval/known_item_v1.jsonl --collection documents \
    --k 10 --chunk-limit 50 --backend torch --output report.json --fail-under-hit-at-k 0.9
```

| Option | Meaning |
|---|---|
| `--questions FILE` | The question file (JSON Lines). Required. |
| `--collection NAME` | The Qdrant collection to search. Required; it must exist, with vectors of the dimension of the model. |
| `--k N` | Number of documents the metrics look at. Default 10. |
| `--chunk-limit N` | Number of chunks asked to the search per question. Default 50; at most `RAG_ENGINE_SEARCH_MAX_LIMIT`. |
| `--backend torch\|onnx` | Embedding backend. Default: `RAG_ENGINE_BACKEND`. |
| `--output FILE` | Also write the full report as JSON: parameters, metrics, every question with its first documents. |
| `--fail-under-hit-at-k X` | Exit with code 1 when the overall hit@k is below `X` (between 0 and 1). |

The command is the one place of the engine that reads the settings by itself (`Settings.from_env()`; a `.env` file is not read). It prints the report on the standard output. Exit codes: **0** when the evaluation ran, **1** when `--fail-under-hit-at-k` is given and not met, **2** when the evaluation could not run to its end (bad arguments or settings, an unusable question file, an unreachable server or a refused key, a missing collection, a missing embedding backend, a failure midway). `rag-engine --version` prints the version.

**The question file** is JSON Lines, one question per line: `id` (unique), `question`, `expected_doc_ids` (the `doc_id` of every document that answers it, at least one), and optionally `filters` (an object with the field names and the rules of `SearchFilters`), `kind` and `tags` (labels the report groups by), `language` and `rationale` (kept, not used). It is checked strictly when read: a duplicate id, an empty `expected_doc_ids`, an unknown field or filter, a bad date raise an `EvalError` naming the line. Two files are kept under [`eval/`](eval/README.md):

- `eval/known_item_v1.jsonl`: 833 automatic questions that ask for a document by its title, or by its bank, type and date in English and in French, with and without filters. They are built from the corpus manifests by `eval/make_known_item.py`, which never states a date more precisely than the manifest knows it.
- `eval/analytical_v1.jsonl`: questions on the content of the documents, written by hand; the file is added once its questions are validated.

**Scoring is done on documents.** The search returns chunks, and several may come from one document, while an expected answer is a document. For each question the search is asked for `chunk_limit` chunks, which are collapsed into documents: a document takes the place and the score of its best chunk. The metrics look at the first `k` documents of that ranking:

| Metric | Meaning |
|---|---|
| hit@1 | Share of questions whose first document is an expected one. |
| hit@k | Share of questions with an expected document among the first `k`. |
| MRR | Mean of `1 / rank` of the best-placed expected document; a rank beyond `k` counts for 0. |
| recall@k | Mean share of the expected documents found among the first `k`. |

They are given overall, per `kind` and per tag. The questions without any expected document among the first `k` are listed as **misses**, with the three best documents returned and their titles.

`chunk_limit` chunks may hold fewer than `k` documents, when the best chunks come from a few long documents. The report does not hide it: it counts the questions for which fewer than `k` documents were ranked and, among them, those whose search returned the whole chunk budget, for which a larger `--chunk-limit` could have ranked more. The budget is bounded by the search maximum (`RAG_ENGINE_SEARCH_MAX_LIMIT`, default 50), so that the evaluation measures the search a caller gets; raise both to look deeper.

**Documents that are not in the collection.** Before any search, the expected documents of all the questions are looked up in the collection (`QdrantStore.existing_doc_ids`, a filter on `doc_id`). A question none of whose expected documents is there is reported under `not_in_collection` and left out of every metric: counting it as a miss would blame the search for a document that was never ingested. A question with only some of its documents absent is scored; the absent ones are listed, and its recall is the share of the documents that are there.

From Python, with a store and an embedder the caller already has:

```python
from rag_engine.eval import evaluate, format_report, load_questions

questions = load_questions("eval/known_item_v1.jsonl")
report = evaluate(questions, collection="documents", store=store, embedder=embedder, k=10, chunk_limit=50)
print(format_report(report))
report.overall.hit_at_k  # also .hit_at_1, .mrr, .recall_at_k; report.by_kind, report.by_tag
report.misses  # the questions without an expected document among the first k
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

The default suite also runs the evaluation harness end to end, in qdrant-client's local mode, on the fixture of `eval/fixtures/` (12 documents, 10 questions with known answers), and checks the committed question file `eval/known_item_v1.jsonl`.

Two test markers are deselected by default:

- `integration`: needs a real Qdrant, given by `RAG_ENGINE_IT_QDRANT_URL`.
- `models`: downloads embedding models. It holds the test of the real torch model, the parity test of the two backends and the checks of the pinned model files (`tests/test_parity.py`); it needs `pip install -e ".[dev,torch,onnx]"`, and about 1.2 GB of free space in the temporary directory for the altered copy of the ONNX file. CI runs it in its own job, `parity`, with the model files cached under a key that carries the pinned revision.

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:v1.20.0
RAG_ENGINE_IT_QDRANT_URL=http://localhost:6333 pytest -m integration -q
pytest -m models -q -s  # -s prints the measured parity
```

The integration tests of ingestion parse real documents, so they also need the `ingest` extra and skip without it; the CI integration job installs it and fails if the parsers cannot be imported.

## Licence

MIT, see [LICENSE](LICENSE). Portions are derived from eigenmind (MIT); see [NOTICE](NOTICE).
