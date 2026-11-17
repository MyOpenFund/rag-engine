# Changelog

All notable changes to rag-engine are recorded here. Versions follow [semantic versioning](https://semver.org).

## 0.1.0 — Unreleased

First version: ingestion, the two embedding backends, and filtered search.

### Delivered

- **Settings** (`rag_engine.config`): `Settings.from_env()` reads the environment when it is called, never at import; every value is validated, and the Qdrant API key is never printed.
- **Payload schema v1 and point ids** (`rag_engine.schema`): the payload and the deterministic ids the data-orchestrator already writes. An id is derived from the `doc_id`, the page and the chunk index only.
- **Qdrant store** (`rag_engine.store`): connection, strict collection check or creation (`ensure_collection`), batched writes that report how many points were written before a failure, authentication refusals reported as `QdrantAuthError`.
- **Payload indexes** (`rag_engine.schema.PAYLOAD_INDEXES`, `QdrantStore.ensure_payload_indexes`): keyword indexes on `doc_id`, `source_code`, `doc_type`, `language` and `corpus`, an integer index on `year`, a datetime index on `date`. They are what the search filters rely on. The call creates the missing indexes only, can be repeated, works on a collection that already holds points without rewriting any, and refuses with `CollectionMismatchError` a field already indexed with another kind instead of replacing that index. A `date` that is not a full date (`2016-03`, `2016`, empty, null, absent) is never an error; such a point matches no date range.
- **Embedders** (`rag_engine.embedding`): the `Embedder` interface, the PyTorch backend (`TorchEmbedder`, extra `torch`), the ONNX backend (`OnnxEmbedder`, extra `onnx`: fastembed on the ONNX runtime, without PyTorch), and a deterministic `FakeEmbedder` for tests. The passage and query prefixes of the model are applied in one place.
- **Backend parity**: `RAG_ENGINE_BACKEND=onnx` is meant for a service that embeds search queries against passages embedded with `torch`. A test (marker `models`, CI job `parity`) compares the two backends on 50 fixed multilingual texts and on a small search; it fails below `RAG_ENGINE_PARITY_MIN` (new setting, default 0.999). Measured for `intfloat/multilingual-e5-base`: minimum cosine above 0.99999999 for passages and for queries, identical top-5 rankings. `rag_engine.embedding.cosine_parity` is the measure.
- **Chunking** (`rag_engine.chunking`): `chunk_file` for PDF, DOCX, XLSX, CSV, PPTX, TXT and MD (extra `ingest`), with an explicit OCR mode (`auto`, `always`, `never`), explicit OCR languages and explicit chunker options.
- **Ingestion** (`rag_engine.ingest`): `ingest_document` chunks one file, embeds its chunks and writes one point per chunk. For the same file and metadata it writes exactly the points the previous pipeline wrote (ids, payloads, vectors): a golden fixture captured from that pipeline pins it.
- **Search** (`rag_engine.search`): `search(query, *, collection, store, embedder, filters=None, limit=None, max_limit=None, normalize_query=True)` embeds the query and returns the closest chunks as `Hit` objects, best first, in one request (`QdrantStore.query`).
  - `SearchFilters`: `source_codes`, `doc_types`, `languages`, `doc_ids` (lists), `date_from`, `date_to`, `corpus`. Fields are combined with AND, the values of a list with OR. Checked at construction: a bare string instead of a list and an empty list are refused (`None` is "no restriction"); an unknown value simply returns nothing.
  - Dates: both bounds inclusive, whole days in UTC; the upper bound is sent as "before the next day", so a document stored with a time of day stays in its day. A bound is a `datetime.date` or a `YYYY-MM-DD` string and nothing else: `"2016"` or `"2016-03"` raise a `SearchError` instead of reaching qdrant-client, which reads `"2016"` as a Unix timestamp. A chunk without a full `date` is left out as soon as a bound is set.
  - `limit` defaults to 10 and may not exceed `max_limit` (default 50): above it the search is refused with an error naming the maximum, never capped silently. `search` reads no settings; the caller passes `settings.search_limit` and `settings.search_max_limit`.
  - The query is put in Unicode NFKD before it is embedded (`normalize_query=True`), the form the chunker stores and embeds the chunks in. Checked on the tokenizer of `intfloat/multilingual-e5-base`: no effect in French, German, Spanish or Greek; in Japanese (voiced kana), Korean and Vietnamese only the NFKD query is cut into the tokens of the stored passages.
  - `Hit`: `text`, `score`, `doc_id`, `title`, `date`, `source_code`, `doc_type`, `language`, `page` (`None` without pages), `chunk_number`, `corpus`, plus `point_id` and the full `payload`; `to_dict()` for JSON.
- **Errors** (`rag_engine.errors`): every exception class of the engine derives from `RagEngineError`. `ConfigError`, `UnsupportedFormatError` and `SearchError` are also `ValueError`.
- **Packaging**: light base install (`qdrant-client>=1.10`, the first release with the query API; `numpy`), optional extras, `py.typed` marker, `__all__` in every module. The package root exposes `__version__` only.

### Differences from the previous engine

For a valid document and valid metadata the points written are the same. What differs:

- **OCR languages are explicit.** The previous engine used every language found in the tessdata directory. The languages are now a value the caller passes (`ocr_languages`, from `RAG_ENGINE_OCR_LANGUAGES`, default `eng`). To keep the previous result on a host with several languages installed, list them, for example `eng+fra`.
- **Invalid UTF-8 is replaced, not dropped.** In a `.txt` or `.md` file, each invalid byte sequence becomes `�` (U+FFFD) and a warning is logged; the previous engine dropped those bytes silently. The chunks that contain such bytes differ from the ones ingested before.
- **OCR switches are read at each call.** `ENABLE_OCR` and `TESSDATA_PREFIX` are read each time a file is chunked; the previous engine read `ENABLE_OCR` once, at import. With `ocr="always"`, a `TESSDATA_PREFIX` set to the parent of a `tessdata` folder is now corrected too, whatever `ENABLE_OCR` says.
- **Reserved payload keys in the caller's metadata are refused.** Metadata that sets `filename`, `chunk_number`, `ingestion_date`, `text` or `page`, or a `doc_id` different from the document's, raises `ValueError` before the file is read. The previous pipeline accepted such metadata without a word: its own `filename`, `chunk_number`, `ingestion_date`, `text` and `page` replaced the caller's, and a `doc_id` in the metadata replaced the document's in the payload. A `doc_id` equal to the document's id is accepted, as the vault source sends it.
