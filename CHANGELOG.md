# Changelog

All notable changes to rag-engine are recorded here. Versions follow [semantic versioning](https://semver.org).

## 0.1.0 — Unreleased

First version: ingestion, and the two embedding backends. Search is not part of it yet.

### Delivered

- **Settings** (`rag_engine.config`): `Settings.from_env()` reads the environment when it is called, never at import; every value is validated, and the Qdrant API key is never printed.
- **Payload schema v1 and point ids** (`rag_engine.schema`): the payload and the deterministic ids the data-orchestrator already writes. An id is derived from the `doc_id`, the page and the chunk index only.
- **Qdrant store** (`rag_engine.store`): connection, strict collection check or creation (`ensure_collection`), batched writes that report how many points were written before a failure, authentication refusals reported as `QdrantAuthError`.
- **Embedders** (`rag_engine.embedding`): the `Embedder` interface, the PyTorch backend (`TorchEmbedder`, extra `torch`), the ONNX backend (`OnnxEmbedder`, extra `onnx`: fastembed on the ONNX runtime, without PyTorch), and a deterministic `FakeEmbedder` for tests. The passage and query prefixes of the model are applied in one place.
- **Backend parity**: `RAG_ENGINE_BACKEND=onnx` is meant for a service that embeds search queries against passages embedded with `torch`. A test (marker `models`, CI job `parity`) compares the two backends on 50 fixed multilingual texts and on a small search; it fails below `RAG_ENGINE_PARITY_MIN` (new setting, default 0.999). Measured for `intfloat/multilingual-e5-base`: minimum cosine above 0.99999999 for passages and for queries, identical top-5 rankings. `rag_engine.embedding.cosine_parity` is the measure.
- **Chunking** (`rag_engine.chunking`): `chunk_file` for PDF, DOCX, XLSX, CSV, PPTX, TXT and MD (extra `ingest`), with an explicit OCR mode (`auto`, `always`, `never`), explicit OCR languages and explicit chunker options.
- **Ingestion** (`rag_engine.ingest`): `ingest_document` chunks one file, embeds its chunks and writes one point per chunk. For the same file and metadata it writes exactly the points the previous pipeline wrote (ids, payloads, vectors): a golden fixture captured from that pipeline pins it.
- **Errors** (`rag_engine.errors`): every exception class of the engine derives from `RagEngineError`. `ConfigError` and `UnsupportedFormatError` are also `ValueError`.
- **Packaging**: light base install (`qdrant-client>=1.8`, `numpy`), optional extras, `py.typed` marker, `__all__` in every module. The package root exposes `__version__` only.

### Not available yet

- Search: no search function; `RAG_ENGINE_SEARCH_LIMIT` and `RAG_ENGINE_SEARCH_MAX_LIMIT` are read and checked but not used.

### Differences from the previous engine

For a valid document and valid metadata the points written are the same. What differs:

- **OCR languages are explicit.** The previous engine used every language found in the tessdata directory. The languages are now a value the caller passes (`ocr_languages`, from `RAG_ENGINE_OCR_LANGUAGES`, default `eng`). To keep the previous result on a host with several languages installed, list them, for example `eng+fra`.
- **Invalid UTF-8 is replaced, not dropped.** In a `.txt` or `.md` file, each invalid byte sequence becomes `�` (U+FFFD) and a warning is logged; the previous engine dropped those bytes silently. The chunks that contain such bytes differ from the ones ingested before.
- **OCR switches are read at each call.** `ENABLE_OCR` and `TESSDATA_PREFIX` are read each time a file is chunked; the previous engine read `ENABLE_OCR` once, at import. With `ocr="always"`, a `TESSDATA_PREFIX` set to the parent of a `tessdata` folder is now corrected too, whatever `ENABLE_OCR` says.
- **Reserved payload keys in the caller's metadata are refused.** Metadata that sets `filename`, `chunk_number`, `ingestion_date`, `text` or `page`, or a `doc_id` different from the document's, raises `ValueError` before the file is read. The previous pipeline accepted such metadata without a word: its own `filename`, `chunk_number`, `ingestion_date`, `text` and `page` replaced the caller's, and a `doc_id` in the metadata replaced the document's in the payload. A `doc_id` equal to the document's id is accepted, as the vault source sends it.
