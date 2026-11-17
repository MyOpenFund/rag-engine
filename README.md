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
| `rag-engine[ingest]` | `chunknorris`, `markitdown[pptx]` | document parsing and chunking |
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
| `RAG_ENGINE_PASSAGE_PREFIX` | `passage: ` (with the trailing space) | Text put before each chunk when embedding |
| `RAG_ENGINE_QUERY_PREFIX` | `query: ` (with the trailing space) | Text put before each query when embedding |
| `RAG_ENGINE_BACKEND` | `torch` | Embedding backend: `torch` or `onnx` |
| `RAG_ENGINE_MODEL_CACHE` | none | Directory for model files; unset leaves the backend default |
| `RAG_ENGINE_BATCH_SIZE` | `100` | Chunks embedded and written per batch (at least 1) |
| `RAG_ENGINE_OCR_LANGUAGES` | `eng` | Tesseract languages joined by `+`, e.g. `eng+fra` |
| `RAG_ENGINE_SEARCH_LIMIT` | `10` | Hits returned when no limit is given |
| `RAG_ENGINE_SEARCH_MAX_LIMIT` | `50` | Largest limit a caller may ask for; not below the default limit |

An unset or blank variable gives the default. The two prefixes are the exception: they are never stripped, and a prefix set to the empty string means "no prefix". An invalid value raises `ConfigError` (a `ValueError`) naming the variable and the value.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

Two test markers are deselected by default:

- `integration`: needs a real Qdrant, given by `RAG_ENGINE_IT_QDRANT_URL`.
- `models`: downloads embedding models.

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:v1.20.0
RAG_ENGINE_IT_QDRANT_URL=http://localhost:6333 pytest -m integration -q
```

## Licence

MIT, see [LICENSE](LICENSE). Portions are derived from eigenmind (MIT); see [NOTICE](NOTICE).
