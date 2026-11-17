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
