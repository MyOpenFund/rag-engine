"""Loads the evaluation fixture of ``eval/fixtures/`` into a collection, for the tests.

``corpus.json`` holds thirteen small documents (thirty-two chunks) and the model the fixture is
meant for: a ``FakeEmbedder`` whose passage and query prefixes are both empty, so that a
question whose text is the text of a chunk gets the vector of that chunk. The unit tests
(qdrant-client's local mode) and the integration test (a real server) load the same points.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qdrant_client import models

from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.schema import build_payload, page_for_id, point_id

FIXTURES = Path(__file__).resolve().parent.parent / "eval" / "fixtures"
CORPUS = FIXTURES / "corpus.json"
QUESTIONS = FIXTURES / "questions.jsonl"
INGESTED_AT = "2026-11-17T10:00:00"


def fixture_corpus() -> dict[str, Any]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def fixture_spec() -> ModelSpec:
    return ModelSpec(**fixture_corpus()["model"])


def fixture_embedder() -> FakeEmbedder:
    return FakeEmbedder(fixture_spec())


def fixture_points(embedder: FakeEmbedder | None = None) -> list[models.PointStruct]:
    """The corpus as schema v1 points, one per chunk, with the vectors of a ``FakeEmbedder``."""
    embedder = embedder if embedder is not None else fixture_embedder()
    points = []
    for document in fixture_corpus()["documents"]:
        doc_id, texts = document["doc_id"], document["chunks"]
        for index, vector in enumerate(embedder.encode_passage(texts)):
            payload = build_payload(
                doc_id,
                document["metadata"],
                filename=f"{doc_id}.pdf",
                chunk_number=index,
                ingestion_date=INGESTED_AT,
                text=texts[index],
                page=index,
            )
            points.append(
                models.PointStruct(
                    id=point_id(doc_id, page_for_id(index), index), vector=vector.tolist(), payload=payload
                )
            )
    return points


def load_fixture(store: Any, collection: str) -> int:
    """Create ``collection`` and write the fixture corpus into it; returns the number of points."""
    store.ensure_collection(collection, fixture_spec().dim)
    return store.upsert(collection, fixture_points())
