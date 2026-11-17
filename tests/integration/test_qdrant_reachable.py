"""Smoke test: the Qdrant given by RAG_ENGINE_IT_QDRANT_URL answers."""

import os

import pytest
from qdrant_client import QdrantClient

pytestmark = pytest.mark.integration

QDRANT_URL = os.environ.get("RAG_ENGINE_IT_QDRANT_URL", "").strip()


@pytest.mark.skipif(not QDRANT_URL, reason="RAG_ENGINE_IT_QDRANT_URL is not set")
def test_qdrant_reachable():
    client = QdrantClient(url=QDRANT_URL)
    try:
        assert client.get_collections().collections is not None
    finally:
        client.close()
