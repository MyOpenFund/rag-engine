"""The small corpus and the filter matrix shared by the search tests.

The unit tests (qdrant-client's local mode) and the integration tests (a real server) ask
the same questions and expect the same documents. ``conftest.py`` runs a test that takes a
``search_case`` argument once per entry of ``SEARCH_CASES``.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import models

from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.schema import build_payload, page_for_id, point_id

SEARCH_SPEC = ModelSpec(name="fake", dim=8, passage_prefix="passage: ", query_prefix="query: ")
SEARCH_WHEN = "2026-11-17T10:00:00"

# doc_id -> (metadata, pages of its chunks). What each document is there for:
#   ecb-a    dated 2016-03-10 without time, two chunks on two pages
#   ecb-b    dated the same day with a time of day: the inclusive upper bound must keep it
#   fed-a    the day before; fed-b the day after, in French
#   snb-a    an earlier year, in German, three chunks
#   boj-a    no date key, no page (an unpaginated document), no title
#   boj-b    date and year null
#   boe-a    an imprecise date ("2016-03"), which no date range matches
#   press-a  another corpus, with a source_code the central-bank corpus has too
SEARCH_DOCUMENTS: dict[str, tuple[dict[str, Any], list[int | None]]] = {
    "ecb-a": (
        {
            "corpus": "central-bank",
            "source_code": "ecb",
            "doc_type": "statement",
            "title": "Monetary policy statement",
            "date": "2016-03-10",
            "year": 2016,
            "language": "en",
        },
        [0, 1],
    ),
    "ecb-b": (
        {
            "corpus": "central-bank",
            "source_code": "ecb",
            "doc_type": "minutes",
            "title": "Account of the meeting",
            "date": "2016-03-10T15:30:00Z",
            "year": 2016,
            "language": "en",
        },
        [0],
    ),
    "fed-a": (
        {
            "corpus": "central-bank",
            "source_code": "fed",
            "doc_type": "statement",
            "title": "FOMC statement",
            "date": "2016-03-09",
            "year": 2016,
            "language": "en",
        },
        [0],
    ),
    "fed-b": (
        {
            "corpus": "central-bank",
            "source_code": "fed",
            "doc_type": "minutes",
            "title": "Minutes",
            "date": "2016-03-11",
            "year": 2016,
            "language": "fr",
        },
        [0, 2],
    ),
    "snb-a": (
        {
            "corpus": "central-bank",
            "source_code": "snb",
            "doc_type": "speech",
            "title": "Geldpolitische Lagebeurteilung",
            "date": "2015-06-18",
            "year": 2015,
            "language": "de",
        },
        [0, 1, 2],
    ),
    "boj-a": (
        {"corpus": "central-bank", "source_code": "boj", "doc_type": "statement", "language": "ja"},
        [None],
    ),
    "boj-b": (
        {
            "corpus": "central-bank",
            "source_code": "boj",
            "doc_type": "minutes",
            "title": "Minutes of the meeting",
            "date": None,
            "year": None,
            "language": "ja",
        },
        [0],
    ),
    "boe-a": (
        {
            "corpus": "central-bank",
            "source_code": "boe",
            "doc_type": "report",
            "title": "Inflation report",
            "date": "2016-03",
            "year": 2016,
            "language": "en",
        },
        [0],
    ),
    "press-a": (
        {
            "corpus": "press",
            "source_code": "ecb",
            "doc_type": "article",
            "title": "An article",
            "date": "2017-01-01",
            "year": 2017,
            "language": "en",
        },
        [None],
    ),
}
ALL_DOCUMENTS = frozenset(SEARCH_DOCUMENTS)
SEARCH_POINT_COUNT = sum(len(pages) for _metadata, pages in SEARCH_DOCUMENTS.values())


def search_chunk_text(doc_id: str, index: int) -> str:
    return f"chunk {index} of {doc_id}"


def search_points(embedder: FakeEmbedder | None = None) -> list[models.PointStruct]:
    """The corpus as schema v1 points, one per chunk, with the vectors of a ``FakeEmbedder``."""
    embedder = embedder if embedder is not None else FakeEmbedder(SEARCH_SPEC)
    points = []
    for doc_id, (metadata, pages) in SEARCH_DOCUMENTS.items():
        texts = [search_chunk_text(doc_id, index) for index in range(len(pages))]
        for index, (page, vector) in enumerate(zip(pages, embedder.encode_passage(texts))):
            payload = build_payload(
                doc_id,
                metadata,
                filename=f"{doc_id}.pdf",
                chunk_number=index,
                ingestion_date=SEARCH_WHEN,
                text=texts[index],
                page=page,
            )
            points.append(
                models.PointStruct(
                    id=point_id(doc_id, page_for_id(page), index), vector=vector.tolist(), payload=payload
                )
            )
    return points


@dataclass(frozen=True)
class SearchCase:
    """One question of the matrix: the filters, and the documents whose chunks must come back."""

    label: str
    filters: dict[str, Any]
    documents: frozenset[str] = field(default_factory=frozenset)


def _case(label: str, filters: dict[str, Any], *documents: str) -> SearchCase:
    return SearchCase(label, filters, frozenset(documents))


DATED = ("ecb-a", "ecb-b", "fed-a", "fed-b", "snb-a", "press-a")

SEARCH_CASES = [
    _case("no filter", {}, *ALL_DOCUMENTS),
    # Each field alone.
    _case("one source", {"source_codes": ["fed"]}, "fed-a", "fed-b"),
    _case("one doc type", {"doc_types": ["minutes"]}, "ecb-b", "fed-b", "boj-b"),
    _case("one language", {"languages": ["ja"]}, "boj-a", "boj-b"),
    _case("one document", {"doc_ids": ["ecb-a"]}, "ecb-a"),
    _case("corpus", {"corpus": "press"}, "press-a"),
    _case("the other corpus", {"corpus": "central-bank"}, *(ALL_DOCUMENTS - {"press-a"})),
    # Lists are OR.
    _case("two sources", {"source_codes": ["fed", "boj"]}, "fed-a", "fed-b", "boj-a", "boj-b"),
    _case("a tuple of doc types", {"doc_types": ("report", "article")}, "boe-a", "press-a"),
    _case("two languages", {"languages": ["fr", "ja"]}, "fed-b", "boj-a", "boj-b"),
    _case("two documents", {"doc_ids": ["ecb-b", "boe-a"]}, "ecb-b", "boe-a"),
    _case("a known and an unknown value", {"source_codes": ["nowhere", "boe"]}, "boe-a"),
    # Fields are AND.
    _case("source and doc type", {"source_codes": ["ecb"], "doc_types": ["statement"]}, "ecb-a"),
    _case("source and corpus", {"source_codes": ["ecb"], "corpus": "central-bank"}, "ecb-a", "ecb-b"),
    _case(
        "lists in two fields",
        {"source_codes": ["ecb", "fed"], "doc_types": ["minutes", "article"]},
        "ecb-b",
        "fed-b",
        "press-a",
    ),
    _case("fields that exclude each other", {"source_codes": ["boj"], "languages": ["en"]}),
    _case(
        "every field at once",
        {
            "source_codes": ["ecb", "fed"],
            "doc_types": ["statement", "minutes"],
            "languages": ["en"],
            "doc_ids": ["ecb-a", "ecb-b", "fed-a", "fed-b"],
            "date_from": "2016-03-10",
            "date_to": "2016-03-10",
            "corpus": "central-bank",
        },
        "ecb-a",
        "ecb-b",
    ),
    # Unknown values are not errors.
    _case("unknown source", {"source_codes": ["nowhere"]}),
    _case("unknown document", {"doc_ids": ["no-such-document"]}),
    _case("unknown corpus", {"corpus": "nothing"}),
    _case("a value in another case", {"source_codes": ["ECB"]}),
    # Dates: both bounds inclusive; a document without a full date matches no bound.
    _case("from a day, inclusive", {"date_from": "2016-03-10"}, "ecb-a", "ecb-b", "fed-b", "press-a"),
    _case(
        "to a day, inclusive of its afternoon", {"date_to": "2016-03-10"}, "ecb-a", "ecb-b", "fed-a", "snb-a"
    ),
    _case("one day", {"date_from": "2016-03-10", "date_to": "2016-03-10"}, "ecb-a", "ecb-b"),
    _case("up to the day before", {"date_to": "2016-03-09"}, "fed-a", "snb-a"),
    _case("the day before only", {"date_from": "2016-03-09", "date_to": "2016-03-09"}, "fed-a"),
    _case("from the day after", {"date_from": "2016-03-11"}, "fed-b", "press-a"),
    _case("a range around everything dated", {"date_from": "1900-01-01", "date_to": "2100-12-31"}, *DATED),
    _case("from the first day of the calendar", {"date_from": "0001-01-01"}, *DATED),
    _case("to the last day of the calendar", {"date_to": "9999-12-31"}, *DATED),
    _case("a range before everything", {"date_to": "2000-01-01"}),
    _case("a range after everything", {"date_from": "2030-01-01"}),
    _case(
        "date objects",
        {"date_from": datetime.date(2016, 3, 10), "date_to": datetime.date(2016, 3, 11)},
        "ecb-a",
        "ecb-b",
        "fed-b",
    ),
    _case("dates and a source", {"date_from": "2016-03-10", "source_codes": ["fed"]}, "fed-b"),
    _case("dates and an undated source", {"date_from": "1900-01-01", "source_codes": ["boj", "boe"]}),
    _case("the undated source without dates", {"source_codes": ["boj", "boe"]}, "boj-a", "boj-b", "boe-a"),
]
