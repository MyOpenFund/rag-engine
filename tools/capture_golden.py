"""Capture the golden fixture of ``tests/test_ingest.py`` from the previous ingestion pipeline.

Purpose: provenance. ``tests/fixtures/golden/expected.json`` holds the points (id, payload,
vector) that the data-orchestrator's ``core.ingest_item`` produced for the two documents of
``tests/fixtures/golden/``, before that code moved into this package. The golden test asserts
that ``rag_engine.ingest.ingest_document`` produces the same points, so the points already in
Qdrant stay valid. This script is how the file was made; ``tests/fixtures/golden/README.md``
records the command, the versions and the commits.

It is not part of the package and is not run in CI. It needs a checkout of the
data-orchestrator and of the engine it drives (eigenmind), both on ``PYTHONPATH``, in an
environment that has ChunkNorris and this package:

    PYTHONPATH=<data-orchestrator>:<eigenmind> python tools/capture_golden.py

``--make-pdf`` writes the PDF fixture first. It was used once: PyMuPDF does not write the same
bytes twice (creation date, document id), so the PDF is committed and never generated again.
An existing PDF is not overwritten.

Run it again only to capture from a known-good pipeline; never to make a failing test pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden"
EXPECTED = GOLDEN_DIR / "expected.json"
MARKDOWN = "monetary-policy-statement.md"
PDF = "inflation-report-extract.pdf"

# The fake embedder of the capture and of the test: a small dimension keeps the file small.
MODEL = {"name": "golden-fake", "dim": 8, "passage_prefix": "passage: ", "query_prefix": "query: "}
SEED = 0
# "never" rather than the default "auto": what "auto" does depends on the host (ENABLE_OCR,
# TESSDATA_PREFIX), and the fixture must not.
OCR = "never"

# Metadata as the vault source of the data-orchestrator provides it. ``sha256`` is filled in
# from the file. The PDF carries one key outside the usual columns (``series``).
DOCUMENTS = (
    {
        "file": MARKDOWN,
        "collection": "central-bank-e5b-v1",
        "doc_id": "ecb/monetary-policy-statement/2025-10-30",
        "metadata": {
            "corpus": "central-bank",
            "source_code": "ecb",
            "doc_type": "monetary_policy_statement",
            "title": "Décision de politique monétaire",
            "date": "2025-10-30",
            "year": 2025,
            "language": "fr",
            "provenance": "manifest",
        },
    },
    {
        "file": PDF,
        "collection": "central-bank-e5b-v1",
        "doc_id": "boe/inflation-report/2024-08-01",
        "metadata": {
            "series": "Monetary Policy Report",
            "corpus": "central-bank",
            "source_code": "boe",
            "doc_type": "inflation_report",
            "title": "Monetary Policy Report, August 2024 (extract)",
            "date": "2024-08-01",
            "year": 2024,
            "language": "en",
            "provenance": "manifest",
        },
    },
)

PDF_PAGES = (
    (
        "Monetary Policy Report",
        (
            (
                "The Monetary Policy Committee sets monetary policy to meet the two per cent inflation "
                "target, and in a way that helps to sustain growth and employment. At its meeting "
                "ending on 31 July, the Committee voted by a majority of five to four to reduce Bank "
                "Rate by a quarter of a percentage point, to five per cent."
            ),
            (
                "Twelve-month consumer price inflation was at the target in both May and June. The "
                "Committee expects the fall in headline inflation, and the normalisation of many "
                "indicators of inflation expectations, to continue to feed through to weaker pay and "
                "price-setting dynamics over the coming quarters."
            ),
            (
                "Gross domestic product has picked up quite sharply so far this year, but underlying "
                "momentum appears weaker. Household consumption has been subdued, business investment "
                "has recovered only slowly, and surveys of output point to a more modest pace of "
                "expansion in the second half of the year."
            ),
        ),
    ),
    (
        "Demand and output",
        (
            (
                "A margin of slack is projected to emerge in the economy as activity falls below "
                "potential and the labour market eases further. Unemployment is expected to rise "
                "gradually over the forecast period."
            ),
        ),
    ),
    (
        "Costs and prices",
        (
            (
                "Services price inflation has remained elevated, at 5.7 per cent in June. Private "
                "sector regular average weekly earnings growth has fallen to 5.6 per cent in the three "
                "months to May, and is expected to ease further as past increases in the minimum wage "
                "drop out of the annual comparison."
            ),
            (
                "Energy prices are projected to push headline inflation up to around 2.75 per cent in "
                "the second half of this year, as declines in energy prices last year fall out of the "
                "annual comparison, revealing more clearly the prevailing persistence of domestic "
                "inflationary pressures."
            ),
            (
                "Conditional on market interest rates, inflation is projected to fall back to 1.7 per "
                "cent in two years' time and to 1.5 per cent in three years, reflecting the continued "
                "restrictiveness of the policy stance and the emergence of spare capacity."
            ),
        ),
    ),
)


def make_pdf(target: Path) -> None:
    """Write the three-page PDF fixture: a title and body paragraphs per page, the second page short."""
    import pymupdf

    document = pymupdf.open()
    for title, paragraphs in PDF_PAGES:
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 96), title, fontsize=18, fontname="hebo")
        top = 130
        for paragraph in paragraphs:
            rect = pymupdf.Rect(72, top, 523, top + 110)
            unused = page.insert_textbox(rect, paragraph, fontsize=11, fontname="helv")
            if unused < 0:
                raise RuntimeError(f"paragraph does not fit its box on page {title!r}")
            top += 120
    document.save(target, garbage=4, deflate=True)
    document.close()


class RecordingClient:
    """The part of a Qdrant client that ``ingest_item`` uses; keeps what it is asked to write."""

    def __init__(self) -> None:
        self.collections: list[str] = []
        self.points: list = []

    def upsert(self, collection_name, points, **_ignored) -> None:
        self.collections.append(collection_name)
        self.points.extend(points)


class RecordingStore:
    def __init__(self) -> None:
        self.client = RecordingClient()


def capture() -> dict:
    # The previous pipeline, imported as it is: nothing of it is replaced.
    from data_orchestrator.core import SourceItem, ingest_item

    from rag_engine.embedding import FakeEmbedder, ModelSpec

    documents = []
    for spec in DOCUMENTS:
        path = GOLDEN_DIR / spec["file"]
        metadata = {**spec["metadata"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        store = RecordingStore()
        written = ingest_item(
            SourceItem(doc_id=spec["doc_id"], path=path, payload=dict(metadata)),
            spec["collection"],
            store=store,
            embedder=FakeEmbedder(ModelSpec(**MODEL), seed=SEED),
            ocr=OCR,
        )
        points = store.client.points
        if written != len(points) or not points:
            raise RuntimeError(f"{spec['file']}: {written} chunks reported, {len(points)} points recorded")
        if set(store.client.collections) != {spec["collection"]}:
            raise RuntimeError(f"{spec['file']}: written to {store.client.collections}")
        dates = {point.payload["ingestion_date"] for point in points}
        if len(dates) != 1:
            raise RuntimeError(f"{spec['file']}: several ingestion dates: {sorted(dates)}")
        documents.append(
            {
                "file": spec["file"],
                "collection": spec["collection"],
                "doc_id": spec["doc_id"],
                "metadata": metadata,
                "ingestion_date": dates.pop(),
                "points": [
                    {"id": point.id, "payload": point.payload, "vector": point.vector} for point in points
                ],
            }
        )
    return {
        "captured_with": {
            "python": platform.python_version(),
            "chunknorris": version("chunknorris"),
            "pymupdf": version("PyMuPDF"),
        },
        "embedder": {"model": MODEL, "seed": SEED},
        "ocr": OCR,
        "documents": documents,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--make-pdf", action="store_true", help="write the PDF fixture first (never overwrites)"
    )
    args = parser.parse_args(argv)
    if args.make_pdf:
        target = GOLDEN_DIR / PDF
        if target.exists():
            print(f"{target} exists: not overwritten", file=sys.stderr)
            return 1
        make_pdf(target)
        print(f"wrote {target} ({target.stat().st_size} bytes)")
    result = capture()
    EXPECTED.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for document in result["documents"]:
        pages = sorted({point["payload"].get("page") for point in document["points"]}, key=str)
        print(f"{document['file']}: {len(document['points'])} points, pages {pages}")
    print(f"wrote {EXPECTED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
