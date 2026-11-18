"""The dated corpus, the random ranges and the oracle shared by the tests of the date filters.

The unit tests (qdrant-client's local mode) and the integration tests (a real server) search
the same documents with the same ranges and expect what ``passes`` says. The oracle is plain
Python on days: it shares no code with ``rag_engine.schema.date_interval`` nor with the filter
``rag_engine.search`` builds, so an error in either shows as a difference.

One document is one point. A document says how its date is stored, with which precision, and
two facts that are written by hand, not computed:

``day``
    The UTC day the datetime index of Qdrant reads ``date`` as; ``None`` when it reads none
    (no key, a null, an empty string, a partial date).
``engine_day``
    The UTC day the engine reads ``date`` as when it derives the interval; ``None`` when it
    derives none. It is ``day`` for every form but the few the two do not read alike.
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass
from typing import Any

# The engine first: importing it sets the default that keeps the ONNX runtime quiet, and
# qdrant-client loads that runtime when fastembed is installed.
from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.schema import DERIVED_KEYS, build_payload, point_id

# isort: split
from qdrant_client import models

DATE_SPEC = ModelSpec(name="fake", dim=8, passage_prefix="passage: ", query_prefix="query: ")
DATE_WHEN = "2026-11-18T10:00:00"
DATE_QUERY = "what was decided, and when?"

MISSING: Any = object()
SAME: Any = object()
ONE_DAY = datetime.timedelta(days=1)


def D(text: str) -> datetime.date:
    """A day written ``YYYY-MM-DD``: a short constructor for the tables below."""
    return datetime.date.fromisoformat(text)


@dataclass(frozen=True)
class DatedDocument:
    doc_id: str
    date: Any = MISSING  # the value stored under "date"; MISSING for no key
    precision: Any = MISSING  # the value stored under "date_precision"; MISSING for no key
    day: datetime.date | None = None
    engine_day: Any = SAME

    @property
    def read_by_engine(self) -> datetime.date | None:
        return self.day if self.engine_day is SAME else self.engine_day

    @property
    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"corpus": "dates", "source_code": self.doc_id.split("-")[0]}
        if self.date is not MISSING:
            metadata["date"] = self.date
        if self.precision is not MISSING:
            metadata["date_precision"] = self.precision
        metadata["date_kind"] = "meeting"
        return metadata

    @property
    def period(self) -> tuple[datetime.date, datetime.date] | None:
        """The first and the last day the document is dated to, both included; ``None`` without a date."""
        day = self.read_by_engine
        if day is None:
            return None
        if self.precision == "month":
            first = datetime.date(day.year, day.month, 1)
            after = datetime.date(day.year + 1, 1, 1) if day.month == 12 else None
            if after is None:
                after = datetime.date(day.year, day.month + 1, 1)
            return first, after - ONE_DAY
        if self.precision == "year":
            return datetime.date(day.year, 1, 1), datetime.date(day.year, 12, 31)
        return day, day


def _doc(doc_id: str, date: Any = MISSING, precision: Any = MISSING, day: str | None = None, **more: Any):
    return DatedDocument(doc_id, date, precision, D(day) if day is not None else None, **more)


# What each hand-written document is there for.
ADVERSARIAL_DOCUMENTS: list[DatedDocument] = [
    # Day precision, stated, not stated, and stated with a word the engine does not know.
    _doc("day-stated", "2016-03-10", "day", "2016-03-10"),
    _doc("day-no-key", "2016-03-10", MISSING, "2016-03-10"),
    _doc("day-null", "2016-03-11", None, "2016-03-11"),
    _doc("day-garbage", "2016-07-01", "fortnight", "2016-07-01"),  # garbage precision: a day
    _doc("day-wrong-case", "2016-07-01", "Month", "2016-07-01"),  # not "month": a day too
    _doc("day-leap", "2016-02-29", "day", "2016-02-29"),
    _doc("day-new-year-eve", "2016-12-31", "day", "2016-12-31"),
    _doc("day-new-year", "2017-01-01", "day", "2017-01-01"),
    # Datetime strings, with and without an offset: the UTC day counts.
    _doc("time-utc", "2016-03-10T15:30:00Z", "day", "2016-03-10"),
    _doc("time-last-instant", "2016-03-10T23:59:59.999999Z", MISSING, "2016-03-10"),
    _doc("time-plus", "2016-03-11T00:30:00+02:00", "day", "2016-03-10"),  # the day before, in UTC
    _doc("time-minus", "2016-03-10T23:30:00-05:00", "day", "2016-03-11"),  # the day after, in UTC
    _doc("time-naive", "2016-03-09 12:00:00", MISSING, "2016-03-09"),
    _doc("time-lenient", "2016-3-9", MISSING, "2016-03-09"),
    # Month precision: stored at the first day, as the producer does.
    _doc("month-july", "2016-07-01", "month", "2016-07-01"),  # a one-day range inside the month
    _doc("month-leap-february", "2016-02-01", "month", "2016-02-01"),
    _doc("month-plain-february", "2015-02-01", "month", "2015-02-01"),
    _doc("month-december", "2016-12-01", "month", "2016-12-01"),  # ends with the year
    _doc("month-thirty-days", "2016-04-01", "month", "2016-04-01"),
    _doc("month-mid", "2016-09-17", "month", "2016-09-17"),  # not stored at the first day
    _doc("month-offset", "2016-06-01T00:30:00+02:00", "month", "2016-05-31"),  # May, in UTC
    # Year precision: at both edges of its year.
    _doc("year-2016", "2016-01-01", "year", "2016-01-01"),
    _doc("year-2015", "2015-01-01", "year", "2015-01-01"),
    _doc("year-mid", "2017-06-15", "year", "2017-06-15"),
    # The ends of the calendar.
    _doc("first-day", "0001-01-01", "day", "0001-01-01"),
    _doc("last-day", "9999-12-31", "day", "9999-12-31"),
    _doc("last-year", "9999-01-01", "year", "9999-01-01"),
    # No date at all, in every way it can be missing; the precision changes nothing.
    _doc("undated-no-key"),
    _doc("undated-null", None, None),
    _doc("undated-empty", "", "day"),
    _doc("undated-partial-month", "2016-03", "month"),
    _doc("undated-partial-year", "2016", "year"),
    _doc("undated-precision-only", MISSING, "month"),
    # Read by the engine and not by the datetime index: found by its interval only.
    _doc("engine-only", "2016-03-10T12:00Z", "day", None, engine_day=D("2016-03-10")),
]

# Read by the datetime index of a real server and not by the engine: no interval, found by its
# date, in every mode. For the integration tests only.
SERVER_ONLY_DOCUMENTS: list[DatedDocument] = [
    _doc("index-only-space", " 2016-03-10", "month", "2016-03-10", engine_day=None),
    _doc("index-only-list", ["2016-03-10"], "year", "2016-03-10", engine_day=None),
]


def random_documents(count: int, seed: int) -> list[DatedDocument]:
    """``count`` documents dated between 2014 and 2018 with a random precision, some undated."""
    rng = random.Random(seed)
    first = datetime.date(2014, 1, 1).toordinal()
    last = datetime.date(2018, 12, 31).toordinal()
    documents = []
    for index in range(count):
        day = datetime.date.fromordinal(rng.randint(first, last))
        precision = rng.choice(["day", "day", "month", "month", "year", None, MISSING, "week"])
        if precision == "month" and rng.random() < 0.8:
            day = day.replace(day=1)  # the convention of the producer; a few are left mid-period
        if precision == "year" and rng.random() < 0.8:
            day = day.replace(month=1, day=1)
        if rng.random() < 0.08:
            documents.append(DatedDocument(f"random-{index:03d}", None, precision))
            continue
        stored = day.isoformat() + rng.choice(["", "", "", "T00:00:00Z", "T18:45:00Z"])
        documents.append(DatedDocument(f"random-{index:03d}", stored, precision, day))
    return documents


DATE_DOCUMENTS: list[DatedDocument] = ADVERSARIAL_DOCUMENTS + random_documents(70, seed=20261118)
DOCUMENT_COUNT = len(DATE_DOCUMENTS)

#: How the points of a collection were written.
#: - "v2": every point by ``build_payload``, with its interval;
#: - "v1": as before schema v2, without ``date_start`` and ``date_end`` (the production collection);
#: - "half": every other document upgraded, the others still as in "v1".
STATES = ("v1", "v2", "half")


def has_interval(state: str, index: int) -> bool:
    """Whether the document at ``index`` carries its interval in a collection in ``state``."""
    return state == "v2" or (state == "half" and index % 2 == 0)


def date_points(state: str, documents: list[DatedDocument] | None = None) -> list[models.PointStruct]:
    """One point per document, written as the collection ``state`` says."""
    documents = DATE_DOCUMENTS if documents is None else documents
    texts = [f"the text of {document.doc_id}" for document in documents]
    vectors = FakeEmbedder(DATE_SPEC).encode_passage(texts)
    points = []
    for index, (document, text, vector) in enumerate(zip(documents, texts, vectors)):
        payload = build_payload(
            document.doc_id,
            document.metadata,
            filename=f"{document.doc_id}.pdf",
            chunk_number=0,
            ingestion_date=DATE_WHEN,
            text=text,
            page=0,
        )
        if not has_interval(state, index):
            payload = {key: value for key, value in payload.items() if key not in DERIVED_KEYS}
        points.append(
            models.PointStruct(id=point_id(document.doc_id, 0, 0), vector=vector.tolist(), payload=payload)
        )
    return points


def _in_range(first: datetime.date, last: datetime.date, date_from, date_to) -> bool:
    """Whether the days ``first`` to ``last`` and the range have a day in common, bounds included."""
    return (date_to is None or first <= date_to) and (date_from is None or last >= date_from)


def passes(
    document: DatedDocument,
    *,
    interval_stored: bool,
    mode: str,
    date_from: datetime.date | None,
    date_to: datetime.date | None,
    include_undated: bool = False,
) -> bool:
    """Whether ``document`` must come back from a search with these bounds: the oracle.

    ``interval_stored`` says whether its point carries ``date_start`` and ``date_end``;
    ``mode`` is the filter actually applied, ``"legacy"`` or ``"interval"``.
    """
    assert mode in ("legacy", "interval")
    assert date_from is not None or date_to is not None
    by_date = document.day is not None and _in_range(document.day, document.day, date_from, date_to)
    if mode == "legacy":
        return by_date or (include_undated and document.day is None)
    period = document.period if interval_stored else None
    by_interval = period is not None and _in_range(*period, date_from, date_to)
    undated = period is None and document.day is None
    return by_interval or by_date or (include_undated and undated)


def expected_documents(
    state: str,
    mode: str,
    date_from: datetime.date | None,
    date_to: datetime.date | None,
    include_undated: bool = False,
    documents: list[DatedDocument] | None = None,
) -> set[str]:
    documents = DATE_DOCUMENTS if documents is None else documents
    return {
        document.doc_id
        for index, document in enumerate(documents)
        if passes(
            document,
            interval_stored=has_interval(state, index),
            mode=mode,
            date_from=date_from,
            date_to=date_to,
            include_undated=include_undated,
        )
    }


# Ranges written by hand: every edge the fixtures above were made for.
EDGE_RANGES: list[tuple[str | None, str | None]] = [
    ("2016-07-14", "2016-07-14"),  # one day inside a month-precision document
    ("2016-07-01", "2016-07-01"),  # its first day: the garbage-precision day is there too
    ("2016-07-31", "2016-07-31"),  # its last day
    ("2016-08-01", "2016-08-01"),  # the day after
    ("2016-06-30", "2016-06-30"),  # the day before
    ("2016-02-29", "2016-02-29"),  # the last day of a leap February
    ("2016-03-01", "2016-03-01"),
    ("2015-02-28", "2015-02-28"),
    ("2015-03-01", "2015-03-01"),
    ("2016-04-30", "2016-04-30"),  # a one-day range on the last day of a month
    ("2016-05-01", "2016-05-01"),
    ("2016-05-31", "2016-05-31"),  # the month a date with an offset really falls in
    ("2016-06-01", "2016-06-01"),
    ("2016-12-31", "2016-12-31"),  # 31 December: a year, a month and a day end there
    ("2017-01-01", "2017-01-01"),
    ("2015-12-31", "2015-12-31"),  # a year-precision document at the edge of a range
    ("2016-01-01", "2016-01-01"),
    (None, "2015-12-31"),
    (None, "2016-01-01"),
    ("2016-12-31", None),
    ("2017-01-01", None),
    ("2016-03-10", "2016-03-10"),  # the datetime strings, with and without offsets
    ("2016-03-11", "2016-03-11"),
    ("2016-03-09", "2016-03-09"),
    ("2016-03-10", "2016-03-11"),
    ("0001-01-01", None),
    (None, "9999-12-31"),
    ("0001-01-01", "9999-12-31"),
    ("0001-01-01", "0001-01-01"),
    ("9999-12-31", "9999-12-31"),
    ("9999-06-01", "9999-06-30"),
    (None, "0001-01-01"),
    ("9999-12-31", None),
    ("2000-01-01", "2000-12-31"),  # nothing
    ("2016-01-01", "2016-12-31"),
]


def random_ranges(count: int, seed: int) -> list[tuple[datetime.date | None, datetime.date | None]]:
    """``count`` ranges around the corpus: short and long, one-sided, and snapped to the edges of periods."""
    rng = random.Random(seed)
    first = datetime.date(2013, 10, 1).toordinal()
    last = datetime.date(2019, 3, 31).toordinal()
    ranges: list[tuple[datetime.date | None, datetime.date | None]] = []
    while len(ranges) < count:
        start = datetime.date.fromordinal(rng.randint(first, last))
        kind = rng.random()
        if kind < 0.25:  # snapped to the first or the last day of a month, give or take a day
            start = start.replace(day=1) + rng.choice([-1, 0, 0, 1]) * ONE_DAY
        elif kind < 0.35:  # snapped to the edge of a year
            start = start.replace(month=1, day=1) + rng.choice([-1, 0, 0, 1]) * ONE_DAY
        length = rng.choice([0, 0, 0, 1, 2, 6, 27, 30, 31, 89, 364, 365, 800])
        length = rng.randint(0, length) if rng.random() < 0.5 else length
        end = start + length * ONE_DAY
        side = rng.random()
        if side < 0.1:
            ranges.append((start, None))
        elif side < 0.2:
            ranges.append((None, end))
        else:
            ranges.append((start, end))
    return ranges


def all_ranges(count: int, seed: int) -> list[tuple[datetime.date | None, datetime.date | None]]:
    """The hand-written edges, then ``count`` random ranges."""
    edges = [(D(a) if a else None, D(b) if b else None) for a, b in EDGE_RANGES]
    return edges + random_ranges(count, seed)
