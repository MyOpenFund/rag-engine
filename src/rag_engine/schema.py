"""Payload schema v2 and deterministic point ids.

Schema v1 is the payload the MyOpenFund data-orchestrator wrote before this package existed.
Schema v2 (``PAYLOAD_SCHEMA_VERSION``) is schema v1 plus two keys the engine derives from the
metadata, ``date_start`` and ``date_end``. Nothing is renamed or removed and ids are computed
the same way, so existing points stay valid and ingesting a document again overwrites its
points in place. The version is the name of this layout; it is not stored in the points: a
point of version 1 is a point without the two derived keys, and
``QdrantStore.set_payload`` adds them without touching its vector.

One point is one chunk of one document. Its payload holds:

Keys written by the engine (``RESERVED_KEYS``; caller metadata may not set them):

- ``doc_id``: identifier of the document, given by the caller.
- ``filename``: base name of the ingested file.
- ``chunk_number``: 0-based position of the chunk in the document. The key is
  ``chunk_number``, not ``chunk_index``.
- ``ingestion_date``: the engine's own timestamp, an ISO 8601 string, identical
  on every point of a document. It says when the point was written, nothing
  about the document itself.
- ``text``: the text of the chunk.
- ``page``: page the chunk starts on. Absent for formats without pages
  (markdown, plain text, ...); never written as null or 0 in that case.

Keys copied from the caller's metadata. The engine does not change them; the
MyOpenFund vault source provides these today, each one left out when the vault
has no value for it:

- ``corpus``: corpus the document belongs to.
- ``source_code``: code of the source inside the corpus.
- ``doc_type``: kind of document.
- ``title``: title of the document.
- ``date``: publication date of the document, an ISO 8601 string. Not to be
  confused with ``ingestion_date``. When the document is known to the month or
  to the year only, the producer writes the first day of the period here and
  says so in ``date_precision``.
- ``date_precision``: how precise ``date`` is: ``day``, ``month`` or ``year``
  (``DATE_PRECISIONS``). Absent, null or anything else means "not stated".
- ``date_kind``: what ``date`` is (``meeting``, ``publication``, ``period``).
  Never read by the engine.
- ``year``: publication year, an integer.
- ``language``: language code.
- ``sha256``: hash of the source file.
- ``provenance``: where the file came from.

Any other metadata key is copied as is. ``date_precision`` and ``date_kind``
are ordinary metadata: copied like the others, and not reserved.

Keys derived by the engine from ``date`` and ``date_precision``
(``DERIVED_KEYS``, new in schema v2; ``date_interval`` computes them):

- ``date_start``: the first day of the period the document is dated to.
- ``date_end``: the last day of that period, **included**.

Both are RFC 3339 strings at midnight UTC (``2016-03-01T00:00:00Z``): the day
itself for a ``day`` precision, the first and the last day of the month for
``month``, 1 January and 31 December for ``year``. A date whose precision is
not stated is read as a day, which is what a point of version 1 means. A
document without a readable ``date`` has neither key. They are written after
every key of version 1, so the payload of version 1 is the beginning of the
payload of version 2, key for key. Caller metadata may not set them.

The point id depends on ``(doc_id, page, chunk_index)`` only, where
``chunk_index`` is the value stored as ``chunk_number`` and ``page`` is 0 when
the chunk has no page (see ``page_for_id``).

Payload indexes (``PAYLOAD_INDEXES``) are what the search filters rely on. An
index never changes a point: it can be added to a collection that already
holds points, and a payload value an index cannot read is not an error. The
point stays stored and is still returned by a search without a filter on that
field, but a filter on that field no longer sees it.
"""

from __future__ import annotations

import calendar
import datetime
import hashlib
import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "DATE_PRECISIONS",
    "DERIVED_KEYS",
    "PAYLOAD_INDEXES",
    "PAYLOAD_SCHEMA_VERSION",
    "RESERVED_KEYS",
    "build_payload",
    "date_interval",
    "date_interval_fields",
    "page_for_id",
    "point_id",
]

#: The version of the payload layout this module writes: 1 plus ``date_start`` and ``date_end``.
PAYLOAD_SCHEMA_VERSION = 2

RESERVED_KEYS: tuple[str, ...] = ("doc_id", "filename", "chunk_number", "ingestion_date", "text", "page")

#: The keys the engine computes from the metadata (:func:`date_interval`), in payload order.
#: Caller metadata may not set them at ingestion. They are not reserved for
#: ``QdrantStore.set_payload``, which is how they are added to points written before version 2.
DERIVED_KEYS: tuple[str, ...] = ("date_start", "date_end")

#: The values of ``date_precision`` the engine reads. Anything else is read as ``day``.
DATE_PRECISIONS: tuple[str, ...] = ("day", "month", "year")

#: The payload fields the engine indexes, as ``(field, kind)``; a kind is the name of a
#: Qdrant payload index type. ``QdrantStore.ensure_payload_indexes`` creates them.
#:
#: - keyword (exact match): ``doc_id``, ``source_code``, ``doc_type``, ``language``, ``corpus``.
#:   The index reads strings (and lists of strings). A number or a boolean is not read: a
#:   filter, which matches strings, does not find that point.
#: - integer (match and range): ``year``. The index reads integers, and floats with an
#:   integral value (``2016.0``). A ``year`` stored as the string ``"2016"`` or as ``2016.5``
#:   is not read: once the index exists, it matches no integer range.
#: - datetime (range): ``date``, the publication date as the vault gives it. The index reads
#:   a full date (``2016-03-10``, taken as midnight UTC); a full date and a time, after ``T``
#:   or a space (``2016-03-10T12:00:00``, ``2016-03-10 12:00:00``, ``2016-03-10T12:00``);
#:   the same with ``Z`` or an offset (``2016-03-10T12:00:00+02:00``); and the lenient
#:   ``2016-3-10``. It does not read ``2016-03``, ``2016``, an empty string, a null, a
#:   missing key, an integer (``20160310``, a Unix timestamp) or ``10/03/2016``.
#: - datetime (range): ``date_start`` and ``date_end``, the interval the engine derives from
#:   ``date`` and ``date_precision`` (:func:`date_interval`), always written at midnight UTC.
#:   **These two indexes are also the signal a search reads**: with
#:   ``date_filter_mode="auto"``, a date bound uses the intervals as soon as both exist on
#:   the collection (``rag_engine.search``).
#:
#: A value an index cannot read is never an error, neither when the index is built nor when
#: the point is written. The point is stored as it is, and a search without a filter on that
#: field still returns it; a filter on that field no longer sees it. Before creating the
#: indexes on a collection that already holds points, check the types of the stored values
#: of ``year`` and of the keyword fields. (Checked on Qdrant 1.20.)
#:
#: The bounds of a date range must be full dates as well. The REST API of Qdrant refuses
#: ``"2016"`` and ``"2016-03"`` with HTTP 400. Through the Python client, ``"2016-03"`` is a
#: validation error raised by the client, but ``"2016"`` is accepted and read as a Unix
#: timestamp (``1970-01-01T00:33:36Z``): no error, and a range that matches almost
#: everything or almost nothing depending on the side of the bound. ``rag_engine.search``
#: therefore accepts only a ``datetime.date`` or a ``YYYY-MM-DD`` string.
PAYLOAD_INDEXES: tuple[tuple[str, str], ...] = (
    ("doc_id", "keyword"),
    ("source_code", "keyword"),
    ("doc_type", "keyword"),
    ("language", "keyword"),
    ("corpus", "keyword"),
    ("year", "integer"),
    ("date", "datetime"),
    ("date_start", "datetime"),
    ("date_end", "datetime"),
)

# A stored date: a full date, alone or followed by a time of day and an optional UTC offset.
_STORED_DATE = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{1,2})-(?P<day>[0-9]{1,2})"
    r"(?:[Tt ](?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{1,2})(?::(?P<second>[0-9]{1,2})(?:\.[0-9]+)?)?"
    r" ?(?:(?P<utc>[Zz])|(?P<sign>[+-])(?P<offset_hours>[0-9]{2})(?::?(?P<offset_minutes>[0-9]{2}))?)?)?"
)


def _utc_day(value: object) -> datetime.date | None:
    """The UTC day ``value`` falls on; ``None`` when it is not a date the engine reads."""
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            try:
                value = value.astimezone(datetime.timezone.utc)
            except OverflowError:  # the first or the last day of the calendar, pushed out by its offset
                return None
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if not isinstance(value, str):
        return None
    found = _STORED_DATE.fullmatch(value)
    if found is None:
        return None
    try:
        day = datetime.date(int(found["year"]), int(found["month"]), int(found["day"]))
        if found["hour"] is None:
            return day
        moment = datetime.datetime.combine(
            day, datetime.time(int(found["hour"]), int(found["minute"]), int(found["second"] or 0))
        )
        if found["sign"] is not None:
            offset_hours, offset_minutes = int(found["offset_hours"]), int(found["offset_minutes"] or 0)
            if offset_hours > 23 or offset_minutes > 59:
                return None
            offset = datetime.timedelta(hours=offset_hours, minutes=offset_minutes)
            moment = moment - offset if found["sign"] == "+" else moment + offset
    except (ValueError, OverflowError):  # a day or a time that does not exist, or outside the calendar
        return None
    return moment.date()


def _midnight_utc(day: datetime.date) -> str:
    return f"{day.year:04d}-{day.month:02d}-{day.day:02d}T00:00:00Z"


def date_interval(date: object, date_precision: object = None) -> tuple[str, str] | None:
    """The ``(date_start, date_end)`` a document dated ``date`` is stored with; ``None`` without a date.

    This is the one place the interval is computed: :func:`build_payload` calls it at
    ingestion, and a caller that corrects a date afterwards calls it too, then writes the
    result with ``QdrantStore.set_payload`` (:func:`date_interval_fields` gives it as the
    two payload keys).

    ``date_precision`` says how much of ``date`` is known (:data:`DATE_PRECISIONS`):

    - ``"day"``: the day itself, twice;
    - ``"month"``: the first and the last day of the month ``date`` falls in;
    - ``"year"``: 1 January and 31 December of the year ``date`` falls in;
    - anything else (``None``, a missing value, another word, another case such as
      ``"Month"``, a value that is not a string): read as ``"day"``. A point written before
      the precision existed is searched by its day, and so is a date whose precision is not
      stated.

    **The stored form.** Both ends are RFC 3339 strings at midnight UTC, and ``date_end`` is
    the **last day, included**, not the day after: ``("2016-03-01T00:00:00Z",
    "2016-03-31T00:00:00Z")`` for March 2016. An interval overlaps the days ``first`` to
    ``last`` (both included) exactly when ``date_start <= last at 00:00Z`` and
    ``date_end >= first at 00:00Z``, two comparisons a Qdrant datetime range makes as they
    are; nothing depends on how many days a month has or on a time of day.

    ``date`` is read the way the datetime index of Qdrant reads the payload field: a full
    date (``"2016-03-10"``, or the lenient ``"2016-3-10"``), alone or followed, after ``T``
    or a space, by a time (``"2016-03-10T15:30:00Z"``, ``"2016-03-10 15:30"``, with or
    without seconds and a fraction, with ``Z``, an offset such as ``+02:00``, or no zone,
    which means UTC). A date with a time counts for the **UTC day** of that instant:
    ``"2016-03-11T00:30:00+02:00"`` is 10 March. A ``datetime.date`` and a
    ``datetime.datetime`` (UTC when it has no time zone) are read as well.

    ``None`` is returned for no date and for a date that cannot be read: ``None``, an empty
    string, a partial date (``"2016-03"``, ``"2016"``), a day that does not exist, a number,
    any other type. Such a document has no interval, whatever its precision. Nothing is
    raised: like the payload indexes, an unreadable value is not an error.
    """
    day = _utc_day(date)
    if day is None:
        return None
    first, last = day, day
    if date_precision == "month":
        first = day.replace(day=1)
        last = day.replace(day=calendar.monthrange(day.year, day.month)[1])
    elif date_precision == "year":
        first, last = day.replace(month=1, day=1), day.replace(month=12, day=31)
    return _midnight_utc(first), _midnight_utc(last)


def date_interval_fields(date: object, date_precision: object = None) -> dict[str, str | None]:
    """:func:`date_interval` as the two payload keys, for ``QdrantStore.set_payload``.

    Always both keys of :data:`DERIVED_KEYS`. Without a readable date their values are
    ``None``: given to ``set_payload``, a null removes the key from the points (and a stored
    null would be, to every date filter, what a missing key is), so a document whose date
    was withdrawn loses its interval in the same write, and a document that never had a date
    is left as it is.
    """
    start, end = date_interval(date, date_precision) or (None, None)
    return {"date_start": start, "date_end": end}


def point_id(doc_id: str, page: int, chunk_index: int) -> int:
    """Return the Qdrant point id of one chunk: a stable integer in ``[0, 2**63)``.

    The algorithm is frozen: changing it would move the id of every point
    already stored.
    """
    key = f"{doc_id}::{page}::{chunk_index}"
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) >> 1


def page_for_id(page: int | None) -> int:
    """Return the page to use in ``point_id``: 0 for a chunk without a page.

    The payload keeps the real absence instead (``build_payload`` writes no
    ``page`` key for ``None``).
    """
    return 0 if page is None else page


def build_payload(
    doc_id: str,
    metadata: Mapping[str, Any],
    *,
    filename: str,
    chunk_number: int,
    ingestion_date: str,
    text: str,
    page: int | None,
) -> dict[str, Any]:
    """Return the schema v2 payload of one chunk as a new dict.

    ``metadata`` is copied untouched between ``doc_id`` and the engine's own
    keys. It may not contain a reserved key: that raises ``ValueError`` naming
    every offending key. The one exception is ``doc_id`` equal to the
    ``doc_id`` argument, which is accepted.

    ``date_start`` and ``date_end`` are added last, when ``metadata`` holds a
    readable ``date``: ``date_interval(metadata["date"], metadata["date_precision"])``.
    ``metadata`` may not set them itself (``DERIVED_KEYS``): a value that disagreed
    with ``date`` would make a search answer two ways. That is a ``ValueError`` too.
    """
    clashes = [
        key for key in RESERVED_KEYS if key in metadata and not (key == "doc_id" and metadata[key] == doc_id)
    ]
    if clashes:
        raise ValueError(
            f"metadata may not set reserved payload key(s): {', '.join(clashes)} (document {doc_id!r})"
        )
    derived = [key for key in DERIVED_KEYS if key in metadata]
    if derived:
        raise ValueError(
            f"metadata may not set derived payload key(s): {', '.join(derived)} (document {doc_id!r}): "
            "the engine computes them from date and date_precision"
        )
    payload: dict[str, Any] = {
        "doc_id": doc_id,
        **metadata,
        "filename": filename,
        "chunk_number": chunk_number,
        "ingestion_date": ingestion_date,
        "text": text,
    }
    if page is not None:
        payload["page"] = page
    interval = date_interval(metadata.get("date"), metadata.get("date_precision"))
    if interval is not None:
        payload["date_start"], payload["date_end"] = interval
    return payload
