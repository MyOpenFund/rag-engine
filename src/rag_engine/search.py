"""Search what was ingested: one dense cosine search, narrowed by filters on the payload.

``search`` embeds the query, asks Qdrant for the closest chunks that pass the filters and
returns them as :class:`Hit` objects, best first. The filters are a :class:`SearchFilters`,
checked when it is built: a filter that could be read in two ways is refused instead of
being guessed.

Importing this module loads no embedding backend: the embedder is given by the caller.
"""

from __future__ import annotations

import datetime
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from qdrant_client import models

from .config import Settings
from .embedding import Embedder
from .errors import RagEngineError
from .store import QdrantStore

__all__ = ["QUERY_NORMALIZATION", "Hit", "SearchError", "SearchFilters", "is_searchable", "search"]

#: The Unicode form a query is put in before it is embedded (``normalize_query=True``): the
#: form the chunker stores, and therefore embeds, the text of the chunks in.
QUERY_NORMALIZATION = "NFKD"

_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_ONE_DAY = datetime.timedelta(days=1)

# The fields of SearchFilters that hold a list of accepted values, in the order of the filter.
_LIST_FIELDS = (
    ("source_codes", "source_code"),
    ("doc_types", "doc_type"),
    ("languages", "language"),
    ("doc_ids", "doc_id"),
)


class SearchError(RagEngineError, ValueError):
    """A search was asked with an argument or a filter that cannot be used as it is."""


def _encodable(text: str) -> bool:
    """Whether ``text`` can be written as UTF-8: false for a string that holds a lone surrogate."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _not_encodable(name: str, verb: str, text: str) -> SearchError:
    # ascii(): the message can itself be encoded, whatever the string holds.
    return SearchError(
        f"{name} {verb} a string that cannot be encoded as UTF-8 (a lone surrogate, as a JSON "
        f"escape such as \\ud800 gives): {text!a}"
    )


def is_searchable(text: str, *, normalized: bool = False) -> bool:
    """Whether ``text`` holds something to search for: the rule :func:`search` applies to a query.

    True when ``text`` holds at least one letter, number, symbol or punctuation mark (Unicode
    general categories L, N, S and P), in any script; false for a string made only of
    combining marks, of invisible or control characters and of spaces, and for an empty one.

    With ``normalized=True`` the question is asked of ``text`` put in
    :data:`QUERY_NORMALIZATION`, the form a query is embedded in when ``normalize_query`` is
    on: the spacing form of a diacritic (``"\u00a8"``, ``"\u00b4"``) is a symbol as given and
    becomes a space and a combining mark. :func:`search` refuses a query unless both
    ``is_searchable(query)`` and, with ``normalize_query=True``,
    ``is_searchable(query, normalized=True)`` hold; a caller that holds many queries can
    check them all before the first search.
    """
    if normalized:
        text = unicodedata.normalize(QUERY_NORMALIZATION, text)
    return any(unicodedata.category(character)[0] in "LNSP" for character in text)


def _values(name: str, value: object) -> tuple[str, ...] | None:
    """One list field of the filters as a tuple without duplicates; ``SearchError`` otherwise."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        # A bare string is a sequence too: it would be read as one value per character.
        raise SearchError(
            f"{name} must be a list of strings, got the single string {value!r}: write [{value!r}]"
        )
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise SearchError(f"{name} must be a list of strings or None, got {type(value).__name__}")
    if not value:
        raise SearchError(
            f"{name} is an empty list, which could mean 'match nothing' as well as 'no restriction': "
            "use None for no restriction"
        )
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SearchError(f"{name} must hold non-blank strings, got {item!r}")
        if not _encodable(item):
            raise _not_encodable(name, "holds", item)
    ordered = sorted(value) if isinstance(value, (set, frozenset)) else value
    return tuple(dict.fromkeys(ordered))


def _day(name: str, value: object) -> datetime.date | None:
    """One date bound as a ``datetime.date``; ``SearchError`` for anything but a full date."""
    if value is None:
        return None
    # A datetime is a date too, but its time of day and its time zone would be dropped.
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str) and _DATE.fullmatch(value):
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            raise SearchError(f"{name} is not a date of the calendar: {value!r}") from None
    raise SearchError(
        f"{name} must be a datetime.date or a string written YYYY-MM-DD (a full date, e.g. '2016-03-10'), "
        f"got {value!r}"
    )


def _midnight(day: datetime.date) -> datetime.datetime:
    return datetime.datetime(day.year, day.month, day.day, tzinfo=datetime.timezone.utc)


@dataclass(frozen=True)
class SearchFilters:
    """Which chunks a search may return. Every field is optional; ``None`` does not restrict.

    ``source_codes``, ``doc_types``, ``languages``, ``doc_ids``
        Accepted values of the payload fields ``source_code``, ``doc_type``, ``language`` and
        ``doc_id``: a list (or tuple, or set) of non-blank strings, compared exactly, as they
        are (case and surrounding whitespace included).
    ``date_from``, ``date_to``
        Bounds on ``date``, the publication date, **both inclusive**: a ``datetime.date`` or
        a string written exactly ``YYYY-MM-DD``.
    ``corpus``
        The one accepted value of the payload field ``corpus``: a non-blank string.

    How they combine:

    - the fields are combined with AND: a chunk must pass every field that is set;
    - the values of one list are combined with OR: a chunk passes when its value is any of
      them;
    - a value no chunk carries is not an error: the search returns nothing;
    - a chunk whose payload lacks a filtered field does not pass that field.

    Refused at construction, with a :class:`SearchError`:

    - a bare string where a list is expected (``source_codes="ecb"``): it would be read as
      one value per character. Write ``["ecb"]``;
    - **an empty list**: it could mean "match nothing" as well as "no restriction", and the
      two give opposite results. ``None`` is the only way to write "no restriction";
    - a blank or non-string value in a list, a blank or non-string ``corpus``;
    - a value of a list, or a ``corpus``, that cannot be encoded as UTF-8: a string holding a
      lone surrogate, which a JSON escape such as ``"\\ud800"`` gives. No payload holds such a
      value, and the Qdrant client could not send it;
    - a date bound that is not a full date: ``"2016"``, ``"2016-03"``, ``"10/03/2016"``, a
      date with a time, a ``datetime.datetime``, a number, a day that does not exist;
    - ``date_from`` after ``date_to``.

    Dates. A bound is a whole day in UTC, the time zone Qdrant reads a stored date without
    time in: ``date_from="2016-03-10"`` starts at 2016-03-10T00:00:00Z, and
    ``date_to="2016-03-10"`` includes everything up to the end of that day, so a document
    stored as ``2016-03-10`` or as ``2016-03-10T15:30:00Z`` passes both. **As soon as one
    bound is set, a chunk without a readable full date is left out**: no ``date`` key, a
    null, an empty string, or an imprecise date such as ``2016-03`` or ``2016``.

    The filters are stored in a fixed form, whatever was given: the lists as tuples without
    duplicates, the dates as ``datetime.date``. The object is immutable and hashable, and a
    list given by the caller is neither kept nor changed.
    """

    source_codes: tuple[str, ...] | None = None
    doc_types: tuple[str, ...] | None = None
    languages: tuple[str, ...] | None = None
    doc_ids: tuple[str, ...] | None = None
    date_from: datetime.date | None = None
    date_to: datetime.date | None = None
    corpus: str | None = None

    def __post_init__(self) -> None:
        for name, _key in _LIST_FIELDS:
            object.__setattr__(self, name, _values(name, getattr(self, name)))
        for name in ("date_from", "date_to"):
            object.__setattr__(self, name, _day(name, getattr(self, name)))
        if self.date_from is not None and self.date_to is not None and self.date_from > self.date_to:
            raise SearchError(
                f"date_from ({self.date_from.isoformat()}) is after date_to ({self.date_to.isoformat()})"
            )
        if self.corpus is not None and (not isinstance(self.corpus, str) or not self.corpus.strip()):
            raise SearchError(f"corpus must be a non-blank string or None, got {self.corpus!r}")
        if self.corpus is not None and not _encodable(self.corpus):
            raise _not_encodable("corpus", "is", self.corpus)


def _qdrant_filter(filters: SearchFilters | None) -> models.Filter | None:
    """The Qdrant filter of ``filters``; ``None`` when nothing restricts the search."""
    if filters is None:
        return None
    must: list[models.FieldCondition] = []
    if filters.corpus is not None:
        must.append(models.FieldCondition(key="corpus", match=models.MatchValue(value=filters.corpus)))
    for name, key in _LIST_FIELDS:
        accepted = getattr(filters, name)
        if accepted is not None:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(accepted))))
    if filters.date_from is not None or filters.date_to is not None:
        # The bounds are sent as datetimes with their time zone, never as text: given the
        # string "2016", qdrant-client reads a Unix timestamp and matches almost everything.
        lower = _midnight(filters.date_from) if filters.date_from is not None else None
        # Inclusive upper bound = strictly before the next day at 00:00 UTC. A stored date
        # may carry a time ("2016-03-10T15:30:00Z"); "lte 2016-03-10" would mean midnight
        # and leave that document out of its own day, and "lte 23:59:59.999999" depends on
        # the precision dates are stored with. "lt the next day" depends on neither.
        upper = None
        if filters.date_to is not None and filters.date_to < datetime.date.max:
            upper = _midnight(filters.date_to + _ONE_DAY)
        if lower is not None or upper is not None:
            must.append(models.FieldCondition(key="date", range=models.DatetimeRange(gte=lower, lt=upper)))
        else:
            # date_to is the last day of the calendar and has no next day: every readable
            # date passes, and a chunk without one is still left out.
            first = _midnight(datetime.date.min)
            must.append(models.FieldCondition(key="date", range=models.DatetimeRange(gte=first)))
    return models.Filter(must=must) if must else None


@dataclass(frozen=True)
class Hit:
    """One chunk returned by :func:`search`.

    ``text``
        The text of the chunk, as stored.
    ``score``
        Cosine similarity between the query and the chunk; higher is closer.
    ``doc_id``, ``title``, ``date``, ``source_code``, ``doc_type``, ``language``, ``corpus``
        The payload values of the same name, as stored (``date`` is the publication date, a
        string). ``None`` for a key the payload does not have.
    ``page``
        Page the chunk starts on, counted from 0; ``None`` for a document without pages.
    ``chunk_number``
        0-based position of the chunk in its document.
    ``point_id``
        Id of the point in Qdrant.
    ``payload``
        The whole payload of the point, for the keys that have no field here (``year``,
        ``filename``, ``ingestion_date``, ``sha256``, ...). It is a plain dict owned by the
        hit; because of it a hit is not hashable.
    """

    text: str | None
    score: float
    doc_id: str | None
    title: str | None
    date: str | None
    source_code: str | None
    doc_type: str | None
    language: str | None
    page: int | None
    chunk_number: int | None
    corpus: str | None
    point_id: int | str
    payload: Mapping[str, Any]

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        """The hit as a new dict, ready for ``json.dumps``.

        It holds every field but ``payload``, in the order of the class; ``point_id`` is an
        integer, or a string for a UUID id. ``include_payload=True`` adds a copy of the
        payload under ``"payload"`` (it repeats ``text``, and is as JSON-ready as what was
        stored).
        """
        result = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "payload"}
        if include_payload:
            result["payload"] = dict(self.payload)
        return result


_PAYLOAD_FIELDS = (
    "text",
    "doc_id",
    "title",
    "date",
    "source_code",
    "doc_type",
    "language",
    "page",
    "chunk_number",
    "corpus",
)


def _hit(point: Any) -> Hit:
    payload = dict(point.payload or {})
    point_id = point.id if isinstance(point.id, int) else str(point.id)
    return Hit(
        score=float(point.score),
        point_id=point_id,
        payload=payload,
        **{name: payload.get(name) for name in _PAYLOAD_FIELDS},
    )


def _limit(name: str, value: object) -> int:
    if type(value) is not int or value < 1:  # a bool is an int subclass, and is not a count
        raise SearchError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def search(
    query: str,
    *,
    collection: str,
    store: QdrantStore,
    embedder: Embedder,
    filters: SearchFilters | None = None,
    limit: int | None = None,
    max_limit: int | None = None,
    normalize_query: bool = True,
) -> list[Hit]:
    """Return the chunks of ``collection`` closest to ``query``, best first.

    The query is embedded with ``embedder.encode_query`` (which adds the query prefix of the
    model), and Qdrant is asked once for the ``limit`` chunks with the highest cosine
    similarity among those that pass ``filters``. There may be fewer, or none: a filter value
    that no chunk carries is not an error. Hits come in decreasing ``score``; several chunks
    of one document may be returned, nothing is grouped.

    **This function reads no settings and no environment variable.** Its defaults are fixed
    (``Settings.search_limit`` and ``Settings.search_max_limit`` as the class defines them:
    10 and 50); a caller that keeps its configuration in
    :class:`rag_engine.config.Settings` passes the values itself::

        search(query, collection=..., store=store, embedder=embedder,
               limit=settings.search_limit, max_limit=settings.search_max_limit)

    ``RAG_ENGINE_SEARCH_LIMIT`` and ``RAG_ENGINE_SEARCH_MAX_LIMIT`` have no effect otherwise.

    Arguments:

    - ``query``: a non-blank string that holds something to search for: at least one letter,
      number, symbol or punctuation mark (Unicode general categories L, N, S and P), in any
      script. ``"2016"``, ``"€"`` and ``"$AAPL"`` are queries. A query made only of combining
      marks, of zero-width or other format characters, of control characters and of spaces
      is refused: it would be embedded, and a full ranked list returned for a question that
      asks nothing. **Punctuation alone (``"???"``) is accepted**: it is a degenerate query,
      but one that was typed. The rule is applied to the query as given and, when
      ``normalize_query`` is on, to its normalised form too: NFKD writes the spacing form of
      a diacritic (``"¨"``, ``"´"``) as a space and a combining mark, so such a query alone
      is refused, and is accepted with ``normalize_query=False``. The query is embedded as
      given, apart from the normalisation below; it is not stripped.
    - ``collection`` must exist, with the dimension of the embedder. **This function does not
      check it** (no extra request per search): the caller calls
      ``store.ensure_collection(collection, embedder.dim)`` once. The payload indexes
      (``store.ensure_payload_indexes``) make the filters fast; the answers are the same
      without them.
    - ``filters``: a :class:`SearchFilters`, or ``None`` for no restriction.
    - ``limit``: the largest number of hits wanted, an integer of at least 1; ``None`` means
      the default, 10.
    - ``max_limit``: the largest ``limit`` accepted, an integer of at least 1; ``None`` means
      the default, 50. **A ``limit`` above it is refused, never lowered silently**: the
      error names the maximum. This holds for the default limit too, when ``max_limit`` is
      set below it.
    - ``normalize_query``: put the query in Unicode NFKD (:data:`QUERY_NORMALIZATION`) before
      embedding it. The chunker stores, and embeds, the text of the chunks in that form. For
      most languages the tokenizer of the default model gives the same tokens either way;
      for Japanese (voiced kana), Korean and Vietnamese it does not, and only an NFKD query
      is cut into the tokens the chunks were cut into. Leave it on; ``False`` is for a
      collection whose passages were embedded in another form.

    Errors. :class:`SearchError` (a ``ValueError`` and a ``RagEngineError``), raised before
    the embedder or the store is called: a ``query`` or a ``collection`` that is not a
    non-blank string or that cannot be encoded as UTF-8 (a lone surrogate), a ``query``
    without any searchable character, a ``filters`` that is not a ``SearchFilters``, a ``limit`` or a
    ``max_limit`` that is not an integer >= 1, a ``limit`` above ``max_limit``, a
    ``normalize_query`` that is not a boolean. Nothing else is caught: an error of the
    embedder is raised unchanged, a refused API key is a
    :class:`rag_engine.store.QdrantAuthError`, and a collection that does not exist, a
    collection of another dimension or an unreachable server is raised as the Qdrant client
    raised it.
    """
    if not isinstance(query, str) or not query.strip():
        raise SearchError(f"query must be a non-blank string, got {query!r}")
    if not _encodable(query):
        raise _not_encodable("query", "is", query)
    if not is_searchable(query):
        raise SearchError(
            "the query has no searchable character: it holds no letter, number, symbol or punctuation "
            f"mark, only combining marks, invisible or control characters and spaces, got {query!a}"
        )
    if not isinstance(collection, str) or not collection.strip():
        raise SearchError(f"collection must be a non-blank string, got {collection!r}")
    if not _encodable(collection):
        raise _not_encodable("collection", "is", collection)
    if filters is not None and not isinstance(filters, SearchFilters):
        raise SearchError(f"filters must be a SearchFilters or None, got {type(filters).__name__}")
    if not isinstance(normalize_query, bool):
        raise SearchError(f"normalize_query must be True or False, got {normalize_query!r}")
    wanted = Settings.search_limit if limit is None else _limit("limit", limit)
    maximum = Settings.search_max_limit if max_limit is None else _limit("max_limit", max_limit)
    if wanted > maximum:
        raise SearchError(f"limit {wanted} is above the maximum of {maximum}: ask for at most {maximum} hits")

    text = unicodedata.normalize(QUERY_NORMALIZATION, query) if normalize_query else query
    if not is_searchable(text):
        # The spacing form of a diacritic is a symbol; its normalised form is a space and a mark.
        raise SearchError(
            f"the query has no searchable character once put in {QUERY_NORMALIZATION}: it holds only "
            "isolated accents or other marks, which become a space and a combining mark, "
            f"got {query!a}"
        )
    vector = embedder.encode_query(text)
    points = store.query(
        collection,
        vector.tolist() if hasattr(vector, "tolist") else list(vector),
        query_filter=_qdrant_filter(filters),
        limit=wanted,
    )
    return [_hit(point) for point in points]
