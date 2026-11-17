"""Measure the search on a set of questions whose answers are known.

A question names the documents that answer it. :func:`evaluate` runs
:func:`rag_engine.search.search` for every question, turns the chunks returned into a ranking
of documents, and reports where the expected documents stand in it.

The question file (:func:`load_questions`) is JSON Lines, one question per line::

    {"id": "q-001", "question": "What did the ECB decide in March 2016?",
     "expected_doc_ids": ["a5b8ba5e477ad19a"],
     "filters": {"source_codes": ["ecb"], "date_from": "2016-03-01", "date_to": "2016-03-31"},
     "kind": "analytical", "tags": ["bank:ecb"], "language": "en", "rationale": "..."}

``id``, ``question`` and ``expected_doc_ids`` are required. ``filters`` holds the fields of
:class:`rag_engine.search.SearchFilters` under their names and is applied to the search as
it is. ``kind`` and ``tags`` only group the results in the report; ``language`` and
``rationale`` are kept and not used. Any other field is refused: a misspelt ``filters``
must not be ignored silently.

Scoring is done on documents, not on chunks (:func:`evaluate` says how), and a question
whose expected documents are not in the collection at all is reported apart instead of
being counted as a failure of the search.

The report (:class:`EvalReport`, :func:`format_report`) gives every group of questions two
means: over the questions, and over the documents they expect, the questions of one document
being averaged first (they are correlated). Its headline is the second one, on the kinds of
question asked without filters, with an interval obtained by resampling the documents.

Importing this module loads no embedding backend, and needs the standard library only
besides the engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, overload

from . import __version__
from .config import Settings
from .embedding import Embedder
from .errors import RagEngineError
from .search import QUERY_NORMALIZATION, SearchError, SearchFilters, is_searchable, search
from .store import QdrantStore

__all__ = [
    "BOOTSTRAP_SEED",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CHUNK_LIMIT",
    "DEFAULT_INTERVAL_MIN_DOCUMENTS",
    "DEFAULT_K",
    "FILTER_FIELDS",
    "INTERVAL_LEVEL",
    "QUESTION_FIELDS",
    "Aggregate",
    "EvalError",
    "EvalReport",
    "Means",
    "NotInCollection",
    "Question",
    "QuestionResult",
    "QuestionSet",
    "RankedDocument",
    "evaluate",
    "format_report",
    "load_questions",
]

#: Number of documents the metrics look at, unless ``k`` is given.
DEFAULT_K = 10
#: Number of chunks asked to the search for one question, unless ``chunk_limit`` is given.
DEFAULT_CHUNK_LIMIT = 50
#: Number of times the documents are drawn again for an interval, unless ``bootstrap_resamples`` is given.
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
#: Fewest documents a group needs to be given an interval, unless ``interval_min_documents`` is given.
DEFAULT_INTERVAL_MIN_DOCUMENTS = 20
#: Seed of the resampling: fixed, so that the same results always give the same intervals.
BOOTSTRAP_SEED = 1
#: Level of the intervals: the 2.5th and the 97.5th percentiles of the resampled means.
INTERVAL_LEVEL = 0.95

#: The fields a filter of the question file may hold: those of ``SearchFilters``.
FILTER_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(SearchFilters))
#: The fields a question of the question file may hold.
QUESTION_FIELDS: tuple[str, ...] = (
    "id",
    "question",
    "expected_doc_ids",
    "filters",
    "kind",
    "tags",
    "language",
    "rationale",
)

_REQUIRED = ("id", "question", "expected_doc_ids")
_NO_KIND = "(none)"
_MISS_DOCUMENTS = 3
_METRICS = ("hit_at_1", "hit_at_k", "mrr", "recall_at_k")
_INTERVAL_METRICS = ("hit_at_1", "hit_at_k", "mrr")


class EvalError(RagEngineError, ValueError):
    """A question, a question file or an argument of the evaluation cannot be used as it is."""


def _strings(name: str, value: object, *, allow_empty: bool) -> tuple[str, ...]:
    """A list field of a question as a tuple of non-blank strings; ``EvalError`` otherwise."""
    if isinstance(value, (str, bytes)):
        raise EvalError(f"'{name}' must be a list of strings, got the single string {value!r}")
    if not isinstance(value, (list, tuple)):
        raise EvalError(f"'{name}' must be a list of strings, got {type(value).__name__}")
    if not value and not allow_empty:
        raise EvalError(f"'{name}' is empty: a question names at least one document that answers it")
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise EvalError(f"'{name}' must be a list of non-blank strings, got {item!r}")
        if item in seen:
            raise EvalError(f"'{name}' names {item!r} twice")
        seen.add(item)
    return tuple(value)


def _filters_dict(filters: SearchFilters | None) -> dict[str, Any] | None:
    """``filters`` as the question file writes them: the fields that are set, ready for JSON."""
    if filters is None:
        return None
    result: dict[str, Any] = {}
    for name in FILTER_FIELDS:
        value = getattr(filters, name)
        if value is None:
            continue
        if isinstance(value, tuple):
            result[name] = list(value)
        elif isinstance(value, str):
            result[name] = value
        else:
            result[name] = value.isoformat()
    return result or None


@dataclass(frozen=True)
class Question:
    """One question and the documents that answer it.

    ``id``
        A non-blank string, unique in its set.
    ``question``
        The text given to the search: a non-blank string with something to search for.
    ``expected_doc_ids``
        The ``doc_id`` of every document that answers the question: at least one, none twice.
    ``filters``
        The :class:`rag_engine.search.SearchFilters` the search runs with, or ``None``.
    ``kind``, ``tags``
        Labels the report groups the results by: one optional string, and a list of strings.
    ``language``, ``rationale``
        Free text, kept and not used.
    ``line``
        Line of the question in its file, counted from 1; ``None`` for a question built in code.

    The fields are checked when the question is built (``EvalError``); lists are kept as
    tuples.
    """

    id: str
    question: str
    expected_doc_ids: tuple[str, ...]
    filters: SearchFilters | None = None
    kind: str | None = None
    tags: tuple[str, ...] = ()
    language: str | None = None
    rationale: str | None = None
    line: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise EvalError(f"'id' must be a non-blank string, got {self.id!r}")
        if not isinstance(self.question, str) or not self.question.strip():
            raise EvalError(f"'question' must be a non-blank string, got {self.question!r}")
        if not is_searchable(self.question):
            raise EvalError(f"'question' has no searchable character, got {self.question!a}")
        expected = _strings("expected_doc_ids", self.expected_doc_ids, allow_empty=False)
        object.__setattr__(self, "expected_doc_ids", expected)
        if self.filters is not None and not isinstance(self.filters, SearchFilters):
            raise EvalError(f"'filters' must be a SearchFilters or None, got {type(self.filters).__name__}")
        if self.kind is not None and (not isinstance(self.kind, str) or not self.kind.strip()):
            raise EvalError(f"'kind' must be a non-blank string, got {self.kind!r}")
        object.__setattr__(self, "tags", _strings("tags", self.tags, allow_empty=True))
        for name in ("language", "rationale"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise EvalError(f"'{name}' must be a string, got {value!r}")

    def to_dict(self) -> dict[str, Any]:
        """The question as its line of the question file: the fields that are set."""
        result: dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "expected_doc_ids": list(self.expected_doc_ids),
        }
        filters = _filters_dict(self.filters)
        if filters is not None:
            result["filters"] = filters
        if self.kind is not None:
            result["kind"] = self.kind
        if self.tags:
            result["tags"] = list(self.tags)
        for name in ("language", "rationale"):
            if getattr(self, name) is not None:
                result[name] = getattr(self, name)
        return result


@dataclass(frozen=True)
class QuestionSet(Sequence[Question]):
    """The questions of one file, in the order of the file, with the identity of the file.

    A sequence of :class:`Question`. ``path`` is the path it was read from, as given, and
    ``sha256`` the hash of the bytes of the file: the report records both, so that two
    reports can be told to answer the same questions.
    """

    questions: tuple[Question, ...]
    path: str | None = None
    sha256: str | None = None

    @overload
    def __getitem__(self, index: int) -> Question: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[Question, ...]: ...
    def __getitem__(self, index: int | slice) -> Question | tuple[Question, ...]:
        return self.questions[index]

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions)


def _question_of(record: object) -> Question:
    """The question of one parsed line; ``EvalError`` or ``SearchError`` when it is not one."""
    if not isinstance(record, dict):
        raise EvalError(f"a line must hold one JSON object, got {type(record).__name__}")
    for name in record:
        if name not in QUESTION_FIELDS:
            raise EvalError(f"unknown field {name!r} (the fields are: {', '.join(QUESTION_FIELDS)})")
    for name in _REQUIRED:
        if name not in record:
            raise EvalError(f"'{name}' is missing")
    filters = record.get("filters")
    if filters is not None:
        if not isinstance(filters, dict):
            raise EvalError(f"'filters' must be a JSON object, got {type(filters).__name__}")
        # An unknown filter, a bad value: a SearchError, reported with the line like an EvalError.
        filters = SearchFilters.from_mapping(filters)
        if _filters_dict(filters) is None:
            filters = None  # {} and a filter of nulls restrict nothing
    for name in ("kind", "language", "rationale"):
        if record.get(name) is not None and not isinstance(record[name], str):
            raise EvalError(f"'{name}' must be a string, got {record[name]!r}")
    question = Question(
        id=record["id"],
        question=record["question"],
        expected_doc_ids=record["expected_doc_ids"],
        filters=filters,
        kind=record.get("kind"),
        tags=record["tags"] if record.get("tags") is not None else (),
        language=record.get("language"),
        rationale=record.get("rationale"),
    )
    _check_normalised(question)
    return question


def _check_normalised(question: Question) -> None:
    """``EvalError`` when the text of ``question`` has nothing to search for once normalised."""
    if not is_searchable(question.question, normalized=True):
        raise EvalError(
            f"'question' has no searchable character once put in {QUERY_NORMALIZATION}, the form it is "
            f"searched in: it holds only isolated accents or other marks, got {question.question!a}"
        )


def load_questions(path: str | os.PathLike[str]) -> QuestionSet:
    """Read and check a question file; returns its questions in the order of the file.

    The file is UTF-8 JSON Lines (a byte order mark is accepted, blank lines are skipped).
    Each line is one object with the fields of :data:`QUESTION_FIELDS`:

    - ``id``: a non-blank string, used once in the file;
    - ``question``: a non-blank string that holds something to search for, as given and
      once put in NFKD, the form the search embeds it in (an isolated accent such as
      ``"\u00b4"`` is a symbol as given, and a space and a combining mark in NFKD);
    - ``expected_doc_ids``: a non-empty list of non-blank strings, none twice;
    - ``filters`` (optional): an object whose keys are fields of
      :class:`rag_engine.search.SearchFilters` (:data:`FILTER_FIELDS`) and whose values
      follow its rules: lists of strings, never a bare string or an empty list; dates
      written ``YYYY-MM-DD``; ``date_from`` not after ``date_to``. A null value, and an
      empty object, restrict nothing;
    - ``kind`` (optional string), ``tags`` (optional list of strings): labels the report
      groups by;
    - ``language``, ``rationale`` (optional strings): kept, not used.

    An optional field may be left out or null. Nothing is guessed or repaired. The first
    line that breaks a rule raises an :class:`EvalError` naming the file and the line number:
    text that is not JSON or not an object, a missing or unknown field, an unknown filter, a
    value of the wrong type, a bad date, an id already used (the error names the line of its
    first use). A file without any question, and a file that is not UTF-8, are ``EvalError``
    too. A file that cannot be read raises the ``OSError`` of the system.
    """
    name = os.fspath(path)
    with open(name, "rb") as handle:
        raw = handle.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EvalError(
            f"{name}: the question file is not UTF-8 ({exc.reason} at byte {exc.start})"
        ) from None
    questions: list[Question] = []
    first_use: dict[str, int] = {}
    # Split on line feeds only: str.splitlines also cuts on characters a JSON string may hold.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise EvalError(f"not valid JSON ({exc})") from None
            question = _question_of(record)
            if question.id in first_use:
                raise EvalError(f"duplicate id {question.id!r} (first used on line {first_use[question.id]})")
        except (EvalError, SearchError) as exc:
            raise EvalError(f"{name}, line {number}: {exc}") from None
        first_use[question.id] = number
        object.__setattr__(question, "line", number)
        questions.append(question)
    if not questions:
        raise EvalError(f"{name}: the file holds no question")
    return QuestionSet(tuple(questions), path=name, sha256=hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True)
class RankedDocument:
    """One document of the ranking of a question.

    ``score`` is the score of its best chunk among those returned, ``title`` the title that
    chunk carries (``None`` without one), and ``chunks`` the number of chunks of the document
    among those returned.
    """

    doc_id: str
    title: str | None
    score: float
    chunks: int

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "title": self.title, "score": self.score, "chunks": self.chunks}


@dataclass(frozen=True)
class QuestionResult:
    """What the search gave for one question that was scored.

    ``absent_doc_ids``
        The expected documents that are not in the collection (the others are).
    ``rank``
        Rank, from 1, of the best-placed expected document among the documents ranked;
        ``None`` when none of them was ranked. It may be greater than ``k``.
    ``hit_at_1``, ``hit_at_k``
        Whether that rank is 1; whether it is at most ``k``.
    ``reciprocal_rank``
        ``1 / rank`` when the rank is at most ``k``, else 0.
    ``recall_at_k``
        Share of the expected documents present in the collection that are among the first
        ``k`` documents.
    ``chunks_returned``, ``documents_ranked``
        Number of chunks the search returned, and of distinct documents among them.
    ``chunk_budget_exhausted``
        Fewer than ``k`` documents were ranked although the search returned as many chunks
        as it was asked for: more chunks could have ranked more documents.
    ``tied_at_decision``
        ``hit_at_1`` or ``hit_at_k`` depends on the order given to documents of exactly the
        same score: an expected document is tied with documents that are not expected, across
        rank 1 or rank ``k``, and another order of the tied documents would change the answer.
    ``top_documents``
        The first ``k`` documents of the ranking, best first; the first three when ``k`` is
        smaller, since a miss is shown with the three best documents.
    """

    id: str
    question: str
    kind: str | None
    tags: tuple[str, ...]
    expected_doc_ids: tuple[str, ...]
    absent_doc_ids: tuple[str, ...]
    filters: SearchFilters | None
    rank: int | None
    hit_at_1: bool
    hit_at_k: bool
    reciprocal_rank: float
    recall_at_k: float
    chunks_returned: int
    documents_ranked: int
    chunk_budget_exhausted: bool
    top_documents: tuple[RankedDocument, ...]
    tied_at_decision: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.id, "question": self.question}
        if self.kind is not None:
            result["kind"] = self.kind
        if self.tags:
            result["tags"] = list(self.tags)
        result["expected_doc_ids"] = list(self.expected_doc_ids)
        result["absent_doc_ids"] = list(self.absent_doc_ids)
        filters = _filters_dict(self.filters)
        if filters is not None:
            result["filters"] = filters
        for name in (
            "rank",
            "hit_at_1",
            "hit_at_k",
            "reciprocal_rank",
            "recall_at_k",
            "chunks_returned",
            "documents_ranked",
            "chunk_budget_exhausted",
            "tied_at_decision",
        ):
            result[name] = getattr(self, name)
        result["top_documents"] = [document.to_dict() for document in self.top_documents]
        return result

    def _miss(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_doc_ids": list(self.expected_doc_ids),
            "rank": self.rank,
            "top_documents": [document.to_dict() for document in self.top_documents[:_MISS_DOCUMENTS]],
        }


@dataclass(frozen=True)
class NotInCollection:
    """A question left out of the metrics: none of its expected documents is in the collection."""

    id: str
    question: str
    expected_doc_ids: tuple[str, ...]
    kind: str | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "expected_doc_ids": list(self.expected_doc_ids),
        }
        if self.kind is not None:
            result["kind"] = self.kind
        if self.tags:
            result["tags"] = list(self.tags)
        return result


@dataclass(frozen=True)
class Means:
    """The four metrics of a group, each a mean: ``hit_at_1``, ``hit_at_k``, ``mrr``, ``recall_at_k``."""

    hit_at_1: float
    hit_at_k: float
    mrr: float
    recall_at_k: float

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in _METRICS}


def _values(result: QuestionResult) -> tuple[float, float, float, float]:
    return (float(result.hit_at_1), float(result.hit_at_k), result.reciprocal_rank, result.recall_at_k)


def _mean(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """The mean of each column of ``rows``, summed exactly so that the order does not matter."""
    return tuple(math.fsum(column) / len(rows) for column in zip(*rows, strict=True))


def _per_document(results: Sequence[QuestionResult]) -> dict[str, tuple[float, ...]]:
    """For each expected document that is in the collection, the means of the questions that expect it."""
    questions: dict[str, list[tuple[float, float, float, float]]] = {}
    for result in results:
        values = _values(result)
        for doc_id in result.expected_doc_ids:
            if doc_id not in result.absent_doc_ids:
                questions.setdefault(doc_id, []).append(values)
    return {doc_id: _mean(rows) for doc_id, rows in sorted(questions.items())}


def _percentile(ordered: Sequence[float], share: float) -> float:
    """The value at ``share`` of ``ordered`` (nearest rank)."""
    return ordered[min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))]


def _intervals(
    documents: Sequence[tuple[float, ...]], *, resamples: int, seed: object
) -> dict[str, tuple[float, float]]:
    """Percentile intervals of the mean over ``documents`` of hit@1, hit@k and MRR.

    The documents are drawn with replacement, as many as there are, ``resamples`` times;
    the interval of a metric goes from the 2.5th to the 97.5th percentile of its means.
    """
    generator = random.Random(seed)
    count = len(documents)
    places = range(count)
    draws: list[list[float]] = [[] for _name in _INTERVAL_METRICS]
    for _ in range(resamples):
        drawn = generator.choices(places, k=count)
        for column, values in enumerate(draws):
            values.append(math.fsum(documents[place][column] for place in drawn) / count)
    low, high = (1.0 - INTERVAL_LEVEL) / 2.0, (1.0 + INTERVAL_LEVEL) / 2.0
    result = {}
    for name, values in zip(_INTERVAL_METRICS, draws, strict=True):
        values.sort()
        result[name] = (_percentile(values, low), _percentile(values, high))
    return result


@dataclass(frozen=True)
class Aggregate:
    """The metrics of a group of scored questions.

    ``questions``
        Number of questions of the group.
    ``documents``
        Number of distinct documents these questions expect, among those that are in the
        collection.
    ``hit_at_1``, ``hit_at_k``, ``mrr``, ``recall_at_k``
        The means **over the questions**: the shares of questions whose best expected
        document is first, and among the first ``k``; the mean reciprocal rank (cut at
        ``k``); the mean recall. ``None`` for an empty group.
    ``per_document``
        The same four as means **over the documents** (:class:`Means`): for each document,
        the mean over the questions that expect it, then the mean of these over the
        documents. The questions of one document are correlated (the same document is
        easy or hard to find however it is asked for), so this is the mean whose
        uncertainty the number of documents tells. A question that expects several
        documents counts for each of them. ``None`` for an empty group.
    ``intervals``
        For ``hit_at_1``, ``hit_at_k`` and ``mrr``, the bounds ``(low, high)`` of a 95%
        interval of the per-document mean, from resampling the documents with replacement
        (a percentile bootstrap, with a fixed seed). ``None`` when no interval was asked
        for this group, or when it has too few documents.
    ``few_documents``
        An interval was asked and not given: the group has fewer documents than the minimum.
    """

    questions: int
    documents: int
    hit_at_1: float | None
    hit_at_k: float | None
    mrr: float | None
    recall_at_k: float | None
    per_document: Means | None = None
    intervals: Mapping[str, tuple[float, float]] | None = None
    few_documents: bool = False

    @classmethod
    def of(
        cls,
        results: Sequence[QuestionResult],
        *,
        interval: str | None = None,
        resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
        min_documents: int = DEFAULT_INTERVAL_MIN_DOCUMENTS,
    ) -> Aggregate:
        """The aggregate of ``results``.

        ``interval`` asks for the intervals: it is the name of the group, which with
        :data:`BOOTSTRAP_SEED` seeds its resampling, so that the interval of a group does
        not depend on the other groups. They are computed from ``resamples`` draws when the
        group has at least ``min_documents`` documents.
        """
        if not results:
            return cls(0, 0, None, None, None, None)
        documents = _per_document(results)
        per_question = _mean([_values(result) for result in results])
        per_document = Means(*_mean(list(documents.values()))) if documents else None
        intervals = None
        few = False
        if interval is not None:
            few = len(documents) < min_documents
            if not few:
                intervals = _intervals(
                    list(documents.values()), resamples=resamples, seed=f"{BOOTSTRAP_SEED}:{interval}"
                )
        return cls(len(results), len(documents), *per_question, per_document, intervals, few)

    def to_dict(self) -> dict[str, Any]:
        """The aggregate as a new dict, ready for ``json.dumps``: the exact values."""
        result: dict[str, Any] = {"questions": self.questions, "documents": self.documents}
        result.update({name: getattr(self, name) for name in _METRICS})
        result["per_document"] = None if self.per_document is None else self.per_document.to_dict()
        result["intervals"] = (
            None
            if self.intervals is None
            else {name: list(bounds) for name, bounds in self.intervals.items()}
        )
        result["few_documents"] = self.few_documents
        return result


@dataclass(frozen=True)
class EvalReport:
    """The result of :func:`evaluate`.

    ``parameters``
        What the evaluation ran with: ``collection``, ``k``, ``chunk_limit``, ``max_limit``,
        ``model`` and ``backend`` of the embedder, its ``query_prefix``, ``normalize_query``
        and ``query_normalization``, ``questions_file`` and ``questions_sha256`` (``None``
        for questions not read from a file), ``bootstrap_resamples``, ``bootstrap_seed``,
        ``interval_level`` and ``interval_min_documents``, ``engine_version``.
    ``results``
        One :class:`QuestionResult` per scored question, in the order of the questions.
    ``not_in_collection``
        The questions left out because none of their expected documents is in the
        collection. They count in no metric.

    Everything else is derived from these:

    ``by_kind``, ``by_tag``
        One :class:`Aggregate` per ``kind`` (``"(none)"`` for the questions without one) and
        per tag. The kinds carry intervals, the tags do not.
    ``filtered_kinds``, ``headline_kinds``
        The kinds with at least one scored question that carries filters, and the others,
        both sorted. Filters narrow the candidates: when they leave fewer than ``k``
        documents, ``hit_at_k`` and ``recall_at_k`` are true by construction, so a filtered
        kind is read on ``hit_at_1`` and ``mrr`` only and is kept out of the headline.
    ``headline``
        The :class:`Aggregate`, with intervals, of the questions of the ``headline_kinds``:
        the kinds asked without filters. Its ``per_document`` means are the headline of the
        report, and what ``rag-engine eval --fail-under-hit-at-k`` looks at.
    ``overall``
        The :class:`Aggregate` of all the scored questions, pooled: every kind, filtered or
        not, each question counted once. Kept for reference; it is not the headline.

    and the :attr:`misses` and the counts.
    """

    parameters: Mapping[str, Any]
    results: tuple[QuestionResult, ...]
    not_in_collection: tuple[NotInCollection, ...] = ()
    overall: Aggregate = field(init=False)
    headline: Aggregate = field(init=False)
    headline_kinds: tuple[str, ...] = field(init=False)
    filtered_kinds: tuple[str, ...] = field(init=False)
    by_kind: dict[str, Aggregate] = field(init=False)
    by_tag: dict[str, Aggregate] = field(init=False)

    def __post_init__(self) -> None:
        kinds: dict[str, list[QuestionResult]] = {}
        tags: dict[str, list[QuestionResult]] = {}
        filtered: set[str] = set()
        for result in self.results:
            kind = result.kind if result.kind is not None else _NO_KIND
            kinds.setdefault(kind, []).append(result)
            if result.filters is not None:
                filtered.add(kind)
            for tag in result.tags:
                tags.setdefault(tag, []).append(result)
        sampling = {
            "resamples": self.parameters.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES),
            "min_documents": self.parameters.get("interval_min_documents", DEFAULT_INTERVAL_MIN_DOCUMENTS),
        }
        unfiltered = tuple(sorted(name for name in kinds if name not in filtered))
        headline = [result for result in self.results if (result.kind or _NO_KIND) in unfiltered]
        values = {
            "overall": Aggregate.of(self.results),
            "headline": Aggregate.of(headline, interval="headline", **sampling),
            "headline_kinds": unfiltered,
            "filtered_kinds": tuple(sorted(filtered)),
            "by_kind": {
                name: Aggregate.of(group, interval=f"kind:{name}", **sampling)
                for name, group in kinds.items()
            },
            "by_tag": {name: Aggregate.of(group) for name, group in tags.items()},
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def misses(self) -> tuple[QuestionResult, ...]:
        """The scored questions without any expected document among the first ``k``."""
        return tuple(result for result in self.results if not result.hit_at_k)

    @property
    def partially_absent(self) -> int:
        """Number of scored questions with some, not all, expected documents out of the collection."""
        return sum(1 for result in self.results if result.absent_doc_ids)

    @property
    def fewer_than_k_documents(self) -> int:
        """Number of scored questions for which fewer than ``k`` documents were ranked."""
        k = self.parameters["k"]
        return sum(1 for result in self.results if result.documents_ranked < k)

    @property
    def chunk_budget_exhausted(self) -> int:
        """Among them, those whose search returned every chunk it was asked for."""
        return sum(1 for result in self.results if result.chunk_budget_exhausted)

    @property
    def tied_at_decision(self) -> int:
        """Number of scored questions whose hit@1 or hit@k hangs on the order of equal scores."""
        return sum(1 for result in self.results if result.tied_at_decision)

    def to_dict(self) -> dict[str, Any]:
        """The whole report as a new dict, ready for ``json.dumps``, with the exact values.

        Keys, in this order: ``parameters``; ``summary`` (the counts); ``metrics``
        (``headline``, ``overall``, ``by_kind``, ``by_tag``: each an aggregate with its means
        over the questions, its ``per_document`` means and its ``intervals``; ``headline``
        and ``overall`` say what they aggregate, and each kind whether it is ``filtered``);
        ``misses`` (id, question, expected ids, rank, the three best documents with their
        titles); ``not_in_collection``; ``results`` (every scored question with its first
        documents).
        """
        return {
            "parameters": dict(self.parameters),
            "summary": {
                "questions": len(self.results) + len(self.not_in_collection),
                "scored": len(self.results),
                "not_in_collection": len(self.not_in_collection),
                "partially_absent": self.partially_absent,
                "fewer_than_k_documents": self.fewer_than_k_documents,
                "chunk_budget_exhausted": self.chunk_budget_exhausted,
                "tied_at_decision": self.tied_at_decision,
            },
            "metrics": {
                "headline": {
                    "aggregation": "per document, over the kinds asked without filters",
                    "kinds": list(self.headline_kinds),
                    **self.headline.to_dict(),
                },
                "overall": {
                    "aggregation": "pooled: every scored question of every kind, counted once",
                    **self.overall.to_dict(),
                },
                "by_kind": {
                    name: {"filtered": name in self.filtered_kinds, **group.to_dict()}
                    for name, group in self.by_kind.items()
                },
                "by_tag": {name: group.to_dict() for name, group in self.by_tag.items()},
            },
            "misses": [result._miss() for result in self.misses],
            "not_in_collection": [skipped.to_dict() for skipped in self.not_in_collection],
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """:meth:`to_dict` as JSON text, non-ASCII characters written as they are."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _count(name: str, value: object) -> int:
    if type(value) is not int or value < 1:  # a bool is an int subclass, and is not a count
        raise EvalError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def _backend(embedder: object) -> str:
    """``torch``, ``onnx`` or ``fake`` for the embedders of the engine, else the class name."""
    name = type(embedder).__name__
    known = {"TorchEmbedder": "torch", "OnnxEmbedder": "onnx", "FakeEmbedder": "fake"}
    return known.get(name, name)


def _in_order(hits: Sequence[Any]) -> list[Any]:
    """``hits`` by decreasing score, then by ``doc_id``, then by ``chunk_number``.

    Qdrant returns chunks of exactly the same score in an order of its own, which may change
    from one request to the next; this order does not.
    """

    def key(hit: Any) -> tuple[float, str, int]:
        doc_id = hit.doc_id if isinstance(hit.doc_id, str) else ""
        number = hit.chunk_number if type(hit.chunk_number) is int else -1
        return (-hit.score, doc_id, number)

    return sorted(hits, key=key)


def _rank_documents(hits: Sequence[Any]) -> list[RankedDocument]:
    """The documents of ``hits``, each at the place of its first (best) chunk."""
    order: list[str] = []
    best: dict[str, Any] = {}
    chunks: dict[str, int] = {}
    for hit in hits:
        doc_id = hit.doc_id
        if not isinstance(doc_id, str) or not doc_id:
            continue  # a chunk that names no document cannot be a document of the ranking
        if doc_id not in best:
            order.append(doc_id)
            best[doc_id] = hit
        chunks[doc_id] = chunks.get(doc_id, 0) + 1
    return [
        RankedDocument(
            doc_id=doc_id,
            title=best[doc_id].title if isinstance(best[doc_id].title, str) else None,
            score=best[doc_id].score,
            chunks=chunks[doc_id],
        )
        for doc_id in order
    ]


def _tie_decides(ranking: Sequence[RankedDocument], expected: Sequence[str], within: int) -> bool:
    """Whether the order of documents of equal score decides if an expected one is among the first ``within``.

    True when the documents at ranks ``within`` and ``within + 1`` have the same score, no
    expected document is ranked above the documents of that score, some of them are
    expected, and those that are not could fill every place up to ``within``.
    """
    if len(ranking) <= within or ranking[within - 1].score != ranking[within].score:
        return False
    score = ranking[within - 1].score
    start = next(place for place, document in enumerate(ranking) if document.score == score)
    if any(document.doc_id in expected for document in ranking[:start]):
        return False
    tied = [document.doc_id in expected for document in ranking[start:] if document.score == score]
    wanted = sum(tied)
    return wanted > 0 and len(tied) - wanted >= within - start


def evaluate(
    questions: Iterable[Question],
    *,
    collection: str,
    store: QdrantStore,
    embedder: Embedder,
    k: int = DEFAULT_K,
    chunk_limit: int = DEFAULT_CHUNK_LIMIT,
    max_limit: int | None = None,
    normalize_query: bool = True,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    interval_min_documents: int = DEFAULT_INTERVAL_MIN_DOCUMENTS,
) -> EvalReport:
    """Run the search for every question and score it on documents.

    **From chunks to documents.** The search returns chunks, and several of them may belong
    to one document; an expected answer is a document. For each question
    :func:`rag_engine.search.search` is called once, with the filters of the question, for
    ``chunk_limit`` chunks (default 50). The chunks come best first; they are collapsed into
    documents by their ``doc_id``: a document takes the place and the score of its best
    chunk, so the documents are ranked in the order their first chunk appears. A chunk
    without a ``doc_id`` is ignored. The metrics look at the first ``k`` documents of that
    ranking (default 10).

    **Equal scores.** Two chunks may have exactly the same score (the same text stored in
    two documents), and the order Qdrant returns them in is not fixed. The chunks returned
    are therefore put in one order before they are collapsed: decreasing score, then
    ``doc_id``, then ``chunk_number``. A question whose ``hit_at_1`` or ``hit_at_k`` hangs on
    that order (an expected document tied with documents that are not expected, across rank
    1 or rank ``k``) is marked ``tied_at_decision`` and the report counts them. What the
    harness cannot order is what it was not given: when chunks are tied at the edge of the
    chunk budget, which of them Qdrant returns is its choice.

    ``chunk_limit`` is the chunk budget, and it is the same search a caller of the engine
    gets: it may not exceed ``max_limit`` (default 50, the default maximum of ``search``; a
    caller that keeps its configuration in :class:`rag_engine.config.Settings` passes
    ``settings.search_max_limit``). A ``chunk_limit`` above it is refused, never lowered; so
    is a ``k`` above ``chunk_limit``, which could never be reached. When the best chunks all
    come from a few long documents, ``chunk_limit`` chunks may hold fewer than ``k``
    documents. This is not hidden: each result says how many documents were ranked, and the
    report counts the questions with fewer than ``k`` (``fewer_than_k_documents``) and, among
    them, those whose search returned the full ``chunk_limit``
    (``chunk_budget_exhausted``): for these a larger budget could have ranked more
    documents, and a miss may be due to the budget. The others simply have fewer than ``k``
    documents passing their filters.

    **Documents that are not there.** Before any search, the expected documents of all the
    questions are looked up in the collection, in one batched pass
    (:meth:`rag_engine.store.QdrantStore.existing_doc_ids`). A question none of whose
    expected documents is in the collection is not searched and not scored: it is listed in
    ``not_in_collection``. Counting it as a miss would blame the search for a document that
    was never ingested. A question with some of its documents absent is scored, and the
    absent ones are listed (``absent_doc_ids``).

    **Per question** (:class:`QuestionResult`): the rank of the best-placed expected
    document (``None`` when none was ranked), ``hit_at_1`` (rank 1), ``hit_at_k`` (rank at
    most ``k``), the reciprocal rank (``1 / rank`` up to ``k``, 0 beyond: MRR is cut at
    ``k``), and ``recall_at_k``, the share of the expected documents present in the
    collection that are among the first ``k``.

    **Aggregates** (:class:`Aggregate`, :class:`EvalReport`): for each ``kind``, each tag,
    the headline (the kinds asked without filters) and all scored questions pooled, the
    means of these four over the questions and over the documents, and the number of
    distinct documents. The kinds and the headline also get a 95% interval of the
    per-document hit@1, hit@k and MRR, from ``bootstrap_resamples`` resamplings of their
    documents (default 1000, seed :data:`BOOTSTRAP_SEED`); a group with fewer than
    ``interval_min_documents`` documents (default 20) gets none and is marked
    ``few_documents``: an interval from a handful of documents would look more certain than
    it is.

    ``questions`` is a :class:`QuestionSet` (from :func:`load_questions`) or any iterable of
    :class:`Question` with distinct ids. ``collection`` must exist with the dimension of the
    embedder; it is not checked here. ``normalize_query`` is the argument of ``search``.

    Errors. :class:`EvalError`, before anything is asked, for no question, something that is
    not a ``Question``, an id used twice, a question that holds nothing to search for once
    normalised (with ``normalize_query``), a ``k``, ``chunk_limit``, ``max_limit``,
    ``bootstrap_resamples`` or ``interval_min_documents`` that is not an integer >= 1, a
    ``chunk_limit`` above the maximum, a ``k`` above ``chunk_limit``, a blank
    ``collection``, a ``normalize_query`` that is not a boolean. Everything else is
    raised as the search or the store raised it, at the first question that fails; no partial
    report is returned.
    """
    asked = tuple(questions)
    if not asked:
        raise EvalError("there is no question to evaluate")
    seen: set[str] = set()
    for question in asked:
        if not isinstance(question, Question):
            raise EvalError(f"questions must hold Question objects, got {type(question).__name__}")
        if question.id in seen:
            raise EvalError(f"duplicate id {question.id!r} among the questions")
        seen.add(question.id)
    if not isinstance(collection, str) or not collection.strip():
        raise EvalError(f"collection must be a non-blank string, got {collection!r}")
    if not isinstance(normalize_query, bool):
        raise EvalError(f"normalize_query must be True or False, got {normalize_query!r}")
    if normalize_query:
        for question in asked:  # the search would refuse it, midway through the questions
            try:
                _check_normalised(question)
            except EvalError as exc:
                raise EvalError(f"question {question.id!r}: {exc}") from None
    _count("k", k)
    _count("chunk_limit", chunk_limit)
    _count("bootstrap_resamples", bootstrap_resamples)
    _count("interval_min_documents", interval_min_documents)
    maximum = Settings.search_max_limit if max_limit is None else _count("max_limit", max_limit)
    if chunk_limit > maximum:
        raise EvalError(
            f"chunk_limit {chunk_limit} is above the maximum of {maximum} chunks a search may return "
            "(max_limit; RAG_ENGINE_SEARCH_MAX_LIMIT for the command line)"
        )
    if k > chunk_limit:
        raise EvalError(
            f"k={k} documents cannot be ranked from chunk_limit={chunk_limit} chunks: "
            "ask for at least as many chunks as documents"
        )

    wanted = sorted({doc_id for question in asked for doc_id in question.expected_doc_ids})
    present = store.existing_doc_ids(collection, wanted)

    results: list[QuestionResult] = []
    skipped: list[NotInCollection] = []
    for question in asked:
        absent = tuple(doc_id for doc_id in question.expected_doc_ids if doc_id not in present)
        reachable = [doc_id for doc_id in question.expected_doc_ids if doc_id in present]
        if not reachable:
            skipped.append(
                NotInCollection(
                    question.id, question.question, question.expected_doc_ids, question.kind, question.tags
                )
            )
            continue
        hits = search(
            question.question,
            collection=collection,
            store=store,
            embedder=embedder,
            filters=question.filters,
            limit=chunk_limit,
            max_limit=maximum,
            normalize_query=normalize_query,
        )
        ranking = _rank_documents(_in_order(hits))
        places = {document.doc_id: place for place, document in enumerate(ranking, start=1)}
        ranks = [places[doc_id] for doc_id in reachable if doc_id in places]
        rank = min(ranks) if ranks else None
        within_k = rank is not None and rank <= k
        results.append(
            QuestionResult(
                id=question.id,
                question=question.question,
                kind=question.kind,
                tags=question.tags,
                expected_doc_ids=question.expected_doc_ids,
                absent_doc_ids=absent,
                filters=question.filters,
                rank=rank,
                hit_at_1=rank == 1,
                hit_at_k=within_k,
                reciprocal_rank=1.0 / rank if within_k else 0.0,
                recall_at_k=sum(1 for place in ranks if place <= k) / len(reachable),
                chunks_returned=len(hits),
                documents_ranked=len(ranking),
                chunk_budget_exhausted=len(ranking) < k and len(hits) >= chunk_limit,
                top_documents=tuple(ranking[: max(k, _MISS_DOCUMENTS)]),
                tied_at_decision=_tie_decides(ranking, reachable, 1) or _tie_decides(ranking, reachable, k),
            )
        )

    parameters = {
        "collection": collection,
        "k": k,
        "chunk_limit": chunk_limit,
        "max_limit": maximum,
        "model": embedder.model_name,
        "backend": _backend(embedder),
        "query_prefix": embedder.spec.query_prefix,
        "normalize_query": normalize_query,
        "query_normalization": QUERY_NORMALIZATION if normalize_query else None,
        "questions_file": getattr(questions, "path", None),
        "questions_sha256": getattr(questions, "sha256", None),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interval_level": INTERVAL_LEVEL,
        "interval_min_documents": interval_min_documents,
        "engine_version": __version__,
    }
    return EvalReport(parameters=parameters, results=tuple(results), not_in_collection=tuple(skipped))


# The width the lines of the text report whose length depends on a path are cut to.
_LINE_WIDTH = 110


def _cut(text: str, width: int) -> str:
    """``text`` on one line, cut to ``width`` characters."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _cut_start(text: str, width: int) -> str:
    """``text`` on one line, cut to ``width`` characters by dropping its start: the end is kept."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else "…" + flat[len(flat) - (width - 1) :]


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _row(name: str, group: Aggregate, *, pooled: bool = False) -> tuple[str, ...]:
    """One line of the table: the name, the counts, then hit@1, hit@k, MRR, recall@k and a mark."""
    means = group if pooled else group.per_document
    cells = []
    for metric in _METRICS:
        cell = _number(None if means is None else getattr(means, metric))
        if group.intervals is not None and metric in group.intervals:
            low, high = group.intervals[metric]
            cell += f" [{low:.2f}, {high:.2f}]"
        cells.append(cell)
    mark = "(few documents)" if group.few_documents else ""
    return (name, str(group.questions), str(group.documents), *cells, mark)


def _table(rows: Sequence[tuple[str, ...]], k: int) -> list[str]:
    header = ("", "questions", "documents", "hit@1", f"hit@{k}", "MRR", f"recall@{k}", "")
    rows = [header, *((_cut(row[0], 48), *row[1:]) for row in rows)]
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    lines = []
    for row in rows:
        cells = [f"{row[0]:<{widths[0]}}", f"{row[1]:>{widths[1]}}", f"{row[2]:>{widths[2]}}"]
        cells += [f"{cell:<{width}}" for cell, width in zip(row[3:], widths[3:], strict=True)]
        lines.append("  ".join(cells).rstrip())
    return lines


def format_report(report: EvalReport) -> str:
    """The report as a short text for a terminal: parameters, metrics, what was left out, misses.

    The first table has one line per ``kind`` (a ``*`` marks a kind asked with filters),
    then the ``headline`` (the kinds asked without filters) and ``overall (pooled)``. Every
    line gives the number of questions and of distinct expected documents; the lines of the
    kinds and of the headline give the **per-document** means, with their 95% interval in
    brackets or the mark ``(few documents)``; the pooled line gives the means over the
    questions. A legend says so under the table. A second table gives the per-document means
    per tag, without intervals.

    Rates are written to two decimals, which is as much as a few hundred documents can
    tell (``n/a`` for a group without questions); the exact values are in
    :meth:`EvalReport.to_dict`. Then come the questions left out, the ties, and each miss
    with the question, the expected documents and the three best documents returned, with
    their titles. The line that names the question file is cut to 110 characters whatever
    the length of its path, the end of the path being kept. The text has no trailing line
    break.
    """
    parameters = report.parameters
    k = parameters["k"]
    scored, left_out = len(report.results), len(report.not_in_collection)
    title = (
        f"Search evaluation of collection {parameters['collection']!r}: {scored} questions scored "
        f"({scored + left_out} read, {left_out} not in the collection)"
    )
    settings = (
        f"k={k}, chunk_limit={parameters['chunk_limit']}, model {parameters['model']} "
        f"({parameters['backend']} backend), query normalisation "
        f"{parameters['query_normalization'] or 'off'}, rag-engine {parameters['engine_version']}"
    )
    lines = [title, settings]
    if parameters.get("questions_file"):
        digest = parameters.get("questions_sha256") or ""
        # The whole line fits the width, wherever the file lives; the end of a path names the file.
        before, after = "questions: ", f" (sha256 {digest[:12]})"
        room = _LINE_WIDTH - len(before) - len(after)
        lines.append(f"{before}{_cut_start(str(parameters['questions_file']), room)}{after}")

    resamples = parameters.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)
    minimum = parameters.get("interval_min_documents", DEFAULT_INTERVAL_MIN_DOCUMENTS)
    seed = parameters.get("bootstrap_seed", BOOTSTRAP_SEED)
    filtered = set(report.filtered_kinds)
    rows = []
    if list(report.by_kind) != [_NO_KIND]:
        for name, group in sorted(report.by_kind.items()):
            rows.append(_row(f"kind: {name}{' *' if name in filtered else ''}", group))
    rows.append(_row("headline", report.headline))
    rows.append(_row("overall (pooled)", report.overall, pooled=True))
    lines.append("")
    lines += _table(rows, k)
    lines.append("")
    kinds = ", ".join(report.headline_kinds) or "none"
    lines += [
        "Per-document means: the questions that expect one document are averaged first, then the documents",
        "  (the questions on one document are correlated).",
        f"[low, high]: 95% interval from {resamples} resamples of the documents (seed {seed}).",
        f"(few documents): fewer than {minimum} documents, no interval is given.",
        _cut(f"headline: the kinds asked without filters ({kinds}).", 110),
        f"  --fail-under-hit-at-k looks at its hit@{k}.",
    ]
    if filtered:
        lines += [
            "*: asked with filters, and kept out of the headline. Only hit@1 and MRR are informative:",
            f"  where the filters leave fewer than {k} documents, hit@{k} and recall@{k} are certain.",
        ]
    lines += [
        "overall (pooled): the means over all scored questions, every kind, each question counted once;",
        "  it is not the headline.",
    ]
    if report.by_tag:
        lines.append("")
        lines.append("Per tag (per-document means):")
        lines += _table([_row(f"tag: {name}", group) for name, group in sorted(report.by_tag.items())], k)

    if report.not_in_collection:
        lines.append("")
        lines.append(
            f"Not in the collection ({left_out}, left out of the metrics: none of their expected "
            "documents was ingested):"
        )
        for entry in report.not_in_collection:
            lines.append(f"  {_cut(entry.id, 60)}: {_cut(', '.join(entry.expected_doc_ids), 90)}")
    partial = [result for result in report.results if result.absent_doc_ids]
    if partial:
        lines.append("")
        lines.append(f"Scored with some expected documents not in the collection ({len(partial)}):")
        for result in partial:
            lines.append(f"  {_cut(result.id, 60)}: absent {_cut(', '.join(result.absent_doc_ids), 80)}")
    if report.fewer_than_k_documents:
        lines.append("")
        lines.append(
            f"Fewer than {k} documents ranked: {report.fewer_than_k_documents} questions, of which "
            f"{report.chunk_budget_exhausted} used the whole chunk budget"
        )
        if report.chunk_budget_exhausted:
            lines.append(
                f"  (chunk_limit={parameters['chunk_limit']} chunks held fewer than {k} documents: "
                "a larger chunk budget could rank more)"
            )
    if report.tied_at_decision:
        lines.append("")
        lines.append(
            f"Equal scores: hit@1 or hit@{k} hangs on the order of documents of exactly the same score for "
            f"{report.tied_at_decision} question{'' if report.tied_at_decision == 1 else 's'}"
        )
        lines.append("  (ordered here by doc_id; see tied_at_decision in the JSON report)")
    misses = report.misses
    lines.append("")
    lines.append(f"Misses ({len(misses)}): no expected document among the first {k}")
    for result in misses:
        place = "not ranked" if result.rank is None else f"rank {result.rank}"
        lines.append(f"  {_cut(result.id, 60)}  {_cut(result.question, 90)}")
        lines.append(f"    expected: {_cut(', '.join(result.expected_doc_ids), 100)} ({place})")
        for position, document in enumerate(result.top_documents[:_MISS_DOCUMENTS], start=1):
            title = _cut(document.title, 70) if document.title else "(no title)"
            lines.append(f"    {position}. {document.score:.3f}  {_cut(document.doc_id, 40)}  {title}")
        if not result.top_documents:
            lines.append("    nothing was returned")
    return "\n".join(lines)
