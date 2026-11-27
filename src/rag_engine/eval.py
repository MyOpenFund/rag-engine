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

A question may say more (the second version of the format; a file of the first one reads
and scores as before)::

    {"id": "ex-01", "question": "How did the vote on the rate cut split?",
     "expected_behavior": "answer",
     "essential_ids": ["a5b8ba5e477ad19a"],
     "essential_any_of": [["1f2e3d4c5b6a7980", "0a1b2c3d4e5f6071"]],
     "acceptable_ids": ["0c1d2e3f4a5b6c7d"],
     "filters": {"source_codes": ["gb"]}, "run_unfiltered": true,
     "must_include": ["5-4"], "must_not_include": ["unanimous"],
     "kind": "analytical", "tags": ["votes"], "language": "en", "rationale": "..."}

``essential_ids`` and ``acceptable_ids`` split the expected documents into those the answer
needs and those that are merely relevant; ``essential_any_of`` lists groups of documents
that carry the same fact, any one of which will do; ``expected_doc_ids`` may then be left
out. ``expected_behavior`` is ``answer``, ``correct_premise`` or ``abstain``. A question that
abstains is one the corpus does not answer: it is searched and reported apart, never scored,
and the documents it lists are context documents, none of them essential. ``run_unfiltered``
asks for a second run without the filters. ``must_include`` and ``must_not_include`` describe
the right answer: they are kept for a scorer of answers and are not read here, because the
engine returns documents and writes no answer.

Scoring is done on documents, not on chunks (:func:`evaluate` says how), and a question
whose expected documents are not in the collection at all is reported apart instead of
being counted as a failure of the search.

The report (:class:`EvalReport`, :func:`format_report`) gives every group of questions two
means: over the questions, and over the documents they expect, the questions of one document
being averaged first (they are correlated). Its headline is the second one, on the kinds of
question asked without filters, with an interval obtained by resampling the documents.
The essential metrics are read per question: the share of questions that have all they need
among the first ``k``, with an interval obtained by resampling the questions.

A second, smaller scorer grades **passages** instead of documents (:func:`evaluate_passages`): it is
for corpora of a few long documents, where the document is found by any search and chunk ids differ
between chunkers. A question carries ``expected_spans``, short verbatim texts of the answer, and a
retrieved passage is a hit when it holds one (:func:`span_hit`).

This module imports no embedding backend and no parser, and needs the standard library only
besides the engine: ``torch`` is never imported unless the torch backend is built. Through the
store it imports ``qdrant-client``, which may itself load ``fastembed`` (and with it the ONNX
runtime) when that package is installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, overload

from . import __version__
from .config import Settings
from .embedding import Embedder
from .errors import RagEngineError
from .search import QUERY_NORMALIZATION, Hit, SearchError, SearchFilters, is_searchable, search
from .store import QdrantStore

__all__ = [
    "BEHAVIORS",
    "BOOTSTRAP_SEED",
    "DEFAULT_BEHAVIOR",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CHUNK_LIMIT",
    "DEFAULT_INTERVAL_MIN_DOCUMENTS",
    "DEFAULT_INTERVAL_MIN_QUESTIONS",
    "DEFAULT_K",
    "DEFAULT_SCORE_LEVELS",
    "FILTER_FIELDS",
    "INTERVAL_LEVEL",
    "QUESTION_FIELDS",
    "SPAN_GAP",
    "UNFILTERED_SUFFIX",
    "UNFILTERED_TAG",
    "Abstention",
    "Aggregate",
    "EssentialAggregate",
    "EssentialMeans",
    "EvalError",
    "EvalReport",
    "FilterEffect",
    "Means",
    "NotInCollection",
    "PassageAbstention",
    "PassageReport",
    "PassageResult",
    "Question",
    "QuestionResult",
    "QuestionSet",
    "RankedDocument",
    "TopScores",
    "evaluate",
    "evaluate_passages",
    "format_report",
    "load_questions",
    "span_held",
    "span_hit",
    "span_parts",
]

#: Number of documents the metrics look at, unless ``k`` is given.
DEFAULT_K = 10
#: Number of chunks asked to the search for one question, unless ``chunk_limit`` is given.
DEFAULT_CHUNK_LIMIT = 50
#: Number of times the documents are drawn again for an interval, unless ``bootstrap_resamples`` is given.
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
#: Fewest documents a group needs to be given an interval, unless ``interval_min_documents`` is given.
DEFAULT_INTERVAL_MIN_DOCUMENTS = 20
#: Fewest questions a group needs for an interval obtained by resampling its questions (those of the
#: essential metrics), unless ``interval_min_questions`` is given.
DEFAULT_INTERVAL_MIN_QUESTIONS = 20
#: Seed of the resampling: fixed, so that the same results always give the same intervals.
BOOTSTRAP_SEED = 1
#: Level of the intervals: the 2.5th and the 97.5th percentiles of the resampled means.
INTERVAL_LEVEL = 0.95

#: What a question expects of the answer: ``answer`` (the documents answer it), ``correct_premise``
#: (its premise is false, and the documents show it), ``abstain`` (the collection does not hold the
#: information). Only the documents are scored here; the behaviour groups the results.
BEHAVIORS: tuple[str, ...] = ("answer", "correct_premise", "abstain")
#: The behaviour of a question that states none.
DEFAULT_BEHAVIOR = "answer"
#: The scores the documents returned for an abstention question are counted against, unless
#: ``score_levels`` is given. Cosine similarities; which levels tell anything depends on the model.
DEFAULT_SCORE_LEVELS: tuple[float, ...] = (0.7, 0.75, 0.8, 0.85, 0.9)
#: What is added to the id and to the kind of a question for its second run, without its filters.
UNFILTERED_SUFFIX = "#unfiltered"
#: The tag that second run carries, besides the tags of its question.
UNFILTERED_TAG = "unfiltered-rerun"

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
    "expected_behavior",
    "essential_ids",
    "acceptable_ids",
    "must_include",
    "must_not_include",
    "run_unfiltered",
    "essential_any_of",
    "expected_spans",
    "span_match",
    "difficulty",
)

#: How the ``expected_spans`` of a question are matched: ``any`` (one of them is enough, the default)
#: or ``all`` (each must be found among the first ``k`` passages: a question that needs two filings).
SPAN_MATCHES: tuple[str, ...] = ("any", "all")
#: The span matching of a question that states none.
DEFAULT_SPAN_MATCH = "any"
#: The difficulty labels a question may carry (``eval/filings_README.md`` defines them); a label only,
#: the report groups by it.
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
#: The marker that splits one span into fragments one passage must hold in this order, whatever lies
#: between them (a table's year row and its answer row): ``"2024 | | 2025 [...] | Capex | 1 | 2"``.
SPAN_GAP = "[...]"

_REQUIRED = ("id", "question")
_NO_KIND = "(none)"
_MISS_DOCUMENTS = 3
_METRICS = ("hit_at_1", "hit_at_k", "mrr", "recall_at_k")
_INTERVAL_METRICS = ("hit_at_1", "hit_at_k", "mrr")
_ESSENTIAL_METRICS = ("essential_recall_at_k", "all_essential_at_k")
_QUESTION_INTERVAL_METRICS = ("essential_recall_at_k", "all_essential_at_k", "pooled_essential_recall_at_k")
_DOCUMENT_LISTS = ("expected_doc_ids", "essential_ids", "acceptable_ids")


class EvalError(RagEngineError, ValueError):
    """A question, a question file or an argument of the evaluation cannot be used as it is."""


def _strings(name: str, value: object) -> tuple[str, ...]:
    """A list field of a question as a tuple of distinct non-blank strings; ``EvalError`` otherwise."""
    if isinstance(value, (str, bytes)):
        raise EvalError(f"'{name}' must be a list of strings, got the single string {value!r}")
    if not isinstance(value, (list, tuple)):
        raise EvalError(f"'{name}' must be a list of strings, got {type(value).__name__}")
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise EvalError(f"'{name}' must be a list of non-blank strings, got {item!r}")
        if item in seen:
            raise EvalError(f"'{name}' names {item!r} twice")
        seen.add(item)
    return tuple(value)


def _span_group(value: object) -> tuple[str, ...]:
    """One group of alternative spans: a non-empty list of non-blank strings."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or not value:
        raise EvalError(f"a group of 'expected_spans' must be a non-empty list of strings, got {value!r}")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise EvalError(f"a group of 'expected_spans' must hold non-blank strings, got {item!r}")
    return tuple(value)


def _doc_ids(name: str, value: object) -> tuple[str, ...]:
    """A list of document ids: :func:`_strings`, and no id with a space before or after it."""
    ids = _strings(name, value)
    for doc_id in ids:
        if doc_id != doc_id.strip():
            raise EvalError(
                f"'{name}' holds a document id with leading or trailing whitespace, {doc_id!r}: "
                "it would never match a document of the collection"
            )
    return ids


def _groups(value: object) -> tuple[tuple[str, ...], ...]:
    """``essential_any_of`` as a tuple of groups of document ids; ``EvalError`` otherwise."""
    name = "essential_any_of"
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise EvalError(f"'{name}' must be a list of groups, each a list of document ids, got {value!r}")
    groups = []
    where: dict[str, int] = {}
    for number, group in enumerate(value, start=1):
        if isinstance(group, (str, bytes)) or not isinstance(group, (list, tuple)):
            raise EvalError(f"'{name}' must be a list of groups, each a list of document ids, got {group!r}")
        if not group:
            raise EvalError(f"'{name}' holds an empty group: a group names at least one document")
        ids = _doc_ids(f"{name}[{number}]", group)
        for doc_id in ids:
            if doc_id in where:
                raise EvalError(
                    f"'{name}' names {doc_id!r} in two groups ({where[doc_id]} and {number}): "
                    "a document belongs to one requirement"
                )
            where[doc_id] = number
        groups.append(ids)
    return tuple(groups)


def _filters_dict(filters: SearchFilters | None) -> dict[str, Any] | None:
    """``filters`` as the question file writes them: the fields that are set, ready for JSON."""
    if filters is None:
        return None
    result: dict[str, Any] = {}
    for name in FILTER_FIELDS:
        value = getattr(filters, name)
        if value is None or value is False:  # False: the default of a boolean filter, not written
            continue
        if isinstance(value, bool):
            result[name] = value
        elif isinstance(value, tuple):
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
        The ``doc_id`` of every document that answers the question, none twice: the essential
        ones, the members of the either-of groups and the acceptable ones together. At least
        one, unless ``expected_behavior`` is ``"abstain"``. It may be left out when
        ``essential_ids``, ``essential_any_of`` or ``acceptable_ids`` is given (it is then the
        three in that order); given with them, it must name the same documents, in any order.
    ``filters``
        The :class:`rag_engine.search.SearchFilters` the search runs with, or ``None``.
    ``kind``, ``tags``
        Labels the report groups the results by: one optional string, and a list of strings.
    ``language``, ``rationale``
        Free text, kept and not used.
    ``line``
        Line of the question in its file, counted from 1; ``None`` for a question built in code.
    ``expected_behavior``
        One of :data:`BEHAVIORS`; ``"answer"`` unless stated. Another label the report groups
        by. A question that abstains is never scored, whether it lists documents or not: it is
        searched and reported apart (:class:`Abstention`). The documents it lists are its
        *context documents*, those a system may reasonably show while it says that the corpus
        does not answer; none of them is essential, so its ``essential_ids`` must be empty and
        it carries no ``essential_any_of``.
    ``essential_ids``, ``acceptable_ids``
        The documents without which the question cannot be answered, and those that are
        relevant without being necessary: two lists without a common document. Both are
        ``None`` for a question that does not say which of its documents are essential (it
        gives ``expected_doc_ids`` only); when one is given the other is empty unless given too.
    ``essential_any_of``
        Groups of documents that carry the same fact (the decision release and the combined
        document of one day; a later account that restates a total): the answer needs any one
        document of each group. Each group is a non-empty list of distinct ``doc_id``; a
        document is in one group at most, and is neither in ``essential_ids`` nor in
        ``acceptable_ids``. Empty unless given. :attr:`requirements` puts the two together.

        A question says which of its documents are essential as soon as one of the three
        fields is given, an empty list included: ``essential_any_of=[]`` alone means what
        ``essential_ids=[]`` alone means. ``None``, and a field left out, give nothing.
    ``must_include``, ``must_not_include``
        Short texts a right answer holds, and typical false statements it must not hold. Kept
        for a scorer of answers: the engine returns documents and writes no answer, so nothing
        here reads them.
    ``run_unfiltered``
        ``True`` to score the question a second time without its filters. Refused on a
        question that carries none.
    ``expected_spans``
        Short verbatim texts of the filing that answer the question, for the passage-level
        evaluation (:func:`evaluate_passages`): a retrieved passage is a hit when it holds one
        of them (:func:`span_hit`). Distinct non-blank strings; empty unless given, and always
        empty on a question that abstains. :func:`evaluate` does not read them. An item may be
        a list of strings instead of a string: a *group* of alternative spans (the same fact in
        another sentence or another filing), allowed with ``span_match="all"`` only. Once built,
        ``expected_spans`` is the flat tuple of every span, groups unrolled in order. A span may
        hold :data:`SPAN_GAP`: its fragments must then all be held, in order, by one passage
        (a table's year row and its answer row), none of them blank.
    ``span_groups``
        The spans as groups: one tuple per item of ``expected_spans`` as given (a lone string is
        a group of one). Derived when left out (``None``); given, it must hold the same spans as
        ``expected_spans`` (which may then be left empty). Groups only matter with ``"all"``.
    ``span_match``
        One of :data:`SPAN_MATCHES`: ``"any"`` (the default), a passage holding any one span is
        a hit; ``"all"``, every group must be found, by some passage among the first ``k``
        holding one of its spans (a question whose answer needs two facts, often from two
        filings), and the rank is the position where the last missing group is first found.
        ``"all"`` needs at least two groups. With groups of one span, ``"all"`` means every
        span, as before groups existed.
    ``difficulty``
        ``None`` or one of :data:`DIFFICULTIES`: a label the passage report groups by.

    The fields are checked when the question is built (``EvalError``); lists are kept as
    tuples. A document id may not start or end with a space, ``must_include`` and
    ``must_not_include`` may not hold the same text, and a ``kind`` may not end with
    :data:`UNFILTERED_SUFFIX`, which names the second run of a kind.
    """

    id: str
    question: str
    expected_doc_ids: tuple[str, ...] | None = None
    filters: SearchFilters | None = None
    kind: str | None = None
    tags: tuple[str, ...] = ()
    language: str | None = None
    rationale: str | None = None
    line: int | None = None
    expected_behavior: str = DEFAULT_BEHAVIOR
    essential_ids: tuple[str, ...] | None = None
    acceptable_ids: tuple[str, ...] | None = None
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    run_unfiltered: bool = False
    essential_any_of: tuple[tuple[str, ...], ...] | None = None  # a tuple once built, never None
    expected_spans: tuple[str, ...] = ()
    span_match: str = DEFAULT_SPAN_MATCH
    difficulty: str | None = None
    span_groups: tuple[tuple[str, ...], ...] | None = None  # a tuple once built, never None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise EvalError(f"'id' must be a non-blank string, got {self.id!r}")
        if not isinstance(self.question, str) or not self.question.strip():
            raise EvalError(f"'question' must be a non-blank string, got {self.question!r}")
        if not is_searchable(self.question):
            raise EvalError(f"'question' has no searchable character, got {self.question!a}")
        if not isinstance(self.expected_behavior, str) or self.expected_behavior not in BEHAVIORS:
            raise EvalError(
                f"'expected_behavior' must be one of {', '.join(BEHAVIORS)}, got {self.expected_behavior!r}"
            )
        self._check_documents()
        if self.filters is not None and not isinstance(self.filters, SearchFilters):
            raise EvalError(f"'filters' must be a SearchFilters or None, got {type(self.filters).__name__}")
        if self.kind is not None and (not isinstance(self.kind, str) or not self.kind.strip()):
            raise EvalError(f"'kind' must be a non-blank string, got {self.kind!r}")
        if self.kind is not None and self.kind.endswith(UNFILTERED_SUFFIX):
            raise EvalError(
                f"'kind' may not end with {UNFILTERED_SUFFIX!r}, got {self.kind!r}: the suffix names the "
                "second run of a kind, without its filters"
            )
        for name in ("tags", "must_include", "must_not_include"):
            object.__setattr__(self, name, _strings(name, getattr(self, name)))
        self._check_spans()
        gapped = [span for span in self.expected_spans if any(not part.strip() for part in span_parts(span))]
        if gapped:
            raise EvalError(f"a span has a blank fragment around {SPAN_GAP!r}: {gapped[0]!r}")
        if self.expected_spans and self.expected_behavior == "abstain":
            raise EvalError("'expected_spans' is given on a question whose 'expected_behavior' is 'abstain'")
        if not isinstance(self.span_match, str) or self.span_match not in SPAN_MATCHES:
            raise EvalError(f"'span_match' must be one of {', '.join(SPAN_MATCHES)}, got {self.span_match!r}")
        if self.span_match == "all" and len(self.span_groups) < 2:
            raise EvalError(
                "'span_match' is 'all' with fewer than two 'expected_spans' (groups): it would mean 'any'"
            )
        if self.span_match != "all" and any(len(group) > 1 for group in self.span_groups):
            raise EvalError(
                "'expected_spans' holds a group of alternatives on a question whose 'span_match' is not "
                "'all': with 'any' every span is already an alternative, list them flat"
            )
        if self.difficulty is not None and self.difficulty not in DIFFICULTIES:
            raise EvalError(f"'difficulty' must be one of {', '.join(DIFFICULTIES)}, got {self.difficulty!r}")
        both = [text for text in self.must_include if text in self.must_not_include]
        if both:
            raise EvalError(
                f"'must_include' and 'must_not_include' both hold {', '.join(map(repr, both))}: "
                "a right answer cannot hold a text and not hold it"
            )
        for name in ("language", "rationale"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise EvalError(f"'{name}' must be a string, got {value!r}")
        if not isinstance(self.run_unfiltered, bool):
            raise EvalError(f"'run_unfiltered' must be true or false, got {self.run_unfiltered!r}")
        if self.run_unfiltered and _filters_dict(self.filters) is None:
            raise EvalError(
                "'run_unfiltered' is true on a question without filters: there is no filter to remove"
            )

    def _check_spans(self) -> None:
        """Check ``expected_spans`` (strings and groups of strings) and ``span_groups``, store both."""
        spans = self.expected_spans
        if isinstance(spans, (str, bytes)) or not isinstance(spans, (list, tuple)):
            _strings("expected_spans", spans)  # raises the error of a list field
        nested = any(isinstance(item, (list, tuple)) for item in spans)
        groups: tuple[tuple[str, ...], ...]
        if self.span_groups is not None:
            if nested:
                raise EvalError(
                    "'span_groups' is given with groups in 'expected_spans': give one or the other"
                )
            if isinstance(self.span_groups, (str, bytes)) or not isinstance(self.span_groups, (list, tuple)):
                raise EvalError(f"'span_groups' must be a list of lists of strings, got {self.span_groups!r}")
            groups = tuple(_span_group(group) for group in self.span_groups)
            flat = tuple(span for group in groups for span in group)
            if spans and tuple(spans) != flat:
                raise EvalError(
                    "'expected_spans' and 'span_groups' do not hold the same spans in the same order"
                )
        else:
            groups = tuple(
                _span_group(item) if isinstance(item, (list, tuple)) else (item,) for item in spans
            )
            flat = tuple(span for group in groups for span in group)
        object.__setattr__(self, "expected_spans", _strings("expected_spans", flat))
        object.__setattr__(self, "span_groups", groups)

    def _check_documents(self) -> None:
        """Check the lists of documents against each other, and store them as tuples."""
        expected = self.expected_doc_ids
        essential, acceptable = self.essential_ids, self.acceptable_ids
        groups = _groups(self.essential_any_of if self.essential_any_of is not None else ())
        members = tuple(doc_id for group in groups for doc_id in group)
        # Given, even empty: an empty list of groups states "no group", as an empty list of
        # essential documents states "none is essential".
        graded = essential is not None or acceptable is not None or self.essential_any_of is not None
        if graded:
            essential = _doc_ids("essential_ids", essential if essential is not None else ())
            acceptable = _doc_ids("acceptable_ids", acceptable if acceptable is not None else ())
            both = [doc_id for doc_id in essential if doc_id in acceptable]
            if both:
                raise EvalError(
                    f"'essential_ids' and 'acceptable_ids' both name {', '.join(map(repr, both))}: "
                    "a document is one or the other"
                )
            for name, ids in (("essential_ids", essential), ("acceptable_ids", acceptable)):
                both = [doc_id for doc_id in ids if doc_id in members]
                if both:
                    raise EvalError(
                        f"'{name}' and 'essential_any_of' both name {', '.join(map(repr, both))}: a document "
                        "is essential alone, a member of one group, or acceptable"
                    )
            together = essential + members + acceptable
            if expected is None:
                expected = together
            else:
                expected = _doc_ids("expected_doc_ids", expected)
                lacks = [doc_id for doc_id in together if doc_id not in expected]
                adds = [doc_id for doc_id in expected if doc_id not in together]
                if lacks or adds:
                    parts = []
                    if lacks:
                        parts.append(f"it lacks {', '.join(map(repr, lacks))}")
                    if adds:
                        parts.append(f"it adds {', '.join(map(repr, adds))}")
                    names = "'essential_ids' and 'acceptable_ids'"
                    if groups:
                        names = "'essential_ids', 'essential_any_of' and 'acceptable_ids'"
                    raise EvalError(
                        f"'expected_doc_ids' must name the documents of {names} together, in any order: "
                        f"{' and '.join(parts)} (it may also be left out)"
                    )
        elif expected is None:
            raise EvalError("'expected_doc_ids' is missing")
        else:
            expected = _doc_ids("expected_doc_ids", expected)
        if self.expected_behavior == "abstain":
            for name, ids in (("essential_ids", essential or ()), ("essential_any_of", members)):
                if ids:
                    raise EvalError(
                        f"'{name}' names {', '.join(map(repr, ids))} on a question whose 'expected_behavior' "
                        "is 'abstain': a question the corpus does not answer has no essential document "
                        "(the documents a system may show all the same go in 'acceptable_ids')"
                    )
        elif not expected:
            name = "'essential_ids' and 'acceptable_ids' are" if graded else "'expected_doc_ids' is"
            raise EvalError(
                f"{name} empty: a question names at least one document that answers it, unless its "
                "'expected_behavior' is 'abstain'"
            )
        object.__setattr__(self, "expected_doc_ids", expected)
        object.__setattr__(self, "essential_ids", essential if graded else None)
        object.__setattr__(self, "acceptable_ids", acceptable if graded else None)
        object.__setattr__(self, "essential_any_of", groups)

    @property
    def requirements(self) -> tuple[tuple[str, ...], ...] | None:
        """What the answer needs: one group per essential document, then the either-of groups.

        A requirement is satisfied when any document of its group is among the first ``k``.
        ``None`` for a question that does not say which of its documents are essential.
        """
        if self.essential_ids is None:
            return None
        return tuple((doc_id,) for doc_id in self.essential_ids) + self.essential_any_of

    def to_dict(self) -> dict[str, Any]:
        """The question as its line of the question file: the fields that are set."""
        result: dict[str, Any] = {"id": self.id, "question": self.question}
        if self.expected_behavior != DEFAULT_BEHAVIOR:
            result["expected_behavior"] = self.expected_behavior
        if self.essential_ids is not None and self.acceptable_ids is not None:
            result["essential_ids"] = list(self.essential_ids)
            if self.essential_any_of:
                result["essential_any_of"] = [list(group) for group in self.essential_any_of]
            result["acceptable_ids"] = list(self.acceptable_ids)
        result["expected_doc_ids"] = list(self.expected_doc_ids or ())
        filters = _filters_dict(self.filters)
        if filters is not None:
            result["filters"] = filters
        if self.run_unfiltered:
            result["run_unfiltered"] = True
        for name in ("must_include", "must_not_include"):
            if getattr(self, name):
                result[name] = list(getattr(self, name))
        if self.expected_spans:
            result["expected_spans"] = [
                group[0] if len(group) == 1 else list(group) for group in (self.span_groups or ())
            ]
        if self.span_match != DEFAULT_SPAN_MATCH:
            result["span_match"] = self.span_match
        if self.difficulty is not None:
            result["difficulty"] = self.difficulty
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
    for name in ("kind", "language", "rationale", "expected_behavior", "span_match", "difficulty"):
        if record.get(name) is not None and not isinstance(record[name], str):
            raise EvalError(f"'{name}' must be a string, got {record[name]!r}")
    rerun = record.get("run_unfiltered")
    if rerun is not None and not isinstance(rerun, bool):
        raise EvalError(f"'run_unfiltered' must be true or false, got {rerun!r}")
    graded = any(
        record.get(name) is not None for name in ("essential_ids", "essential_any_of", "acceptable_ids")
    )
    if "expected_doc_ids" not in record and not graded:
        raise EvalError("'expected_doc_ids' is missing")
    if not graded and record["expected_doc_ids"] is None:
        raise EvalError("'expected_doc_ids' must be a list of strings, got NoneType")
    question = Question(
        id=record["id"],
        question=record["question"],
        expected_doc_ids=record.get("expected_doc_ids"),
        filters=filters,
        kind=record.get("kind"),
        tags=record["tags"] if record.get("tags") is not None else (),
        language=record.get("language"),
        rationale=record.get("rationale"),
        expected_behavior=(
            record["expected_behavior"] if record.get("expected_behavior") is not None else DEFAULT_BEHAVIOR
        ),
        essential_ids=record.get("essential_ids"),
        acceptable_ids=record.get("acceptable_ids"),
        must_include=record["must_include"] if record.get("must_include") is not None else (),
        must_not_include=record["must_not_include"] if record.get("must_not_include") is not None else (),
        run_unfiltered=bool(rerun),
        essential_any_of=record.get("essential_any_of"),
        expected_spans=record["expected_spans"] if record.get("expected_spans") is not None else (),
        span_match=record["span_match"] if record.get("span_match") is not None else DEFAULT_SPAN_MATCH,
        difficulty=record.get("difficulty"),
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


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """The object of ``pairs``; ``EvalError`` when a key is given twice (JSON would keep the last one)."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvalError(f"the key {key!r} is given twice in one object")
        result[key] = value
    return result


def load_questions(path: str | os.PathLike[str]) -> QuestionSet:
    """Read and check a question file; returns its questions in the order of the file.

    The file is UTF-8 JSON Lines (a byte order mark is accepted, blank lines are skipped).
    Each line is one object with the fields of :data:`QUESTION_FIELDS`:

    - ``id``: a non-blank string, used once in the file;
    - ``question``: a non-blank string that holds something to search for, as given and
      once put in NFKD, the form the search embeds it in (an isolated accent such as
      ``"\u00b4"`` is a symbol as given, and a space and a combining mark in NFKD);
    - ``expected_doc_ids``: a non-empty list of non-blank strings, none twice. It may be
      empty when ``expected_behavior`` is ``"abstain"``, and left out when ``essential_ids``,
      ``essential_any_of`` or ``acceptable_ids`` is given. A document id carries no space
      before or after it, here and in the three lists below;
    - ``filters`` (optional): an object whose keys are fields of
      :class:`rag_engine.search.SearchFilters` (:data:`FILTER_FIELDS`) and whose values
      follow its rules: lists of strings, never a bare string or an empty list; dates
      written ``YYYY-MM-DD``; ``date_from`` not after ``date_to``. A null value, and an
      empty object, restrict nothing;
    - ``kind`` (optional string), ``tags`` (optional list of strings): labels the report
      groups by. A kind may not end with ``#unfiltered``, the name of a second run;
    - ``language``, ``rationale`` (optional strings): kept, not used;
    - ``expected_behavior`` (optional): ``"answer"`` (the default), ``"correct_premise"`` or
      ``"abstain"``;
    - ``essential_ids``, ``acceptable_ids`` (optional): lists of non-blank strings, none
      twice, without a common document. When one of them is given, ``expected_doc_ids`` is
      the two together: left out, it is the first followed by the second; given, it must
      name exactly the same documents, in any order;
    - ``essential_any_of`` (optional): a list of groups, each a non-empty list of distinct
      document ids; the answer needs any one document of each group. A document is in one
      group at most, and in neither of the two lists above; the members of the groups count
      among the expected documents, between the essential and the acceptable ones. On a
      question that abstains, ``essential_ids`` is empty and there is no group: the
      documents it lists are context documents (``acceptable_ids`` or ``expected_doc_ids``).
      One of these three fields given, even as an empty list, makes the question one that
      states its essential documents; ``null`` counts as left out;
    - ``must_include``, ``must_not_include`` (optional): lists of non-blank strings, none
      twice, and none in both; kept, not scored;
    - ``expected_spans`` (optional): a list of non-blank strings, none twice, the verbatim
      passages that answer the question, scored by :func:`evaluate_passages`; not allowed on
      a question that abstains. With ``span_match: "all"`` an item may be a non-empty list of
      strings, a group of alternatives (one of them is enough for that group). A span may hold
      ``[...]`` (:data:`SPAN_GAP`): one passage must hold its fragments in order;
    - ``span_match`` (optional): ``"any"`` (the default) or ``"all"`` (every span, or every
      group, is needed; at least two);
    - ``difficulty`` (optional): ``"easy"``, ``"medium"`` or ``"hard"``, a label;
    - ``run_unfiltered`` (optional): ``true`` or ``false``; ``true`` only on a question
      whose ``filters`` restrict something.

    An optional field may be left out or null. Nothing is guessed or repaired. The first
    line that breaks a rule raises an :class:`EvalError` naming the file and the line number:
    text that is not JSON or not an object, a key given twice in one object, a missing or
    unknown field, an unknown filter, a
    value of the wrong type, a bad date, an id already used (the error names the line of its
    first use), an ``expected_doc_ids`` that differs from the essential and acceptable
    documents, a document both essential and acceptable, no expected document on a question
    that does not abstain, ``run_unfiltered`` without filters, an id that is the id of another
    question followed by ``#unfiltered`` when that question is run a second time. A file
    without any question, and a file that is not UTF-8, are ``EvalError`` too. A file that
    cannot be read raises the ``OSError`` of the system.
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
                record = json.loads(line, object_pairs_hook=_no_duplicate_keys)
            except EvalError:
                raise
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
    for question in questions:
        derived = question.id + UNFILTERED_SUFFIX
        if question.run_unfiltered and derived in first_use:
            raise EvalError(
                f"{name}, line {first_use[derived]}: the id {derived!r} is the one the second run of "
                f"{question.id!r} (line {question.line}, 'run_unfiltered') is reported under"
            )
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
    ``expected_behavior``, ``essential_ids``, ``acceptable_ids``, ``must_include``, ``must_not_include``
        As the question gives them. The last two are carried for a scorer of answers and are
        not scored.
    ``essential_any_of``
        The either-of groups of the question, as it gives them.
    ``essential_present``
        Number of documents of ``essential_ids`` that are in the collection; ``None`` when the
        question does not say which of its documents are essential.
    ``requirements_present``, ``requirements_satisfied``
        A requirement is one essential document or one either-of group. The first is the
        number of requirements with at least one document in the collection (the others are
        left out, like an essential document that is absent); the second, how many of these
        have a document among the first ``k``. ``None`` when the question does not say which
        documents are essential; the second also when no requirement has a document there.
    ``essential_recall_at_k``
        ``requirements_satisfied / requirements_present``; ``None`` when there is no
        requirement to look for. Without either-of groups this is the share of the essential
        documents present in the collection that are among the first ``k``.
    ``all_essential_at_k``
        Whether every requirement is satisfied; ``None`` likewise.
    ``essential_unreachable``
        There are more requirements than ``k``: ``all_essential_at_k`` is false and
        ``essential_recall_at_k`` below 1 whatever the search returns.
    ``expected_beyond_k``
        More expected documents are in the collection than ``k``: ``recall_at_k`` is below 1
        whatever the search returns.
    ``rerun_of``
        For the second run of a question, without its filters (``run_unfiltered``): the id of
        the question. The result then carries the id and the kind of the question followed by
        :data:`UNFILTERED_SUFFIX`, its tags and :data:`UNFILTERED_TAG`, and no filters.
        ``None`` for a question scored as it is written.

    ``rank``, ``hit_at_1``, ``hit_at_k`` and ``reciprocal_rank`` look at the best-placed
    expected document, essential or acceptable: an acceptable document at rank 1 is a hit.
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
    expected_behavior: str = DEFAULT_BEHAVIOR
    essential_ids: tuple[str, ...] | None = None
    acceptable_ids: tuple[str, ...] | None = None
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    essential_present: int | None = None
    essential_recall_at_k: float | None = None
    all_essential_at_k: bool | None = None
    essential_unreachable: bool = False
    rerun_of: str | None = None
    essential_any_of: tuple[tuple[str, ...], ...] = ()
    requirements_present: int | None = None
    requirements_satisfied: int | None = None
    expected_beyond_k: bool = False

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
        result["expected_behavior"] = self.expected_behavior
        for name in ("essential_ids", "acceptable_ids"):
            value = getattr(self, name)
            result[name] = None if value is None else list(value)
        result["must_include"] = list(self.must_include)
        result["must_not_include"] = list(self.must_not_include)
        for name in (
            "essential_present",
            "essential_recall_at_k",
            "all_essential_at_k",
            "essential_unreachable",
            "rerun_of",
        ):
            result[name] = getattr(self, name)
        result["essential_any_of"] = [list(group) for group in self.essential_any_of]
        for name in ("requirements_present", "requirements_satisfied", "expected_beyond_k"):
            result[name] = getattr(self, name)
        return result

    def _requirements(self) -> tuple[int, int] | None:
        """``(present, satisfied)`` for a question with requirements to look for, else ``None``."""
        if self.essential_recall_at_k is None or self.essential_ids is None:
            return None
        if self.requirements_present is not None and self.requirements_satisfied is not None:
            return self.requirements_present, self.requirements_satisfied
        # A result built without the two counts: one requirement per essential document.
        present = self.essential_present or 0
        return present, round(self.essential_recall_at_k * present)

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
class Abstention:
    """What the search returned for a question that abstains (``expected_behavior`` ``"abstain"``).

    The corpus does not answer such a question: it has no right document, so it has no rank
    and counts in no mean, whether it lists documents or not. It is searched all the same,
    so that the scores the search gives when nothing answers can be looked at. Nothing here
    says whether the engine would have abstained: no score is a threshold, and there is no
    pass or fail.

    ``context_doc_ids``, ``absent_context_doc_ids``, ``context_at_k``
        The documents the question lists are its context documents: those a system may
        reasonably show while it says that the corpus does not answer (the document that
        proves the premise wrong, the nearest one on the subject). The first field gives
        them, the second those that are not in the collection, the third how many of the
        others are among the first ``k`` documents (``None`` for a question that lists
        none). It is information, not a score: returning a context document is not a hit.

    ``filters``, ``rerun_of``
        The filters the search ran with; for the second run of the question, without its
        filters, the id of the question (the entry then carries that id followed by
        :data:`UNFILTERED_SUFFIX`).
    ``chunks_returned``, ``documents_ranked``
        Number of chunks the search returned, and of distinct documents among them.
    ``top_documents``
        The first ``k`` documents of the ranking, best first (the first three when ``k`` is
        smaller); the first one and its score are :attr:`top_document` and :attr:`top_score`.
    ``documents_at_or_above``
        For each score level of the evaluation, in increasing order, the level and the number
        of documents ranked (all of them, not the first ``k`` only) whose score is at least
        that level.
    ``must_include``, ``must_not_include``
        As the question gives them; not scored.
    """

    id: str
    question: str
    kind: str | None
    tags: tuple[str, ...]
    filters: SearchFilters | None
    chunks_returned: int
    documents_ranked: int
    top_documents: tuple[RankedDocument, ...]
    documents_at_or_above: tuple[tuple[float, int], ...]
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    rerun_of: str | None = None
    context_doc_ids: tuple[str, ...] = ()
    absent_context_doc_ids: tuple[str, ...] = ()
    context_at_k: int | None = None

    @property
    def top_document(self) -> RankedDocument | None:
        """The first document returned; ``None`` when nothing was."""
        return self.top_documents[0] if self.top_documents else None

    @property
    def top_score(self) -> float | None:
        """Its score; ``None`` when nothing was returned."""
        return self.top_documents[0].score if self.top_documents else None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.id, "question": self.question}
        if self.kind is not None:
            result["kind"] = self.kind
        if self.tags:
            result["tags"] = list(self.tags)
        filters = _filters_dict(self.filters)
        if filters is not None:
            result["filters"] = filters
        result["rerun_of"] = self.rerun_of
        result["chunks_returned"] = self.chunks_returned
        result["documents_ranked"] = self.documents_ranked
        result["top_score"] = self.top_score
        result["top_document"] = None if self.top_document is None else self.top_document.to_dict()
        result["documents_at_or_above"] = [
            {"score": level, "documents": count} for level, count in self.documents_at_or_above
        ]
        result["top_documents"] = [document.to_dict() for document in self.top_documents]
        result["must_include"] = list(self.must_include)
        result["must_not_include"] = list(self.must_not_include)
        result["context_doc_ids"] = list(self.context_doc_ids)
        result["absent_context_doc_ids"] = list(self.absent_context_doc_ids)
        result["context_documents_at_k"] = self.context_at_k
        return result


@dataclass(frozen=True)
class TopScores:
    """The lowest, median and highest score of the first document returned, over ``questions`` questions."""

    questions: int
    min: float
    median: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        return {"questions": self.questions, "min": self.min, "median": self.median, "max": self.max}


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
    documents: Sequence[tuple[float, ...]],
    *,
    resamples: int,
    seed: object,
    metrics: Sequence[str] = _INTERVAL_METRICS,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals of the mean over ``documents`` of their first columns, named by ``metrics``.

    The documents are drawn with replacement, as many as there are, ``resamples`` times;
    the interval of a metric goes from the 2.5th to the 97.5th percentile of its means.
    """
    generator = random.Random(seed)
    count = len(documents)
    places = range(count)
    draws: list[list[float]] = [[] for _name in metrics]
    for _ in range(resamples):
        drawn = generator.choices(places, k=count)
        for column, values in enumerate(draws):
            values.append(math.fsum(documents[place][column] for place in drawn) / count)
    low, high = (1.0 - INTERVAL_LEVEL) / 2.0, (1.0 + INTERVAL_LEVEL) / 2.0
    result = {}
    for name, values in zip(metrics, draws, strict=True):
        values.sort()
        result[name] = (_percentile(values, low), _percentile(values, high))
    return result


def _question_intervals(
    questions: Sequence[tuple[int, int]], *, resamples: int, seed: object
) -> dict[str, tuple[float, float]]:
    """Percentile intervals of the three essential metrics, from resampling ``questions``.

    Each question is ``(requirements, satisfied)``. The questions are drawn with replacement,
    as many as there are, ``resamples`` times; each draw gives the mean share of requirements
    satisfied, the share of questions with all of them, and the pooled share (satisfied over
    requirements, all the questions drawn together).
    """
    generator = random.Random(seed)
    count = len(questions)
    places = range(count)
    shares = [satisfied / present for present, satisfied in questions]
    complete = [float(satisfied == present) for present, satisfied in questions]
    draws: list[list[float]] = [[], [], []]
    for _ in range(resamples):
        drawn = generator.choices(places, k=count)
        draws[0].append(math.fsum(shares[place] for place in drawn) / count)
        draws[1].append(math.fsum(complete[place] for place in drawn) / count)
        draws[2].append(
            sum(questions[place][1] for place in drawn) / sum(questions[place][0] for place in drawn)
        )
    low, high = (1.0 - INTERVAL_LEVEL) / 2.0, (1.0 + INTERVAL_LEVEL) / 2.0
    result = {}
    for name, values in zip(_QUESTION_INTERVAL_METRICS, draws, strict=True):
        values.sort()
        result[name] = (_percentile(values, low), _percentile(values, high))
    return result


@dataclass(frozen=True)
class EssentialMeans:
    """The two essential metrics of a group, as means: ``essential_recall_at_k``, ``all_essential_at_k``."""

    essential_recall_at_k: float
    all_essential_at_k: float

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in _ESSENTIAL_METRICS}


@dataclass(frozen=True)
class EssentialAggregate:
    """The essential metrics of a group: over its questions that have requirements to look for.

    A requirement is one essential document or one either-of group of a question, with at
    least one document in the collection. The other questions of the group (they do not say
    which documents are essential, or none of these is in the collection) are left out of
    these numbers, not counted as zeros.

    ``questions``
        Number of questions the metrics are computed on.
    ``documents``
        Number of distinct documents their requirements name, among those in the collection.
    ``unreachable``
        Number of these questions with more requirements than ``k``: they cannot have all of
        them satisfied among the first ``k``, and pull the means down whatever the search does.
    ``all_essential_at_k``, ``all_essential_questions``
        The **share of the questions** whose requirements are all satisfied, and their number.
    ``essential_recall_at_k``
        The **mean over the questions** of the share of their requirements that are satisfied.
    ``pooled_essential_recall_at_k``, ``requirements``, ``requirements_satisfied``
        The same share **pooled over the requirements**: those satisfied over all of them, the
        questions together. A question with twelve requirements weighs twelve times here and
        once in the mean over the questions.
    ``question_intervals``, ``few_questions``
        For the three metrics above (``essential_recall_at_k``, ``all_essential_at_k``,
        ``pooled_essential_recall_at_k``), the bounds of a 95% interval from resampling the
        **questions** with replacement. ``None`` when no interval was asked for this group or
        when it has too few questions; ``few_questions`` says which.
    ``per_document``, ``intervals``, ``few_documents``
        Kept from the first form of these metrics, with their meaning, and not printed: the
        two metrics as means **over the documents** of the requirements (for each document
        the mean over the questions that name it, then the mean of these), and their interval
        from resampling these documents. A question counts there once for each document of
        its requirements, so ``per_document.all_essential_at_k`` is not a share of questions.
    """

    questions: int
    documents: int
    unreachable: int
    essential_recall_at_k: float
    all_essential_at_k: float
    per_document: EssentialMeans
    intervals: Mapping[str, tuple[float, float]] | None = None
    few_documents: bool = False
    all_essential_questions: int = 0
    requirements: int = 0
    requirements_satisfied: int = 0
    pooled_essential_recall_at_k: float | None = None
    question_intervals: Mapping[str, tuple[float, float]] | None = None
    few_questions: bool = False

    @classmethod
    def of(
        cls,
        results: Sequence[QuestionResult],
        *,
        interval: str | None = None,
        resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
        min_documents: int = DEFAULT_INTERVAL_MIN_DOCUMENTS,
        min_questions: int = DEFAULT_INTERVAL_MIN_QUESTIONS,
    ) -> EssentialAggregate | None:
        """The essential metrics of ``results``; ``None`` when none of them has any."""
        rows: list[tuple[float, float]] = []
        counts: list[tuple[int, int]] = []
        questions: dict[str, list[tuple[float, float]]] = {}
        unreachable = 0
        for result in results:
            requirements = result._requirements()
            if requirements is None or result.essential_ids is None:
                continue
            values = (result.essential_recall_at_k or 0.0, float(bool(result.all_essential_at_k)))
            rows.append(values)
            counts.append(requirements)
            unreachable += result.essential_unreachable
            members = (doc_id for group in result.essential_any_of for doc_id in group)
            for doc_id in (*result.essential_ids, *members):
                if doc_id not in result.absent_doc_ids:
                    questions.setdefault(doc_id, []).append(values)
        if not rows:
            return None
        documents = [_mean(values) for _doc_id, values in sorted(questions.items())]
        intervals = question_intervals = None
        few = few_questions = False
        if interval is not None:
            few = len(documents) < min_documents
            if not few:
                intervals = _intervals(
                    documents,
                    resamples=resamples,
                    seed=f"{BOOTSTRAP_SEED}:{interval}:essential",
                    metrics=_ESSENTIAL_METRICS,
                )
            few_questions = len(rows) < min_questions
            if not few_questions:
                question_intervals = _question_intervals(
                    counts, resamples=resamples, seed=f"{BOOTSTRAP_SEED}:{interval}:essential:questions"
                )
        present = sum(count for count, _satisfied in counts)
        satisfied = sum(found for _count, found in counts)
        return cls(
            len(rows),
            len(documents),
            unreachable,
            *_mean(rows),
            EssentialMeans(*_mean(documents)),
            intervals,
            few,
            all_essential_questions=sum(1 for _recall, complete in rows if complete),
            requirements=present,
            requirements_satisfied=satisfied,
            pooled_essential_recall_at_k=satisfied / present,
            question_intervals=question_intervals,
            few_questions=few_questions,
        )

    def to_dict(self) -> dict[str, Any]:
        """The aggregate as a new dict, ready for ``json.dumps``: the exact values."""

        def bounds(intervals: Mapping[str, tuple[float, float]] | None) -> dict[str, list[float]] | None:
            return None if intervals is None else {name: list(pair) for name, pair in intervals.items()}

        result: dict[str, Any] = {
            "questions": self.questions,
            "documents": self.documents,
            "unreachable": self.unreachable,
        }
        result.update({name: getattr(self, name) for name in _ESSENTIAL_METRICS})
        result["per_document"] = self.per_document.to_dict()
        result["intervals"] = bounds(self.intervals)
        result["few_documents"] = self.few_documents
        for name in (
            "all_essential_questions",
            "requirements",
            "requirements_satisfied",
            "pooled_essential_recall_at_k",
        ):
            result[name] = getattr(self, name)
        result["question_intervals"] = bounds(self.question_intervals)
        result["few_questions"] = self.few_questions
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
    ``essential``
        The essential metrics of the group (:class:`EssentialAggregate`): how many of the
        requirements of its questions are satisfied among the first ``k``. ``None`` when no
        question of the group has a requirement with a document in the collection.

    The first four metrics look at the expected documents, essential and acceptable alike.
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
    essential: EssentialAggregate | None = None

    @classmethod
    def of(
        cls,
        results: Sequence[QuestionResult],
        *,
        interval: str | None = None,
        resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
        min_documents: int = DEFAULT_INTERVAL_MIN_DOCUMENTS,
        min_questions: int = DEFAULT_INTERVAL_MIN_QUESTIONS,
    ) -> Aggregate:
        """The aggregate of ``results``.

        ``interval`` asks for the intervals: it is the name of the group, which with
        :data:`BOOTSTRAP_SEED` seeds its resampling, so that the interval of a group does
        not depend on the other groups. They are computed from ``resamples`` draws when the
        group has at least ``min_documents`` documents; those of the essential metrics, which
        resample questions, when it has at least ``min_questions`` questions with requirements.
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
        essential = EssentialAggregate.of(
            results,
            interval=interval,
            resamples=resamples,
            min_documents=min_documents,
            min_questions=min_questions,
        )
        return cls(len(results), len(documents), *per_question, per_document, intervals, few, essential)

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
        result["essential"] = None if self.essential is None else self.essential.to_dict()
        return result


@dataclass(frozen=True)
class FilterEffect:
    """The questions scored with their filters and again without them, side by side.

    ``pairs``
        For each such question, its result as written and its result without filters.
    ``filtered``, ``unfiltered``
        The means **over these questions** of ``hit_at_1``, ``hit_at_k``, ``mrr`` and
        ``essential_recall_at_k``, with and without the filters. The last one is the mean
        over the ``essential_questions`` pairs that have requirements to look for (essential
        documents or either-of groups in the collection), and over these only; ``None`` when
        there is none.
    """

    pairs: tuple[tuple[QuestionResult, QuestionResult], ...]
    filtered: Mapping[str, float | None] = field(init=False)
    unfiltered: Mapping[str, float | None] = field(init=False)
    essential_questions: int = field(init=False)

    def __post_init__(self) -> None:
        graded = [pair for pair in self.pairs if all(r.essential_recall_at_k is not None for r in pair)]
        for name, place in (("filtered", 0), ("unfiltered", 1)):
            means = _mean([_values(pair[place])[:3] for pair in self.pairs])
            values: dict[str, float | None] = dict(zip(_INTERVAL_METRICS, means, strict=True))
            values["essential_recall_at_k"] = (
                math.fsum(pair[place].essential_recall_at_k or 0.0 for pair in graded) / len(graded)
                if graded
                else None
            )
            object.__setattr__(self, name, values)
        object.__setattr__(self, "essential_questions", len(graded))

    @property
    def questions(self) -> int:
        """Number of questions scored both ways."""
        return len(self.pairs)

    def to_dict(self) -> dict[str, Any]:
        def one(result: QuestionResult) -> dict[str, Any]:
            return {
                "rank": result.rank,
                "hit_at_1": result.hit_at_1,
                "hit_at_k": result.hit_at_k,
                "reciprocal_rank": result.reciprocal_rank,
                "essential_recall_at_k": result.essential_recall_at_k,
            }

        return {
            "aggregation": "means over the questions scored with their filters and without them",
            "questions": self.questions,
            "essential_questions": self.essential_questions,
            "filtered": dict(self.filtered),
            "unfiltered": dict(self.unfiltered),
            "pairs": [
                {"id": written.id, "filtered": one(written), "unfiltered": one(rerun)}
                for written, rerun in self.pairs
            ],
        }


@dataclass(frozen=True)
class EvalReport:
    """The result of :func:`evaluate`.

    ``parameters``
        What the evaluation ran with: ``collection``, ``k``, ``chunk_limit``, ``max_limit``,
        ``model`` and ``backend`` of the embedder, its ``query_prefix``, ``normalize_query``
        and ``query_normalization``, ``questions_file`` and ``questions_sha256`` (``None``
        for questions not read from a file), ``bootstrap_resamples``, ``bootstrap_seed``,
        ``interval_level`` and ``interval_min_documents``, ``engine_version``,
        ``score_levels``, ``interval_min_questions``.
    ``results``
        One :class:`QuestionResult` per scored question, as it is written, in the order of
        the questions.
    ``not_in_collection``
        The questions left out because none of their expected documents is in the
        collection. They count in no metric.
    ``unfiltered_reruns``
        One more :class:`QuestionResult` for each scored question that asks for it
        (``run_unfiltered``): the same question without its filters, under the id and the
        kind of the question followed by ``#unfiltered``.
    ``abstentions``
        The questions that abstain (:class:`Abstention`), with or without context documents,
        and their second run without filters when they ask for one. They count in no metric.

    Everything else is derived from these:

    ``by_kind``, ``by_tag``
        One :class:`Aggregate` per ``kind`` (``"(none)"`` for the questions without one) and
        per tag. The kinds carry intervals, the tags do not. The second runs count here: each
        under its own kind (``<kind>#unfiltered``), under the tags of its question and under
        ``unfiltered-rerun``.
    ``filtered_kinds``, ``headline_kinds``
        The kinds with at least one scored question that carries filters, and the others,
        both sorted. Filters narrow the candidates: when they leave fewer than ``k``
        documents, ``hit_at_k`` and ``recall_at_k`` are true by construction, so a filtered
        kind is read on ``hit_at_1`` and ``mrr`` only and is kept out of the headline. A
        second run carries no filters and a kind of its own: that kind is not a filtered one.
    ``headline``
        The :class:`Aggregate`, with intervals, of the questions of the ``headline_kinds``:
        the kinds asked without filters, the second runs among them. Its ``per_document``
        means are the headline of the report, and what ``rag-engine eval
        --fail-under-hit-at-k`` looks at.
    ``overall``
        The :class:`Aggregate` of all the scored questions, pooled: every kind, filtered or
        not, each question counted once, as it is written (the second runs are not in it).
        Kept for reference; it is not the headline.
    ``by_behavior``
        One :class:`Aggregate` per ``expected_behavior`` that has a scored question, in the
        order of :data:`BEHAVIORS`, each question counted once as it is written. No intervals.
        :func:`evaluate` scores no question that abstains, so its reports have no ``abstain``
        entry here: these questions are in ``abstentions``.
    ``filter_effect``
        The questions scored with and without their filters, compared (:class:`FilterEffect`);
        ``None`` when there is none.
    ``top_scores``
        The lowest, the median and the highest score of the first document returned, over the
        scored questions as they are written (:class:`TopScores`): what the scores of the
        abstention questions compare with. ``None`` when nothing was scored.

    and the :attr:`misses` and the counts.
    """

    parameters: Mapping[str, Any]
    results: tuple[QuestionResult, ...]
    not_in_collection: tuple[NotInCollection, ...] = ()
    unfiltered_reruns: tuple[QuestionResult, ...] = ()
    abstentions: tuple[Abstention, ...] = ()
    overall: Aggregate = field(init=False)
    headline: Aggregate = field(init=False)
    headline_kinds: tuple[str, ...] = field(init=False)
    filtered_kinds: tuple[str, ...] = field(init=False)
    by_kind: dict[str, Aggregate] = field(init=False)
    by_tag: dict[str, Aggregate] = field(init=False)
    by_behavior: dict[str, Aggregate] = field(init=False)

    def __post_init__(self) -> None:
        kinds: dict[str, list[QuestionResult]] = {}
        tags: dict[str, list[QuestionResult]] = {}
        filtered: set[str] = set()
        for result in self.searches:
            kind = result.kind if result.kind is not None else _NO_KIND
            kinds.setdefault(kind, []).append(result)
            if result.filters is not None:
                filtered.add(kind)
            for tag in result.tags:
                tags.setdefault(tag, []).append(result)
        sampling = {
            "resamples": self.parameters.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES),
            "min_documents": self.parameters.get("interval_min_documents", DEFAULT_INTERVAL_MIN_DOCUMENTS),
            "min_questions": self.parameters.get("interval_min_questions", DEFAULT_INTERVAL_MIN_QUESTIONS),
        }
        unfiltered = tuple(sorted(name for name in kinds if name not in filtered))
        headline = [result for result in self.searches if (result.kind or _NO_KIND) in unfiltered]
        behaviors = {
            name: [result for result in self.results if result.expected_behavior == name]
            for name in BEHAVIORS
        }
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
            "by_behavior": {name: Aggregate.of(group) for name, group in behaviors.items() if group},
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def searches(self) -> tuple[QuestionResult, ...]:
        """The results and the second runs together: each second run right after its question."""
        reruns: dict[str | None, list[QuestionResult]] = {}
        for rerun in self.unfiltered_reruns:
            reruns.setdefault(rerun.rerun_of, []).append(rerun)
        ordered: list[QuestionResult] = []
        for result in self.results:
            ordered.append(result)
            ordered += reruns.pop(result.id, [])
        for left in reruns.values():  # a second run whose question is not among the results
            ordered += left
        return tuple(ordered)

    @property
    def misses(self) -> tuple[QuestionResult, ...]:
        """The scored questions, and second runs, without any expected document among the first ``k``."""
        return tuple(result for result in self.searches if not result.hit_at_k)

    @property
    def partially_absent(self) -> int:
        """Number of scored questions with some, not all, expected documents out of the collection."""
        return sum(1 for result in self.results if result.absent_doc_ids)

    @property
    def fewer_than_k_documents(self) -> int:
        """Number of scored questions, and second runs, for which fewer than ``k`` documents were ranked."""
        k = self.parameters["k"]
        return sum(1 for result in self.searches if result.documents_ranked < k)

    @property
    def chunk_budget_exhausted(self) -> int:
        """Among them, those whose search returned every chunk it was asked for."""
        return sum(1 for result in self.searches if result.chunk_budget_exhausted)

    @property
    def tied_at_decision(self) -> int:
        """Number of scored questions, and second runs, whose hit@1 or hit@k hangs on equal scores."""
        return sum(1 for result in self.searches if result.tied_at_decision)

    @property
    def essential_unreachable(self) -> int:
        """Number of scored questions with more requirements to look for than ``k``."""
        return sum(1 for result in self.results if result.essential_unreachable)

    @property
    def expected_beyond_k(self) -> int:
        """Number of scored questions with more expected documents in the collection than ``k``."""
        return sum(1 for result in self.results if result.expected_beyond_k)

    @property
    def abstention_questions(self) -> int:
        """Number of questions that abstain (their second runs without filters are not counted)."""
        return sum(1 for entry in self.abstentions if entry.rerun_of is None)

    @property
    def filter_effect(self) -> FilterEffect | None:
        """The questions scored with and without their filters, compared; ``None`` without any."""
        written = {result.id: result for result in self.results}
        pairs = tuple(
            (written[rerun.rerun_of], rerun) for rerun in self.unfiltered_reruns if rerun.rerun_of in written
        )
        return FilterEffect(pairs) if pairs else None

    @property
    def top_scores(self) -> TopScores | None:
        """The scores of the first document over the scored questions as written; ``None`` without any."""
        scores = [result.top_documents[0].score for result in self.results if result.top_documents]
        if not scores:
            return None
        return TopScores(len(scores), min(scores), statistics.median(scores), max(scores))

    def to_dict(self) -> dict[str, Any]:
        """The whole report as a new dict, ready for ``json.dumps``, with the exact values.

        Keys, in this order: ``parameters``; ``summary`` (the counts); ``metrics``
        (``headline``, ``overall``, ``by_kind``, ``by_tag``, ``by_expected_behavior``: each an
        aggregate with its means over the questions, its ``per_document`` means, its
        ``intervals`` and its ``essential`` metrics; ``headline`` and ``overall`` say what
        they aggregate, and each kind whether it is ``filtered``); ``misses`` (id, question,
        expected ids, rank, the three best documents with their titles);
        ``not_in_collection``; ``results`` (every scored question, as written, with its first
        documents); ``unfiltered_reruns`` (the second runs, in the same form);
        ``filter_effect`` (``None`` without second runs); ``abstentions`` (the score levels,
        the top scores of the scored questions, and one entry per question that abstains).

        The keys of the first version of the report keep their names and their meaning; the
        others were added to them. In ``essential``, ``per_document``, ``intervals`` and
        ``few_documents`` are kept as they were first computed (over the documents of the
        requirements); the text report reads ``all_essential_at_k``, ``essential_recall_at_k``,
        ``pooled_essential_recall_at_k`` and ``question_intervals``.
        """
        effect = self.filter_effect
        top = self.top_scores
        reruns = len(self.unfiltered_reruns)
        return {
            "parameters": dict(self.parameters),
            "summary": {
                "questions": len(self.results) + len(self.not_in_collection) + self.abstention_questions,
                "scored": len(self.results),
                "not_in_collection": len(self.not_in_collection),
                "partially_absent": self.partially_absent,
                "fewer_than_k_documents": self.fewer_than_k_documents,
                "chunk_budget_exhausted": self.chunk_budget_exhausted,
                "tied_at_decision": self.tied_at_decision,
                "unfiltered_reruns": reruns,
                "abstentions": self.abstention_questions,
                "essential_unreachable": self.essential_unreachable,
                "expected_beyond_k": self.expected_beyond_k,
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
                "by_expected_behavior": {name: group.to_dict() for name, group in self.by_behavior.items()},
            },
            "misses": [result._miss() for result in self.misses],
            "not_in_collection": [skipped.to_dict() for skipped in self.not_in_collection],
            "results": [result.to_dict() for result in self.results],
            "unfiltered_reruns": [result.to_dict() for result in self.unfiltered_reruns],
            "filter_effect": None if effect is None else effect.to_dict(),
            "abstentions": {
                "score_levels": list(self.parameters.get("score_levels", DEFAULT_SCORE_LEVELS)),
                "top_score_of_scored_questions": None if top is None else top.to_dict(),
                "questions": [entry.to_dict() for entry in self.abstentions],
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """:meth:`to_dict` as JSON text, non-ASCII characters written as they are."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _count(name: str, value: object) -> int:
    if type(value) is not int or value < 1:  # a bool is an int subclass, and is not a count
        raise EvalError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def _levels(value: object) -> tuple[float, ...]:
    """``score_levels`` in increasing order; ``EvalError`` unless they are distinct finite numbers."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise EvalError(f"score_levels must be a list of numbers, got {value!r}")
    levels = list(value)
    for level in levels:
        if isinstance(level, bool) or not isinstance(level, (int, float)) or not math.isfinite(level):
            raise EvalError(f"score_levels must be finite numbers, got {level!r}")
    if not levels:
        raise EvalError("score_levels is empty: give at least one score")
    if len(set(levels)) != len(levels):
        raise EvalError(f"score_levels names a score twice: {levels!r}")
    return tuple(sorted(float(level) for level in levels))


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
    score_levels: Sequence[float] = DEFAULT_SCORE_LEVELS,
    interval_min_questions: int = DEFAULT_INTERVAL_MIN_QUESTIONS,
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

    **Essential and acceptable documents.** A question may say which of its expected
    documents are essential (``essential_ids``) and which are only acceptable
    (``acceptable_ids``). The four metrics above do not tell them apart: the rank is the one
    of the best-placed expected document, so an acceptable document at rank 1 is a hit. Two
    more metrics look at what the answer needs. A *requirement* is one essential document, or
    one group of ``essential_any_of``: several documents that carry the same fact, any one of
    which will do. A requirement is satisfied when one of its documents is among the first
    ``k``; one whose documents are all absent from the collection is left out.
    ``essential_recall_at_k`` is the share of the requirements of the question that are
    satisfied, and ``all_essential_at_k`` whether all of them are. They are ``None`` for a
    question that does not state its essential documents, or has no requirement to look for,
    and such a question is left out of their means. A question with more requirements than
    ``k`` cannot have them all satisfied: it is marked ``essential_unreachable`` and counted,
    and the remedy is a larger ``k`` (itself at most ``chunk_limit``). Each aggregate carries
    these metrics (:class:`EssentialAggregate`): the share of questions with all their
    requirements, the mean share over the questions, and the share pooled over the
    requirements, with intervals from resampling the questions where the group has at least
    ``interval_min_questions`` of them (default 20).

    **Long lists of expected documents.** ``recall_at_k`` cannot reach 1 for a question with
    more expected documents in the collection than ``k``: it is marked ``expected_beyond_k``
    and the report counts them. In a per-document mean such a question counts once for each of
    its documents; the means over the questions, which every aggregate carries too, count it
    once.

    **Without the filters.** A question with ``run_unfiltered`` is searched and scored
    twice: as it is written, and without its filters. The second result is kept apart
    (``unfiltered_reruns``) under the id and the kind of the question followed by
    ``#unfiltered``, with the tag ``unfiltered-rerun``: it counts in that kind, which is
    asked without filters and so belongs to the headline, and in its tags, not in the kind
    of its question nor in the pooled ``overall``. :attr:`EvalReport.filter_effect` compares
    the two runs of these questions.

    **Questions that abstain** (``expected_behavior`` ``"abstain"``) are questions the corpus
    does not answer: nothing is the right document, so they are searched, never scored, and
    left out of every mean, of the headline, of the kinds, the tags and the behaviours,
    whether they list documents or not. They are listed in ``abstentions`` with the first
    document returned, its score, and the number of documents ranked at or above each of
    ``score_levels`` (default :data:`DEFAULT_SCORE_LEVELS`). The documents such a question
    lists are context documents, which a system may show while saying that the corpus does
    not answer: the entry says how many of them are among the first ``k``, for information.
    :attr:`EvalReport.top_scores` gives, for comparison, the lowest, median and highest
    score of the first document over the scored questions. No threshold is applied and
    nothing passes or fails: the engine returns documents, and whether to answer is decided
    by its caller. A ``"correct_premise"`` question is scored on its documents like any
    other; :attr:`EvalReport.by_behavior` gives one aggregate per behaviour that is scored.
    ``must_include`` and ``must_not_include`` are carried into the results and never read.

    ``questions`` is a :class:`QuestionSet` (from :func:`load_questions`) or any iterable of
    :class:`Question` with distinct ids. ``collection`` must exist with the dimension of the
    embedder; it is not checked here. ``normalize_query`` is the argument of ``search``.

    Errors. :class:`EvalError`, before anything is asked, for no question, something that is
    not a ``Question``, an id used twice, a question that holds nothing to search for once
    normalised (with ``normalize_query``), a ``k``, ``chunk_limit``, ``max_limit``,
    ``bootstrap_resamples``, ``interval_min_documents`` or ``interval_min_questions`` that is
    not an integer >= 1, a
    ``chunk_limit`` above the maximum, a ``k`` above ``chunk_limit``, a blank
    ``collection``, a ``normalize_query`` that is not a boolean, ``score_levels`` that are
    not distinct finite numbers (at least one), an id that is also the id the second run of
    another question is reported under. Everything else is
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
    for question in asked:
        if question.run_unfiltered and question.id + UNFILTERED_SUFFIX in seen:
            raise EvalError(
                f"the id {question.id + UNFILTERED_SUFFIX!r} is the one the second run of "
                f"{question.id!r} is reported under"
            )
    levels = _levels(score_levels)
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
    _count("interval_min_questions", interval_min_questions)
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
    present = store.existing_doc_ids(collection, wanted) if wanted else set()

    def ranked(question: Question, filters: SearchFilters | None) -> tuple[int, list[RankedDocument]]:
        hits = search(
            question.question,
            collection=collection,
            store=store,
            embedder=embedder,
            filters=filters,
            limit=chunk_limit,
            max_limit=maximum,
            normalize_query=normalize_query,
        )
        return len(hits), _rank_documents(_in_order(hits))

    results: list[QuestionResult] = []
    reruns: list[QuestionResult] = []
    skipped: list[NotInCollection] = []
    abstentions: list[Abstention] = []
    for question in asked:
        # As it is written, then without its filters when it asks for that.
        runs = [(question.filters, False)] + ([(None, True)] if question.run_unfiltered else [])
        if question.expected_behavior == "abstain":  # nothing is the right document: not scored
            context = question.expected_doc_ids
            there = [doc_id for doc_id in context if doc_id in present]
            for filters, again in runs:
                returned, ranking = ranked(question, filters)
                first = {document.doc_id for document in ranking[:k]}
                abstentions.append(
                    Abstention(
                        id=question.id + UNFILTERED_SUFFIX if again else question.id,
                        question=question.question,
                        kind=question.kind,
                        tags=question.tags,
                        filters=filters,
                        chunks_returned=returned,
                        documents_ranked=len(ranking),
                        top_documents=tuple(ranking[: max(k, _MISS_DOCUMENTS)]),
                        documents_at_or_above=tuple(
                            (level, sum(1 for document in ranking if document.score >= level))
                            for level in levels
                        ),
                        must_include=question.must_include,
                        must_not_include=question.must_not_include,
                        rerun_of=question.id if again else None,
                        context_doc_ids=context,
                        absent_context_doc_ids=tuple(doc_id for doc_id in context if doc_id not in present),
                        context_at_k=sum(1 for doc_id in there if doc_id in first) if context else None,
                    )
                )
            continue
        absent = tuple(doc_id for doc_id in question.expected_doc_ids if doc_id not in present)
        reachable = [doc_id for doc_id in question.expected_doc_ids if doc_id in present]
        if not reachable:
            skipped.append(
                NotInCollection(
                    question.id, question.question, question.expected_doc_ids, question.kind, question.tags
                )
            )
            continue
        essential = [doc_id for doc_id in question.essential_ids or () if doc_id in present]
        # A requirement is looked for when one of its documents at least is in the collection.
        requirements = [
            [doc_id for doc_id in group if doc_id in present] for group in question.requirements or ()
        ]
        requirements = [group for group in requirements if group]
        graded = question.essential_ids is not None
        for filters, again in runs:
            returned, ranking = ranked(question, filters)
            places = {document.doc_id: place for place, document in enumerate(ranking, start=1)}
            ranks = [places[doc_id] for doc_id in reachable if doc_id in places]
            rank = min(ranks) if ranks else None
            within_k = rank is not None and rank <= k
            found = sum(
                1 for group in requirements if any(places.get(doc_id, k + 1) <= k for doc_id in group)
            )
            tags = question.tags
            if again and UNFILTERED_TAG not in tags:
                tags = (*tags, UNFILTERED_TAG)
            result = QuestionResult(
                id=question.id + UNFILTERED_SUFFIX if again else question.id,
                question=question.question,
                kind=(question.kind or _NO_KIND) + UNFILTERED_SUFFIX if again else question.kind,
                tags=tags,
                expected_doc_ids=question.expected_doc_ids,
                absent_doc_ids=absent,
                filters=filters,
                rank=rank,
                hit_at_1=rank == 1,
                hit_at_k=within_k,
                reciprocal_rank=1.0 / rank if within_k else 0.0,
                recall_at_k=sum(1 for place in ranks if place <= k) / len(reachable),
                chunks_returned=returned,
                documents_ranked=len(ranking),
                chunk_budget_exhausted=len(ranking) < k and returned >= chunk_limit,
                top_documents=tuple(ranking[: max(k, _MISS_DOCUMENTS)]),
                tied_at_decision=_tie_decides(ranking, reachable, 1) or _tie_decides(ranking, reachable, k),
                expected_behavior=question.expected_behavior,
                essential_ids=question.essential_ids,
                acceptable_ids=question.acceptable_ids,
                must_include=question.must_include,
                must_not_include=question.must_not_include,
                essential_present=len(essential) if graded else None,
                essential_recall_at_k=found / len(requirements) if requirements else None,
                all_essential_at_k=found == len(requirements) if requirements else None,
                essential_unreachable=len(requirements) > k,
                rerun_of=question.id if again else None,
                essential_any_of=question.essential_any_of,
                requirements_present=len(requirements) if graded else None,
                requirements_satisfied=found if requirements else None,
                expected_beyond_k=len(reachable) > k,
            )
            (reruns if again else results).append(result)

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
        "score_levels": list(levels),
        "interval_min_questions": interval_min_questions,
    }
    return EvalReport(
        parameters=parameters,
        results=tuple(results),
        not_in_collection=tuple(skipped),
        unfiltered_reruns=tuple(reruns),
        abstentions=tuple(abstentions),
    )


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


def _wrapped(label: str, parts: Sequence[str]) -> list[str]:
    """``label`` followed by ``parts`` separated by commas, on as many lines of the width as it takes."""
    lines: list[str] = []
    current = label
    for place, part in enumerate(parts):
        piece = part + ("," if place < len(parts) - 1 else "")
        if len(current) + 1 + len(piece) > _LINE_WIDTH and current.strip():
            lines.append(current)
            current = " " * 4 + piece
        else:
            current = f"{current} {piece}"
    lines.append(current)
    return lines


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


def _table(
    rows: Sequence[tuple[str, ...]],
    k: int,
    *,
    notes: Sequence[tuple[str, str, str]] = (),
    width: int | None = None,
) -> list[str]:
    """The lines of a table of metrics: a header, then ``rows`` (as :func:`_row` makes them).

    ``notes`` are lines without metrics, printed last: a name, a number of questions and a
    text in place of the metrics. With ``width``, a table that would be wider is laid out
    with the intervals, and the mark of a line, on a line of their own under each line.
    """
    header = ("", "questions", "documents", "hit@1", f"hit@{k}", "MRR", f"recall@{k}", "")
    rows = [header, *((_cut(row[0], 48), *row[1:]) for row in rows)]
    notes = [(_cut(name, 48), count, text) for name, count, text in notes]

    def lines_of(rows: Sequence[tuple[str, ...]]) -> list[str]:
        widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
        for name, count, _text in notes:
            widths[0], widths[1] = max(widths[0], len(name)), max(widths[1], len(count))
        lines = []
        for row in rows:
            cells = [f"{row[0]:<{widths[0]}}", f"{row[1]:>{widths[1]}}", f"{row[2]:>{widths[2]}}"]
            cells += [f"{cell:<{width}}" for cell, width in zip(row[3:], widths[3:], strict=True)]
            lines.append("  ".join(cells).rstrip())
        for name, count, text in notes:
            lines.append(f"{name:<{widths[0]}}  {count:>{widths[1]}}  {text}")
        return lines

    lines = lines_of(rows)
    if width is None or max(len(line) for line in lines) <= width:
        return lines
    split: list[tuple[str, ...]] = [header]
    for row in rows[1:]:
        values = [cell.partition(" [") for cell in row[3:7]]
        split.append((*row[:3], *(value for value, _mark, _interval in values), ""))
        if any(interval for _value, _mark, interval in values):
            intervals = ("[" + interval if interval else "" for _value, _mark, interval in values)
            split.append(("  95% interval", "", "", *intervals, ""))
        if row[7]:
            split.append((f"  {row[7]}", "", "", "", "", "", "", ""))
    return lines_of(split)


def _aligned(rows: Sequence[Sequence[str]], *, right: Sequence[int] = (), name_width: int = 36) -> list[str]:
    """``rows`` (the header first) as lines of aligned columns; the columns of ``right`` are right-aligned."""
    rows = [(_cut(row[0], name_width), *row[1:]) for row in rows]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = []
    for row in rows:
        cells = [
            f"{cell:>{width}}" if column in right else f"{cell:<{width}}"
            for column, (cell, width) in enumerate(zip(row, widths, strict=True))
        ]
        lines.append("  ".join(cells).rstrip())
    return lines


def _second_version(report: EvalReport) -> bool:
    """Whether the questions use the second version of the format: the report then has its parts."""
    return bool(
        report.abstentions
        or report.unfiltered_reruns
        or any(
            result.essential_ids is not None or result.expected_behavior != DEFAULT_BEHAVIOR
            for result in report.results
        )
    )


def _named_groups(report: EvalReport) -> list[tuple[str, Aggregate]]:
    """The kinds (a ``*`` after a filtered one) and the headline, as the tables name them."""
    filtered = set(report.filtered_kinds)
    groups = []
    if list(report.by_kind) != [_NO_KIND]:
        for name, group in sorted(report.by_kind.items()):
            groups.append((f"kind: {name}{' *' if name in filtered else ''}", group))
    groups.append(("headline", report.headline))
    return groups


def _per_question_lines(report: EvalReport, k: int) -> list[str]:
    header = ("", "questions", "hit@1", f"hit@{k}", "MRR", f"recall@{k}")
    rows = [header]
    for name, group in _named_groups(report):
        rows.append((name, str(group.questions), *(_number(getattr(group, metric)) for metric in _METRICS)))
    lines = ["", "Per question (the same lines as means over the questions: each question counts once):"]
    lines += _aligned(rows, right=(1,))
    lines += [
        "Which to read. Per document (the first table) when several questions ask for one document, as in",
        "  a known-item batch: a document asked for four times is not four observations. Per question",
        "  (this table, and overall (pooled) above) when questions expect many documents each: in a",
        "  per-document mean a question that expects 44 documents weighs 44 times one that expects one.",
    ]
    beyond = [result.id for result in report.results if result.expected_beyond_k]
    if beyond:
        count = len(beyond)
        some = f"{count} question{'' if count == 1 else 's'}: recall@{k} cannot reach 1 for "
        some += "it" if count == 1 else "them"
        lines += _wrapped(f"More expected documents than k={k} ({some}):", beyond)
    return lines


def _behavior_lines(report: EvalReport, k: int) -> list[str]:
    rows = [_row(f"expected_behavior: {name}", group) for name, group in report.by_behavior.items()]
    apart = report.abstention_questions
    notes = [("expected_behavior: abstain", str(apart), "see the Abstention section")] if apart else []
    lines = [
        "",
        "Per expected behaviour (per-document means; each question once, as it is written):",
        *_table(rows, k, notes=notes),
    ]
    if "correct_premise" in report.by_behavior:
        lines += [
            "correct_premise: scored on the documents that show the premise is false, like any question.",
            "  Whether the response corrects the premise is a property of the answer: it is not judged here.",
        ]
    if apart:
        lines.append("abstain: never scored, whether the question lists documents or not; in no line above.")
    return lines


def _essential_cells(group: Aggregate) -> tuple[str, str, str, str]:
    """The cells of one line of the essential table: all, mean share, pooled share, unreachable."""
    essential = group.essential
    if essential is None:
        return ("n/a", "n/a", "n/a", "0")
    complete = f"({essential.all_essential_questions}/{essential.questions})"
    pooled = f"({essential.requirements_satisfied}/{essential.requirements})"
    return (
        f"{_number(essential.all_essential_at_k)} {complete}",
        _number(essential.essential_recall_at_k),
        f"{_number(essential.pooled_essential_recall_at_k)} {pooled}",
        str(essential.unreachable),
    )


def _essential_lines(report: EvalReport, k: int, *, behaviors: bool) -> list[str]:
    minimum = report.parameters.get("interval_min_questions", DEFAULT_INTERVAL_MIN_QUESTIONS)
    resamples = report.parameters.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)
    seed = report.parameters.get("bootstrap_seed", BOOTSTRAP_SEED)
    header = ("", f"all_essential@{k}", f"essential_recall@{k}", "pooled", "unreachable")
    named = _named_groups(report)
    rows = [header, *((name, *_essential_cells(group)) for name, group in named)]
    rows.append(("overall (pooled)", *_essential_cells(report.overall)))
    if behaviors:
        for name, group in report.by_behavior.items():
            rows.append((f"expected_behavior: {name}", *_essential_cells(group)))
    lines = ["", "Essential documents (what an answer needs, on the questions that state it):"]
    lines += _aligned(rows, right=(4,))
    with_interval = [(name, group.essential) for name, group in named if group.essential is not None]
    if any(essential.question_intervals is not None for _name, essential in with_interval):
        bounds = [("", *header[1:4])]
        few = []
        for name, essential in with_interval:
            if essential.question_intervals is None:
                few.append(name)
                continue
            cells = (essential.question_intervals[metric] for metric in _QUESTION_INTERVAL_METRICS)
            # The order of the columns: all, mean share, pooled share.
            recall, complete, pooled = (f"[{low:.2f}, {high:.2f}]" for low, high in cells)
            bounds.append((name, complete, recall, pooled))
        lines.append(f"95% intervals, from {resamples} resamples of the questions (seed {seed}):")
        lines += _aligned(bounds)
        lines += [f"{name}: fewer than {minimum} questions: no interval" for name in few]
    elif any(essential.few_questions for _name, essential in with_interval):
        lines.append(f"No interval: every line has fewer than {minimum} questions that state requirements.")
    lines += [
        "requirement: one document of essential_ids, or one group of essential_any_of (documents that",
        f"  carry the same fact): satisfied when one of its documents is among the first {k}. One whose",
        "  documents are all absent from the collection is left out.",
        f"all_essential@{k}: share of the questions with every requirement satisfied (their number / the",
        "  questions of the line that state requirements; the others are left out, n/a without any).",
        f"essential_recall@{k}: mean over these questions of the share of their requirements satisfied.",
        "pooled: requirements satisfied / requirements, these questions together (a long one weighs more).",
        f"unreachable: questions with more requirements than k={k}: they can never have them all.",
        "  For chain and aggregation questions raise --k (at most --chunk-limit, itself bounded by",
        "  RAG_ENGINE_SEARCH_MAX_LIMIT).",
        f"hit@1, hit@{k}, MRR and recall@{k} look at every expected document: essential, either-of or",
        "  acceptable.",
    ]
    out_of_reach = [result for result in report.results if result.essential_unreachable]
    if out_of_reach:
        lines.append("")
        lines.append(f"More requirements than k={k} ({len(out_of_reach)}):")
        for result in out_of_reach:
            requirements = result._requirements()
            count = result.essential_present if requirements is None else requirements[0]
            lines.append(f"  {_cut(result.id, 60)}: {count} requirements")
    return lines


def _filter_lines(effect: FilterEffect, k: int) -> list[str]:
    header = ("", "questions", "hit@1", f"hit@{k}", "MRR", f"essential_recall@{k}")
    rows = [header]
    for name, means in (("with their filters", effect.filtered), ("without filters", effect.unfiltered)):
        cells = [_number(means[metric]) for metric in (*_INTERVAL_METRICS, "essential_recall_at_k")]
        rows.append((name, str(effect.questions), *cells))
    count = effect.questions
    plural = "" if count == 1 else "s"
    return [
        "",
        f"What the filter brings ({count} question{plural} scored with their filters, then without):",
        *_aligned(rows, right=(1,)),
        "Means over these questions. The second runs are the kinds named #unfiltered above; they are",
        "  in the headline and not in overall (pooled).",
    ]


def _abstention_lines(report: EvalReport, k: int) -> list[str]:
    levels = [level for level, _count in report.abstentions[0].documents_at_or_above]
    count = report.abstention_questions
    plural = "" if count == 1 else "s"
    title = f"Abstention ({count} question{plural} whose expected behaviour is to abstain: searched, in no "
    lines = ["", title + "mean, not judged):"]
    header = ("", "top score", *(f">={level:g}" for level in levels), f"context@{k}", "top document")
    rows = [header]
    for entry in report.abstentions:
        top = entry.top_document
        if top is None:
            document = "nothing was returned"
        else:
            document = f"{_cut(top.doc_id, 24)}  {_cut(top.title, 22) if top.title else '(no title)'}"
        score = "n/a" if entry.top_score is None else f"{entry.top_score:.3f}"
        if entry.context_at_k is None:
            context = "-"
        else:
            there = len(entry.context_doc_ids) - len(entry.absent_context_doc_ids)
            context = f"{entry.context_at_k}/{there}"
        counts = (str(number) for _level, number in entry.documents_at_or_above)
        rows.append((entry.id, score, *counts, context, document))
    lines += _aligned(rows, right=tuple(range(1, len(levels) + 2)), name_width=30)
    lines += [
        ">=x: documents ranked with a score of at least x, among all those returned for the question.",
        f"context@{k}: the context documents of the question among the first {k}, over those that are in",
        "  the collection. A context document is one the question lists: a system may show it while it",
        "  says that the corpus does not answer. Returning one is neither a hit nor a fault; - without any.",
    ]
    top = report.top_scores
    scored = len(report.results)
    if top is None:
        lines.append("No question was scored: there is no top score to compare with.")
    else:
        scores = f"min {top.min:.3f}, median {top.median:.3f}, max {top.max:.3f}."
        if top.questions == scored:
            lines.append(f"For comparison, top score of the {top.questions} scored questions: {scores}")
        else:  # a search that returned nothing gives no top score
            nothing = scored - top.questions
            some = f"{top.questions} of the {scored} scored questions ({nothing} returned no document)"
            lines += [f"For comparison, top score of {some}:", f"  {scores}"]
    lines.append("No threshold is applied: whether to answer is decided by the caller of the search.")
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

    More parts are printed when the questions use the second version of the format (a
    question states its essential documents or another behaviour than ``answer``, abstains,
    or is scored again without its filters), each with its legend. Under the first legend, the
    lines of the first table again as means **per question**, with the number of questions
    that expect more documents than ``k``; then, between the tags and the questions left out:
    the metrics per ``expected_behavior`` (when a question has another one than ``answer``),
    the essential documents (when a question states them: the share of questions with all
    their requirements, the mean and the pooled share of requirements satisfied, intervals
    from resampling the questions), what the filter brings (questions scored again without
    their filters), and the questions that abstain with the scores they were given. In such a
    report a first table wider than 110 characters puts its intervals on a line of their own.
    A report of questions that use none of this is the report of the first version, to the
    character.
    """
    parameters = report.parameters
    k = parameters["k"]
    scored, left_out = len(report.results), len(report.not_in_collection)
    apart = report.abstention_questions
    second_version = _second_version(report)
    title = (
        f"Search evaluation of collection {parameters['collection']!r}: {scored} questions scored "
        f"({scored + left_out + apart} read, {left_out} not in the collection)"
    )
    settings = (
        f"k={k}, chunk_limit={parameters['chunk_limit']}, model {parameters['model']} "
        f"({parameters['backend']} backend), query normalisation "
        f"{parameters['query_normalization'] or 'off'}, rag-engine {parameters['engine_version']}"
    )
    lines = [title]
    extra = []
    if apart:
        extra.append(f"{apart} question{'' if apart == 1 else 's'} that abstain (searched, not scored)")
    if report.unfiltered_reruns:
        again = len(report.unfiltered_reruns)
        extra.append(f"{again} second run{'' if again == 1 else 's'} without filters")
    if extra:
        lines.append("and " + ", ".join(extra))
    lines.append(settings)
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
    rows = [_row(name, group) for name, group in _named_groups(report)]
    rows.append(_row("overall (pooled)", report.overall, pooled=True))
    lines.append("")
    lines += _table(rows, k, width=_LINE_WIDTH if second_version else None)
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
    if report.unfiltered_reruns:
        lines += [
            "#unfiltered: the questions of a kind that ask for it (run_unfiltered), scored again without",
            "  their filters: a kind of its own, asked without filters; not in overall (pooled).",
        ]
    if second_version:
        lines += _per_question_lines(report, k)
    if report.by_tag:
        lines.append("")
        lines.append("Per tag (per-document means):")
        lines += _table([_row(f"tag: {name}", group) for name, group in sorted(report.by_tag.items())], k)

    behaviors = any(name != DEFAULT_BEHAVIOR for name in report.by_behavior)
    if behaviors or apart:
        lines += _behavior_lines(report, k)
    if report.overall.essential is not None or report.headline.essential is not None:
        lines += _essential_lines(report, k, behaviors=behaviors)
    effect = report.filter_effect
    if effect is not None:
        lines += _filter_lines(effect, k)
    if report.abstentions:
        lines += _abstention_lines(report, k)

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
        if second_version:
            lines += [
                f"  tied_at_decision concerns hit@1 and hit@{k} only: recall@{k} and the essential metrics",
                "  are not checked for ties.",
            ]
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


# --- passages ---------------------------------------------------------------------------------------

#: Salt of the seed of the passage intervals, so they do not repeat the draws of the document ones.
_PASSAGE_INTERVAL_SEED = "passages"
#: The attributes of a :class:`PassageResult` a :class:`PassageReport` can be split by.
_PASSAGE_GROUPS = ("kind", "difficulty", "language")


_FOLDED = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
        "\u2212": "-",
        "\u00ad": None, "\u200b": None, "\u200c": None, "\u200d": None, "\u2060": None, "\ufeff": None,
    }
)  # fmt: skip


def _squash(text: str) -> str:
    """The form :func:`span_hit` compares: NFKD (the form chunks are stored in), curly quotes and
    dashes in ASCII, zero-width characters dropped, case-folded, white space runs as one space."""
    text = unicodedata.normalize(QUERY_NORMALIZATION, text).translate(_FOLDED)
    return " ".join(unicodedata.normalize(QUERY_NORMALIZATION, text.casefold()).split())


def span_parts(span: str) -> tuple[str, ...]:
    """The fragments of ``span`` split at :data:`SPAN_GAP`, as given (one fragment without a gap)."""
    return tuple(span.split(SPAN_GAP))


def span_held(haystack: str, needle: str) -> bool:
    """Whether the squashed ``haystack`` holds the squashed ``needle``: every fragment of it
    (:func:`span_parts`), in order and without overlap; a needle without a gap is a plain
    containment. A blank needle or fragment is never held."""
    at = 0
    for part in span_parts(needle):
        part = part.strip()
        found = haystack.find(part, at) if part else -1
        if found < 0:
            return False
        at = found + len(part)
    return True


def span_hit(hit_text: str | None, spans: Sequence[str]) -> bool:
    """Whether the passage ``hit_text`` holds one of ``spans``, ignoring case, white space and typography.

    Both sides are put in :data:`QUERY_NORMALIZATION` (the form the index stores text in), curly
    quotes and apostrophes become ASCII ones, en and em dashes become ``-``, zero-width characters
    and soft hyphens are dropped, text is case-folded and every run of white space (line breaks of
    a table or of a hard-wrapped paragraph included) becomes one space before the containment is
    tested. A span holding :data:`SPAN_GAP` is held when the passage holds each of its fragments,
    in order, whatever lies between them (a table's year row, then the answer row a few rows
    below): one passage must hold them all. A passage without text, no span, and a blank span
    give ``False``.
    """
    if not hit_text:
        return False
    haystack = _squash(hit_text)
    return any(span_held(haystack, needle) for needle in map(_squash, spans))


@dataclass(frozen=True)
class PassageResult:
    """One scored question of :func:`evaluate_passages`.

    ``rank`` is the 1-based position of the first passage holding an expected span among the
    ``k`` returned, or ``None``. ``returned`` is the number of passages the run gave (at most
    ``k`` are read).
    """

    id: str
    rank: int | None
    returned: int
    kind: str | None = None
    difficulty: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class PassageAbstention:
    """A question the corpus does not answer: searched, never scored.

    ``returned`` passages came back; ``top_score`` is the score of the first one, ``None`` when
    there was none.
    """

    id: str
    returned: int
    top_score: float | None


@dataclass(frozen=True)
class PassageReport:
    """The result of :func:`evaluate_passages`: passage metrics with intervals over the questions.

    ``results`` holds one :class:`PassageResult` per scored question, in the order of the
    questions; ``abstentions`` the questions that abstain, reported apart. ``k`` is the depth
    searched; ``hit_at(n)`` and ``mrr_at(n)`` read any ``n <= k``. Intervals are percentile
    intervals obtained by resampling the questions (``bootstrap_resamples`` draws, seed
    :data:`BOOTSTRAP_SEED`, level :data:`INTERVAL_LEVEL`), given only for a group of at least
    ``interval_min_questions`` questions. A sibling of :class:`EvalReport`, which scores
    documents and cannot say "this passage, not just this document".
    """

    k: int
    results: tuple[PassageResult, ...]
    abstentions: tuple[PassageAbstention, ...] = ()
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    interval_min_questions: int = DEFAULT_INTERVAL_MIN_QUESTIONS
    questions_file: str | None = None
    questions_sha256: str | None = None

    def _values(self, metric: str) -> list[float]:
        name, _, depth = metric.rpartition("_")
        if name not in ("hit_at", "mrr"):
            raise EvalError(f"unknown metric {metric!r} (hit_at_<n> or mrr_<n>)")
        if not depth.isdigit() or not 1 <= int(depth) <= self.k:
            raise EvalError(f"the depth of {metric!r} must be an integer from 1 to k={self.k}")
        n = int(depth)
        if name == "hit_at":
            return [float(r.rank is not None and r.rank <= n) for r in self.results]
        return [1.0 / r.rank if r.rank is not None and r.rank <= n else 0.0 for r in self.results]

    def value(self, metric: str) -> float:
        values = self._values(metric)
        return math.fsum(values) / len(values) if values else float("nan")

    def hit_at(self, n: int) -> float:
        """Share of the scored questions with an expected span among the first ``n`` passages."""
        return self.value(f"hit_at_{n}")

    def mrr_at(self, n: int) -> float:
        """Mean reciprocal rank of the first passage holding a span, ``0`` past ``n`` or never."""
        return self.value(f"mrr_{n}")

    def interval(self, metric: str) -> tuple[float, float] | None:
        """Interval of ``metric`` (``hit_at_5``, ``mrr_10``, ...), or ``None`` for too few questions."""
        values = self._values(metric)
        if not values or len(values) < self.interval_min_questions:
            return None
        return _intervals(
            [(v,) for v in values],
            resamples=self.bootstrap_resamples,
            seed=f"{BOOTSTRAP_SEED}:{_PASSAGE_INTERVAL_SEED}",
            metrics=("m",),
        )["m"]

    def groups(self, attribute: str) -> dict[str, PassageReport]:
        """The report split by ``attribute`` of the results (``kind``, ``difficulty`` or ``language``):
        one report per value, in order of first appearance, ``(none)`` for questions without one.
        Each keeps the depth and the interval settings, so a small group gets no interval."""
        if attribute not in _PASSAGE_GROUPS:
            raise EvalError(f"unknown grouping {attribute!r} (one of {', '.join(_PASSAGE_GROUPS)})")
        labels = [getattr(r, attribute) or _NO_KIND for r in self.results]
        return {
            label: replace(
                self,
                results=tuple(r for r, own in zip(self.results, labels, strict=True) if own == label),
                abstentions=(),
            )
            for label in dict.fromkeys(labels)
        }

    def _metric_names(self) -> list[str]:
        depths = sorted({n for n in (1, 3, 5, 10, self.k) if n <= self.k})
        return [f"{kind}_{n}" for n in depths for kind in ("hit_at", "mrr")]

    def to_dict(self) -> dict[str, Any]:
        """The report as plain data: each metric with its value and interval, the misses, the abstentions."""
        result: dict[str, Any] = {
            "k": self.k,
            "questions": len(self.results),
            "questions_file": self.questions_file,
            "questions_sha256": self.questions_sha256,
            "bootstrap_resamples": self.bootstrap_resamples,
            "interval_level": INTERVAL_LEVEL,
        }
        for metric in self._metric_names():
            interval = self.interval(metric)
            result[metric] = {
                "value": None if math.isnan(self.value(metric)) else self.value(metric),
                "interval": None if interval is None else list(interval),
            }
        result["misses"] = [r.id for r in self.results if r.rank is None]
        result["abstentions"] = [
            {"id": a.id, "returned": a.returned, "top_score": a.top_score} for a in self.abstentions
        ]
        return result

    def format(self) -> str:
        """The report as text: one line per metric, then the abstentions."""
        lines = [f"{len(self.results)} questions scored on passages, k={self.k}"]
        for metric in self._metric_names():
            kind, _, depth = metric.rpartition("_")
            label = f"hit@{depth}" if kind == "hit_at" else f"MRR@{depth}"
            interval = self.interval(metric)
            span = "" if interval is None else f"  [{interval[0]:.3f}, {interval[1]:.3f}]"
            lines.append(f"  {label:<8} {self.value(metric):.3f}{span}")
        if self.abstentions:
            tops = [a.top_score for a in self.abstentions if a.top_score is not None]
            best = f", highest top score {max(tops):.3f}" if tops else ""
            lines.append(f"  {len(self.abstentions)} abstention questions searched, not scored{best}")
        return "\n".join(lines)


def _passage_rank(question: Question, hits: Sequence[Hit]) -> int | None:
    """The 1-based rank at which ``hits`` answer ``question``, or ``None``.

    A passage counts only if it comes from one of the question's ``expected_doc_ids`` (when it
    lists any). With ``span_match="any"`` the rank is that of the first passage holding a span;
    with ``"all"`` it is the smallest ``r`` such that every group of spans is held, by one of
    its spans, by some passage among the first ``r`` (``None`` when a group is never found).
    """
    wanted = question.expected_doc_ids or ()
    groups = question.span_groups or tuple((span,) for span in question.expected_spans)
    if question.span_match != "all":
        groups = (question.expected_spans,)  # "any": a single group of every span
    firsts: list[int | None] = []
    for group in groups:
        firsts.append(
            next(
                (
                    i
                    for i, hit in enumerate(hits, start=1)
                    if (not wanted or hit.doc_id in wanted) and span_hit(hit.text, group)
                ),
                None,
            )
        )
    found = [rank for rank in firsts if rank is not None]
    if question.span_match == "all":
        return max(found) if found and len(found) == len(firsts) else None
    return min(found) if found else None


def evaluate_passages(
    questions: Sequence[Question],
    run: Callable[[Question], Sequence[Hit]],
    *,
    k: int = DEFAULT_K,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    interval_min_questions: int = DEFAULT_INTERVAL_MIN_QUESTIONS,
) -> PassageReport:
    """Score ``run`` on passages: where is the first passage holding an expected span?

    ``run(question)`` returns the passages (hits, best first) for the question: any retrieval
    (dense, hybrid, selected), the caller decides; at most ``k`` are read. A question that
    abstains is run and reported apart (:class:`PassageAbstention`), never scored; every other
    question needs ``expected_spans`` (``EvalError`` otherwise). The report gives hit@n and
    MRR@n for any ``n <= k``. When the question lists ``expected_doc_ids``, a passage counts
    only if it comes from one of them: a sentence repeated by another year's filing is no hit.
    A question with ``span_match="all"`` is a hit at the rank where its last group of spans is
    first found (every group must be held, by one of its spans, by a passage among the first
    ``k``; a lone span is a group of one).

    Returns a :class:`PassageReport` rather than an :class:`EvalReport`: that one ranks
    documents, has no notion of a passage, and its aggregates are built around expected
    documents. The interval machinery (:func:`_intervals`, level, seed, resamples) is shared.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise EvalError(f"'k' must be an integer >= 1, got {k!r}")
    results: list[PassageResult] = []
    abstentions: list[PassageAbstention] = []
    for question in questions:
        if question.expected_behavior != "abstain" and not question.expected_spans:
            raise EvalError(
                f"question {question.id!r} has no 'expected_spans': it cannot be scored on passages"
            )
        hits = list(run(question))[:k]
        if question.expected_behavior == "abstain":
            top = hits[0].score if hits else None
            abstentions.append(PassageAbstention(question.id, len(hits), top))
            continue
        rank = _passage_rank(question, hits)
        results.append(
            PassageResult(question.id, rank, len(hits), question.kind, question.difficulty, question.language)
        )
    return PassageReport(
        k=k,
        results=tuple(results),
        abstentions=tuple(abstentions),
        bootstrap_resamples=bootstrap_resamples,
        interval_min_questions=interval_min_questions,
        questions_file=getattr(questions, "path", None),
        questions_sha256=getattr(questions, "sha256", None),
    )
