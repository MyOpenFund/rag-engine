"""Check a question file without any collection: what it holds, and whether its documents exist.

:func:`rag_engine.eval.load_questions` refuses a file that breaks a rule of the format.
:func:`check_questions` then counts what a file that loads is made of (kinds, tags, expected
behaviours, languages, filters, how many documents the questions expect) and points at what
is allowed and probably not meant: a question text used twice, a kind that mixes questions
with and without filters, more essential documents than ``k``. Given the catalogue of a
corpus (:func:`load_catalogue`: the JSON Lines manifest files), it also looks every document
a question names up in it.

``rag-engine eval-check`` prints the result (:func:`format_check`). Nothing here connects to
Qdrant or embeds anything, and only the standard library is used besides the question loader.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .eval import BEHAVIORS, DEFAULT_K, EvalError, Question, _filters_dict

__all__ = [
    "CatalogueEntry",
    "DocumentReference",
    "QuestionCheck",
    "check_questions",
    "format_check",
    "load_catalogue",
]

_NONE = "(none)"
_NOT_STATED = "not stated"
_LINE_WIDTH = 110


@dataclass(frozen=True)
class CatalogueEntry:
    """A document of the catalogue: its ``doc_id``, and the ``mime_type`` and ``doc_type`` of its row."""

    doc_id: str
    mime_type: str | None = None
    doc_type: str | None = None


def load_catalogue(patterns: Iterable[str | os.PathLike[str]]) -> dict[str, CatalogueEntry]:
    """The documents of the manifest files matching ``patterns`` (paths or glob patterns), by ``doc_id``.

    A manifest is UTF-8 JSON Lines, one document per line, with a ``doc_id`` (a non-blank
    string) and, when known, a ``mime_type`` and a ``doc_type``; the other fields are not
    read. A ``doc_id`` met again keeps its first row. :class:`rag_engine.eval.EvalError` for a
    pattern that matches no file, a line that is not a JSON object, a row without a
    ``doc_id``, a ``mime_type`` or ``doc_type`` that is not a string; the ``OSError`` of the
    system for a file that cannot be read.
    """
    paths: list[str] = []
    for pattern in patterns:
        found = sorted(glob.glob(os.fspath(pattern)))
        if not found:
            raise EvalError(f"no manifest file matches {os.fspath(pattern)!r}")
        paths.extend(path for path in found if path not in paths)
    if not paths:
        raise EvalError("no manifest file was given")
    catalogue: dict[str, CatalogueEntry] = {}
    for path in paths:
        with open(path, "rb") as handle:
            raw = handle.read()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise EvalError(f"{path}: the manifest is not UTF-8 ({exc.reason} at byte {exc.start})") from None
        for number, line in enumerate(text.split("\n"), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise EvalError(f"{path}, line {number}: not valid JSON ({exc})") from None
            if not isinstance(row, dict):
                raise EvalError(f"{path}, line {number}: a line must hold one JSON object")
            doc_id = row.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise EvalError(f"{path}, line {number}: 'doc_id' is missing or not a non-blank string")
            for name in ("mime_type", "doc_type"):
                if row.get(name) is not None and not isinstance(row[name], str):
                    raise EvalError(f"{path}, line {number}: '{name}' must be a string, got {row[name]!r}")
            catalogue.setdefault(doc_id, CatalogueEntry(doc_id, row.get("mime_type"), row.get("doc_type")))
    return catalogue


@dataclass(frozen=True)
class DocumentReference:
    """A document some questions name, that the catalogue does not hold or holds outside the scope.

    ``questions`` gives, for each question that names it, its id, its line in the file
    (``None`` for a question built in code) and how it names it: ``essential``, ``acceptable``
    or ``expected`` (a question that does not say which documents are essential).
    ``mime_type`` and ``doc_type`` are those of the catalogue, ``None`` for a document that is
    not in it.
    """

    doc_id: str
    questions: tuple[tuple[str, int | None, str], ...]
    mime_type: str | None = None
    doc_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "mime_type": self.mime_type,
            "doc_type": self.doc_type,
            "questions": [{"id": id_, "line": line, "as": role} for id_, line, role in self.questions],
        }


@dataclass(frozen=True)
class QuestionCheck:
    """What :func:`check_questions` found.

    ``questions``
        Number of questions.
    ``by_kind``, ``by_tag``, ``by_behavior``, ``by_language``
        Number of questions per kind, tag, expected behaviour and language (``"(none)"`` for
        the questions without a kind or a language), most frequent first.
    ``with_filters``, ``run_unfiltered``
        Number of questions that carry filters, and of those that ask to be scored again
        without them.
    ``essential_counts``, ``expected_counts``
        How many questions name how many essential documents, and how many expected
        documents (essential and acceptable together): the number of documents, then the
        number of questions, in increasing order. ``essential_not_stated`` counts the
        questions that do not say which documents are essential.
    ``without_documents``
        Number of questions that expect no document (they abstain): searched, never scored.
    ``more_essential_than_k``
        The ids of the questions with more essential documents than ``k``: at that ``k`` they
        can never have all of them among the first ``k`` documents.
    ``duplicate_texts``
        The groups of questions with the same text (compared without regard to case and to
        spacing): each group is the ids of its questions, in the order of the file.
    ``mixed_kinds``
        The kinds that hold questions with filters and questions without: the report treats
        such a kind as a filtered one, and keeps all of it out of the headline.
    ``documents``
        Number of distinct documents the questions name.
    ``catalogue_documents``
        Number of documents of the catalogue; ``None`` when no catalogue was given, and the
        four fields below are then empty.
    ``missing``
        The documents named that are not in the catalogue (:class:`DocumentReference`).
    ``out_of_scope``
        Those that are in it with another ``mime_type`` or ``doc_type`` than the scope given.
    ``by_doc_type``, ``by_mime_type``
        Number of distinct documents named, per ``doc_type`` and per ``mime_type`` of the
        catalogue.

    :attr:`ok` is false when a document is missing or out of scope.
    """

    questions: int
    k: int
    by_kind: Mapping[str, int]
    by_tag: Mapping[str, int]
    by_behavior: Mapping[str, int]
    by_language: Mapping[str, int]
    with_filters: int
    run_unfiltered: int
    essential_counts: Mapping[int, int]
    essential_not_stated: int
    expected_counts: Mapping[int, int]
    without_documents: int
    more_essential_than_k: tuple[str, ...]
    duplicate_texts: tuple[tuple[str, ...], ...]
    mixed_kinds: tuple[str, ...]
    documents: int
    catalogue_documents: int | None = None
    missing: tuple[DocumentReference, ...] = ()
    out_of_scope: tuple[DocumentReference, ...] = ()
    by_doc_type: Mapping[str, int] | None = None
    by_mime_type: Mapping[str, int] | None = None
    scope_doc_types: tuple[str, ...] | None = None
    scope_mime_types: tuple[str, ...] | None = None

    @property
    def ok(self) -> bool:
        """Every document named is in the catalogue and in the scope (true without a catalogue)."""
        return not self.missing and not self.out_of_scope


def _frequency(counter: Counter[str]) -> dict[str, int]:
    """``counter`` as a dict, most frequent first, then by name."""
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _scope(name: str, values: Iterable[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise EvalError(f"{name} must be a list of strings, got the single string {values!r}")
    scope = tuple(values)
    if not scope or not all(isinstance(value, str) and value.strip() for value in scope):
        raise EvalError(f"{name} must be a non-empty list of non-blank strings, got {scope!r}")
    return scope


def check_questions(
    questions: Iterable[Question],
    *,
    catalogue: Mapping[str, CatalogueEntry] | None = None,
    doc_types: Iterable[str] | None = None,
    mime_types: Iterable[str] | None = None,
    k: int = DEFAULT_K,
) -> QuestionCheck:
    """Count what ``questions`` are made of and, with ``catalogue``, look their documents up in it.

    ``questions`` is what :func:`rag_engine.eval.load_questions` returned, or any iterable
    of :class:`rag_engine.eval.Question`: they are valid already, and nothing is searched.
    ``catalogue`` is what :func:`load_catalogue` returned. ``doc_types`` and ``mime_types``
    narrow it to the scope of a collection (the doc types and the MIME types that were
    ingested): a document named by a question and found in the catalogue outside that scope
    is reported as out of scope. Both need a catalogue. ``k`` is the number of documents the
    evaluation will look at (default 10): the questions with more essential documents are
    listed.

    :class:`rag_engine.eval.EvalError` for a ``k`` that is not an integer >= 1, a scope
    without a catalogue, a scope that is not a list of strings.
    """
    asked = tuple(questions)
    if type(k) is not int or k < 1:
        raise EvalError(f"k must be an integer >= 1, got {k!r}")
    types, mimes = _scope("doc_types", doc_types), _scope("mime_types", mime_types)
    if catalogue is None and (types is not None or mimes is not None):
        raise EvalError("a scope (doc_types, mime_types) needs a catalogue to be checked against")

    kinds: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    behaviors: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    essential: Counter[int] = Counter()
    expected: Counter[int] = Counter()
    texts: dict[str, list[str]] = {}
    filtered_kinds: set[str] = set()
    unfiltered_kinds: set[str] = set()
    references: dict[str, list[tuple[str, int | None, str]]] = {}
    with_filters = reruns = not_stated = 0
    beyond: list[str] = []
    for question in asked:
        kind = question.kind if question.kind is not None else _NONE
        kinds[kind] += 1
        tags.update(question.tags)
        behaviors[question.expected_behavior] += 1
        languages[question.language if question.language is not None else _NONE] += 1
        if _filters_dict(question.filters) is not None:
            with_filters += 1
            filtered_kinds.add(kind)
        elif question.expected_doc_ids:  # a question without documents is not scored: in no kind
            unfiltered_kinds.add(kind)
        reruns += question.run_unfiltered
        expected[len(question.expected_doc_ids or ())] += 1
        if question.essential_ids is None:
            not_stated += 1
        else:
            essential[len(question.essential_ids)] += 1
            if len(question.essential_ids) > k:
                beyond.append(question.id)
        texts.setdefault(" ".join(question.question.split()).casefold(), []).append(question.id)
        for doc_id in question.expected_doc_ids or ():
            if question.essential_ids is None:
                role = "expected"
            else:
                role = "essential" if doc_id in question.essential_ids else "acceptable"
            references.setdefault(doc_id, []).append((question.id, question.line, role))

    missing: list[DocumentReference] = []
    outside: list[DocumentReference] = []
    by_doc_type: Counter[str] | None = None
    by_mime_type: Counter[str] | None = None
    if catalogue is not None:
        by_doc_type, by_mime_type = Counter(), Counter()
        for doc_id, users in sorted(references.items()):
            entry = catalogue.get(doc_id)
            if entry is None:
                missing.append(DocumentReference(doc_id, tuple(users)))
                continue
            by_doc_type[entry.doc_type if entry.doc_type is not None else _NONE] += 1
            by_mime_type[entry.mime_type if entry.mime_type is not None else _NONE] += 1
            in_types = types is None or entry.doc_type in types
            in_mimes = mimes is None or entry.mime_type in mimes
            if not (in_types and in_mimes):
                outside.append(DocumentReference(doc_id, tuple(users), entry.mime_type, entry.doc_type))

    return QuestionCheck(
        questions=len(asked),
        k=k,
        by_kind=_frequency(kinds),
        by_tag=_frequency(tags),
        by_behavior={name: behaviors[name] for name in BEHAVIORS if behaviors[name]},
        by_language=_frequency(languages),
        with_filters=with_filters,
        run_unfiltered=reruns,
        essential_counts=dict(sorted(essential.items())),
        essential_not_stated=not_stated,
        expected_counts=dict(sorted(expected.items())),
        without_documents=expected[0],
        more_essential_than_k=tuple(beyond),
        duplicate_texts=tuple(tuple(ids) for ids in texts.values() if len(ids) > 1),
        mixed_kinds=tuple(sorted(filtered_kinds & unfiltered_kinds)),
        documents=len(references),
        catalogue_documents=None if catalogue is None else len(catalogue),
        missing=tuple(missing),
        out_of_scope=tuple(outside),
        by_doc_type=None if by_doc_type is None else _frequency(by_doc_type),
        by_mime_type=None if by_mime_type is None else _frequency(by_mime_type),
        scope_doc_types=types,
        scope_mime_types=mimes,
    )


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


def _counts(label: str, counts: Mapping[Any, int]) -> list[str]:
    return _wrapped(label, [f"{name} {count}" for name, count in counts.items()] or ["none"])


def _reference_lines(reference: DocumentReference) -> list[str]:
    users = []
    for id_, line, role in reference.questions:
        users.append(f"{id_} ({role}{'' if line is None else f', line {line}'})")
    label = f"  {reference.doc_id}"
    if reference.doc_type is not None or reference.mime_type is not None:
        label += f" [{reference.doc_type or _NONE}, {reference.mime_type or _NONE}]"
    return _wrapped(label + ":", users)


def format_check(check: QuestionCheck, *, name: str | None = None, sha256: str | None = None) -> str:
    """``check`` as a short text for a terminal; ``name`` and ``sha256`` identify the question file.

    The counts first, then what deserves a look (texts used twice, kinds that mix questions
    with and without filters, more essential documents than ``k``), then, when a catalogue was
    given, the documents that are not in it or not in the scope. The last line says whether
    the file passed. The text has no trailing line break.
    """
    plural = "" if check.questions == 1 else "s"
    title = f"{check.questions} question{plural}"
    if name:
        title = f"{name}: {title}"
    if sha256:
        title += f" (sha256 {sha256[:12]})"
    lines = [title]
    lines += _counts("kind:", check.by_kind)
    lines += _counts("tag:", check.by_tag)
    lines += _counts("expected_behavior:", check.by_behavior)
    lines += _counts("language:", check.by_language)
    lines.append(
        f"filters: {check.with_filters} with filters, {check.run_unfiltered} of them scored again without "
        "(run_unfiltered)"
    )
    essential: dict[Any, int] = {}
    if check.essential_not_stated:
        essential[_NOT_STATED] = check.essential_not_stated
    essential.update(check.essential_counts)
    lines += _wrapped(
        "essential documents per question (documents: questions):",
        [f"{number}: {count}" for number, count in essential.items()],
    )
    lines += _wrapped(
        "expected documents per question (documents: questions):",
        [f"{number}: {count}" for number, count in check.expected_counts.items()],
    )
    lines.append(f"distinct documents named: {check.documents}")
    if check.without_documents:
        lines.append(
            f"without any document: {check.without_documents} (they abstain: searched and reported apart, "
            "never scored)"
        )

    lines.append("")
    lines.append(f"duplicate question texts: {len(check.duplicate_texts)}")
    for ids in check.duplicate_texts:
        lines += _wrapped("  the same text:", list(ids))
    if check.mixed_kinds:
        lines += _wrapped("kinds with and without filters:", list(check.mixed_kinds))
        lines += [
            "  One question with filters makes its kind a filtered kind: the whole kind is kept out of the",
            "  headline. Give the questions with filters a kind of their own to keep the others in it.",
        ]
    if check.more_essential_than_k:
        lines += _wrapped(f"more essential documents than k={check.k}:", list(check.more_essential_than_k))
        lines.append(
            f"  At k={check.k} they can never have all their essential documents among the first {check.k}: "
            "run with a larger --k."
        )

    if check.catalogue_documents is not None:
        lines.append("")
        lines.append(f"catalogue: {check.catalogue_documents} documents")
        if check.scope_doc_types is not None or check.scope_mime_types is not None:
            scope = []
            if check.scope_doc_types is not None:
                scope.append("doc types " + ", ".join(check.scope_doc_types))
            if check.scope_mime_types is not None:
                scope.append("MIME types " + ", ".join(check.scope_mime_types))
            lines += _wrapped("scope:", ["; ".join(scope)])
        lines += _counts("documents named, per doc type:", check.by_doc_type or {})
        lines += _counts("documents named, per MIME type:", check.by_mime_type or {})
        lines.append(f"not in the catalogue: {len(check.missing)}")
        for reference in check.missing:
            lines += _reference_lines(reference)
        if check.scope_doc_types is not None or check.scope_mime_types is not None:
            lines.append(f"in the catalogue, outside the scope: {len(check.out_of_scope)}")
            for reference in check.out_of_scope:
                lines += _reference_lines(reference)

    lines.append("")
    if check.ok:
        checked = (
            "" if check.catalogue_documents is None else ", and every document it names is in the catalogue"
        )
        lines.append(f"The file is valid{checked}.")
    else:
        count = len(check.missing) + len(check.out_of_scope)
        lines.append(
            f"The file is valid, but {count} document{'' if count == 1 else 's'} it names "
            f"{'is' if count == 1 else 'are'} not in the catalogue or not in the scope."
        )
    return "\n".join(lines)
