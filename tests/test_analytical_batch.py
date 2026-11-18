"""The analytical question file: what it holds, and its report on a collection made for the test.

The documents of the real corpus are not here. A small collection holds one chunk per document the
questions name, and the text of some questions as a second chunk of a document they require, so
that every part of the report has something to show. The numbers mean nothing; the shape does.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.eval import evaluate, format_report, load_questions
from rag_engine.eval_check import check_questions
from rag_engine.store import QdrantStore

BATCH = Path(__file__).resolve().parent.parent / "eval" / "analytical_v2.jsonl"
SPEC = ModelSpec(name="tiny", dim=8, passage_prefix="", query_prefix="")
# The questions the corpus cannot answer: three of the first batch, then an-71 to an-82.
ABSTAINING = ["an-25", "an-53", "an-64b", *(f"an-{number}" for number in range(71, 83))]


@pytest.fixture(scope="module")
def questions():
    return load_questions(BATCH)


@pytest.fixture(scope="module")
def report(questions):
    texts: dict[str, list[str]] = {}
    for question in questions:
        for doc_id in question.expected_doc_ids:
            texts.setdefault(doc_id, [f"document {doc_id}"])
    for number, question in enumerate(questions):
        if question.requirements and number % 3:  # two questions of three find a document they require
            texts[question.requirements[0][0]].append(question.question)
    store = QdrantStore(client=QdrantClient(":memory:"))
    store.ensure_collection("c", SPEC.dim)
    embedder = FakeEmbedder(SPEC)
    points = []
    for doc_id, chunks in sorted(texts.items()):
        for index, vector in enumerate(embedder.encode_passage(chunks)):
            payload = {
                "doc_id": doc_id,
                "text": chunks[index],
                "chunk_number": index,
                "title": f"Title {doc_id}",
            }
            points.append(models.PointStruct(id=len(points) + 1, vector=vector.tolist(), payload=payload))
    store.upsert("c", points)
    return evaluate(questions, collection="c", store=store, embedder=FakeEmbedder(SPEC))


def test_what_the_batch_holds(questions):
    assert len(questions) == 91
    behaviors = Counter(question.expected_behavior for question in questions)
    assert behaviors == {"answer": 72, "correct_premise": 4, "abstain": 15}
    assert Counter(question.language for question in questions) == {"en": 81, "fr": 10}
    assert Counter(question.kind for question in questions) == {"analytical": 86, "analytical+filters": 5}
    assert all(question.essential_ids is not None for question in questions)
    assert len({doc_id for question in questions for doc_id in question.expected_doc_ids}) == 424


def test_the_questions_that_abstain_list_context_documents_only(questions):
    abstaining = [question for question in questions if question.expected_behavior == "abstain"]
    assert [question.id for question in abstaining] == ABSTAINING
    for question in abstaining:
        assert question.essential_ids == () and question.essential_any_of == ()
        assert question.acceptable_ids and question.must_include
        assert "out-of-corpus" in question.tags
    # The twelve added later say where the true answer is, and which of the two absences they test.
    added = [
        question
        for question in abstaining
        if question.id[:3] == "an-" and question.id[3:].isdigit() and int(question.id[3:]) >= 71
    ]
    assert len(added) == 12
    for question in added:
        assert "https://" in question.rationale and "True answer" in question.rationale, question.id
        assert len({"absent-bank", "coverage-gap"} & set(question.tags)) == 1, question.id
    assert sum(question.language == "fr" for question in added) == 2
    # A fact about a bank that is not in the corpus, reported by another bank's document, is an answer.
    indirect = [question for question in questions if "indirect-source" in question.tags]
    assert [question.id for question in indirect] == ["an-55", "an-57"]
    assert all(question.expected_behavior == "answer" and question.essential_ids for question in indirect)


def test_the_variants_are_tagged(questions):
    suffixed = {question.id: question.tags for question in questions if not question.id[-1].isdigit()}
    assert sorted(suffixed) == [
        "an-01a",
        "an-03a",
        "an-04b",
        "an-07a",
        "an-08a",
        "an-13a",
        "an-14b",
        "an-17a",
        "an-20b",
        "an-64b",
    ]
    for id_, tags in suffixed.items():
        assert ("variant:easy" in tags) == id_.endswith("a"), id_
        assert ("variant:extra" in tags) == id_.endswith("b"), id_
    tagged = [
        question.id for question in questions if any(tag.startswith("variant:") for tag in question.tags)
    ]
    assert sorted(tagged) == sorted(suffixed)


def test_a_rationale_describes_the_documents(questions):
    for question in questions:
        assert question.rationale and question.rationale.strip() == question.rationale
        process = re.search(r"\b(v1|required by|asked for)\b", question.rationale, re.IGNORECASE)
        assert process is None, question.id


def test_either_of_groups_of_the_batch(questions):
    by_id = {question.id: question for question in questions}
    check = check_questions(questions)
    assert (check.either_of_questions, check.either_of_groups) == (37, 94)
    assert check.more_requirements_than_k == ("an-23",)
    assert len(by_id["an-23"].requirements) == 12
    # Twenty-four sets of minutes carry the same proposal: two requirements, not three documents picked.
    kiuchi = by_id["an-67"]
    assert [len(group) for group in kiuchi.requirements] == [11, 8] and len(kiuchi.acceptable_ids) == 5
    # Eight meetings, each in its statement or its minutes.
    assert [len(group) for group in by_id["an-66"].requirements] == [2] * 8
    assert max(len(question.requirements) for question in questions if question.id != "an-23") <= 8


def test_the_report_of_the_batch(report):
    assert len(report.results) == 76 and len(report.unfiltered_reruns) == 5
    assert [entry.id for entry in report.abstentions] == ABSTAINING
    assert list(report.by_behavior) == ["answer", "correct_premise"]
    assert all(result.expected_behavior != "abstain" for result in report.searches)
    assert report.expected_beyond_k == 18 and report.essential_unreachable == 1
    essential = report.by_kind["analytical"].essential
    assert essential.questions == 71 and essential.question_intervals is not None
    text = format_report(report)
    lines = text.splitlines()
    # Everything above the misses fits the width: the tables, their legends, the lists of ids.
    above = lines[: next(index for index, line in enumerate(lines) if line.startswith("Misses ("))]
    assert [line for line in above if len(line) > 110] == []
    assert "and 15 questions that abstain (searched, not scored), 5 second runs without filters" in lines
    assert any(line.startswith("  95% interval") for line in lines)  # the wide first table, laid out
    assert "Abstention (15 questions whose expected behaviour is to abstain" in text
    assert any(line.split()[:1] == ["an-64b"] and "/2" in line for line in lines)
    assert "More requirements than k=10 (1):\n  an-23: 12 requirements" in text
    (count,) = [line for line in lines if line.startswith("kind: analytical  ") and "(" in line]
    assert re.search(r"\d\.\d\d \(\d+/71\)", count)
