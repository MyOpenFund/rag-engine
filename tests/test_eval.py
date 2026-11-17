"""The evaluation harness: the question file, document-level scoring, the report.

Run against qdrant-client's local mode with a ``FakeEmbedder``. The fixture of
``eval/fixtures/`` is built so that every ranking the metrics depend on is certain without
any semantics: a question is the text of a chunk, or its filters isolate the answer.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from fractions import Fraction

import pytest
from eval_fixture import QUESTIONS, fixture_corpus, fixture_embedder, fixture_points, load_fixture
from qdrant_client import QdrantClient, models

import rag_engine
from rag_engine.config import Settings
from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.errors import RagEngineError
from rag_engine.eval import (
    BOOTSTRAP_SEED,
    Aggregate,
    EvalError,
    EvalReport,
    Means,
    Question,
    QuestionResult,
    QuestionSet,
    RankedDocument,
    evaluate,
    format_report,
    load_questions,
)
from rag_engine.search import Hit, SearchFilters
from rag_engine.store import QdrantStore

COLLECTION = "eval-fixture"


@pytest.fixture(scope="module")
def store():
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client, settings=Settings(batch_size=16))
    assert load_fixture(store, COLLECTION) == 32
    yield store
    client.close()


@pytest.fixture(scope="module")
def questions():
    return load_questions(QUESTIONS)


@pytest.fixture(scope="module")
def report(store, questions):
    return evaluate(questions, collection=COLLECTION, store=store, embedder=fixture_embedder())


def write(tmp_path, *lines) -> str:
    path = tmp_path / "questions.jsonl"
    text = "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines)
    path.write_text(text + "\n", encoding="utf-8")
    return str(path)


def q(**changes):
    base = {"id": "q1", "question": "What was decided?", "expected_doc_ids": ["d1"]}
    base.update(changes)
    return {key: value for key, value in base.items() if value is not ...}


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------
def test_the_fixture_is_what_its_documentation_says():
    corpus = fixture_corpus()
    assert len(corpus["documents"]) == 13
    assert sum(len(document["chunks"]) for document in corpus["documents"]) == 32
    assert len(fixture_points()) == 32
    # The combined document repeats one chunk of the decision and one of the statement, nothing else.
    chunks = {document["doc_id"]: document["chunks"] for document in corpus["documents"]}
    assert chunks["fx-ecb-combined-2016-03"] == [
        chunks["fx-ecb-decision-2016-03"][2],
        chunks["fx-ecb-statement-2016-03"][1],
    ]
    texts = [text for document in corpus["documents"] for text in document["chunks"]]
    assert len(set(texts)) == len(texts) - 2
    # A question equal to a chunk gets the vector of the chunk only if both prefixes are equal.
    assert corpus["model"]["passage_prefix"] == corpus["model"]["query_prefix"]


# ---------------------------------------------------------------------------
# load_questions
# ---------------------------------------------------------------------------
def test_load_the_fixture_questions(questions):
    assert isinstance(questions, QuestionSet)
    assert len(questions) == 13
    assert [question.id for question in questions] == [f"fx-q{n:02d}" for n in range(1, 14)]
    assert questions.sha256 == hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()
    assert questions.path == str(QUESTIONS)
    first, third, sixth = questions[0], questions[2], questions[5]
    assert isinstance(first, Question)
    assert first.expected_doc_ids == ("fx-ecb-decision-2016-03",)
    assert (first.filters, first.kind, first.line) == (None, "verbatim", 1)
    assert first.tags == ("bank:ecb", "doc_type:A1")
    # Every question of the fixture carries several tags: a breakdown must count each of them.
    assert all(len(question.tags) >= 2 for question in questions)
    assert first.rationale.startswith("The question is the text")
    assert third.filters == SearchFilters(doc_types=["A3"])
    assert third.expected_doc_ids == ("fx-gb-minutes-2016-08", "fx-gb-report-2016-08")
    assert (sixth.language, sixth.line) == ("fr", 6)
    assert questions[4].filters.date_from == datetime.date(2016, 5, 1)


def test_load_minimal_question_and_every_filter(tmp_path):
    filters = {
        "source_codes": ["ecb"],
        "doc_types": ["A1", "A2"],
        "languages": ["en"],
        "doc_ids": ["d1"],
        "date_from": "2016-01-01",
        "date_to": "2016-12-31",
        "corpus": "central-bank",
    }
    path = write(tmp_path, q(), "", "   ", q(id="q2", filters=filters), q(id="q3", filters={}))
    loaded = load_questions(path)
    assert [question.id for question in loaded] == ["q1", "q2", "q3"]
    assert [question.line for question in loaded] == [1, 4, 5]
    one, two, three = loaded
    assert (one.kind, one.tags, one.language, one.rationale, one.filters) == (None, (), None, None, None)
    assert two.filters == SearchFilters(**filters)
    assert three.filters is None
    assert one.to_dict() == {"id": "q1", "question": "What was decided?", "expected_doc_ids": ["d1"]}
    assert two.to_dict()["filters"] == filters


def test_a_null_filter_field_does_not_restrict(tmp_path):
    path = write(tmp_path, q(filters={"source_codes": None, "doc_types": ["A1"]}), q(id="q2", filters=None))
    one, two = load_questions(path)
    assert one.filters == SearchFilters(doc_types=["A1"])
    assert two.filters is None


@pytest.mark.parametrize(
    ("lines", "line", "match"),
    [
        (["{not json"], 1, "not valid JSON"),
        (["[1, 2]"], 1, "JSON object"),
        ([q(), '"text"'], 2, "JSON object"),
        ([q(), q()], 2, "duplicate id 'q1'.*line 1"),
        ([q(id=...)], 1, "'id' is missing"),
        ([q(id="")], 1, "'id'"),
        ([q(id=7)], 1, "'id'"),
        ([q(question=...)], 1, "'question' is missing"),
        ([q(question="   ")], 1, "'question'"),
        ([q(question="\u0301\u200b")], 1, "searchable"),
        ([q(), q(id="q2", question="\u00b4")], 2, "no searchable character once put in NFKD"),
        ([q(question=" \u00a8 \u02dc ")], 1, "NFKD"),
        ([q(expected_doc_ids=...)], 1, "'expected_doc_ids' is missing"),
        ([q(expected_doc_ids=[])], 1, "'expected_doc_ids'.*empty"),
        ([q(expected_doc_ids="d1")], 1, "'expected_doc_ids'.*list"),
        ([q(expected_doc_ids=["d1", ""])], 1, "'expected_doc_ids'"),
        ([q(expected_doc_ids=["d1", 3])], 1, "'expected_doc_ids'"),
        ([q(expected_doc_ids=["d1", "d1"])], 1, "'expected_doc_ids'.*twice"),
        ([q(), q(id="q2"), q(id="q3", filters={"source": ["ecb"]})], 3, "unknown filter 'source'"),
        ([q(filters=["ecb"])], 1, "'filters'.*object"),
        ([q(filters={"date_from": "2016"})], 1, "date_from"),
        ([q(filters={"date_from": "2016-02-30"})], 1, "date_from"),
        ([q(filters={"date_from": "2016-03-02", "date_to": "2016-03-01"})], 1, "after date_to"),
        ([q(filters={"source_codes": "ecb"})], 1, "source_codes"),
        ([q(filters={"doc_types": []})], 1, "doc_types"),
        ([q(expected=["d1"])], 1, "unknown field 'expected'"),
        ([q(kind=3)], 1, "'kind'"),
        ([q(kind="")], 1, "'kind'"),
        ([q(tags="a")], 1, "'tags'"),
        ([q(tags=["a", 1])], 1, "'tags'"),
        ([q(language=["fr"])], 1, "'language'"),
        ([q(rationale=1)], 1, "'rationale'"),
    ],
)
def test_a_bad_question_file_is_refused_with_its_line(tmp_path, lines, line, match):
    path = write(tmp_path, *lines)
    with pytest.raises(EvalError, match=match) as info:
        load_questions(path)
    assert f"line {line}" in str(info.value)
    assert "questions.jsonl" in str(info.value)


def test_eval_error_is_a_value_error_of_the_engine():
    assert issubclass(EvalError, ValueError)
    assert issubclass(EvalError, RagEngineError)


def test_an_empty_question_file_is_refused(tmp_path):
    with pytest.raises(EvalError, match="no question"):
        load_questions(write(tmp_path, ""))


def test_a_missing_question_file_is_an_os_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_questions(tmp_path / "absent.jsonl")


def test_a_question_file_that_is_not_utf8_is_refused(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_bytes(b'{"id": "q1", "question": "caf\xe9", "expected_doc_ids": ["d1"]}\n')
    with pytest.raises(EvalError, match="UTF-8"):
        load_questions(path)


def test_a_byte_order_mark_is_accepted(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(q()).encode() + b"\n")
    assert [question.id for question in load_questions(path)] == ["q1"]


# ---------------------------------------------------------------------------
# evaluate on the fixture: exact values
# ---------------------------------------------------------------------------
EXPECTED_RANKS = {
    "fx-q01": 1,
    "fx-q02": 1,
    "fx-q03": 1,
    "fx-q04": 1,
    "fx-q05": 1,
    "fx-q06": 1,
    "fx-q07": 2,
    "fx-q08": None,
    "fx-q10": 1,
    "fx-q11": 1,
    "fx-q12": 2,
    "fx-q13": 1,
}


def exactly(group: Aggregate, questions, documents, per_question, per_document):
    """Check the counts and the eight means of ``group`` against exact fractions."""
    assert (group.questions, group.documents) == (questions, documents)
    got = (group.hit_at_1, group.hit_at_k, group.mrr, group.recall_at_k)
    assert got == pytest.approx([float(value) for value in per_question], abs=1e-12)
    assert isinstance(group.per_document, Means)
    means = group.per_document
    got = (means.hit_at_1, means.hit_at_k, means.mrr, means.recall_at_k)
    assert got == pytest.approx([float(value) for value in per_document], abs=1e-12)


F = Fraction


def test_fixture_ranks(report):
    assert isinstance(report, EvalReport)
    assert {result.id: result.rank for result in report.results} == EXPECTED_RANKS
    assert [result.id for result in report.results] == list(EXPECTED_RANKS)  # the order of the file


def test_fixture_pooled_metrics_are_exact(report):
    # Twelve scored questions, each counted once: hit@1 9, hit@10 11, reciprocal ranks 10, recall 10.5.
    # Eleven documents are expected by them and present.
    overall = report.overall
    assert isinstance(overall, Aggregate)
    assert (overall.questions, overall.documents) == (12, 11)
    assert overall.hit_at_1 == pytest.approx(9 / 12, abs=1e-12)
    assert overall.hit_at_k == pytest.approx(11 / 12, abs=1e-12)
    assert overall.mrr == pytest.approx(10 / 12, abs=1e-12)
    assert overall.recall_at_k == pytest.approx(10.5 / 12, abs=1e-12)
    assert overall.intervals is None and overall.few_documents is False  # no interval is asked for it


def test_the_headline_is_the_unfiltered_kinds_averaged_per_document(report):
    assert report.headline_kinds == ("verbatim",)
    assert report.filtered_kinds == ("filtered", "second")
    # Six questions on four documents. The decision has three (first, second by a tie, first):
    # its means are 2/3 and 5/6; the three other documents are always first.
    exactly(
        report.headline,
        6,
        4,
        per_question=(F(5, 6), 1, F(11, 12), 1),
        per_document=(F(11, 12), 1, F(23, 24), 1),
    )
    assert report.headline.per_document.hit_at_1 != pytest.approx(report.headline.hit_at_1)
    assert report.headline.per_document == report.by_kind["verbatim"].per_document


def test_fixture_per_question_values(report):
    by_id = {result.id: result for result in report.results}
    third, seventh, eighth, tenth = by_id["fx-q03"], by_id["fx-q07"], by_id["fx-q08"], by_id["fx-q10"]
    assert (third.hit_at_1, third.hit_at_k, third.reciprocal_rank, third.recall_at_k) == (
        True,
        True,
        1.0,
        0.5,
    )
    assert (seventh.hit_at_1, seventh.hit_at_k, seventh.reciprocal_rank) == (False, True, 0.5)
    assert seventh.recall_at_k == 1.0
    assert [document.doc_id for document in seventh.top_documents] == [
        "fx-ecb-decision-2016-03",
        "fx-ecb-account-2016-04",
    ]
    assert seventh.top_documents[0].score == pytest.approx(1.0, abs=1e-5)
    assert seventh.top_documents[0].title == "Monetary policy decisions, 10 March 2016"
    assert seventh.top_documents[0].chunks == 3
    assert (seventh.chunks_returned, seventh.documents_ranked) == (6, 2)
    assert (eighth.hit_at_1, eighth.hit_at_k, eighth.reciprocal_rank, eighth.recall_at_k) == (
        False,
        False,
        0.0,
        0.0,
    )
    assert eighth.documents_ranked == 8
    assert tenth.absent_doc_ids == ("fx-us-presser-2016-03",)
    assert (tenth.rank, tenth.recall_at_k) == (1, 1.0)
    assert by_id["fx-q01"].absent_doc_ids == ()
    assert by_id["fx-q01"].documents_ranked == 13  # 50 chunks asked, 32 in the corpus: every document


def test_fixture_ties(report):
    by_id = {result.id: result for result in report.results}
    twelfth, thirteenth = by_id["fx-q12"], by_id["fx-q13"]
    # The same chunk is in the combined document and in the decision: the same score, to the last bit.
    first, second = twelfth.top_documents[:2]
    assert (first.doc_id, second.doc_id) == ("fx-ecb-combined-2016-03", "fx-ecb-decision-2016-03")
    assert first.score == second.score
    assert (twelfth.rank, twelfth.hit_at_1, twelfth.tied_at_decision) == (2, False, True)
    # When both are expected the order of the two changes nothing.
    assert (thirteenth.rank, thirteenth.recall_at_k, thirteenth.tied_at_decision) == (1, 1.0, False)
    assert [result.id for result in report.results if result.tied_at_decision] == ["fx-q12"]
    assert report.tied_at_decision == 1


def test_fixture_breakdown_by_kind(report):
    assert set(report.by_kind) == {"verbatim", "filtered", "second"}
    exactly(
        report.by_kind["verbatim"],
        6,
        4,
        per_question=(F(5, 6), 1, F(11, 12), 1),
        per_document=(F(11, 12), 1, F(23, 24), 1),
    )
    # fx-q03, 04, 05, 06 and 08 (fx-q09 is not scored); fx-q03 expects two documents, one found.
    exactly(
        report.by_kind["filtered"],
        5,
        6,
        per_question=(F(4, 5), F(4, 5), F(4, 5), F(7, 10)),
        per_document=(F(5, 6), F(5, 6), F(5, 6), F(2, 3)),
    )
    exactly(report.by_kind["second"], 1, 1, per_question=(0, 1, F(1, 2), 1), per_document=(0, 1, F(1, 2), 1))


def test_fixture_breakdown_by_tag_counts_every_tag_of_a_question(report):
    assert set(report.by_tag) == {
        "bank:ecb",
        "bank:us",
        "bank:gb",
        "bank:au",
        "bank:ca",
        "bank:jp",
        "doc_type:A1",
        "doc_type:A3",
        "doc_type:B1",
        "doc_type:E1",
        "doc_type:F1",
        "equivalent-documents",
        "tie",
    }
    # First tags: fx-q01, 07, 12 and 13 on the ECB (decision: 1, 0, 1; account: 0; combined: 1).
    exactly(
        report.by_tag["bank:ecb"],
        4,
        3,
        per_question=(F(1, 2), 1, F(3, 4), 1),
        per_document=(F(5, 9), 1, F(7, 9), 1),
    )
    exactly(report.by_tag["bank:us"], 4, 3, per_question=(1, 1, 1, 1), per_document=(1, 1, 1, 1))
    exactly(report.by_tag["bank:jp"], 1, 1, per_question=(0, 0, 0, 0), per_document=(0, 0, 0, 0))
    # Second tags: fx-q02, 03, 07, 08 and 11 are about minutes, whatever their bank.
    exactly(
        report.by_tag["doc_type:A3"],
        5,
        5,
        per_question=(F(3, 5), F(4, 5), F(7, 10), F(7, 10)),
        per_document=(F(3, 5), F(4, 5), F(7, 10), F(3, 5)),
    )
    exactly(report.by_tag["doc_type:B1"], 1, 1, per_question=(1, 1, 1, 1), per_document=(1, 1, 1, 1))
    # Third tags.
    exactly(report.by_tag["tie"], 1, 1, per_question=(0, 1, F(1, 2), 1), per_document=(0, 1, F(1, 2), 1))
    assert report.by_tag["equivalent-documents"].documents == 2
    assert sum(group.questions for name, group in report.by_tag.items() if name.startswith("bank:")) == 12
    assert sum(group.questions for name, group in report.by_tag.items() if name.startswith("doc_type:")) == 12
    assert all(group.intervals is None and not group.few_documents for group in report.by_tag.values())


def test_a_question_whose_documents_are_all_absent_is_reported_and_not_scored(report):
    assert [skipped.id for skipped in report.not_in_collection] == ["fx-q09"]
    assert report.not_in_collection[0].expected_doc_ids == ("fx-ch-statement-2016-06",)
    assert "fx-q09" not in {result.id for result in report.results}
    assert "fx-q09" not in {result.id for result in report.misses}
    assert report.partially_absent == 1


def test_fixture_misses(report):
    assert [result.id for result in report.misses] == ["fx-q08"]
    miss = report.to_dict()["misses"][0]
    assert miss["id"] == "fx-q08"
    assert miss["question"] == "What did the Bank of Japan decide about negative interest rates?"
    assert miss["expected_doc_ids"] == ["fx-jp-minutes"]
    assert len(miss["top_documents"]) == 3
    assert set(miss["top_documents"][0]) == {"doc_id", "title", "score", "chunks"}
    assert all(document["title"] for document in miss["top_documents"])


def test_fixture_documents_ranked_counts(report):
    # Filters leave fewer than ten documents for six questions; the chunk budget is never the cause.
    assert report.fewer_than_k_documents == 6
    assert report.chunk_budget_exhausted == 0


def test_fixture_with_k_1(store, questions):
    report = evaluate(questions, collection=COLLECTION, store=store, embedder=fixture_embedder(), k=1)
    overall = report.overall
    assert (overall.questions, overall.hit_at_1) == (12, pytest.approx(9 / 12))
    assert overall.hit_at_k == pytest.approx(9 / 12)
    assert overall.mrr == pytest.approx(9 / 12)  # a rank beyond k counts for nothing
    # fx-q03 finds one of two; fx-q13 one of two as well, the second document being second.
    assert overall.recall_at_k == pytest.approx(8 / 12)
    assert report.headline.per_document.hit_at_k == pytest.approx(11 / 12)
    seventh = {result.id: result for result in report.results}["fx-q07"]
    assert (seventh.rank, seventh.hit_at_k, seventh.reciprocal_rank, seventh.recall_at_k) == (
        2,
        False,
        0.0,
        0.0,
    )
    assert len(seventh.top_documents) == 2  # the three best when k is smaller, and there are two only
    assert [result.id for result in report.misses] == ["fx-q07", "fx-q08", "fx-q12"]
    assert report.fewer_than_k_documents == 0
    assert report.tied_at_decision == 1  # with k = 1, hit@k is hit@1: the same tie


def test_a_chunk_budget_too_small_for_k_documents_is_reported(store, questions):
    report = evaluate(
        questions, collection=COLLECTION, store=store, embedder=fixture_embedder(), k=10, chunk_limit=10
    )
    # Ten chunks at most, so ten documents only if every chunk came from another document.
    assert report.overall.hit_at_1 == pytest.approx(9 / 12)
    exhausted = {result.id for result in report.results if result.chunk_budget_exhausted}
    # fx-q04, fx-q05, fx-q06 and fx-q07 match fewer than ten chunks: the budget is not the cause.
    assert exhausted == {"fx-q01", "fx-q02", "fx-q03", "fx-q08", "fx-q10", "fx-q11", "fx-q12", "fx-q13"}
    assert report.chunk_budget_exhausted == 8
    assert report.fewer_than_k_documents == 12
    assert "chunk budget" in format_report(report)


def test_parameters_of_the_report(report, questions):
    assert report.parameters == {
        "collection": COLLECTION,
        "k": 10,
        "chunk_limit": 50,
        "max_limit": 50,
        "model": "fake",
        "backend": "fake",
        "query_prefix": "",
        "normalize_query": True,
        "query_normalization": "NFKD",
        "questions_file": str(QUESTIONS),
        "questions_sha256": questions.sha256,
        "bootstrap_resamples": 1000,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interval_level": 0.95,
        "interval_min_documents": 20,
        "engine_version": rag_engine.__version__,
        "score_levels": [0.7, 0.75, 0.8, 0.85, 0.9],
        "interval_min_questions": 20,
    }


def test_report_as_json(report):
    data = json.loads(report.to_json())
    assert data == json.loads(json.dumps(report.to_dict()))
    # The keys of the first version, in their order; the others were added after them.
    assert list(data)[:6] == ["parameters", "summary", "metrics", "misses", "not_in_collection", "results"]
    assert list(data)[6:] == ["unfiltered_reruns", "filter_effect", "abstentions"]
    assert data["summary"] == {
        "questions": 13,
        "scored": 12,
        "not_in_collection": 1,
        "partially_absent": 1,
        "fewer_than_k_documents": 6,
        "chunk_budget_exhausted": 0,
        "tied_at_decision": 1,
        "unfiltered_reruns": 0,
        "abstentions": 0,
        "essential_unreachable": 0,
        "expected_beyond_k": 0,
    }
    assert list(data["metrics"]) == ["headline", "overall", "by_kind", "by_tag", "by_expected_behavior"]
    # The pooled row is kept, and says what it is; the values are exact, not rounded.
    assert data["metrics"]["overall"] == {
        "aggregation": "pooled: every scored question of every kind, counted once",
        "questions": 12,
        "documents": 11,
        "hit_at_1": 9 / 12,
        "hit_at_k": 11 / 12,
        "mrr": 10 / 12,
        "recall_at_k": 10.5 / 12,
        "per_document": {
            "hit_at_1": pytest.approx(float(F(2, 3) + 8) / 11),
            "hit_at_k": pytest.approx(10 / 11),
            "mrr": pytest.approx(float(F(5, 6) + F(1, 2) + 8) / 11),
            "recall_at_k": pytest.approx(9 / 11),
        },
        "intervals": None,
        "few_documents": False,
        "essential": None,
    }
    headline = data["metrics"]["headline"]
    assert headline["aggregation"] == "per document, over the kinds asked without filters"
    assert headline["kinds"] == ["verbatim"]
    assert (headline["questions"], headline["documents"]) == (6, 4)
    assert headline["per_document"]["hit_at_1"] == pytest.approx(11 / 12, abs=1e-12)
    assert headline["hit_at_1"] == pytest.approx(5 / 6, abs=1e-12)
    assert (headline["intervals"], headline["few_documents"]) == (None, True)
    kinds = data["metrics"]["by_kind"]
    assert {name: kind["filtered"] for name, kind in kinds.items()} == {
        "verbatim": False,
        "filtered": True,
        "second": True,
    }
    assert "filtered" not in data["metrics"]["by_tag"]["bank:ecb"]
    assert data["not_in_collection"] == [
        {
            "id": "fx-q09",
            "question": "What did the Swiss National Bank say about the franc?",
            "expected_doc_ids": ["fx-ch-statement-2016-06"],
            "kind": "filtered",
            "tags": ["bank:ch", "doc_type:A2"],
        }
    ]
    third = data["results"][2]
    assert third["id"] == "fx-q03"
    assert third["filters"] == {"doc_types": ["A3"]}
    assert third["rank"] == 1 and third["recall_at_k"] == 0.5
    assert third["tied_at_decision"] is False
    assert third["top_documents"][0]["doc_id"] == "fx-gb-minutes-2016-08"
    assert data["results"][4]["filters"] == {"date_from": "2016-05-01", "date_to": "2016-05-31"}
    assert "filters" not in data["results"][0]
    assert [result["id"] for result in data["results"] if result["tied_at_decision"]] == ["fx-q12"]
    assert report.to_json(indent=None).count("\n") == 0


def table(text: str) -> dict[str, list[str]]:
    """The lines of the tables of a report, by their name, cut into cells."""
    rows = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("  ") if cell.strip()]
        if cells and (
            cells[0].startswith(("kind: ", "tag: ")) or cells[0] in ("headline", "overall (pooled)")
        ):
            rows[cells[0]] = cells[1:]
    return rows


def test_human_readable_report(report):
    text = format_report(report)
    lines = text.splitlines()
    assert "eval-fixture" in lines[0]
    assert "12 questions scored" in text
    assert "hit@1" in text and "hit@10" in text and "MRR" in text and "recall@10" in text
    assert "Not in the collection (1" in text and "fx-q09" in text
    assert "fx-us-presser-2016-03" in text  # the partially absent document
    assert "Misses (1)" in text and "fx-q08" in text and "fx-jp-minutes" in text
    assert report.parameters["questions_sha256"][:12] in text
    assert all(len(line) <= 120 for line in lines)
    assert not text.endswith("\n")


def questions_line(report, path: str) -> str:
    """The line of the report that names the question file, for a report read from ``path``."""
    moved = dataclasses.replace(report, parameters={**report.parameters, "questions_file": path})
    lines = format_report(moved).splitlines()
    assert all(len(line) <= 120 for line in lines)
    (line,) = [line for line in lines if line.startswith("questions: ")]
    return line


def test_the_report_names_a_short_question_file_in_full(report):
    digest = report.parameters["questions_sha256"][:12]
    assert (
        questions_line(report, "eval/questions.jsonl") == f"questions: eval/questions.jsonl (sha256 {digest})"
    )


@pytest.mark.parametrize("depth", [8, 40])
def test_the_report_keeps_the_end_of_a_long_question_file_path(report, depth):
    # Wherever the repository lives: the line fits, and shows the end of the path, the file name.
    path = "/" + "/".join(f"a-rather-long-folder-name-{level:02d}" for level in range(depth))
    path += "/eval/fixtures/questions.jsonl"
    digest = report.parameters["questions_sha256"][:12]
    line = questions_line(report, path)
    assert len(line) == 110
    assert line.startswith("questions: …") and line.endswith(
        f"/eval/fixtures/questions.jsonl (sha256 {digest})"
    )
    shown = line[len("questions: …") : -len(f" (sha256 {digest})")]
    assert path.endswith(shown)


def test_the_report_cuts_the_question_file_path_exactly_at_the_width(report):
    digest = report.parameters["questions_sha256"][:12]
    room = 110 - len(f"questions:  (sha256 {digest})")
    assert questions_line(report, "p" * room) == f"questions: {'p' * room} (sha256 {digest})"
    cut = questions_line(report, "x" + "p" * room)
    assert cut == f"questions: …{'p' * (room - 1)} (sha256 {digest})" and len(cut) == 110


def test_the_report_gives_the_kinds_first_then_the_headline_then_the_pooled_line(report):
    text = format_report(report)
    rows = table(text)
    names = list(rows)
    assert names[:5] == [
        "kind: filtered *",
        "kind: second *",
        "kind: verbatim",
        "headline",
        "overall (pooled)",
    ]
    assert all(name.startswith("tag: ") for name in names[5:])
    # questions, documents, then the per-document means to two decimals.
    assert rows["kind: verbatim"] == ["6", "4", "0.92", "1.00", "0.96", "1.00", "(few documents)"]
    assert rows["headline"] == ["6", "4", "0.92", "1.00", "0.96", "1.00", "(few documents)"]
    assert rows["kind: filtered *"] == ["5", "6", "0.83", "0.83", "0.83", "0.67", "(few documents)"]
    # The pooled line gives the means over the questions.
    assert rows["overall (pooled)"] == ["12", "11", "0.75", "0.92", "0.83", "0.88"]
    assert rows["tag: bank:ecb"] == ["4", "3", "0.56", "1.00", "0.78", "1.00"]
    assert rows["tag: doc_type:A3"] == ["5", "5", "0.60", "0.80", "0.70", "0.60"]
    assert not any(character.isdigit() for name in rows for cell in rows[name][2:] for character in cell[4:5])


def test_the_report_explains_its_lines(report):
    text = format_report(report)
    assert "Per-document means" in text
    assert "headline: the kinds asked without filters (verbatim)" in text
    assert "--fail-under-hit-at-k looks at its hit@10" in text
    assert "*: asked with filters" in text and "Only hit@1 and MRR are informative" in text
    assert "overall (pooled)" in text and "not the headline" in text
    assert "1000 resamples of the documents (seed 1)" in text
    assert "(few documents): fewer than 20 documents" in text
    assert "Equal scores" in text and "for 1 question\n" in text


def test_intervals_are_printed_for_the_kinds_and_the_headline_when_there_are_enough_documents(
    store, questions
):
    report = evaluate(
        questions,
        collection=COLLECTION,
        store=store,
        embedder=fixture_embedder(),
        interval_min_documents=4,
        bootstrap_resamples=200,
    )
    assert report.parameters["interval_min_documents"] == 4
    assert report.parameters["bootstrap_resamples"] == 200
    headline = report.headline
    assert headline.few_documents is False
    assert set(headline.intervals) == {"hit_at_1", "hit_at_k", "mrr"}
    low, high = headline.intervals["hit_at_1"]
    # Four documents, of means 2/3, 1, 1 and 1: a resampled mean lies between 2/3 and 1.
    assert 2 / 3 <= low <= 11 / 12 <= high <= 1.0 and low < high
    assert headline.intervals["hit_at_k"] == (1.0, 1.0)
    assert report.by_kind["second"].few_documents is True and report.by_kind["second"].intervals is None
    rows = table(format_report(report))
    assert rows["headline"] == [
        "6",
        "4",
        f"0.92 [{low:.2f}, {high:.2f}]",
        "1.00 [1.00, 1.00]",
        "0.96 [{:.2f}, {:.2f}]".format(*headline.intervals["mrr"]),
        "1.00",
    ]
    assert rows["kind: second *"][-1] == "(few documents)"
    assert "200 resamples" in format_report(report) and "fewer than 4 documents" in format_report(report)
    data = report.to_dict()["metrics"]["headline"]
    assert data["intervals"]["hit_at_1"] == [low, high]


# ---------------------------------------------------------------------------
# evaluate: arguments and edge cases
# ---------------------------------------------------------------------------
SPEC = ModelSpec(name="tiny", dim=8, passage_prefix="", query_prefix="")


def small_store(documents: dict[str, list[str]], titles: dict[str, str] | None = None):
    """A local store holding ``documents`` (doc_id -> chunk texts) in the collection ``c``."""
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client)
    store.ensure_collection("c", SPEC.dim)
    embedder = FakeEmbedder(SPEC)
    points = []
    for doc_id, texts in documents.items():
        for index, vector in enumerate(embedder.encode_passage(texts)):
            payload = {"doc_id": doc_id, "text": texts[index], "chunk_number": index}
            if titles and doc_id in titles:
                payload["title"] = titles[doc_id]
            points.append(models.PointStruct(id=len(points) + 1, vector=vector.tolist(), payload=payload))
    if points:
        store.upsert("c", points)
    return store


def ask(store, questions, **kwargs):
    return evaluate(questions, collection="c", store=store, embedder=FakeEmbedder(SPEC), **kwargs)


def question(id="q1", text="alpha", expected=("a",), **kwargs) -> Question:
    return Question(id=id, question=text, expected_doc_ids=tuple(expected), **kwargs)


def test_evaluate_takes_plain_questions_without_a_file():
    store = small_store({"a": ["alpha"], "b": ["beta"]})
    report = ask(store, [question()])
    exactly(report.overall, 1, 1, per_question=(1, 1, 1, 1), per_document=(1, 1, 1, 1))
    assert report.parameters["questions_file"] is None
    assert report.parameters["questions_sha256"] is None
    assert report.parameters["model"] == "tiny"
    assert list(report.by_kind) == ["(none)"]
    exactly(report.by_kind["(none)"], 1, 1, per_question=(1, 1, 1, 1), per_document=(1, 1, 1, 1))
    assert report.by_tag == {}
    # Questions without a kind and without filters are the headline; the kind line is not printed.
    assert report.headline_kinds == ("(none)",)
    assert report.headline.per_document == Means(1.0, 1.0, 1.0, 1.0)
    assert "kind:" not in format_report(report) and "headline" in format_report(report)


def test_documents_are_ranked_by_their_best_chunk():
    # "a" has the chunk that equals the question and four others; "b" has one chunk.
    store = small_store({"a": ["alpha", "a1", "a2", "a3", "a4"], "b": ["beta"]})
    report = ask(store, [question(expected=("b",))], k=2)
    result = report.results[0]
    assert [document.doc_id for document in result.top_documents] == ["a", "b"]
    assert [document.chunks for document in result.top_documents] == [5, 1]
    assert (result.rank, result.reciprocal_rank, result.hit_at_k) == (2, 0.5, True)
    scores = [document.score for document in result.top_documents]
    assert scores == sorted(scores, reverse=True) and scores[0] == pytest.approx(1.0, abs=1e-5)


def test_a_long_document_can_use_up_the_chunk_budget():
    # Two chunks are asked for and both belong to "a": nothing says a second document is out of reach
    # or absent, only that the budget was used up before two documents were ranked.
    chunks = [f"a{n}" for n in range(30)]
    store = small_store({"a": ["alpha", *chunks]})
    report = ask(store, [question(text="alpha", expected=("a",))], k=2, chunk_limit=2)
    result = report.results[0]
    assert (result.chunks_returned, result.documents_ranked, result.chunk_budget_exhausted) == (2, 1, True)
    assert (report.fewer_than_k_documents, report.chunk_budget_exhausted) == (1, 1)
    assert result.rank == 1


def test_fewer_documents_than_k_in_the_collection_is_not_a_budget_problem():
    store = small_store({"a": ["alpha"], "b": ["beta"]})
    report = ask(store, [question()], k=10)
    result = report.results[0]
    assert (result.documents_ranked, result.chunk_budget_exhausted) == (2, False)
    assert (report.fewer_than_k_documents, report.chunk_budget_exhausted) == (1, 0)


def test_recall_counts_the_documents_of_the_collection_only():
    store = small_store({"a": ["alpha"], "b": ["beta"], "c": ["gamma"]})
    wanted = question(expected=("a", "b", "ghost"), filters=SearchFilters(doc_ids=["a", "c"]))
    result = ask(store, [wanted]).results[0]
    # "ghost" is not in the collection; "b" is, and the filter keeps it out: one of two.
    assert result.absent_doc_ids == ("ghost",)
    assert result.recall_at_k == 0.5


def test_the_rank_is_the_one_of_the_best_expected_document():
    store = small_store({"a": ["alpha"], "b": ["beta"], "c": ["gamma"]})
    result = ask(store, [question(text="beta", expected=("a", "b"))]).results[0]
    assert (result.rank, result.hit_at_1, result.recall_at_k) == (1, True, 1.0)


def test_chunks_without_a_doc_id_are_not_documents():
    store = small_store({"a": ["alpha"]})
    vector = FakeEmbedder(SPEC).encode_passage(["orphan"])[0].tolist()
    store.upsert("c", [models.PointStruct(id=99, vector=vector, payload={"text": "orphan"})])
    result = ask(store, [question(text="orphan")]).results[0]
    assert (result.chunks_returned, result.documents_ranked, result.rank) == (2, 1, 1)


def test_an_empty_collection_scores_nothing():
    store = small_store({})
    report = ask(store, [question()])
    assert [skipped.id for skipped in report.not_in_collection] == ["q1"]
    assert report.overall == Aggregate(0, 0, None, None, None, None)
    assert report.headline == Aggregate(0, 0, None, None, None, None)
    assert (report.headline_kinds, report.filtered_kinds) == ((), ())
    assert report.results == () and report.misses == ()
    text = format_report(report)
    assert "0 questions scored" in text and "n/a" in text
    metrics = json.loads(report.to_json())["metrics"]
    assert metrics["overall"]["hit_at_k"] is None and metrics["overall"]["per_document"] is None
    assert metrics["headline"]["per_document"] is None and metrics["headline"]["kinds"] == []


def test_the_existence_of_the_expected_documents_is_asked_once_for_all_questions():
    store = small_store({"a": ["alpha"], "b": ["beta"]})
    calls = []
    original = store.existing_doc_ids

    def recording(collection, doc_ids, **kwargs):
        calls.append((collection, sorted(doc_ids)))
        return original(collection, doc_ids, **kwargs)

    store.existing_doc_ids = recording
    ask(store, [question(), question(id="q2", expected=("b", "ghost")), question(id="q3", expected=("a",))])
    assert calls == [("c", ["a", "b", "ghost"])]


def test_the_query_is_embedded_once_per_scored_question_and_normalised():
    store = small_store({"a": ["alpha"]})
    embedder = FakeEmbedder(SPEC)
    questions = [question(text="caf\u00e9"), question(id="q2", expected=("ghost",))]
    evaluate(questions, collection="c", store=store, embedder=embedder)
    assert embedder.calls == ["cafe\u0301"]
    embedder = FakeEmbedder(SPEC)
    report = evaluate(questions, collection="c", store=store, embedder=embedder, normalize_query=False)
    assert embedder.calls == ["caf\u00e9"]
    assert report.parameters["normalize_query"] is False
    assert report.parameters["query_normalization"] is None


def test_max_limit_bounds_the_chunk_budget():
    store = small_store({"a": ["alpha"]})
    with pytest.raises(EvalError, match="chunk_limit 51 is above the maximum of 50"):
        ask(store, [question()], chunk_limit=51)
    with pytest.raises(EvalError, match="chunk_limit 50 is above the maximum of 20"):
        ask(store, [question()], max_limit=20)
    report = ask(store, [question()], chunk_limit=200, max_limit=200)
    assert (report.parameters["chunk_limit"], report.parameters["max_limit"]) == (200, 200)


def test_k_may_not_exceed_the_chunk_budget():
    store = small_store({"a": ["alpha"]})
    with pytest.raises(EvalError, match="k=11.*chunk_limit=10"):
        ask(store, [question()], k=11, chunk_limit=10)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": 0}, "k must be"),
        ({"k": True}, "k must be"),
        ({"k": 2.0}, "k must be"),
        ({"chunk_limit": 0}, "chunk_limit must be"),
        ({"chunk_limit": "50"}, "chunk_limit must be"),
        ({"max_limit": 0}, "max_limit must be"),
        ({"bootstrap_resamples": 0}, "bootstrap_resamples must be"),
        ({"bootstrap_resamples": 10.0}, "bootstrap_resamples must be"),
        ({"interval_min_documents": 0}, "interval_min_documents must be"),
        ({"normalize_query": 1}, "normalize_query"),
    ],
)
def test_bad_arguments_are_refused_before_anything_is_asked(kwargs, match):
    store = small_store({"a": ["alpha"]})
    embedder = FakeEmbedder(SPEC)
    with pytest.raises(EvalError, match=match):
        evaluate([question()], collection="c", store=store, embedder=embedder, **kwargs)
    assert embedder.calls == []


def test_bad_questions_are_refused_before_anything_is_asked():
    store = small_store({"a": ["alpha"]})
    with pytest.raises(EvalError, match="no question"):
        ask(store, [])
    with pytest.raises(EvalError, match="duplicate id 'q1'"):
        ask(store, [question(), question()])
    with pytest.raises(EvalError, match="Question"):
        ask(store, [{"id": "q1"}])
    with pytest.raises(EvalError, match="collection"):
        evaluate([question()], collection=" ", store=store, embedder=FakeEmbedder(SPEC))


def test_a_question_that_normalisation_empties_is_refused_before_anything_is_asked():
    # An isolated acute accent is a symbol: a Question. In NFKD it is a space and a combining mark.
    store = small_store({"a": ["alpha"]})
    accent = Question(id="q2", question="\u00b4", expected_doc_ids=("a",))
    embedder = FakeEmbedder(SPEC)
    with pytest.raises(EvalError, match="question 'q2'.*no searchable character once put in NFKD"):
        evaluate([question(), accent], collection="c", store=store, embedder=embedder)
    assert embedder.calls == []  # not after the first question was searched
    # Without normalisation it is a query like another.
    report = evaluate(
        [question(), accent], collection="c", store=store, embedder=FakeEmbedder(SPEC), normalize_query=False
    )
    assert len(report.results) == 2


def test_the_question_file_refuses_what_the_default_search_would_refuse(tmp_path):
    path = write(tmp_path, q(), q(id="q2"), q(id="q3", question="\u00b4"))
    with pytest.raises(EvalError, match="line 3.*NFKD"):
        load_questions(path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"id": ""}, "id"),
        ({"question": "  "}, "question"),
        ({"expected_doc_ids": ()}, "expected_doc_ids"),
        ({"expected_doc_ids": "abc"}, "expected_doc_ids"),
        ({"expected_doc_ids": ("a", "a")}, "twice"),
        ({"filters": {"source_codes": ["ecb"]}}, "SearchFilters"),
        ({"tags": "abc"}, "tags"),
        ({"kind": ""}, "kind"),
    ],
)
def test_a_question_is_checked_when_it_is_built(kwargs, match):
    values = {"id": "q1", "question": "alpha", "expected_doc_ids": ("a",)}
    values.update(kwargs)
    with pytest.raises(EvalError, match=match):
        Question(**values)


def test_a_question_keeps_lists_as_tuples():
    built = Question(id="q1", question="alpha", expected_doc_ids=["a", "b"], tags=["x"])
    assert (built.expected_doc_ids, built.tags) == (("a", "b"), ("x",))


def test_a_missing_collection_is_raised_as_the_client_raised_it():
    store = small_store({"a": ["alpha"]})
    with pytest.raises(ValueError, match="nowhere"):
        evaluate([question()], collection="nowhere", store=store, embedder=FakeEmbedder(SPEC))


def test_the_backend_name_comes_from_the_embedder():
    class CustomBackend(FakeEmbedder):
        pass

    store = small_store({"a": ["alpha"]})
    report = evaluate([question()], collection="c", store=store, embedder=CustomBackend(SPEC))
    assert report.parameters["backend"] == "CustomBackend"


def test_misses_show_three_documents_and_the_report_shows_untitled_ones():
    documents = {name: [name] for name in ("a", "b", "c", "d", "e")}
    store = small_store(documents, titles={"a": "The first"})
    report = ask(store, [question(text="a", expected=("e",))], k=1)
    miss = report.to_dict()["misses"][0]
    assert len(miss["top_documents"]) == 3
    assert miss["top_documents"][0] == {
        "doc_id": "a",
        "title": "The first",
        "score": pytest.approx(1.0, abs=1e-5),
        "chunks": 1,
    }
    assert miss["top_documents"][1]["title"] is None
    assert miss["rank"] in (2, 3, 4, 5)
    text = format_report(report)
    assert "The first" in text and "(no title)" in text


# ---------------------------------------------------------------------------
# Documents, not chunks; equal scores
# ---------------------------------------------------------------------------
def test_hit_at_k_counts_documents_not_chunks():
    # Four chunks of "a" equal the question, so they are the four best chunks, whatever the rest;
    # the chunk of "b" is the fifth chunk at best, and "b" is the second document.
    store = small_store({"a": ["alpha"] * 4, "b": ["beta"]})
    result = ask(store, [question(expected=("b",))], k=2).results[0]
    assert [document.doc_id for document in result.top_documents] == ["a", "b"]
    assert result.top_documents[0].chunks == 4
    assert (result.chunks_returned, result.documents_ranked) == (5, 2)
    assert (result.rank, result.hit_at_k, result.reciprocal_rank, result.recall_at_k) == (2, True, 0.5, 1.0)
    assert result.hit_at_1 is False


def hit(doc_id, score, chunk_number=0, title=None) -> Hit:
    return Hit(
        text="t",
        score=score,
        doc_id=doc_id,
        title=title,
        date=None,
        source_code=None,
        doc_type=None,
        language=None,
        page=None,
        chunk_number=chunk_number,
        corpus=None,
        point_id=f"{doc_id}-{chunk_number}",
        payload={},
    )


def returning(monkeypatch, hits):
    """Make the search of the harness return ``hits`` as they are, whatever the question."""
    monkeypatch.setattr("rag_engine.eval.search", lambda *args, **kwargs: list(hits))
    return small_store({name: ["x"] for name in {h.doc_id for h in hits if isinstance(h.doc_id, str)}})


def test_equal_scores_are_put_in_one_order_whatever_the_order_they_come_in(monkeypatch):
    hits = [
        hit("c", 0.5, 1, title="second chunk of c"),
        hit("b", 0.9),
        hit("d", 0.9),
        hit("c", 0.5, 0, title="first chunk of c"),
        hit("a", 0.9),
        hit(None, 0.9),
        hit("e", 0.95),
    ]
    rankings = []
    for order in (hits, hits[::-1], sorted(hits, key=lambda h: str(h.doc_id), reverse=True)):
        store = returning(monkeypatch, order)
        result = ask(store, [question(expected=("d",))], k=5).results[0]
        rankings.append([(document.doc_id, document.score) for document in result.top_documents])
        # The best chunk of a document is the one of the lowest number among equal scores.
        assert result.top_documents[-1].title == "first chunk of c"
        assert result.rank == 4
    assert rankings[0] == [("e", 0.95), ("a", 0.9), ("b", 0.9), ("d", 0.9), ("c", 0.5)]
    assert rankings[1] == rankings[0] and rankings[2] == rankings[0]


@pytest.mark.parametrize(
    ("scores", "expected", "k", "tied"),
    [
        # Two documents with the same score: which is first is the order of the harness.
        ({"a": 0.9, "b": 0.9}, ("b",), 10, True),
        ({"a": 0.9, "b": 0.9}, ("a",), 10, True),
        ({"a": 0.9, "b": 0.9}, ("a", "b"), 10, False),
        ({"a": 0.9, "b": 0.8}, ("b",), 10, False),
        # Four tied documents and k = 2: any expected one may be in or out.
        ({"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9}, ("c",), 2, True),
        ({"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9}, ("a",), 2, True),
        # Three of the four are expected: one of them is among the first two whatever the order,
        # but the first may be the one that is not expected.
        ({"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9}, ("b", "c", "d"), 2, True),
        ({"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9}, ("a", "b", "c", "d"), 2, False),
        # The tie is across rank k, and an expected document is above it: nothing hangs on it.
        ({"a": 1.0, "b": 0.5, "c": 0.5}, ("a", "b"), 2, False),
        ({"a": 1.0, "b": 0.5, "c": 0.5}, ("b",), 2, True),
        ({"a": 1.0, "b": 0.5, "c": 0.5}, ("c",), 2, True),
        # The tie is inside the first k, or beyond them: it decides nothing.
        ({"a": 1.0, "b": 0.5, "c": 0.5}, ("c",), 3, False),
        ({"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.4}, ("c",), 3, False),
        ({"a": 1.0, "b": 0.8, "c": 0.5, "d": 0.5}, ("d",), 2, False),
        ({"a": 1.0, "b": 0.8, "c": 0.5, "d": 0.5}, ("b",), 2, False),
        # Two expected documents in a tie of three across k = 2, one place left: one of them may miss it,
        # the other cannot both miss it.
        ({"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.5}, ("b", "c"), 2, True),
        ({"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.5}, ("b", "c"), 3, False),
    ],
)
def test_a_tie_that_decides_hit_at_1_or_hit_at_k_is_detected(monkeypatch, scores, expected, k, tied):
    store = returning(monkeypatch, [hit(doc_id, score) for doc_id, score in scores.items()])
    result = ask(store, [question(expected=expected)], k=k).results[0]
    assert result.tied_at_decision is tied


def test_ties_are_counted_in_the_report():
    store = small_store({"a": ["alpha"], "b": ["alpha"], "c": ["gamma"]})
    report = ask(store, [question(expected=("b",)), question(id="q2", text="gamma", expected=("c",))])
    first, second = report.results
    assert [document.doc_id for document in first.top_documents[:2]] == ["a", "b"]
    assert (first.rank, first.tied_at_decision, second.tied_at_decision) == (2, True, False)
    assert report.tied_at_decision == 1
    assert report.to_dict()["summary"]["tied_at_decision"] == 1
    assert "Equal scores" in format_report(report)
    untied = ask(store, [question(id="q2", text="gamma", expected=("c",))])
    assert untied.tied_at_decision == 0 and "Equal scores" not in format_report(untied)


# ---------------------------------------------------------------------------
# Aggregates: per question, per document, intervals
# ---------------------------------------------------------------------------
def scored(id, expected, rank, *, kind=None, tags=(), filters=None, absent=(), k=10) -> QuestionResult:
    within = rank is not None and rank <= k
    return QuestionResult(
        id=id,
        question="q",
        kind=kind,
        tags=tuple(tags),
        expected_doc_ids=tuple(expected),
        absent_doc_ids=tuple(absent),
        filters=filters,
        rank=rank,
        hit_at_1=rank == 1,
        hit_at_k=within,
        reciprocal_rank=1.0 / rank if within else 0.0,
        recall_at_k=1.0 if within else 0.0,
        chunks_returned=10,
        documents_ranked=10,
        chunk_budget_exhausted=False,
        top_documents=(RankedDocument("x", None, 1.0, 1),),
    )


def test_the_questions_of_one_document_are_averaged_before_the_documents():
    # "a" is asked for three times and always found first; "b" once, and missed.
    results = [scored(f"a{n}", ["a"], 1) for n in range(3)] + [scored("b", ["b"], None)]
    group = Aggregate.of(results)
    exactly(group, 4, 2, per_question=(F(3, 4), F(3, 4), F(3, 4), F(3, 4)), per_document=(F(1, 2),) * 4)
    assert group.to_dict()["per_document"] == {
        "hit_at_1": 0.5,
        "hit_at_k": 0.5,
        "mrr": 0.5,
        "recall_at_k": 0.5,
    }


def test_a_question_that_expects_several_documents_counts_for_each_present_one():
    results = [
        scored("both", ["a", "b"], 1),
        scored("a-only", ["a"], None),
        scored("ghost", ["b", "ghost"], 2, absent=["ghost"]),
    ]
    group = Aggregate.of(results)
    # a: (1 + 0) / 2; b: (1 + 0) / 2 for hit@1, and (1 + 1/2) / 2 for the reciprocal rank.
    assert group.documents == 2  # "ghost" is not in the collection: no document stands behind it
    exactly(
        group,
        3,
        2,
        per_question=(F(1, 3), F(2, 3), F(1, 2), F(2, 3)),
        per_document=(F(1, 2), F(3, 4), F(5, 8), F(3, 4)),
    )


def test_an_empty_group_has_no_means():
    assert Aggregate.of([]) == Aggregate(0, 0, None, None, None, None, None, None, False)
    assert Aggregate.of([], interval="x").few_documents is False


def test_intervals_come_from_resampling_documents_not_questions():
    # Every document is found once and missed once: each has a mean of 1/2, so whatever documents are
    # drawn the mean is 1/2. Resampling the questions instead would spread it.
    results = []
    for n in range(30):
        results += [scored(f"hit-{n}", [f"d{n}"], 1), scored(f"miss-{n}", [f"d{n}"], None)]
    group = Aggregate.of(results, interval="x")
    assert group.documents == 30 and group.few_documents is False
    assert group.intervals == {"hit_at_1": (0.5, 0.5), "hit_at_k": (0.5, 0.5), "mrr": (0.5, 0.5)}


def test_intervals_are_wider_when_one_document_carries_many_questions():
    # Forty questions, half of them hits. On forty documents the mean of a resample moves little; on
    # twenty documents that are each all hits or all misses it moves more, and the questions are the same.
    spread = [scored(f"q{n}", [f"d{n}"], 1 if n % 2 else None) for n in range(40)]
    clustered = [scored(f"q{n}", [f"d{n // 2}"], 1 if (n // 2) % 2 else None) for n in range(40)]
    one, other = Aggregate.of(spread, interval="x"), Aggregate.of(clustered, interval="x")
    assert one.hit_at_1 == other.hit_at_1 == 0.5
    assert (one.documents, other.documents) == (40, 20)

    def width(group):
        return group.intervals["hit_at_1"][1] - group.intervals["hit_at_1"][0]

    assert 0.2 < width(one) < width(other) < 0.6
    for group in (one, other):
        low, high = group.intervals["hit_at_1"]
        assert 0.0 < low < 0.5 < high < 1.0


def test_intervals_are_the_same_for_the_same_results_in_any_order():
    results = [scored(f"q{n}", [f"d{n % 25}"], 1 if n % 3 else 4) for n in range(60)]
    first = Aggregate.of(results, interval="headline")
    again = Aggregate.of(results[::-1], interval="headline")
    assert first.intervals == again.intervals and first.per_document == again.per_document
    assert Aggregate.of(results, interval="kind:other").intervals != first.intervals  # another draw
    assert Aggregate.of(results).intervals is None
    low, high = first.intervals["mrr"]
    assert low < first.per_document.mrr < high
    # The number of resamples is a parameter; with one draw the interval is a point.
    single = Aggregate.of(results, interval="headline", resamples=1)
    assert single.intervals["hit_at_1"][0] == single.intervals["hit_at_1"][1]


def test_a_group_with_few_documents_gets_no_interval():
    results = [scored(f"q{n}", [f"d{n}"], 1) for n in range(19)]
    group = Aggregate.of(results, interval="x")
    assert (group.documents, group.intervals, group.few_documents) == (19, None, True)
    enough = Aggregate.of([*results, scored("q19", ["d19"], 1)], interval="x")
    assert (enough.documents, enough.few_documents) == (20, False)
    assert enough.intervals["hit_at_1"] == (1.0, 1.0)
    assert Aggregate.of(results, interval="x", min_documents=19).few_documents is False


def report_of(results, **parameters) -> EvalReport:
    base = {
        "collection": "c",
        "k": 10,
        "chunk_limit": 50,
        "model": "tiny",
        "backend": "fake",
        "query_normalization": "NFKD",
        "engine_version": rag_engine.__version__,
    }
    return EvalReport(parameters={**base, **parameters}, results=tuple(results))


def test_the_headline_leaves_out_every_kind_asked_with_filters():
    filters = SearchFilters(doc_types=["A3"])
    results = [scored(f"t{n}", [f"d{n}"], 1, kind="title") for n in range(4)]
    results += [scored(f"d{n}", [f"d{n}"], None, kind="described") for n in range(2)]
    results += [scored(f"f{n}", [f"d{n}"], 1, kind="described+filters", filters=filters) for n in range(4)]
    # One question with filters makes its kind a filtered kind, with all its questions.
    results += [
        scored("m0", ["d0"], None, kind="mixed"),
        scored("m1", ["d1"], None, kind="mixed", filters=filters),
    ]
    report = report_of(results)
    assert report.headline_kinds == ("described", "title")
    assert report.filtered_kinds == ("described+filters", "mixed")
    # d0 and d1: one title found, one description missed; d2 and d3: the title found.
    exactly(report.headline, 6, 4, per_question=(F(2, 3),) * 4, per_document=(F(3, 4),) * 4)
    # The pooled line counts everything.
    assert (report.overall.questions, report.overall.hit_at_1) == (12, pytest.approx(8 / 12))
    text = format_report(report)
    assert "headline: the kinds asked without filters (described, title)" in text
    assert "kind: described+filters *" in text and "kind: mixed *" in text
    assert "kind: described *" not in text and "kind: title *" not in text


def test_the_report_reads_the_resampling_parameters():
    results = [scored(f"q{n}", [f"d{n}"], 1 if n % 2 else None, kind="title") for n in range(30)]
    default = report_of(results)
    assert default.headline.intervals is not None and default.by_kind["title"].intervals is not None
    assert default.overall.intervals is None
    assert report_of(results, interval_min_documents=31).headline.few_documents is True
    few = report_of(results, bootstrap_resamples=1)
    assert few.headline.intervals["hit_at_1"][0] == few.headline.intervals["hit_at_1"][1]
    assert " [0." in format_report(default) and " [0." not in format_report(report_of(results[:5]))


def test_the_report_shown_in_the_documentation_is_the_one_of_the_fixture(store, questions):
    from pathlib import Path

    report = evaluate(
        questions, collection=COLLECTION, store=store, embedder=fixture_embedder(), interval_min_documents=4
    )
    shown = format_report(report).split("\n\n")[1]
    assert shown.startswith(" ") and "headline" in shown and "[0.75, 1.00]" in shown
    root = Path(__file__).resolve().parent.parent
    for name in ("README.md", "eval/README.md"):
        assert shown in (root / name).read_text(encoding="utf-8"), name
