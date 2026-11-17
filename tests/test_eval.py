"""The evaluation harness: the question file, document-level scoring, the report.

Run against qdrant-client's local mode with a ``FakeEmbedder``. The fixture of
``eval/fixtures/`` is built so that every ranking the metrics depend on is certain without
any semantics: a question is the text of a chunk, or its filters isolate the answer.
"""

from __future__ import annotations

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
    Aggregate,
    EvalError,
    EvalReport,
    Question,
    QuestionSet,
    evaluate,
    format_report,
    load_questions,
)
from rag_engine.search import SearchFilters
from rag_engine.store import QdrantStore

COLLECTION = "eval-fixture"


@pytest.fixture(scope="module")
def store():
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client, settings=Settings(batch_size=16))
    assert load_fixture(store, COLLECTION) == 30
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
    assert len(corpus["documents"]) == 12
    assert sum(len(document["chunks"]) for document in corpus["documents"]) == 30
    assert len(fixture_points()) == 30
    # A question equal to a chunk gets the vector of the chunk only if both prefixes are equal.
    assert corpus["model"]["passage_prefix"] == corpus["model"]["query_prefix"]


# ---------------------------------------------------------------------------
# load_questions
# ---------------------------------------------------------------------------
def test_load_the_fixture_questions(questions):
    assert isinstance(questions, QuestionSet)
    assert len(questions) == 10
    assert [question.id for question in questions] == [f"fx-q{n:02d}" for n in range(1, 11)]
    assert questions.sha256 == hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()
    assert questions.path == str(QUESTIONS)
    first, third, sixth = questions[0], questions[2], questions[5]
    assert isinstance(first, Question)
    assert first.expected_doc_ids == ("fx-ecb-decision-2016-03",)
    assert (first.filters, first.kind, first.tags, first.line) == (None, "verbatim", ("bank:ecb",), 1)
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
}


def test_fixture_ranks(report):
    assert isinstance(report, EvalReport)
    assert {result.id: result.rank for result in report.results} == EXPECTED_RANKS
    assert [result.id for result in report.results] == list(EXPECTED_RANKS)  # the order of the file


def test_fixture_metrics_are_exact(report):
    overall = report.overall
    assert isinstance(overall, Aggregate)
    assert overall.questions == 9
    assert overall.hit_at_1 == pytest.approx(7 / 9, abs=1e-12)
    assert overall.hit_at_k == pytest.approx(8 / 9, abs=1e-12)
    assert overall.mrr == pytest.approx(7.5 / 9, abs=1e-12)
    assert overall.recall_at_k == pytest.approx(7.5 / 9, abs=1e-12)


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
    assert eighth.documents_ranked == 7
    assert tenth.absent_doc_ids == ("fx-us-presser-2016-03",)
    assert (tenth.rank, tenth.recall_at_k) == (1, 1.0)
    assert by_id["fx-q01"].absent_doc_ids == ()
    assert by_id["fx-q01"].documents_ranked == 12  # 50 chunks asked, 30 in the corpus: every document


def test_fixture_breakdowns(report):
    assert set(report.by_kind) == {"verbatim", "filtered", "second"}
    assert report.by_kind["verbatim"] == Aggregate(4, 1.0, 1.0, 1.0, 3.5 / 4)
    assert report.by_kind["filtered"].questions == 4  # fx-q09 is not scored
    assert report.by_kind["filtered"].hit_at_k == 0.75
    assert report.by_kind["second"] == Aggregate(1, 0.0, 1.0, 0.5, 1.0)
    assert set(report.by_tag) == {"bank:ecb", "bank:us", "bank:gb", "bank:au", "bank:ca", "bank:jp"}
    assert report.by_tag["bank:ecb"] == Aggregate(2, 0.5, 1.0, 0.75, 1.0)
    assert report.by_tag["bank:us"] == Aggregate(3, 1.0, 1.0, 1.0, 1.0)
    assert report.by_tag["bank:jp"] == Aggregate(1, 0.0, 0.0, 0.0, 0.0)


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
    assert (overall.questions, overall.hit_at_1) == (9, pytest.approx(7 / 9))
    assert overall.hit_at_k == pytest.approx(7 / 9)
    assert overall.mrr == pytest.approx(7 / 9)  # a rank beyond k counts for nothing
    assert overall.recall_at_k == pytest.approx(6.5 / 9)
    seventh = {result.id: result for result in report.results}["fx-q07"]
    assert (seventh.rank, seventh.hit_at_k, seventh.reciprocal_rank, seventh.recall_at_k) == (
        2,
        False,
        0.0,
        0.0,
    )
    assert len(seventh.top_documents) == 2  # the three best when k is smaller, and there are two only
    assert [result.id for result in report.misses] == ["fx-q07", "fx-q08"]
    assert report.fewer_than_k_documents == 0


def test_a_chunk_budget_too_small_for_k_documents_is_reported(store, questions):
    report = evaluate(
        questions, collection=COLLECTION, store=store, embedder=fixture_embedder(), k=10, chunk_limit=10
    )
    # Ten chunks at most, so ten documents only if every chunk came from another document.
    assert report.overall.hit_at_1 == pytest.approx(7 / 9)
    exhausted = {result.id for result in report.results if result.chunk_budget_exhausted}
    # fx-q04, fx-q05, fx-q06 and fx-q07 match fewer than ten chunks: the budget is not the cause.
    assert exhausted == {"fx-q01", "fx-q02", "fx-q03", "fx-q08", "fx-q10"}
    assert report.chunk_budget_exhausted == 5
    assert report.fewer_than_k_documents == 9
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
        "engine_version": rag_engine.__version__,
    }


def test_report_as_json(report):
    data = json.loads(report.to_json())
    assert data == json.loads(json.dumps(report.to_dict()))
    assert list(data) == ["parameters", "summary", "metrics", "misses", "not_in_collection", "results"]
    assert data["summary"] == {
        "questions": 10,
        "scored": 9,
        "not_in_collection": 1,
        "partially_absent": 1,
        "fewer_than_k_documents": 6,
        "chunk_budget_exhausted": 0,
    }
    assert list(data["metrics"]) == ["overall", "by_kind", "by_tag"]
    assert data["metrics"]["overall"] == {
        "questions": 9,
        "hit_at_1": 7 / 9,
        "hit_at_k": 8 / 9,
        "mrr": float(Fraction(15, 18)),
        "recall_at_k": float(Fraction(15, 18)),
    }
    assert data["not_in_collection"] == [
        {
            "id": "fx-q09",
            "question": "What did the Swiss National Bank say about the franc?",
            "expected_doc_ids": ["fx-ch-statement-2016-06"],
            "kind": "filtered",
            "tags": ["bank:ch"],
        }
    ]
    third = data["results"][2]
    assert third["id"] == "fx-q03"
    assert third["filters"] == {"doc_types": ["A3"]}
    assert third["rank"] == 1 and third["recall_at_k"] == 0.5
    assert third["top_documents"][0]["doc_id"] == "fx-gb-minutes-2016-08"
    assert data["results"][4]["filters"] == {"date_from": "2016-05-01", "date_to": "2016-05-31"}
    assert "filters" not in data["results"][0]
    assert report.to_json(indent=None).count("\n") == 0


def test_human_readable_report(report):
    text = format_report(report)
    lines = text.splitlines()
    assert "eval-fixture" in lines[0]
    assert "9 questions scored" in text
    assert "hit@1" in text and "hit@10" in text and "MRR" in text and "recall@10" in text
    assert "overall" in text and "0.778" in text and "0.889" in text and "0.833" in text
    assert "kind: verbatim" in text and "tag: bank:ecb" in text
    assert "Not in the collection (1" in text and "fx-q09" in text
    assert "fx-us-presser-2016-03" in text  # the partially absent document
    assert "Misses (1)" in text and "fx-q08" in text and "fx-jp-minutes" in text
    assert report.parameters["questions_sha256"][:12] in text
    assert all(len(line) <= 160 for line in lines)
    assert not text.endswith("\n")


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
    assert report.overall == Aggregate(1, 1.0, 1.0, 1.0, 1.0)
    assert report.parameters["questions_file"] is None
    assert report.parameters["questions_sha256"] is None
    assert report.parameters["model"] == "tiny"
    assert report.by_kind == {"(none)": Aggregate(1, 1.0, 1.0, 1.0, 1.0)}
    assert report.by_tag == {}


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
    assert report.overall == Aggregate(0, None, None, None, None)
    assert report.results == () and report.misses == ()
    text = format_report(report)
    assert "0 questions scored" in text and "n/a" in text
    assert json.loads(report.to_json())["metrics"]["overall"]["hit_at_k"] is None


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
