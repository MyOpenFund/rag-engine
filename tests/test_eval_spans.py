"""Passage-level scoring: ``expected_spans``, :func:`span_hit`, :func:`evaluate_passages`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rag_engine.eval import (
    EvalError,
    PassageReport,
    PassageResult,
    Question,
    QuestionSet,
    evaluate_passages,
    load_questions,
    span_hit,
)
from rag_engine.search import Hit

EVAL = Path(__file__).resolve().parent.parent / "eval"


def _hit(text, doc_id="d"):
    return Hit(
        text=text, score=1.0, doc_id=doc_id, title=None, date=None, source_code=None, doc_type=None,
        language=None, page=None, chunk_number=0, corpus=None, point_id=1, payload={},
    )  # fmt: skip


def _question(qid, spans, **more):
    return Question(id=qid, question=f"question {qid}", expected_doc_ids=("d",), expected_spans=spans, **more)


def test_existing_files_still_load():
    known = load_questions(EVAL / "known_item_v1.jsonl")
    analytical = load_questions(EVAL / "analytical_v2.jsonl")
    assert (len(known), len(analytical)) == (797, 91)
    assert all(q.expected_spans == () for q in (*known, *analytical))


def test_span_hit_normalises_whitespace_and_case():
    assert span_hit("Net  revenue\n rose\t12% in\nfiscal 2024.", ["NET REVENUE rose 12%"])
    assert span_hit("alpha beta", ["gamma", "beta"])  # any span will do
    assert not span_hit("alpha beta", ["gamma"])
    assert not span_hit("alpha beta", [])
    assert not span_hit(None, ["alpha"])
    assert not span_hit("alpha", ["  "])  # a blank span matches nothing


def test_expected_spans_are_read_checked_and_written_back(tmp_path):
    path = tmp_path / "q.jsonl"
    line = {"id": "a", "question": "q?", "expected_doc_ids": ["d"], "expected_spans": ["one", "two"]}
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    question = load_questions(path)[0]
    assert question.expected_spans == ("one", "two")
    assert question.to_dict()["expected_spans"] == ["one", "two"]
    for bad in ("one", ["one", "one"], [""], [3]):
        line["expected_spans"] = bad
        path.write_text(json.dumps(line) + "\n", encoding="utf-8")
        with pytest.raises(EvalError):
            load_questions(path)


def test_an_abstention_carries_no_span():
    with pytest.raises(EvalError, match="abstain"):
        Question(
            id="a", question="q?", expected_doc_ids=(), expected_behavior="abstain", expected_spans=("x",)
        )


def test_evaluate_passages_hit_at_k():
    runs = {
        "q1": [_hit("nothing"), _hit("the Answer is here")],  # rank 2
        "q2": [_hit("the answer"), _hit("x")],  # rank 1
        "q3": [_hit("a"), _hit("b"), _hit("c"), _hit("d"), _hit("e"), _hit("the answer")],  # rank 6
        "q4": [_hit("a"), _hit("b")],  # miss
    }
    questions = QuestionSet(tuple(_question(q, ("the answer",)) for q in runs))
    report = evaluate_passages(
        questions, lambda q: runs[q.id], k=10, interval_min_questions=4, bootstrap_resamples=200
    )
    assert isinstance(report, PassageReport)
    assert [r.rank for r in report.results] == [2, 1, 6, None]
    assert report.hit_at(5) == 0.5
    assert report.hit_at(10) == 0.75
    assert report.mrr_at(10) == pytest.approx((1 / 2 + 1 + 1 / 6) / 4)
    assert report.mrr_at(5) == pytest.approx((1 / 2 + 1) / 4)
    low, high = report.interval("hit_at_10")
    assert 0 <= low <= 0.75 <= high <= 1
    assert report.to_dict()["hit_at_10"]["value"] == 0.75
    assert "hit@10" in report.format()


def test_hits_beyond_k_are_ignored_and_few_questions_get_no_interval():
    questions = QuestionSet((_question("q", ("x",)),))
    report = evaluate_passages(questions, lambda q: [_hit("a")] * 3 + [_hit("x")], k=3)
    assert report.results[0].rank is None
    assert report.interval("hit_at_3") is None  # one question: under interval_min_questions


def test_abstentions_are_reported_apart():
    questions = QuestionSet(
        (
            _question("ans", ("x",)),
            Question(id="abs", question="unanswerable?", expected_doc_ids=(), expected_behavior="abstain"),
        )
    )
    report = evaluate_passages(questions, lambda q: [_hit("x")], k=5)
    assert len(report.results) == 1 and report.hit_at(5) == 1.0
    assert [a.id for a in report.abstentions] == ["abs"]
    assert report.abstentions[0].returned == 1


def test_a_question_without_span_is_refused():
    questions = QuestionSet((Question(id="n", question="q?", expected_doc_ids=("d",)),))
    with pytest.raises(EvalError, match="expected_spans"):
        evaluate_passages(questions, lambda q: [], k=5)


def test_k_must_be_a_positive_integer():
    with pytest.raises(EvalError):
        evaluate_passages(QuestionSet((_question("q", ("x",)),)), lambda q: [], k=0)


# --- the filings eval script ----------------------------------------------------------------------


def _script():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "filings_eval.py"
    spec = importlib.util.spec_from_file_location("filings_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def _synthetic_corpus(root):
    topics = {
        "acme-2024": (
            "Acme Corp 10-K 2024",
            "Liquidity",
            "The company holds cash and marketable securities of 4.2 billion dollars.",
        ),
        "acme-2023": (
            "Acme Corp 10-K 2023",
            "Risk factors",
            "Supply chain disruption could reduce our gross margin in fiscal 2023.",
        ),
        "beta-2024": (
            "Beta Inc 10-K 2024",
            "Legal proceedings",
            "A patent infringement suit was filed against Beta in the Eastern District of Texas.",
        ),
    }
    for doc_id, (title, heading, sentence) in topics.items():
        filler = "\n\n".join(
            f"Paragraph {i} of {doc_id} describes ordinary operations in some detail." for i in range(30)
        )
        body = f"{sentence} " + " ".join(f"{doc_id}-note{i}" for i in range(150))  # long enough to be kept
        (root / f"{doc_id}.md").write_text(
            f"# {title}\n\n## {heading}\n\n{body}\n\n## Operations\n\n{filler}\n", encoding="utf-8"
        )
        (root / f"{doc_id}.json").write_text(
            json.dumps({"title": title, "form": "10-K", "date": "2024-02-01"}), encoding="utf-8"
        )
    questions = [
        {
            "id": "q1",
            "question": "cash and marketable securities",
            "expected_doc_ids": ["acme-2024"],
            "expected_spans": ["cash and marketable securities of 4.2 billion"],
        },
        {
            "id": "q2",
            "question": "supply chain disruption gross margin",
            "expected_doc_ids": ["acme-2023"],
            "expected_spans": ["Supply chain disruption could reduce"],
        },
        {
            "id": "q3",
            "question": "patent infringement suit Eastern District of Texas",
            "expected_doc_ids": ["beta-2024"],
            "expected_spans": ["patent infringement suit was filed"],
        },
        {
            "id": "q4",
            "question": "dividend per share in Mars colony",
            "expected_behavior": "abstain",
            "expected_doc_ids": [],
        },
    ]
    qfile = root / "questions.jsonl"
    qfile.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")
    return qfile


def test_filings_eval_runs_on_a_synthetic_corpus(tmp_path, capsys):
    script = _script()
    qfile = _synthetic_corpus(tmp_path)
    out = tmp_path / "rows.json"
    code = script.main(
        [str(tmp_path), str(qfile), "--backend", "fake", "--sweep-threshold", "0.5,0.9"]
        + ["--sweep-keep-top", "0,1", "--sweep-solver", "lp_greedy", "--resamples", "50", "--json", str(out)]
    )  # fmt: skip
    assert code == 0
    printed = capsys.readouterr().out
    for needle in (
        "hit@5",
        "hit@10",
        "MRR@10",
        "s/10-K",
        "dense",
        "hybrid",
        "hybrid+eigenmind[t=0.5,keep=1,lp_greedy]",
        "v2 >= v1",
    ):
        assert needle in printed
    rows = json.loads(out.read_text())
    assert {(r["chunking"], r["retrieval"]) for r in rows} >= {("v1", "dense"), ("v2", "hybrid")}
    # words are shared with the question: the lexical side finds the passage even with meaningless vectors
    hybrid = next(r for r in rows if r["chunking"] == "v1" and r["retrieval"] == "hybrid")
    assert hybrid["report"]["hit_at_10"]["value"] == 1.0
    assert len(hybrid["report"]["abstentions"]) == 1


def test_eigenmind_verdict_needs_a_non_overlapping_interval():
    script = _script()

    def report(ranks):
        results = tuple(PassageResult(f"q{i}", r, 10) for i, r in enumerate(ranks))
        return PassageReport(k=10, results=results, bootstrap_resamples=200, interval_min_questions=5)

    base = report([1] * 10 + [None] * 10)  # hit@10 = 0.5
    better_overlapping = report([1] * 12 + [None] * 8)  # 0.6, intervals overlap
    much_better = report([1] * 40 + [None] * 0)  # 1.0
    assert script.eigenmind_verdict(base, better_overlapping)[0] is False
    assert script.eigenmind_verdict(report([1] * 20 + [None] * 20), much_better)[0] is True
    assert script.eigenmind_verdict(much_better, base)[0] is False
    assert script.eigenmind_verdict(report([1, None]), much_better)[0] is False  # too few questions


# --- review round 1 -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("span", "stored"),
    [
        (
            "The Company\u2019s cash totaled $4.2 billion \u2014 Caf\u00e9\u2122 brand",
            "the company's cash totaled $4.2 billion - cafe\u0301TM brand",
        ),
        ("\u201cquoted\u201d text", '"quoted" text'),
        ("2023\u20132024 growth", "2023-2024 growth"),
        ("zero\u200bwidth and soft\u00adhyphen", "zerowidth and softhyphen"),
    ],
)
def test_span_hit_folds_unicode_typography_on_both_sides(span, stored):
    import unicodedata

    assert span_hit(unicodedata.normalize("NFKD", stored), [span])
    assert span_hit(span, [stored])
    assert span_hit("prefix " + unicodedata.normalize("NFKD", span) + " suffix", [span])


def test_a_hit_must_come_from_an_expected_filing():
    question = _question("q", ("repeated sentence",))
    hits = [_hit("a repeated sentence here", doc_id="fy2023"), _hit("a repeated sentence here", doc_id="d")]
    report = evaluate_passages(QuestionSet((question,)), lambda q: hits, k=5)
    assert report.results[0].rank == 2


def test_k_below_ten_is_refused_before_any_work():
    script = _script()
    with pytest.raises(ValueError, match="k must be"):
        script.check_inputs([_question("q", ("x",))], ["v1"], 5)
    with pytest.raises(ValueError, match="profile"):
        script.check_inputs([_question("q", ("x",))], ["v3"], 10)
    with pytest.raises(ValueError, match="expected_spans"):
        script.check_inputs([Question(id="n", question="q?", expected_doc_ids=("d",))], ["v1"], 10)


def test_an_empty_report_writes_null_not_nan():
    report = PassageReport(k=10, results=())
    assert json.loads(json.dumps(report.to_dict(), allow_nan=False))["hit_at_10"]["value"] is None
