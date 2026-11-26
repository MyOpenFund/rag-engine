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
    pytest.importorskip("chunknorris")  # the ingest extra (not installed on Python 3.10 in CI)
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


# --- step 5-6: the per-section cut, the span checker, the committed set ---------------------------


def _checker():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "check_filings_eval.py"
    spec = importlib.util.spec_from_file_location("check_filings_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_section_pieces_cut_like_the_mcp():
    script = _script()
    text = "AAAA" + "1111" + "BB" + "2222" + "   " + "3333" + "CC"
    sections = [("Item 2", 10, 14), ("Item 1", 4, 8), ("Item 3", 17, 21), ("Item 1A", 6, 9)]
    sections.append(("Item 9", 40, 50))
    pieces = script.section_pieces(text, sections)
    # sorted by start; Item 1A clipped to start after Item 1 (8..9); blank gap and past-the-end dropped
    assert pieces == [
        (None, "AAAA"),
        ("Item 1", "1111"),
        ("Item 1A", "B"),
        (None, "B"),
        ("Item 2", "2222"),
        ("Item 3", "3333"),
        (None, "CC"),
    ]
    assert "".join(piece for _, piece in pieces) == text.replace("   ", "")
    assert script.section_pieces("plain text", []) == [(None, "plain text")]


def test_section_cut_never_spans_two_sections(tmp_path):
    pytest.importorskip("chunknorris")  # the ingest extra (not installed on Python 3.10 in CI)
    script = _script()
    first = "# Item 7\n\n" + " ".join(f"alpha{i}" for i in range(60))
    second = "\n\n# Item 8\n\n" + " ".join(f"omega{i}" for i in range(60))
    text = first + second
    filing = script.Filing("d", text, "T", "10-K", "2024-01-01", (("Item 7", 0, len(first)),))
    options = script.PROFILES["v2"]()
    whole = script.filing_chunks(filing, options=options, tokenizer=script._WordTokenizer(), cut="document")
    split = script.filing_chunks(filing, options=options, tokenizer=script._WordTokenizer(), cut="section")
    assert any("alpha" in c.text and "omega" in c.text for c in whole)
    assert not any("alpha" in c.text and "omega" in c.text for c in split)
    with pytest.raises(ValueError, match="sections"):
        script.filing_chunks(
            script.Filing("d", text, "T", "10-K", "2024-01-01"),
            options=options,
            tokenizer=script._WordTokenizer(),
            cut="section",
        )


def test_load_filings_reads_and_checks_sections(tmp_path):
    script = _script()
    (tmp_path / "d.md").write_text("0123456789", encoding="utf-8")
    meta = {"title": "T", "form": "10-K", "date": "2024-01-01"}
    (tmp_path / "d.json").write_text(json.dumps(meta), encoding="utf-8")
    assert script.load_filings(tmp_path)[0].sections is None
    meta["sections"] = [{"name": "Item 7", "start": 2, "end": 5}]
    (tmp_path / "d.json").write_text(json.dumps(meta), encoding="utf-8")
    assert script.load_filings(tmp_path)[0].sections == (("Item 7", 2, 5),)
    for bad in ([{"name": "Item 7", "start": 2, "end": 50}], [{"name": "Item 7", "start": 5, "end": 2}], "x"):
        meta["sections"] = bad
        (tmp_path / "d.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="section"):
            script.load_filings(tmp_path)


def test_filing_title_is_the_mcp_title():
    """The no-vault MCP's `filing_title` (ruling R40): the chunk context, so the eval must match it."""
    script = _script()
    assert script.filing_title("AMAZON COM INC", "10-K", "2022-12-31", "2023-02-03") == (
        "AMAZON COM INC 10-K for the period ended 2022-12-31, filed 2023-02-03"
    )
    without_period = script.filing_title("AMAZON COM INC", "8-K", None, "2023-02-03")
    assert without_period == "AMAZON COM INC 8-K, filed 2023-02-03"


def test_load_filings_builds_the_title_from_the_company(tmp_path):
    script = _script()
    (tmp_path / "d.md").write_text("0123456789", encoding="utf-8")
    meta = {"company": "MICROSOFT CORP", "form": "10-K", "date": "2024-07-30", "report_date": "2024-06-30"}
    (tmp_path / "d.json").write_text(json.dumps(meta), encoding="utf-8")
    title = "MICROSOFT CORP 10-K for the period ended 2024-06-30, filed 2024-07-30"
    assert script.load_filings(tmp_path)[0].title == title
    del meta["report_date"]
    (tmp_path / "d.json").write_text(json.dumps(meta), encoding="utf-8")
    assert script.load_filings(tmp_path)[0].title == "MICROSOFT CORP 10-K, filed 2024-07-30"
    meta["title"] = "Microsoft 10-K FY2024"  # a title that is not the built one: refused, not silently used
    (tmp_path / "d.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="title"):
        script.load_filings(tmp_path)
    meta.pop("title")
    meta.pop("company")
    (tmp_path / "d.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="title or company"):
        script.load_filings(tmp_path)


def test_filings_eval_section_cut_end_to_end(tmp_path, capsys):
    pytest.importorskip("chunknorris")  # the ingest extra (not installed on Python 3.10 in CI)
    script = _script()
    qfile = _synthetic_corpus(tmp_path)
    for sidecar in tmp_path.glob("*.json"):
        meta = json.loads(sidecar.read_text())
        text = sidecar.with_suffix(".md").read_text()
        cut = text.index("## Operations")
        meta["sections"] = [{"name": "Item 7", "start": 0, "end": cut}]
        sidecar.write_text(json.dumps(meta))
    out = tmp_path / "rows.json"
    args = [str(tmp_path), str(qfile), "--backend", "fake", "--cuts", "document,section", "--resamples", "50"]
    assert script.main([*args, "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "v2/section" in printed
    assert "v2 >= v1 on hit@10 (hybrid, section cut)" in printed
    rows = json.loads(out.read_text())
    assert {r["chunking"] for r in rows} == {"v1", "v2", "v1/section", "v2/section"}
    assert {len(r["questions_sha256"]) for r in rows} == {64}
    with pytest.raises(ValueError, match="cut"):
        script.check_inputs([_question("q", ("x",))], ["v1"], 10, ["pages"])


def test_section_cut_needs_sections_before_any_work(tmp_path):
    script = _script()
    qfile = _synthetic_corpus(tmp_path)  # sidecars without sections
    with pytest.raises(ValueError, match="no 'sections'"):
        script.main([str(tmp_path), str(qfile), "--backend", "fake", "--cuts", "section"])


def test_span_checker_reports_each_rule(tmp_path, capsys):
    checker = _checker()
    (tmp_path / "a-2024.md").write_text(
        "Cash totaled $4.2 billion at year end.\n\nThe board met.\n\nRepeated boilerplate sentence here.\n"
        "Twice: capital expenditures of $9 billion; again capital expenditures of $9 billion;"
        " thrice capital expenditures of $9 billion.",
        encoding="utf-8",
    )
    (tmp_path / "a-2023.md").write_text("Repeated boilerplate sentence here.", encoding="utf-8")

    def row(id_, doc, span):
        return {"id": id_, "question": "q?", "expected_doc_ids": [doc], "expected_spans": [span]}

    rows = [
        row("ok", "a-2024", "cash  TOTALED $4.2 billion"),
        row("missing", "a-2024", "never written anywhere"),
        row("short", "a-2024", "----------------------"),
        row("shared", "a-2024", "Repeated boilerplate sentence"),
        row("thrice", "a-2024", "capital expenditures of $9 billion"),
        row("nofile", "b-2024", "Cash totaled $4.2 billion"),
        {"id": "abs", "question": "q?", "expected_behavior": "abstain", "expected_doc_ids": []},
    ]
    qfile = tmp_path / "q.jsonl"
    qfile.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert checker.main([str(qfile), str(tmp_path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert not any(line.startswith(("ok", "abs")) for line in lines)
    joined = "\n".join(lines)
    assert "missing 'never written anywhere': not found in any expected filing" in joined
    assert "short '" in joined and "too short or no letter" in joined
    assert "also in filings not expected: a-2023" in joined
    assert "found 3 times in a-2024" in joined
    assert "expected filing 'b-2024' not in the directory" in joined
    assert "7 questions (6 answering, 1 abstaining)" in joined
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[-1]) + "\n", encoding="utf-8")
    assert checker.main([str(good), str(tmp_path)]) == 0


def test_the_committed_filings_set_is_well_formed():
    questions = load_questions(EVAL / "filings_v1.jsonl")
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids)) and len(ids) >= 40
    abstaining = [q for q in questions if q.expected_behavior == "abstain"]
    assert len(abstaining) >= 4
    for question in questions:
        if question.expected_behavior != "abstain":
            assert 1 <= len(question.expected_spans) <= 3
            assert question.expected_doc_ids


# --- filings v2: difficulty, span_match "all", breakdowns, the checker's new rules -----------------


def test_span_match_all_needs_every_span_and_ranks_at_the_last_one():
    both = _question("both", ("alpha fact", "omega fact"), span_match="all")
    texts = ("noise", "the alpha fact", "noise", "an omega fact", "alpha fact again")
    hits = [_hit(text) for text in texts]
    report = evaluate_passages(QuestionSet((both,)), lambda q: hits, k=10)
    assert report.results[0].rank == 4  # alpha at 2, omega at 4: answered once both are in
    assert report.hit_at(3) == 0.0 and report.hit_at(5) == 1.0
    one_missing = evaluate_passages(QuestionSet((both,)), lambda q: hits[:3], k=10)
    assert one_missing.results[0].rank is None
    either = _question("either", ("alpha fact", "omega fact"))  # the default: any span will do
    assert evaluate_passages(QuestionSet((either,)), lambda q: hits, k=10).results[0].rank == 2


def test_span_match_all_counts_each_span_only_from_an_expected_filing():
    spans = ("from a", "from b")
    question = Question(
        id="q", question="q?", expected_doc_ids=("a", "b"), expected_spans=spans, span_match="all"
    )
    hits = [_hit("from a", doc_id="a"), _hit("from b", doc_id="other"), _hit("from b", doc_id="b")]
    assert evaluate_passages(QuestionSet((question,)), lambda q: hits, k=10).results[0].rank == 3


def test_span_match_and_difficulty_are_checked_and_written_back(tmp_path):
    path = tmp_path / "q.jsonl"
    line = {
        "id": "a",
        "question": "q?",
        "expected_doc_ids": ["d"],
        "expected_spans": ["one", "two"],
        "span_match": "all",
        "difficulty": "hard",
    }
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    question = load_questions(path)[0]
    assert (question.span_match, question.difficulty) == ("all", "hard")
    assert question.to_dict()["span_match"] == "all" and question.to_dict()["difficulty"] == "hard"
    plain = Question(id="p", question="q?", expected_doc_ids=("d",), expected_spans=("x",))
    assert "span_match" not in plain.to_dict() and "difficulty" not in plain.to_dict()
    bads = (
        {"span_match": "most"},
        {"difficulty": "trivial"},
        {"span_match": "all", "expected_spans": ["one"]},
        {"difficulty": 3},
    )
    for bad in bads:
        path.write_text(json.dumps({**line, **bad}) + "\n", encoding="utf-8")
        with pytest.raises(EvalError):
            load_questions(path)


def test_a_passage_report_splits_by_difficulty_and_language():
    results = (
        PassageResult("e1", 1, 10, "capex", "easy", "en"),
        PassageResult("e2", None, 10, "capex", "easy", "fr"),
        PassageResult("h1", 4, 10, "risk", "hard", "fr"),
        PassageResult("n1", 2, 10, None, None, None),
    )
    report = PassageReport(k=10, results=results, interval_min_questions=2, bootstrap_resamples=50)
    by_difficulty = report.groups("difficulty")
    assert list(by_difficulty) == ["easy", "hard", "(none)"]
    assert by_difficulty["easy"].hit_at(10) == 0.5 and by_difficulty["hard"].mrr_at(10) == 0.25
    assert by_difficulty["hard"].interval("hit_at_10") is None  # one question: no interval
    assert [r.id for r in report.groups("language")["fr"].results] == ["e2", "h1"]
    with pytest.raises(EvalError, match="grouping"):
        report.groups("issuer")


def test_evaluate_passages_carries_difficulty_and_language():
    question = _question("q", ("x",), difficulty="medium", language="fr")
    result = evaluate_passages(QuestionSet((question,)), lambda q: [_hit("x")], k=5).results[0]
    assert (result.difficulty, result.language) == ("medium", "fr")


def test_the_runner_breaks_metrics_down_by_difficulty_and_language():
    script = _script()
    results = (
        PassageResult("a", 1, 10, None, "easy", "en"),
        PassageResult("b", None, 10, None, "hard", "fr"),
        PassageResult("c", 7, 10, None, "hard", "en"),
    )
    report = PassageReport(k=10, results=results)
    assert script.breakdown(report, "difficulty") == {
        "easy": {"questions": 1, "hit_at_5": 1.0, "hit_at_10": 1.0, "mrr_10": 1.0},
        "hard": {"questions": 2, "hit_at_5": 0.0, "hit_at_10": 0.5, "mrr_10": pytest.approx(1 / 14)},
    }
    row = script.Row("v2", "hybrid", report, 1.0, 2.0, 3)
    table = script.format_breakdown([row], "language")
    assert "language" in table.splitlines()[0]
    assert any(line.split()[:4] == ["v2", "hybrid", "fr", "1"] for line in table.splitlines())


def test_the_runner_prints_and_writes_the_breakdowns(tmp_path, capsys):
    pytest.importorskip("chunknorris")  # the ingest extra (not installed on Python 3.10 in CI)
    script = _script()
    qfile = _synthetic_corpus(tmp_path)
    lines = [json.loads(line) for line in qfile.read_text().splitlines()]
    labels = [("easy", "en"), ("hard", "fr"), ("hard", "en"), ("easy", "en")]
    for line, (difficulty, language) in zip(lines, labels, strict=True):
        line.update(difficulty=difficulty, language=language)
    qfile.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    out = tmp_path / "rows.json"
    args = [str(tmp_path), str(qfile), "--backend", "fake", "--resamples", "20", "--json", str(out)]
    assert script.main(args) == 0
    printed = capsys.readouterr().out
    assert "by difficulty" in printed and "by language" in printed
    row = json.loads(out.read_text())[0]
    assert set(row["by_difficulty"]) == {"easy", "hard"}  # the abstention is not scored
    assert row["by_difficulty"]["hard"]["questions"] == 2 and set(row["by_language"]) == {"en", "fr"}


def test_span_checker_new_rules(tmp_path, capsys):
    checker = _checker()
    (tmp_path / "acme-2024.md").write_text(
        "Capital expenditures were $9.1 billion in 2024. We hold a stake in Widget Labs.", encoding="utf-8"
    )
    for name, text in (
        ("acme-2023", "Capital expenditures were $7.4 billion in 2023."),
        ("beta-2024", "Our philanthropic work; NovaChip is a supplier."),
    ):
        (tmp_path / f"{name}.md").write_text(text, encoding="utf-8")
    rows = [
        {
            "id": "ok",
            "question": "q?",
            "expected_doc_ids": ["acme-2024"],
            "difficulty": "easy",
            "expected_spans": ["Capital expenditures were $9.1 billion in 2024"],
        },
        {
            "id": "idle-filing",
            "question": "q?",
            "expected_doc_ids": ["acme-2024", "acme-2023"],
            "expected_spans": ["Capital expenditures were $9.1 billion in 2024"],
        },
        {
            "id": "abs-ok",
            "question": "q?",
            "expected_behavior": "abstain",
            "expected_doc_ids": [],
            "difficulty": "hard",
            "tags": ["absent:Gizmo Corp", "absent@beta:anthropic", "absent@acme:NovaChip"],
        },
        {
            "id": "abs-bad",
            "question": "q?",
            "expected_behavior": "abstain",
            "expected_doc_ids": [],
            "difficulty": "hard",
            "tags": ["absent:widget labs", "absent@beta:NovaChip"],
        },
    ]
    qfile = tmp_path / "q.jsonl"
    qfile.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert checker.main([str(qfile), str(tmp_path), "--require-difficulty", "--report"]) == 1
    out = capsys.readouterr().out
    assert "idle-filing: expected filing 'acme-2023' holds none of the spans" in out
    assert "idle-filing: no difficulty" in out
    assert "abs-bad: absent term 'widget labs' found in acme-2024" in out
    assert "abs-bad: absent term 'NovaChip' (beta filings) found in beta-2024" in out
    assert not any(line.startswith("abs-ok:") for line in out.splitlines())  # 'anthropic' is not a whole word
    # the report: the span is in no other filing, but its wording is (the 2023 sentence, other figures)
    expected_line = (
        "ok: 'Capital expenditures were $9.1 billion in 2024' expected [acme-2024x1] other filings 0"
    )
    assert expected_line in out
    assert "same wording in 1 other filing(s)" in out
    assert checker.wording("Capital expenditures were $9.1 billion in 2024") == "capital expenditures were"
    assert checker.wording("$4.0 billion in April 2026") == ""  # every stretch is under 15 characters


def test_the_committed_v2_set_is_well_formed():
    questions = load_questions(EVAL / "filings_v2.jsonl")
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids)) and 60 <= len(ids) <= 90
    assert all(q.difficulty in ("easy", "medium", "hard") for q in questions)
    abstaining = [q for q in questions if q.expected_behavior == "abstain"]
    assert 0.07 <= len(abstaining) / len(questions) <= 0.13
    assert {q.difficulty for q in abstaining} == {"easy", "medium", "hard"}
    french = [q for q in questions if q.language == "fr"]
    assert len(french) >= 6 and {q.difficulty for q in french} == {"easy", "medium", "hard"}
    for difficulty, low, high in (("easy", 0.2, 0.4), ("medium", 0.35, 0.55), ("hard", 0.18, 0.32)):
        assert low <= sum(q.difficulty == difficulty for q in questions) / len(questions) <= high
    tables = [s for q in questions for s in q.expected_spans if s.startswith("|")]
    assert len(tables) >= 8
    for question in questions:
        if question.expected_behavior == "abstain":
            assert not question.expected_spans and any(t.startswith("absent") for t in question.tags)
            continue
        assert 1 <= len(question.expected_spans) <= 3 and question.expected_doc_ids
        if question.span_match == "all":
            assert question.difficulty == "hard"
    issuers_years = {
        (t1, t2)
        for q in questions
        for t1 in q.tags
        if t1.startswith("issuer:")
        for t2 in q.tags
        if t2.startswith("fy:")
    }
    assert len(issuers_years) == 20  # every filing is asked about
