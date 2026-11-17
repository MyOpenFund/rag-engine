"""The ``rag-engine`` command: ``--version`` and ``eval``.

The command builds its store and its embedder from the environment. The tests replace these
two builders with qdrant-client's local mode and a ``FakeEmbedder`` holding the fixture of
``eval/fixtures/``; everything else is the real command.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from eval_fixture import QUESTIONS, fixture_embedder, load_fixture
from qdrant_client import QdrantClient

import rag_engine
from rag_engine import cli
from rag_engine.config import Settings
from rag_engine.eval import evaluate, format_report, load_questions
from rag_engine.store import QdrantStore

COLLECTION = "eval-fixture"
ENVIRONMENT = (
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_API_KEY",
    "QDRANT_HTTPS",
    "RAG_ENGINE_MODEL",
    "RAG_ENGINE_DIM",
    "RAG_ENGINE_PASSAGE_PREFIX",
    "RAG_ENGINE_QUERY_PREFIX",
    "RAG_ENGINE_BACKEND",
    "RAG_ENGINE_BATCH_SIZE",
    "RAG_ENGINE_SEARCH_LIMIT",
    "RAG_ENGINE_SEARCH_MAX_LIMIT",
    "RAG_ENGINE_MODEL_CACHE",
    "RAG_ENGINE_ONNX_THREADS",
)


@pytest.fixture(scope="module")
def fixture_store():
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client)
    load_fixture(store, COLLECTION)
    yield store
    client.close()


@pytest.fixture
def built(monkeypatch, fixture_store):
    """The command wired to the fixture; records the settings its builders were given."""
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_ENGINE_DIM", "16")  # the dimension of the fixture collection
    seen: dict[str, object] = {}

    def store(settings: Settings) -> QdrantStore:
        seen["store_settings"] = settings
        return fixture_store

    def embedder(settings: Settings):
        seen["embedder_settings"] = settings
        seen["embedder"] = fixture_embedder()
        return seen["embedder"]

    monkeypatch.setattr(cli, "_store", store)
    monkeypatch.setattr(cli, "_embedder", embedder)
    return seen


def run(*arguments: str) -> int:
    return cli.main(["eval", "--questions", str(QUESTIONS), "--collection", COLLECTION, *arguments])


# ---------------------------------------------------------------------------
# The command itself
# ---------------------------------------------------------------------------
def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"rag-engine {rag_engine.__version__}"


def test_without_a_subcommand_is_a_usage_error(capsys):
    assert cli.main([]) == 2
    captured = capsys.readouterr()
    assert "usage: rag-engine" in captured.err and "eval" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["eval"],
        ["eval", "--questions", "q.jsonl"],
        ["eval", "--collection", "c"],
        ["nothing"],
        ["eval", "--questions", "q.jsonl", "--collection", "c", "--k", "ten"],
        ["eval", "--questions", "q.jsonl", "--collection", "c", "--backend", "tensorflow"],
        ["eval", "--questions", "q.jsonl", "--collection", "c", "--fail-under-hit-at-k", "1.5"],
        ["eval", "--questions", "q.jsonl", "--collection", "c", "--fail-under-hit-at-k", "-0.1"],
        ["eval", "--questions", "q.jsonl", "--collection", "c", "--fail-under-hit-at-k", "nan"],
        ["eval", "--questions", "q.jsonl", "--collection", "c", "--fail-under-hit-at-k", "most"],
        ["eval", "--questions", "q.jsonl", "--collection", "c", "--bootstrap-resamples", "many"],
    ],
)
def test_bad_arguments_are_usage_errors(arguments, capsys, built):
    assert cli.main(arguments) == 2
    assert "usage: rag-engine" in capsys.readouterr().err
    assert built == {}  # nothing was built


def test_help_exits_with_0(capsys):
    assert cli.main(["eval", "--help"]) == 0
    text = capsys.readouterr().out
    for option in ("--questions", "--collection", "--k", "--chunk-limit", "--backend", "--output"):
        assert option in text
    assert "--fail-under-hit-at-k" in text and "headline" in text
    assert "--allow-empty" in text and "--bootstrap-resamples" in text


def test_the_module_can_be_run():
    result = subprocess.run(
        [sys.executable, "-m", "rag_engine.cli", "--version"], capture_output=True, text=True, check=False
    )
    assert (result.returncode, result.stdout.strip()) == (0, f"rag-engine {rag_engine.__version__}")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def test_eval_prints_the_report_and_exits_with_0(built, capsys, fixture_store):
    assert run() == 0
    captured = capsys.readouterr()
    expected = evaluate(
        load_questions(str(QUESTIONS)),
        collection=COLLECTION,
        store=fixture_store,
        embedder=fixture_embedder(),
    )
    assert captured.out == format_report(expected) + "\n"
    assert captured.err == ""
    assert built["embedder"]._released is True


def test_eval_writes_the_json_report(built, capsys, tmp_path):
    output = tmp_path / "report.json"
    assert run("--output", str(output), "--k", "1", "--chunk-limit", "20") == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["parameters"]["k"] == 1 and data["parameters"]["chunk_limit"] == 20
    assert data["parameters"]["collection"] == COLLECTION
    assert data["summary"]["scored"] == 12
    assert [miss["id"] for miss in data["misses"]] == ["fx-q07", "fx-q08", "fx-q12"]
    assert data["parameters"]["bootstrap_resamples"] == 1000
    assert data["metrics"]["headline"]["kinds"] == ["verbatim"]
    assert output.read_text(encoding="utf-8").endswith("}\n")
    assert "hit@1" in capsys.readouterr().out


def test_eval_output_in_a_missing_directory_is_refused_before_the_evaluation(built, capsys, tmp_path):
    assert run("--output", str(tmp_path / "absent" / "report.json")) == 2
    assert "absent" in capsys.readouterr().err
    assert "embedder" not in built


@pytest.mark.parametrize(("threshold", "code"), [("0.91", 0), ("0.9166", 0), ("0.92", 1), ("1", 1), ("0", 0)])
def test_fail_under_hit_at_k_looks_at_the_headline(built, capsys, threshold, code, tmp_path):
    # With k = 1 the headline (the unfiltered kind, per document) is 11/12 = 0.9167. The pooled hit@1
    # is 9/12 and the per-question one of the same kind 5/6: neither is what the threshold reads.
    output = tmp_path / "report.json"
    assert run("--k", "1", "--fail-under-hit-at-k", threshold, "--output", str(output)) == code
    captured = capsys.readouterr()
    assert "headline" in captured.out  # the report is printed either way
    assert output.is_file()  # and written either way
    assert ("below" in captured.err) is (code == 1)
    if code == 1:
        assert "headline hit@1" in captured.err and "0.917" in captured.err


def test_fail_under_is_met_by_the_headline_even_when_filtered_kinds_miss(built, capsys):
    # Every unfiltered question finds its document among ten; a filtered one (fx-q08) does not.
    assert run("--fail-under-hit-at-k", "1") == 0
    assert "Misses (1)" in capsys.readouterr().out


def test_fail_under_is_not_met_without_a_headline(built, capsys, tmp_path):
    # Every question of the file carries filters: there is no kind for a headline.
    questions = tmp_path / "q.jsonl"
    line = {
        "id": "q1",
        "question": "anything",
        "expected_doc_ids": ["fx-us-projections-2015-12"],
        "filters": {"doc_types": ["F1"]},
    }
    questions.write_text(json.dumps(line) + "\n")
    arguments = ["eval", "--questions", str(questions), "--collection", COLLECTION]
    assert cli.main(arguments) == 0
    assert "1 questions scored" in capsys.readouterr().out
    assert cli.main([*arguments, "--fail-under-hit-at-k", "0.5"]) == 1
    assert "no question asked without filters could be scored" in capsys.readouterr().err


def test_a_collection_without_any_expected_document_is_refused(built, capsys, tmp_path):
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        '{"id": "q1", "question": "anything", "expected_doc_ids": ["ghost"]}\n'
        '{"id": "q2", "question": "anything else", "expected_doc_ids": ["ghost", "phantom"]}\n'
    )
    output = tmp_path / "report.json"
    arguments = ["eval", "--questions", str(questions), "--collection", COLLECTION, "--output", str(output)]
    assert cli.main(arguments) == 2
    assert_refused(capsys, built, "none of the 2 documents", repr(COLLECTION), "--allow-empty")
    assert not output.exists()


def test_allow_empty_runs_on_a_collection_without_any_expected_document(built, capsys, tmp_path):
    questions = tmp_path / "q.jsonl"
    questions.write_text('{"id": "q1", "question": "anything", "expected_doc_ids": ["ghost"]}\n')
    arguments = ["eval", "--questions", str(questions), "--collection", COLLECTION, "--allow-empty"]
    assert cli.main(arguments) == 0
    captured = capsys.readouterr()
    assert "0 questions scored" in captured.out
    assert captured.err.startswith("rag-engine: warning: ") and "none of the 1 documents" in captured.err
    assert cli.main([*arguments, "--fail-under-hit-at-k", "0.5"]) == 1
    assert "no question asked without filters could be scored" in capsys.readouterr().err


def test_allow_empty_changes_nothing_when_documents_are_there(built, capsys):
    assert run("--allow-empty") == 0
    captured = capsys.readouterr()
    assert "12 questions scored" in captured.out and captured.err == ""


def test_the_number_of_resamples_is_an_option(built, capsys, tmp_path):
    output = tmp_path / "report.json"
    assert run("--bootstrap-resamples", "50", "--output", str(output)) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["parameters"]["bootstrap_resamples"] == 50
    assert "50 resamples" in capsys.readouterr().out
    assert run("--bootstrap-resamples", "0") == 2
    assert "--bootstrap-resamples" in capsys.readouterr().err


def test_the_settings_come_from_the_environment(built, monkeypatch, capsys):
    monkeypatch.setenv("QDRANT_HOST", "qdrant.example")
    monkeypatch.setenv("RAG_ENGINE_BACKEND", "onnx")
    monkeypatch.setenv("RAG_ENGINE_SEARCH_MAX_LIMIT", "200")
    assert run("--chunk-limit", "200") == 0
    assert built["store_settings"].qdrant_host == "qdrant.example"
    assert built["embedder_settings"].backend == "onnx"
    assert "chunk_limit=200" in capsys.readouterr().out


def test_backend_option_wins_over_the_environment(built, monkeypatch):
    monkeypatch.setenv("RAG_ENGINE_BACKEND", "onnx")
    assert run("--backend", "torch") == 0
    assert built["embedder_settings"].backend == "torch"
    assert run("--backend", "onnx") == 0
    assert built["embedder_settings"].backend == "onnx"
    monkeypatch.delenv("RAG_ENGINE_BACKEND")
    assert run() == 0
    assert built["embedder_settings"].backend == "torch"


def assert_refused(capsys, built, *fragments: str) -> None:
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("rag-engine: error: ")
    for fragment in fragments:
        assert fragment in captured.err
    assert "Traceback" not in captured.err
    assert "embedder" not in built  # refused before the model was loaded


def test_a_chunk_limit_above_the_configured_maximum_is_refused(built, capsys):
    assert run("--chunk-limit", "51") == 2
    assert_refused(capsys, built, "chunk_limit 51", "50", "RAG_ENGINE_SEARCH_MAX_LIMIT")


@pytest.mark.parametrize(
    "arguments", [["--k", "0"], ["--chunk-limit", "0"], ["--k", "11", "--chunk-limit", "10"]]
)
def test_impossible_counts_are_refused(built, capsys, arguments):
    assert run(*arguments) == 2
    assert_refused(capsys, built)


def test_a_missing_question_file_is_refused(built, capsys, tmp_path):
    missing = tmp_path / "absent.jsonl"
    assert cli.main(["eval", "--questions", str(missing), "--collection", COLLECTION]) == 2
    assert_refused(capsys, built, "absent.jsonl")
    assert "store_settings" not in built


def test_a_bad_question_file_is_refused_with_its_line(built, capsys, tmp_path):
    questions = tmp_path / "q.jsonl"
    questions.write_text('{"id": "q1", "question": "x", "expected_doc_ids": []}\n')
    assert cli.main(["eval", "--questions", str(questions), "--collection", COLLECTION]) == 2
    assert_refused(capsys, built, "q.jsonl, line 1", "expected_doc_ids")


def test_a_bad_environment_is_refused(built, capsys, monkeypatch):
    monkeypatch.setenv("QDRANT_PORT", "sixty")
    assert run() == 2
    assert_refused(capsys, built, "QDRANT_PORT")


def test_a_collection_that_does_not_exist_is_refused(built, capsys):
    assert cli.main(["eval", "--questions", str(QUESTIONS), "--collection", "nowhere"]) == 2
    assert_refused(capsys, built, "'nowhere'", "does not exist")


def test_an_unreachable_server_is_refused(built, capsys, monkeypatch):
    class Unreachable:
        host, port = "qdrant.example", 6333

        def is_reachable(self) -> bool:
            return False

    monkeypatch.setattr(cli, "_store", lambda settings: Unreachable())
    assert run() == 2
    assert_refused(capsys, built, "qdrant.example", "6333", "QDRANT_API_KEY")


def test_a_missing_backend_is_refused_with_the_install_hint(built, capsys, monkeypatch):
    def missing(settings):
        raise ImportError("install it with: pip install 'rag-engine[torch]'")

    monkeypatch.setattr(cli, "_embedder", missing)
    assert run() == 2
    assert_refused(capsys, built, "rag-engine[torch]")


def test_a_collection_of_another_dimension_is_refused_before_the_model_is_loaded(built, capsys, monkeypatch):
    monkeypatch.setenv("RAG_ENGINE_DIM", "768")
    assert run() == 2
    # assert_refused checks that no embedder was built: the cheap check comes first.
    assert_refused(capsys, built, "768", "16", repr(COLLECTION), "RAG_ENGINE_DIM")
    assert "store_settings" in built


def test_a_model_of_another_dimension_than_announced_is_refused_too(built, capsys, monkeypatch):
    # The settings announce 16, the dimension of the collection, and the model built produces 768.
    class Wide:
        dim = 768
        model_name = "wide"
        released = False

        def release(self) -> None:
            Wide.released = True

    monkeypatch.setattr(cli, "_embedder", lambda settings: Wide())
    assert run() == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("rag-engine: error: ")
    assert "768" in captured.err and "16" in captured.err
    assert Wide.released is True


def named_vectors_store(vectors_config):
    from qdrant_client import models

    client = QdrantClient(":memory:")
    client.create_collection("named", vectors_config=vectors_config)
    return QdrantStore(client=client), models


def test_a_collection_with_named_vectors_gets_a_clear_message(built, capsys, monkeypatch):
    from qdrant_client import models

    config = {
        "title": models.VectorParams(size=16, distance=models.Distance.COSINE),
        "body": models.VectorParams(size=16, distance=models.Distance.COSINE),
    }
    store, _ = named_vectors_store(config)
    monkeypatch.setattr(cli, "_store", lambda settings: store)
    assert cli.main(["eval", "--questions", str(QUESTIONS), "--collection", "named"]) == 2
    assert_refused(capsys, built, "'named'", "named vectors", "'body', 'title'", "unnamed")
    store.client.close()


def test_the_checks_come_in_order_of_cost(built, capsys, monkeypatch, fixture_store):
    """Dimension, then the presence of the expected documents, then the model."""
    calls = []
    original = fixture_store.existing_doc_ids

    def recording(collection, doc_ids, **kwargs):
        calls.append("documents")
        return original(collection, doc_ids, **kwargs)

    def embedder(settings):
        calls.append("model")
        return fixture_embedder()

    monkeypatch.setattr(fixture_store, "existing_doc_ids", recording)
    monkeypatch.setattr(cli, "_embedder", embedder)
    assert run() == 0
    assert calls[:2] == ["documents", "model"]
    capsys.readouterr()
    calls.clear()
    monkeypatch.setenv("RAG_ENGINE_DIM", "768")
    assert run() == 2
    assert calls == []  # the dimension is wrong: the documents are not even looked up


def test_a_failure_during_the_evaluation_is_exit_code_2_not_1(built, capsys, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the server went away")

    monkeypatch.setattr(cli, "evaluate", broken)
    assert run("--fail-under-hit-at-k", "0.5") == 2
    captured = capsys.readouterr()
    assert "rag-engine: error: " in captured.err and "RuntimeError: the server went away" in captured.err
    assert built["embedder"]._released is True


def test_a_model_the_backend_refuses_is_a_configuration_error(built, capsys, monkeypatch):
    def refusing(settings):
        raise ValueError("embedding model 'x' produces 384-dimensional vectors (check RAG_ENGINE_DIM)")

    monkeypatch.setattr(cli, "_embedder", refusing)
    assert run() == 2
    assert_refused(capsys, built, "RAG_ENGINE_DIM")


def test_the_real_builders_follow_the_settings(monkeypatch):
    settings = Settings(qdrant_host="qdrant.example", qdrant_port=7000)
    built = {}
    monkeypatch.setattr(cli, "QdrantStore", lambda **kwargs: built.setdefault("store", kwargs))
    monkeypatch.setattr(cli, "make_embedder", lambda given: built.setdefault("embedder", given))
    assert cli._store(settings) == {"settings": settings}
    assert cli._embedder(settings) is settings
