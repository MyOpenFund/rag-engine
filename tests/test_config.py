"""Settings: defaults, environment parsing, validation, no import-time side effect."""

import dataclasses
import os
import subprocess
import sys

import pytest

from rag_engine.config import ConfigError, Settings

DEFAULTS = {
    "qdrant_host": "localhost",
    "qdrant_port": 6333,
    "qdrant_api_key": None,
    "qdrant_https": False,
    "model_name": "intfloat/multilingual-e5-base",
    "model_dim": 768,
    "passage_prefix": "passage: ",
    "query_prefix": "query: ",
    "backend": "torch",
    "batch_size": 100,
    "ocr_languages": "eng",
    "search_limit": 10,
    "search_max_limit": 50,
    "model_cache": None,
}

# Every variable where a blank value means "use the default".
BLANK_FALLS_BACK = [
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_API_KEY",
    "QDRANT_HTTPS",
    "RAG_ENGINE_MODEL",
    "RAG_ENGINE_DIM",
    "RAG_ENGINE_BACKEND",
    "RAG_ENGINE_BATCH_SIZE",
    "RAG_ENGINE_OCR_LANGUAGES",
    "RAG_ENGINE_SEARCH_LIMIT",
    "RAG_ENGINE_SEARCH_MAX_LIMIT",
    "RAG_ENGINE_MODEL_CACHE",
]


def test_defaults_with_empty_environment():
    assert dataclasses.asdict(Settings.from_env({})) == DEFAULTS


def test_direct_construction_uses_the_same_defaults():
    assert Settings() == Settings.from_env({})


def test_every_field_is_covered_by_the_defaults_table():
    assert {f.name for f in dataclasses.fields(Settings)} == set(DEFAULTS)


def test_settings_are_frozen():
    settings = Settings.from_env({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.qdrant_port = 1


def test_all_values_read_from_the_mapping():
    env = {
        "QDRANT_HOST": " qdrant.internal ",
        "QDRANT_PORT": " 6334 ",
        "QDRANT_API_KEY": "  s3cret  ",
        "QDRANT_HTTPS": "true",
        "RAG_ENGINE_MODEL": "intfloat/multilingual-e5-small",
        "RAG_ENGINE_DIM": "384",
        "RAG_ENGINE_PASSAGE_PREFIX": "doc: ",
        "RAG_ENGINE_QUERY_PREFIX": "q: ",
        "RAG_ENGINE_BACKEND": "ONNX",
        "RAG_ENGINE_BATCH_SIZE": "32",
        "RAG_ENGINE_OCR_LANGUAGES": "eng+fra",
        "RAG_ENGINE_SEARCH_LIMIT": "5",
        "RAG_ENGINE_SEARCH_MAX_LIMIT": "20",
        "RAG_ENGINE_MODEL_CACHE": "/var/cache/models",
    }
    assert dataclasses.asdict(Settings.from_env(env)) == {
        "qdrant_host": "qdrant.internal",
        "qdrant_port": 6334,
        "qdrant_api_key": "s3cret",
        "qdrant_https": True,
        "model_name": "intfloat/multilingual-e5-small",
        "model_dim": 384,
        "passage_prefix": "doc: ",
        "query_prefix": "q: ",
        "backend": "onnx",
        "batch_size": 32,
        "ocr_languages": "eng+fra",
        "search_limit": 5,
        "search_max_limit": 20,
        "model_cache": "/var/cache/models",
    }


def test_from_env_defaults_to_os_environ(monkeypatch):
    monkeypatch.setenv("QDRANT_PORT", "7000")
    assert Settings.from_env().qdrant_port == 7000


@pytest.mark.parametrize("name", BLANK_FALLS_BACK)
@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_value_falls_back_to_default(name, blank):
    assert dataclasses.asdict(Settings.from_env({name: blank})) == DEFAULTS


# --- booleans ---------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on", " On "])
def test_https_true_values(raw):
    assert Settings.from_env({"QDRANT_HTTPS": raw}).qdrant_https is True


@pytest.mark.parametrize("raw", ["", "0", "false", "False", "NO", "off", " off "])
def test_https_false_values(raw):
    assert Settings.from_env({"QDRANT_HTTPS": raw}).qdrant_https is False


def test_https_invalid_value_names_variable_and_value():
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({"QDRANT_HTTPS": "maybe"})
    assert "QDRANT_HTTPS" in str(excinfo.value)
    assert "'maybe'" in str(excinfo.value)


# --- integers ---------------------------------------------------------------


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("QDRANT_PORT", "abc"),
        ("QDRANT_PORT", "6333.0"),
        ("QDRANT_PORT", "0"),
        ("QDRANT_PORT", "-1"),
        ("QDRANT_PORT", "65536"),
        ("RAG_ENGINE_DIM", "wide"),
        ("RAG_ENGINE_DIM", "0"),
        ("RAG_ENGINE_BATCH_SIZE", "0"),
        ("RAG_ENGINE_BATCH_SIZE", "-5"),
        ("RAG_ENGINE_BATCH_SIZE", "many"),
        ("RAG_ENGINE_SEARCH_LIMIT", "0"),
        ("RAG_ENGINE_SEARCH_LIMIT", "ten"),
        ("RAG_ENGINE_SEARCH_MAX_LIMIT", "0"),
        ("RAG_ENGINE_SEARCH_MAX_LIMIT", "1e2"),
        ("RAG_ENGINE_BACKEND", "tensorflow"),
        ("RAG_ENGINE_OCR_LANGUAGES", "eng,fra"),
        ("RAG_ENGINE_OCR_LANGUAGES", "eng fra"),
        ("RAG_ENGINE_OCR_LANGUAGES", "eng++fra"),
        ("RAG_ENGINE_OCR_LANGUAGES", "+eng"),
    ],
)
def test_invalid_value_names_variable_and_value(name, raw):
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({name: raw})
    message = str(excinfo.value)
    assert name in message
    assert raw in message


@pytest.mark.parametrize("raw", ["1", "65535"])
def test_port_bounds_accepted(raw):
    assert Settings.from_env({"QDRANT_PORT": raw}).qdrant_port == int(raw)


# --- search limits ----------------------------------------------------------


def test_search_limit_may_equal_the_maximum():
    settings = Settings.from_env({"RAG_ENGINE_SEARCH_LIMIT": "50"})
    assert (settings.search_limit, settings.search_max_limit) == (50, 50)


def test_search_limit_above_maximum_is_refused():
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({"RAG_ENGINE_SEARCH_LIMIT": "51"})
    message = str(excinfo.value)
    assert "RAG_ENGINE_SEARCH_LIMIT" in message and "RAG_ENGINE_SEARCH_MAX_LIMIT" in message
    assert "51" in message and "50" in message


def test_maximum_below_default_limit_is_refused():
    with pytest.raises(ConfigError):
        Settings.from_env({"RAG_ENGINE_SEARCH_MAX_LIMIT": "5"})


# --- prefixes ---------------------------------------------------------------


def test_prefixes_are_not_stripped():
    settings = Settings.from_env(
        {"RAG_ENGINE_PASSAGE_PREFIX": "  passage:  ", "RAG_ENGINE_QUERY_PREFIX": "query:\t"}
    )
    assert settings.passage_prefix == "  passage:  "
    assert settings.query_prefix == "query:\t"


def test_empty_prefix_means_no_prefix():
    settings = Settings.from_env({"RAG_ENGINE_PASSAGE_PREFIX": "", "RAG_ENGINE_QUERY_PREFIX": ""})
    assert settings.passage_prefix == ""
    assert settings.query_prefix == ""


def test_one_prefix_set_leaves_the_other_at_its_default():
    settings = Settings.from_env({"RAG_ENGINE_QUERY_PREFIX": ""})
    assert settings.passage_prefix == "passage: "
    assert settings.query_prefix == ""


# --- OCR languages ----------------------------------------------------------


@pytest.mark.parametrize("raw", ["eng", "eng+fra", "chi_sim+eng", "script/Latin", " deu+eng "])
def test_ocr_languages_accepted(raw):
    assert Settings.from_env({"RAG_ENGINE_OCR_LANGUAGES": raw}).ocr_languages == raw.strip()


def test_ocr_settings_of_the_previous_engine_are_not_read():
    # ENABLE_OCR / TESSDATA_PREFIX never held a language list: they do not change the settings.
    env = {"ENABLE_OCR": "true", "TESSDATA_PREFIX": "/nonexistent/tessdata"}
    assert Settings.from_env(env).ocr_languages == "eng"


# --- direct construction is validated too -----------------------------------


def test_direct_construction_is_validated():
    with pytest.raises(ConfigError):
        Settings(search_limit=100, search_max_limit=50)
    with pytest.raises(ConfigError):
        Settings(backend="tensorflow")
    with pytest.raises(ConfigError):
        Settings(batch_size=0)


# --- no side effect at import -----------------------------------------------


def test_import_does_not_read_the_environment():
    env = dict(os.environ, QDRANT_PORT="abc", QDRANT_HTTPS="maybe", RAG_ENGINE_BACKEND="nope")
    code = (
        "import sys\n"
        "import rag_engine.config as config\n"
        "assert 'dotenv' not in sys.modules\n"
        "try:\n"
        "    config.Settings.from_env()\n"
        "except config.ConfigError as exc:\n"
        "    assert 'QDRANT_PORT' in str(exc), exc\n"
        "    print('from_env raised')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "from_env raised"
