"""Settings: defaults, environment parsing, validation, no import-time side effect."""

import dataclasses
import os
import subprocess
import sys

import pytest

from rag_engine.config import ConfigError, Settings

SECRET = "k3y-Zx9-DISTINCTIVE-41f7"

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
    "parity_min": 0.999,
    "onnx_threads": None,
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
    "RAG_ENGINE_PARITY_MIN",
    "RAG_ENGINE_ONNX_THREADS",
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
        "RAG_ENGINE_PARITY_MIN": " 0.9995 ",
        "RAG_ENGINE_ONNX_THREADS": " 2 ",
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
        "parity_min": 0.9995,
        "onnx_threads": 2,
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
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({"RAG_ENGINE_SEARCH_MAX_LIMIT": "5"})
    message = str(excinfo.value)
    assert "RAG_ENGINE_SEARCH_LIMIT=10" in message
    assert "RAG_ENGINE_SEARCH_MAX_LIMIT=5" in message


# --- backend ----------------------------------------------------------------


@pytest.mark.parametrize("raw", ["onnx", "ONNX", " Onnx "])
def test_backend_is_case_insensitive_in_the_environment(raw):
    assert Settings.from_env({"RAG_ENGINE_BACKEND": raw}).backend == "onnx"


def test_invalid_backend_echoes_the_raw_value():
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({"RAG_ENGINE_BACKEND": "  TensorFlow "})
    message = str(excinfo.value)
    assert "RAG_ENGINE_BACKEND='  TensorFlow '" in message
    assert "torch" in message and "onnx" in message


# --- prefixes ---------------------------------------------------------------


PREFIXES = [
    ("RAG_ENGINE_PASSAGE_PREFIX", "passage_prefix", "passage: "),
    ("RAG_ENGINE_QUERY_PREFIX", "query_prefix", "query: "),
]


def test_prefixes_are_not_stripped():
    settings = Settings.from_env(
        {"RAG_ENGINE_PASSAGE_PREFIX": "  passage:  ", "RAG_ENGINE_QUERY_PREFIX": "query:\t"}
    )
    assert settings.passage_prefix == "  passage:  "
    assert settings.query_prefix == "query:\t"


@pytest.mark.parametrize(("name", "field", "default"), PREFIXES)
def test_empty_prefix_variable_gives_the_default(name, field, default):
    # A compose passthrough of an unset variable arrives as the empty string.
    assert getattr(Settings.from_env({name: ""}), field) == default


@pytest.mark.parametrize(("name", "field", "default"), PREFIXES)
@pytest.mark.parametrize("raw", ["none", "None", "NONE", "  none\t", " nOnE "])
def test_none_means_no_prefix(name, field, default, raw):
    assert getattr(Settings.from_env({name: raw}), field) == ""


@pytest.mark.parametrize(("name", "field", "default"), PREFIXES)
@pytest.mark.parametrize("raw", [" ", "   ", "none: ", "no", "passage:", " x "])
def test_any_other_prefix_is_taken_verbatim(name, field, default, raw):
    assert getattr(Settings.from_env({name: raw}), field) == raw


def test_one_prefix_set_leaves_the_other_at_its_default():
    settings = Settings.from_env({"RAG_ENGINE_QUERY_PREFIX": "none"})
    assert settings.passage_prefix == "passage: "
    assert settings.query_prefix == ""


def test_direct_construction_takes_prefixes_verbatim():
    settings = Settings(passage_prefix="", query_prefix="none")
    assert settings.passage_prefix == ""
    assert settings.query_prefix == "none"


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
    with pytest.raises(ConfigError, match="RAG_ENGINE_SEARCH_LIMIT=100.*RAG_ENGINE_SEARCH_MAX_LIMIT=50"):
        Settings(search_limit=100, search_max_limit=50)
    with pytest.raises(ConfigError, match="RAG_ENGINE_BACKEND='tensorflow'"):
        Settings(backend="tensorflow")
    with pytest.raises(ConfigError, match="RAG_ENGINE_BATCH_SIZE=0"):
        Settings(batch_size=0)


INT_FIELDS = [
    ("qdrant_port", "QDRANT_PORT"),
    ("model_dim", "RAG_ENGINE_DIM"),
    ("batch_size", "RAG_ENGINE_BATCH_SIZE"),
    ("search_limit", "RAG_ENGINE_SEARCH_LIMIT"),
    ("search_max_limit", "RAG_ENGINE_SEARCH_MAX_LIMIT"),
]

STR_FIELDS = [
    ("qdrant_host", "QDRANT_HOST"),
    ("model_name", "RAG_ENGINE_MODEL"),
    ("passage_prefix", "RAG_ENGINE_PASSAGE_PREFIX"),
    ("query_prefix", "RAG_ENGINE_QUERY_PREFIX"),
    ("backend", "RAG_ENGINE_BACKEND"),
    ("ocr_languages", "RAG_ENGINE_OCR_LANGUAGES"),
]


@pytest.mark.parametrize(("field", "variable"), INT_FIELDS)
@pytest.mark.parametrize("value", [True, False, 5.0, 5.5, "5", None, [5]])
def test_direct_integers_must_be_real_integers(field, variable, value):
    with pytest.raises(ConfigError) as excinfo:
        Settings(**{field: value})
    message = str(excinfo.value)
    assert variable in message
    assert repr(value) in message
    assert type(value).__name__ in message


@pytest.mark.parametrize("value", ["false", "true", "", 0, 1, None])
def test_direct_https_must_be_a_real_boolean(value):
    with pytest.raises(ConfigError) as excinfo:
        Settings(qdrant_https=value)
    message = str(excinfo.value)
    assert "QDRANT_HTTPS" in message
    assert repr(value) in message


@pytest.mark.parametrize(("field", "variable"), STR_FIELDS)
@pytest.mark.parametrize("value", [None, 5, b"torch", ["eng"]])
def test_direct_strings_must_be_strings(field, variable, value):
    with pytest.raises(ConfigError) as excinfo:
        Settings(**{field: value})
    message = str(excinfo.value)
    assert variable in message
    assert repr(value) in message


@pytest.mark.parametrize(
    ("field", "variable"), [("qdrant_host", "QDRANT_HOST"), ("model_name", "RAG_ENGINE_MODEL")]
)
@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_direct_empty_host_or_model_is_refused(field, variable, value):
    with pytest.raises(ConfigError) as excinfo:
        Settings(**{field: value})
    assert variable in str(excinfo.value)


@pytest.mark.parametrize("field", ["qdrant_api_key", "model_cache"])
@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_direct_blank_optional_string_becomes_none(field, value):
    assert getattr(Settings(**{field: value}), field) is None


def test_direct_optional_strings_must_be_strings_or_none():
    with pytest.raises(ConfigError, match="RAG_ENGINE_MODEL_CACHE=5"):
        Settings(model_cache=5)
    with pytest.raises(ConfigError, match="QDRANT_API_KEY"):
        Settings(qdrant_api_key=12345)


@pytest.mark.parametrize("value", ["ONNX", "Torch", " onnx", "onnx "])
def test_direct_backend_is_matched_exactly(value):
    with pytest.raises(ConfigError) as excinfo:
        Settings(backend=value)
    assert f"RAG_ENGINE_BACKEND={value!r}" in str(excinfo.value)


def test_direct_construction_accepts_valid_values():
    settings = Settings(
        qdrant_host="qdrant.internal",
        qdrant_port=6334,
        qdrant_api_key=SECRET,
        qdrant_https=True,
        backend="onnx",
        model_cache="/var/cache/models",
    )
    assert settings.qdrant_api_key == SECRET
    assert settings.qdrant_https is True
    assert settings.backend == "onnx"


# --- the API key is never printed -------------------------------------------


def test_api_key_is_absent_from_repr_and_str():
    for settings in (Settings(qdrant_api_key=SECRET), Settings.from_env({"QDRANT_API_KEY": SECRET})):
        assert settings.qdrant_api_key == SECRET
        for text in (repr(settings), str(settings), f"{settings}", f"{settings!r}", f"{[settings]}"):
            assert SECRET not in text
            assert "qdrant_api_key=<set>" in text


def test_repr_says_when_no_api_key_is_set():
    text = repr(Settings())
    assert "qdrant_api_key=<unset>" in text
    assert "qdrant_host='localhost'" in text
    assert "qdrant_port=6333" in text


def test_repr_shows_every_field():
    text = repr(Settings())
    assert text.startswith("Settings(") and text.endswith(")")
    for field in dataclasses.fields(Settings):
        assert f"{field.name}=" in text


INVALID_ENV = [
    {"QDRANT_PORT": "abc"},
    {"QDRANT_PORT": "0"},
    {"QDRANT_HTTPS": "maybe"},
    {"RAG_ENGINE_DIM": "0"},
    {"RAG_ENGINE_BACKEND": "tensorflow"},
    {"RAG_ENGINE_BATCH_SIZE": "many"},
    {"RAG_ENGINE_OCR_LANGUAGES": "eng,fra"},
    {"RAG_ENGINE_SEARCH_LIMIT": "51"},
    {"RAG_ENGINE_SEARCH_MAX_LIMIT": "5"},
    {"RAG_ENGINE_PARITY_MIN": "high"},
    {"RAG_ENGINE_PARITY_MIN": "0"},
    {"RAG_ENGINE_PARITY_MIN": "1.5"},
    {"RAG_ENGINE_ONNX_THREADS": "0"},
    {"RAG_ENGINE_ONNX_THREADS": "all"},
]

INVALID_KWARGS = [
    {"qdrant_host": ""},
    {"qdrant_host": 5},
    {"qdrant_port": "6333"},
    {"qdrant_port": 0},
    {"qdrant_https": "false"},
    {"model_name": ""},
    {"model_dim": 1.5},
    {"passage_prefix": None},
    {"query_prefix": 3},
    {"backend": "ONNX"},
    {"batch_size": True},
    {"ocr_languages": "eng,fra"},
    {"search_limit": 100},
    {"search_max_limit": 5},
    {"model_cache": 5},
    {"parity_min": 1},
    {"parity_min": 0.0},
    {"onnx_threads": 0},
    {"onnx_threads": "2"},
]


@pytest.mark.parametrize("env", INVALID_ENV)
def test_api_key_is_absent_from_environment_errors(env):
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({"QDRANT_API_KEY": SECRET, **env})
    assert SECRET not in str(excinfo.value)
    assert SECRET not in repr(excinfo.value)


@pytest.mark.parametrize("kwargs", INVALID_KWARGS)
def test_api_key_is_absent_from_direct_construction_errors(kwargs):
    with pytest.raises(ConfigError) as excinfo:
        Settings(qdrant_api_key=SECRET, **kwargs)
    assert SECRET not in str(excinfo.value)
    assert SECRET not in repr(excinfo.value)


@pytest.mark.parametrize("value", [SECRET.encode(), [SECRET], {"key": SECRET}])
def test_api_key_of_the_wrong_type_is_refused_without_echoing_it(value):
    with pytest.raises(ConfigError) as excinfo:
        Settings(qdrant_api_key=value)
    message = str(excinfo.value)
    assert "QDRANT_API_KEY" in message
    assert type(value).__name__ in message
    assert SECRET not in message


# --- echoed values are truncated --------------------------------------------

HUGE = "x" * 5000


@pytest.mark.parametrize(
    "env",
    [
        {"QDRANT_PORT": HUGE},
        {"QDRANT_HTTPS": HUGE},
        {"RAG_ENGINE_BACKEND": HUGE},
        {"RAG_ENGINE_OCR_LANGUAGES": HUGE + "+"},
        {"RAG_ENGINE_SEARCH_LIMIT": "9" * 4000},
        {"RAG_ENGINE_SEARCH_MAX_LIMIT": "-" + "9" * 4000},
    ],
)
def test_huge_environment_value_is_truncated_in_the_error(env):
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env(env)
    message = str(excinfo.value)
    assert next(iter(env)) in message
    assert len(message) < 300
    assert "…" in message
    assert "x" * 81 not in message and "9" * 81 not in message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"qdrant_host": [HUGE]},
        {"qdrant_port": HUGE},
        {"qdrant_https": HUGE},
        {"backend": HUGE},
        {"ocr_languages": HUGE + "+"},
        {"model_cache": [HUGE]},
        {"search_limit": 10**400},
    ],
)
def test_huge_direct_value_is_truncated_in_the_error(kwargs):
    with pytest.raises(ConfigError) as excinfo:
        Settings(**kwargs)
    message = str(excinfo.value)
    assert len(message) < 300
    assert "…" in message


def test_integer_too_large_to_print_still_raises_a_config_error():
    with pytest.raises(ConfigError) as excinfo:
        Settings(qdrant_port=10**5000)
    message = str(excinfo.value)
    assert "QDRANT_PORT" in message
    assert len(message) < 300


def test_short_values_are_echoed_in_full():
    value = "y" * 70
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({"RAG_ENGINE_BACKEND": value})
    assert repr(value) in str(excinfo.value)
    assert "…" not in str(excinfo.value)


# --- no side effect at import -----------------------------------------------


def _run(code, env, cwd=None):
    return subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=cwd, capture_output=True, text=True, check=False
    )


def _clean_environment():
    """The current environment without any variable the settings read."""
    return {
        name: value for name, value in os.environ.items() if not name.startswith(("QDRANT_", "RAG_ENGINE_"))
    }


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
    result = _run(code, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "from_env raised"


def test_variable_changed_after_import_is_seen():
    # Nothing is snapshotted at import: the environment is read on every call.
    env = dict(_clean_environment(), QDRANT_PORT="7001", RAG_ENGINE_PASSAGE_PREFIX="before: ")
    code = (
        "import os\n"
        "import rag_engine.config as config\n"
        "os.environ['QDRANT_PORT'] = '7123'\n"
        "os.environ['QDRANT_HOST'] = 'late.internal'\n"
        "os.environ['RAG_ENGINE_PASSAGE_PREFIX'] = 'after: '\n"
        "first = config.Settings.from_env()\n"
        "print(first.qdrant_port, first.qdrant_host, repr(first.passage_prefix))\n"
        "del os.environ['QDRANT_PORT']\n"
        "print(config.Settings.from_env().qdrant_port)\n"
    )
    result = _run(code, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["7123 late.internal 'after: '", "6333"]


def test_dotenv_file_in_the_working_directory_is_ignored(tmp_path):
    (tmp_path / ".env").write_text(
        "QDRANT_PORT=7999\nQDRANT_HOST=from-dotenv.internal\nRAG_ENGINE_BACKEND=onnx\n",
        encoding="utf-8",
    )
    code = (
        "import os, sys\n"
        "import rag_engine.config as config\n"
        "assert os.path.exists('.env'), os.getcwd()\n"
        "settings = config.Settings.from_env()\n"
        "assert 'dotenv' not in sys.modules\n"
        "assert 'QDRANT_PORT' not in os.environ\n"
        "print(settings.qdrant_port, settings.qdrant_host, settings.backend)\n"
    )
    result = _run(code, _clean_environment(), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "6333 localhost torch"


# ---------------------------------------------------------------------------
# parity_min
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("raw", "expected"), [("1", 1.0), ("1.0", 1.0), ("0.5", 0.5), ("9.99e-1", 0.999)])
def test_parity_min_is_read_as_a_number(raw, expected):
    value = Settings.from_env({"RAG_ENGINE_PARITY_MIN": raw}).parity_min
    assert type(value) is float
    assert value == expected


@pytest.mark.parametrize("raw", ["high", "0,999", "0.999x"])
def test_parity_min_that_is_not_a_number_is_refused(raw):
    with pytest.raises(ConfigError, match="RAG_ENGINE_PARITY_MIN=.*not a number"):
        Settings.from_env({"RAG_ENGINE_PARITY_MIN": raw})


@pytest.mark.parametrize("raw", ["0", "-0.5", "1.0001", "2", "nan", "inf", "-inf"])
def test_parity_min_outside_its_range_is_refused(raw):
    with pytest.raises(ConfigError, match="RAG_ENGINE_PARITY_MIN=.*above 0 and at most 1"):
        Settings.from_env({"RAG_ENGINE_PARITY_MIN": raw})


@pytest.mark.parametrize("value", [0.0, -0.1, 1.0000001, float("nan"), float("inf")])
def test_direct_parity_min_outside_its_range_is_refused(value):
    with pytest.raises(ConfigError, match="RAG_ENGINE_PARITY_MIN"):
        Settings(parity_min=value)


@pytest.mark.parametrize("value", [1, True, "0.999", None])
def test_direct_parity_min_must_be_a_float(value):
    with pytest.raises(ConfigError, match="RAG_ENGINE_PARITY_MIN=.*must be a float"):
        Settings(parity_min=value)


def test_direct_parity_min_accepts_the_bounds_of_its_range():
    assert Settings(parity_min=1.0).parity_min == 1.0
    assert Settings(parity_min=1e-9).parity_min == 1e-9


# ---------------------------------------------------------------------------
# onnx_threads
# ---------------------------------------------------------------------------
def test_onnx_threads_default_leaves_the_choice_to_the_runtime():
    assert Settings().onnx_threads is None
    assert Settings.from_env({}).onnx_threads is None
    assert Settings.from_env({"RAG_ENGINE_ONNX_THREADS": "  "}).onnx_threads is None


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), (" 4 ", 4), ("64", 64)])
def test_onnx_threads_is_read_as_an_integer(raw, expected):
    value = Settings.from_env({"RAG_ENGINE_ONNX_THREADS": raw}).onnx_threads
    assert type(value) is int
    assert value == expected


@pytest.mark.parametrize("raw", ["two", "2.0", "1,5", "auto"])
def test_onnx_threads_that_is_not_an_integer_is_refused(raw):
    with pytest.raises(ConfigError, match="RAG_ENGINE_ONNX_THREADS=.*not an integer"):
        Settings.from_env({"RAG_ENGINE_ONNX_THREADS": raw})


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_onnx_threads_below_one_is_refused(raw):
    with pytest.raises(ConfigError, match="RAG_ENGINE_ONNX_THREADS=.*at least 1"):
        Settings.from_env({"RAG_ENGINE_ONNX_THREADS": raw})


@pytest.mark.parametrize("value", [0, -2])
def test_direct_onnx_threads_below_one_is_refused(value):
    with pytest.raises(ConfigError, match="RAG_ENGINE_ONNX_THREADS=.*at least 1"):
        Settings(onnx_threads=value)


@pytest.mark.parametrize("value", [True, 2.0, "2"])
def test_direct_onnx_threads_must_be_an_integer_or_none(value):
    with pytest.raises(ConfigError, match="RAG_ENGINE_ONNX_THREADS=.*must be an integer or None"):
        Settings(onnx_threads=value)


def test_direct_onnx_threads_accepts_a_positive_integer_and_none():
    assert Settings(onnx_threads=1).onnx_threads == 1
    assert Settings(onnx_threads=None).onnx_threads is None
