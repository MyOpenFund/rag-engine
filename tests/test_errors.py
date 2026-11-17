"""The exception classes of the engine share one base, and keep the bases they had."""

import pytest

from rag_engine.chunking import UnsupportedFormatError
from rag_engine.config import ConfigError, Settings
from rag_engine.errors import RagEngineError
from rag_engine.store import CollectionMismatchError, QdrantAuthError, StoreError


def test_the_base_is_a_plain_exception():
    assert RagEngineError.__bases__ == (Exception,)


@pytest.mark.parametrize(
    "error", [ConfigError, StoreError, QdrantAuthError, CollectionMismatchError, UnsupportedFormatError]
)
def test_every_error_of_the_engine_derives_from_the_base(error):
    assert issubclass(error, RagEngineError)


@pytest.mark.parametrize("error", [ConfigError, UnsupportedFormatError])
def test_the_errors_that_were_value_errors_still_are(error):
    assert issubclass(error, ValueError)
    with pytest.raises(ValueError):
        raise error("x")


def test_a_store_error_is_not_a_value_error():
    assert not issubclass(StoreError, ValueError)
    assert issubclass(QdrantAuthError, StoreError) and issubclass(CollectionMismatchError, StoreError)


def test_a_config_error_is_caught_as_the_base_and_as_a_value_error():
    with pytest.raises(RagEngineError) as caught:
        Settings(qdrant_port=0)
    assert isinstance(caught.value, ConfigError) and isinstance(caught.value, ValueError)
    assert "QDRANT_PORT" in str(caught.value)
