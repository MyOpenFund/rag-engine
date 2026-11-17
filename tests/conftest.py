"""Shared test configuration: the search tests run once per case of the filter matrix."""

from __future__ import annotations

import pytest
from search_corpus import SEARCH_CASES


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Run a test that takes a ``search_case`` argument once per entry of ``SEARCH_CASES``."""
    if "search_case" in metafunc.fixturenames:
        metafunc.parametrize("search_case", SEARCH_CASES, ids=[case.label for case in SEARCH_CASES])
