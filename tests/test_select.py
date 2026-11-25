"""select_diverse: eigenmind's Supported Diversity over the fused hybrid candidates (offline)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from rag_engine.search import Hit
from rag_engine.select import select_diverse


def _hit(rank: int, score: float) -> Hit:
    return Hit(
        text=f"chunk {rank}",
        score=score,
        doc_id="d",
        title=None,
        date=None,
        source_code=None,
        doc_type=None,
        language=None,
        page=None,
        chunk_number=rank,
        corpus=None,
        point_id=rank,
        payload={"text": f"chunk {rank}"},
    )


def _hits(n: int) -> list[Hit]:
    # Fused RRF scores, best first, as hybrid_search hands them over.
    return [_hit(rank, 1.0 / (60 + rank + 1)) for rank in range(n)]


def _clusters(sizes: list[int], *, dim: int = 16, spread: float = 0.02, seed: int = 0) -> tuple:
    """Unit vectors in tight clusters around orthogonal centres; returns (vectors, labels)."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for label, size in enumerate(sizes):
        centre = np.zeros(dim)
        centre[label] = 1.0
        for _ in range(size):
            rows.append(centre + spread * rng.normal(size=dim))
            labels.append(label)
    vectors = np.array(rows)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True), labels


def _interleaved(labels: list[int], vectors: np.ndarray, order: list[int]) -> tuple:
    return vectors[order], [labels[i] for i in order]


# --- with eigenmind --------------------------------------------------------------------------------


def test_three_clusters_one_each():
    pytest.importorskip("eigenmind")
    vectors, labels = _clusters([5, 5, 5])
    # Rank order mixes the clusters: the top 3 by fused score would be two of cluster 0 and one of 1.
    order = [0, 1, 5, 2, 6, 10, 3, 7, 11, 4, 8, 12, 9, 13, 14]
    vectors, labels = _interleaved(labels, vectors, order)
    hits = _hits(15)
    chosen = select_diverse(hits, vectors, 3)
    assert len(chosen) == 3
    ranks = [hits.index(hit) for hit in chosen]
    assert sorted({labels[r] for r in ranks}) == [0, 1, 2]
    # Utilities are the fused scores: each cluster gives its best-ranked member.
    assert ranks == [0, 2, 5]


def test_deterministic():
    pytest.importorskip("eigenmind")
    rng = np.random.default_rng(7)
    vectors, _ = _clusters([6, 4, 4, 3, 3], spread=0.3, seed=3)
    vectors = vectors[rng.permutation(len(vectors))]
    hits = _hits(len(vectors))
    first = select_diverse(hits, vectors, 5)
    second = select_diverse(hits, vectors, 5)
    assert [h.point_id for h in first] == [h.point_id for h in second]
    assert len(first) == 5


def test_output_is_sorted_by_fused_rank():
    pytest.importorskip("eigenmind")
    vectors, _ = _clusters([4, 4, 4])
    hits = _hits(12)
    # Put the clusters' best members at ranks 3, 7, 11 by reversing the order.
    chosen = select_diverse(hits[::-1], vectors[::-1], 3)
    ranks = [hits[::-1].index(hit) for hit in chosen]
    assert ranks == sorted(ranks)


def test_fills_up_with_the_best_remaining_hits_when_too_few_are_selected():
    pytest.importorskip("eigenmind")
    # Two clusters only: the stable set holds at most 2, limit 4 is filled by fused rank.
    vectors, _ = _clusters([4, 4])
    hits = _hits(8)
    chosen = select_diverse(hits, vectors, 4)
    ranks = [hits.index(hit) for hit in chosen]
    assert len(chosen) == 4
    assert ranks == sorted(ranks)
    assert {0, 4} <= set(ranks)  # one per cluster, the best of each
    assert ranks == [0, 1, 2, 4]  # then ranks 1 and 2, the best left


def test_isolated_candidates_are_not_selected_by_eigenmind_but_fill_up():
    pytest.importorskip("eigenmind")
    # Rank 0 is an isolate (no neighbour above the threshold): Supported Diversity drops it.
    vectors, _ = _clusters([1, 3, 3])
    hits = _hits(7)
    chosen = select_diverse(hits, vectors, 2)
    assert [hits.index(hit) for hit in chosen] == [1, 4]
    chosen = select_diverse(hits, vectors, 3)
    assert [hits.index(hit) for hit in chosen] == [0, 1, 4]


def test_keep_top_zero_pins_the_isolate_behaviour():
    pytest.importorskip("eigenmind")
    vectors, _ = _clusters([1, 3, 3])  # rank 0 is an isolate
    hits = _hits(7)
    assert [hits.index(h) for h in select_diverse(hits, vectors, 2, keep_top=0)] == [1, 4]


def test_keep_top_one_keeps_the_best_hit_even_when_isolated():
    pytest.importorskip("eigenmind")
    vectors, _ = _clusters([1, 3, 3])
    hits = _hits(7)
    assert [hits.index(h) for h in select_diverse(hits, vectors, 2, keep_top=1)] == [0, 1]
    assert [hits.index(h) for h in select_diverse(hits, vectors, 3, keep_top=1)] == [0, 1, 4]


def test_keep_top_leaves_the_kept_hits_near_duplicates_out_of_the_selection():
    pytest.importorskip("eigenmind")
    vectors, _ = _clusters([3, 3])
    hits = _hits(6)
    # Rank 0 is kept; ranks 1 and 2 are its near-duplicates, so the other slot goes to rank 3.
    assert [hits.index(h) for h in select_diverse(hits, vectors, 2, keep_top=1)] == [0, 3]


def test_keep_top_at_least_limit_gives_the_first_hits(monkeypatch):
    monkeypatch.setitem(sys.modules, "eigenmind", None)  # not needed, so not imported
    vectors, _ = _clusters([3, 3])
    hits = _hits(6)
    assert select_diverse(hits, vectors, 2, keep_top=2) == hits[:2]
    assert select_diverse(hits, vectors, 2, keep_top=5) == hits[:2]


def test_the_default_time_limit_is_two_seconds():
    import inspect

    assert inspect.signature(select_diverse).parameters["time_limit"].default == 2.0


def test_solver_failure_falls_back_to_fused_rank(monkeypatch):
    eigenmind = pytest.importorskip("eigenmind")

    def broken(*_args, **_kwargs):
        raise RuntimeError("LP failed")

    monkeypatch.setattr(eigenmind, "supported_diversity", broken)
    vectors, _ = _clusters([3, 3])
    hits = _hits(6)
    assert select_diverse(hits, vectors, 2) == hits[:2]


def test_threshold_is_a_parameter():
    pytest.importorskip("eigenmind")
    # Two tight groups whose centres have a cosine of 0.8.
    rng = np.random.default_rng(5)
    a, b = np.zeros(16), np.zeros(16)
    a[0], b[0], b[1] = 1.0, 0.8, 0.6
    rows = [c + 0.01 * rng.normal(size=16) for c in (a, a, a, b, b, b)]
    vectors = np.array(rows)
    hits = _hits(6)
    # Above 0.8 the groups are two components: one hit from each.
    assert [hits.index(h) for h in select_diverse(hits, vectors, 2, threshold=0.9)] == [0, 3]
    # At the default (0.65) the six form one clique: one hit, filled up by fused rank.
    assert [hits.index(h) for h in select_diverse(hits, vectors, 2)] == [0, 1]


# --- without eigenmind -----------------------------------------------------------------------------


def test_missing_eigenmind_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "eigenmind", None)
    vectors, _ = _clusters([3, 3])
    with pytest.raises(ImportError) as caught:
        select_diverse(_hits(6), vectors, 2)
    assert "graph" in str(caught.value)
    assert "pip install 'myopenfund-rag-engine[graph]'" in str(caught.value)


def test_limit_at_least_the_candidates_returns_them_unchanged(monkeypatch):
    monkeypatch.setitem(sys.modules, "eigenmind", None)  # not needed, so not imported
    hits = _hits(4)
    vectors, _ = _clusters([4])
    assert select_diverse(hits, vectors, 4) == hits
    assert select_diverse(hits, vectors, 9) == hits


def test_importing_the_module_does_not_import_eigenmind():
    code = "import sys, rag_engine.select; print('eigenmind' in sys.modules, 'scipy' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["False", "False"]


def test_no_hits_gives_no_hits(monkeypatch):
    monkeypatch.setitem(sys.modules, "eigenmind", None)
    assert select_diverse([], np.zeros((0, 8)), 3) == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_refuses_a_bad_limit(limit):
    vectors, _ = _clusters([3])
    with pytest.raises(ValueError, match="limit"):
        select_diverse(_hits(3), vectors, limit)


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones(16),  # the query vector alone, not one row per candidate
        np.ones((5, 16)),  # one row short
        np.full((6, 16), np.nan),
        np.zeros((6, 16)),
    ],
)
def test_refuses_vectors_not_aligned_with_the_hits(vectors):
    with pytest.raises(ValueError, match="vectors"):
        select_diverse(_hits(6), vectors, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"role": "spectral_attention"},
        {"threshold": 1.0},
        {"threshold": -0.1},
        {"time_limit": 0},
        {"time_limit": True},
        {"keep_top": -1},
        {"keep_top": True},
        {"keep_top": 1.0},
    ],
)
def test_refuses_bad_options(kwargs):
    vectors, _ = _clusters([3, 3])
    with pytest.raises(ValueError):
        select_diverse(_hits(6), vectors, 2, **kwargs)
