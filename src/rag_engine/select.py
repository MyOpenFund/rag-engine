"""Pick ``limit`` passages among the fused hybrid candidates so they cover distinct themes.

:func:`select_diverse` runs **Supported Diversity**, one of the three mechanisms of eigenmind
0.2 (``eigenmind.supported_diversity``), on the cosine graph of the candidates:

1. Graph. The candidate vectors are put to unit length (``eigenmind.normalize_embeddings``,
   ``center=False``: no mean-centring, no principal direction removed), their cosines
   computed (``eigenmind.cosine_matrix``) and every pair whose cosine is strictly above
   ``threshold`` joined by an edge weighted by that cosine (``eigenmind.threshold_graph``).
   ``threshold`` defaults to 0.65, eigenmind's own ``SIMILARITY_THRESHOLD``. A threshold
   graph was chosen over a k-NN graph because an edge then means one thing at every query
   ("these two passages are near-duplicates"), whatever the number of candidates.
2. Selection. ``supported_diversity(W, utilities, budget=limit, ...)`` solves the budgeted
   maximum-weight stable set exactly (HiGHS MILP through ``scipy.optimize.milp``,
   clique-strengthened, with an LP bound): at most ``limit`` candidates, no two of them
   joined by an edge, maximising the sum of their utilities. The utilities are the fused
   RRF scores (``Hit.score``), so among near-duplicates the best-ranked one is kept.
   ``min_degree``, ``relaxation``, ``solver`` and ``time_limit`` are passed through with
   eigenmind's defaults (``0.0``, ``"cliques"``, ``"milp"``, ``10.0`` seconds).
3. Fill-up. Supported Diversity only returns *supported* candidates (weighted degree > 0):
   a candidate with no neighbour above the threshold (an isolate) is never chosen by it,
   and the stable set may hold fewer than ``limit`` candidates (fewer themes than
   ``limit``). The result is then filled up with the best remaining candidates by fused
   rank, so it always holds ``min(limit, len(hits))`` hits. When eigenmind raises
   ``RuntimeError`` (its LP or feasibility checks failed), the result is the first
   ``limit`` candidates by fused rank.

The hits come back in their fused order (best first). The selection is deterministic for
the same input: no randomness, and HiGHS returns the same optimum for the same problem.
One exception: a MILP stopped by ``time_limit`` returns its best incumbent at that time,
which depends on the machine (a 50-candidate problem is solved in milliseconds).

It plugs into :func:`rag_engine.local.hybrid_search` as ``selector=select_diverse``: the
search passes the stored vector of every candidate, one row per hit in the same order.

To tune (evaluated in B5, on real collections): ``threshold``, since real embeddings are
anisotropic (their cosines crowd in a narrow band, so at 0.65 most candidates may form one
clique, the stable set holds one or two of them and the fill-up gives back the fused order);
and mean-centring or removal of the top principal directions before the graph
(``eigenmind.normalize_embeddings(center=..., remove_top=...)``, not exposed yet). Ties in
the fused scores (a dense rank r and an FTS-only rank r) are broken by HiGHS: stable for one
HiGHS version, not specified.

eigenmind is an optional dependency (the ``graph`` extra); it is imported only when a
selection is needed, so ``limit >= len(hits)`` works without it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .search import Hit

__all__ = ["DEFAULT_THRESHOLD", "ROLES", "select_diverse"]

#: Cosine above which two candidates are joined (eigenmind's ``SIMILARITY_THRESHOLD``).
DEFAULT_THRESHOLD = 0.65
#: The eigenmind mechanisms :func:`select_diverse` can run.
ROLES = ("supported_diversity",)

_INSTALL_HINT = (
    "select_diverse needs eigenmind, which the graph extra installs: "
    "pip install 'myopenfund-rag-engine[graph]'"
)


def _check_limit(limit: object) -> int:
    if type(limit) is not int or limit < 1:
        raise ValueError(f"limit must be an integer >= 1, got {limit!r}")
    return limit


def _check_vectors(vectors: object, count: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=float)
    if array.ndim != 2 or array.shape[0] != count:
        raise ValueError(f"vectors must hold one row per hit, shape ({count}, dim), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("vectors must be finite")
    if count and np.any(np.linalg.norm(array, axis=1) == 0):
        raise ValueError("vectors must not hold a zero row")
    return array


def select_diverse(
    hits: list[Hit],
    vectors: np.ndarray,
    limit: int,
    *,
    role: str = "supported_diversity",
    threshold: float = DEFAULT_THRESHOLD,
    min_degree: float = 0.0,
    relaxation: str = "cliques",
    solver: str = "milp",
    time_limit: float = 10.0,
) -> list[Hit]:
    """Up to ``limit`` of ``hits`` that cover distinct themes, in their fused order.

    ``hits`` are the fused candidates, best first; ``vectors`` holds one embedding per hit,
    in the same order (shape ``(len(hits), dim)``). See the module docstring for the graph,
    the selection and the fill-up.

    - ``limit >= len(hits)`` returns ``list(hits)`` unchanged, without importing eigenmind;
      no hits gives ``[]``.
    - ``role``: the eigenmind mechanism; only ``"supported_diversity"`` (:data:`ROLES`).
    - ``threshold``: cosine strictly above which two candidates are near-duplicates, in
      ``[0, 1)``.
    - ``min_degree``, ``relaxation``, ``solver``, ``time_limit``: passed to
      ``eigenmind.supported_diversity`` (eigenmind's defaults).

    Raises ``ValueError`` for a ``limit`` that is not an integer >= 1, ``vectors`` not
    aligned with ``hits`` (including the query vector alone), non-finite or with a zero row,
    or a bad option; ``ImportError`` naming the ``graph`` extra when eigenmind is missing.
    """
    _check_limit(limit)
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must lie in [0, 1), got {threshold!r}")
    if not (isinstance(time_limit, (int, float)) and time_limit > 0):
        raise ValueError(f"time_limit must be a positive number of seconds, got {time_limit!r}")
    array = _check_vectors(vectors, len(hits))
    if limit >= len(hits):
        return list(hits)

    try:
        import eigenmind
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error

    unit = eigenmind.normalize_embeddings(array, center=False)
    graph = eigenmind.threshold_graph(eigenmind.cosine_matrix(unit), threshold)
    utilities = np.array([hit.score for hit in hits], dtype=float)
    try:
        result = eigenmind.supported_diversity(
            graph,
            utilities,
            budget=limit,
            min_degree=min_degree,
            relaxation=relaxation,
            solver=solver,
            time_limit=time_limit,
        )
        chosen = {int(i) for i in result["selected_indices"][:limit]}
    except RuntimeError:
        chosen = set()
    return _fill(hits, chosen, limit)


def _fill(hits: Sequence[Hit], chosen: set[int], limit: int) -> list[Hit]:
    """``chosen`` plus the best other ranks up to ``limit``, in fused order."""
    for rank in range(len(hits)):
        if len(chosen) >= limit:
            break
        chosen.add(rank)
    return [hits[rank] for rank in sorted(chosen)]
