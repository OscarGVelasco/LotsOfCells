"""Internal statistical primitives: transforms and contingency tables."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# --- Transformations used everywhere ---------------------------------------------------

def pseudo_count_arcsin(counts: np.ndarray) -> np.ndarray:
    """`counts + sqrt(counts^2 + 1)` — matches the R pseudocount in entropyScore.R."""
    return counts + np.sqrt(counts * counts + 1)


def asrt(p: np.ndarray) -> np.ndarray:
    """Arcsin square-root transform (Anscombe-style)."""
    return np.arcsin(np.sqrt(np.clip(p, 0, 1)))


def geom_mean(x: np.ndarray) -> float:
    """Geometric mean over the strictly positive entries of ``x``.

    Note: this intentionally diverges from R's literal ``exp(mean(log(x)))``,
    which collapses to 0 whenever **any** entry is 0. In the symmetric
    divergence formula used by `entropyScore`, a zero in
    ``|p * log2(p/q)|`` means ``p[i] == q[i]`` (the two distributions agree
    on cell type ``i``); such a term should contribute *nothing* to the
    divergence — not zero out the entire score.

    The 1-class test makes this critical: random partitions inside a single
    condition often share integer totals after the ``int(sqrt(count_s))``
    crowd sizing, which forces ``p[i] == q[i]`` for any cell type missing
    from both subsamples. With strict R semantics every iteration collapses
    to 0; with this version the geom_mean is taken over the cell types
    that actually disagree.

    If every entry is zero, the divergence really is 0.
    """
    x = np.asarray(x, dtype=float)
    nonzero = x[x > 0]
    if nonzero.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(nonzero))))


# --- Contingency tables --------------------------------------------------------------

def _table(groups: Sequence[str], covariable: Sequence[str]) -> pd.DataFrame:
    """Equivalent of R `table(data.frame(groups, covariable))`."""
    return (
        pd.crosstab(pd.Series(groups, name="groups"),
                    pd.Series(covariable, name="covariable"))
    )


def _ensure_rows(tab: pd.DataFrame, label_order: Sequence[str]) -> pd.DataFrame:
    """Add zero rows for any missing labels and reindex."""
    missing = [l for l in label_order if l not in tab.index]
    if missing:
        z = pd.DataFrame(0, index=missing, columns=tab.columns)
        tab = pd.concat([tab, z])
    return tab.reindex(label_order)


def _ensure_cols(tab: pd.DataFrame, indexes: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in indexes if c not in tab.columns]
    if missing:
        for m in missing:
            tab[m] = 0
    return tab[list(indexes)]
