"""Internal statistical primitives.

Direct ports of the R helpers `cellToGamma`, `cellToGammaOriginal` and
`cellToMontecarlo`. Implementation choices (pseudocounts, transforms) match
the R version exactly so results are comparable.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# --- Transformations used everywhere ---------------------------------------------------

def pseudo_count(counts: np.ndarray) -> np.ndarray:
    """`counts + 0.5` — matches the R pseudocount in lotsOfCells.R."""
    return counts + 0.5


def pseudo_count_arcsin(counts: np.ndarray) -> np.ndarray:
    """`counts + sqrt(counts^2 + 1)` — matches the R pseudocount in entropyScore.R."""
    return counts + np.sqrt(counts * counts + 1)


def asrt(p: np.ndarray) -> np.ndarray:
    """Arcsin square-root transform (Anscombe-style)."""
    return np.arcsin(np.sqrt(np.clip(p, 0, 1)))


def logit(f: np.ndarray) -> np.ndarray:
    return np.log(f / (1 - f))


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


# --- Goodman & Kruskal gamma rank correlation ----------------------------------------

def _ranked_proportions(
    tab: pd.DataFrame,
    label_order: Sequence[str],
    indexes: Sequence[str],
) -> np.ndarray:
    """Rows=label_order, cols=covariables.

    Computes per-covariable proportions then ranks across labels.
    Mirrors `t(apply(dftmp,2,function(row){row/(sum(row)+0.1)}))[labelOrder, indexes]`
    followed by `t(apply(.,1,rank))`.
    """
    tab = _ensure_rows(tab, label_order)
    tab = _ensure_cols(tab, indexes)
    # column-wise proportions: row/(sum(row)+0.1) per column => divide each column by (col_sum+0.1)
    col_sums = tab.values.sum(axis=0) + 0.1  # shape (n_cov,)
    contig = tab.values / col_sums[np.newaxis, :]  # rows = labels in label_order
    # rank within each row across covariables (R: apply(contig_tab,1,rank))
    # 'average' ties to mirror base::rank's default
    ranks = np.apply_along_axis(_rank_avg, 1, contig)
    return ranks  # shape (n_labels, n_cov)


def _rank_avg(x: np.ndarray) -> np.ndarray:
    """Equivalent of R base::rank(x, ties.method='average')."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average over ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=float)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _concordant_discordant(
    ranks: np.ndarray, rank_index: np.ndarray, original: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """For each covariable column, count concordant and discordant pairs.

    If `original=False` (random/null): concordant means
    sign(ranks[i]-ranks[i+1:]) == -1 (matches the R cellToGamma which assumes
    monotonic 1..N and sign always = -1). Discordant counts where
    `ranks[i] != ranks[k]` and sign != -1.

    If `original=True`: compare against the actual rank_index sign pattern.
    """
    n_labels, n_cov = ranks.shape
    nconc = np.zeros(n_cov, dtype=int)
    ndisc = np.zeros(n_cov, dtype=int)
    for i in range(n_labels - 1):
        ri = ranks[i]
        rj = ranks[i + 1 :]  # (rest, n_cov)
        diff_r = ri[np.newaxis, :] - rj  # (rest, n_cov)
        if original:
            idx_diff = rank_index[i] - rank_index[i + 1 :]
            target_sign = np.sign(idx_diff)[:, np.newaxis]  # (rest, 1)
            nconc += np.sum(np.sign(diff_r) == target_sign, axis=0)
            mask_neq = diff_r != 0
            ndisc += np.sum((np.sign(diff_r) != target_sign) & mask_neq, axis=0)
        else:
            nconc += np.sum(np.sign(diff_r) == -1, axis=0)
            mask_neq = diff_r != 0
            ndisc += np.sum((np.sign(diff_r) != -1) & mask_neq, axis=0)
    return nconc, ndisc


def cell_to_gamma(
    covariable: np.ndarray,
    groups: np.ndarray,
    label_order: Sequence[str],
    indexes: Sequence[str],
    cell_crowd: Dict[str, int],
    rank_index: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random null distribution: mix all covariables, then subsample per-group.

    Returns (n_concordant, n_discordant) per covariable column (length n_cov).
    """
    pieces_cov, pieces_grp = [], []
    for label in label_order:
        n = int(cell_crowd[label])
        sample = rng.choice(covariable, size=n, replace=True)
        pieces_cov.append(sample)
        pieces_grp.append(np.repeat(label, n))
    cov = np.concatenate(pieces_cov)
    grp = np.concatenate(pieces_grp)
    tab = _table(grp, cov)
    ranks = _ranked_proportions(tab, label_order, indexes)
    return _concordant_discordant(ranks, rank_index, original=False)


def cell_to_gamma_original(
    covariable: np.ndarray,
    groups: np.ndarray,
    label_order: Sequence[str],
    indexes: Sequence[str],
    cell_crowd: Dict[str, int],
    rank_index: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Original-data subsampling: subsample within each group preserving labels."""
    pieces_cov, pieces_grp = [], []
    for label in label_order:
        n = int(cell_crowd[label])
        pool = covariable[groups == label]
        if len(pool) == 0:
            continue
        replace = n > len(pool)
        sample = rng.choice(pool, size=n, replace=replace)
        pieces_cov.append(sample)
        pieces_grp.append(np.repeat(label, n))
    cov = np.concatenate(pieces_cov)
    grp = np.concatenate(pieces_grp)
    tab = _table(grp, cov)
    ranks = _ranked_proportions(tab, label_order, indexes)
    return _concordant_discordant(ranks, rank_index, original=True)


# --- Monte Carlo for 2-condition fold-change -----------------------------------------

def _proportions_from_table(
    tab: pd.DataFrame,
    label_order: Sequence[str],
    indexes: Sequence[str],
    pseudo: bool = True,
) -> np.ndarray:
    """`pseudo_count(tab)` then column-wise proportions, indexed by label_order/indexes."""
    tab = _ensure_rows(tab, label_order)
    tab = _ensure_cols(tab, indexes)
    vals = tab.values.astype(float)
    if pseudo:
        vals = pseudo_count(vals)
    col_sums = vals.sum(axis=0) + 1.0
    return vals / col_sums[np.newaxis, :]  # (n_labels, n_cov)


def cell_to_montecarlo(
    covariable: np.ndarray,
    groups: np.ndarray,
    label_order: Sequence[str],
    indexes: Sequence[str],
    cell_crowd: Union[Dict[str, int], Dict[str, List[int]]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mixed-pool fold change, original-resampled fold change).

    Both are arrays of length len(indexes), holding
    log2( asrt(p1) / asrt(p2) ).
    """
    def _build_mixed(crowd_for_label):
        if isinstance(crowd_for_label, (list, np.ndarray)):
            sizes = np.asarray(crowd_for_label, dtype=int)
            return np.concatenate(
                [rng.choice(covariable, size=int(s), replace=True) for s in sizes]
            )
        return rng.choice(covariable, size=int(crowd_for_label), replace=True)

    def _build_orig(crowd_for_label, label):
        pool = covariable[groups == label]
        if len(pool) == 0:
            return np.array([], dtype=covariable.dtype)
        if isinstance(crowd_for_label, (list, np.ndarray)):
            sizes = np.asarray(crowd_for_label, dtype=int)
            return np.concatenate(
                [rng.choice(pool, size=int(s), replace=True) for s in sizes]
            )
        n = int(crowd_for_label)
        return rng.choice(pool, size=n, replace=True)

    mixed_cov, mixed_grp, orig_cov, orig_grp = [], [], [], []
    for label in label_order:
        cm = _build_mixed(cell_crowd[label])
        co = _build_orig(cell_crowd[label], label)
        mixed_cov.append(cm)
        mixed_grp.append(np.repeat(label, len(cm)))
        orig_cov.append(co)
        orig_grp.append(np.repeat(label, len(co)))

    mixed_tab = _table(np.concatenate(mixed_grp), np.concatenate(mixed_cov))
    orig_tab = _table(np.concatenate(orig_grp), np.concatenate(orig_cov))

    p_mixed = _proportions_from_table(mixed_tab, label_order, indexes, pseudo=True)
    p_orig = _proportions_from_table(orig_tab, label_order, indexes, pseudo=True)

    fc_mixed = np.log2(asrt(p_mixed[0]) / asrt(p_mixed[1]))
    fc_orig = np.log2(asrt(p_orig[0]) / asrt(p_orig[1]))
    return fc_mixed, fc_orig
