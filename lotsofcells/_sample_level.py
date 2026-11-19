"""Sample-level (patient-level) statistics shared by the tests.

The unit of independence is the sample, not the cell. Each sample's
composition is computed from ALL of its cells; samples are then combined
with optional weights. The observed statistic, the permutation null
(sample labels reshuffled across groups) and the bootstrap CI (samples
resampled within groups) all use the same deterministic formula, so the
p-value compares like with like.

Weights: `sqrt` gives larger sections a larger — but sub-linear and capped —
say; `equal` gives every sample one vote. Neither lets section size (tissue
area) drive the group estimate the way pooling all cells does.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial, prod
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._stats import pseudo_count_arcsin

METHODS = ("auto", "sample_permutation", "montecarlo")
ALTERNATIVES = ("two-sided", "greater", "less")


# ---------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------

def resolve_method(method: str, metadata: pd.DataFrame, main_variable: str,
                   sample_id: Optional[str], label_order: Sequence[str],
                   verbose: bool = True) -> str:
    """Return the concrete method to run ("sample_permutation" or "montecarlo").

    `auto` picks the sample-level test when `sample_id` is given and every
    group has at least two samples; otherwise it falls back to Monte Carlo.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "montecarlo":
        return "montecarlo"
    if sample_id is None:
        if method == "sample_permutation":
            raise ValueError("method='sample_permutation' requires `sample_id`.")
        if verbose:
            print("method='auto': no sample_id → using 'montecarlo' (cell-level).")
        return "montecarlo"

    n_samples = (
        metadata.assign(_g=metadata[main_variable].astype(str),
                        _s=metadata[sample_id].astype(str))
        .groupby("_g")["_s"].nunique()
    )
    too_few = [l for l in label_order if n_samples.get(l, 0) < 2]
    if too_few:
        if method == "sample_permutation":
            raise ValueError(
                f"method='sample_permutation' needs ≥ 2 samples per group; "
                f"too few in {too_few}."
            )
        if verbose:
            print(f"method='auto': < 2 samples in {too_few} → using 'montecarlo'.")
        return "montecarlo"
    if verbose:
        print("method='auto' → 'sample_permutation' (sample-level test).")
    return "sample_permutation"


# ---------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------

def sample_count_matrix(groups: np.ndarray, samples: np.ndarray,
                        covariable: np.ndarray, label_order: Sequence[str],
                        indexes: Sequence[str]):
    """Per-sample count matrix.

    Returns ``(counts, sample_names, group_idx, n_s)`` where ``counts`` is
    ``(n_samples, n_cov)``, ``group_idx[i]`` is the position of sample i's
    group in ``label_order`` and ``n_s`` is cells per sample. Samples are
    ordered by group, so the observed labelling is ``group_idx`` itself.
    """
    label_order = list(label_order)
    df = pd.DataFrame({"g": np.asarray(groups).astype(str),
                       "s": np.asarray(samples).astype(str),
                       "c": np.asarray(covariable).astype(str)})
    groups_per_sample = df.groupby("s")["g"].nunique()
    shared = groups_per_sample[groups_per_sample > 1].index.tolist()
    if shared:
        raise ValueError(
            f"Each sample must belong to exactly one group; these span several: "
            f"{shared[:5]}"
        )
    sample_group = df.groupby("s")["g"].first()
    order = sorted(sample_group.index,
                   key=lambda s: (label_order.index(sample_group[s]), s))
    counts = (pd.crosstab(df["s"], df["c"])
                .reindex(index=order, columns=list(indexes), fill_value=0))
    group_idx = np.array([label_order.index(sample_group[s]) for s in order])
    for k, label in enumerate(label_order):
        if (group_idx == k).sum() < 2:
            raise ValueError(f"Group {label!r} has fewer than 2 samples.")
    values = counts.to_numpy(dtype=float)
    return values, np.array(order), group_idx, values.sum(axis=1)


def sample_weights(n_s: np.ndarray, scheme: str = "sqrt",
                   cap: Optional[float] = 2.0) -> np.ndarray:
    """Per-sample weights.

    ``sqrt``: √n_s, capped at ``cap × median(√n)`` so one huge section
    cannot dominate. ``equal``: one vote per sample.
    """
    n_s = np.asarray(n_s, dtype=float)
    if scheme == "equal":
        return np.ones_like(n_s)
    if scheme != "sqrt":
        raise ValueError(f"weights must be 'sqrt' or 'equal', got {scheme!r}")
    w = np.sqrt(n_s)
    if cap is not None:
        w = np.minimum(w, cap * np.median(w))
    return w


def per_sample_props(counts: np.ndarray, kind: str = "loc") -> np.ndarray:
    """Per-sample composition using the package's pseudocounts.

    ``loc``     — ``(c + 1/4) / (n + 1/2)``. With this pseudocount
                  ``asin(sqrt(p))`` has no first-order bias in the sample size
                  (bias ∝ (a/2 − 1/8)/n for a pseudocount a), so samples of
                  different size (e.g. tissue area) are not shifted relative to
                  each other. A fixed 0.5 made small samples look enriched in
                  rare cell types; see simulation_benchmark/plasmode/pseudocount.
    ``entropy`` — ``c + sqrt(c² + 1)`` row-normalised, as in ``entropy_score``.
    """
    if kind == "loc":
        return (counts + 0.25) / (counts.sum(axis=1, keepdims=True) + 0.5)
    if kind == "entropy":
        v = pseudo_count_arcsin(counts)
        return v / v.sum(axis=1, keepdims=True)
    raise ValueError(f"Unknown kind {kind!r}")


# ---------------------------------------------------------------------
# Relabellings (permutation null)
# ---------------------------------------------------------------------

def n_relabellings(group_idx: np.ndarray) -> int:
    """Distinct ways to assign the samples to groups with the same sizes."""
    sizes = np.bincount(group_idx)
    return factorial(int(sizes.sum())) // prod(factorial(int(s)) for s in sizes)


def _exact_assignments(group_idx: np.ndarray) -> np.ndarray:
    """Every distinct assignment of samples to groups (same group sizes).

    The observed labelling is always row 0.
    """
    sizes = np.bincount(group_idx)
    n = len(group_idx)

    if len(sizes) == 2:
        # Vectorised fast path: choose the positions of group 0.
        chosen = np.array(list(combinations(range(n), int(sizes[0]))), dtype=np.int32)
        out = np.ones((len(chosen), n), dtype=np.int16)
        out[np.arange(len(chosen))[:, np.newaxis], chosen] = 0
    else:
        rows = []

        def rec(remaining: Tuple[int, ...], g: int, current: np.ndarray):
            if g == len(sizes) - 1:
                current = current.copy()
                current[list(remaining)] = g
                rows.append(current)
                return
            for chosen in combinations(remaining, int(sizes[g])):
                nxt = current.copy()
                nxt[list(chosen)] = g
                taken = set(chosen)
                rec(tuple(i for i in remaining if i not in taken), g + 1, nxt)

        rec(tuple(range(n)), 0, np.full(n, -1, dtype=np.int16))
        out = np.vstack(rows)
    # Samples are ordered by group, so the first combination generated is
    # the observed labelling; assert it to be safe.
    if not np.array_equal(out[0], group_idx):
        obs_row = np.where((out == group_idx).all(axis=1))[0][0]
        out[[0, obs_row]] = out[[obs_row, 0]]
    return out


def relabellings(group_idx: np.ndarray, max_exact: int, permutations: int,
                 rng: np.random.Generator):
    """Assignment matrix for the null.

    Returns ``(assign, exact)``. When the number of distinct relabellings is
    ≤ ``max_exact`` every one is enumerated (``exact=True``, observed
    labelling included as row 0). Otherwise ``permutations`` random
    shuffles of the labels are drawn (``exact=False``; the observed
    labelling is not included and is accounted for by the +1 in the
    p-value).
    """
    if n_relabellings(group_idx) <= max_exact:
        return _exact_assignments(group_idx), True
    assign = np.vstack([rng.permutation(group_idx) for _ in range(int(permutations))])
    return assign.astype(np.int16), False


# ---------------------------------------------------------------------
# Vectorised group statistics
# ---------------------------------------------------------------------

def group_weighted_means(assign: np.ndarray, values: np.ndarray,
                         weights: np.ndarray, n_groups: int,
                         chunk: int = 20_000) -> np.ndarray:
    """Weighted mean of per-sample ``values`` per group, for every assignment.

    ``assign`` is ``(N, n_samples)``; ``values`` is ``(n_samples, k)``.
    Returns ``(n_groups, N, k)``.
    """
    assign = np.atleast_2d(assign)
    n_rows = assign.shape[0]
    out = np.empty((n_groups, n_rows, values.shape[1]))
    for start in range(0, n_rows, chunk):
        block = assign[start:start + chunk]
        for g in range(n_groups):
            m = (block == g) * weights[np.newaxis, :]
            out[g, start:start + chunk] = (m @ values) / m.sum(axis=1, keepdims=True)
    return out


def bootstrap_indices(group_idx: np.ndarray, n_bootstrap: int,
                      rng: np.random.Generator) -> np.ndarray:
    """``(n_bootstrap, n_samples)`` sample indices resampled within each group.

    Position i keeps the group of sample i, so the labelling is unchanged;
    only which samples fill each group changes. Weights travel with samples.
    """
    n = len(group_idx)
    out = np.empty((int(n_bootstrap), n), dtype=int)
    for g in np.unique(group_idx):
        pos = np.where(group_idx == g)[0]
        out[:, pos] = rng.choice(pos, size=(int(n_bootstrap), len(pos)), replace=True)
    return out


def bootstrap_group_means(boot_idx: np.ndarray, group_idx: np.ndarray,
                          values: np.ndarray, weights: np.ndarray,
                          n_groups: int) -> np.ndarray:
    """Weighted group means for each bootstrap resample. Returns ``(n_groups, B, k)``."""
    out = np.empty((n_groups, boot_idx.shape[0], values.shape[1]))
    for g in range(n_groups):
        pos = np.where(group_idx == g)[0]
        idx = boot_idx[:, pos]                      # (B, n_g)
        w = weights[idx]                            # (B, n_g)
        v = values[idx]                             # (B, n_g, k)
        out[g] = (w[..., np.newaxis] * v).sum(axis=1) / w.sum(axis=1, keepdims=True)
    return out


def gamma_per_column(values: np.ndarray, rank_assign: np.ndarray,
                     chunk: int = 2_000) -> np.ndarray:
    """Goodman–Kruskal gamma between group rank and each column of ``values``.

    ``values`` is ``(n_samples, k)``; ``rank_assign`` is ``(N, n_samples)``
    of ordinal group indices. Pairs tied in either variable are excluded.
    Returns ``(N, k)``.
    """
    rank_assign = np.atleast_2d(rank_assign)
    s_v = np.sign(values[:, np.newaxis, :] - values[np.newaxis, :, :])  # (n, n, k)
    abs_v = np.abs(s_v)
    out = np.empty((rank_assign.shape[0], values.shape[1]))
    for start in range(0, rank_assign.shape[0], chunk):
        r = rank_assign[start:start + chunk].astype(float)
        s_r = np.sign(r[:, :, np.newaxis] - r[:, np.newaxis, :])         # (b, n, n)
        num = np.einsum("bij,ijk->bk", s_r, s_v)
        den = np.einsum("bij,ijk->bk", np.abs(s_r), abs_v)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:start + chunk] = np.where(den > 0, num / den, 0.0)
    return out


def kendall_gamma_per_column(values: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Single-labelling convenience wrapper around :func:`gamma_per_column`."""
    return gamma_per_column(values, np.asarray(ranks)[np.newaxis, :])[0]


# ---------------------------------------------------------------------
# p-values
# ---------------------------------------------------------------------

def perm_pvalue(null: np.ndarray, obs, exact: bool,
                alternative: str = "two-sided") -> np.ndarray:
    """Permutation p-value.

    Exact enumeration already contains the observed labelling, so
    ``p = count / N`` (counted once). Random relabellings do not, so
    ``p = (count + 1) / (N + 1)``.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}")
    null = np.asarray(null, dtype=float)
    obs = np.asarray(obs, dtype=float)
    tol = 1e-10 * np.maximum(1.0, np.abs(obs))
    if alternative == "two-sided":
        hits = np.abs(null) >= np.abs(obs) - tol
    elif alternative == "greater":
        hits = null >= obs - tol
    else:
        hits = null <= obs + tol
    count = hits.sum(axis=0)
    n = null.shape[0]
    return count / n if exact else (count + 1) / (n + 1)


def maxt_adjust(null: np.ndarray, obs, exact: bool,
                alternative: str = "two-sided") -> np.ndarray:
    """Westfall–Young step-down maxT adjusted p-values (family-wise error rate).

    Every statistic is standardised by the SD of its own permutation null,
    ``z_j = obs_j / SD(null_j)``, so that cell types with different spreads
    are comparable. The adjusted p-value of the k-th most extreme type is the
    share of relabellings whose largest standardised statistic over that type
    and every less extreme one reaches it (made monotone). The joint null keeps
    the dependence between cell types (compositions sum to one), which is what
    lets small cohorts pass, where Benjamini–Hochberg on per-type p-values
    cannot get below the per-type floor times the number of types.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}")
    null = np.asarray(null, dtype=float)
    obs = np.asarray(obs, dtype=float)
    sd = null.std(axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.inf)   # constant null: z = 0
    z, zn = np.nan_to_num(obs / sd), np.nan_to_num(null / sd)
    if alternative == "two-sided":
        z, zn = np.abs(z), np.abs(zn)
    elif alternative == "less":
        z, zn = -z, -zn
    order = np.argsort(-z, kind="stable")                  # most extreme first
    # u[b, i]: max over the i-th and every less extreme type in relabelling b
    u = np.maximum.accumulate(zn[:, order][:, ::-1], axis=1)[:, ::-1]
    tol = 1e-10 * np.maximum(1.0, np.abs(z[order]))
    count = (u >= z[order] - tol).sum(axis=0)
    n = null.shape[0]
    p = count / n if exact else (count + 1) / (n + 1)
    p = np.minimum(np.maximum.accumulate(p), 1.0)
    out = np.empty_like(p)
    out[order] = p
    return out


def report_design(label_order: Sequence[str], group_idx: np.ndarray,
                  n_perms: int, exact: bool, p_min: float, verbose: bool,
                  one_sided_hint: bool = True) -> None:
    """Print the cohort design and the smallest achievable p-value."""
    if not verbose:
        return
    sizes = ", ".join(f"{l}: {int((group_idx == k).sum())}"
                      for k, l in enumerate(label_order))
    kind = "exact enumeration" if exact else "random relabellings"
    print(f"Sample-level permutation test — samples per group ({sizes}); "
          f"{n_perms:,} {kind}; smallest achievable p = {p_min:.4g}")
    if p_min > 0.05:
        hint = (" (or a one-sided `alternative` if the direction was pre-specified)"
                if one_sided_hint else "")
        print("  ⚠ With this cohort size the test cannot reach p < 0.05; "
              f"more samples per group are needed{hint}.")


def perm_pmin(n_perms: int, exact: bool, mirror_tie: bool = False) -> float:
    """Smallest p-value the permutation test can return.

    With equal group sizes and a symmetric (or two-sided) statistic, the
    A↔B swap of the observed labelling always ties with it.
    """
    if not exact:
        return 1.0 / (n_perms + 1)
    return (2.0 if mirror_tie else 1.0) / n_perms
