"""Symmetric divergence (KL-based) entropy score, plus the 1-class abundance test."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._parallel import run_permutations
from ._stats import (
    _ensure_cols,
    _ensure_rows,
    _table,
    geom_mean,
    pseudo_count_arcsin,
)
from ._utils import get_metadata


def _proportions_arcsin(
    tab: pd.DataFrame, label_order: Sequence[str], indexes: Sequence[str]
) -> np.ndarray:
    """Per-group proportions across covariables (each row sums to 1).

    Mirrors the R `entropyScore` normalisation. Note: in R the *random*
    contig table is built from `data.frame(covariable, groups)` (covariable
    first) so `table()` produces shape (ncov, ngroups) and the code applies
    `apply(., 2, row/sum(row))` followed by `t()` — which is mathematically
    equivalent to row-normalising on a (ngroups, ncov) matrix. Since
    `pd.crosstab(groups, covariable)` already returns (ngroups, ncov) here,
    a single function works for both observed and random tables.
    """
    tab = _ensure_rows(tab, label_order)
    tab = _ensure_cols(tab, indexes)
    vals = pseudo_count_arcsin(tab.values.astype(float))
    row_sums = vals.sum(axis=1, keepdims=True)
    return vals / row_sums

# Deprecated
#def _distance_surprise(p: np.ndarray, q: np.ndarray) -> float:
#    return geom_mean(np.abs(p * np.log2(p / q))) + geom_mean(np.abs(q * np.log2(q / p)))

def _distance_surprise(p: np.ndarray, q: np.ndarray) -> float:
    return np.mean(np.sum(np.tanh(np.abs(np.log2(p / q))) * (np.abs(p-q)/(p+q))))


def entropy_score(
    sc_object,
    main_variable: str,
    subtype_variable: str,
    label_order: Sequence[str],
    sample_id: Optional[str] = None,
    permutations: int = 1000,
    seed: Optional[int] = None,
    n_cores: Optional[int] = None,
    table: Optional[str] = None,
    plot: bool = True,
    verbose: bool = True,
    pdf_file: Optional[str] = None,
):
    """Symmetric divergence score for global proportion dysregulation between 2 groups.

    Returns a `pandas.Series` with per-covariable relative entropies plus the
    summary fields (``entropy_score``, ``p.val``, ``mean.random.entropy``,
    ``sd.random.entropy``).

    If ``len(label_order) == 1``, runs the 1-class permutation test on
    ``sample_id`` (analogue of the R `oneClassTest`) and returns a small
    summary dict instead.
    """
    metadata = get_metadata(sc_object, table=table)

    main_vals = metadata[main_variable].astype(str).to_numpy()
    if not all(l in np.unique(main_vals) for l in label_order):
        missing = [l for l in label_order if l not in np.unique(main_vals)]
        raise ValueError(f"Some groups in label_order not in data: {missing}")

    metadata = metadata.loc[np.isin(main_vals, list(label_order))].copy()
    groups = metadata[main_variable].astype(str).to_numpy()
    covariable = metadata[subtype_variable].astype(str).to_numpy()

    if len(label_order) == 0:
        raise ValueError("label_order must be specified.")

    if len(label_order) == 1:
        if sample_id is None:
            raise ValueError("In 1-class mode you must specify `sample_id`.")
        return _one_class_test(
            metadata,
            sample_id,
            covariable,
            permutations,
            seed=seed,
            n_cores=n_cores,
            plot=plot,
            verbose=verbose,
            pdf_file=pdf_file,
        )

    if len(label_order) > 2:
        raise ValueError(
            f"Only 2 labels are allowed for entropy estimation, got "
            f"{len(label_order)}: {label_order}"
        )

    if verbose:
        print(
            "Computing entropy proportion over covariables for groups: "
            f"{label_order[0]} vs {label_order[1]}"
        )
    obs_tab = _table(groups, covariable)
    indexes = list(obs_tab.columns)
    contig = _proportions_arcsin(obs_tab, label_order, indexes)

    # Per-covariable relative entropies (matches R apply over rows... in the R it's
    # apply(contig_tab, 1, function(x) abs(log2((x[1]*log2(x[2]))/(x[1]*log2(x[1])))));
    # since R contig_tab is rows=labels, columns=covariables, apply over rows iterates
    # COLUMNS — so we replicate by iterating columns here)
    rel_entropies = np.empty(len(indexes))
    for j in range(len(indexes)):
        x = contig[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_entropies[j] = np.abs(
                np.log2((x[0] * np.log2(x[1])) / (x[0] * np.log2(x[0])))
            )

    obs_score = _distance_surprise(contig[0], contig[1])

    # Build cell-crowd for null sampling
    if sample_id is not None:
        samples = metadata[sample_id].astype(str).to_numpy()
        n_per_sample = (
            pd.crosstab(pd.Series(groups), pd.Series(samples)).reindex(label_order)
        )
        n_per_sample = np.sqrt(n_per_sample)
        cell_crowd = {}
        for cond in label_order:
            row = n_per_sample.loc[cond]
            cell_crowd[cond] = list(row[row != 0].astype(int).to_numpy())
    else:
        counts = pd.Series(groups).value_counts().to_dict()
        cell_crowd = {l: int(round(np.sqrt(counts.get(l, 0)))) for l in label_order}

    if verbose:
        print(f"Starting Monte-Carlo simulation with n. permutations: {permutations}")

    def _perm_chunk(chunk_size, seed_seq):
        rng = np.random.default_rng(seed_seq)
        out = np.empty((chunk_size, 1))
        for i in range(chunk_size):
            pieces_cov, pieces_grp = [], []
            for label in label_order:
                crowd = cell_crowd[label]
                if isinstance(crowd, list):
                    for n in crowd:
                        s = rng.choice(covariable, size=int(n), replace=True)
                        pieces_cov.append(s)
                        pieces_grp.append(np.repeat(label, len(s)))
                else:
                    s = rng.choice(covariable, size=int(crowd), replace=True)
                    pieces_cov.append(s)
                    pieces_grp.append(np.repeat(label, len(s)))
            cov = np.concatenate(pieces_cov)
            grp = np.concatenate(pieces_grp)
            rand_tab = _table(grp, cov)
            p = _proportions_arcsin(rand_tab, label_order, indexes)
            out[i, 0] = _distance_surprise(p[0], p[1])
        return out

    null_scores = run_permutations(
        _perm_chunk, permutations,
        n_cores=n_cores, seed=seed, verbose=verbose,
    )[:, 0]

    p_val = float((null_scores >= obs_score).sum() / permutations)

    if plot:
        try:
            _plot_entropy(
                contig=contig,
                indexes=indexes,
                label_order=label_order,
                obs_score=obs_score,
                null_scores=null_scores,
                p_val=p_val,
                subtype_variable=subtype_variable,
                pdf_file=pdf_file,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"(Plot skipped: {e})")

    out = pd.Series(rel_entropies, index=indexes)
    out["entropy_score"] = obs_score
    out["p.val"] = p_val
    out["mean.random.entropy"] = float(null_scores.mean())
    out["sd.random.entropy"] = float(null_scores.std(ddof=1))
    return out


def _plot_entropy(
    contig, indexes, label_order, obs_score, null_scores, p_val,
    subtype_variable, pdf_file=None,
):
    import matplotlib.pyplot as plt
    from ._utils import save_to_pdf

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    n = len(indexes)
    width = 0.35
    x = np.arange(n)
    palette = ["#9ECAE1", "#3182BD"]
    for i, label in enumerate(label_order):
        ax.bar(x + (i - 0.5) * width, contig[i], width, label=label, color=palette[i])
    ax.set_xticks(x)
    ax.set_xticklabels(indexes, rotation=45, ha="right")
    ax.set_ylabel("proportion")
    ax.set_title(
        f"Symmetric Divergence Score: {obs_score:.3f} | p.val.adj: {p_val:.3f}"
    )
    ax.legend(title=f"Class: {subtype_variable}")

    ax2 = axes[1]
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.1, 0.1, size=len(null_scores))
    ax2.scatter(jitter, null_scores, color="#D5BADB", alpha=0.5, s=15)
    ax2.axhline(np.median(null_scores), color="#86608E", lw=1)
    ax2.scatter([0], [obs_score], color="#F08080", s=80, zorder=5)
    ax2.set_xticks([])
    ax2.set_ylabel("symmetric divergence")
    plt.tight_layout()
    save_to_pdf(fig, pdf_file)


def _one_class_test(
    metadata,
    sample_id,
    covariable,
    permutations,
    seed=None,
    n_cores=None,
    plot=True,
    verbose=True,
    pdf_file=None,
):
    """Permutation test for sample-level proportion variation in a single condition.

    Departs from R's `oneClassTest` in one important way: the null draws each
    sample's cells from THAT SAMPLE'S own covariable distribution, not from
    the global pool. The R version sampled every cell from the global pool,
    which collapses both random pseudo-groups onto the same global
    distribution and produces a null that is essentially zero — so the user
    never observes any spread no matter how heterogeneous the real samples
    are. Drawing from per-sample pools preserves real per-sample structure
    and lets random partitions of those samples yield a null distribution
    whose spread reflects across-sample heterogeneity, which is what this
    test is meant to assess.
    """
    samples = metadata[sample_id].astype(str).to_numpy()
    obs_tab = _table(samples, covariable)
    indexes = list(obs_tab.columns)
    n_per_sample = pd.Series(samples).value_counts()
    sqrt_n = np.sqrt(n_per_sample)
    sqrt_n[sqrt_n == 0] = 10
    cell_crowd = sqrt_n.to_dict()

    # Build a per-sample pool of covariable values (preserves the real cell
    # composition of each sample for the null draw).
    sample_pools = {
        s: covariable[samples == s] for s in n_per_sample.index
    }
    unique_samples = list(n_per_sample.index)
    if len(unique_samples) < 2:
        raise ValueError(
            "1-class entropy test needs at least 2 samples in `sample_id`."
        )
    n_g1 = max(1, round(len(unique_samples) / 2))

    # Mirror R's iteration count: seq(100) * seq(permutations/10) = 10*perms.
    n_iter = max(int(permutations) * 10, 100)
    if verbose:
        print(f"Starting 1-class Monte-Carlo simulation: {n_iter} iterations")

    def _perm_chunk(chunk_size, seed_seq):
        rng = np.random.default_rng(seed_seq)
        out = np.empty((chunk_size, 1))
        for i in range(chunk_size):
            perm = rng.permutation(len(unique_samples))
            g1 = [unique_samples[k] for k in perm[:n_g1]]
            g2 = [unique_samples[k] for k in perm[n_g1:]]
            pieces_cov, pieces_grp = [], []
            for s in g1:
                n = max(int(cell_crowd[s]), 1)
                pool = sample_pools[s]
                if len(pool) == 0:
                    continue
                draw = rng.choice(pool, size=n, replace=True)
                pieces_cov.append(draw)
                pieces_grp.append(np.repeat("group1", n))
            for s in g2:
                n = max(int(cell_crowd[s]), 1)
                pool = sample_pools[s]
                if len(pool) == 0:
                    continue
                draw = rng.choice(pool, size=n, replace=True)
                pieces_cov.append(draw)
                pieces_grp.append(np.repeat("group2", n))
            cov = np.concatenate(pieces_cov)
            grp = np.concatenate(pieces_grp)
            rand_tab = _table(grp, cov)
            p = _proportions_arcsin(rand_tab, ["group1", "group2"], indexes)
            out[i, 0] = _distance_surprise(p[0], p[1])
        return out

    null_scores = run_permutations(
        _perm_chunk, n_iter,
        n_cores=n_cores, seed=seed, verbose=verbose,
    )[:, 0]

    mean_null = float(null_scores.mean())
    sd_null = float(null_scores.std(ddof=1))
    median_null = float(np.median(null_scores))
    cv = float(sd_null / mean_null * 100) if mean_null > 0 else float("inf")
    if median_null > 0:
        relative_iqr = float(
            (np.percentile(null_scores, 75) - np.percentile(null_scores, 25))
            / median_null
        )
    else:
        relative_iqr = float("nan")
    if cv <= 35:
        variation = "Low"
    elif cv <= 50:
        variation = "Medium"
    else:
        variation = "High"

    if verbose:
        print(f"Coefficient of Variation: {cv:.2f} %")
        print(f"Variation across samples is considered: {variation}")
        print(f"Relative IQR: {relative_iqr:.3f}")

    if plot:
        try:
            import matplotlib.pyplot as plt
            from ._utils import save_to_pdf

            fig, ax = plt.subplots(figsize=(3.5, 5))
            jitter_rng = np.random.default_rng(seed if seed is not None else 0)
            jitter = jitter_rng.uniform(-0.1, 0.1, size=len(null_scores))
            ax.scatter(jitter, null_scores, color="#D5BADB", alpha=0.5, s=15)
            ax.axhline(median_null, color="#86608E", lw=1)
            ax.set_xlim(-0.5, 0.5)
            lo = float(min(0.0, null_scores.min()))
            hi = float(null_scores.max())
            pad = max(1e-3, 0.1 * (hi - lo))
            ax.set_ylim(lo, hi + pad)
            ax.set_xticks([])
            ax.set_ylabel("symmetric divergence (null)")
            ax.set_title(
                f"1-class null distribution\n"
                f"median={median_null:.4f}  CV={cv:.1f}%  ({variation})"
            )
            plt.tight_layout()
            save_to_pdf(fig, pdf_file)
        except Exception:
            pass

    return {
        "cv": cv,
        "variation": variation,
        "relative_iqr": relative_iqr,
        "mean.random.entropy": mean_null,
        "sd.random.entropy": sd_null,
        "median.random.entropy": median_null,
    }
