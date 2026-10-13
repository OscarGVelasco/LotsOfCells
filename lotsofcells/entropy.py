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
    return np.mean(np.tanh(np.abs(np.log2(p / q)) * (np.abs(p-q)/(p+q))))


def _bootstrap_observed_score(
    metadata: pd.DataFrame,
    main_variable: str,
    subtype_variable: str,
    sample_id: Optional[str],
    label_order: Sequence[str],
    indexes,
    n_bootstrap: int,
    rng: np.random.Generator,
    verbose: bool = True,
) -> np.ndarray:
    """Within-group bootstrap of the observed entropy score.

    Mirrors the "resample-from-original" step that :func:`lots_of_cells`
    uses to build its `CI95low`/`CI95high` — resample within each group
    (with replacement) so per-group structure is preserved, then recompute
    the score. The spread across replicates estimates the internal
    variability of the observed score.

    - If ``sample_id`` is provided, resamples at the SAMPLE level:
      draws samples with replacement per group, keeps every cell of the
      drawn samples. That's the biologically appropriate unit of
      independence.
    - Otherwise resamples cells with replacement within each group as a
      fallback.

    Returns an array of length ``n_bootstrap``.
    """
    if verbose:
        unit = "samples" if sample_id is not None else "cells"
        # Report how many units per group are being drawn so users can
        # see the bootstrap preserves group size rather than subsetting.
        sizes = []
        for label in label_order:
            group_mask = (metadata[main_variable].astype(str).to_numpy() == label)
            if sample_id is not None:
                n_units = len(
                    metadata.loc[group_mask, sample_id].astype(str).unique()
                )
            else:
                n_units = int(group_mask.sum())
            sizes.append(f"{label}: {n_units} {unit}")
        print(
            f"Bootstrap: {n_bootstrap} within-group resamples of "
            f"{unit} with replacement ({'; '.join(sizes)}) "
            "→ observed-score variability"
        )
    boot_scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        grp_pieces, cov_pieces = [], []
        for label in label_order:
            group_mask = (
                metadata[main_variable].astype(str).to_numpy() == label
            )
            if sample_id is not None:
                samples_in_group = (
                    metadata.loc[group_mask, sample_id].astype(str).unique()
                )
                if len(samples_in_group) == 0:
                    continue
                boot_samples = rng.choice(
                    samples_in_group, size=len(samples_in_group), replace=True,
                )
                sample_col = metadata[sample_id].astype(str).to_numpy()
                sub_col = metadata[subtype_variable].astype(str).to_numpy()
                for s in boot_samples:
                    m = group_mask & (sample_col == s)
                    n = int(m.sum())
                    if n == 0:
                        continue
                    cov_pieces.append(sub_col[m])
                    grp_pieces.append(np.repeat(label, n))
            else:
                idxs = np.where(group_mask)[0]
                boot_idxs = rng.choice(idxs, size=len(idxs), replace=True)
                cov_pieces.append(
                    metadata[subtype_variable].astype(str).to_numpy()[boot_idxs]
                )
                grp_pieces.append(np.repeat(label, len(boot_idxs)))
        cov = np.concatenate(cov_pieces)
        grp = np.concatenate(grp_pieces)
        boot_tab = _table(grp, cov)
        contig_boot = _proportions_arcsin(boot_tab, label_order, indexes)
        boot_scores[i] = _distance_surprise(contig_boot[0], contig_boot[1])
    return boot_scores


def entropy_score(
    sc_object,
    main_variable: str,
    subtype_variable: str,
    label_order: Sequence[str],
    sample_id: Optional[str] = None,
    permutations: int = 1000,
    n_bootstrap: int = 100,
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

    If ``len(label_order) == 1``, runs the 1-class tests on ``sample_id``
    (see :func:`_one_class_test_2`) and returns a tuple
    ``(individual_data, total_data)`` of two DataFrames instead:
    pairwise scores of the individual test and per-sample scores of the
    total test.
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
        return _one_class_test_2(
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
    subtype_variable, boot_scores=None, pdf_file=None,
):
    import matplotlib.pyplot as plt
    from ._utils import draw_threshold_bands, save_to_pdf

    # Widened last column so the threshold legend has room without
    # overlapping the scatter cloud.
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1.3]},
    )
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
    # Scatter first so the y-axis auto-scales to the data before we shade.
    ax2.scatter(jitter, null_scores, color="#D5BADB", alpha=0.6, s=15, zorder=3)
    ax2.axhline(np.median(null_scores), color="#86608E", lw=1, zorder=4)

    # Observed dot with an ASYMMETRIC bootstrap percentile CI.
    # ± SD is misleading when the bootstrap distribution is skewed (which
    # is common with small-sample cluster bootstraps — one heavy-composition
    # sample drawn 3-of-4 times inflates the upper tail). Percentile CIs
    # give an honest picture of the actual spread on each side of the
    # observed value.
    if boot_scores is not None and len(boot_scores) > 1:
        lo = float(np.quantile(boot_scores, 0.025))
        hi = float(np.quantile(boot_scores, 0.975))
        # Clip negative lower bound at 0 — the score is non-negative.
        lo = max(lo, 0.0)
        yerr = [[max(obs_score - lo, 0.0)], [max(hi - obs_score, 0.0)]]
        ax2.errorbar(
            [0], [obs_score], yerr=yerr,
            fmt="o", color="#F08080", markersize=10,
            markeredgecolor="black", markeredgewidth=0.5,
            ecolor="#B22222", elinewidth=1.2, capsize=5, capthick=1.0,
            zorder=6,
            label=f"observed with bootstrap 95% CI (n={len(boot_scores)})",
        )
    else:
        ax2.scatter(
            [0], [obs_score], color="#F08080", s=80, zorder=5,
            edgecolor="black", linewidth=0.4, label="observed",
        )
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xticks([])
    ax2.set_ylabel("symmetric divergence")

    # Shade the six qualitative-change bands behind the cloud. Bands are
    # calibrated on the bounded_fc_mean scale (see
    # examples/simulation_calibration_realistic.py) — keep them as a
    # reference against which the observed score can be read visually.
    draw_threshold_bands(ax2, alpha=0.30, zorder=0, min_top=0.20)

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
            from ._utils import draw_threshold_bands, save_to_pdf

            # Wider figure so the threshold legend has room outside.
            fig, ax = plt.subplots(figsize=(5.5, 5))
            jitter_rng = np.random.default_rng(seed if seed is not None else 0)
            jitter = jitter_rng.uniform(-0.1, 0.1, size=len(null_scores))
            ax.scatter(jitter, null_scores, color="#D5BADB", alpha=0.6, s=15, zorder=3)
            ax.axhline(median_null, color="#86608E", lw=1, zorder=4)
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

            # Six qualitative-change bands behind the null cloud (grey →
            # deep purple as intensity grows). Calibrated on bounded_fc_mean
            # — see examples/simulation_calibration_realistic.py.
            draw_threshold_bands(ax, alpha=0.30, zorder=0, min_top=0.20)
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




def _one_class_test_2(
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
    """Alternative to `_one_class_test`: compares the samples directly.

    - individual_test: every sample against every other sample (pairwise scores)
    - total_test: every sample against a random draw from all remaining samples

    Same signature as `_one_class_test`, so that switching between the two
    variants only requires changing the function name in `entropy_score`.
    `n_cores` is accepted for compatibility but not used (yet).

    Returns
    -------
    individual_data : pandas.DataFrame
        One row per ordered sample pair with columns
        ``sample``, ``compared_to``, ``Score``.
    total_data : pandas.DataFrame
        One row per sample with columns
        ``sample``, ``Mean Score``, ``Std of Score``, ``Median Score``.
    """
    rng = np.random.default_rng(seed)

    samples = metadata[sample_id].astype(str).to_numpy()
    obs_tab = _table(samples, covariable)
    indexes = list(obs_tab.columns)
    if verbose:
        print("obs_tab")
        print(obs_tab)

    # Per-sample pool of cell-type labels (the real composition of each sample)
    n_per_sample = pd.Series(samples).value_counts()
    sample_pools = {s: covariable[samples == s] for s in n_per_sample.index}
    unique_samples = list(n_per_sample.index)

    if len(unique_samples) < 2:
        raise ValueError(
            "1-class entropy test needs at least 2 samples in `sample_id`."
        )

    individual_data, score_matrix = individual_test(unique_samples, sample_pools, indexes)
    total_data, _ = total_test(permutations, unique_samples, sample_pools, indexes, rng)

    if verbose:
        print("individual_data")
        print(individual_data)
        print("total_data")
        print(total_data)

    if plot:
        # Plotting errors must not abort the test (same behaviour as `_one_class_test`)
        try:
            import matplotlib.pyplot as plt

            fig_bar, _ = plot_bar(
                total_data["sample"], total_data["Mean Score"], total_data["Std of Score"]
            )
            g = plot_heatmap(individual_data, unique_samples, score_matrix=score_matrix)

            # Both figures are written into a single PDF if a path is given
            if pdf_file is not None:
                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages(pdf_file) as pdf:
                    pdf.savefig(fig_bar, bbox_inches="tight")
                    pdf.savefig(g.fig, bbox_inches="tight")
            plt.show()
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"(Plot skipped: {e})")

    return individual_data, total_data


def individual_test(unique_samples, sample_pools, indexes):
    """Compute exactly ONE distance-surprise score for every sample pair.

    Each pair is scored directly from the observed cell-type distributions
    (no resampling). The score is symmetric (Score(A, B) == Score(B, A)),
    so every pair is computed only once and mirrored into the matrix.

    Returns
    -------
    data : pandas.DataFrame
        One row per ordered pair (``sample``, ``compared_to``, ``Score``).
    score_matrix : numpy.ndarray
        Symmetric (n_samples x n_samples) matrix of scores, NaN on the diagonal.
    """
    n_samples = len(unique_samples)
    score_matrix = np.full((n_samples, n_samples), np.nan)

    for k in range(n_samples):
        pool_k = sample_pools[unique_samples[k]]
        if len(pool_k) == 0:
            continue
        for j in range(k + 1, n_samples):
            pool_j = sample_pools[unique_samples[j]]
            if len(pool_j) == 0:
                continue

            cov = np.concatenate([pool_k, pool_j])
            grp = np.concatenate([
                np.repeat("sample_0", len(pool_k)),
                np.repeat("sample_1", len(pool_j)),
            ])
            tab = _table(grp, cov)
            p = _proportions_arcsin(tab, ["sample_0", "sample_1"], indexes)
            score = _distance_surprise(p[0], p[1])
            score_matrix[k, j] = score
            score_matrix[j, k] = score

    # Long format: one row per ordered pair (both directions listed)
    rows = []
    for k, s0 in enumerate(unique_samples):
        for j, s in enumerate(unique_samples):
            if s == s0:
                continue
            rows.append({
                "sample": s0,
                "compared_to": s,
                "Score": score_matrix[k, j],
            })

    data = pd.DataFrame(rows)
    return data, score_matrix


def total_test(permutations, unique_samples, sample_pools, indexes, rng):
    """Compare every sample against an artificial reference built from all other samples.

    For each sample s0, the reference consists of n cells drawn at random
    (with replacement) from the pooled cells of all remaining samples, where
    n is the number of cells in s0. This is repeated ``permutations`` times.

    Returns
    -------
    data : pandas.DataFrame
        One row per sample with ``Mean Score``, ``Std of Score`` and
        ``Median Score`` across the random draws.
    null_scores : numpy.ndarray
        All individual scores, shape (permutations, n_samples).
    """
    n_samples = len(unique_samples)
    null_scores = np.empty((permutations, n_samples))
    rows = []

    for k, s0 in enumerate(unique_samples):
        pool_s0 = sample_pools[s0]
        n0 = len(pool_s0)
        # Depends only on s0, so it is built once instead of in every repetition
        pool_rest = np.concatenate([sample_pools[s] for s in unique_samples if s != s0])
        grp = np.concatenate([np.repeat("sample_0", n0), np.repeat("rest", n0)])

        for i in range(permutations):
            draw = rng.choice(pool_rest, size=n0, replace=True)
            cov = np.concatenate([pool_s0, draw])
            rand_tab = _table(grp, cov)
            p = _proportions_arcsin(rand_tab, ["sample_0", "rest"], indexes)
            null_scores[i, k] = _distance_surprise(p[0], p[1])

        col = null_scores[:, k]
        rows.append((s0, col.mean(), col.std(ddof=1), np.median(col)))

    data = pd.DataFrame(rows, columns=["sample", "Mean Score", "Std of Score", "Median Score"])
    return data, null_scores


# Colour scale of the one-class plots: a continuous transition from green
# (low heterogeneity) over blue (medium) to purple (high). The medium range
# lies between the two bounds; the colour is pure blue in its middle.
HEATMAP_BANDS = (0.5, 0.7)    # pairwise scores (individual test)
BAR_BANDS = (0.63, 0.78)      # mean scores (total test)
HETEROGENEITY_COLORS = [
    "#a6dba0",   # score 0:            light green (very similar)
    "#1b7837",   # lower bound:        green
    "#2166ac",   # middle of the band: blue
    "#762a83",   # upper bound:        purple
    "#40004b",   # score 1:            dark purple (very different)
]


def _heterogeneity_cmap(bands):
    """Colormap on the score range [0, 1]: green below the lower bound,
    transition through blue inside the medium range, purple above the upper bound."""
    from matplotlib.colors import LinearSegmentedColormap

    lo, hi = bands
    positions = [0.0, lo, (lo + hi) / 2, hi, 1.0]
    return LinearSegmentedColormap.from_list(
        "heterogeneity", list(zip(positions, HETEROGENEITY_COLORS))
    )


def _mark_bands_on_colorbar(cbar_ax, bands):
    """Draw the two bounds on a vertical colour bar and label them."""
    lo, hi = bands
    for bound in bands:
        cbar_ax.axhline(bound, color="white", linewidth=1.5)
    cbar_ax.set_yticks([0.0, lo, hi, 1.0])
    cbar_ax.set_yticklabels(["0", f"{lo:g}", f"{hi:g}", "1"])


def plot_bar(label_order, scores, errors, test_name="total", bands=BAR_BANDS):
    """Bar chart (mean ± SD) of the total-test score per sample.

    The colour of each bar shows the heterogeneity of the sample: a continuous
    transition from green (low) over blue (medium, between the two values of
    ``bands``) to purple (high). The bounds are drawn as dashed lines.

    Note: the error bars only reflect the variation between the random
    reference draws, not biological variability.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    label_order = np.asarray(label_order)
    scores = np.asarray(scores, dtype=float)
    errors = np.asarray(errors, dtype=float)
    x = np.arange(len(label_order))

    cmap = _heterogeneity_cmap(bands)
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x, scores, yerr=errors, capsize=10, width=0.6,
           color=cmap(norm(scores)), edgecolor="black", linewidth=0.5)
    for bound in bands:
        ax.axhline(bound, color="black", linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(label_order, rotation=45, ha="right")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Score")
    ax.set_title(f"Results: {test_name}-test\nMean ± SD across random reference draws")

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cbar.set_label("Heterogeneity (score)")
    _mark_bands_on_colorbar(cbar.ax, bands)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(0, ymin), max(ymax, bands[1] * 1.05))
    fig.tight_layout()
    return fig, ax


def plot_heatmap(individual_data, unique_samples, score_matrix=None,
                 test_name="individual", bands=HEATMAP_BANDS):
    """Clustered heatmap of the pairwise scores from the individual test.

    The scores are themselves distances between samples and are therefore
    used directly for the hierarchical clustering. The colour shows the
    heterogeneity of each pair: a continuous transition from green (low) over
    blue (medium, between the two values of ``bands``) to purple (high).
    The diagonal (a sample compared with itself) is left blank.
    """
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    if score_matrix is not None:
        score_df = pd.DataFrame(score_matrix, index=unique_samples, columns=unique_samples)
    else:
        score_df = individual_data.pivot_table(
            index="sample", columns="compared_to", values="Score", aggfunc="mean"
        )
        score_df = score_df.combine_first(score_df.T)

    # No self-comparison exists on the diagonal -> 0 (distance to itself) for
    # the clustering; it is masked in the plot.
    score_df = score_df.fillna(0)
    mask = np.eye(score_df.shape[0], dtype=bool)

    # Average-linkage clustering on the score matrix used as a distance matrix
    link = linkage(squareform(score_df.to_numpy(), checks=False), method="average")

    n = score_df.shape[0]
    show_labels = n <= 50  # labels become unreadable with many samples

    g = sns.clustermap(
        score_df,
        row_linkage=link,
        col_linkage=link,
        mask=mask,
        cmap=_heterogeneity_cmap(bands),
        vmin=0.0,
        vmax=1.0,
        figsize=(14, 14),
        xticklabels=show_labels,
        yticklabels=show_labels,
        cbar_kws={"label": "Heterogeneity (score)"},
    )
    _mark_bands_on_colorbar(g.ax_cbar, bands)

    g.fig.suptitle(f"Results: {test_name}-test", fontsize=14, fontweight="bold", y=1.02)
    return g