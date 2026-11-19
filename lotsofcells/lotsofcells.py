"""Main `lots_of_cells` function: 2-group Monte-Carlo and >2-group Goodman & Kruskal gamma."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._parallel import run_permutations
from ._sample_level import (
    bootstrap_group_means,
    bootstrap_indices,
    gamma_per_column,
    group_weighted_means,
    maxt_adjust,
    perm_pmin,
    perm_pvalue,
    per_sample_props,
    relabellings,
    report_design,
    resolve_method,
    sample_count_matrix,
    sample_weights,
)
from ._stats import (
    _proportions_from_table,
    _table,
    asrt,
    cell_to_gamma,
    cell_to_gamma_original,
    cell_to_montecarlo,
)
from ._utils import get_metadata


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR. Equivalent to R p.adjust(., 'fdr')."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # cummin from the right
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def _gamma_godkrus(nc: np.ndarray, nd: np.ndarray, denom: float) -> np.ndarray:
    """Goodman-Kruskal gamma: (nc - nd) / exp(mean(log(N))) — N is denom (scalar here)."""
    return (nc - nd) / np.exp(np.log(denom))


def lots_of_cells(
    sc_object,
    main_variable: str,
    subtype_variable: str,
    label_order: Sequence[str],
    sample_id: Optional[str] = None,
    permutations: Optional[int] = None,
    seed: Optional[int] = None,
    n_cores: Optional[int] = None,
    table: Optional[str] = None,
    plot: bool = True,
    verbose: bool = True,
    pdf_file: Optional[str] = None,
    method: str = "auto",
    weights: str = "sqrt",
    weight_cap: Optional[float] = 2.0,
    alternative: str = "two-sided",
    n_bootstrap: int = 1000,
    max_exact: int = 20_000,
) -> pd.DataFrame:
    """Compute proportion tests on single-cell metadata.

    Parameters
    ----------
    sc_object
        AnnData / SpatialData / MuData / pandas.DataFrame.
    main_variable
        Column in ``.obs`` (or DataFrame) with the main grouping (e.g.
        ``"condition"``).
    subtype_variable
        Column with the covariable to test (e.g. ``"cell_type"``).
    label_order
        Order of labels in ``main_variable`` to compare.
        - 2 labels  → log2 fold-change of arcsin-sqrt proportions.
        - >2 labels → Goodman & Kruskal's gamma rank correlation.
    sample_id
        Optional column with sample IDs.
    permutations
        ``montecarlo``: Monte-Carlo permutations (default 1000).
        ``sample_permutation``: random relabellings drawn when there are more
        than ``max_exact`` possible ones (default: ``max_exact``).
    seed
        Random seed for reproducibility.
    table
        For SpatialData/MuData with multiple tables/modalities.
    plot
        If True and 2 labels, show the abundance test plot.
    method
        ``"sample_permutation"`` — the sample is the unit: each sample's
        composition uses all its cells, groups are weighted means of the
        per-sample arcsin-sqrt proportions, the null permutes sample labels
        across groups (exactly, when feasible) and the CI resamples samples
        within groups. Requires ``sample_id`` and ≥ 2 samples per group.
        ``"montecarlo"`` — the original cell-level Monte-Carlo test (pooled
        observed statistic, √n-sized draws for the null).
        ``"auto"`` (default) — ``sample_permutation`` when possible,
        otherwise ``montecarlo``.
    weights
        ``sample_permutation`` only: ``"sqrt"`` (√ cells, capped) or
        ``"equal"`` (one vote per sample).
    weight_cap
        Cap for ``sqrt`` weights, as a multiple of the median weight
        (``None`` disables).
    alternative
        ``sample_permutation`` only: ``"two-sided"``, ``"greater"`` or
        ``"less"`` (direction refers to ``label_order[0]`` vs
        ``label_order[1]``; for >2 groups to an increasing trend).
    n_bootstrap
        ``sample_permutation`` only: sample-bootstrap resamples for the CI.
    max_exact
        ``sample_permutation`` only: the exact number of relabellings is
        computed first; when it is at most ``max_exact`` (default 20,000)
        every one is enumerated, otherwise ``max_exact`` random relabellings
        are drawn (or ``permutations``, if given).

    Returns
    -------
    pandas.DataFrame with one row per covariable level. With
    ``sample_permutation`` the extra columns ``p.adj.fwer`` (Westfall–Young
    step-down maxT: family-wise error rate from the joint permutation null),
    ``p.val``, ``p.min`` (smallest achievable p-value — e.g. 0.10 for a
    two-sided 3 vs 3) and ``n_perms`` are appended. ``result.attrs["method"]`` records the method used.
    """
    metadata = get_metadata(sc_object, table=table)

    main_vals = metadata[main_variable].astype(str).to_numpy()

    if isinstance(label_order[0], list) or isinstance(label_order[1], list):
        if verbose:
            print(f"Multiple sub-groups detected.")
        # If several levels, process:
        flat_order = np.array([a for b in label_order for a in b]).astype(str)
        group_1 = np.array(label_order[0]).astype(str)
        group_2 = np.array(label_order[1]).astype(str)
        # Clean data with unwanted levels:
        mask = np.isin(main_vals, flat_order)
        metadata = metadata.loc[mask].copy()
        # Update target labels:
        main_vals = metadata[main_variable].astype(str).to_numpy()

        # Obtain group labels:
        mask_g1 = np.isin(main_vals, group_1)
        mask_g2 = np.isin(main_vals, group_2)
        # Define new labels:
        label_1 = "loc_group_one [" + " ".join(group_1)+"]"
        label_2 = "loc_group_two [" + " ".join(group_2)+"]"
        # Create synthetic labels:
        metadata.loc[mask_g1,"loc_tmp_group"] = label_1
        metadata.loc[mask_g2,"loc_tmp_group"] = label_2
        # relevel and recompute
        main_variable = "loc_tmp_group"
        main_vals = metadata[main_variable].astype(str).to_numpy()
        # Copy original and push new:
        label_order_original = label_order
        label_order = [label_1, label_2]

    if not all(l in np.unique(main_vals) for l in label_order):
        missing = [l for l in label_order if l not in np.unique(main_vals)]
        raise ValueError(f"Some groups in label_order not found in data: {missing}")

    mask = np.isin(main_vals, list(label_order))
    metadata = metadata.loc[mask].copy()
    groups = metadata[main_variable].astype(str).to_numpy()
    covariable = metadata[subtype_variable].astype(str).to_numpy()

    min_cells = 10

    if len(label_order) < 2:
        raise ValueError("label_order must have at least 2 entries.")

    resolved = resolve_method(method, metadata, main_variable, sample_id,
                              label_order, verbose=verbose)

    # Sample-level null: every relabelling when there are at most `max_exact`,
    # otherwise that many random ones (or `permutations`, if given). The
    # cell-level Monte Carlo keeps its own default of 1000.
    sample_perms = int(max_exact if permutations is None else permutations)
    permutations = 1000 if permutations is None else int(permutations)
    if resolved == "sample_permutation":
        samples = metadata[sample_id].astype(str).to_numpy()
        common = dict(
            groups=groups, covariable=covariable, samples=samples,
            label_order=list(label_order), weights=weights,
            weight_cap=weight_cap, alternative=alternative,
            permutations=sample_perms, n_bootstrap=n_bootstrap,
            max_exact=max_exact, seed=seed, verbose=verbose,
        )
        if len(label_order) > 2:
            return _sample_gamma_path(**common)
        return _sample_permutation_path(
            **common, subtype_variable=subtype_variable,
            plot=plot, pdf_file=pdf_file,
        )

    if len(label_order) > 2:
        return _gamma_path(
            covariable, groups, label_order, permutations, min_cells,
            seed, n_cores, verbose,
        )

    return _montecarlo_path(
        metadata,
        covariable,
        groups,
        label_order,
        sample_id,
        main_variable,
        subtype_variable,
        permutations,
        min_cells,
        seed,
        n_cores,
        verbose,
        plot,
        pdf_file,
    )


# --- 2-condition Monte Carlo path ----------------------------------------------------

def _montecarlo_path(
    metadata,
    covariable,
    groups,
    label_order,
    sample_id,
    main_variable,
    subtype_variable,
    permutations,
    min_cells,
    seed,
    n_cores,
    verbose,
    plot,
    pdf_file=None,
):
    # Local rng for the deterministic (non-permutation) synthetic-sample
    # step. The permutation loop below uses run_permutations with its own
    # SeedSequence chunking.
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Only 2 groups detected. Computing FC for {label_order[0]} vs {label_order[1]}")

    if sample_id is not None:
        if verbose:
            print(f"Additional sub-level for testing: {sample_id}")
        samples = metadata[sample_id].astype(str).to_numpy()
        n_per_sample = pd.crosstab(pd.Series(groups), pd.Series(samples)).reindex(label_order)

        # Synthetic samples (per-condition resampling) — mirrors R lotsOfCells.R
        synth_meta = metadata[[main_variable, subtype_variable, sample_id]].copy()
        mult_factor = 2
        new_samples = int(round((n_per_sample != 0).sum(axis=1).mean())) * mult_factor
        synth_rows = []
        for i in range(1, new_samples + 1):
            for cond in label_order:
                row = n_per_sample.loc[cond]
                nonzero = row[row != 0]
                if len(nonzero) == 0:
                    continue
                n = int(rng.integers(int(nonzero.min()), int(nonzero.max()) + 1))
                pool = covariable[groups == cond]
                synth_cov = rng.choice(pool, size=n, replace=True)
                synth_rows.append(pd.DataFrame({
                    main_variable: cond,
                    subtype_variable: synth_cov,
                    sample_id: f"synthetic_sample_{cond}_{i}",
                }))
        if synth_rows:
            synth_meta = pd.concat([synth_meta, *synth_rows], ignore_index=True)

        groups_synth = synth_meta[main_variable].astype(str).to_numpy()
        covariable_synth = synth_meta[subtype_variable].astype(str).to_numpy()

        cell_crowd = {}
        for cond in label_order:
            row = n_per_sample.loc[cond]
            nonzero = row[row != 0].to_numpy()
            cell_crowd[cond] = list(np.maximum(np.sqrt(nonzero), min_cells).astype(int))
    else:
        groups_synth = groups
        covariable_synth = covariable
        counts_per_group = pd.Series(groups).value_counts().to_dict()
        cell_crowd = {
            l: int(round(max(np.sqrt(counts_per_group.get(l, 0)), min_cells)))
            for l in label_order
        }

    # Observed fold-change
    obs_tab = _table(groups, covariable)
    p_obs = _proportions_from_table(obs_tab, label_order, list(obs_tab.columns), pseudo=True)
    indexes = list(obs_tab.columns)
    obs_fc = np.log2(asrt(p_obs[0]) / asrt(p_obs[1]))

    if verbose:
        print("- Starting Monte-Carlo simulation of fold changes")

    def _perm_chunk(chunk_size, seed_seq):
        # Each worker's own rng, seeded independently → deterministic across
        # any n_cores value for the same base `seed`.
        chunk_rng = np.random.default_rng(seed_seq)
        # Stack null and real together so run_permutations concat works;
        # shape is (chunk_size, 2, n_indexes). We unpack after the join.
        out = np.empty((chunk_size, 2, len(indexes)))
        for k in range(chunk_size):
            m, o = cell_to_montecarlo(
                covariable_synth, groups_synth, label_order, indexes,
                cell_crowd, chunk_rng,
            )
            out[k, 0] = m
            out[k, 1] = o
        return out

    stacked = run_permutations(
        _perm_chunk, permutations,
        n_cores=n_cores, seed=seed, verbose=verbose,
    )
    null_fcs = stacked[:, 0, :]
    real_fcs = stacked[:, 1, :]

    higher = (np.sum(null_fcs >= obs_fc, axis=0) + 1) / (permutations + 1)
    lower = (np.sum(null_fcs <= obs_fc, axis=0) + 1) / (permutations + 1)
    p_vals = np.where(obs_fc > 0, higher, lower)
    p_adj = _bh_fdr(p_vals)
    sd_mc = null_fcs.std(axis=0, ddof=1)
    ci_low = np.quantile(real_fcs, 0.025, axis=0)
    ci_high = np.quantile(real_fcs, 0.975, axis=0)

    pct1 = np.round(p_obs[0], 3)
    pct2 = np.round(p_obs[1], 3)
    table_results = pd.DataFrame(
        {
            "groupFC": obs_fc,
            f"percent_in_{label_order[0]}": pct1,
            f"percent_in_{label_order[1]}": pct2,
            "p.adj": np.round(p_adj, 5),
            "sd.montecarlo": sd_mc,
            "CI95low": ci_low,
            "CI95high": ci_high,
            # Raw p-value. Note: this method reports the one-sided tail in
            # the direction of the observed FC (as the R original does).
            "p.val": np.round(p_vals, 5),
        },
        index=indexes,
    )
    table_results.attrs["method"] = "montecarlo"

    # Ensure CIs encompass observed
    bad_low = ~(table_results["CI95low"] < table_results["groupFC"])
    table_results.loc[bad_low, "CI95low"] = table_results.loc[bad_low, "groupFC"]
    bad_high = ~(table_results["CI95high"] > table_results["groupFC"])
    table_results.loc[bad_high, "CI95high"] = table_results.loc[bad_high, "groupFC"]

    if plot:
        try:
            from .plots import plot_abundance_test
            plot_abundance_test(
                table_results,
                subtype_variable=subtype_variable,
                pdf_file=pdf_file,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"(Plot skipped: {e})")

    return table_results


# --- >2-condition Goodman-Kruskal gamma path -----------------------------------------

def _gamma_path(
    covariable, groups, label_order, permutations, min_cells,
    seed, n_cores, verbose,
):
    if verbose:
        print(
            "More than 2 groups detected. Computing Goodman-Kruskal gamma rank "
            f"correlation in order: {' vs '.join(label_order)}"
        )

    # Local rng for the small (fixed-size) CI-subsample loop, which stays
    # serial. The two big permutation loops use run_permutations with their
    # own SeedSequence chunking so results are reproducible for any n_cores.
    rng = np.random.default_rng(seed)

    counts_per_group = pd.Series(groups).value_counts().to_dict()
    cell_crowd = {
        l: int(round(max(np.sqrt(counts_per_group.get(l, 0)), min_cells)))
        for l in label_order
    }
    kendall_denom = (len(label_order) * (len(label_order) - 1)) / 2
    rank_index = np.arange(1, len(label_order) + 1)

    obs_tab = _table(groups, covariable)
    indexes = list(obs_tab.columns)

    # --- Observed gamma: aggregate over `permutations` subsamplings of the
    # original data. Chunk returns per-permutation (nc, nd), summed after
    # the join.
    def _obs_chunk(chunk_size, seed_seq):
        r = np.random.default_rng(seed_seq)
        out = np.empty((chunk_size, 2, len(indexes)))
        for k in range(chunk_size):
            nc, nd = cell_to_gamma_original(
                covariable, groups, label_order, indexes, cell_crowd,
                rank_index, r,
            )
            out[k, 0] = nc
            out[k, 1] = nd
        return out

    obs_stack = run_permutations(
        _obs_chunk, permutations,
        n_cores=n_cores, seed=seed, verbose=verbose,
    )
    nc_orig = obs_stack[:, 0, :].sum(axis=0)
    nd_orig = obs_stack[:, 1, :].sum(axis=0)
    obs_gamma = _gamma_godkrus(nc_orig, nd_orig, kendall_denom * permutations)

    # --- CI: 10 sub-samples × 100 aggregations each. Fixed cost — stays
    # serial; parallel dispatch would add more overhead than it saves.
    sub_gammas = np.empty((10, len(indexes)))
    for s in range(10):
        nc_s = np.zeros(len(indexes))
        nd_s = np.zeros(len(indexes))
        for _ in range(100):
            nc, nd = cell_to_gamma_original(
                covariable, groups, label_order, indexes, cell_crowd, rank_index, rng
            )
            nc_s += nc
            nd_s += nd
        sub_gammas[s] = _gamma_godkrus(nc_s, nd_s, kendall_denom * 100)
    ci_low = np.nanquantile(sub_gammas, 0.025, axis=0)
    ci_high = np.nanquantile(sub_gammas, 0.975, axis=0)

    if verbose:
        print("- Starting gamma rank permutation analysis, this can take a while...")

    # --- Null distribution: `permutations` outer × 10 inner. Each outer
    # iteration returns one gamma vector; chunk returns (chunk_size, k).
    n_random_observations = 10

    def _null_chunk(chunk_size, seed_seq):
        r = np.random.default_rng(seed_seq)
        out = np.empty((chunk_size, len(indexes)))
        for k in range(chunk_size):
            nc_p = np.zeros(len(indexes))
            nd_p = np.zeros(len(indexes))
            for _ in range(n_random_observations):
                nc, nd = cell_to_gamma(
                    covariable, groups, label_order, indexes, cell_crowd,
                    rank_index, r,
                )
                nc_p += nc
                nd_p += nd
            out[k] = _gamma_godkrus(nc_p, nd_p, kendall_denom * n_random_observations)
        return out

    # Use a different-but-deterministic seed for the null so it's not
    # correlated with the observed pass.
    null_seed = None if seed is None else int(seed) + 1
    null_gamma = run_permutations(
        _null_chunk, permutations,
        n_cores=n_cores, seed=null_seed, verbose=verbose,
    )

    with np.errstate(invalid="ignore"):
        # +1/+1: random draws never include the observed configuration.
        higher = (np.sum(null_gamma >= obs_gamma, axis=0) + 1) / (permutations + 1)
        lower = (np.sum(null_gamma <= obs_gamma, axis=0) + 1) / (permutations + 1)
    p_vals = np.where(obs_gamma > 0, higher, lower)
    p_adj = _bh_fdr(np.nan_to_num(p_vals, nan=1.0))

    # Per-condition proportions (unnormalised contig table -> per-row proportions)
    contig_tab = _table(groups, covariable).reindex(label_order)
    proportions = contig_tab.div(contig_tab.sum(axis=1), axis=0).reindex(
        index=label_order, columns=indexes
    )

    df = pd.DataFrame({"groupGammaCor": np.round(obs_gamma, 4)}, index=indexes)
    for l in label_order:
        df[f"percent_in_{l}"] = np.round(proportions.loc[l].values, 3)
    df["p.adj"] = np.round(p_adj, 5)
    df["CI95low"] = np.round(ci_low, 4)
    df["CI95high"] = np.round(ci_high, 4)
    df.attrs["method"] = "montecarlo"
    return df


# --- Sample-level paths --------------------------------------------------------------

def _sample_setup(groups, covariable, samples, label_order, weights, weight_cap):
    """Shared preparation: per-sample counts, weights and compositions."""
    indexes = list(_table(groups, covariable).columns)
    counts, names, group_idx, n_s = sample_count_matrix(
        groups, samples, covariable, label_order, indexes,
    )
    w = sample_weights(n_s, weights, weight_cap)
    props = per_sample_props(counts, "loc")
    return indexes, names, group_idx, n_s, w, props


def _sample_permutation_path(
    groups, covariable, samples, label_order, weights, weight_cap, alternative,
    permutations, n_bootstrap, max_exact, seed, verbose,
    subtype_variable, plot, pdf_file,
):
    """Two-group test with the sample as the unit of independence.

    observed  T_g = Σ w_s·asin(√p_s) / Σ w_s,  groupFC = log2(T_A / T_B)
    null      the same statistic under every (or random) relabelling of
              samples across the two groups
    CI        the same statistic on samples resampled within each group
    """
    indexes, _, group_idx, _, w, props = _sample_setup(
        groups, covariable, samples, label_order, weights, weight_cap,
    )
    t = asrt(props)                                       # (n_samples, n_cov)
    rng = np.random.default_rng(seed)

    def fc_of(assign):
        means = group_weighted_means(assign, t, w, 2)
        return np.log2(means[0] / means[1])

    obs_fc = fc_of(group_idx[np.newaxis, :])[0]
    assign, exact = relabellings(group_idx, max_exact, permutations, rng)
    null_fc = fc_of(assign)
    n_perms = assign.shape[0]

    p_vals = perm_pvalue(null_fc, obs_fc, exact, alternative)
    p_adj = _bh_fdr(p_vals)
    p_fwer = maxt_adjust(null_fc, obs_fc, exact, alternative)
    equal_sizes = (group_idx == 0).sum() == (group_idx == 1).sum()
    p_min = perm_pmin(n_perms, exact,
                      mirror_tie=equal_sizes and alternative == "two-sided")
    report_design(label_order, group_idx, n_perms, exact, p_min, verbose)

    boot_rng = np.random.default_rng(None if seed is None else int(seed) + 7919)
    boot_idx = bootstrap_indices(group_idx, n_bootstrap, boot_rng)
    boot_means = bootstrap_group_means(boot_idx, group_idx, t, w, 2)
    boot_fc = np.log2(boot_means[0] / boot_means[1])
    ci_low = np.quantile(boot_fc, 0.025, axis=0)
    ci_high = np.quantile(boot_fc, 0.975, axis=0)

    composition = group_weighted_means(group_idx[np.newaxis, :], props, w, 2)[:, 0, :]

    table_results = pd.DataFrame(
        {
            "groupFC": obs_fc,
            f"percent_in_{label_order[0]}": np.round(composition[0], 3),
            f"percent_in_{label_order[1]}": np.round(composition[1], 3),
            "p.adj": np.round(p_adj, 5),
            "p.adj.fwer": np.round(p_fwer, 5),
            # SD of the permutation null; name kept for plot_abundance_test.
            "sd.montecarlo": null_fc.std(axis=0, ddof=1),
            "CI95low": ci_low,
            "CI95high": ci_high,
            "p.val": np.round(p_vals, 5),
            "p.min": p_min,
            "n_perms": n_perms,
        },
        index=indexes,
    )
    table_results.attrs.update(method="sample_permutation", weights=weights,
                               alternative=alternative, exact=exact)

    if plot:
        try:
            from .plots import plot_abundance_test
            plot_abundance_test(table_results, subtype_variable=subtype_variable,
                                pdf_file=pdf_file)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"(Plot skipped: {e})")
    return table_results


def _sample_gamma_path(
    groups, covariable, samples, label_order, weights, weight_cap, alternative,
    permutations, n_bootstrap, max_exact, seed, verbose,
):
    """>2 ordered groups with the sample as the unit.

    Per cell type, Goodman–Kruskal gamma between each sample's group rank
    (position in ``label_order``) and its asin(√p) value. Rank-based, so
    sample weights do not enter the statistic (they are used for the
    reported group compositions). Null: sample labels permuted across
    groups. CI: samples resampled within groups.
    """
    indexes, _, group_idx, _, w, props = _sample_setup(
        groups, covariable, samples, label_order, weights, weight_cap,
    )
    t = asrt(props)
    rng = np.random.default_rng(seed)

    obs_gamma = gamma_per_column(t, group_idx[np.newaxis, :])[0]
    assign, exact = relabellings(group_idx, max_exact, permutations, rng)
    null_gamma = gamma_per_column(t, assign)
    n_perms = assign.shape[0]

    p_vals = perm_pvalue(null_gamma, obs_gamma, exact, alternative)
    p_adj = _bh_fdr(p_vals)
    p_fwer = maxt_adjust(null_gamma, obs_gamma, exact, alternative)
    # Reversing the group order is itself a relabelling when the group sizes
    # are mirror-symmetric; it flips the sign of gamma, so it ties two-sided.
    sizes = np.bincount(group_idx)
    p_min = perm_pmin(n_perms, exact,
                      mirror_tie=alternative == "two-sided"
                      and np.array_equal(sizes, sizes[::-1]))
    report_design(label_order, group_idx, n_perms, exact, p_min, verbose)

    boot_rng = np.random.default_rng(None if seed is None else int(seed) + 7919)
    boot_idx = bootstrap_indices(group_idx, n_bootstrap, boot_rng)
    boot_gamma = np.vstack([
        gamma_per_column(t[row], group_idx[np.newaxis, :])[0] for row in boot_idx
    ])
    ci_low = np.nanquantile(boot_gamma, 0.025, axis=0)
    ci_high = np.nanquantile(boot_gamma, 0.975, axis=0)

    composition = group_weighted_means(
        group_idx[np.newaxis, :], props, w, len(label_order),
    )[:, 0, :]

    df = pd.DataFrame({"groupGammaCor": np.round(obs_gamma, 4)}, index=indexes)
    for k, l in enumerate(label_order):
        df[f"percent_in_{l}"] = np.round(composition[k], 3)
    df["p.adj"] = np.round(p_adj, 5)
    df["p.adj.fwer"] = np.round(p_fwer, 5)
    df["CI95low"] = np.round(ci_low, 4)
    df["CI95high"] = np.round(ci_high, 4)
    df["p.val"] = np.round(p_vals, 5)
    df["p.min"] = p_min
    df["n_perms"] = n_perms
    df.attrs.update(method="sample_permutation", weights=weights,
                    alternative=alternative, exact=exact)
    return df
