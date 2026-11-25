"""Main `lots_of_cells` function: differential abundance with the sample as the unit.

Two groups: log2 fold change of the √n-weighted mean per-sample proportions, tested
against every (or 20,000 random) relabelling of the samples. More than two ordered
groups: Goodman–Kruskal gamma between group rank and per-sample composition.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

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
    sample_count_matrix,
    sample_weights,
)
from ._stats import _table, asrt
from ._utils import get_metadata

# Fixed design of the test (benchmarked in simulation_benchmark/plasmode)
N_RELABEL = 20_000      # every relabelling when there are at most this many, else this many random
N_BOOTSTRAP = 1_000     # sample-bootstrap resamples for the 95% CI
WEIGHT_CAP = 2.0        # sample weight √n, capped at 2 × the median weight
EPS_CELLS = 0.25        # stabiliser of the group means, in cells at the median sample size


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


def lots_of_cells(
    sc_object,
    main_variable: str,
    subtype_variable: str,
    label_order: Sequence[str],
    sample_id: str,
    seed: Optional[int] = None,
    alternative: str = "two-sided",
    table: Optional[str] = None,
    plot: bool = True,
    verbose: bool = True,
    pdf_file: Optional[str] = None,
) -> pd.DataFrame:
    """Differential abundance of cell types (or any covariable) between groups,
    with the sample as the unit of replication.

    Two groups (``label_order = [A, B]``), per covariable level k:

    - each sample s contributes its proportion p_sk = c_sk / n_s (no pseudocount),
      with weight w_s = √n_s capped at 2 × the median weight;
    - each group's composition is the weighted mean m_gk = Σ w_s p_sk / Σ w_s;
    - ``groupFC`` = log2((m_Ak + ε) / (m_Bk + ε)), with ε = ¼ / median(n_s), a
      quarter of a cell at the typical sample size. ε is the same for both
      groups, so it does not depend on which group has the smaller samples;
    - null: the same statistic under every relabelling of the samples across
      the two groups (exact when there are at most 20,000, otherwise 20,000
      random ones); p-value two-sided unless ``alternative`` says otherwise;
    - 95% CI: 1,000 bootstrap resamples of the samples within each group.

    The mean of c/n is the cell type's proportion whatever the sample size, so
    groups that differ in cells per sample (depth, tissue area) are compared
    without bias, also for rare cell types.

    More than two ordered groups: Goodman–Kruskal gamma between each sample's
    group rank and its composition, with the same relabelling null and
    bootstrap (column ``groupGammaCor``).

    Parameters
    ----------
    sc_object
        AnnData / SpatialData / MuData / pandas.DataFrame.
    main_variable
        Column in ``.obs`` (or DataFrame) with the grouping (e.g. ``"condition"``).
    subtype_variable
        Column with the covariable to test (e.g. ``"cell_type"``).
    label_order
        Labels of ``main_variable`` to compare, in order. Two labels: groupFC is
        ``label_order[0]`` vs ``label_order[1]``. More than two: ordered groups.
        An entry may be a list of labels, merged into one group.
    sample_id
        Column with the sample (donor, section) of each cell. Every group needs
        at least two samples; each sample must belong to one group.
    seed
        Random seed (random relabellings and bootstrap).
    alternative
        ``"two-sided"`` (default), ``"greater"`` or ``"less"``: direction of
        ``label_order[0]`` vs ``label_order[1]`` (for >2 groups, an increasing
        trend). One-sided only when the direction was specified beforehand.
    table
        For SpatialData/MuData with multiple tables/modalities.
    plot
        Two groups: draw the abundance plot.
    verbose
        Print the design: samples per group, number of relabellings and the
        smallest achievable p-value.
    pdf_file
        Also save the plot to this PDF.

    Returns
    -------
    pandas.DataFrame, one row per covariable level. Two groups: ``groupFC``,
    ``percent_in_<A>``, ``percent_in_<B>`` (weighted mean proportions),
    ``p.adj`` (Benjamini–Hochberg), ``p.adj.fwer`` (Westfall–Young step-down
    maxT, family-wise error rate), ``sd.null`` (SD of the permutation null),
    ``CI95low``, ``CI95high`` and ``p.val``. ``result.attrs`` holds ``p_min``
    (smallest achievable p-value, e.g. 0.10 for a two-sided 3 vs 3),
    ``n_perms``, ``exact``, ``alternative`` and ``epsilon``.
    """
    metadata = get_metadata(sc_object, table=table)
    for col in (main_variable, subtype_variable, sample_id):
        if col not in metadata.columns:
            raise ValueError(f"Column {col!r} not found in the metadata.")

    main_vals = metadata[main_variable].astype(str).to_numpy()

    if len(label_order) < 2:
        raise ValueError("label_order must have at least 2 entries.")

    # Two groups given as lists of labels: merge each list into one group
    if isinstance(label_order[0], list) or isinstance(label_order[1], list):
        if verbose:
            print("Multiple sub-groups detected.")
        group_1 = np.array(label_order[0]).astype(str)
        group_2 = np.array(label_order[1]).astype(str)
        metadata = metadata.loc[np.isin(main_vals, np.concatenate([group_1, group_2]))].copy()
        main_vals = metadata[main_variable].astype(str).to_numpy()
        label_1 = "loc_group_one [" + " ".join(group_1) + "]"
        label_2 = "loc_group_two [" + " ".join(group_2) + "]"
        metadata["loc_tmp_group"] = np.where(np.isin(main_vals, group_1), label_1, label_2)
        main_variable = "loc_tmp_group"
        main_vals = metadata[main_variable].astype(str).to_numpy()
        label_order = [label_1, label_2]

    if not all(l in np.unique(main_vals) for l in label_order):
        missing = [l for l in label_order if l not in np.unique(main_vals)]
        raise ValueError(f"Some groups in label_order not found in data: {missing}")

    mask = np.isin(main_vals, list(label_order))
    metadata = metadata.loc[mask].copy()
    groups = metadata[main_variable].astype(str).to_numpy()
    covariable = metadata[subtype_variable].astype(str).to_numpy()
    samples = metadata[sample_id].astype(str).to_numpy()
    common = dict(groups=groups, covariable=covariable, samples=samples,
                  label_order=list(label_order), alternative=alternative,
                  seed=seed, verbose=verbose)
    if len(label_order) > 2:
        return _sample_gamma_path(**common)
    return _two_group_test(**common, subtype_variable=subtype_variable,
                           plot=plot, pdf_file=pdf_file)


def _two_group_test(groups, covariable, samples, label_order, alternative, seed,
                    verbose, subtype_variable, plot, pdf_file):
    indexes = list(_table(groups, covariable).columns)
    counts, _, group_idx, n_s = sample_count_matrix(
        groups, samples, covariable, label_order, indexes,
    )
    w = sample_weights(n_s, "sqrt", WEIGHT_CAP)
    props = counts / n_s[:, np.newaxis]                   # raw per-sample proportions
    eps = EPS_CELLS / float(np.median(n_s))

    def fc_of(means):
        return np.log2((means[0] + eps) / (means[1] + eps))

    rng = np.random.default_rng(seed)
    composition = group_weighted_means(group_idx[np.newaxis, :], props, w, 2)
    obs_fc = fc_of(composition)[0]
    assign, exact = relabellings(group_idx, N_RELABEL, N_RELABEL, rng)
    null_fc = fc_of(group_weighted_means(assign, props, w, 2))
    n_perms = assign.shape[0]

    p_vals = perm_pvalue(null_fc, obs_fc, exact, alternative)
    p_adj = _bh_fdr(p_vals)
    p_fwer = maxt_adjust(null_fc, obs_fc, exact, alternative)
    equal_sizes = (group_idx == 0).sum() == (group_idx == 1).sum()
    p_min = perm_pmin(n_perms, exact,
                      mirror_tie=equal_sizes and alternative == "two-sided")
    report_design(label_order, group_idx, n_perms, exact, p_min, verbose)

    boot_rng = np.random.default_rng(None if seed is None else int(seed) + 7919)
    boot_idx = bootstrap_indices(group_idx, N_BOOTSTRAP, boot_rng)
    boot_fc = fc_of(bootstrap_group_means(boot_idx, group_idx, props, w, 2))

    table_results = pd.DataFrame(
        {
            "groupFC": obs_fc,
            f"percent_in_{label_order[0]}": np.round(composition[0, 0], 3),
            f"percent_in_{label_order[1]}": np.round(composition[1, 0], 3),
            "p.adj": np.round(p_adj, 5),
            "p.adj.fwer": np.round(p_fwer, 5),
            "sd.null": null_fc.std(axis=0, ddof=1),
            "CI95low": np.quantile(boot_fc, 0.025, axis=0),
            "CI95high": np.quantile(boot_fc, 0.975, axis=0),
            "p.val": np.round(p_vals, 5),
        },
        index=indexes,
    )
    # one value per test, not per row: kept in attrs (and printed when verbose)
    table_results.attrs.update(alternative=alternative, exact=exact, p_min=p_min,
                               n_perms=n_perms, epsilon=eps)

    if plot:
        try:
            from .plots import plot_abundance_test
            plot_abundance_test(table_results, subtype_variable=subtype_variable,
                                pdf_file=pdf_file)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"(Plot skipped: {e})")
    return table_results


def _sample_gamma_path(groups, covariable, samples, label_order, alternative, seed,
                       verbose):
    """>2 ordered groups with the sample as the unit.

    Per cell type, Goodman–Kruskal gamma between each sample's group rank
    (position in ``label_order``) and its asin(√p) value. Rank-based, so
    sample weights do not enter the statistic (they are used for the
    reported group compositions). Null: sample labels permuted across
    groups. CI: samples resampled within groups.
    """
    indexes = list(_table(groups, covariable).columns)
    counts, _, group_idx, n_s = sample_count_matrix(
        groups, samples, covariable, label_order, indexes,
    )
    w = sample_weights(n_s, "sqrt", WEIGHT_CAP)
    props = per_sample_props(counts, "loc")
    t = asrt(props)
    rng = np.random.default_rng(seed)

    obs_gamma = gamma_per_column(t, group_idx[np.newaxis, :])[0]
    assign, exact = relabellings(group_idx, N_RELABEL, N_RELABEL, rng)
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
    boot_idx = bootstrap_indices(group_idx, N_BOOTSTRAP, boot_rng)
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
    df.attrs.update(alternative=alternative, exact=exact, p_min=p_min, n_perms=n_perms)
    return df
