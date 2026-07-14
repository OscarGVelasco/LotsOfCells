"""Gene-source decomposition test.

For a chosen gene G, the fraction of G's total transcripts in a sample
that originate from each cell type is a **joint** measure of:

- cell-type abundance in that sample, and
- per-cell-type transcriptional intensity for gene G.

Testing whether that per-cell-type fraction differs between patient groups
therefore asks a distinct question from the compositional tests in
:func:`lots_of_cells` (which only sees cell counts) — as noted by Sibai
et al. (2026) in the DUTRENEO study, source-defined bimodality of PDL1
and CTLA4 was not solely explained by cell-type abundance.

The test permutes **sample labels**, not cells:

- The unit of biological independence is the patient, not the cell.
- With cohorts of ≤ ~12 patients we can enumerate the null distribution
  exactly; with more we fall back to random sampling.

Two contrast modes:

- 2 labels → observed difference of group means; two-sided permutation
  p-value.
- >2 ordered labels → Kendall gamma rank correlation between the
  per-sample fractional contribution and the ordered group index.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._parallel import run_permutations
from ._utils import get_metadata, get_numerical_variable, get_palette, save_to_pdf


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

class _GeneSourceContext:
    """Bundle of context (contribution matrix, sample→group series, etc.)
    attached to a :func:`gene_source_score` result so the follow-up plot
    functions can be one-liners.

    Wrapped in a class rather than a plain dict because pandas' ``concat``
    (called internally when a wide DataFrame is truncated for terminal
    display) compares ``attrs == attrs`` element-wise. A raw DataFrame in
    the dict makes that comparison raise "ambiguous truth value" and
    breaks ``print(result)``. Overriding ``__eq__`` to identity keeps
    pandas happy.
    """

    __slots__ = ("contrib", "sample_to_group", "label_order", "gene")

    def __init__(self, contrib, sample_to_group, label_order, gene):
        self.contrib = contrib
        self.sample_to_group = sample_to_group
        self.label_order = list(label_order)
        self.gene = gene

    def __eq__(self, other):  # identity — safe inside dict comparisons
        return other is self

    def __hash__(self):
        return id(self)


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def _contribution_matrix(
    metadata: pd.DataFrame,
    gene_expr: np.ndarray,
    celltype_key: str,
    sample_id: str,
) -> pd.DataFrame:
    """Per-sample per-cell-type fractional contribution to gene total.

    Returns a DataFrame with samples on rows, cell types on columns, and
    each row summing to 1 (or NaN if the sample has 0 total).
    """
    df = metadata.assign(_expr=gene_expr)
    agg = (
        df.groupby([sample_id, celltype_key], observed=True)["_expr"]
          .sum()
          .unstack(fill_value=0)
    )
    totals = agg.sum(axis=1)
    return agg.div(totals.where(totals > 0, np.nan), axis=0)


def _kendall_gamma_per_column(values: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Kendall gamma between each column of `values` and integer `ranks`.

    Ties in either variable are excluded from concordant/discordant counts.
    """
    n, k = values.shape
    out = np.zeros(k)
    for j in range(k):
        val_rank = np.argsort(np.argsort(values[:, j]))
        c = d = 0
        for i in range(n - 1):
            s_g = np.sign(ranks[i] - ranks[i + 1 :])
            s_x = np.sign(val_rank[i] - val_rank[i + 1 :])
            mask = (s_g != 0) & (s_x != 0)
            c += int(np.sum((s_g == s_x) & mask))
            d += int(np.sum((s_g != s_x) & mask))
        out[j] = (c - d) / (c + d) if (c + d) > 0 else 0.0
    return out


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------

def gene_source_score(
    sc_object,
    gene: str,
    celltype_key: str,
    sample_id: str,
    main_variable: str,
    label_order: Sequence[str],
    layer: Optional[str] = None,
    permutations: int = 10000,
    exact_if_possible: bool = True,
    max_exact: int = 200_000,
    seed: Optional[int] = None,
    n_cores: Optional[int] = None,
    table: Optional[str] = None,
    plot: bool = True,
    pdf_file: Optional[str] = None,
    figsize=(11, 5),
    colors: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Test whether the cellular source of ``gene`` transcripts differs
    between groups.

    For each sample S and cell type C, compute::

        contribution(C, S) = sum(gene G expression in cells of type C in S)
                             / sum(gene G expression in all cells in S)

    Then test — via a **sample-level** permutation test — whether that
    fractional contribution differs between the groups in ``label_order``.

    Parameters
    ----------
    sc_object
        AnnData / SpatialData / MuData / DataFrame with per-cell metadata
        and gene expression.
    gene
        Gene name (must be in ``adata.var_names``), or column name in a
        DataFrame.
    celltype_key, sample_id, main_variable
        Column names in ``.obs`` (or the DataFrame).
    label_order
        Order of the contrast. 2 labels → difference-of-means test.
        >2 labels → Kendall gamma rank correlation between the fractional
        contribution and the ordered group index (e.g.
        ``["NO", "PARTIAL", "COMPLETE"]``).
    layer
        For AnnData: pull expression from ``adata.layers[layer]`` instead
        of ``.X``. Use raw counts if possible — fractional contributions
        are most interpretable then.
    permutations
        Random permutations when exact enumeration is infeasible.
    exact_if_possible
        If True and ``n_choose_k(n_samples, n_smaller_group) <= max_exact``
        (2-group mode), enumerate the null distribution exactly.
    max_exact
        Cap on the number of exact permutations before falling back to
        random sampling.
    seed
        Random seed for reproducibility.
    plot
        If True, render the Figure 2A-style stacked bar of the contribution
        matrix and save to ``pdf_file`` if given.

    Returns
    -------
    pandas.DataFrame indexed by cell type. Common columns:

    - ``mean_frac_<label>`` — mean fractional contribution per group
    - ``diff`` (2 groups) or ``gammaCor`` (>2 groups) — observed effect
    - ``p.val`` — permutation p-value (two-sided)
    - ``p.adj`` — BH-adjusted across cell types
    - ``n_perms_used`` — actual number of permutations
    """
    metadata = get_metadata(sc_object, table=table)
    metadata = metadata.loc[
        metadata[main_variable].astype(str).isin(label_order)
    ].copy()
    if len(metadata) == 0:
        raise ValueError(f"No cells match label_order={list(label_order)}")

    gene_expr = get_numerical_variable(sc_object, gene, metadata, layer=layer)

    # Sanity: warn if the expression clearly looks normalised (non-integer)
    finite = gene_expr[np.isfinite(gene_expr)]
    if len(finite) and verbose:
        sample_vals = finite[:2000]
        if not np.all(sample_vals == sample_vals.astype(int)):
            print(
                f"[gene_source_score] `{gene}` contains non-integer values. "
                "Fractional contributions still make sense, but consider "
                "passing layer='counts' if raw counts are available."
            )

    contrib = _contribution_matrix(metadata, gene_expr, celltype_key, sample_id)
    contrib = contrib.dropna(how="all")
    if len(contrib) < 2:
        raise ValueError(
            f"Only {len(contrib)} samples have non-zero `{gene}` expression — "
            "cannot run a permutation test."
        )
    contrib = contrib.fillna(0.0)

    # Sample → group lookup
    sample_to_group = (
        metadata.groupby(sample_id, observed=True)[main_variable]
                .first()
                .astype(str)
    )
    sample_to_group = sample_to_group.loc[contrib.index]

    if len(label_order) < 2:
        raise ValueError("label_order must have at least 2 entries")
    if len(label_order) == 2:
        result = _two_group_test(
            contrib, sample_to_group, label_order,
            permutations, exact_if_possible, max_exact,
            seed, n_cores, verbose,
        )
    else:
        result = _gamma_rank_test(
            contrib, sample_to_group, label_order,
            permutations, seed, n_cores, verbose,
        )

    # Attach the derived inputs first so the boxplot's one-liner form works
    # and so downstream callers can access `contrib` / `sample_to_group`
    # without recomputing. Everything is wrapped in a single class so
    # pandas' internal `attrs == attrs` comparison stays safe (see the
    # docstring on `_GeneSourceContext`).
    result.attrs["_gsc_context"] = _GeneSourceContext(
        contrib=contrib, sample_to_group=sample_to_group,
        label_order=label_order, gene=gene,
    )

    if plot:
        try:
            import matplotlib.pyplot as plt

            # Default combined figure — two stacked panels:
            #   top: Figure 2A-style stacked bar per sample
            #   bottom: single-panel boxplot per cell type
            # Constrained_layout handles the outside-axes legends without
            # tight_layout's "not compatible" warning.
            width = (figsize[0] if figsize else 12.0)
            fig, (ax_bar, ax_box) = plt.subplots(
                2, 1,
                figsize=(width, 10.0),
                gridspec_kw={"height_ratios": [1, 1]},
                constrained_layout=True,
            )
            gene_source_bar_chart(
                contrib, sample_to_group, label_order, gene,
                colors=colors, ax=ax_bar,
            )
            gene_source_boxplot(
                result, ax=ax_box, colors=colors,
            )
            save_to_pdf(fig, pdf_file)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[gene_source_score] plot skipped: {e}")

    return result


# ---------------------------------------------------------------------
# Statistical backends
# ---------------------------------------------------------------------

def _two_group_test(
    contrib: pd.DataFrame,
    sample_to_group: pd.Series,
    label_order: Sequence[str],
    permutations: int,
    exact_if_possible: bool,
    max_exact: int,
    seed: Optional[int],
    n_cores: Optional[int],
    verbose: bool,
) -> pd.DataFrame:
    groups = sample_to_group.to_numpy()
    n_total = len(groups)
    n_a = int((groups == label_order[0]).sum())
    n_b = int((groups == label_order[1]).sum())
    if n_a == 0 or n_b == 0:
        raise ValueError(
            f"Empty group in label_order: {label_order[0]}={n_a}, "
            f"{label_order[1]}={n_b}"
        )

    values = contrib.to_numpy()
    obs_diff = values[groups == label_order[0]].mean(axis=0) - values[
        groups == label_order[1]
    ].mean(axis=0)

    n_exact = comb(n_total, min(n_a, n_b))
    use_exact = exact_if_possible and n_exact <= max_exact
    if use_exact:
        # Exact enumeration is deterministic — parallelising it just adds
        # IPC overhead for no gain, so it stays serial.
        null = np.empty((n_exact, values.shape[1]))
        for k, combo in enumerate(combinations(range(n_total), n_a)):
            idx_a = np.array(combo)
            mask_a = np.zeros(n_total, dtype=bool)
            mask_a[idx_a] = True
            null[k] = values[mask_a].mean(axis=0) - values[~mask_a].mean(axis=0)
        n_used = n_exact
        if verbose:
            print(f"[gene_source_score] exact enumeration ({n_exact:,} permutations)")
    else:
        def _perm_chunk(chunk_size: int, seed_seq) -> np.ndarray:
            rng = np.random.default_rng(seed_seq)
            out = np.empty((chunk_size, values.shape[1]))
            for k in range(chunk_size):
                perm = rng.permutation(n_total)
                idx_a = perm[:n_a]
                mask_a = np.zeros(n_total, dtype=bool)
                mask_a[idx_a] = True
                out[k] = values[mask_a].mean(axis=0) - values[~mask_a].mean(axis=0)
            return out

        null = run_permutations(
            _perm_chunk, permutations,
            n_cores=n_cores, seed=seed, verbose=verbose,
        )
        n_used = permutations
        if verbose:
            print(
                f"[gene_source_score] random sampling: {permutations:,} "
                f"permutations (exact would need {n_exact:,})"
            )

    # Two-sided p-value with +1/+1 correction (never returns 0)
    obs_abs = np.abs(obs_diff)
    p_vals = (np.sum(np.abs(null) >= obs_abs, axis=0) + 1) / (n_used + 1)
    p_adj = _bh_fdr(p_vals)

    out = pd.DataFrame(
        {
            f"mean_frac_{label_order[0]}": values[groups == label_order[0]].mean(axis=0),
            f"mean_frac_{label_order[1]}": values[groups == label_order[1]].mean(axis=0),
            "diff": obs_diff,
            "p.val": np.round(p_vals, 5),
            "p.adj": np.round(p_adj, 5),
            "n_perms_used": n_used,
        },
        index=contrib.columns,
    )
    return out.iloc[np.argsort(-np.abs(out["diff"].to_numpy()))]


def _gamma_rank_test(
    contrib: pd.DataFrame,
    sample_to_group: pd.Series,
    label_order: Sequence[str],
    permutations: int,
    seed: Optional[int],
    n_cores: Optional[int],
    verbose: bool,
) -> pd.DataFrame:
    rank_map = {lbl: k for k, lbl in enumerate(label_order)}
    ranks = np.array([rank_map[g] for g in sample_to_group.to_numpy()])
    values = contrib.to_numpy()

    obs_gamma = _kendall_gamma_per_column(values, ranks)

    def _perm_chunk(chunk_size: int, seed_seq) -> np.ndarray:
        rng = np.random.default_rng(seed_seq)
        out = np.empty((chunk_size, values.shape[1]))
        for k in range(chunk_size):
            out[k] = _kendall_gamma_per_column(values, rng.permutation(ranks))
        return out

    null = run_permutations(
        _perm_chunk, permutations,
        n_cores=n_cores, seed=seed, verbose=verbose,
    )

    p_vals = (np.sum(np.abs(null) >= np.abs(obs_gamma), axis=0) + 1) / (permutations + 1)
    p_adj = _bh_fdr(p_vals)

    means = {
        f"mean_frac_{lbl}": values[ranks == r].mean(axis=0)
        for r, lbl in enumerate(label_order)
    }
    out = pd.DataFrame(means, index=contrib.columns)
    out["gammaCor"] = obs_gamma
    out["p.val"] = np.round(p_vals, 5)
    out["p.adj"] = np.round(p_adj, 5)
    out["n_perms_used"] = permutations
    return out.iloc[np.argsort(-np.abs(out["gammaCor"].to_numpy()))]


# ---------------------------------------------------------------------
# Plot — Figure 2A style
# ---------------------------------------------------------------------

def gene_source_bar_chart(
    contrib: pd.DataFrame,
    sample_to_group: pd.Series,
    label_order: Sequence[str],
    gene: str,
    colors: Optional[Sequence[str]] = None,
    figsize=(11, 5),
    pdf_file: Optional[str] = None,
    ax=None,
):
    """Stacked bar per sample showing the cellular source of `gene` transcripts.

    Mirrors Figure 2A of Sibai et al. 2026 (DUTRENEO): each bar is a
    patient, each segment is a cell type, segment height = fraction of
    `gene`'s total transcripts contributed by that cell type. Bars are
    grouped and ordered by ``label_order``, with a coloured band and
    label below the axis marking group boundaries.

    Pass ``ax=`` to draw into an existing ``Axes`` (used by
    :func:`gene_source_score` to compose the combined 2-panel figure).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    groups = sample_to_group.loc[contrib.index]

    # Order: bars grouped by label_order, within each group sorted by
    # descending contribution of the group-mean-dominant cell type.
    order = []
    for lbl in label_order:
        idx = contrib.index[groups == lbl]
        if len(idx) == 0:
            continue
        sub = contrib.loc[idx]
        top_col = sub.mean().idxmax()
        sub = sub.sort_values(top_col, ascending=False)
        order.extend(sub.index.tolist())
    contrib = contrib.loc[order]
    groups = sample_to_group.loc[contrib.index]

    cell_types = list(contrib.columns)
    palette = get_palette(use_palette=colors, n_colors=len(cell_types))
    color_map = dict(zip(cell_types, palette))

    ax_provided = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    bottom = np.zeros(len(contrib))
    for ct in cell_types:
        vals = contrib[ct].values
        ax.bar(
            range(len(contrib)), vals, bottom=bottom,
            color=color_map[ct], label=ct, width=0.85, linewidth=0,
        )
        bottom += vals

    ax.set_xticks(range(len(contrib)))
    ax.set_xticklabels(contrib.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"Fraction of {gene} transcripts")
    ax.set_ylim(-0.09, 1.0)
    ax.set_title(f"Cellular source of {gene} transcripts")

    # Group band below the axis
    ymin, ymax = -0.06, -0.02
    band_palette = get_palette(n_colors=max(len(label_order), 3))
    band_map = dict(zip(label_order, band_palette))
    runs = []
    start = 0
    for i in range(1, len(groups)):
        if groups.iloc[i] != groups.iloc[i - 1]:
            runs.append((start, i - 1, groups.iloc[start]))
            start = i
    runs.append((start, len(groups) - 1, groups.iloc[start]))
    for s, e, lbl in runs:
        ax.add_patch(mpatches.Rectangle(
            (s - 0.5, ymin), e - s + 1, ymax - ymin,
            color=band_map[lbl], clip_on=False, zorder=3,
        ))
        ax.text(
            (s + e) / 2, (ymin + ymax) / 2, lbl,
            ha="center", va="center", color="white",
            fontsize=9, fontweight="bold", zorder=4,
        )

    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        title="Cell type", fontsize=8,
    )
    if not ax_provided:
        fig.tight_layout()
    save_to_pdf(fig, pdf_file)
    return ax


# ---------------------------------------------------------------------
# Plot — Figure S6C style: per-cell-type boxplot across groups
# ---------------------------------------------------------------------

def _pval_stars(p: float) -> str:
    """Common star notation for p-values."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def gene_source_boxplot(
    result_or_contrib,
    sample_to_group: Optional[pd.Series] = None,
    label_order: Optional[Sequence[str]] = None,
    gene: Optional[str] = None,
    result: Optional[pd.DataFrame] = None,
    colors: Optional[Sequence[str]] = None,
    figsize: Optional[tuple] = None,
    show_points: bool = True,
    show_pvalues: bool = True,
    pdf_file: Optional[str] = None,
    seed: Optional[int] = 0,
    ax=None,
):
    """Single-panel per-cell-type boxplot of fractional contribution.

    All cell types share one panel: the x-axis is the cell type (tilted
    45°), and at each x position there is one box per level of
    ``main_variable`` in ``label_order``. This mirrors Figure S6C of
    Sibai et al. 2026 (DUTRENEO) but keeps every cell type visually
    comparable in a single view.

    Individual samples are overlaid as jittered points so small cohorts
    remain readable.

    Two usage patterns:

    - **One-liner** — pass the DataFrame returned by
      :func:`gene_source_score` directly. All other inputs (``contrib``,
      ``sample_to_group``, ``label_order``, ``gene``) are read from the
      result's ``.attrs``::

          res = loc.gene_source_score(adata, gene='PDL1', ...)
          loc.gene_source_boxplot(res)

    - **Manual** — pass ``contrib`` and the accompanying series
      explicitly.

    Parameters
    ----------
    result_or_contrib
        Either the ``pandas.DataFrame`` returned by
        :func:`gene_source_score` (preferred; carries everything needed in
        ``.attrs``), or a plain ``samples × cell types`` contribution
        matrix. In the latter case you must also pass ``sample_to_group``,
        ``label_order`` and ``gene``.
    sample_to_group
        Series mapping sample id → group label.
    label_order
        Group order shown side-by-side within each cell type.
    gene
        Gene name, used in the figure title and y-axis label only.
    result
        Optional DataFrame from :func:`gene_source_score` (only needed if
        you passed a raw contribution matrix as the first argument and
        still want p-value annotations / effect-size ordering).
    colors
        Palette for the group boxes. Uses :func:`get_palette` if omitted.
    figsize
        ``(width, height)`` for the figure. Auto-sized by default.
    show_points
        Overlay individual sample values as jittered dots.
    show_pvalues
        Annotate each cell type with the BH-adjusted p-value when
        ``result`` is provided.
    pdf_file
        Optional output PDF path.
    seed
        Random seed for the point-jitter (visual only).
    ax
        If provided, draw into this Matplotlib ``Axes`` instead of creating
        a new figure. Used by :func:`gene_source_score` to compose the
        combined bar + boxplot figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Resolve the one-liner form: user passed the result DataFrame directly.
    ctx = None
    if isinstance(result_or_contrib, pd.DataFrame):
        ctx = result_or_contrib.attrs.get("_gsc_context")
    if ctx is not None:
        result = result_or_contrib
        contrib = ctx.contrib
        if sample_to_group is None:
            sample_to_group = ctx.sample_to_group
        if label_order is None:
            label_order = ctx.label_order
        if gene is None:
            gene = ctx.gene
    else:
        contrib = result_or_contrib
        if sample_to_group is None or label_order is None or gene is None:
            raise ValueError(
                "When passing a raw contribution matrix, `sample_to_group`, "
                "`label_order` and `gene` must all be provided."
            )

    # Cell-type order: by |effect size| from `result` if given, else by
    # descending mean contribution.
    if result is not None:
        effect_col = "diff" if "diff" in result.columns else "gammaCor"
        if effect_col in result.columns:
            ordered = result.reindex(
                result[effect_col].abs().sort_values(ascending=False).index
            ).index.tolist()
            cell_types = [c for c in ordered if c in contrib.columns]
        else:
            cell_types = list(result.index)
    else:
        cell_types = list(contrib.mean().sort_values(ascending=False).index)

    n_types = len(cell_types)
    n_groups = len(label_order)

    # Grouped-box layout: each cell type occupies a "slot" of unit width;
    # inside that slot the n_groups boxes are centred, each `box_width`
    # wide, spaced by `cluster_span / n_groups`. Adjacent cell types stay
    # visually separated.
    cluster_span = 0.72
    slot_step = cluster_span / n_groups
    box_width = slot_step * 0.9
    group_offsets = (np.arange(n_groups) - (n_groups - 1) / 2) * slot_step

    ax_provided = ax is not None
    if ax is None:
        if figsize is None:
            figsize = (max(1.15 * n_types + 3.0, 8.0), 5.0)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    band_palette = get_palette(use_palette=colors, n_colors=max(n_groups, 3))
    band_map = dict(zip(label_order, band_palette))

    groups = sample_to_group.loc[contrib.index]
    jitter_rng = np.random.default_rng(seed)

    for i, ct in enumerate(cell_types):
        for g, lbl in enumerate(label_order):
            data = contrib.loc[groups == lbl, ct].dropna().to_numpy()
            if len(data) == 0:
                continue
            x_pos = i + group_offsets[g]

            bp = ax.boxplot(
                [data],
                positions=[x_pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"color": "#666666", "linewidth": 0.8},
                capprops={"color": "#666666", "linewidth": 0.8},
                boxprops={"linewidth": 0.6, "edgecolor": "#666666"},
            )
            bp["boxes"][0].set_facecolor(band_map[lbl])
            bp["boxes"][0].set_alpha(0.4)

            if show_points:
                jitter = jitter_rng.uniform(
                    -box_width * 0.3, box_width * 0.3, size=len(data)
                )
                ax.scatter(
                    x_pos + jitter, data,
                    s=16, color=band_map[lbl],
                    edgecolor="white", linewidth=0.4,
                    alpha=0.85, zorder=3,
                )

    # X-axis: one tick per cell type, tilted 45°.
    ax.set_xticks(np.arange(n_types))
    ax.set_xticklabels(cell_types, rotation=45, ha="right", fontsize=9)
    ax.set_xlim(-0.5, n_types - 0.5)

    ax.set_ylabel(f"Fraction of {gene} transcripts", fontsize=9)
    ax.set_ylim(-0.03, 1.10)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)

    # P-value annotations, centred over each cell type slot.
    if show_pvalues and result is not None and "p.adj" in result.columns:
        for i, ct in enumerate(cell_types):
            if ct not in result.index:
                continue
            p = float(result.loc[ct, "p.adj"])
            if not np.isfinite(p):
                continue
            stars = _pval_stars(p)
            ax.text(
                i, 1.03, f"p={p:.3g} {stars}".strip(),
                ha="center", va="bottom", fontsize=7,
                fontweight="bold" if p < 0.05 else "normal",
                color="#444444",
            )

    handles = [
        mpatches.Patch(color=band_map[l], alpha=0.6, label=l)
        for l in label_order
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Group",
        fontsize=8,
        frameon=False,
    )
    ax.set_title(
        f"Cell-type source of {gene} transcripts, by group",
        fontsize=10,
    )

    if not ax_provided:
        fig.tight_layout()
    save_to_pdf(fig, pdf_file)
    return fig
