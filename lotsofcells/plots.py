"""Visualizations: bar, waffle, polar, density-ridge, dynamics, abundance test.

All functions accept either an `AnnData` (or `SpatialData`/`MuData`) or a
`pandas.DataFrame` containing the metadata.

Returns either a `matplotlib.figure.Figure` or `matplotlib.axes.Axes`.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from ._utils import (
    desaturate,
    get_metadata,
    get_numerical_variable,
    get_palette,
    lighten,
    save_to_pdf,
)


# ------------------------------------------------------------------
# Bar chart
# ------------------------------------------------------------------

def bar_chart(
    sc_object,
    main_variable: str,
    subtype_variable: str,
    sample_id: Optional[str] = None,
    subtype_only: Optional[str] = None,
    contribution: bool = False,
    colors: Optional[Sequence[str]] = None,
    table: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize=(7, 5),
    pdf_file: Optional[str] = None,
):
    """Stacked barplot of subtype proportions per main_variable level.

    Mirrors the R ``bar_chart``. Pass ``pdf_file="path.pdf"`` to also save
    the figure to disk.
    """
    metadata = get_metadata(sc_object, table=table)
    groups = metadata[main_variable].astype(str)
    covariable = metadata[subtype_variable].astype(str)
    order = list(covariable.value_counts(ascending=True).index)  # smallest first
    palette = get_palette(use_palette=colors, n_colors=len(order))
    color_map = dict(zip(order[::-1], palette))  # largest avg → first color

    if sample_id is not None:
        samples = metadata[sample_id].astype(str)
        if contribution:
            return _bar_chart_contribution(
                groups, covariable, samples, order, color_map,
                main_variable, subtype_variable, sample_id, figsize,
                pdf_file=pdf_file,
            )
        bar_keys = (groups + "_" + samples).to_numpy()
        df = pd.DataFrame({"groups": bar_keys, "covariable": covariable.values})
        labels_main = groups
    else:
        df = pd.DataFrame({"groups": groups.values, "covariable": covariable.values})
        labels_main = groups

    contig = pd.crosstab(df["groups"], df["covariable"])
    contig = contig.div(contig.sum(axis=1), axis=0)
    if subtype_only is not None:
        contig = contig[[subtype_only]]

    # Order bars within main group by descending value of largest covariable
    bar_keys = list(contig.index)
    if sample_id is not None:
        bar_main = [k.split("_")[0] for k in bar_keys]
    else:
        bar_main = bar_keys
    main_levels = sorted(set(bar_main))

    if subtype_only is None:
        sort_col = order[-1]
    else:
        sort_col = subtype_only
    sort_vals = contig[sort_col]
    bar_order = sorted(
        bar_keys,
        key=lambda k: (
            main_levels.index(k.split("_")[0] if sample_id is not None else k),
            -sort_vals[k],
        ),
    )
    contig = contig.loc[bar_order]
    bar_main = [k.split("_")[0] if sample_id is not None else k for k in bar_order]

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    bottom = np.zeros(len(contig))
    cov_order = order[::-1]  # largest at bottom
    if subtype_only is not None:
        cov_order = [subtype_only]

    # If subtype_only, color bars by main group
    if subtype_only is not None:
        group_colors = _group_colors(main_levels)
        gc_map = {m: c for m, c in zip(main_levels, group_colors)}
        bar_colors = [gc_map[m] for m in bar_main]
        ax.bar(range(len(contig)), contig[subtype_only].values, color=bar_colors)
    else:
        for cov in cov_order:
            ax.bar(
                range(len(contig)),
                contig[cov].values,
                bottom=bottom,
                color=color_map[cov],
                label=cov,
            )
            bottom += contig[cov].values
        ax.legend(title=f"Class: {subtype_variable}", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax.set_xticks(range(len(contig)))
    ax.set_xticklabels(contig.index, rotation=45, ha="right")
    ax.set_ylabel("percentage")
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels([f"{int(100 * v)}" for v in np.linspace(0, 1, 11)])
    title = f"Proportions of {subtype_variable} by {main_variable}"
    if subtype_only:
        ax.set_title(f"{title}\nClass: {subtype_only}")
    elif sample_id:
        ax.set_title(f"{title}\nIndividual sub-level by: {sample_id}")
    else:
        ax.set_title(title)

    # Annotate group bands at the bottom
    _annotate_groups(ax, bar_main, main_levels)
    fig.tight_layout()
    save_to_pdf(fig, pdf_file)
    return ax


def _group_colors(levels):
    base = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854",
            "#FFD92F", "#E5C494", "#B3B3B3"]
    palette = get_palette(use_palette=base, n_colors=len(levels))
    return [desaturate(c, 0.16) for c in palette]


def _annotate_groups(ax, bar_main, main_levels):
    palette = _group_colors(main_levels)
    color_for = dict(zip(main_levels, palette))
    n = len(bar_main)
    ymin, ymax = -0.05, -0.02
    runs = []
    start = 0
    for i in range(1, n):
        if bar_main[i] != bar_main[i - 1]:
            runs.append((start, i - 1, bar_main[start]))
            start = i
    runs.append((start, n - 1, bar_main[start]))
    for s, e, lbl in runs:
        ax.add_patch(mpatches.Rectangle(
            (s - 0.5, ymin), (e - s + 1), (ymax - ymin),
            color=color_for[lbl], clip_on=False, zorder=3,
        ))
        ax.text((s + e) / 2, (ymin + ymax) / 2, lbl,
                ha="center", va="center", color="white",
                fontsize=8, style="italic", zorder=4)


def _bar_chart_contribution(
    groups, covariable, samples, order, color_map,
    main_variable, subtype_variable, sample_id, figsize,
    pdf_file=None,
):
    fig, ax = plt.subplots(figsize=figsize)
    main_levels = sorted(groups.unique())
    width = 0.7
    for i, m in enumerate(main_levels):
        sub = (groups == m)
        df = pd.DataFrame({
            "samples": samples[sub].values,
            "covariable": covariable[sub].values,
        })
        contig = pd.crosstab(df["samples"], df["covariable"])
        contig = contig / contig.values.sum()
        bottom = 0.0
        cov_order = order[::-1]
        for cov in cov_order:
            base_color = color_map[cov]
            samples_present = list(contig.index)
            n_s = len(samples_present)
            if n_s == 0:
                continue
            shades = [
                lighten(base_color, t)
                for t in np.linspace(-0.2, 0.2, n_s)
            ]
            shades = [c if not c.startswith("-") else base_color for c in shades]
            for idx, s in enumerate(samples_present):
                v = contig.loc[s, cov] if cov in contig.columns else 0
                ax.bar(i, v, width, bottom=bottom, color=shades[idx])
                bottom += v
    ax.set_xticks(range(len(main_levels)))
    ax.set_xticklabels(main_levels, rotation=45, ha="right")
    ax.set_ylabel("percentage")
    ax.set_title(
        f"Proportions of {subtype_variable} by {main_variable}\n"
        f"Contribution by {sample_id}"
    )
    handles = [
        mpatches.Patch(color=color_map[c], label=c) for c in order[::-1]
    ]
    ax.legend(
        handles=handles, title=f"Class: {subtype_variable}",
        bbox_to_anchor=(1.02, 1), loc="upper left",
    )
    fig.tight_layout()
    save_to_pdf(fig, pdf_file)
    return ax


# ------------------------------------------------------------------
# Waffle chart (each tile = 1%)
# ------------------------------------------------------------------

def waffle_chart(
    sc_object,
    main_variable: str,
    subtype_variable: str,
    sample_id: Optional[str] = None,
    subtype_only: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    table: Optional[str] = None,
    figsize=None,
    pdf_file: Optional[str] = None,
):
    metadata = get_metadata(sc_object, table=table)
    groups = metadata[main_variable].astype(str)
    covariable = metadata[subtype_variable].astype(str)

    if subtype_only is not None:
        if subtype_only not in covariable.unique():
            raise ValueError(
                f"subtype_only '{subtype_only}' not found in {subtype_variable}."
            )
        cov = np.where(covariable == subtype_only, subtype_only, "All Other")
        order = [subtype_only, "All Other"]
        # alternating dim/main shades per main group
        coloresSubtype = [
            "#DBECDA", "#92C791", "#BEDAEC", "#7EB6D9", "#DDC7E2", "#86608E",
        ]
        coloresSubtype = [desaturate(c, 0.16) for c in coloresSubtype]
        subtype_palette = coloresSubtype
    else:
        cov = covariable.to_numpy()
        order = list(covariable.value_counts(ascending=False).index)[::-1]
        subtype_palette = None

    if sample_id is not None:
        keys = (groups + "_" + metadata[sample_id].astype(str)).to_numpy()
    else:
        keys = groups.to_numpy()

    df = pd.DataFrame({"groups": keys, "covariable": cov})
    contig = pd.crosstab(df["groups"], df["covariable"])
    if subtype_only is not None:
        ncells = contig.get(subtype_only, pd.Series(0, index=contig.index))
    else:
        ncells = contig.sum(axis=1)
    contig = contig.div(contig.sum(axis=1), axis=0)
    contig = contig.reindex(columns=order, fill_value=0)

    palette = (
        get_palette(use_palette=colors, n_colors=len(order))
        if subtype_palette is None
        else subtype_palette
    )
    n_panels = len(contig)
    ncol = max(1, int(np.ceil(np.sqrt(n_panels))))
    nrow = int(np.ceil(n_panels / ncol))
    if figsize is None:
        # +1 column reserved for the legend → 3*ncol for waffles, 1.5 for legend
        figsize = (3 * ncol + 1.8, 3 * nrow)

    # GridSpec: nrow x (ncol + 1). The last column is a dedicated, axis-off
    # area where the legend lives, so it never overlaps the waffles.
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrow, ncol + 1,
        width_ratios=[1.0] * ncol + [0.45],
        wspace=0.15, hspace=0.25,
    )
    axes = np.empty((nrow, ncol), dtype=object)
    for r in range(nrow):
        for c in range(ncol):
            axes[r, c] = fig.add_subplot(gs[r, c])
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis("off")

    # Map main groups → color pair indices for subtype_only mode
    main_order = sorted({k.split("_")[0] if sample_id is not None else k
                         for k in contig.index})
    main_idx = {m: i for i, m in enumerate(main_order)}

    i = -1
    for i, (gname, row) in enumerate(contig.iterrows()):
        ax = axes[i // ncol][i % ncol]
        percentages = (row * 100).round().astype(int).to_numpy()
        percentages = _balance_to_100(percentages)
        if subtype_only is not None:
            mg = gname.split("_")[0] if sample_id is not None else gname
            pi = main_idx[mg] * 2
            colors_panel = [palette[pi % len(palette)],
                            palette[(pi + 1) % len(palette)]]
        else:
            colors_panel = [palette[order.index(o)] for o in order]
        _draw_waffle(ax, percentages, colors_panel, order, gname, ncells.get(gname, 0))
        if subtype_only is not None and len(percentages) >= 1:
            ax.text(4.5, 8, f"{percentages[0]:.0f}%",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=colors_panel[0])

    # Hide unused panels in the bottom-right of the waffle grid.
    for j in range(i + 1, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    handles = [
        mpatches.Patch(
            color=(palette[order.index(o)] if subtype_only is None
                   else (palette[1] if o == subtype_only else palette[0])),
            label=o,
        )
        for o in order
    ]
    legend_ax.legend(
        handles=handles,
        title=f"Class: {subtype_variable}",
        loc="center",
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
    )

    save_to_pdf(fig, pdf_file)

    # Display once in Jupyter inline (the inline backend auto-flushes
    # at cell end AND the Figure has `_repr_*_` that re-renders if
    # returned). Show the figure here, then close it so the cell does
    # not re-display, and return None.
    try:
        from matplotlib import get_backend
        if "inline" in get_backend().lower():
            from IPython.display import display
            display(fig)
            plt.close(fig)
            return None
    except Exception:
        pass
    return None


def _balance_to_100(arr):
    arr = arr.astype(int)
    diff = 100 - int(arr.sum())
    if diff == 0:
        return arr
    arr = arr.copy()
    if diff > 0:
        arr[arr.argmax()] += diff
    else:
        arr[arr.argmax()] += diff
    return arr


def _draw_waffle(ax, percentages, colors_panel, order, title, ncells):
    grid = np.zeros(100, dtype=int)
    cum = np.cumsum(percentages)
    for k in range(100):
        grid[k] = int(np.searchsorted(cum, k, side="right"))
    grid = grid.reshape(10, 10)
    for i in range(10):
        for j in range(10):
            idx = grid[i, j]
            idx = min(idx, len(colors_panel) - 1)
            ax.add_patch(mpatches.Rectangle(
                (j, i), 0.85, 0.85, color=colors_panel[idx],
            ))
    ax.set_xlim(-0.1, 10)
    ax.set_ylim(-0.1, 10)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(title))
    if ncells:
        ax.set_xlabel(f"n. cells: {int(ncells):,}",
                      fontsize=8, style="italic", color="grey")


# ------------------------------------------------------------------
# Polar / circular barplot
# ------------------------------------------------------------------

def polar_chart(
    sc_object,
    main_variable: str,
    subtype_variable: str,
    sample_id: Optional[str] = None,
    subtype_only: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    table: Optional[str] = None,
    figsize=(8, 8),
    pdf_file: Optional[str] = None,
):
    metadata = get_metadata(sc_object, table=table)
    groups = metadata[main_variable].astype(str)
    covariable = metadata[subtype_variable].astype(str)
    order = list(covariable.value_counts(ascending=True).index)  # smallest first
    palette = get_palette(use_palette=colors, n_colors=len(order))
    color_map = dict(zip(order[::-1], palette))

    if sample_id is not None:
        keys = (groups + "_" + metadata[sample_id].astype(str)).to_numpy()
    else:
        keys = groups.to_numpy()
    df = pd.DataFrame({"groups": keys, "covariable": covariable.values})
    contig = pd.crosstab(df["groups"], df["covariable"])
    if subtype_only is not None:
        contig = contig[[subtype_only]]

    # Sort by main_group then preserve order
    if sample_id is not None:
        main_levels = sorted({k.split("_")[0] for k in contig.index})
        contig = contig.reindex(
            sorted(contig.index, key=lambda k: (main_levels.index(k.split("_")[0]), k))
        )
    else:
        contig = contig.sort_index()

    n_bars = len(contig)
    angles = np.linspace(0, 2 * np.pi, n_bars, endpoint=False)
    width = 2 * np.pi / n_bars * 0.9

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    bottom = np.zeros(n_bars)
    cov_order = order[::-1]
    if subtype_only is not None:
        cov_order = [subtype_only]
    for cov in cov_order:
        if cov not in contig.columns:
            continue
        vals = contig[cov].values
        ax.bar(angles, vals, width=width, bottom=bottom,
               color=color_map.get(cov, "#999999"), label=cov, edgecolor="white")
        bottom += vals

    ax.set_xticks(angles)
    ax.set_xticklabels(contig.index, fontsize=7)
    ax.set_yticklabels([])
    ax.set_title(f"Proportions of {subtype_variable} by {main_variable}",
                 fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.2, 1), loc="upper left",
              title=f"Class: {subtype_variable}")
    fig.tight_layout()
    save_to_pdf(fig, pdf_file)
    return ax


# ------------------------------------------------------------------
# Density (ridge) chart
# ------------------------------------------------------------------

def density_chart(
    sc_object,
    main_variable: str,
    subtype_variable: str,
    numerical_variable: str,
    sample_id: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    table: Optional[str] = None,
    figsize=(9, 7),
    pdf_file: Optional[str] = None,
):
    """Ridge-style density plot of a numerical variable across covariate levels.

    The numerical variable can be a column in ``.obs`` or — when ``sc_object``
    is an AnnData — a feature name (gene); expression values from ``.X`` will
    be used.
    """
    metadata = get_metadata(sc_object, table=table)
    metadata = metadata.dropna(subset=[subtype_variable])
    groups = metadata[main_variable].astype(str)
    covariable = metadata[subtype_variable].astype(str)
    order = list(covariable.value_counts(ascending=True).index)
    palette = get_palette(use_palette=colors, n_colors=len(order))
    color_map = dict(zip(order, palette))

    values = get_numerical_variable(sc_object, numerical_variable, metadata)
    metadata = metadata.assign(_val=values)
    metadata = metadata.dropna(subset=["_val"])

    if sample_id is not None:
        sub_label = (covariable + "_" + groups + "_" + metadata[sample_id].astype(str))
    else:
        sub_label = covariable + "_" + groups

    levels = []
    for o in order:
        sub_levels = sorted(sub_label[covariable == o].unique())
        levels.extend(sub_levels)
    metadata = metadata.assign(_label=sub_label)
    metadata["_label"] = pd.Categorical(metadata["_label"], categories=levels, ordered=True)

    fig, ax = plt.subplots(figsize=figsize)
    overlap = 0.7
    y = 0
    for label in levels:
        cov_name = label.split("_")[0]
        color = color_map[cov_name]
        vals = metadata.loc[metadata["_label"] == label, "_val"].to_numpy()
        if len(vals) < 2:
            y += 1
            continue
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(vals)
            xs = np.linspace(np.min(vals), np.max(vals), 200)
            ys = kde(xs)
            ys = ys / ys.max() * (1 + overlap)
            ax.fill_between(xs, y, y + ys, color=color, alpha=0.6, lw=0)
            ax.plot(xs, y + ys, color=color, lw=0.5)
            med = np.median(vals)
            ax.vlines(med, y, y + np.interp(med, xs, ys), color="black", lw=0.7)
        except Exception:
            pass
        y += 1

    ax.set_yticks(np.arange(len(levels)) + 0.3)
    ax.set_yticklabels(levels, fontsize=8)
    ax.set_xlabel(numerical_variable)
    title = f"Density distribution of {numerical_variable} across {subtype_variable}"
    if sample_id:
        title += f"\nsplit across {sample_id}"
    ax.set_title(title)
    fig.tight_layout()
    save_to_pdf(fig, pdf_file)
    return ax


# ------------------------------------------------------------------
# Dynamics chart (proportion trends across >2 conditions)
# ------------------------------------------------------------------

def dynamics_chart(
    gamma_results: pd.DataFrame,
    scale_data: bool = False,
    figsize=(10, 8),
    pdf_file: Optional[str] = None,
):
    """Visualize per-cell-type proportion dynamics across ordered groups."""
    df = gamma_results.copy()
    summary_cols = {"groupGammaCor", "p.adj", "CI95low", "CI95high"}
    pct_cols = [c for c in df.columns if c not in summary_cols]
    if scale_data:
        scaled = df[pct_cols].apply(lambda r: (r - r.mean()) / r.std(ddof=1), axis=1)
        df[pct_cols] = scaled

    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    palette = get_palette(n_colors=len(df))
    color_map = dict(zip(df.index, palette))
    x_labels = [c.replace("percent_in_", "proportion ") for c in pct_cols]
    for cov in df.index:
        ax.plot(range(len(pct_cols)), df.loc[cov, pct_cols].values,
                marker="s", color=color_map[cov], label=cov, lw=1.2)
        ax.text(len(pct_cols) - 0.95, df.loc[cov, pct_cols[-1]],
                f"cor. {df.loc[cov, 'groupGammaCor']:.2f}",
                fontsize=8, fontweight="bold", color=color_map[cov])
    ax.set_xticks(range(len(pct_cols)))
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel("proportion")
    ax.set_title("Proportion dynamics across groups", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    ax2 = axes[1]
    sorted_cov = df.sort_values("groupGammaCor").index.tolist()
    for cov in sorted_cov:
        ax2.scatter(cov, df.loc[cov, "groupGammaCor"],
                    s=120, color=color_map[cov], edgecolor="black", zorder=4)
    ax2.axhline(0, color="darkgrey", lw=0.6)
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("Kendall correlation")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save_to_pdf(fig, pdf_file)
    # Same Jupyter inline double-render issue as `waffle_chart` — display
    # once explicitly and close so the cell return doesn't re-display.
    try:
        from matplotlib import get_backend
        if "inline" in get_backend().lower():
            from IPython.display import display
            display(fig)
            plt.close(fig)
            return None
    except Exception:
        pass
    return None


# ------------------------------------------------------------------
# Abundance test plot (after lots_of_cells with 2 groups)
# ------------------------------------------------------------------

def plot_abundance_test(
    table_results: pd.DataFrame,
    subtype_variable: str = "covariable",
    figsize=(8, 6),
    pdf_file: Optional[str] = None,
):
    """Beautiful bubble plot of FC ± Monte-Carlo SD shown as pink ribbon."""
    df = table_results.sort_values("groupFC").copy()
    df["classLabel"] = df.index
    cols = list(df.columns)
    on_right = cols[1].split("percent_in_")[1]
    on_left = cols[2].split("percent_in_")[1]
    guide = float(np.ceil(max(np.abs(df[["CI95low", "CI95high"]].fillna(0).to_numpy()).max(), 0))) + 0.5
    p_adj = df["p.adj"].to_numpy()
    p_adj[p_adj == 0] = 1e-5
    significance = np.sign(df["groupFC"].values) * -np.log10(p_adj)

    cmap = LinearSegmentedColormap.from_list(
        "fc_cmap",
        ["#122A53", "#43587D", "#8BBCD4", "#C1DEEF", "#EEF6FF", "#FDFFFF",
         "#F6F3FF", "#DDCFFF", "#D1AADB", "#76608E", "#463955"],
    )

    fig, ax = plt.subplots(figsize=figsize)
    # SD ribbon
    for i, (cls, row) in enumerate(df.iterrows()):
        ax.add_patch(mpatches.Rectangle(
            (-row["sd.montecarlo"], i - 0.4),
            2 * row["sd.montecarlo"], 0.8,
            color="pink", alpha=0.3, zorder=1,
        ))
    # CI bars
    for i, (cls, row) in enumerate(df.iterrows()):
        ax.hlines(i, row["CI95low"], row["CI95high"],
                  colors="#70508E", lw=0.6, zorder=2)
    # Points
    norm = plt.Normalize(-3, 3)
    sc = ax.scatter(
        df["groupFC"], range(len(df)),
        c=significance, cmap=cmap, norm=norm,
        s=140, edgecolors="black", linewidths=0.3, zorder=3,
    )
    ax.axvline(0, color="#86608E", lw=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    ax.set_xlim(-guide, guide)
    ax.set_xlabel(f"log2(proportion_FC) : ({on_right}/{on_left})")
    ax.set_title(
        f"Fold-Change difference in proportion\n"
        f"Monte-Carlo simulation on {subtype_variable}"
    )
    ax.text(-1, -0.6, on_left, color="grey", ha="center")
    ax.text(1, -0.6, on_right, color="grey", ha="center")
    cbar = plt.colorbar(sc, ax=ax, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar.set_label("signed -log10(p.adj)")
    fig.tight_layout()
    save_to_pdf(fig, pdf_file)
    return ax
