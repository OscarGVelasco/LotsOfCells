"""Internal helpers: metadata extraction and color palette."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

# Default categorical palette — scanpy `default_20` (Vega-20 with
# yellow/green hues swapped for higher contrast). The de facto standard in
# the single-cell community, so colours feel native to scanpy figures.
_DEFAULT_PALETTE = [
    "#1F77B4", "#FF7F0E", "#279E68", "#D62728", "#AA40FC",
    "#8C564B", "#E377C2", "#B5BD61", "#17BECF", "#AEC7E8",
    "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5", "#C49C94",
    "#F7B6D2", "#DBDB8D", "#9EDAE5", "#AD494A", "#8C6D31",
]

def _is_anndata(obj) -> bool:
    """Return True if obj quacks like an AnnData (has .obs)."""
    try:
        import anndata  # noqa: F401
    except Exception:
        anndata = None  # type: ignore
    if anndata is not None and isinstance(obj, anndata.AnnData):
        return True
    return hasattr(obj, "obs") and isinstance(getattr(obj, "obs"), pd.DataFrame)


def _is_spatialdata(obj) -> bool:
    try:
        import spatialdata  # type: ignore
        return isinstance(obj, spatialdata.SpatialData)
    except Exception:
        return False


def _is_mudata(obj) -> bool:
    try:
        import mudata  # type: ignore
        return isinstance(obj, mudata.MuData)
    except Exception:
        return False


def get_metadata(sc_object, table: Optional[str] = None) -> pd.DataFrame:
    """Return a metadata DataFrame from a scanpy/spatial/dataframe object.

    Parameters
    ----------
    sc_object
        One of: ``pandas.DataFrame``, ``anndata.AnnData``, ``mudata.MuData``,
        or ``spatialdata.SpatialData``. AnnData/Mu/Spatial objects expose their
        cell-level metadata via ``.obs``; this is the analogue of
        ``Seurat[[]]`` / ``SingleCellExperiment::colData``.
    table
        Only used when ``sc_object`` is a ``SpatialData`` (the name of the
        table whose ``.obs`` should be returned) or ``MuData`` (the modality
        name). If ``None`` and the object has a single table/modality, that
        one is used.
    """
    if sc_object is None:
        raise ValueError("At least an AnnData/SpatialData/DataFrame is required.")

    if isinstance(sc_object, pd.DataFrame):
        return sc_object.copy()

    if _is_spatialdata(sc_object):
        tables = dict(sc_object.tables)
        if not tables:
            raise ValueError("SpatialData object has no tables.")
        if table is None:
            if len(tables) > 1:
                raise ValueError(
                    f"SpatialData has multiple tables {list(tables)}; "
                    "specify `table=...`."
                )
            table = next(iter(tables))
        return tables[table].obs.copy()

    if _is_mudata(sc_object):
        if table is None:
            return sc_object.obs.copy()
        return sc_object[table].obs.copy()

    if _is_anndata(sc_object):
        return sc_object.obs.copy()

    raise TypeError(
        "Unsupported object type for metadata extraction. "
        "Pass a pandas.DataFrame or AnnData/MuData/SpatialData."
    )


def get_numerical_variable(
    sc_object, numerical_variable: str, metadata: pd.DataFrame,
    layer: Optional[str] = None,
) -> np.ndarray:
    """Resolve a numerical variable from .obs OR feature counts (gene name).

    Mirrors the R behaviour of `density_chart`: if the column is in
    metadata, return it; otherwise look for a feature in the AnnData and
    return its expression vector aligned to ``metadata.index``.

    Parameters
    ----------
    layer
        For AnnData objects, pull the gene column from
        ``adata.layers[layer]`` instead of ``adata.X``. Useful when ``.X``
        is normalised but you need raw counts (e.g. for the gene-source
        contribution test).
    """
    if numerical_variable in metadata.columns:
        return metadata[numerical_variable].to_numpy()

    if _is_anndata(sc_object):
        adata = sc_object
        if numerical_variable in adata.var_names:
            idx = adata.var_names.get_loc(numerical_variable)
            if layer is not None:
                if layer not in adata.layers:
                    raise ValueError(
                        f"Layer '{layer}' not found in adata.layers "
                        f"(available: {list(adata.layers.keys())})"
                    )
                X = adata.layers[layer]
            else:
                X = adata.X
            col = X[:, idx]
            if hasattr(col, "toarray"):
                col = col.toarray().ravel()
            else:
                col = np.asarray(col).ravel()
            # Align to metadata row order
            obs_idx = metadata.index
            full = pd.Series(col, index=adata.obs_names)
            return full.loc[obs_idx].to_numpy()

    raise ValueError(
        f"Variable '{numerical_variable}' not found in metadata columns "
        "or feature names."
    )


def get_palette(
    use_palette: Optional[Sequence[str]] = None, n_colors: int = 20
) -> list:
    """Return a list of `n_colors` colors.

    If `use_palette` is None, the default lotsOfCells palette is used.
    If more colors than provided are requested, a linear interpolation in RGB
    space (analogue of `colorRampPalette`) is performed.
    """
    base = list(use_palette) if use_palette is not None else list(_DEFAULT_PALETTE)
    if n_colors <= len(base):
        return base[:n_colors]
    return _ramp_palette(base, n_colors)


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _rgb_to_hex(rgb: Union[np.ndarray, Sequence[float]]) -> str:
    rgb = np.clip(np.asarray(rgb), 0, 1)
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255)) for c in rgb))


def _ramp_palette(colors: Sequence[str], n: int) -> list:
    """Equivalent of grDevices::colorRampPalette in linear RGB."""
    rgbs = np.stack([_hex_to_rgb(c) for c in colors])  # (k, 3)
    if n == 1:
        return [_rgb_to_hex(rgbs[0])]
    src = np.linspace(0, 1, len(colors))
    tgt = np.linspace(0, 1, n)
    interp = np.stack(
        [np.interp(tgt, src, rgbs[:, c]) for c in range(3)], axis=1
    )
    return [_rgb_to_hex(rgb) for rgb in interp]


def lighten(color: str, amount: float = 0.2) -> str:
    """Lighten an HSV-based color by `amount` (0..1). Analogue of colorspace::lighten."""
    import colorsys

    r, g, b = _hex_to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l + amount * (1 - l)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex((r, g, b))


def darken(color: str, amount: float = 0.2) -> str:
    """Darken color by `amount` (0..1). Analogue of colorspace::darken."""
    import colorsys

    r, g, b = _hex_to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l * (1 - amount)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex((r, g, b))


def desaturate(color: str, amount: float = 0.16) -> str:
    """Reduce saturation. Analogue of colorspace::desaturate."""
    import colorsys

    r, g, b = _hex_to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, s * (1 - amount))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex((r, g, b))


# -------------------------------------------------------------------
# Qualitative-change thresholds for composition-divergence scores.
#
# Anchored at 75% penetrance × 50% cell types dysregulated = 0.151 in
# the bounded_fc_mean calibration study (see
# examples/simulation_calibration_realistic.py). Every other band is
# picked to match a biologically defensible scenario at that qualitative
# level. Colour ramp goes grey → deep purple as intensity grows.
#
# Note: bands are calibrated for the bounded_fc_mean scale. When the
# underlying `entropy_score` metric is KL-symmetric divergence the scores
# span a different range — the bands are still visually informative for
# comparison against a reference, but the exact category labels will not
# apply verbatim.
# -------------------------------------------------------------------
CATEGORY_THRESHOLDS = {
    "None":        (0.00, 0.09),
    "Minor":        (0.09, 0.12),
    "Mild":    (0.12, 0.15),
    "Moderate": (0.15, 0.20),
    "Substantial":   (0.20, 0.28),
    "Extensive":      (0.28, float("inf")),
}
"""
CATEGORY_COLORS = {
    "None":        "#F2F2F2",   # very light grey
    "Minor":        "#DCD0E4",   # pale lavender
    "Mild":    "#B698CE",   # light purple
    "Moderate": "#86608E",   # mid purple (matches existing package palette)
    "Substantial":   "#613269",   # deep purple (already in the default palette)
    "Extensive":      "#2A1240",   # near-black purple
}
"""
CATEGORY_COLORS = {
    "None":        "#CCCCCC",
    "Minor":        "#5CCD9ACB",
    "Moderate":    "#F9BE8D",
    "Substantial": "#B25356",
    "Extensive":   "#613269",
}

def draw_threshold_bands(ax, alpha: float = 0.30, zorder: int = 0,
                          min_top: float = 0.20, add_legend: bool = True,
                          legend_kwargs: Optional[dict] = None) -> None:
    """Shade the six qualitative-change threshold bands behind an Axes.

    Parameters
    ----------
    ax
        Matplotlib Axes to shade. The bands run horizontally along the
        y-axis (score axis).
    alpha
        Band transparency. Default 0.30 keeps overlaid scatter readable.
    zorder
        Drawing order. Default 0 puts the bands behind data.
    min_top
        Extend the y-axis at least this high (default 0.20 = start of
        Extensive) so the ★ Substantial anchor is always visible.
    add_legend
        Attach a compact legend (patches) explaining the six bands.
    legend_kwargs
        Overrides for the legend call (loc, fontsize, bbox_to_anchor…).
    """
    import matplotlib.patches as mpatches

    ylim = ax.get_ylim()
    new_top = max(ylim[1], min_top)
    for name, (lo, hi) in CATEGORY_THRESHOLDS.items():
        if lo >= new_top:
            break
        hi_draw = min(hi, new_top)
        ax.axhspan(lo, hi_draw, color=CATEGORY_COLORS[name],
                   alpha=alpha, zorder=zorder, linewidth=0)
    ax.set_ylim(ylim[0], new_top)

    if add_legend:
        handles = [
            mpatches.Patch(color=CATEGORY_COLORS[name], alpha=alpha, label=name)
            for name in CATEGORY_THRESHOLDS
        ]
        defaults = dict(
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=7, frameon=False, title="Change level",
            title_fontsize=7, handlelength=1.2, handletextpad=0.4,
            borderaxespad=0.0,
        )
        if legend_kwargs:
            defaults.update(legend_kwargs)
        ax.legend(handles=handles, **defaults)


def save_to_pdf(fig, pdf_file: Optional[str]) -> None:
    """Save a matplotlib Figure to PDF if `pdf_file` is provided.

    Used by every plotting function when the user passes ``pdf_file=...``.
    Uses ``bbox_inches="tight"`` so that legends placed outside the axes
    are included and not clipped.
    """
    if pdf_file is None:
        return
    if fig is None:
        import matplotlib.pyplot as plt

        fig = plt.gcf()
    fig.savefig(pdf_file, format="pdf", bbox_inches="tight")
