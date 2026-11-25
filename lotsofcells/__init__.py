"""lotsofcells: proportion-test statistics and visualization on single-cell metadata.

Python port of the R package `lotsOfCells`, designed for the scanpy / AnnData
framework. Compatible with single-cell (`AnnData`) and spatial transcriptomics
(`SpatialData` / `MuData`) objects, since metadata is read from `.obs`.

References
----------
Óscar González-Velasco; lotsOfCells: data visualization and statistics of
single cell metadata. bioRxiv 2024.05.23.595582;
https://doi.org/10.1101/2024.05.23.595582
"""

from ._utils import get_metadata, get_palette
from .lotsofcells import lots_of_cells
from .entropy import entropy_score
from .gene_source import (
    gene_source_score,
    gene_source_bar_chart,
    gene_source_boxplot,
)
from .plots import (
    bar_chart,
    waffle_chart,
    polar_chart,
    density_chart,
    dynamics_chart,
    plot_abundance_test,
)

__all__ = [
    "get_metadata",
    "get_palette",
    "lots_of_cells",
    "entropy_score",
    "gene_source_score",
    "gene_source_bar_chart",
    "gene_source_boxplot",
    "bar_chart",
    "waffle_chart",
    "polar_chart",
    "density_chart",
    "dynamics_chart",
    "plot_abundance_test",
]

def _get_version() -> str:
    """Version of the installed package (set by setuptools-scm from the git tag at
    build time). From a source checkout that is not installed: the version
    setuptools-scm derives from the git history, if available."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("lotsofcells")
    except PackageNotFoundError:
        pass
    try:
        from pathlib import Path
        from setuptools_scm import get_version
        return get_version(root=str(Path(__file__).resolve().parents[1]))
    except Exception:  # noqa: BLE001 — no setuptools-scm, or not a git checkout
        return "0+unknown"


__version__ = _get_version()
