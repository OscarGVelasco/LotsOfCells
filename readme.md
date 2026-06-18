# lotsOfCells

Proportion-test statistics and visualisation on single-cell metadata. A
simple Python package for single-cell metadata exploration, built on the
**scanpy / AnnData** stack and natively compatible with **spatial
transcriptomics** (`spatialdata`, `mudata`) — any object that exposes its
metadata via `.obs`.

### Installation

Install the latest release from PyPI:

```bash
pip install lotsofcells
```

Or install the development version directly from GitHub:

```bash
pip install git+https://github.com/OscarGVelasco/LotsOfCells.git
```

Optional extras for spatial transcriptomics and multi-modal data:

```bash
pip install "lotsofcells[scanpy,spatial]"
```

### How to cite

A pre-print describing the statistical tests and showing example use on a
public lung adenocarcinoma dataset is available at
[https://doi.org/10.1101/2024.05.23.595582](https://doi.org/10.1101/2024.05.23.595582):

```
Óscar González-Velasco; lotsOfCells: data visualization and statistics of
single cell metadata. bioRxiv 2024.05.23.595582;
doi: https://doi.org/10.1101/2024.05.23.595582
```

### Updates

* **v0.3.0**
  - Native AnnData / SpatialData / MuData support — pass the object directly,
    metadata is read from `.obs`.
  - Added an option to save any plot to PDF via `pdf_file="file.pdf"`.
  - Customisable colour palettes for bar, waffle, polar, and density plots.
  - Density ridge plots support both `.obs` columns and gene/feature
    expression (looked up in `adata.X`).
  - Differential proportion barplots are sorted by fold-change.

# Introduction

Single-cell sequencing unveils a treasure trove into the biological and
molecular characteristics of samples. `lotsOfCells` is a Python package
designed to explore the intricate landscape of phenotype data within
single-cell and spatial studies. An archetypal use is testing differences
in cell-type proportions between conditions, but it generalises to any
covariate: for example, testing whether a specific cell-type or class
proportion is dependent on sequencing date as a quality check.

### Overview of tests and visualisations

`lotsOfCells` exposes nine top-level functions:

| Statistics            | Visualisation        |
| --------------------- | -------------------- |
| `lots_of_cells`       | `bar_chart`          |
| `entropy_score`       | `waffle_chart`       |
|                       | `polar_chart`        |
|                       | `density_chart`      |
|                       | `dynamics_chart`     |
|                       | `plot_abundance_test`|

# Manual

`lotsOfCells` accepts an `AnnData`, `SpatialData`, `MuData`, or a plain
`pandas.DataFrame` containing the metadata. All visualisation and test
functions share the same first arguments:

- `sc_object`: an `AnnData` (or `SpatialData` / `MuData`), or a
  `pandas.DataFrame` with the metadata.
- `main_variable`: name of the column with the main class variable
  (e.g. condition, treatment, tissue).
- `subtype_variable`: name of the column with the sub-class variable
  (e.g. cell type, sequencing date, cluster).
- `sample_id` *(optional)*: name of the column with the sample / patient
  identifier. When provided, tests simulate per-sample proportion
  variability and plots show each individual.
- `pdf_file` *(optional)*: path to a PDF file. If given, the plot is also
  saved to disk via `fig.savefig(..., bbox_inches="tight")`.

For spatial transcriptomics or multi-modal objects with multiple tables,
pass `table="<table_or_modality_name>"` so the right `.obs` is picked.

### Example dataset

We will construct a simulated dataset of single-cell metadata consisting of
six samples with two conditions (mutant and wild type), four cell types
(A, B, C, D) and three time points simulating treatment (0h, 2h, 4h):

```python
import numpy as np
import pandas as pd
import anndata as ad
import lotsofcells as loc

rng = np.random.default_rng(0)

sample_blocks = [
    ("A", "time 0h", "wt",  [("CellTypeA", 700),  ("CellTypeB", 300),
                              ("CellTypeC", 500),  ("CellTypeD", 1000)]),
    ("B", "time 0h", "mut", [("CellTypeA", 1700), ("CellTypeB", 350),
                              ("CellTypeC", 550),  ("CellTypeD", 800)]),
    ("C", "time 2h", "wt",  [("CellTypeA", 1200), ("CellTypeB", 200),
                              ("CellTypeC", 420),  ("CellTypeD", 800)]),
    ("D", "time 2h", "mut", [("CellTypeA", 500),  ("CellTypeB", 1000),
                              ("CellTypeC", 10),   ("CellTypeD", 1200)]),
    ("E", "time 4h", "wt",  [("CellTypeA", 550),  ("CellTypeB", 990),
                              ("CellTypeC", 10),   ("CellTypeD", 1100)]),
    ("F", "time 4h", "mut", [("CellTypeA", 1350), ("CellTypeB", 590),
                              ("CellTypeC", 300),  ("CellTypeD", 600)]),
]
rows = []
for sample, t, cond, parts in sample_blocks:
    for ct, n in parts:
        rows.extend([(sample, ct, t, cond)] * n)
meta = pd.DataFrame(rows, columns=["sample", "cell_type", "times", "condition"])
meta.head()
```

You can use `meta` directly or wrap it in an `AnnData`:

```python
adata = ad.AnnData(np.zeros((len(meta), 1), dtype=np.float32), obs=meta.copy())
adata.obs_names = adata.obs_names.astype(str)
```

Every plot and test below works identically on `meta` and `adata`.

The functions below accept an optional `subtype_only` argument to focus
the visualisation on a single class from `subtype_variable` (useful when
you want, for instance, to display the proportion of one specific cell
type across conditions).

### Barplots

Barplots are arranged so that the class with the largest average
proportion sits at the bottom of the stack, which makes the smaller
groups easier to compare at the top.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. All cells together per group
loc.bar_chart(meta, main_variable="condition",
              subtype_variable="cell_type", ax=axes[0, 0])

# B. One stacked bar per sample, grouped by condition
loc.bar_chart(meta, main_variable="condition", subtype_variable="cell_type",
              sample_id="sample", ax=axes[0, 1])

# C. One class only, per sample
loc.bar_chart(meta, main_variable="condition", subtype_variable="cell_type",
              sample_id="sample", subtype_only="CellTypeD", ax=axes[1, 0])

# D. Distribution of time-points by condition
loc.bar_chart(meta, main_variable="condition",
              subtype_variable="times", ax=axes[1, 1])

plt.tight_layout()
```

The `subtype_variable` does not have to be a cell type. Switching it to a
covariate such as time points lets you investigate how strongly a
covariate contributes to the main condition, which is a useful quality
check for time points, sequencing dates, or tissues of origin.

### Contribution of a class to the global population

If you want to visualise how each sample contributes to the overall
proportion of a class (for example, how much each sample contributes to a
specific cell type within a condition), pass `contribution=True`. Each
sample is then rendered as a different shade of its cell-type colour:

```python
loc.bar_chart(meta, main_variable="condition", subtype_variable="cell_type",
              sample_id="sample", contribution=True)

loc.bar_chart(meta, main_variable="times", subtype_variable="cell_type",
              sample_id="sample", contribution=True)
```

### Waffle plots

Waffle plots are easier to read than barplots when proportions are small.
Each tile represents 1% of the population:

```python
# A. All cells together per group
loc.waffle_chart(meta, main_variable="condition",
                 subtype_variable="cell_type")

# B. One waffle per sample
loc.waffle_chart(meta, main_variable="condition",
                 subtype_variable="cell_type", sample_id="sample")

# C. A specific class only, per sample
loc.waffle_chart(meta, main_variable="condition",
                 subtype_variable="cell_type",
                 sample_id="sample", subtype_only="CellTypeD")

# D. A specific class only, samples pooled by condition
loc.waffle_chart(meta, main_variable="condition",
                 subtype_variable="cell_type",
                 subtype_only="CellTypeD")
```

### Density ridge plots

To visualise a numerical variable over groups and sub-categories, use
`density_chart`. Pass the name of the column in `.obs` (or in a
`DataFrame`) through `numerical_variable`. The function draws one ridge
per `subtype_variable × main_variable` (and per sample when `sample_id` is
provided):

```python
# Simulate the number of RNA features detected per cell
mut_mask = meta["condition"] == "mut"
meta["n_features_RNA"] = np.where(
    mut_mask,
    np.abs(rng.normal(loc=3200, scale=800, size=len(meta))),
    np.abs(rng.normal(loc=2000, scale=400, size=len(meta))),
)

# A. All cells per group
loc.density_chart(meta, main_variable="condition",
                  subtype_variable="cell_type",
                  numerical_variable="n_features_RNA")

# B. Per individual sample
loc.density_chart(meta, main_variable="condition",
                  subtype_variable="cell_type",
                  numerical_variable="n_features_RNA",
                  sample_id="sample")
```

When `sc_object` is an `AnnData`, `numerical_variable` can also be a
gene/feature name. The values are looked up in `adata.X` and aligned to
the metadata index, so you can plot a gene's expression across cell types
and conditions directly:

```python
# Pretend `adata` has a gene called "INSR" in adata.var_names
loc.density_chart(adata,
                  main_variable="condition",
                  subtype_variable="cell_type",
                  numerical_variable="INSR")
```

### Personalising the colours

Colours can be customised for bar, waffle, polar, and density plots via
the `colors=` argument. Pass any sequence of hex strings or named
matplotlib colours. If you provide fewer colours than there are classes,
the palette is interpolated to fit; if you pass none, the default
`lotsOfCells` palette is used:

```python
loc.bar_chart(meta, main_variable="condition", subtype_variable="cell_type",
              sample_id="sample",
              colors=["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072",
                      "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5"])

loc.waffle_chart(meta, main_variable="condition", subtype_variable="cell_type",
                 sample_id="sample",
                 colors=["#B3E2CD", "#FDCDAC", "#CBD5E8", "#F4CAE4",
                         "#E6F5C9", "#FFF2AE"])
```

### Polar plots

Polar plots are useful when you want to compare the raw number of cells
per class, for example to spot samples whose cell count is much larger or
smaller than the rest and that could disproportionately influence
downstream analyses:

```python
loc.polar_chart(meta, main_variable="condition",
                subtype_variable="cell_type", sample_id="sample")

loc.polar_chart(meta, main_variable="cell_type",
                subtype_variable="sample")
```

### Reproducibility

Every statistical function takes a `seed=` argument. Internally the
package uses `numpy.random.default_rng(seed)` so the same seed gives the
same numbers across runs and operating systems:

```python
results = loc.lots_of_cells(meta, main_variable="condition",
                            subtype_variable="cell_type",
                            sample_id="sample",
                            label_order=["mut", "wt"],
                            permutations=1000, seed=42)
```

### Differential proportion test — two conditions

The headline feature of `lotsOfCells` is the comparison of proportions
combined with a Monte-Carlo simulation that gauges how extreme the
observed values are relative to a random null. To compare the proportions
of each class between two conditions, define the two labels and the order
of the contrast via `label_order` (the fold-change is computed as
`label_order[0] / label_order[1]`). Optionally, providing `sample_id`
accounts for per-sample heterogeneity in the simulation.

> ***NOTE*** *for significance testing we recommend an α value of 0.001.*

A bubble plot of the differences in proportions is rendered alongside the
result table. The pink ribbon around each point shows the standard
deviation of the Monte-Carlo simulations and the horizontal bar shows the
95% confidence interval of the observed fold change.

```python
results = loc.lots_of_cells(
    sc_object=meta,
    main_variable="condition",
    subtype_variable="cell_type",
    sample_id="sample",
    label_order=["mut", "wt"],
    permutations=1000,
    seed=0,
)
print(results)
```

The returned `pandas.DataFrame` has one row per class found in
`subtype_variable`:

| column                   | meaning                                                       |
| ------------------------ | ------------------------------------------------------------- |
| `groupFC`                | log2 fold-change of arcsin-sqrt proportions: log2(asrt(p1)/asrt(p2)) |
| `percent_in_<label1>`    | proportion of this class in `label_order[0]`                  |
| `percent_in_<label2>`    | proportion of this class in `label_order[1]`                  |
| `p.adj`                  | Benjamini–Hochberg FDR-adjusted p-value                       |
| `sd.montecarlo`          | standard deviation of the Monte-Carlo fold-change distribution|
| `CI95low`, `CI95high`    | 95% confidence interval for the observed `groupFC`            |

If you only want the numerical results without the plot, pass
`plot=False`. You can also redraw the plot at any time:

```python
loc.plot_abundance_test(results, subtype_variable="cell_type",
                        pdf_file="abundance_test.pdf")
```

### Symmetric Divergence Score — global dysregulation in class proportions

When you want to assess whether *the majority* of class proportions
change simultaneously between two conditions, use `entropy_score`. The
test computes a symmetric divergence score based on the Kullback–Leibler
(KL) divergence, then simulates a random distribution to estimate how
extreme the observed score is. Higher divergence scores suggest stronger
simultaneous changes across many classes.

```python
ent = loc.entropy_score(
    sc_object=meta,
    main_variable="condition",
    subtype_variable="cell_type",
    label_order=["mut", "wt"],
    permutations=10000,
    seed=0,
)
print(ent)
```

The function returns a `pandas.Series` containing per-class relative
entropies plus the summary fields:

- `entropy_score` — observed symmetric divergence
- `p.val` — proportion of null scores that meet or exceed the observed
- `mean.random.entropy`, `sd.random.entropy` — summary of the null

### Single-class permutation test (one condition, sample-level)

`entropy_score` also supports a one-class mode that quantifies how
heterogeneous the samples are *within a single condition*. Pass
`label_order=["<single_condition>"]` and a `sample_id` column. The
function randomly partitions the samples into two halves, draws cells
from each sample's own composition, and measures the divergence between
the two halves; the distribution of those divergences tells you how much
real per-sample variability exists.

```python
result = loc.entropy_score(
    sc_object=meta,
    main_variable="condition",
    subtype_variable="cell_type",
    label_order=["mut"],
    sample_id="sample",
    permutations=1000,
    seed=0,
)
print(result)   # dict with CV, variation level, IQR, mean/sd/median of the null
```

The reported coefficient of variation is classified as `Low` (≤35%),
`Medium` (≤50%), or `High` (>50%).

### Differential proportion test — more than two conditions

The proportion test in `lotsOfCells` can also be run with more than two
ordered groups, in which case it computes a Goodman & Kruskal gamma rank
correlation across the labels in the order specified via `label_order`.
Positive correlation means the proportion of that class trends up across
the ordered conditions; negative means it trends down.

```python
gamma = loc.lots_of_cells(
    sc_object=meta,
    main_variable="times",
    subtype_variable="cell_type",
    sample_id="sample",
    label_order=["time 0h", "time 2h", "time 4h"],
    permutations=1000,
    seed=0,
)
print(gamma)

# Visualise the dynamics across the ordered groups
loc.dynamics_chart(gamma)
```

`dynamics_chart` shows one line per class across the ordered groups in
the top panel, and the gamma rank correlation coefficient per class in
the bottom panel. The shaded amber bands indicate increasing levels of
correlation strength.

# Working with AnnData and spatial objects

Every function accepts an `AnnData` directly:

```python
import scanpy as sc

adata = sc.read_h5ad("my_dataset.h5ad")
loc.bar_chart(adata, main_variable="condition",
              subtype_variable="cell_type", sample_id="patient")
```

For `SpatialData`, pass the object as `sc_object`. If the object has more
than one table, use `table=` to pick the one whose `.obs` should be read:

```python
import spatialdata as sd

sdata = sd.read_zarr("my_visium_dataset.zarr")
loc.bar_chart(sdata, main_variable="region", subtype_variable="annotation",
              table="table")

loc.entropy_score(sdata, main_variable="region",
                  subtype_variable="annotation",
                  label_order=["tumor", "stroma"],
                  permutations=1000, seed=0,
                  table="table")
```

For `MuData`, pass the modality name as `table`:

```python
import mudata as md

mdata = md.read_h5mu("my_multiome.h5mu")
loc.lots_of_cells(mdata, main_variable="condition",
                  subtype_variable="cell_type",
                  label_order=["healthy", "disease"],
                  permutations=1000, seed=0,
                  table="rna")
```

# Saving plots to PDF

Every plotting function accepts a `pdf_file=` argument. When given, the
figure is saved with `bbox_inches="tight"` so that legends positioned
outside the axes are preserved:

```python
loc.bar_chart(meta, main_variable="condition", subtype_variable="cell_type",
              sample_id="sample", pdf_file="bar.pdf")

loc.waffle_chart(meta, main_variable="condition", subtype_variable="cell_type",
                 sample_id="sample", pdf_file="waffle.pdf")

loc.polar_chart(meta, main_variable="condition", subtype_variable="cell_type",
                sample_id="sample", pdf_file="polar.pdf")

loc.density_chart(meta, main_variable="condition", subtype_variable="cell_type",
                  numerical_variable="n_features_RNA", pdf_file="density.pdf")

# Plots auto-rendered by tests can also be saved
loc.lots_of_cells(meta, main_variable="condition", subtype_variable="cell_type",
                  sample_id="sample", label_order=["mut", "wt"],
                  permutations=1000, seed=0, pdf_file="abundance.pdf")

loc.entropy_score(meta, main_variable="condition", subtype_variable="cell_type",
                  label_order=["mut", "wt"], permutations=10000, seed=0,
                  pdf_file="entropy.pdf")

loc.dynamics_chart(gamma, pdf_file="dynamics.pdf")
```

# API reference

| Function                | Purpose                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| `lots_of_cells`         | Differential proportion test (2 labels) or gamma rank correlation (>2 labels) |
| `entropy_score`         | Symmetric divergence score for global proportion dysregulation; also 1-class permutation test |
| `bar_chart`             | Stacked barplot of proportions per group                         |
| `waffle_chart`          | Waffle plot (each tile = 1%)                                     |
| `polar_chart`           | Circular barplot of cell counts                                  |
| `density_chart`         | Ridge-style density plot of a numerical variable                 |
| `dynamics_chart`        | Per-class trends across ordered groups with gamma correlation    |
| `plot_abundance_test`   | Bubble plot of FC ± Monte-Carlo SD                               |
| `get_metadata`          | Extract `.obs` from AnnData / SpatialData / MuData / DataFrame   |
| `get_palette`           | Return a list of colours, interpolating if necessary             |

# License

MIT.