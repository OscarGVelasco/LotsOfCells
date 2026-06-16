# lotsofcells (Python port)

Python translation of the R package
[`lotsOfCells`](https://github.com/OscarGVelasco/lotsOfCells), built on the
**scanpy / AnnData** stack and compatible with **spatial transcriptomics**
(`spatialdata`, `mudata`) — any object that exposes its metadata via `.obs`.

`lotsofcells` provides proportion-test statistics and visualisations for
single-cell metadata:

- **2-condition** log-fold-change of arcsin-sqrt proportions, with a
  Monte-Carlo null distribution.
- **>2-condition** Goodman & Kruskal gamma rank-correlation across an
  ordered set of groups.
- **Symmetric divergence (entropy) score** to detect global proportion
  dysregulation.
- Bar, waffle, polar, density-ridge, dynamics, and abundance-test plots.

## Install

From the `python/` folder:

```bash
pip install -e .
# optional extras
pip install -e .[scanpy,spatial]
```

## Quick start

```python
import numpy as np
import pandas as pd
import anndata as ad
import lotsofcells as loc

# --- simulate metadata identical to the R example ---
def sim():
    counts = [(700, 300, 500, 1000),
              (1700, 350, 550, 800),
              (1200, 200, 420, 800),
              (500, 1000, 10, 1200),
              (550, 990, 10, 1100),
              (1350, 590, 300, 600)]
    samples, types = [], []
    for i, c in enumerate(counts):
        s = chr(ord("A") + i)
        for ct, n in zip(["CellTypeA", "CellTypeB", "CellTypeC", "CellTypeD"], c):
            samples += [s] * n
            types  += [ct] * n
    df = pd.DataFrame({"sample": samples, "cell_type": types})
    df["times"] = np.repeat(["time 0h", "time 0h", "time 2h",
                             "time 2h", "time 4h", "time 4h"],
                            [sum(c) for c in counts])
    df["condition"] = np.where(df["sample"].isin(["B", "D", "F"]), "mut", "wt")
    return df

meta = sim()

# Use as a plain DataFrame...
loc.bar_chart(meta, main_variable="condition", subtype_variable="cell_type",
              sample_id="sample")

# ... or as an AnnData (uses .obs)
adata = ad.AnnData(np.zeros((len(meta), 1)), obs=meta)
loc.waffle_chart(adata, main_variable="condition",
                 subtype_variable="cell_type", sample_id="sample")

# Differential proportion test (2 conditions)
results = loc.lots_of_cells(
    adata,
    main_variable="condition",
    subtype_variable="cell_type",
    sample_id="sample",
    label_order=["mut", "wt"],
    permutations=1000,
    seed=0,
)
print(results)

# Symmetric divergence score
loc.entropy_score(
    adata,
    main_variable="condition",
    subtype_variable="cell_type",
    label_order=["mut", "wt"],
    permutations=1000,
    seed=0,
)

# >2 conditions: gamma rank correlation
gamma = loc.lots_of_cells(
    adata,
    main_variable="times",
    subtype_variable="cell_type",
    sample_id="sample",
    label_order=["time 0h", "time 2h", "time 4h"],
    permutations=200,
    seed=0,
)
loc.dynamics_chart(gamma)
```

## Spatial transcriptomics

For `SpatialData`, pass the object directly. If it has more than one table,
use the `table=` keyword:

```python
import spatialdata as sd
sdata = sd.read_zarr("my_spatial_dataset.zarr")
loc.bar_chart(sdata, main_variable="region", subtype_variable="annotation",
              table="table")
```

For `MuData`, pass the modality name as `table`:

```python
import mudata as md
mdata = md.read_h5mu("my_multiome.h5mu")
loc.lots_of_cells(mdata, main_variable="sample", subtype_variable="cell_type",
                  label_order=["healthy", "disease"],
                  table="rna")
```

## Mapping from R → Python

| R                  | Python                                   |
| ------------------ | ---------------------------------------- |
| `lotsOfCells()`    | `lots_of_cells()`                        |
| `entropyScore()`   | `entropy_score()`                        |
| `bar_chart()`      | `bar_chart()`                            |
| `waffle_chart()`   | `waffle_chart()`                         |
| `polar_chart()`    | `polar_chart()`                          |
| `density_chart()`  | `density_chart()`                        |
| `dynamics_chart()` | `dynamics_chart()`                       |
| `getMetadata()`    | `get_metadata()`                         |
| `getPalette()`     | `get_palette()`                          |
| `plotAbundanceTest()` | `plot_abundance_test()`               |

## Notes on parity with the R version

- Pseudocounts, arcsin-sqrt transform, log2 fold-change formula, and the BH
  FDR adjustment all mirror the R sources exactly (`R/lotsOfCells.R`,
  `R/cellToMontecarlo.R`, `R/entropyScore.R`).
- Random seeding is deterministic via `numpy.random.default_rng(seed)`.
- Plots are matplotlib-based and intentionally minimal — they aim for the
  same information density as the ggplot originals rather than pixel-perfect
  replication.
