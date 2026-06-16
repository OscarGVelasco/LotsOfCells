"""End-to-end example mirroring the R README.

Run from the `python/` folder:
    python -m pip install -e .
    python example.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import lotsofcells as loc


def simulated_metadata() -> pd.DataFrame:
    """Reproduce the simulated dataset from the R README."""
    sample_blocks = [
        ("A", "time 0h", "wt",  [("CellTypeA", 700), ("CellTypeB", 300), ("CellTypeC", 500), ("CellTypeD", 1000)]),
        ("B", "time 0h", "mut", [("CellTypeA", 1700), ("CellTypeB", 350), ("CellTypeC", 550), ("CellTypeD", 800)]),
        ("C", "time 2h", "wt",  [("CellTypeA", 1200), ("CellTypeB", 200), ("CellTypeC", 420), ("CellTypeD", 800)]),
        ("D", "time 2h", "mut", [("CellTypeA", 500),  ("CellTypeB", 1000), ("CellTypeC", 10), ("CellTypeD", 1200)]),
        ("E", "time 4h", "wt",  [("CellTypeA", 550),  ("CellTypeB", 990),  ("CellTypeC", 10), ("CellTypeD", 1100)]),
        ("F", "time 4h", "mut", [("CellTypeA", 1350), ("CellTypeB", 590),  ("CellTypeC", 300), ("CellTypeD", 600)]),
    ]
    rows = []
    for sample, t, cond, ct_counts in sample_blocks:
        for ct, n in ct_counts:
            for _ in range(n):
                rows.append((sample, ct, t, cond))
    return pd.DataFrame(rows, columns=["sample", "cell_type", "times", "condition"])


def main():
    rng = np.random.default_rng(0)
    meta = simulated_metadata()
    print("Metadata head:")
    print(meta.head())
    print("Shape:", meta.shape)

    print("\n--- 2-condition test (mut vs wt) ---")
    res = loc.lots_of_cells(
        meta,
        main_variable="condition",
        subtype_variable="cell_type",
        sample_id="sample",
        label_order=["mut", "wt"],
        permutations=300,
        seed=0,
        plot=False,
    )
    print(res)

    print("\n--- >2-condition gamma rank (time 0h, 2h, 4h) ---")
    gamma = loc.lots_of_cells(
        meta,
        main_variable="times",
        subtype_variable="cell_type",
        sample_id="sample",
        label_order=["time 0h", "time 2h", "time 4h"],
        permutations=100,
        seed=0,
        plot=False,
    )
    print(gamma)

    print("\n--- Symmetric divergence score (mut vs wt) ---")
    ent = loc.entropy_score(
        meta,
        main_variable="condition",
        subtype_variable="cell_type",
        label_order=["mut", "wt"],
        permutations=200,
        seed=0,
        plot=False,
    )
    print(ent)

    print("\n--- AnnData round-trip ---")
    try:
        import anndata as ad
        adata = ad.AnnData(np.zeros((len(meta), 1), dtype=float), obs=meta.copy())
        adata.obs_names = adata.obs_names.astype(str)
        # Simulate a numerical feature so density_chart works on .obs
        adata.obs["n_features_RNA"] = np.abs(
            rng.normal(loc=2500, scale=600, size=adata.n_obs)
        )
        res2 = loc.lots_of_cells(
            adata,
            main_variable="condition",
            subtype_variable="cell_type",
            sample_id="sample",
            label_order=["mut", "wt"],
            permutations=100,
            seed=0,
            plot=False,
        )
        assert res2.shape[0] == res.shape[0]
        print("AnnData path OK. Same #covariables in result.")
    except ImportError:
        print("(anndata not installed — skipping AnnData round-trip)")


if __name__ == "__main__":
    main()
