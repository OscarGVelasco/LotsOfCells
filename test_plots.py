"""Smoke test: render every plot to disk to ensure they work without error."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lotsofcells as loc

OUT = "_smoke_plots"
os.makedirs(OUT, exist_ok=True)


def make_meta():
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
            rows.extend([(sample, ct, t, cond)] * n)
    df = pd.DataFrame(rows, columns=["sample", "cell_type", "times", "condition"])
    rng = np.random.default_rng(0)
    df["n_features_RNA"] = np.abs(
        rng.normal(loc=2500 + (df["condition"] == "mut") * 700, scale=400)
    )
    return df


def save(name):
    plt.savefig(os.path.join(OUT, f"{name}.png"), dpi=80, bbox_inches="tight")
    plt.close("all")


def main():
    meta = make_meta()
    print("Rendering bar_chart...")
    loc.bar_chart(meta, "condition", "cell_type"); save("bar_basic")
    loc.bar_chart(meta, "condition", "cell_type", sample_id="sample"); save("bar_sample")
    loc.bar_chart(meta, "condition", "cell_type", sample_id="sample",
                  subtype_only="CellTypeD"); save("bar_subtypeonly")
    loc.bar_chart(meta, "condition", "times"); save("bar_times")

    print("Rendering waffle_chart...")
    loc.waffle_chart(meta, "condition", "cell_type"); save("waffle_basic")
    loc.waffle_chart(meta, "condition", "cell_type", sample_id="sample"); save("waffle_sample")
    loc.waffle_chart(meta, "condition", "cell_type", subtype_only="CellTypeD"); save("waffle_subtypeonly")

    print("Rendering polar_chart...")
    loc.polar_chart(meta, "condition", "cell_type", sample_id="sample"); save("polar")

    print("Rendering density_chart...")
    loc.density_chart(meta, "condition", "cell_type",
                      numerical_variable="n_features_RNA"); save("density_basic")
    loc.density_chart(meta, "condition", "cell_type",
                      numerical_variable="n_features_RNA",
                      sample_id="sample"); save("density_sample")

    print("Rendering plot_abundance_test via lots_of_cells (2 conds)...")
    res = loc.lots_of_cells(meta, "condition", "cell_type",
                            sample_id="sample",
                            label_order=["mut", "wt"],
                            permutations=200, seed=0)
    save("abundance_test")
    print(res)

    print("Rendering dynamics_chart (>2 conds)...")
    gamma = loc.lots_of_cells(meta, "times", "cell_type",
                              sample_id="sample",
                              label_order=["time 0h", "time 2h", "time 4h"],
                              permutations=80, seed=0, plot=False)
    loc.dynamics_chart(gamma); save("dynamics")
    print(gamma)

    print("Rendering entropy_score plot...")
    loc.entropy_score(meta, "condition", "cell_type",
                      label_order=["mut", "wt"],
                      permutations=200, seed=0)
    save("entropy")

    files = sorted(os.listdir(OUT))
    print(f"\nWrote {len(files)} plot(s) to {OUT}/")
    for f in files:
        print(" ", f)


if __name__ == "__main__":
    main()
