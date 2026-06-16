"""Smoke test: verify pdf_file= save and the 1-class entropy null fix."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lotsofcells as loc  # noqa: E402

OUT = "_smoke_pdf"
os.makedirs(OUT, exist_ok=True)


def make_meta_with_real_heterogeneity():
    """Samples within a single condition that have meaningfully different
    cell-type compositions (so the 1-class null should pick that up)."""
    rng = np.random.default_rng(0)
    sample_compositions = {
        "S1": [("A", 1500), ("B", 200),  ("C", 200),  ("D", 100)],   # A-dominant
        "S2": [("A", 1300), ("B", 250),  ("C", 250),  ("D", 200)],
        "S3": [("A", 200),  ("B", 1500), ("C", 100),  ("D", 200)],   # B-dominant
        "S4": [("A", 250),  ("B", 1300), ("C", 200),  ("D", 250)],
        "S5": [("A", 300),  ("B", 200),  ("C", 1400), ("D", 100)],   # C-dominant
        "S6": [("A", 200),  ("B", 300),  ("C", 1300), ("D", 200)],
    }
    rows = []
    for s, comps in sample_compositions.items():
        for ct, n in comps:
            rows.extend([(s, ct)] * n)
    df = pd.DataFrame(rows, columns=["sample", "cell_type"])
    df["condition"] = "diseased"
    df["times"] = "T0"
    return df


def main():
    meta = make_meta_with_real_heterogeneity()
    print(f"Synthesised metadata: {len(meta)} cells across "
          f"{meta['sample'].nunique()} samples in 1 condition")

    print("\n=== 1-class entropy_score (real per-sample heterogeneity) ===")
    out = loc.entropy_score(
        meta, "condition", "cell_type",
        label_order=["diseased"],
        sample_id="sample",
        permutations=200,
        seed=0,
        plot=True,
        pdf_file=os.path.join(OUT, "oneclass_entropy.pdf"),
    )
    print(out)
    assert out["mean.random.entropy"] > 0.005, (
        f"1-class null should NOT be ~0 with real heterogeneity, got "
        f"{out['mean.random.entropy']:.6f}"
    )
    print("✓ 1-class null is non-zero with real per-sample heterogeneity")

    # Re-add a wt condition for the rest of the smoke tests
    meta2 = meta.copy()
    extra_rows = []
    rng = np.random.default_rng(1)
    # Make wt samples that ARE more uniform
    for s in ["W1", "W2", "W3"]:
        for ct, n in [("A", 600), ("B", 600), ("C", 600), ("D", 600)]:
            extra_rows.extend([(s, ct, "control", "T0")] * n)
    meta2 = pd.concat(
        [meta2.assign(condition=meta2["condition"]),
         pd.DataFrame(extra_rows, columns=["sample", "cell_type", "condition", "times"])],
        ignore_index=True,
    )

    print("\n=== bar_chart with pdf_file ===")
    loc.bar_chart(meta2, "condition", "cell_type", sample_id="sample",
                  pdf_file=os.path.join(OUT, "bar.pdf"))

    print("\n=== waffle_chart with pdf_file ===")
    loc.waffle_chart(meta2, "condition", "cell_type", sample_id="sample",
                     pdf_file=os.path.join(OUT, "waffle.pdf"))

    print("\n=== polar_chart with pdf_file ===")
    loc.polar_chart(meta2, "condition", "cell_type", sample_id="sample",
                    pdf_file=os.path.join(OUT, "polar.pdf"))

    print("\n=== density_chart with pdf_file ===")
    meta2["nFeatures"] = np.abs(np.random.default_rng(2).normal(2500, 400, len(meta2)))
    loc.density_chart(meta2, "condition", "cell_type",
                      numerical_variable="nFeatures",
                      pdf_file=os.path.join(OUT, "density.pdf"))

    print("\n=== lots_of_cells (2-cond) with pdf_file ===")
    loc.lots_of_cells(
        meta2, "condition", "cell_type", sample_id="sample",
        label_order=["diseased", "control"], permutations=100, seed=0,
        pdf_file=os.path.join(OUT, "abundance.pdf"),
    )

    print("\n=== lots_of_cells (>2-cond) + dynamics_chart with pdf_file ===")
    # Add second time point to meta2
    g = loc.lots_of_cells(meta2, "sample", "cell_type",
                          label_order=list(sorted(meta2["sample"].unique()))[:3],
                          permutations=80, seed=0, plot=False)
    loc.dynamics_chart(g, pdf_file=os.path.join(OUT, "dynamics.pdf"))

    print("\n=== entropy_score (2-cond) with pdf_file ===")
    loc.entropy_score(meta2, "condition", "cell_type",
                      label_order=["diseased", "control"],
                      permutations=200, seed=0,
                      pdf_file=os.path.join(OUT, "entropy_2cond.pdf"))

    pdfs = sorted(f for f in os.listdir(OUT) if f.endswith(".pdf"))
    print(f"\nWrote {len(pdfs)} PDF(s):")
    for f in pdfs:
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f}  ({size} bytes)")
    assert len(pdfs) == 8, f"Expected 8 PDFs, got {len(pdfs)}"
    print("\n✓ All plots saved to PDF successfully")


if __name__ == "__main__":
    main()
