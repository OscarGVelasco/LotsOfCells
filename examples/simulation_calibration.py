"""Calibration reference for the bounded_fc_mean score.

Goal
----
Not a hypothesis test — a *lookup table*. Given a fixed cohort (samples,
cells, biological variability), what scores do we expect when
`fraction_changing` percent of cell types are dysregulated at `fold`
magnitude?

Once these quantile bands are computed, a user with a real observed
score can locate it on the reference chart and read off an approximate
level of global dysregulation.

Design
------
- 10 cell types, mixed common/mid/rare baseline (same as before).
- 6 samples per group, 5000 cells per sample, α_bio = 100.
- Sweep:
    fraction_changing ∈ {0, 10%, 20%, ..., 100%}
    fold ∈ {2×, 3×, 5×}
- 500 replicates per (fraction, fold) combination.
- Report per-scenario quantiles: 5, 25, 50, 75, 95%.
- Produce:
    (a) reference table (fraction × fold → score quantiles)
    (b) calibration curves (median ± IQR)
    (c) heatmap (fraction × fold → median score)
    (d) an `interpret_score(score, fold=4.0)` helper.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
N_CELL_TYPES = 10
N_SAMPLES_PER_GROUP = 6
N_CELLS = 5_000
N_REPLICATES = 500
ALPHA_BIO = 100.0

FRACTIONS = np.linspace(0.0, 1.0, 11)          # 0%, 10%, ..., 100%
FOLDS = [2.0, 3.0, 5.0]                        # modest, dramatic, severe
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

SEED = 0

BASELINE = np.array([
    0.22, 0.18, 0.15, 0.12,
    0.08, 0.06,
    0.04, 0.03, 0.02, 0.01,
])
BASELINE = BASELINE / BASELINE.sum()

OUT = Path(__file__).resolve().parent / "simulation_output_calibration"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------
EPS = 1e-9


def bounded_fc_mean(p, q):
    p = np.asarray(p) + EPS
    q = np.asarray(q) + EPS
    lfc = np.abs(np.log2(p / q))
    return float(np.mean(np.tanh(lfc) * np.abs(p - q) / (p + q)))


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------

def perturb(baseline, n_changing, fold, rare_first=True):
    """Perturb `n_changing` cell types by `fold`, alternating up/down."""
    p = baseline.copy()
    idx = np.argsort(p) if rare_first else np.argsort(p)[::-1]
    q = p.copy()
    for k, i in enumerate(idx[:n_changing]):
        direction = 1 if k % 2 == 0 else -1
        q[i] = p[i] * (fold if direction > 0 else 1.0 / fold)
    return q / q.sum()


def draw_group(true_props, n_samples, alpha, n_cells, rng):
    bio = rng.dirichlet(alpha * true_props, size=n_samples)
    return np.stack([rng.multinomial(n_cells, p) / n_cells for p in bio])


def one_replicate(p_true, q_true, rng):
    A = draw_group(p_true, N_SAMPLES_PER_GROUP, ALPHA_BIO, N_CELLS, rng)
    B = draw_group(q_true, N_SAMPLES_PER_GROUP, ALPHA_BIO, N_CELLS, rng)
    return bounded_fc_mean(A.mean(axis=0), B.mean(axis=0))


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

def run_simulation():
    rng = np.random.default_rng(SEED)
    records = []
    for fold in FOLDS:
        for frac in FRACTIONS:
            n_changing = int(round(frac * N_CELL_TYPES))
            p_true = BASELINE
            q_true = perturb(BASELINE, n_changing, fold)
            for rep in range(N_REPLICATES):
                score = one_replicate(p_true, q_true, rng)
                records.append({
                    "fold": fold,
                    "fraction_changing": frac,
                    "n_changing": n_changing,
                    "rep": rep,
                    "score": score,
                })
    return pd.DataFrame(records)


def quantile_table(df: pd.DataFrame) -> pd.DataFrame:
    def q_row(s):
        return pd.Series({f"q{int(q*100):02d}": np.quantile(s, q) for q in QUANTILES})
    out = df.groupby(["fold", "fraction_changing"])["score"].apply(q_row).unstack(-1)
    out.index.names = ["fold", "fraction_changing"]
    return out.round(4)


# ---------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------

def build_interpreter(df: pd.DataFrame):
    """Return a function score -> (fraction estimate, 90% CI, matched fold)."""
    # Median score per (fold, fraction)
    med = df.groupby(["fold", "fraction_changing"])["score"].median().reset_index()

    def interpret(score: float, fold: float = 4.0) -> dict:
        # Nearest available fold in the reference
        available_folds = np.array(sorted(df["fold"].unique()))
        use_fold = float(available_folds[np.argmin(np.abs(available_folds - fold))])

        sub = med[med["fold"] == use_fold].sort_values("fraction_changing")
        # Interpolate fraction from score, then look up 5-95% band.
        fracs = sub["fraction_changing"].to_numpy()
        scores = sub["score"].to_numpy()

        # Score is monotonically non-decreasing in fraction (usually), interpolate.
        # np.interp expects xp increasing; if not, sort.
        sort_idx = np.argsort(scores)
        est_frac = float(np.interp(score, scores[sort_idx], fracs[sort_idx]))

        # 5-95% band at the closest fraction
        closest = df[(df["fold"] == use_fold) &
                     (df["fraction_changing"] == fracs[np.argmin(np.abs(scores - score))])]
        lo = float(np.quantile(closest["score"], 0.05))
        hi = float(np.quantile(closest["score"], 0.95))

        return {
            "score": score,
            "reference_fold": use_fold,
            "estimated_fraction_changing": est_frac,
            "matched_scenario_5-95_band": (lo, hi),
        }

    return interpret


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_calibration_curves(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4E79A7", "#E15759", "#59A14F"]
    for fold, color in zip(FOLDS, colors):
        sub = df[df["fold"] == fold].groupby("fraction_changing")["score"]
        med = sub.median()
        q05 = sub.quantile(0.05)
        q95 = sub.quantile(0.95)
        q25 = sub.quantile(0.25)
        q75 = sub.quantile(0.75)

        ax.fill_between(med.index, q05, q95, color=color, alpha=0.12,
                         label=f"{fold}× fold — 5–95%")
        ax.fill_between(med.index, q25, q75, color=color, alpha=0.28,
                         label=f"{fold}× fold — IQR")
        ax.plot(med.index, med.values, color=color, marker="o",
                linewidth=1.8, label=f"{fold}× fold — median")

    ax.set_xlabel("Fraction of cell types dysregulated")
    ax.set_ylabel("bounded_fc_mean")
    ax.set_title(
        f"Calibration reference — score vs fraction of cell types changing\n"
        f"({N_CELL_TYPES} cell types, {N_SAMPLES_PER_GROUP} samples/group, "
        f"{N_CELLS} cells/sample, {N_REPLICATES} reps/point)",
        fontsize=10,
    )
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels([f"{int(f*100)}%" for f in FRACTIONS])
    ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out: Path) -> None:
    piv = df.groupby(["fold", "fraction_changing"])["score"].median().unstack(-1)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(piv.values, aspect="auto", cmap="magma_r",
                   vmin=0, vmax=piv.values.max())
    ax.set_xticks(np.arange(len(FRACTIONS)))
    ax.set_xticklabels([f"{int(f*100)}%" for f in FRACTIONS])
    ax.set_yticks(np.arange(len(FOLDS)))
    ax.set_yticklabels([f"{f}×" for f in FOLDS])
    ax.set_xlabel("Fraction of cell types dysregulated")
    ax.set_ylabel("Fold change")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > piv.values.max() * 0.55 else "black",
                    fontsize=8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("median bounded_fc_mean")
    ax.set_title("Reference table (median score) — locate your observed score here",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_score_bands(df: pd.DataFrame, fold: float, out: Path) -> None:
    """Ridge-style violin per fraction level, for a chosen fold."""
    sub = df[df["fold"] == fold]
    fracs = sorted(sub["fraction_changing"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(fracs))
    data = [sub.loc[sub["fraction_changing"] == f, "score"].to_numpy() for f in fracs]
    parts = ax.violinplot(data, positions=positions, widths=0.75,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#B25356")
        pc.set_alpha(0.45)
        pc.set_edgecolor("#5F1C1E")
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(f*100)}%" for f in fracs])
    ax.set_xlabel("Fraction of cell types dysregulated")
    ax.set_ylabel("bounded_fc_mean")
    ax.set_title(
        f"Score distribution per dysregulation level (fold = {fold}×)\n"
        f"Read this chart: your observed score → nearest violin → that's the "
        f"estimated fraction",
        fontsize=10,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    df = run_simulation()

    print("=" * 82)
    print(f"Calibration: {N_CELL_TYPES} cell types, {N_SAMPLES_PER_GROUP} samples/group, "
          f"{N_CELLS} cells/sample, α_bio={ALPHA_BIO}")
    print(f"{N_REPLICATES} replicates × {len(FOLDS)} folds × {len(FRACTIONS)} fractions "
          f"= {N_REPLICATES * len(FOLDS) * len(FRACTIONS):,} simulations")
    print("=" * 82)

    qtab = quantile_table(df)
    print("\nQuantile reference table (rows = fold × fraction_changing):\n")
    print(qtab.to_string())

    # Save the reference table for programmatic use.
    qtab.to_csv(OUT / "calibration_reference_table.csv")
    print(f"\nSaved: {OUT / 'calibration_reference_table.csv'}")

    plot_calibration_curves(df, OUT / "1_calibration_curves.pdf")
    plot_heatmap(df, OUT / "2_score_heatmap.pdf")
    plot_score_bands(df, fold=4.0 if 4.0 in FOLDS else FOLDS[-1], out=OUT / "3_violins_5x.pdf")
    plot_score_bands(df, fold=FOLDS[0], out=OUT / "3_violins_2x.pdf")
    plot_score_bands(df, fold=FOLDS[-1], out=OUT / "3_violins_5x.pdf")

    # Interpretation function demo
    interpret = build_interpreter(df)
    print("\nExample lookups (given an observed score, estimate fraction dysregulated):")
    for score in [0.05, 0.10, 0.18, 0.30, 0.45, 0.60]:
        r = interpret(score, fold=3.0)
        lo, hi = r["matched_scenario_5-95_band"]
        print(f"  score={score:.2f} @ 3× fold → "
              f"~{r['estimated_fraction_changing']*100:.0f}% dysregulated  "
              f"(closest-scenario 5-95%: [{lo:.3f}, {hi:.3f}])")

    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    main()
