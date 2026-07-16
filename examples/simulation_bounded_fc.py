"""Simulation study: bounded fold-change composition score.

Question: does the metric

    per_ct(p, q) = tanh(|log2(p/q)|) * |p - q| / (p + q)
    global_score  = mean(per_ct across cell types)

behave sensibly as more cell types undergo dramatic composition changes?

Setup
-----
- 10 cell types with a mixed baseline (common + mid + rare).
- Scenarios: n_changing ∈ {0, 1, 3, 6, 8}. In each scenario the RAREST
  n_changing cell types undergo a `FOLD`-fold change, direction
  alternating so total mass is roughly conserved. That targets the
  motivating case: rare cells changing dramatically.
- For each scenario: 100 finite-cell replicates via multinomial sampling
  of N_CELLS cells per group. This mimics real experimental sampling
  noise.
- Three scores compared:
    * `bounded_fc_mean`     — the proposed metric
    * `kl_symmetric`        — current entropy_score-like KL sum
    * `total_variation`     — L1/2 for reference (bounded [0,1])

Outputs (PDFs written to `examples/simulation_output/`):
    1. score_vs_ncells.pdf   — score curves across scenarios
    2. per_ct_breakdown.pdf  — which cells drive the score, one scenario
    3. truth_composition.pdf — the true p and q used in each scenario
"""
from __future__ import annotations

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
N_CELLS = 5_000               # cells per group per replicate
N_REPLICATES = 100
FOLD = 4.0                    # "dramatic" fold-change
SCENARIOS = [0, 1, 3, 6, 8]
SEED = 42

# Baseline: 4 common + 2 mid + 4 rare, summing to 1.
BASELINE = np.array([
    0.22, 0.18, 0.15, 0.12,       # common
    0.08, 0.06,                   # mid
    0.04, 0.03, 0.02, 0.01,       # rare
])
BASELINE = BASELINE / BASELINE.sum()

CT_NAMES = [f"CT{i:02d} ({p*100:.1f}%)" for i, p in enumerate(BASELINE)]

OUT = Path(__file__).resolve().parent / "simulation_output"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
EPS = 1e-9


def bounded_fc_per_ct(p, q):
    """Per-cell-type: tanh(|log2(p/q)|) × |p - q| / (p + q). Bounded [0, 1]."""
    p = np.asarray(p) + EPS
    q = np.asarray(q) + EPS
    lfc = np.abs(np.log2(p / q))
    return np.tanh(lfc) * np.abs(p - q) / (p + q)


def bounded_fc_mean(p, q):
    return float(np.mean(bounded_fc_per_ct(p, q)))


def kl_symmetric(p, q):
    """KL(p||q) + KL(q||p) — the current entropy_score family."""
    p = np.asarray(p) + EPS
    q = np.asarray(q) + EPS
    return float(np.sum(p * np.log2(p / q)) + np.sum(q * np.log2(q / p)))


def total_variation(p, q):
    """L1 / 2. Bounded [0, 1]. All cell types weighted by absolute diff."""
    return float(0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q))))


METRICS = {
    "bounded_fc_mean": bounded_fc_mean,
    "kl_symmetric":    kl_symmetric,
    "total_variation": total_variation,
}


# ---------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------

def perturb(baseline: np.ndarray, n_changing: int, fold: float,
            rare_first: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Perturb the `n_changing` rarest cell types by `fold`.

    Direction alternates up/down so the sum stays close to 1 before the
    final renormalisation.
    """
    p = baseline.copy()
    idx = np.argsort(p)  # ascending → rarest first
    if not rare_first:
        idx = idx[::-1]

    q = p.copy()
    for k, i in enumerate(idx[:n_changing]):
        direction = 1 if k % 2 == 0 else -1
        q[i] = p[i] * (fold if direction > 0 else 1.0 / fold)

    p = p / p.sum()
    q = q / q.sum()
    return p, q


def sample_multinomial(p: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    counts = rng.multinomial(n, p)
    return counts / counts.sum()


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

def run_simulation() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records = []
    for n_changing in SCENARIOS:
        p_true, q_true = perturb(BASELINE, n_changing, FOLD)
        for rep in range(N_REPLICATES):
            p_obs = sample_multinomial(p_true, N_CELLS, rng)
            q_obs = sample_multinomial(q_true, N_CELLS, rng)
            rec = {"n_changing": n_changing, "rep": rep}
            for name, fn in METRICS.items():
                rec[name] = fn(p_obs, q_obs)
            records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_score_curves(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.0))
    for ax, (name, _) in zip(axes, METRICS.items()):
        grouped = df.groupby("n_changing")[name]
        means = grouped.mean()
        stds = grouped.std()

        ax.errorbar(
            means.index, means.values, yerr=stds.values,
            marker="o", capsize=3, color="#3C7DA6", linewidth=1.5,
        )
        # Scatter individual replicates for context
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(df))
        ax.scatter(
            df["n_changing"] + jitter, df[name],
            s=6, alpha=0.15, color="#C1DEEF",
        )
        ax.set_xlabel("cell types changing")
        ax.set_ylabel(name)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
        ax.set_xticks(SCENARIOS)

    fig.suptitle(
        f"Composition-change scores under increasing dysregulation\n"
        f"({N_CELL_TYPES} cell types, {FOLD}× fold change, rarest first, "
        f"{N_REPLICATES} replicates)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_per_ct_breakdown(rng: np.random.Generator, out: Path) -> None:
    """For each scenario, show the per-cell-type contribution to the
    bounded_fc_mean score. Uses the TRUE (noise-free) p, q.
    """
    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(9, 1.9 * len(SCENARIOS)),
                              sharex=True)
    x = np.arange(N_CELL_TYPES)
    colors = ["#B25356" if BASELINE[i] < 0.05 else "#3C7DA6" for i in range(N_CELL_TYPES)]

    for ax, n_changing in zip(axes, SCENARIOS):
        p_true, q_true = perturb(BASELINE, n_changing, FOLD)
        contrib = bounded_fc_per_ct(p_true, q_true)
        ax.bar(x, contrib, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(np.mean(contrib), color="black", linestyle="--", linewidth=0.7,
                   label=f"mean = {np.mean(contrib):.3f}")
        ax.set_ylabel(f"n={n_changing}\nper-CT score", fontsize=8)
        ax.set_ylim(0, max(0.4, contrib.max() * 1.1))
        ax.legend(loc="upper right", fontsize=7, frameon=False)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(CT_NAMES, rotation=45, ha="right", fontsize=8)
    fig.suptitle(
        "Per-cell-type contribution to the bounded_fc score\n"
        "(red = rare cell types < 5% baseline)",
        fontsize=10, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_truth_composition(out: Path) -> None:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(3 * len(SCENARIOS), 3.4),
                              sharey=True)
    x = np.arange(N_CELL_TYPES)
    w = 0.4
    for ax, n_changing in zip(axes, SCENARIOS):
        p_true, q_true = perturb(BASELINE, n_changing, FOLD)
        ax.bar(x - w/2, p_true, width=w, color="#3C7DA6", label="p")
        ax.bar(x + w/2, q_true, width=w, color="#B25356", label="q")
        ax.set_title(f"n_changing = {n_changing}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"CT{i:02d}" for i in range(N_CELL_TYPES)],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 0.35)
        if ax is axes[0]:
            ax.set_ylabel("proportion")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)

    fig.suptitle("True composition p (baseline) vs q (perturbed)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    df = run_simulation()

    print("\nMean score per scenario:")
    print(df.groupby("n_changing")[list(METRICS.keys())].mean().round(4))
    print("\nStandard deviation across replicates:")
    print(df.groupby("n_changing")[list(METRICS.keys())].std().round(4))

    plot_truth_composition(OUT / "1_truth_composition.pdf")
    plot_score_curves(df, OUT / "2_score_vs_ncells.pdf")
    plot_per_ct_breakdown(np.random.default_rng(SEED),
                          OUT / "3_per_ct_breakdown.pdf")

    print(f"\nPDFs written to: {OUT}")


if __name__ == "__main__":
    main()
