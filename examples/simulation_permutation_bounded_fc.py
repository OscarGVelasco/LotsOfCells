"""Permutation-based simulation: bounded fold-change composition score.

Goal
----
Evaluate the sensitivity, specificity, and power of

    per_ct(p, q) = tanh(|log2(p/q)|) * |p - q| / (p + q)
    global_score  = mean(per_ct across cell types)

in a realistic hierarchical design:

- 10 cell types, mixed common/mid/rare baseline.
- Two groups (A vs B), each with N_SAMPLES per-sample compositions drawn
  from Dirichlet(alpha * group_true_proportions) — this models real
  biological variability across patients within a group.
- Each sample's observed proportions come from multinomial sampling of
  N_CELLS cells — the standard finite-cell noise model.
- Group A follows the baseline. Group B has `n_changing` (rarest first)
  cell types perturbed by `FOLD`.
- Observed score computed as mean-of-means: p = mean(A samples),
  q = mean(B samples), then bounded_fc_mean(p, q).
- Null distribution built by exhaustive sample-label permutation
  (identical machinery to gene_source_score).
- p-value = (#null >= observed + 1) / (#permutations + 1).

For each scenario we run N_SIM_REPLICATES independent simulations to
characterise (a) score distribution, (b) p-value distribution, and
(c) statistical power at α = 0.05, 0.01, and 0.001.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
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
N_SAMPLES_PER_GROUP = 6      # C(12, 6) = 924 permutations, exact-enumerable
N_CELLS = 5_000              # cells per sample
N_SIM_REPLICATES = 40        # independent simulations per scenario
FOLD = 4.0                   # dramatic fold-change
ALPHA_BIO = 100.0            # Dirichlet concentration; higher → tighter groups
SCENARIOS = [0, 1, 3, 6, 8]
SEED = 0

# Baseline composition (4 common + 2 mid + 4 rare, ordered high → low)
BASELINE = np.array([
    0.22, 0.18, 0.15, 0.12,
    0.08, 0.06,
    0.04, 0.03, 0.02, 0.01,
])
BASELINE = BASELINE / BASELINE.sum()
CT_NAMES = [f"CT{i:02d}({p*100:.1f}%)" for i, p in enumerate(BASELINE)]

OUT = Path(__file__).resolve().parent / "simulation_output_perm"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------
EPS = 1e-9


def bounded_fc_per_ct(p, q):
    """tanh(|log2(p/q)|) × |p - q| / (p + q) per cell type. Bounded [0, 1]."""
    p = np.asarray(p) + EPS
    q = np.asarray(q) + EPS
    lfc = np.abs(np.log2(p / q))
    return np.tanh(lfc) * np.abs(p - q) / (p + q)


def bounded_fc_mean(p, q):
    return float(np.mean(bounded_fc_per_ct(p, q)))


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------

def perturb(baseline: np.ndarray, n_changing: int, fold: float,
            rare_first: bool = True) -> np.ndarray:
    """Return group B's true proportions (renormalised)."""
    p = baseline.copy()
    idx = np.argsort(p) if rare_first else np.argsort(p)[::-1]
    q = p.copy()
    for k, i in enumerate(idx[:n_changing]):
        direction = 1 if k % 2 == 0 else -1
        q[i] = p[i] * (fold if direction > 0 else 1.0 / fold)
    return q / q.sum()


def draw_samples(true_props: np.ndarray, n_samples: int, alpha: float,
                 n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """Return (n_samples, n_cell_types) matrix of observed proportions.

    Two-level noise:
    1) Biological: each sample's TRUE composition ~ Dirichlet(alpha * true_props)
    2) Technical:  observed counts ~ Multinomial(n_cells, sample_true_props)
    """
    bio = rng.dirichlet(alpha * true_props, size=n_samples)
    obs = np.stack([rng.multinomial(n_cells, p) / n_cells for p in bio])
    return obs


# ---------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------

def compute_observed(A_samples: np.ndarray, B_samples: np.ndarray) -> float:
    p = A_samples.mean(axis=0)
    q = B_samples.mean(axis=0)
    return bounded_fc_mean(p, q)


def permutation_null_exact(all_samples: np.ndarray, n_A: int) -> np.ndarray:
    """Enumerate every possible A/B assignment of the samples."""
    n_total = all_samples.shape[0]
    n_perms = comb(n_total, n_A)
    null = np.empty(n_perms)
    for k, combo in enumerate(combinations(range(n_total), n_A)):
        mask = np.zeros(n_total, dtype=bool)
        mask[list(combo)] = True
        p = all_samples[mask].mean(axis=0)
        q = all_samples[~mask].mean(axis=0)
        null[k] = bounded_fc_mean(p, q)
    return null


# ---------------------------------------------------------------------
# Simulation main loop
# ---------------------------------------------------------------------

def run_simulation():
    rng = np.random.default_rng(SEED)
    records = []
    # Keep one null distribution per scenario for the plots (from rep 0)
    null_examples: dict = {}
    obs_examples: dict = {}

    for n_changing in SCENARIOS:
        p_true = BASELINE
        q_true = perturb(BASELINE, n_changing, FOLD)

        for sim_rep in range(N_SIM_REPLICATES):
            A = draw_samples(p_true, N_SAMPLES_PER_GROUP, ALPHA_BIO, N_CELLS, rng)
            B = draw_samples(q_true, N_SAMPLES_PER_GROUP, ALPHA_BIO, N_CELLS, rng)

            obs = compute_observed(A, B)
            all_samples = np.concatenate([A, B], axis=0)
            null = permutation_null_exact(all_samples, n_A=A.shape[0])
            n_perms = len(null)
            p_val = (np.sum(null >= obs) + 1) / (n_perms + 1)

            records.append({
                "n_changing": n_changing,
                "sim_rep":    sim_rep,
                "obs_score":  obs,
                "null_mean":  float(null.mean()),
                "null_sd":    float(null.std(ddof=1)),
                "null_med":   float(np.median(null)),
                "p_val":      float(p_val),
                "n_perms":    n_perms,
            })

            if sim_rep == 0:
                null_examples[n_changing] = null
                obs_examples[n_changing] = obs

    return pd.DataFrame(records), {"null": null_examples, "obs": obs_examples}


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(f"Simulation: {N_SAMPLES_PER_GROUP} samples/group, {N_CELLS} cells/sample, "
          f"α_bio={ALPHA_BIO}, {FOLD}× fold, {N_SIM_REPLICATES} replicates/scenario")
    n_perms = comb(2 * N_SAMPLES_PER_GROUP, N_SAMPLES_PER_GROUP)
    print(f"Permutations per test: {n_perms} (exact enumeration)")
    print("=" * 78)

    agg = df.groupby("n_changing").agg(
        obs_mean=("obs_score", "mean"),
        obs_sd=("obs_score", "std"),
        null_mean=("null_mean", "mean"),
        pval_mean=("p_val", "mean"),
        pval_med=("p_val", "median"),
    )
    agg["power_0.05"]  = df.groupby("n_changing")["p_val"].apply(lambda s: (s < 0.05).mean())
    agg["power_0.01"]  = df.groupby("n_changing")["p_val"].apply(lambda s: (s < 0.01).mean())
    agg["power_0.001"] = df.groupby("n_changing")["p_val"].apply(lambda s: (s < 0.001).mean())
    print("\n" + agg.round(4).to_string())
    print()


def plot_score_vs_null(df: pd.DataFrame, examples: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(SCENARIOS))

    for x, n_changing in zip(xs, SCENARIOS):
        null = examples["null"][n_changing]
        jitter = np.random.default_rng(int(x)).uniform(-0.15, 0.15, size=len(null))
        ax.scatter(
            np.full_like(null, x, dtype=float) + jitter,
            null,
            s=6, alpha=0.2, color="#B5D4F4",
            label="null (permutation, rep 0)" if x == 0 else None,
        )

    for x, n_changing in zip(xs, SCENARIOS):
        obs = df.loc[df["n_changing"] == n_changing, "obs_score"].to_numpy()
        jitter = np.random.default_rng(100 + int(x)).uniform(-0.15, 0.15, size=len(obs))
        ax.scatter(
            np.full_like(obs, x, dtype=float) + jitter,
            obs,
            s=30, alpha=0.85, color="#B25356", edgecolor="white", linewidth=0.4,
            label="observed (all sim reps)" if x == 0 else None,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(SCENARIOS)
    ax.set_xlabel("Number of cell types changing")
    ax.set_ylabel("bounded_fc_mean")
    n_perms = comb(2 * N_SAMPLES_PER_GROUP, N_SAMPLES_PER_GROUP)
    ax.set_title(
        f"Observed scores vs permutation null\n"
        f"({N_SAMPLES_PER_GROUP} samples/group, {N_SIM_REPLICATES} sim replicates, "
        f"{n_perms} perms each)",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_pvalue_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(3 * len(SCENARIOS), 3), sharey=True)
    for ax, n_changing in zip(axes, SCENARIOS):
        pvals = df.loc[df["n_changing"] == n_changing, "p_val"].to_numpy()
        ax.hist(pvals, bins=np.linspace(0, 1, 21), color="#3C7DA6", alpha=0.7,
                edgecolor="white")
        ax.axvline(0.05, color="#B25356", linewidth=1, linestyle="--", label="α = 0.05")
        ax.set_title(f"n_changing = {n_changing}", fontsize=10)
        ax.set_xlabel("p-value")
        if ax is axes[0]:
            ax.set_ylabel("frequency")
        ax.legend(fontsize=8, frameon=False)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)

    fig.suptitle(
        "P-value distribution across simulation replicates\n"
        "(n_changing=0 should be ~uniform; the rest should skew toward 0)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_power_curve(df: pd.DataFrame, out: Path) -> None:
    alphas = [0.05, 0.01, 0.001]
    colors = ["#3C7DA6", "#B25356", "#276223"]
    power = pd.DataFrame({
        f"α = {a}": df.groupby("n_changing")["p_val"].apply(lambda s: (s < a).mean())
        for a in alphas
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    for (col, c) in zip(power.columns, colors):
        ax.plot(power.index, power[col], marker="o", color=c, label=col, linewidth=1.5)
    ax.set_xlabel("Number of cell types changing")
    ax.set_ylabel("Statistical power")
    ax.set_title(
        f"Power curves — fraction of replicates rejecting H0\n"
        f"(from {N_SIM_REPLICATES} simulation replicates per scenario)",
        fontsize=10,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(SCENARIOS)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    df, examples = run_simulation()
    print_summary(df)

    plot_score_vs_null(df, examples, OUT / "1_observed_vs_null.pdf")
    plot_pvalue_distribution(df, OUT / "2_pvalue_distribution.pdf")
    plot_power_curve(df, OUT / "3_power_curve.pdf")

    print(f"PDFs written to: {OUT}")


if __name__ == "__main__":
    main()
