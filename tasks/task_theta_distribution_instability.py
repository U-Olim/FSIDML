"""Create two-panel theta_hat density plot for instability scenarios."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "documents/outputs/raw/simulations.csv"
OUT_PATH = PROJECT_ROOT / "documents/outputs/figures/paper/fig_theta_distribution_instability.png"

SCENARIOS = [(200, 100), (300, 150)]
LEARNERS = ["ols", "lasso", "elastic_net"]
LABELS = {"ols": "OLS", "lasso": "Lasso", "elastic_net": "Elastic Net"}
COLORS = {"ols": "#264653", "lasso": "#2a9d8f", "elastic_net": "#e76f51"}


def _kde_gaussian(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Simple Gaussian KDE with Silverman's bandwidth."""

    n = samples.size
    if n < 2:
        return np.zeros_like(x_grid)
    std = float(np.std(samples, ddof=1))
    if std <= 0:
        return np.zeros_like(x_grid)
    h = 1.06 * std * (n ** (-1.0 / 5.0))
    if h <= 0:
        return np.zeros_like(x_grid)
    u = (x_grid[:, None] - samples[None, :]) / h
    density = np.mean(np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi), axis=1) / h
    return density


def run() -> Path:
    df = pd.read_csv(RAW_PATH)
    out_dir = OUT_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 220,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)

    # Stable x-range across both panels from relevant subset.
    sub_all = df.loc[
        ((df["n"] == 200) & (df["p"] == 100)) | ((df["n"] == 300) & (df["p"] == 150))
    ]
    x_min = float(np.quantile(sub_all["theta_hat"], 0.001))
    x_max = float(np.quantile(sub_all["theta_hat"], 0.999))
    x_pad = 0.05 * (x_max - x_min)
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 700)

    for ax, (n, p) in zip(axes, SCENARIOS):
        panel = df.loc[(df["n"] == n) & (df["p"] == p)]
        for learner in LEARNERS:
            samples = panel.loc[panel["learner_name"] == learner, "theta_hat"].to_numpy(dtype=float)
            density = _kde_gaussian(samples, x_grid=x_grid)
            ax.plot(
                x_grid,
                density,
                color=COLORS[learner],
                linewidth=2.0,
                label=LABELS[learner],
            )
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"Scenario n={n}, p={p}")
        ax.set_xlabel(r"$\hat{\theta}$")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle(r"Theta-Hat Density by Learner in Instability Scenarios", y=1.14)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)
    return OUT_PATH


if __name__ == "__main__":
    output = run()
    print(output)
