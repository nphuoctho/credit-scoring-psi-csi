"""Chart helpers — every figure the README embeds and the notebook renders.

Pure plotting: callers pass already-computed numbers, these just draw and save a
PNG. Headless Agg backend so it runs in CI / a notebook execute step.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import seaborn as sns                    # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve    # noqa: E402

sns.set_theme(style="whitegrid")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y, scores, path):
    """ROC curves for {label: score}. `scores` also keys the legend AUC."""
    from src.metrics import auc
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for label, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.axvline(0.05, color="crimson", lw=1, ls=":", label="5% FPR operating point")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC — champion vs challenger")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, path)


def plot_lift(lift_df, path, label="challenger"):
    """Decile lift bars (decile 9 = highest predicted risk)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(lift_df.index.astype(str), lift_df["lift"], color="#3b6fb6")
    ax.axhline(1.0, color="grey", ls="--", lw=1, label="population base rate")
    ax.set(xlabel="score decile (9 = riskiest)", ylabel="lift vs base rate",
           title=f"Decile lift — {label}")
    ax.legend()
    _save(fig, path)


def plot_calibration(y, scores, path):
    """Reliability curves: predicted PD vs observed default rate."""
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for label, s in scores.items():
        frac_pos, mean_pred = calibration_curve(y, s, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=label, ms=4)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfectly calibrated")
    ax.set(xlabel="mean predicted probability", ylabel="observed default rate",
           title="Calibration (reliability) curve")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, path)


def plot_psi_trend(psi_score_df, path):
    """Score PSI across monitoring batches with the 0.1 / 0.25 action lines."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(psi_score_df["batch"], psi_score_df["psi"], "o-", color="#b6413b", lw=2)
    ax.axhline(0.10, color="orange", ls="--", lw=1, label="0.10 — monitor")
    ax.axhline(0.25, color="red", ls="--", lw=1, label="0.25 — retrain")
    ax.set(xlabel="batch (simulated month)", ylabel="score PSI vs baseline",
           title="Score PSI trend (population stability)")
    ax.legend()
    _save(fig, path)


def plot_csi_heatmap(csi_matrix, path):
    """Feature x batch CSI heatmap; cells > 0.25 are the drift signal."""
    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(csi_matrix) + 1.5))
    sns.heatmap(csi_matrix, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=0.5,
                cbar_kws={"label": "CSI"}, ax=ax, linewidths=0.5)
    ax.set(title="CSI heatmap (feature drift by batch)", xlabel="", ylabel="")
    _save(fig, path)


def plot_iv(iv_df, path, top=10):
    """Information Value ranking (top features)."""
    d = iv_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ["#c0392b" if s.startswith("suspicious") else "#3b6fb6"
              for s in d["strength"]]
    ax.barh(d["feature"], d["iv"], color=colors)
    ax.axvline(0.5, color="crimson", ls=":", lw=1, label="0.5 — leakage check")
    ax.set(xlabel="Information Value", title="Feature IV (red = IV>0.5, investigate)")
    ax.legend()
    _save(fig, path)
