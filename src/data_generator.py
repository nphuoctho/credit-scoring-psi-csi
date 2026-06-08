"""Synthetic 'Give Me Some Credit'-like dataset + time-batch / drift simulation.

Why synthetic: the real Kaggle CSV needs an account to download. This generator
reproduces the SAME 11-column schema and the SAME data quirks (target imbalance
~6.7%, missing income/dependents, revolving/debt-ratio outliers, the 96/98
sentinel codes in the past-due columns) so the whole pipeline runs end-to-end.
`load_raw()` prefers a real `data/cs-training.csv` if present — schema matches,
so dropping the real file in swaps it transparently.

The 6 'months' are simulated (the dataset has no timestamp). We inject a known
covariate drift on one feature in batches 4-6 and keep a control feature flat,
so PSI/CSI in monitoring can be checked against ground truth.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"
# Exact GMSC column names (hyphens included) so a real CSV is a drop-in.
COLS = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "cs-training.csv"


def _zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-9)


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _solve_intercept(base_logit, prevalence, lo=-20.0, hi=20.0):
    """Bisection on a constant offset so mean default rate == target prevalence.

    Decouples base rate from the risk coefficients, so we can tune model
    separability without disturbing the ~6.7% class balance.
    """
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(base_logit + mid).mean() > prevalence:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_base(n=150_000, seed=42, prevalence=0.067):
    """Build a synthetic GMSC-shaped frame with realistic risk signal + quirks."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(52, 14, n), 21, 95).round()
    income = np.exp(rng.normal(8.6, 0.6, n))                  # lognormal ~ $5k median
    util = np.clip(rng.gamma(1.4, 0.35, n), 0, None)          # mostly 0-1.x
    debt_ratio = np.clip(rng.gamma(1.6, 0.22, n), 0, None)
    open_lines = rng.poisson(8, n)
    realestate = rng.poisson(1.0, n)
    dependents = rng.poisson(0.8, n).astype(float)
    # past-due counts: mostly zero, heavy right skew
    pd30 = rng.poisson(0.18, n)
    pd60 = rng.poisson(0.07, n)
    pd90 = rng.poisson(0.05, n)
    past_due_total = pd30 + pd60 + pd90

    # --- true log-odds of default: linear drivers + a non-linearity ---
    # The step/interaction terms are what gives the tree challenger a real edge
    # over the linear champion (it captures them more fully).
    base_logit = (
        0.60 * _zscore(util)
        + 0.42 * _zscore(debt_ratio)
        - 0.45 * _zscore(np.log(income))
        - 0.30 * _zscore(age)
        + 0.42 * past_due_total
        + 0.55 * (past_due_total >= 2)                        # step effect (non-linear)
        + 0.30 * (_zscore(util) * (debt_ratio > debt_ratio.mean()))  # interaction
    )
    intercept = _solve_intercept(base_logit, prevalence)      # pin base rate ~6.7%
    p = _sigmoid(base_logit + intercept)
    y = rng.binomial(1, p)

    df = pd.DataFrame({
        TARGET: y,
        "RevolvingUtilizationOfUnsecuredLines": util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": pd30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": pd90,
        "NumberRealEstateLoansOrLines": realestate,
        "NumberOfTime60-89DaysPastDueNotWorse": pd60,
        "NumberOfDependents": dependents,
    })

    _inject_quirks(df, rng)
    return df


def _inject_quirks(df, rng):
    """Corrupt observed values exactly like the real dataset does."""
    n = len(df)
    # ~19.8% MonthlyIncome missing, ~2.6% NumberOfDependents missing
    df.loc[rng.random(n) < 0.198, "MonthlyIncome"] = np.nan
    df.loc[rng.random(n) < 0.026, "NumberOfDependents"] = np.nan
    # extreme revolving-utilization / debt-ratio outliers (data entry artefacts)
    out_idx = rng.choice(n, size=max(1, n // 1500), replace=False)
    df.loc[out_idx, "RevolvingUtilizationOfUnsecuredLines"] = rng.uniform(
        2000, 50708, len(out_idx))
    out_idx2 = rng.choice(n, size=max(1, n // 1500), replace=False)
    df.loc[out_idx2, "DebtRatio"] = rng.uniform(2000, 329000, len(out_idx2))
    # one age=0 typo, like the real file
    df.loc[rng.integers(0, n), "age"] = 0
    # sentinel 96 / 98 placeholders in the three past-due columns
    for col in PAST_DUE_COLS:
        sent = rng.choice(n, size=max(1, n // 800), replace=False)
        df.loc[sent, col] = rng.choice([96, 98], len(sent))


def load_raw():
    """Real CSV if present (schema-compatible), else synthetic."""
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
                     errors="ignore")
        return df.reset_index(drop=True)
    return generate_base()


def assign_batches_and_drift(df, seed=42, n_batches=6):
    """Shuffle into N equal 'monthly' batches, then inject a known drift.

    Drift = covariate shift applied to ONE feature in the later batches AFTER
    labels exist, so it is pure input drift the monitor must detect. A second
    feature is left flat as a control (CSI on it must stay quiet).
    """
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["batch"] = (np.arange(len(df)) * n_batches // len(df)) + 1

    drift = {"feature": "DebtRatio", "control": "age",
             "factors": {4: 1.15, 5: 1.25, 6: 1.40}}
    for b, f in drift["factors"].items():
        mask = df["batch"] == b
        df.loc[mask, drift["feature"]] = df.loc[mask, drift["feature"]] * f
    return df, drift


if __name__ == "__main__":
    d = load_raw()
    d, gt = assign_batches_and_drift(d)
    print(d.shape, "| base rate:", round(d[TARGET].mean(), 4))
    print("drift ground truth:", gt)
    print(d.groupby("batch")["DebtRatio"].mean().round(3))
