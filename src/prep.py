"""Data preparation + feature engineering with fit-on-train / apply-to-test.

Every statistic (medians, winsor caps) is learned on the training batches only
and replayed onto the test batches. Computing them on the full frame would leak
future information into the model — the single most common credit-modelling bug,
and the one the JD screens for ("validate, don't accept at face value").
"""
import numpy as np
import pandas as pd

from src.data_generator import PAST_DUE_COLS, TARGET

SENTINELS = (96, 98)
# numeric columns we winsorise at the 1st/99th percentile to tame data-entry outliers
WINSOR_COLS = [
    "RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome", "age",
]


def prep_fit(train):
    """Learn imputation + winsor parameters from the TRAIN batches only."""
    t = _clear_sentinels(train.copy())
    params = {
        "income_median": float(t["MonthlyIncome"].median()),
        "dependents_median": float(t["NumberOfDependents"].median()),
        # median of legitimate (non-sentinel) past-due counts, per column
        "pastdue_median": {c: float(t[c].median()) for c in PAST_DUE_COLS},
        "winsor": {c: (float(t[c].quantile(0.01)), float(t[c].quantile(0.99)))
                   for c in WINSOR_COLS},
    }
    return params


def prep_apply(df, params):
    """Apply learned params + engineer features. Pure function, returns a copy."""
    d = _clear_sentinels(df.copy())

    # --- missing flags BEFORE imputing (the missingness itself can be signal) ---
    d["income_missing"] = d["MonthlyIncome"].isna().astype(int)
    d["dependents_missing"] = d["NumberOfDependents"].isna().astype(int)
    d["MonthlyIncome"] = d["MonthlyIncome"].fillna(params["income_median"])
    d["NumberOfDependents"] = d["NumberOfDependents"].fillna(
        params["dependents_median"])
    for c in PAST_DUE_COLS:                       # sentinel rows are now NaN -> impute
        d[c] = d[c].fillna(params["pastdue_median"][c])

    # --- winsorise outliers using TRAIN caps ---
    for c, (lo, hi) in params["winsor"].items():
        d[c] = d[c].clip(lo, hi)

    # --- engineered features (kept deliberately few: YAGNI) ---
    d["total_past_due"] = d[PAST_DUE_COLS].sum(axis=1)
    d["has_been_late"] = (d["total_past_due"] > 0).astype(int)
    d["monthly_debt"] = d["DebtRatio"] * d["MonthlyIncome"]
    d["income_per_dependent"] = d["MonthlyIncome"] / (d["NumberOfDependents"] + 1)
    return d


def _clear_sentinels(d):
    """Turn the 96/98 'not available' codes into NaN so they get imputed, not
    treated as someone who was 96 times late. Flag that a row had one."""
    had = pd.Series(0, index=d.index)
    for c in PAST_DUE_COLS:
        is_sent = d[c].isin(SENTINELS)
        had = had | is_sent.astype(int)
        d[c] = d[c].mask(is_sent, np.nan)
    d["had_sentinel_pastdue"] = had
    return d


def feature_columns(df):
    """Model feature list = everything except target and the batch marker."""
    return [c for c in df.columns if c not in (TARGET, "batch")]


if __name__ == "__main__":
    from src.data_generator import assign_batches_and_drift, load_raw
    raw, _ = assign_batches_and_drift(load_raw())
    tr = raw[raw["batch"] <= 3]
    te = raw[raw["batch"] >= 4]
    p = prep_fit(tr)
    out = prep_apply(te, p)
    print("test prep shape:", out.shape, "| NaNs:", int(out.isna().sum().sum()))
    print("96/98 remaining:", int(out[PAST_DUE_COLS].isin(SENTINELS).sum().sum()))
    print("new cols:", [c for c in out.columns if c not in raw.columns])
