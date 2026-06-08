"""Weight of Evidence (WoE) and Information Value (IV) from scratch.

Hand-rolled on purpose: a credit interviewer will ask "how is WoE computed, what
does IV mean, why is IV > 0.5 suspicious" and you must read every term, not call
a library `.fit()`.

  WoE(bin)  = ln( %good_in_bin / %bad_in_bin )      good = non-default (y==0)
  IV(bin)   = (%good - %bad) * WoE(bin)
  IV(feat)  = sum over bins

WoE replaces a raw value with the log-odds evidence of its bin, which linearises
the feature->log-odds relationship so a linear model (the LR champion) can use it
and which absorbs outliers and missing into discrete bins. Bins are cut on TRAIN
and replayed on TEST (re-binning test would leak + break comparability).
A high IV is NOT automatically good: IV > 0.5 usually means leakage, not a great
feature, so we flag it for investigation.
"""
import numpy as np
import pandas as pd

SMOOTH = 0.5     # Haldane correction so an all-good / all-bad bin doesn't blow up ln()

IV_STRENGTH = [   # standard credit-scoring interpretation bands
    (0.02, "useless"), (0.1, "weak"), (0.3, "medium"),
    (0.5, "strong"), (np.inf, "suspicious (>0.5 -> check leakage)"),
]


def _strength(iv):
    for hi, label in IV_STRENGTH:
        if iv < hi:
            return label
    return "suspicious"


def _table_from_codes(codes, y):
    """WoE/IV table given integer/categorical bin codes aligned with target y."""
    g = pd.DataFrame({"bin": np.asarray(codes), "y": np.asarray(y)})
    agg = g.groupby("bin", observed=True)["y"].agg(count="count", bad="sum")
    agg["good"] = agg["count"] - agg["bad"]
    gs = agg["good"] + SMOOTH
    bs = agg["bad"] + SMOOTH
    agg["pct_good"] = gs / gs.sum()
    agg["pct_bad"] = bs / bs.sum()
    agg["woe"] = np.log(agg["pct_good"] / agg["pct_bad"])
    agg["iv"] = (agg["pct_good"] - agg["pct_bad"]) * agg["woe"]
    return agg


def woe_iv(feature, target, bins=10):
    """Standalone WoE/IV for one feature. Returns (table, iv)."""
    x = pd.Series(np.asarray(feature))
    if x.nunique() <= bins:                       # low-cardinality -> bin by value
        codes = x.to_numpy()
    else:
        codes = pd.qcut(x, bins, duplicates="drop", labels=False)
    tbl = _table_from_codes(codes, target)
    return tbl, float(tbl["iv"].sum())


def fit_woe(train, features, target, bins=10):
    """Learn bin edges + per-bin WoE on TRAIN for every feature."""
    model = {}
    y = train[target].to_numpy()
    for f in features:
        x = train[f]
        if x.nunique() <= bins:                   # value bins (flags, small counts)
            codes = x.to_numpy()
            tbl = _table_from_codes(codes, y)
            model[f] = {"kind": "value",
                        "map": dict(zip(tbl.index, tbl["woe"])),
                        "iv": float(tbl["iv"].sum())}
        else:                                     # quantile bins for continuous
            codes, edges = pd.qcut(x, bins, duplicates="drop",
                                   labels=False, retbins=True)
            edges = edges.copy()
            edges[0], edges[-1] = -np.inf, np.inf  # open the ends to catch test outliers
            tbl = _table_from_codes(codes, y)
            model[f] = {"kind": "quantile", "edges": edges,
                        "woe": tbl["woe"].reindex(range(len(edges) - 1)).ffill()
                        .bfill().to_numpy(),
                        "iv": float(tbl["iv"].sum())}
    return model


def apply_woe(df, model):
    """Map raw feature values to their TRAIN bin's WoE. Returns a WoE-only frame."""
    out = {}
    for f, m in model.items():
        x = df[f]
        if m["kind"] == "value":
            out[f] = x.map(m["map"]).fillna(0.0).to_numpy()   # unseen value -> neutral
        else:
            codes = pd.cut(x, bins=m["edges"], labels=False, include_lowest=True)
            out[f] = pd.Series(m["woe"][codes.astype(int)], index=df.index).to_numpy()
    return pd.DataFrame(out, index=df.index)


def iv_table(model):
    """Ranked IV summary with the strength band per feature."""
    rows = [{"feature": f, "iv": round(m["iv"], 4), "strength": _strength(m["iv"])}
            for f, m in model.items()]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def select_features(model, lo=0.02, hi=0.5):
    """Keep predictive features (IV in [lo, hi]); drop useless, flag leakage."""
    keep = [f for f, m in model.items() if lo <= m["iv"] <= hi]
    flagged = [f for f, m in model.items() if m["iv"] > hi]
    return keep, flagged


if __name__ == "__main__":
    from src.data_generator import assign_batches_and_drift, load_raw
    from src.prep import feature_columns, prep_apply, prep_fit
    raw, _ = assign_batches_and_drift(load_raw())
    tr = prep_apply(raw[raw["batch"] <= 3], prep_fit(raw[raw["batch"] <= 3]))
    feats = feature_columns(tr)
    model = fit_woe(tr, feats, "SeriousDlqin2yrs")
    print(iv_table(model).to_string(index=False))
    keep, flagged = select_features(model)
    print("\nkept:", len(keep), "| flagged(>0.5):", flagged)
