"""Champion / challenger model training.

Champion  = Logistic Regression on WoE features. It is the *incumbent* not
because it is strongest but because a WoE scorecard is transparent and
regulator-friendly; the challenger must prove a worthwhile lift to justify
trading that interpretability away.

Challenger = LightGBM on raw features (trees bin internally, so no WoE). It can
capture the step/interaction structure the linear champion only approximates.

Cross-validation refits WoE inside every fold, so the champion's CV score never
sees its own validation rows during binning — the correct, leakage-free protocol.
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.metrics import auc
from src.woe import apply_woe, fit_woe


def train_champion(train, features, target, C=1.0):
    """LR on WoE-transformed features.

    Deliberately NOT class-weighted: a scorecard's value is a calibrated PD
    (predicted prob ≈ true default rate), and class re-weighting inflates the
    probabilities. We absorb the ~6.7% imbalance at the decision threshold (a
    fixed 5% FPR operating point) instead, which leaves discrimination ~unchanged
    while keeping the champion honest as a probability. The challenger, by
    contrast, is class-weighted then isotonic-calibrated.
    """
    woe_model = fit_woe(train, features, target)
    X = apply_woe(train, woe_model)
    lr = LogisticRegression(C=C, max_iter=1000)
    lr.fit(X, train[target])
    return {"model": lr, "woe": woe_model, "features": features, "kind": "champion"}


def score_champion(bundle, df):
    return bundle["model"].predict_proba(apply_woe(df, bundle["woe"]))[:, 1]


def new_challenger(scale_pos_weight=1.0):
    """Unfitted LightGBM with the standard challenger params (one source of truth
    so calibration CV reuses identical settings)."""
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=100, subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1,
    )


def train_challenger(train, features, target):
    """LightGBM on raw features; scale_pos_weight handles the imbalance."""
    n_bad = int(train[target].sum())
    n_good = len(train) - n_bad
    gbm = new_challenger(scale_pos_weight=n_good / max(n_bad, 1))
    gbm.fit(train[features], train[target])
    return {"model": gbm, "features": features, "kind": "challenger"}


def score_challenger(bundle, df):
    return bundle["model"].predict_proba(df[bundle["features"]])[:, 1]


def cross_val_auc(train, features, target, kind, n_splits=5, seed=42):
    """5-fold AUC on the training batches. Champion refits WoE per fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fit = train_champion if kind == "champion" else train_challenger
    score = score_champion if kind == "champion" else score_challenger
    aucs = []
    for tr_idx, va_idx in skf.split(train, train[target]):
        trf, vaf = train.iloc[tr_idx], train.iloc[va_idx]
        bundle = fit(trf, features, target)
        aucs.append(auc(vaf[target], score(bundle, vaf)))
    return float(np.mean(aucs)), float(np.std(aucs))


def champion_coefficients(bundle):
    """LR coefficients by feature — sign sanity check (risky feature -> +coef)."""
    import pandas as pd
    lr = bundle["model"]
    return (pd.DataFrame({"feature": bundle["features"], "coef": lr.coef_[0]})
            .sort_values("coef", key=abs, ascending=False).reset_index(drop=True))


if __name__ == "__main__":
    from src.data_generator import TARGET, assign_batches, load_raw
    from src.prep import prep_apply, prep_fit
    from src.woe import fit_woe as _fw, select_features
    raw = assign_batches(load_raw())             # clean eval window (no drift)
    params = prep_fit(raw[raw["batch"] <= 3])
    tr = prep_apply(raw[raw["batch"] <= 3], params)
    te = prep_apply(raw[raw["batch"] >= 4], params)
    feats0 = [c for c in tr.columns if c not in (TARGET, "batch")]
    keep, flagged = select_features(_fw(tr, feats0, TARGET))
    feats = keep + flagged                       # retain flagged util after review
    champ = train_champion(tr, feats, TARGET)
    chall = train_challenger(tr, feats, TARGET)
    from src.metrics import discrimination_summary
    print("champion  test:", discrimination_summary(te[TARGET], score_champion(champ, te)))
    print("challenger test:", discrimination_summary(te[TARGET], score_challenger(chall, te)))
    print("champion CV AUC:", [round(v, 4) for v in cross_val_auc(tr, feats, TARGET, "champion")])
