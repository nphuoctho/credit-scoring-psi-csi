"""Champion-challenger ship/hold decision.

The comparison is pinned at a business operating point (a fixed 5% FPR — i.e. we
accept wrongly declining 5% of good customers) and asks which model catches more
defaulters there. The lift is bootstrapped with PAIRED resampling (both models
scored on the same resampled rows each round) so the confidence interval reflects
the correlated, like-for-like difference rather than two independent noises.

Decision rule is fixed BEFORE seeing the numbers (pre-registered) to avoid
tuning the threshold to the result: ship only if the recall lift clears a
material bar AND the CI lower bound stays positive.
"""
import numpy as np

from src.metrics import recall_at_fpr


def bootstrap_recall_diff(y, s_champ, s_chall, target_fpr=0.05, n=1000, seed=42):
    """Paired bootstrap of (challenger_recall - champion_recall) at fixed FPR."""
    y = np.asarray(y)
    s_champ = np.asarray(s_champ)
    s_chall = np.asarray(s_chall)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))     # same rows for BOTH models
        rc, _ = recall_at_fpr(y[idx], s_champ[idx], target_fpr)
        rh, _ = recall_at_fpr(y[idx], s_chall[idx], target_fpr)
        diffs[i] = rh - rc
    champ_recall, champ_thr = recall_at_fpr(y, s_champ, target_fpr)
    chall_recall, chall_thr = recall_at_fpr(y, s_chall, target_fpr)
    return {
        "target_fpr": target_fpr,
        "champ_recall": champ_recall, "chall_recall": chall_recall,
        "champ_threshold": champ_thr, "chall_threshold": chall_thr,
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def decide(stats, min_lift_pp=3.0):
    """Pre-registered rule: ship iff lift >= min_lift_pp AND CI lower bound > 0."""
    lift_pp = (stats["chall_recall"] - stats["champ_recall"]) * 100
    ci_low_pp = stats["ci_low"] * 100
    material = lift_pp >= min_lift_pp
    significant = ci_low_pp > 0
    if material and significant:
        verdict = "SHIP challenger"
    elif significant and not material:
        verdict = "HOLD — gain is real but below the materiality bar"
    elif material and not significant:
        verdict = "HOLD — gain not statistically robust (CI crosses 0)"
    else:
        verdict = "REJECT — no material, robust gain"
    return {"verdict": verdict, "lift_pp": round(lift_pp, 2),
            "ci_low_pp": round(ci_low_pp, 2), "min_lift_pp": min_lift_pp,
            "material": bool(material), "significant": bool(significant)}


def economic_impact(y, stats, avg_loss_per_default=5000, cost_per_decline=300):
    """Translate the recall lift into money.

    Both models sit at the SAME 5% FPR, so they decline ~the same number of good
    customers -> the extra-false-positive cost is ~zero by construction and the
    net value is simply the extra defaulters caught times the average loss
    averted. avg_loss_per_default / cost_per_decline are ASSUMPTIONS (GMSC carries
    no monetary fields) and are stated as such in the readout.
    """
    n_bad = int(np.asarray(y).sum())
    extra_bad_caught = (stats["chall_recall"] - stats["champ_recall"]) * n_bad
    loss_avoided = extra_bad_caught * avg_loss_per_default
    return {
        "assumption_avg_loss": avg_loss_per_default,
        "assumption_cost_per_decline": cost_per_decline,
        "extra_defaulters_caught": int(round(extra_bad_caught)),
        "loss_avoided": int(round(loss_avoided)),
        "fp_cost": 0,                              # equal FPR -> no extra declines
        "net_impact": int(round(loss_avoided)),
        "note": "FP cost ~0 because both models are compared at an identical 5% FPR.",
    }


if __name__ == "__main__":
    from src.data_generator import TARGET, assign_batches, load_raw
    from src.prep import prep_apply, prep_fit
    from src.train import (score_challenger, score_champion, train_challenger,
                           train_champion)
    from src.woe import fit_woe, select_features
    raw = assign_batches(load_raw())             # clean eval window (no drift)
    params = prep_fit(raw[raw["batch"] <= 3])
    tr, te = prep_apply(raw[raw["batch"] <= 3], params), prep_apply(raw[raw["batch"] >= 4], params)
    feats = sum(select_features(fit_woe(tr, [c for c in tr.columns if c not in (TARGET, "batch")], TARGET)), [])
    champ, chall = train_champion(tr, feats, TARGET), train_challenger(tr, feats, TARGET)
    stats = bootstrap_recall_diff(te[TARGET], score_champion(champ, te), score_challenger(chall, te), n=300)
    print("recall champ/chall:", round(stats["champ_recall"], 3), round(stats["chall_recall"], 3))
    print("lift CI:", round(stats["ci_low"], 4), "..", round(stats["ci_high"], 4))
    print(decide(stats))
    print(economic_impact(te[TARGET], stats))
