"""End-to-end pipeline: data -> prep -> WoE -> models -> decision -> monitoring.

Runs the whole project, writes every chart to outputs/, the fitted models to
models/, and a single results.json the README and the readout PDF read from.
Run: `./.ckenv/bin/python build_pipeline.py`
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict

from src import plots
from src.data_generator import (TARGET, apply_drift, assign_batches, drift_spec,
                                load_raw)
from src.decision import bootstrap_recall_diff, decide, economic_impact
from src.metrics import discrimination_summary, lift_table
from src.prep import feature_columns, prep_apply, prep_fit
from src.psi import drift_table, run_monitoring, verify_drift
from src.train import (champion_coefficients, cross_val_auc, new_challenger,
                       score_challenger, score_champion, train_challenger,
                       train_champion)
from src.woe import apply_woe, fit_woe, iv_table, select_features

ROOT = Path(__file__).parent
OUT, MODELS = ROOT / "outputs", ROOT / "models"
CSI_FEATURES = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio",
                "MonthlyIncome", "age", "NumberOfOpenCreditLinesAndLoans"]


def main():
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    # 1. data: clean batches for modelling, batch 1-3 train / 4-6 test
    raw = assign_batches(load_raw())
    train_raw, test_raw = raw[raw.batch <= 3], raw[raw.batch >= 4]
    params = prep_fit(train_raw)
    train = prep_apply(train_raw, params)
    test = prep_apply(test_raw, params)
    print(f"data: {len(raw)} rows | base rate {raw[TARGET].mean():.4f} "
          f"| train {len(train)} test {len(test)}")

    # 2. WoE / IV + feature screen (retain the flagged-but-legit util after review)
    feats0 = feature_columns(train)
    woe_model = fit_woe(train, feats0, TARGET)
    ivt = iv_table(woe_model)
    keep, flagged = select_features(woe_model)
    feats = keep + flagged                       # util (IV>0.5) kept w/ caveat
    plots.plot_iv(ivt, OUT / "iv_ranking.png")
    print(f"features kept: {len(feats)} | flagged>0.5 (reviewed, retained): {flagged}")

    # 3. train champion (LR/WoE) + challenger (LightGBM/raw)
    champ = train_champion(train, feats, TARGET)
    chall = train_challenger(train, feats, TARGET)
    joblib.dump({"bundle": champ, "prep_params": params, "drift": drift_spec()},
                MODELS / "champion_lr.joblib")
    joblib.dump(chall, MODELS / "challenger_lgbm.joblib")

    s_champ = score_champion(champ, test)
    s_chall = score_challenger(chall, test)
    eval_block = {
        "champion": discrimination_summary(test[TARGET], s_champ),
        "challenger": discrimination_summary(test[TARGET], s_chall),
        "champion_cv_auc": list(np.round(cross_val_auc(train, feats, TARGET, "champion"), 4)),
        "challenger_cv_auc": list(np.round(cross_val_auc(train, feats, TARGET, "challenger"), 4)),
    }
    print("champion ", eval_block["champion"], "\nchallenger", eval_block["challenger"])

    # 4. charts: ROC, lift, calibration (challenger calibrated with OOF isotonic)
    plots.plot_roc(test[TARGET], {"champion (LR+WoE)": s_champ,
                                  "challenger (LightGBM)": s_chall}, OUT / "roc.png")
    plots.plot_lift(lift_table(test[TARGET], s_chall), OUT / "lift.png")
    spw = (len(train) - train[TARGET].sum()) / max(train[TARGET].sum(), 1)
    oof = cross_val_predict(new_challenger(spw), train[feats], train[TARGET],
                            cv=5, method="predict_proba")[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, train[TARGET])
    s_chall_cal = iso.transform(s_chall)
    plots.plot_calibration(test[TARGET], {
        "champion (LR)": s_champ, "challenger raw": s_chall,
        "challenger isotonic": s_chall_cal}, OUT / "calibration.png")

    # 5. champion-challenger decision
    stats = bootstrap_recall_diff(test[TARGET].to_numpy(), s_champ, s_chall, n=1000)
    decision = decide(stats)
    economics = economic_impact(test[TARGET].to_numpy(), stats)
    print(f"decision: {decision['verdict']} | lift {decision['lift_pp']}pp "
          f"CI[{stats['ci_low']*100:.1f},{stats['ci_high']*100:.1f}]pp")

    # 6. monitoring on the DRIFTED simulation (separate from clean eval above)
    drifted_all = apply_drift(raw, drift_spec())
    prepped_drift = prep_apply(drifted_all, params)
    mon_scores = score_champion(champ, prepped_drift)
    psi_score, csi = run_monitoring(drifted_all.reset_index(drop=True), mon_scores, CSI_FEATURES)
    dtable = drift_table(csi)
    verification = verify_drift(csi, drift_spec())
    plots.plot_psi_trend(psi_score, OUT / "psi_trend.png")
    plots.plot_csi_heatmap(csi, OUT / "csi_heatmap.png")
    print(f"monitoring verification passed: {verification['passed']}")

    # 7. persist everything the README + readout read from
    results = {
        "data": {"rows": int(len(raw)), "base_rate": round(float(raw[TARGET].mean()), 4),
                 "train_rows": int(len(train)), "test_rows": int(len(test))},
        "iv_top": ivt.head(8).to_dict(orient="records"),
        "features_used": feats, "flagged_high_iv": flagged,
        "evaluation": eval_block,
        "champion_coefficients": champion_coefficients(champ).round(3).to_dict(orient="records"),
        "decision": {**stats, **decision}, "economics": economics,
        "monitoring": {"score_psi": psi_score.to_dict(orient="records"),
                       "csi": csi.round(4).to_dict(orient="index"),
                       "drift_table": dtable.to_dict(orient="records"),
                       "verification": verification},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT/'results.json'} + 6 charts + 2 models")
    return results


if __name__ == "__main__":
    main()
