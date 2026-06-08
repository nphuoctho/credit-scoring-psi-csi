"""Post-deployment monitoring: PSI (score) + CSI (features).

  PSI = sum_bins (curr% - base%) * ln(curr% / base%)

PSI on the model SCORE is the fire alarm (has the output distribution moved);
CSI is the same formula on each input FEATURE and tells you which room is on fire.
Identical maths, different variable.

The one rule that makes it work: bin edges are frozen on the baseline window and
reused for every later window. Re-binning each window would let every batch
re-balance itself and PSI would read ~0 forever — blind to real drift.

Bands: <0.10 stable | 0.10-0.25 monitor | >0.25 shift (act / retrain).
"""
import numpy as np
import pandas as pd

EPS = 1e-6


def _fit_edges(base, bins=10):
    """Quantile cut-points from the baseline window (interior boundaries only)."""
    q = np.quantile(np.asarray(base, dtype=float), np.linspace(0, 1, bins + 1))
    return np.unique(q[1:-1])          # dedup handles low-cardinality / ties


def psi(base, curr, bins=10, edges=None):
    """Population Stability Index of curr vs base. Returns (value, edges)."""
    base = np.asarray(base, dtype=float)
    curr = np.asarray(curr, dtype=float)
    interior = _fit_edges(base, bins) if edges is None else edges
    n_bins = len(interior) + 1
    bp = np.bincount(np.digitize(base, interior), minlength=n_bins) / len(base)
    cp = np.bincount(np.digitize(curr, interior), minlength=n_bins) / len(curr)
    bp = np.clip(bp, EPS, None)
    cp = np.clip(cp, EPS, None)
    return float(np.sum((cp - bp) * np.log(cp / bp))), interior


def severity(v):
    return "stable" if v < 0.10 else "monitor" if v < 0.25 else "shift"


_ACTION = {"stable": "none", "monitor": "investigate", "shift": "retrain"}


def run_monitoring(raw_df, scores, features, baseline=1, score_col="__score"):
    """Score PSI + per-feature CSI across batches, edges frozen on `baseline`.

    raw_df carries the (drifted) raw feature values + a `batch` column; `scores`
    are the champion PDs aligned row-wise. CSI is computed on raw inputs so a
    covariate shift is visible before downstream transforms can mask it.
    """
    df = raw_df.copy()
    df[score_col] = np.asarray(scores)
    batches = sorted(df["batch"].unique())
    base = df[df["batch"] == baseline]

    score_edges = _fit_edges(base[score_col])
    psi_rows = []
    for b in batches:
        if b == baseline:
            continue
        val, _ = psi(base[score_col], df[df["batch"] == b][score_col], edges=score_edges)
        psi_rows.append({"batch": b, "psi": round(val, 4), "severity": severity(val)})
    psi_score = pd.DataFrame(psi_rows)

    csi = {}
    for f in features:
        edges = _fit_edges(base[f])
        for b in batches:
            if b == baseline:
                continue
            val, _ = psi(base[f], df[df["batch"] == b][f], edges=edges)
            csi.setdefault(f, {})[b] = round(val, 4)
    csi_matrix = pd.DataFrame(csi).T          # rows = feature, cols = batch
    csi_matrix.columns = [f"batch{b}" for b in csi_matrix.columns]
    return psi_score, csi_matrix


def drift_table(csi_matrix):
    """Long-form table of every feature/batch that breached the stable band."""
    rows = []
    for f, series in csi_matrix.iterrows():
        for col, v in series.items():
            sev = severity(v)
            if sev != "stable":
                rows.append({"feature": f, "batch": col, "csi": v,
                             "severity": sev, "action": _ACTION[sev]})
    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)


def verify_drift(csi_matrix, ground_truth):
    """Sanity check: did CSI catch the injected drift and stay quiet on control?

    Pass = injected feature climbs into the >0.25 'shift' band on the
    strongest-drift window AND rises monotonically across the injected windows,
    while the untouched control feature never leaves the <0.10 stable band.
    """
    feat, ctrl = ground_truth["feature"], ground_truth["control"]
    hit_batches = [f"batch{b}" for b in ground_truth["factors"]]
    feat_csi = csi_matrix.loc[feat, hit_batches].to_numpy()
    breached = bool(feat_csi.max() > 0.25)
    progressive = bool(np.all(np.diff(feat_csi) > 0))      # escalating over time
    control_quiet = bool((csi_matrix.loc[ctrl] < 0.10).all())
    return {
        "injected_feature": feat, "injected_batches": list(ground_truth["factors"]),
        "caught_injected_drift": breached, "escalating_trend": progressive,
        "control_feature": ctrl, "control_stayed_stable": control_quiet,
        "passed": breached and progressive and control_quiet,
    }


if __name__ == "__main__":
    from src.data_generator import TARGET, assign_batches_and_drift, load_raw
    from src.prep import prep_apply, prep_fit
    from src.train import score_champion, train_champion
    from src.woe import fit_woe, select_features
    raw, gt = assign_batches_and_drift(load_raw())
    params = prep_fit(raw[raw["batch"] <= 3])
    tr = prep_apply(raw[raw["batch"] <= 3], params)
    prepped_all = prep_apply(raw, params)
    feats = sum(select_features(fit_woe(tr, [c for c in tr.columns if c not in (TARGET, "batch")], TARGET)), [])
    champ = train_champion(tr, feats, TARGET)
    scores = score_champion(champ, prepped_all)
    csi_feats = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome", "age", "NumberOfOpenCreditLinesAndLoans"]
    psi_score, csi = run_monitoring(raw.reset_index(drop=True), scores, csi_feats)
    print("SCORE PSI:\n", psi_score.to_string(index=False))
    print("\nCSI:\n", csi.to_string())
    print("\nVERIFY:", verify_drift(csi, gt))
