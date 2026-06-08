"""Streamlit batch-scoring + drift-check demo (optional, nice-to-have).

Analyst-oriented (batch upload -> portfolio analytics), not a single-customer
form. Reuses the same src/ modules as the notebook/pipeline so scoring, prep and
PSI never diverge. Run: `./.ckenv/bin/python -m streamlit run app.py`
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_generator import assign_batches, load_raw
from src.prep import prep_apply
from src.psi import _fit_edges, psi, severity
from src.train import score_champion

st.set_page_config(page_title="Credit Risk — Batch Scoring", layout="wide")


@st.cache_resource
def load_assets():
    """Champion model + a baseline score reference (batch 1 of the clean data)."""
    m = joblib.load("models/champion_lr.joblib")
    bundle, params = m["bundle"], m["prep_params"]
    base = assign_batches(load_raw())
    base = base[base.batch == 1]
    base_scores = score_champion(bundle, prep_apply(base, params))
    return bundle, params, base_scores, _fit_edges(base_scores, 10)


def main():
    bundle, params, base_scores, edges = load_assets()
    st.title("Credit Risk — Batch Scoring & Drift Check")
    st.caption("Upload a loan portfolio (Give Me Some Credit schema) → PD distribution, "
               "risk deciles, and a PSI drift check against the model's baseline.")

    up = st.file_uploader("Loan CSV", type="csv")
    if not up:
        st.info("Upload a CSV, or generate a sample below.")
        if st.button("Use a synthetic sample (5,000 rows)"):
            st.session_state["sample"] = load_raw().sample(5000, random_state=1)
    df = st.session_state.get("sample") if not up else pd.read_csv(up)
    if df is None:
        return

    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    try:
        scores = score_champion(bundle, prep_apply(df, params))
    except Exception as exc:                       # noqa: BLE001 - surface schema issues
        st.error(f"Could not score — check the columns match the GMSC schema. ({exc})")
        return

    psi_val, _ = psi(base_scores, scores, edges=edges)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows scored", f"{len(df):,}")
    c2.metric("Mean predicted PD", f"{scores.mean():.1%}")
    c3.metric("Score PSI vs baseline", f"{psi_val:.3f}", severity(psi_val))
    if psi_val >= 0.25:
        st.warning("PSI ≥ 0.25 — population has shifted materially; investigate / retrain.")
    elif psi_val >= 0.10:
        st.info("PSI in 0.10–0.25 — monitor.")

    left, right = st.columns(2)
    with left:
        st.subheader("Predicted PD distribution")
        hist, edges_h = np.histogram(scores, bins=20, range=(0, scores.max()))
        st.bar_chart(pd.DataFrame({"count": hist},
                                  index=np.round(edges_h[:-1], 3)))
    with right:
        st.subheader("Risk by decile (10 = riskiest)")
        d = pd.DataFrame({"pd": scores})
        d["decile"] = pd.qcut(d["pd"].rank(method="first"), 10, labels=False) + 1
        tbl = d.groupby("decile")["pd"].agg(mean_pd="mean", customers="count")
        st.dataframe(tbl.sort_index(ascending=False).style.format({"mean_pd": "{:.1%}"}))

    st.subheader("Highest-risk customers")
    out = df.copy()
    out["predicted_pd"] = scores
    st.dataframe(out.sort_values("predicted_pd", ascending=False).head(20))


if __name__ == "__main__":
    main()
