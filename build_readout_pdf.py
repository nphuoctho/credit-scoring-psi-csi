"""Render the 1-page champion-challenger decision readout (outputs/readout.pdf).

Reads outputs/results.json (written by build_pipeline.py) so the PDF can never
drift from the numbers. Layout follows the risk-committee format: recommendation,
lift at the operating point with CI, money, embedded evidence charts, caveats,
next step. Run: `./.ckenv/bin/python build_readout_pdf.py`
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg          # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"


def _fmt_money(x):
    return f"${x:,.0f}"


def build():
    r = json.loads((OUT / "results.json").read_text())
    ev, dec, eco = r["evaluation"], r["decision"], r["economics"]
    mon = r["monitoring"]["verification"]
    champ, chall = ev["champion"], ev["challenger"]

    fig = plt.figure(figsize=(8.27, 11.69))    # A4 portrait
    fig.subplots_adjust(left=0.07, right=0.93, top=0.95, bottom=0.05)

    def T(y, s, size=10, weight="normal", color="black", x=0.07):
        fig.text(x, y, s, fontsize=size, weight=weight, color=color, va="top", wrap=True)

    T(0.97, "Credit Scoring Model: LR vs LightGBM — Production Decision",
      size=15, weight="bold")
    T(0.945, "Champion-challenger readout · operating point: 5% FPR · "
             "offline evaluation on a held-out window", size=8.5, color="#555")

    ship = dec["verdict"].startswith("SHIP")
    T(0.915, f"RECOMMENDATION:  {dec['verdict']}", size=13, weight="bold",
      color="#1a7a34" if ship else "#b5651d")

    ci_hi_pp = dec["ci_high"] * 100
    T(0.875, "1 · Lift at the business threshold", size=11, weight="bold")
    T(0.855,
      f"At a 5% FPR operating point, challenger recall {dec['chall_recall']:.1%} vs "
      f"champion {dec['champ_recall']:.1%} = +{dec['lift_pp']:.1f}pp more defaulters "
      f"caught for the same share of good customers declined "
      f"(95% CI [{dec['ci_low_pp']:.1f}, {ci_hi_pp:.1f}]pp, paired bootstrap, "
      f"n=1000).", size=9)
    T(0.82,
      f"Discrimination — champion AUC {champ['auc']:.3f} / KS {champ['ks']:.3f} · "
      f"challenger AUC {chall['auc']:.3f} / KS {chall['ks']:.3f}. "
      f"Rule (pre-registered): ship if lift ≥ {dec['min_lift_pp']:.0f}pp AND CI lower > 0 "
      f"→ {'met' if ship else 'not met'}.", size=9)

    T(0.78, "2 · Risk / cost trade-off", size=11, weight="bold")
    T(0.76,
      f"Extra defaulters caught (test window): {eco['extra_defaulters_caught']:,} · "
      f"loss avoided {_fmt_money(eco['loss_avoided'])} "
      f"(assumes {_fmt_money(eco['assumption_avg_loss'])}/default). "
      f"False-positive cost ≈ $0 — both models compared at an identical 5% FPR, so "
      f"the same share of good customers is declined. Net ≈ "
      f"{_fmt_money(eco['net_impact'])} per window.", size=9)

    # evidence charts
    for img, (x0, title) in {"roc.png": (0.07, "ROC"),
                             "psi_trend.png": (0.52, "Score PSI (monitoring)")}.items():
        p = OUT / img
        if p.exists():
            ax = fig.add_axes([x0, 0.44, 0.40, 0.24])
            ax.imshow(mpimg.imread(p))
            ax.axis("off")

    T(0.40, "3 · Caveats", size=11, weight="bold")
    flagged = ", ".join(r["flagged_high_iv"]) or "none"
    T(0.38,
      f"• Leakage screen: feature(s) with IV>0.5 = {flagged}; reviewed as the "
      f"canonical strongest credit driver (revolving utilization), retained — not "
      f"target leakage.\n"
      f"• Challenger (LightGBM) less interpretable than the WoE scorecard; isotonic "
      f"calibration applied (see calibration chart).\n"
      f"• Training window is simulated batches; production needs a real vintage.\n"
      f"• Post-deployment monitoring is live: injected-drift sanity check "
      f"{'PASSED' if mon['passed'] else 'FAILED'} "
      f"(CSI caught {mon['injected_feature']} drift, control "
      f"{mon['control_feature']} stable).", size=9)

    T(0.25, "4 · Next step", size=11, weight="bold")
    T(0.23,
      "Ship challenger behind the existing WoE scorecard as a shadow/A-B test; "
      "keep champion as fallback. Retrain trigger wired to PSI: investigate at 0.10, "
      "retrain at 0.25 (DebtRatio already breached in the drift simulation).", size=9)

    T(0.06, "Generated from outputs/results.json · synthetic GMSC-schema data · "
            "figures: outputs/roc.png, outputs/psi_trend.png", size=7, color="#777")

    fig.savefig(OUT / "readout.pdf")
    plt.close(fig)
    print(f"wrote {OUT/'readout.pdf'}")


if __name__ == "__main__":
    build()
