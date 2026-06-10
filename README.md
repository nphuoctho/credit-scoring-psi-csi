# Credit Scoring with Champion-Challenger and PSI/CSI Monitoring

A full credit-scorecard lifecycle on a 150,000-loan consumer portfolio: train two models, decide which one goes to production, then watch it for drift the way a risk team would after deployment. The incumbent is a logistic regression on weight-of-evidence features, the kind of transparent scorecard a credit team can defend to a regulator. The challenger is a LightGBM on raw features. At a fixed 5% false-positive rate the challenger catches 5.6 percentage points more defaulters than the champion (51.1% against 45.5% recall, 95% CI 4.5 to 6.8pp from a paired bootstrap), which clears the bar to ship.

The data is synthetic and seeded, so every number can be checked against a known answer, and the monitoring layer is checked the same way: I inject a known distribution shift into one feature over the later time windows and confirm PSI/CSI catch exactly that and nothing else. The one-page decision writeup is in [outputs/readout.pdf](outputs/readout.pdf); the full narrative with every chart and table is in [notebooks/credit-scoring.ipynb](notebooks/credit-scoring.ipynb).

## Models

| | Champion (LR + WoE) | Challenger (LightGBM) |
|---|---|---|
| AUC (held-out window) | 0.807 | 0.841 |
| KS | 0.479 | 0.533 |
| Gini | 0.614 | 0.683 |
| 5-fold CV AUC | 0.808 (sd 0.010) | 0.842 (sd 0.007) |

The challenger ranks better on every metric, but the call is still not automatic: a LightGBM is harder to explain than a WoE scorecard, so it has to earn the swap. The rule was fixed before looking at the result, ship only if the recall lift is at least 3pp and its confidence interval stays above zero, and both hold. The recommendation is to ship the challenger behind the existing scorecard as a shadow test, with the champion as fallback. Because the two models sit at the same 5% FPR they decline the same share of good customers, so the extra false-positive cost is about zero and the net effect is roughly 283 more defaulters caught per window, on the order of $1.4M at an assumed $5,000 loss per default.

## Charts

| Challenger leads on ROC | Top decile holds the risk | LR calibrated, LightGBM needs isotonic |
|---|---|---|
| ![ROC](outputs/roc.png) | ![Lift](outputs/lift.png) | ![Calibration](outputs/calibration.png) |

| Score PSI climbs as inputs drift | CSI pins the drift to DebtRatio |
|---|---|
| ![PSI trend](outputs/psi_trend.png) | ![CSI heatmap](outputs/csi_heatmap.png) |

## How it was done

A few decisions carry the project:

1. The split is by time batch (windows 1 to 3 train, 4 to 6 test), never random, so the model is never trained on the future. Every prep statistic (medians, winsor caps) is learned on train and replayed on test.
2. The data hides the same traps as the real Give Me Some Credit set: a 6.7% default rate, about 20% missing income, utilization outliers in the tens of thousands, and the 96 and 98 sentinel codes in the past-due columns. The sentinels are read as "not available", not as someone who was 96 times late, which would otherwise teach the model a fake risk spike.
3. WoE and IV are computed by hand. Features with IV below 0.02 are dropped and anything above 0.5 is flagged for a leakage check. Revolving utilization tripped that screen, was reviewed as the canonical strongest credit feature, and kept with a note rather than dropped on reflex.
4. The champion is left unweighted on purpose. Class weighting lifted AUC by a hair but pushed the mean predicted probability to 0.37 against a true 0.067, which is useless as a scorecard. Dropping it keeps the probability calibrated, and the imbalance is handled at the operating point instead. The challenger is class-weighted, then isotonic-calibrated.
5. The decision compares recall at a fixed FPR, not at a 0.5 cutoff, and bootstraps the lift with paired resampling so the interval reflects the like-for-like difference between the two models.
6. PSI on the score is the alarm, CSI per feature is the diagnosis, and the bin edges are frozen on the baseline window. Re-binning each window would let every batch rebalance itself and read a flat 0.00 forever.
7. Model quality and drift robustness are measured separately. Evaluation runs on a clean held-out window; the injected drift is applied only to the monitoring simulation, so the two never get conflated.

## Running it

```bash
python3 setup_env.py                          # builds .ckenv with uv, installs deps
./.ckenv/bin/python build_pipeline.py         # data, models, charts, outputs/results.json
./.ckenv/bin/python build_readout_pdf.py      # outputs/readout.pdf
./.ckenv/bin/python -m streamlit run app.py   # optional batch-scoring demo
```

The generator runs without Kaggle credentials. Drop a real `cs-training.csv` into `data/` and the pipeline uses it instead, since the schema matches. The repo holds no credentials.

## Layout

```
src/         data_generator, prep, woe, metrics, train, decision, psi, plots
build_pipeline.py, build_readout_pdf.py
notebooks/   credit-scoring.ipynb (executed, charts embedded)
outputs/     charts, results.json, readout.pdf
models/      champion and challenger, saved with joblib
app.py       Streamlit batch-scoring and drift-check demo
```

## Limitations

The data is synthetic, a deliberate trade: there is no public dataset I can redistribute, and a seeded generator lets me check both the model and the monitor against a known answer. The time windows are simulated because the source has no real timestamp, and the drift is injected rather than observed. The loss-per-default figure is an assumption that changes the dollar total but not the ship decision. Fairness monitoring (PSI by age or other protected traits) is out of scope and is the obvious next step.

## Stack

Python with pandas, scikit-learn, and LightGBM for the models, matplotlib and seaborn for charts, Streamlit for the demo.

## Tóm tắt (VN)

Dự án mô phỏng trọn vòng đời một scorecard tín dụng: huấn luyện hai mô hình, quyết định mô hình nào lên production bằng khung champion-challenger với bootstrap CI tại ngưỡng FPR 5%, rồi giám sát drift sau triển khai bằng PSI/CSI. Các công thức lõi (WoE/IV, KS, PSI/CSI, bootstrap) đều tự code. Challenger bắt thêm 5,6 điểm phần trăm khách vỡ nợ ở cùng mức từ chối nên được đề xuất ship; lớp monitoring bắt đúng drift đã chủ động tiêm vào DebtRatio.
