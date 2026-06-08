<!-- SCAFFOLD — điền số thật ở Phase 8. Chỗ [TODO] là phần bạn viết. -->

# Credit Scoring + PSI/CSI Monitoring + Champion-Challenger

> [TODO 1-câu pitch] · 📓 [Notebook (nbviewer)](#) · 📄 [Decision readout (PDF)](outputs/readout.pdf)

Mô phỏng vòng đời thật của 1 credit scoring model: train 2 model → quyết định champion-challenger → giám sát drift sau triển khai (PSI/CSI) trên dữ liệu Give Me Some Credit (~150k khoản vay).

## TL;DR (kết quả chính)

- **Champion (LR + WoE):** AUC [TODO] · KS [TODO]
- **Challenger (LightGBM):** AUC [TODO] · KS [TODO]
- **Decision @ 5% FPR:** [ship / hold] — challenger +[TODO]pp recall (95% CI [TODO])
- **Monitoring:** PSI/CSI **bắt được injected drift** ở batch 4–6 ✓ (control giữ < 0.1)

## Charts chính

<!-- embed ở Phase 8: outputs/*.png -->
| ROC | Lift | Calibration | PSI trend | CSI heatmap |
|---|---|---|---|---|
| [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## Method (tóm tắt)

1. **Data prep** — missing/outlier/sentinel handling, split theo time-batch (point-in-time).
2. **WoE/IV** — transform + lọc feature theo IV (flag IV>0.5 nghi leakage).
3. **Models** — LR (champion) trên WoE · LightGBM (challenger) trên raw · 5-fold CV.
4. **Evaluation** — AUC/KS/Gini/calibration/lift trên test.
5. **Decision** — recall @ 5% FPR + bootstrap CI + decision rule pre-registered.
6. **Monitoring** — PSI (score) + CSI (feature) qua 6 batch; verify bắt drift inject.

## Reproduce

```bash
pip install -r requirements.txt
# tải data theo data/README.md → data/cs-training.csv
jupyter notebook notebooks/credit-scoring.ipynb
```

## Limitations & Future work

- Time-window là **giả lập** (dataset không có timestamp); inject drift để test monitoring.
- Fairness/bias (PSI by gender/age): **out of scope** — future work.
- Production sẽ fit WoE bins trong từng CV fold (ở đây fit 1 lần trên train + caveat).

---
*[TODO] Tóm tắt tiếng Việt 2–3 câu cho người đọc VN.*
