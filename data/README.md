# Data — Give Me Some Credit

> ⚠️ CSV **không commit** (Kaggle ToS). Tải về local theo hướng dẫn dưới.

## Download
- Kaggle: <https://www.kaggle.com/c/GiveMeSomeCredit/data>
- File cần: **`cs-training.csv`** (~150k rows). Để vào thư mục này: `data/cs-training.csv`.
- (Tùy chọn) CLI: `kaggle competitions download -c GiveMeSomeCredit` → giải nén lấy `cs-training.csv`.

## Schema (11 cột)
| Cột | Ý nghĩa |
|---|---|
| `SeriousDlqin2yrs` | **TARGET** — 1 = 90+ DPD trong 2 năm |
| `RevolvingUtilizationOfUnsecuredLines` | dư nợ thẻ/tín chấp ÷ hạn mức |
| `age` | tuổi |
| `NumberOfTime30-59DaysPastDueNotWorse` | số lần trễ 30–59 ngày |
| `DebtRatio` | nợ ÷ thu nhập |
| `MonthlyIncome` | thu nhập tháng (**có missing**) |
| `NumberOfOpenCreditLinesAndLoans` | số khoản tín dụng mở |
| `NumberOfTimes90DaysLate` | số lần trễ 90+ ngày |
| `NumberRealEstateLoansOrLines` | số khoản vay BĐS |
| `NumberOfTime60-89DaysPastDueNotWorse` | số lần trễ 60–89 ngày |
| `NumberOfDependents` | số người phụ thuộc (**có missing**) |

> 🔎 Phase 1 EDA: tự đi tìm quirks (imbalance ~6.7%, missing ở Income/Dependents, outlier Revolving/DebtRatio, **sentinel 96/98** trong 3 cột PastDue). Đừng nhận `describe()` ở face value.
