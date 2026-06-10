# Data: Give Me Some Credit

The CSV is not committed (Kaggle terms of service). Download it here before running the pipeline. If it is absent, the generator falls back to synthetic data with the same schema.

## Download

- Kaggle: https://www.kaggle.com/c/GiveMeSomeCredit/data
- Put `cs-training.csv` (about 150,000 rows) in this folder as `data/cs-training.csv`.
- CLI option: `kaggle competitions download -c GiveMeSomeCredit`, then unzip `cs-training.csv`.

## Schema (11 columns)

| Column | Meaning |
|---|---|
| `SeriousDlqin2yrs` | target, 1 = 90+ days past due within 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | revolving balance over credit limit |
| `age` | age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | count of 30 to 59 day delinquencies |
| `DebtRatio` | debt over income |
| `MonthlyIncome` | monthly income (has missing values) |
| `NumberOfOpenCreditLinesAndLoans` | open credit lines and loans |
| `NumberOfTimes90DaysLate` | count of 90+ day delinquencies |
| `NumberRealEstateLoansOrLines` | real estate loans or lines |
| `NumberOfTime60-89DaysPastDueNotWorse` | count of 60 to 89 day delinquencies |
| `NumberOfDependents` | number of dependents (has missing values) |

The set carries the quirks the pipeline handles: about 6.7% default rate, missing income and dependents, revolving and debt-ratio outliers, and the 96 and 98 sentinel codes in the past-due columns.
