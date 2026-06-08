"""Credit-scoring evaluation metrics.

Discrimination (AUC, KS, Gini) + ranking (lift) + an operating-point helper
(recall at a fixed false-positive rate). KS and lift are written out by hand
because the interview question is "what is KS, how does it differ from AUC" — KS
is the single threshold of maximum separation between the good/bad cumulative
distributions, which is exactly how a credit team picks a cut-off; AUC summarises
every threshold at once.

Accuracy is deliberately absent: on a ~6.7% default base rate, "predict everyone
good" scores 93% accuracy and catches zero defaulters. Rank metrics only.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y, score):
    return float(roc_auc_score(y, score))


def gini(y, score):
    return 2 * auc(y, score) - 1


def ks_statistic(y, score):
    """Max gap between the bad and good cumulative distributions over score."""
    y = np.asarray(y)
    score = np.asarray(score)
    order = np.argsort(score)
    ys = y[order]
    n_bad, n_good = ys.sum(), len(ys) - ys.sum()
    cum_bad = np.cumsum(ys) / n_bad
    cum_good = np.cumsum(1 - ys) / n_good
    return float(np.max(np.abs(cum_bad - cum_good)))


def recall_at_fpr(y, score, target_fpr=0.05):
    """Recall (TPR) of defaulters at a fixed FPR operating point + the threshold.

    The risk team reasons as 'I accept wrongly declining 5% of good customers —
    which model then catches more bad ones?'. Comparing recall at a shared FPR is
    apples-to-apples; a raw 0.5 probability cut-off is meaningless once class
    weights have shifted the scores.
    """
    fpr, tpr, thr = roc_curve(y, score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx]), float(thr[idx])


def lift_table(y, score, n=10):
    """Decile lift: bad-rate per score decile divided by the overall base rate."""
    import pandas as pd
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(score)})
    # rank then qcut so ties don't collapse deciles; decile 9 = highest risk
    df["decile"] = pd.qcut(df["s"].rank(method="first"), n, labels=False)
    base = df["y"].mean()
    g = df.groupby("decile")["y"].agg(bad_rate="mean", n="count")
    g["lift"] = g["bad_rate"] / base
    return g.sort_index(ascending=False)


def discrimination_summary(y, score):
    """One-call {auc, ks, gini} for reporting."""
    a = auc(y, score)
    return {"auc": round(a, 4), "ks": round(ks_statistic(y, score), 4),
            "gini": round(2 * a - 1, 4)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.07, 20000)
    # score correlated with y -> sanity: auc>0.5, ks>0
    score = 0.6 * y + rng.normal(0, 1, 20000)
    print("summary:", discrimination_summary(y, score))
    r, t = recall_at_fpr(y, score, 0.05)
    print("recall@5%fpr:", round(r, 3), "| threshold:", round(t, 3))
    print(lift_table(y, score).round(3).to_string())
