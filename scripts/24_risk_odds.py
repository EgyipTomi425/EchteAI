"""Effect size of local deviation on vanishing detections (CPU only).

Logistic model per detector on all FP32 detections above the operating threshold:

    logit P(vanish) = b0 + b1 * log2(local MRE_proj) [+ b2 * margin + b3 * log area]

exp(b1) is the odds ratio per doubling of the local deviation, unadjusted and adjusted for the score
margin and the box size. 95% CIs from 1 000 nonparametric bootstrap resamples of the detections.
Predictors are standardised for the fit and the coefficient is transformed back.
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from pepai.config import load_config, results_dir

DETECTORS = ["frcnn_r50_fpn", "yolov10s", "yolov10x"]
N_BOOT = 1000


def log2_odds_ratio(X, y):
    """exp(coefficient of the first column) of an unpenalised logistic regression."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sd = X.mean(0), X.std(0)
        clf = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-10).fit((X - mu) / sd, y)
    return float(np.exp(clf.coef_[0][0] / sd[0]))


if __name__ == "__main__":
    cfg = load_config()
    rows = []
    for name in DETECTORS:
        path = results_dir(cfg, "risk") / f"{name}_int8fp32_detections.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        y = d.flip_tol.values
        dev = np.log2(d.local_mre.values + 1e-6)
        designs = {"unadjusted": dev[:, None],
                   "adjusted": np.c_[dev, -(d.score - 0.5).abs().values, d.log_area.values]}
        rng = np.random.default_rng(cfg["seed"])
        idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
        for label, X in designs.items():
            est = log2_odds_ratio(X, y)
            boot = np.array(Parallel(n_jobs=64)(delayed(log2_odds_ratio)(X[i], y[i]) for i in idx))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({"model": name, "model_type": label, "odds_ratio_per_doubling": est, "ci_lo": lo,
                         "ci_hi": hi, "boot_median": float(np.median(boot)), "n": len(y), "events": int(y.sum())})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "risk_odds.csv", index=False)
