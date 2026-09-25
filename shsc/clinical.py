from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import RobustScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from .config import CLINICAL_CANDIDATES, SEED


def _cindex(y: np.ndarray, risk: np.ndarray) -> float:
    return concordance_index_censored(y["event"], y["time"], risk)[0]


def _fit_rsf(X: np.ndarray, y: np.ndarray, params: Dict) -> RandomSurvivalForest:
    params = {**params, "random_state": SEED, "n_jobs": -1}
    rsf = RandomSurvivalForest(**params)
    rsf.fit(X, y)
    return rsf


def _preprocess_clinical(
    df: pd.DataFrame, candidates: List[str]
) -> Tuple[np.ndarray, List[str]]:
    categorical = [c for c in candidates if df[c].nunique() <= 5]
    continuous = [c for c in candidates if c not in categorical]
    out = df.copy()
    if continuous:
        out[continuous] = RobustScaler(quantile_range=(15, 85)).fit_transform(
            out[continuous].to_numpy(dtype=np.float64)
        )
    for c in candidates:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.fillna(out.median())
    return out[candidates].to_numpy(dtype=np.float64), candidates


def select_clinical_features(
    X_shs: np.ndarray,
    y: np.ndarray,
    clinical_df: pd.DataFrame,
    shs_params: Dict,
    candidates: List[str] = CLINICAL_CANDIDATES,
    min_improvement: float = 0.0,
    p_threshold: float = 0.05,
) -> List[str]:
    
    X_clin, clin_names = _preprocess_clinical(clinical_df, candidates)

    
    baseline = _fit_rsf(X_shs, y, shs_params)
    baseline_c = _cindex(y, baseline.predict(X_shs))

    improvements = []
    for i, name in enumerate(clin_names):
        X_comb = np.hstack([X_shs, X_clin[:, i : i + 1]])
        rsf = _fit_rsf(X_comb, y, shs_params)
        imp = _cindex(y, rsf.predict(X_comb)) - baseline_c
        improvements.append((name, imp))
    improvements.sort(key=lambda x: x[1], reverse=True)
    stepwise_kept = [name for name, imp in improvements if imp > min_improvement]

    if not stepwise_kept:
        return []


    cox_df = pd.DataFrame(X_clin, columns=clin_names)
    cox_df = cox_df[stepwise_kept].copy()
    cox_df["time"] = y["time"]
    cox_df["event"] = y["event"].astype(int)

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_df, duration_col="time", event_col="event")
        p_values = [
            (name, cph.summary.loc[name, "p"]) for name in stepwise_kept if name in cph.summary.index
        ]
        p_values.sort(key=lambda x: x[1])
        significant = [name for name, p in p_values if p < p_threshold]
    except Exception:
        significant = []

    return significant
