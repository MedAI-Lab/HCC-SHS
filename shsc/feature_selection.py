from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from .config import (
    CENTER_COL,
    L1_COEFF_EPS,
    L1_COX_L1_RATIO,
    L1_COX_PENALIZER,
    RFE_CV_SPLITS,
    RFE_MIN_FEATURES,
    KS_THRESHOLD,
    SEED,
    WASSERSTEIN_THRESHOLD,
)


def remove_zero_variance(
    train: pd.DataFrame, feature_cols: List[str], eps: float = 1e-12
) -> List[str]:
    
    kept = []
    for col in feature_cols:
        values = pd.to_numeric(train[col], errors="coerce")
        var = values.var(ddof=1)
        if np.isfinite(var) and var > eps:
            kept.append(col)
    return kept


def _to_survival(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[("event", "?"), ("time", "float64")],
    )


def l1_cox_select(
    X: np.ndarray, y: np.ndarray, feature_names: List[str]
) -> List[str]:
    cox_df = pd.DataFrame(X, columns=feature_names)
    cox_df["time"] = y["time"]
    cox_df["event"] = y["event"].astype(int)
    cox_df = cox_df.dropna()

    cph = CoxPHFitter(penalizer=L1_COX_PENALIZER, l1_ratio=L1_COX_L1_RATIO)
    cph.fit(cox_df, duration_col="time", event_col="event")

    coefs = pd.Series(cph.params_, index=feature_names)
    selected = [f for f in feature_names if abs(coefs[f]) > L1_COEFF_EPS]
    return selected


def _permutation_importance(
    model: RandomSurvivalForest,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 5,
) -> np.ndarray:
    
    event, time = y["event"], y["time"]
    baseline = concordance_index_censored(event, time, model.predict(X))[0]

    rng = np.random.default_rng(SEED)
    importances = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            col = X_perm[:, j].copy()
            rng.shuffle(col)
            X_perm[:, j] = col
            c = concordance_index_censored(event, time, model.predict(X_perm))[0]
            drops.append(baseline - c)
        importances[j] = float(np.mean(drops))
    return importances


def _rfe_model() -> RandomSurvivalForest:
    return RandomSurvivalForest(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=10,
        random_state=SEED,
        n_jobs=-1,
    )


def _cv_cindex(X: np.ndarray, y: np.ndarray, cv_splits: int) -> float:
    
    event = y["event"].astype(bool)
    n_event = int(event.sum())
    n_cens = int((~event).sum())

    
    if n_event < cv_splits or n_cens < cv_splits:
        rsf = _rfe_model()
        rsf.fit(X, y)
        score = concordance_index_censored(y["event"], y["time"], rsf.predict(X))[0]
        return float(score) if np.isfinite(score) else 0.5

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, val_idx in skf.split(X, event.astype(int)):
        rsf = _rfe_model()
        rsf.fit(X[train_idx], y[train_idx])
        score = concordance_index_censored(
            y["event"][val_idx], y["time"][val_idx], rsf.predict(X[val_idx])
        )[0]
        if np.isfinite(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.5


def rfe_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_splits: int = RFE_CV_SPLITS,
    min_features: int = RFE_MIN_FEATURES,
) -> List[str]:
    if X.shape[1] <= min_features:
        return list(feature_names)

    remaining = list(range(X.shape[1]))
    history: List[Tuple[List[int], float]] = []

    while len(remaining) >= min_features:
        history.append((list(remaining), _cv_cindex(X[:, remaining], y, cv_splits)))
        if len(remaining) == min_features:
            break
        rsf = _rfe_model()
        rsf.fit(X[:, remaining], y)
        importances = _permutation_importance(rsf, X[:, remaining], y)
        remaining.remove(remaining[int(np.argmin(importances))])

    best_subset, _ = max(history, key=lambda item: (item[1], -len(item[0])))
    return [feature_names[i] for i in best_subset]


def _normalised_wasserstein(a: np.ndarray, b: np.ndarray, ref_std: float) -> float:
    return wasserstein_distance(a, b) / (ref_std + 1e-12)


def stable_features(
    train: pd.DataFrame,
    feature_names: List[str],
    ks_threshold: float = KS_THRESHOLD,
    wasserstein_threshold: float = WASSERSTEIN_THRESHOLD,
) -> List[str]:
   
    centers = sorted(train[CENTER_COL].unique())
    if len(centers) < 2:
        return list(feature_names)

    pairs = [(centers[i], centers[j]) for i in range(len(centers)) for j in range(i + 1, len(centers))]
    kept = []
    for feat in feature_names:
        values_by_center = {
            c: pd.to_numeric(train.loc[train[CENTER_COL] == c, feat], errors="coerce").dropna().to_numpy()
            for c in centers
        }
        if any(len(v) < 10 for v in values_by_center.values()):
            continue
        ref_std = np.std(np.concatenate(list(values_by_center.values())), ddof=1)
        stable = True
        for c1, c2 in pairs:
            a, b = values_by_center[c1], values_by_center[c2]
            ks_stat, _ = ks_2samp(a, b)
            wass_norm = _normalised_wasserstein(a, b, ref_std)
            if ks_stat > ks_threshold or wass_norm > wasserstein_threshold:
                stable = False
                break
        if stable:
            kept.append(feat)
    return kept


def select_features(
    train: pd.DataFrame,
    feature_cols: List[str],
    time: np.ndarray,
    event: np.ndarray,
) -> List[str]:
    """Run the full selection pipeline and return the retained feature names."""
    y = _to_survival(time, event)

    # Step 1: zero-variance filtering.
    candidates = remove_zero_variance(train, feature_cols)

    # Standardise for regularised regression.
    from sklearn.preprocessing import StandardScaler

    X = train[candidates].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).to_numpy(dtype=np.float64)
    X_scaled = StandardScaler().fit_transform(X)

    # Step 2: L1-regularised Cox.
    l1_selected = l1_cox_select(X_scaled, y, candidates)

    # Step 3: recursive feature elimination.
    keep_idx = [candidates.index(f) for f in l1_selected]
    rfe_selected = rfe_select(X_scaled[:, keep_idx], y, l1_selected)

    # Step 4: distributional stability across training centres.
    final = stable_features(train, rfe_selected)

    return final
