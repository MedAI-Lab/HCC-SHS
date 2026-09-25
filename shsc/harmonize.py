from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    AGE_COL,
    BATCH_COL,
    CENTER_BATCH_COL,
    CENTER_COL,
    GENDER_COL,
)


def add_center_batch(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[CENTER_BATCH_COL] = (
        out[CENTER_COL].astype(str) + "_" + out[BATCH_COL].astype(str)
    )
    return out


def _validate(dfs: List[pd.DataFrame], feature_cols: List[str]) -> None:
    required = [CENTER_COL, BATCH_COL, AGE_COL, GENDER_COL]
    for i, df in enumerate(dfs):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"cohort {i}: missing columns {missing}")
        if df[BATCH_COL].isna().any() or df[CENTER_COL].isna().any():
            raise ValueError(f"cohort {i}: center/batch contain missing values")
        missing_feats = [c for c in feature_cols if c not in df.columns]
        if missing_feats:
            raise ValueError(f"cohort {i}: missing feature columns {missing_feats[:5]} ...")


def _covariate_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    age = pd.to_numeric(df[AGE_COL], errors="coerce")
    gender = pd.to_numeric(df[GENDER_COL], errors="coerce")
    age = age.fillna(age.median())
    gender = gender.fillna(gender.median())
    return gender.to_numpy(dtype=np.float64), age.to_numpy(dtype=np.float64)



def _ols_beta(design: np.ndarray, X: np.ndarray) -> np.ndarray:
    gram = design.T @ design
    return np.linalg.solve(gram, design.T @ X)


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _aprior(delta_hat: np.ndarray) -> float:
    m = delta_hat.mean()
    s2 = delta_hat.var(ddof=1)
    return (2.0 * s2 + m ** 2) / s2


def _bprior(delta_hat: np.ndarray) -> float:
    m = delta_hat.mean()
    s2 = delta_hat.var(ddof=1)
    return (m * s2 + m ** 3) / s2


def _it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4):
    n = (~np.isnan(sdat)).sum(axis=0).astype(np.float64)
    g_old = g_hat.astype(np.float64)
    d_old = d_hat.astype(np.float64)
    change = np.inf
    while change > conv:
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        resid = sdat - g_new[None, :]
        sum2 = (resid ** 2).sum(axis=0)
        d_new = _postvar(sum2, n, a, b)
        change = max(
            np.nanmax(np.abs(g_new - g_old) / g_old),
            np.nanmax(np.abs(d_new - d_old) / d_old),
        )
        g_old, d_old = g_new, d_new
    return g_new, d_new


def _fit_combat(
    X: np.ndarray, batch_codes: np.ndarray, gender: np.ndarray, age: np.ndarray
) -> Dict:
    n_samples, n_features = X.shape
    batch_levels = list(pd.unique(batch_codes))
    batch_index = {b: i for i, b in enumerate(batch_levels)}
    n_batch = len(batch_levels)
    gender_levels = list(np.unique(gender))

    def batch_onehot(codes):
        out = np.zeros((len(codes), n_batch))
        for i, c in enumerate(codes):
            out[i, batch_index[c]] = 1.0
        return out

    def gender_onehot(g):
        g = np.asarray(g)
        out = np.zeros((len(g), len(gender_levels)))
        for i, lev in enumerate(gender_levels):
            out[g == lev, i] = 1.0
        return out[:, 1:]

    design = np.hstack(
        [batch_onehot(batch_codes), gender_onehot(gender), age.reshape(-1, 1)]
    )
    B_hat = _ols_beta(design, X)

    sample_per_batch = np.array([int((batch_codes == b).sum()) for b in batch_levels])
    grand_mean = (sample_per_batch / n_samples) @ B_hat[:n_batch, :]

    stand_mean = np.tile(grand_mean, (n_samples, 1))
    mod_mean = design[:, n_batch:] @ B_hat[n_batch:, :]

    resid = X - design @ B_hat
    var_pooled = (resid ** 2).mean(axis=0)
    nonzero = var_pooled[var_pooled > 0]
    var_pooled[var_pooled == 0] = np.median(nonzero) if nonzero.size else 1.0

    s_data = (X - stand_mean - mod_mean) / np.sqrt(var_pooled)[None, :]

    gamma_hat = np.zeros((n_batch, n_features))
    delta_hat = np.zeros((n_batch, n_features))
    for i, b in enumerate(batch_levels):
        s_b = s_data[batch_codes == b]
        gamma_hat[i] = s_b.mean(axis=0)
        delta_hat[i] = s_b.var(axis=0, ddof=1)
    delta_hat[delta_hat == 0] = 1.0

    gamma_bar = gamma_hat.mean(axis=0)
    t2 = gamma_hat.var(axis=0, ddof=1)

    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)
    for i, b in enumerate(batch_levels):
        a = _aprior(delta_hat[i])
        b_prior = _bprior(delta_hat[i])
        gamma_star[i], delta_star[i] = _it_sol(
            s_data[batch_codes == b], gamma_hat[i], delta_hat[i], gamma_bar, t2, a, b_prior
        )

    return {
        "batch_index": batch_index,
        "gender_levels": gender_levels,
        "n_batch": n_batch,
        "B_cov": B_hat[n_batch:, :],
        "grand_mean": grand_mean,
        "var_pooled": var_pooled,
        "gamma_star": gamma_star,
        "delta_star": delta_star,
        "gamma_bar": gamma_bar,
        "t2": t2,
    }


def _transform_combat(
    X: np.ndarray,
    batch_codes: np.ndarray,
    gender: np.ndarray,
    age: np.ndarray,
    fit: Dict,
) -> np.ndarray:
    n_samples, _ = X.shape
    gender_levels = fit["gender_levels"]

    def gender_onehot(g):
        g = np.asarray(g)
        out = np.zeros((len(g), len(gender_levels)))
        for i, lev in enumerate(gender_levels):
            out[g == lev, i] = 1.0
        return out[:, 1:]

    cov = np.hstack([gender_onehot(gender), age.reshape(-1, 1)])
    mod_mean = cov @ fit["B_cov"]
    stand_mean = np.tile(fit["grand_mean"], (n_samples, 1))
    var_pooled = fit["var_pooled"]

    s_data = (X - stand_mean - mod_mean) / np.sqrt(var_pooled)[None, :]

    bayesdata = np.empty_like(X)
    for code in pd.unique(batch_codes):
        idx = np.where(batch_codes == code)[0]
        s_b = s_data[idx]
        if code in fit["batch_index"]:
            g_star = fit["gamma_star"][fit["batch_index"][code]]
            d_star = fit["delta_star"][fit["batch_index"][code]]
        else:
            g_hat = s_b.mean(axis=0)
            d_hat = s_b.var(axis=0, ddof=1)
            d_hat[d_hat == 0] = 1.0
            a = _aprior(d_hat)
            b_prior = _bprior(d_hat)
            g_star, d_star = _it_sol(s_b, g_hat, d_hat, fit["gamma_bar"], fit["t2"], a, b_prior)
        bayesdata[idx] = (s_b - g_star[None, :]) / np.sqrt(d_star)[None, :]

    return bayesdata * np.sqrt(var_pooled)[None, :] + stand_mean + mod_mean


def harmonize(
    train: pd.DataFrame,
    internal: pd.DataFrame,
    external: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate([train, internal, external], feature_cols)

    train = add_center_batch(train)
    internal = add_center_batch(internal)
    external = add_center_batch(external)

    X_train = train[feature_cols].to_numpy(dtype=np.float64)
    X_internal = internal[feature_cols].to_numpy(dtype=np.float64)
    X_external = external[feature_cols].to_numpy(dtype=np.float64)
    for X in (X_train, X_internal, X_external):
        if not np.isfinite(X).all():
            raise ValueError("feature matrix contains NaN/infinite values")

    g_train, a_train = _covariate_values(train)
    g_internal, a_internal = _covariate_values(internal)
    g_external, a_external = _covariate_values(external)

    batch_train = train[CENTER_BATCH_COL].astype(str).to_numpy()
    batch_internal = internal[CENTER_BATCH_COL].astype(str).to_numpy()
    batch_external = external[CENTER_BATCH_COL].astype(str).to_numpy()

    fit = _fit_combat(X_train, batch_train, g_train, a_train)

    out_train = train.copy()
    out_train[feature_cols] = _transform_combat(X_train, batch_train, g_train, a_train, fit)
    out_internal = internal.copy()
    out_internal[feature_cols] = _transform_combat(
        X_internal, batch_internal, g_internal, a_internal, fit
    )
    out_external = external.copy()
    out_external[feature_cols] = _transform_combat(
        X_external, batch_external, g_external, a_external, fit
    )

    return out_train, out_internal, out_external
