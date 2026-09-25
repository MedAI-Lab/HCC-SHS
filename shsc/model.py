from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from .config import N_STARTUP_TRIALS, N_TRIALS, SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cindex(y: np.ndarray, risk: np.ndarray) -> float:
    return concordance_index_censored(y["event"], y["time"], risk)[0]


def _rsf_params(trial: optuna.Trial) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 15),
        "max_features": trial.suggest_float("max_features", 0.3, 0.8),
        "max_samples": trial.suggest_float("max_samples", 0.7, 0.95),
        "random_state": SEED,
        "n_jobs": -1,
    }


def tune_rsf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = N_TRIALS,
) -> Dict:


    def objective(trial: optuna.Trial) -> float:
        params = _rsf_params(trial)
        rsf = RandomSurvivalForest(**params)
        rsf.fit(X_train, y_train)
        val_c = _cindex(y_val, rsf.predict(X_val))
        return val_c

    sampler = TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params)


def fit_final_model(
    X: np.ndarray, y: np.ndarray, params: Dict
) -> RandomSurvivalForest:
    params = {**params, "random_state": SEED, "n_jobs": -1}
    model = RandomSurvivalForest(**params)
    model.fit(X, y)
    return model
