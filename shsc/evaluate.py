"""Model evaluation: Harrell's C-index and stratified bootstrap CIs."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sksurv.metrics import concordance_index_censored

from .config import ALPHA, N_BOOT, SEED


def cindex(y: np.ndarray, risk: np.ndarray) -> float:
    """Point estimate of Harrell's concordance index."""
    return concordance_index_censored(y["event"], y["time"], risk)[0]


def bootstrap_cindex(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
) -> Tuple[float, float, float]:
    
    rng = np.random.default_rng(SEED)
    event_idx = np.where(y["event"])[0]
    cens_idx = np.where(~y["event"])[0]

    values = []
    for _ in range(n_boot):
        boot_event = rng.choice(event_idx, size=len(event_idx), replace=True)
        boot_cens = rng.choice(cens_idx, size=len(cens_idx), replace=True)
        boot_idx = np.concatenate([boot_event, boot_cens])
        y_boot = y[boot_idx]
        if y_boot["event"].sum() < 2 or (~y_boot["event"]).sum() < 1:
            continue
        risk = model.predict(X[boot_idx])
        values.append(cindex(y_boot, risk))

    values = np.asarray(values)
    mean = float(np.mean(values))
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return mean, lower, upper
