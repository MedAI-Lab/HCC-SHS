from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    AGE_COL,
    BATCH_COL,
    CENTER_COL,
    EVENT_COL,
    GENDER_COL,
    ID_COL,
    Paths,
    SLIDE_COL,
    TIME_COL,
)
from .evaluate import bootstrap_cindex, cindex
from .feature_selection import remove_zero_variance, select_features
from .harmonize import harmonize
from .clinical import select_clinical_features
from .model import fit_final_model, tune_rsf

# Metadata columns excluded from the spatial feature matrix.
SPATIAL_META = {
    ID_COL,
    CENTER_COL,
    BATCH_COL,
    EVENT_COL,
    TIME_COL,
    AGE_COL,
    GENDER_COL,
    SLIDE_COL,
    "center_batch",
}


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in SPATIAL_META]


def _survival(df: pd.DataFrame) -> np.ndarray:
    return np.array(
        [(bool(e), float(t)) for e, t in zip(df[EVENT_COL], df[TIME_COL])],
        dtype=[("event", "?"), ("time", "float64")],
    )


def _align_clinical(
    clinical: pd.DataFrame, filenames: pd.Series, candidates: List[str]
) -> pd.DataFrame:
    """Return clinical variables for ``filenames``, preserving row order."""
    key = clinical[SLIDE_COL].astype(str)
    clinical = clinical.copy()
    clinical["_key"] = key
    clinical = clinical.drop_duplicates(subset="_key", keep="first")
    lookup = clinical.set_index("_key")[candidates]
    aligned = lookup.reindex(filenames.astype(str))
    return aligned.reset_index(drop=True)


def _evaluate(
    model, X_train, y_train, X_val, y_val, X_test, y_test
) -> Dict[str, Dict[str, float]]:
    results = {}
    for name, X, y in [
        ("train", X_train, y_train),
        ("internal_val", X_val, y_val),
        ("external_test", X_test, y_test),
    ]:
        risk = model.predict(X)
        point = cindex(y, risk)
        _, lo, hi = bootstrap_cindex(X, y, model)
        results[name] = {"c_index": point, "ci_lower": lo, "ci_upper": hi}
    return results


def run(paths: Paths) -> Dict:
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(paths.train_csv)
    internal = pd.read_csv(paths.internal_csv)
    external = pd.read_csv(paths.external_csv)
    clinical = pd.read_csv(paths.clinical_csv)
    for df in (train, internal, external):
        df[ID_COL] = df[ID_COL].astype(str)

    # 1. Zero-variance filtering (training only).
    feature_cols = _feature_columns(train)
    candidates = remove_zero_variance(train, feature_cols)

    # 2. ComBat harmonisation.
    train_c, internal_c, external_c = harmonize(train, internal, external, candidates)

    # 3. Three-step spatial feature selection.
    y_train = _survival(train_c)
    y_val = _survival(internal_c)
    y_test = _survival(external_c)
    shs_features = select_features(
        train_c, candidates, train_c[TIME_COL].to_numpy(), train_c[EVENT_COL].to_numpy()
    )

    # 4. Build the SHS feature matrix and tune it first: its optimal
    # hyperparameters serve as the baseline for clinical-feature selection.
    def _matrix(df, cols):
        return df[cols].to_numpy(dtype=np.float64)

    X_shs = {
        "train": _matrix(train_c, shs_features),
        "val": _matrix(internal_c, shs_features),
        "test": _matrix(external_c, shs_features),
    }

    shs_params = tune_rsf(X_shs["train"], y_train, X_shs["val"], y_val)

    # 5. Clinical feature selection against the tuned SHS baseline (training
    # data only; the external test set is never used for selection).
    from .config import CLINICAL_CANDIDATES

    train_clinical = _align_clinical(clinical, train_c[ID_COL], CLINICAL_CANDIDATES)
    clinical_features = select_clinical_features(
        X_shs["train"], y_train, train_clinical, shs_params
    )

    # 6. Build and evaluate the models. If no clinical predictors are selected,
    # only the SHS model is built (the clinical and SHSC models reduce to SHS).
    models = {"SHS": X_shs}
    if clinical_features:
        clin_train = _align_clinical(clinical, train_c[ID_COL], clinical_features)
        clin_val = _align_clinical(clinical, internal_c[ID_COL], clinical_features)
        clin_test = _align_clinical(clinical, external_c[ID_COL], clinical_features)
        X_clin = {
            "train": clin_train.to_numpy(dtype=np.float64),
            "val": clin_val.to_numpy(dtype=np.float64),
            "test": clin_test.to_numpy(dtype=np.float64),
        }
        X_shsc = {
            k: np.hstack([X_shs[k], X_clin[k]]) for k in ("train", "val", "test")
        }
        models["clinical"] = X_clin
        models["SHSC"] = X_shsc
    all_perf = {}
    all_params = {}
    risk_scores = {}

    for model_name, X in models.items():
        # SHS was already tuned above (as the clinical-selection baseline).
        if model_name == "SHS":
            params = shs_params
        else:
            params = tune_rsf(X["train"], y_train, X["val"], y_val)
        model = fit_final_model(X["train"], y_train, params)

        perf = _evaluate(model, X["train"], y_train, X["val"], y_val, X["test"], y_test)
        all_perf[model_name] = perf
        all_params[model_name] = params

        for cohort, Xc, yc, df in [
            ("train", X["train"], y_train, train_c),
            ("internal_val", X["val"], y_val, internal_c),
            ("external_test", X["test"], y_test, external_c),
        ]:
            risk = model.predict(Xc)
            out = pd.DataFrame({ID_COL: df[ID_COL], "risk_score": risk})
            out.to_csv(paths.output_dir / f"{model_name}_{cohort}_risk.csv", index=False)
            risk_scores[f"{model_name}_{cohort}"] = out

    # 7. Persist summary artefacts.
    summary = {
        "shs_features": shs_features,
        "clinical_features": clinical_features,
        "best_params": all_params,
        "performance": all_perf,
    }
    with (paths.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    perf_rows = []
    for model_name, perf in all_perf.items():
        for cohort, m in perf.items():
            perf_rows.append(
                {
                    "model": model_name,
                    "cohort": cohort,
                    "c_index": round(m["c_index"], 4),
                    "ci_lower": round(m["ci_lower"], 4),
                    "ci_upper": round(m["ci_upper"], 4),
                }
            )
    pd.DataFrame(perf_rows).to_csv(paths.output_dir / "performance.csv", index=False)

    return summary
