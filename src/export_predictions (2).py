"""
export_predictions.py -- Analytics Engineer helper (Group 09)
Creates oof_predictions.csv and test_predictions.csv for the AE pipeline.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROJECT = Path(__file__).resolve().parents[1]
PROC = PROJECT / "data" / "processed"
MODEL_PATH = PROJECT / "src" / "Final_model (1).pkl"
PREP_PATH = PROJECT / "src" / "preprocessing.pkl"

TARGET = "Credit_Risk"
BAD_CLASS = 2


def unscale(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    num_cols = list(preprocessor.transformers_[0][2])
    scaler = preprocessor.named_transformers_["num"]
    scaled = df[[f"num__{c}" for c in num_cols]].to_numpy()
    raw = pd.DataFrame(scaler.inverse_transform(scaled), columns=num_cols, index=df.index)

    out = pd.DataFrame({
        "credit_amount": raw["Credit_Amount"].round(0),
        "duration": raw["Duration_Months"].round(0),
    })
    frac = (raw["Credit_Amount"] - raw["Credit_Amount"].round()).abs().max()
    if frac > 0.01 or out["credit_amount"].min() <= 0 or out["duration"].min() <= 0:
        raise ValueError(
            "Un-scaled values look wrong. preprocessing.pkl may not match train.csv.")
    return out


def main():
    train = pd.read_csv(PROC / "train.csv")
    test = pd.read_csv(PROC / "test.csv")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREP_PATH)

    feats = [c for c in train.columns if c != TARGET]
    X_tr, y_tr = train[feats], train[TARGET]
    X_te, y_te = test[feats], test[TARGET]

    classes = list(model.classes_)
    if BAD_CLASS not in classes:
        raise ValueError(f"Expected class {BAD_CLASS} in model.classes_, got {classes}")
    bad_idx = classes.index(BAD_CLASS)
    print("Model classes:", classes, "-> P(default) = predict_proba column", bad_idx)

    test_out = pd.DataFrame({
        "y_true": (y_te == BAD_CLASS).astype(int).to_numpy(),
        "p_default": model.predict_proba(X_te)[:, bad_idx],
    })
    test_out = pd.concat([test_out, unscale(test, preprocessor).reset_index(drop=True)], axis=1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_p = cross_val_predict(clone(model), X_tr, y_tr, cv=cv,
                              method="predict_proba")[:, bad_idx]
    oof_out = pd.DataFrame({
        "y_true": (y_tr == BAD_CLASS).astype(int).to_numpy(),
        "p_default": oof_p,
    })
    oof_out = pd.concat([oof_out, unscale(train, preprocessor).reset_index(drop=True)], axis=1)

    PROC.mkdir(parents=True, exist_ok=True)
    oof_out.to_csv(PROC / "oof_predictions.csv", index=False)
    test_out.to_csv(PROC / "test_predictions.csv", index=False)

    for name, d in (("OOF (train)", oof_out), ("TEST", test_out)):
        print(f"{name:12s} rows={len(d):4d}  default rate={d['y_true'].mean():.3f}  "
              f"mean P(default)={d['p_default'].mean():.3f}")
    print("Saved oof_predictions.csv and test_predictions.csv to", PROC)


if __name__ == "__main__":
    main()
