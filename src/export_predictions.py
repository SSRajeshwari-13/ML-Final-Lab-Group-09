"""
export_predictions.py  --  Analytics Engineer helper (Group 09)
=================================================================
Creates the two prediction files that src/analytics_engine.py needs:

    data/processed/oof_predictions.csv   <- 5-fold OUT-OF-FOLD predictions on TRAIN
                                            (used to CHOOSE the threshold)
    data/processed/test_predictions.csv  <- predictions on the held-out TEST set
                                            (used to REPORT results once)

Columns written:  y_true (1 = DEFAULT/bad, 0 = good), p_default,
                  credit_amount, duration   (real, un-scaled values)

Why OOF?  The threshold must not be tuned on the test set (brief: "Never manipulate
test data to improve the score").  Each OOF prediction comes from a model that never
saw that row, so it is an honest stand-in for validation data.

Inputs already in the repo:
    data/processed/train.csv, test.csv      (processed features + Credit_Risk, coded 1 = good, 2 = bad)
    src/Final_model (1).pkl                 (tuned Gradient Boosting from model.py)
    src/preprocessing.pkl                   (fitted ColumnTransformer: used only to un-scale
                                             Credit_Amount and Duration_Months)

Run from the project root:
    python src/export_predictions.py
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
BAD_CLASS = 2          # raw coding in the German Credit file: 1 = good, 2 = bad


def unscale(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """Recover real Credit_Amount / Duration_Months from the standardised columns."""
    num_cols = list(preprocessor.transformers_[0][2])           # order used when fitting
    scaler = preprocessor.named_transformers_["num"]
    scaled = df[[f"num__{c}" for c in num_cols]].to_numpy()
    raw = pd.DataFrame(scaler.inverse_transform(scaled), columns=num_cols, index=df.index)

    out = pd.DataFrame({
        "credit_amount": raw["Credit_Amount"].round(0),
        "duration": raw["Duration_Months"].round(0),
    })
    # Sanity checks: these guard against a preprocessor that was refit on other data
    frac = (raw["Credit_Amount"] - raw["Credit_Amount"].round()).abs().max()
    if frac > 0.01 or out["credit_amount"].min() <= 0 or out["duration"].min() <= 0:
        raise ValueError(
            "Un-scaled values look wrong (not whole numbers / not positive). "
            "preprocessing.pkl may not match train.csv - ask the Data Engineer.")
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
    bad_idx = classes.index(BAD_CLASS)                          # column of P(bad)
    print("Model classes:", classes, "-> P(default) = predict_proba column", bad_idx)

    # ---- TEST predictions from the final (already fitted) model --------------
    test_out = pd.DataFrame({
        "y_true": (y_te == BAD_CLASS).astype(int).to_numpy(),
        "p_default": model.predict_proba(X_te)[:, bad_idx],
    })
    test_out = pd.concat([test_out, unscale(test, preprocessor).reset_index(drop=True)], axis=1)

    # ---- OUT-OF-FOLD predictions on TRAIN (same hyper-parameters, refit per fold)
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
