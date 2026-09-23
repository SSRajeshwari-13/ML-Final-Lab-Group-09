"""
MLE Inference Pipeline

Loads the trained credit-risk model and preprocessing pipeline,
validates applicant input data, performs inference, measures
latency, logs operational information, and saves prediction results.
"""

import logging
import time
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_PATH / "src" / "Final_model (1).pkl"
PREPROCESSOR_PATH = PROJECT_PATH / "src" / "preprocessing.pkl"
INPUT_PATH = PROJECT_PATH / "data" / "processed" / "sample_applicants_data.csv"
OUTPUT_PATH = PROJECT_PATH / "data" / "processed" / "prediction_results.csv"


# ============================================================
# 2. SAMPLE INPUT DATA
# ============================================================

data = [
    ["A14", 6, "A34", "A43", 1200, "A65", "A75", 1, "A93", "A101", 4, "A121", 23, "A143", "A152", 1, "A173", 1, "A191", "A202"],
    ["A11", 48, "A31", "A40", 8500, "A61", "A71", 4, "A92", "A101", 4, "A124", 61, "A142", "A152", 2, "A173", 1, "A191", "A201"],
    ["A13", 18, "A34", "A42", 3200, "A64", "A74", 2, "A93", "A101", 3, "A123", 42, "A141", "A152", 1, "A172", 1, "A192", "A201"],
    ["A12", 60, "A30", "A46", 9200, "A65", "A72", 4, "A91", "A103", 1, "A122", 30, "A142", "A153", 3, "A174", 2, "A192", "A201"],
    ["A11", 12, "A32", "A49", 2100, "A61", "A73", 3, "A92", "A101", 2, "A121", 27, "A143", "A152", 1, "A172", 1, "A191", "A202"],
    ["A13", 36, "A33", "A43", 6100, "A62", "A74", 4, "A93", "A101", 4, "A124", 47, "A143", "A151", 4, "A173", 2, "A191", "A201"],
    ["A12", 24, "A32", "A44", 4100, "A63", "A73", 2, "A92", "A102", 3, "A123", 38, "A142", "A152", 2, "A172", 3, "A192", "A201"],
    ["A14", 9, "A33", "A41", 1450, "A64", "A75", 1, "A93", "A101", 4, "A121", 68, "A141", "A151", 1, "A172", 1, "A191", "A201"],
    ["A11", 30, "A31", "A40", 5600, "A65", "A72", 3, "A91", "A103", 2, "A122", 25, "A142", "A153", 2, "A174", 1, "A192", "A201"],
    ["A13", 36, "A33", "A43", 6100, "A62", "A74", 4, "A93", "A101", 4, "A124", 47, "A143", "A151", 4, "A173", 2, "A191", "A202"]
]

columns = [
    "Checking_Status",
    "Duration_Months",
    "Credit_History",
    "Purpose",
    "Credit_Amount",
    "Savings_Status",
    "Employment_Since",
    "Installment_Rate",
    "Personal_Status_Sex",
    "Other_Debtors_Guarantors",
    "Residence_Since",
    "Property",
    "Age",
    "Other_Installment_Plans",
    "Housing",
    "Existing_Credits",
    "Job",
    "Dependents",
    "Telephone",
    "Foreign_Worker",
]


# ============================================================
# 3. LOGGING
# ============================================================

logger = logging.getLogger("MLE_Inference")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ============================================================
# 4. CREATE INPUT DATA
# ============================================================

logger.info("Inference pipeline started.")

input_df = pd.DataFrame(data, columns=columns)

INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
input_df.to_csv(INPUT_PATH, index=False)

print(f"Input data saved to: {INPUT_PATH}")
print("Dataset shape:", input_df.shape)


# ============================================================
# 5. LOAD MODEL AND PREPROCESSOR
# ============================================================

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Model type:", type(model))
print("Preprocessor type:", type(preprocessor))
print("Model classes:", model.classes_)


# ============================================================
# 6. LOAD INPUT DATA
# ============================================================

input_df = pd.read_csv(INPUT_PATH)

expected_columns = list(preprocessor.feature_names_in_)
actual_columns = list(input_df.columns)

print("\nExpected columns:")
print(expected_columns)

print("\nActual columns:")
print(actual_columns)

print("\nExact order match:", expected_columns == actual_columns)

missing_columns = sorted(set(expected_columns) - set(actual_columns))
extra_columns = sorted(set(actual_columns) - set(expected_columns))

print("Missing columns:", missing_columns)
print("Extra columns:", extra_columns)

if expected_columns != actual_columns:
    raise ValueError(
        "Input columns do not match the expected feature names/order."
    )

logger.info("Input schema validation passed.")


# ============================================================
# 7. PREPROCESS INPUT DATA
# ============================================================

start_time = time.perf_counter()

logger.info("Preprocessing started.")

try:
    transformed_data = preprocessor.transform(input_df)

    print("\nPreprocessing successful.")
    print("Raw input shape:", input_df.shape)
    print("Transformed output shape:", transformed_data.shape)
    print("Transformed data type:", type(transformed_data))

    logger.info("Preprocessing completed successfully.")

except Exception as error:
    logger.error("Preprocessing failed: %s", error)
    raise


# ============================================================
# 8. FEATURE COMPATIBILITY CHECK
# ============================================================

model_features = model.n_features_in_
preprocessor_features = transformed_data.shape[1]

print("\nModel expects transformed features:", model_features)
print("Preprocessor produced features:", preprocessor_features)

if model_features != preprocessor_features:
    raise ValueError(
        f"Feature-count mismatch: model expects {model_features}, "
        f"but preprocessor produced {preprocessor_features}."
    )

print("Success: preprocessor output matches model input.")


# ============================================================
# 9. MODEL INFERENCE
# ============================================================

transformed_data_df = pd.DataFrame(
    transformed_data,
    columns=preprocessor.get_feature_names_out()
)

predictions = model.predict(transformed_data_df)
probabilities = model.predict_proba(transformed_data_df)

end_time = time.perf_counter()
latency = end_time - start_time

logger.info("Prediction completed successfully.")
logger.info("Inference latency: %.3f ms", latency * 1000)

print(f"\nInference latency: {latency * 1000:.3f} ms")

print("\nPredicted classes:")
print(predictions)

print("\nProbability matrix shape:")
print(probabilities.shape)

print("\nFirst applicant probability values:")
print(probabilities[0])


# ============================================================
# 10. CREATE AND SAVE PREDICTION RESULTS
# ============================================================

results = input_df.copy()

results["Predicted_Class"] = predictions
results["Probability_Class_1"] = probabilities[:, 0]
results["Probability_Class_2"] = probabilities[:, 1]
results["Prediction_Confidence"] = probabilities.max(axis=1)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print(f"\nPrediction results saved to: {OUTPUT_PATH}")

logger.info("Prediction results saved successfully.")
