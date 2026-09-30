# CrediTrust Capital — AI-Powered Credit Risk Scoring

> **Group 09 · BCADA 5P126 Machine Learning Lab · Final Lab Exam Project**

CrediTrust Capital is an AI-powered credit risk scoring solution designed for retail lending. The project uses machine learning to transform applicant-level financial and demographic information into reproducible **Good/Bad credit-risk classifications**.

The solution focuses not only on conventional classification performance but also on the **business cost of incorrect lending decisions**, particularly the higher cost associated with incorrectly classifying a high-risk applicant as low risk.

---

# 1. Team

## Team Name

**CrediTrust Capital — AI-Powered Credit Risk Scoring**

**Group:** 09  
**Course:** BCADA 5P126 — Machine Learning Lab  
**Project Type:** Final Lab Exam Project

## Team Members

| Student Name | Role |
|---|---|
| **BATCHU LEKHYA REDDY** | Data Engineer |
| **SYED FARHAN BASITH MADNI** | Data Analyst |
| **PATRICIA F T** | Data Scientist |
| **RAJESHWARI S S** | ML Engineer |
| **ROOPASHREE P** | Analytics Engineer |
| **GEDELA NIKHITA** | BI Developer |

---

# 2. Client Persona

## Client: CrediTrust Capital

**Industry:** Retail Banking  
**Business Area:** Credit Risk and Retail Lending

CrediTrust Capital is a retail banking organization that evaluates loan applications using applicant-level financial, employment, demographic, and credit-related information.

The organization requires a machine learning solution that can assist with consistent and data-driven credit-risk assessment.

### Business Need

The client wants to:

- Identify applicants with higher credit risk.
- Support faster and more consistent lending decisions.
- Reduce costly credit-risk misclassifications.
- Use applicant-level characteristics to generate reproducible predictions.
- Evaluate the model using business-oriented cost-sensitive metrics rather than accuracy alone.

---

# 3. Problem Statement

The project focuses on predicting whether a loan applicant represents a **Good** or **Bad** credit risk using the German Credit dataset.

The model uses information such as:

- Checking account status
- Credit history
- Loan purpose
- Credit amount
- Loan duration
- Savings status
- Employment status
- Installment rate
- Housing
- Existing credits
- Age
- Other financial and personal characteristics

The objective is to develop a reproducible machine learning pipeline that transforms applicant information into a credit-risk prediction while explicitly considering the business cost of incorrect classifications.

---

# 4. Business Risk and Cost-Sensitive Classification

Not all classification errors have the same business impact.

For this project, incorrectly treating a **Bad-risk applicant as Good** is considered more costly than incorrectly treating a Good-risk applicant as Bad.

Therefore:

- **Type I Error / False Positive:** A Good-risk applicant is classified as Bad.
- **Type II Error / False Negative:** A Bad-risk applicant is classified as Good.

The project assigns a higher business cost to Type II errors, with the specified cost ratio of:

**Type II : Type I = 5 : 1**

This makes cost-sensitive evaluation important when assessing model performance.

---

# 5. Dataset

## Dataset Used

**UCI Statlog German Credit Data**

The dataset contains information about loan applicants and their associated credit-risk classification.

### Dataset Dimensions

- **Rows:** 1,000
- **Raw Features:** 20
- **Target Column:** 1
- **Total Columns:** 21

### Target Variable

`Credit_Risk`

| Value | Meaning |
|---|---|
| `1` | Good |
| `2` | Bad |

### Class Distribution

| Credit Risk | Number of Applicants | Proportion |
|---|---:|---:|
| Good (`1`) | 700 | 70% |
| Bad (`2`) | 300 | 30% |
| **Total** | **1,000** | **100%** |

The dataset is therefore imbalanced, with Good-risk applicants representing approximately 70% of the observations.

Because of this imbalance, accuracy alone is not sufficient for evaluating the model.

---

# 6. Data Acquisition

The dataset was obtained from the **UCI Machine Learning Repository** and stored in the project repository.

The raw dataset is maintained under:

```text
data/raw/
```

The processed datasets are generated under:

```text
data/processed/
```

---

# 7. Data Engineering and Quality Checks

Before model development, the dataset undergoes data-quality validation.

The following checks were performed.

### Missing Values

No missing values were detected.

```text
Missing values: 0
```

### Duplicate Records

No duplicate rows were detected.

```text
Duplicate rows: 0
```

### Schema Validation

The dataset contains the expected:

```text
21 columns
```

including:

```text
20 features + 1 target
```

### Validity Checks

The following conditions were checked:

- Applicant age below 18
- Credit amount less than or equal to zero
- Loan duration less than or equal to zero

Results:

```text
Age below 18: 0
Credit amount <= 0: 0
Duration <= 0: 0
```

### Outlier Analysis

The IQR method was used to identify potential outliers in:

- Credit Amount
- Duration
- Age

The detected observations were retained because they represent potentially legitimate applicants and may contain useful credit-risk information.

---

# 8. Train-Test Split

The dataset was divided into training and testing subsets using a stratified split.

```text
Training Set: 800 records
Testing Set: 200 records
```

Configuration:

```python
test_size = 0.20
random_state = 42
stratify = y
```

Stratification was used to preserve the Good/Bad class distribution across the training and testing datasets.

---

# 9. Data Preprocessing

The project uses separate preprocessing strategies for numerical and categorical features.

## Numerical Features

Numerical variables are standardized using:

```text
StandardScaler
```

The numerical variables include:

- Duration_Months
- Credit_Amount
- Installment_Rate
- Residence_Since
- Age
- Existing_Credits
- Dependents

## Categorical Features

Categorical variables are converted into numerical representations using:

```text
OneHotEncoder(handle_unknown="ignore")
```

This prevents unseen categories in test or future prediction data from causing encoding errors.

## Processed Dataset

After preprocessing:

```text
Training records: 800
Testing records: 200
Processed features: 61
```

The preprocessing pipeline is fitted only on the training data and then applied to the test data.

---

# 10. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and patterns within the dataset.

The analysis included:

- Target-class distribution
- Numerical feature distributions
- Skewness analysis
- Boxplot-based outlier analysis
- Correlation analysis
- Credit Amount vs Age
- Credit Amount by Credit Risk
- Loan Duration vs Credit Risk
- Checking Status vs Credit Risk
- Numerical feature association with Credit Risk

## Key EDA Findings

### Class Imbalance

Approximately:

```text
70% Good
30% Bad
```

This motivates the use of class-aware and cost-sensitive evaluation metrics.

### Checking Status

Checking Status shows different distributions of Good and Bad risk across its categories, indicating that it contains useful information for classification.

### Loan Duration

Loan duration shows an association with credit risk and is therefore retained as an important model feature.

### Credit Amount

Credit Amount shows substantial variation and positive skewness. Extreme values were retained because they may represent legitimate high-value loans and potentially useful risk signals.

### Strongest Numerical Feature Correlations

The strongest observed numerical correlation was between:

```text
Duration_Months ↔ Credit_Amount
Correlation ≈ 0.638
```

---

# 11. Machine Learning Models

Four classification algorithms were evaluated.

## 1. Logistic Regression

Used as the baseline linear classification model.

## 2. Decision Tree

Used as an interpretable non-linear classification model.

## 3. Random Forest

Used as an ensemble tree-based model.

## 4. Gradient Boosting

Used as a boosting-based classification model and subsequently selected for hyperparameter tuning.

---

# 12. Baseline Model Performance

The initial models were evaluated on the held-out test set.

The target class for precision, recall, and F1-score was the **Bad-risk class (`Credit_Risk = 2`)**.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.780 | 0.667 | 0.533 | 0.593 |
| Decision Tree | 0.630 | 0.379 | 0.367 | 0.373 |
| Random Forest | 0.740 | 0.605 | 0.383 | 0.469 |
| Gradient Boosting | 0.790 | 0.680 | 0.567 | 0.618 |

The initial Gradient Boosting model produced:

```text
Accuracy  = 0.790
Precision = 0.680
Recall    = 0.5667
F1 Score  = 0.6182
```

These values represent the **untuned Gradient Boosting baseline**.

---

# 13. Cross-Validation

A stratified 5-fold cross-validation strategy was used.

Configuration:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The models were evaluated using F1-score during cross-validation.

| Model | Mean CV F1 | Std CV F1 |
|---|---:|---:|
| Logistic Regression | 0.8224 | 0.0303 |
| Decision Tree | 0.7713 | 0.0237 |
| Random Forest | 0.8360 | 0.0163 |
| Gradient Boosting | 0.8138 | 0.0152 |

Cross-validation was used to examine model performance across multiple training/validation splits rather than relying only on a single test result.

---

# 14. Hyperparameter Tuning

Gradient Boosting was selected for further experimentation using `GridSearchCV`.

The following hyperparameters were explored:

```text
n_estimators
learning_rate
max_depth
min_samples_split
min_samples_leaf
```

A total of:

```text
243 parameter combinations
```

were evaluated using:

```text
5-fold cross-validation
```

This resulted in:

```text
1215 model fits
```

## Best Parameters

The selected configuration was:

```text
learning_rate     = 0.1
max_depth         = 5
min_samples_leaf  = 4
min_samples_split = 10
n_estimators      = 50
random_state      = 42
```

The best cross-validation macro F1 score reported by GridSearchCV was:

```text
0.7032
```

---

# 15. Final Tuned Gradient Boosting Model

The tuned Gradient Boosting model was evaluated on the independent 200-record test set.

## Final Test Performance

| Metric | Tuned Gradient Boosting |
|---|---:|
| Accuracy | **0.7450** |
| Macro Precision | **0.6934** |
| Macro Recall | **0.6750** |
| Macro F1 Score | **0.6820** |
| Cost-Weighted F1 | **0.5877** |
| Expected Loss | **0.8550** |

### Classification Report

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Good (`1`) | 0.80 | 0.85 | 0.82 | 140 |
| Bad (`2`) | 0.59 | 0.50 | 0.54 | 60 |
| **Macro Average** | **0.69** | **0.68** | **0.68** | **200** |
| **Weighted Average** | **0.74** | **0.74** | **0.74** | **200** |

---

# 16. Baseline vs Tuned Model

The untuned and tuned Gradient Boosting models were compared on the held-out test set.

| Metric | Baseline Gradient Boosting | Tuned Gradient Boosting |
|---|---:|---:|
| Accuracy | 0.7900 | 0.7450 |
| Bad-class Precision | 0.6800 | 0.5900 |
| Bad-class Recall | 0.5667 | 0.5000 |
| Bad-class F1 | 0.6182 | 0.5400 |
| Cost-Weighted F1 | — | 0.5877 |
| Expected Loss | — | 0.8550 |

The baseline precision, recall, and F1 values are specifically for the **Bad-risk class (`2`)**, while the final model section also reports macro-averaged metrics. Therefore, these metrics should be interpreted with their averaging definition in mind.

---

# 17. Confusion Matrix

The final tuned Gradient Boosting model produced the following confusion matrix:

```text
[[119  21]
 [ 30  30]]
```

Interpreted using:

```text
Rows    = Actual Class
Columns = Predicted Class
```

| | Predicted Good | Predicted Bad |
|---|---:|---:|
| **Actual Good** | 119 | 21 |
| **Actual Bad** | 30 | 30 |

Therefore:

- True Good = 119
- Good classified as Bad = 21
- Bad classified as Good = 30
- True Bad = 30

The **30 Bad-risk applicants classified as Good** represent the Type II error category emphasized in the project's cost-sensitive evaluation.

---

# 18. Primary Target Metrics

The project uses two business-oriented metrics as its primary evaluation criteria.

## Cost-Weighted F1

The Cost-Weighted F1 metric incorporates the higher business cost assigned to Type II errors.

Final reported value:

```text
Cost-Weighted F1 = 0.587705
```

## Expected Loss

Expected Loss summarizes the business cost associated with classification errors.

Final reported value:

```text
Expected Loss = 0.855000
```

These metrics complement conventional metrics such as accuracy, precision, recall, and F1-score.

---

# 19. Feature Importance

Feature importance was extracted from the final tuned Gradient Boosting model.

The most influential features included:

| Feature | Importance |
|---|---:|
| `num__Credit_Amount` | 0.184733 |
| `cat__Checking_Status_A14` | 0.136877 |
| `num__Duration_Months` | 0.105705 |
| `num__Age` | 0.076699 |
| `cat__Savings_Status_A61` | 0.035400 |

This indicates that both numerical applicant characteristics and categorical financial-status variables contribute to the model's predictions.

---

# 20. Individual Applicant Prediction

The trained model can be used to generate an individual applicant's credit-risk prediction.

The prediction workflow is:

```text
Applicant Information
        ↓
Data Validation
        ↓
Preprocessing
        ↓
Feature Transformation
        ↓
Trained Gradient Boosting Model
        ↓
Risk Probability
        ↓
Threshold-Based Classification
        ↓
Good / Bad Credit Risk
```

Importantly, there is **no single default probability assigned to all applicants**.

Each applicant receives an individual probability based on their own input characteristics.

The project's classification threshold is:

```text
0.13
```

The individual applicant probability is evaluated against this threshold to produce the final risk classification.

---

# 21. Model Saving

The final trained model is saved using `joblib`.

Output file:

```text
final_gradient_boosting_model.pkl
```

The model was successfully saved and loaded during validation.

Example:

```python
import joblib

joblib.dump(
    best_gb_model,
    "final_gradient_boosting_model.pkl"
)

model = joblib.load(
    "final_gradient_boosting_model.pkl"
)
```

---

# 22. Project Structure

```text
ML-Final-Lab-Group-09/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── german_credit (1).csv
│   │
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       ├── X_train_processed.csv
│       ├── X_test_processed.csv
│       └── data_quality_log.csv
│
├── notebooks/
│   └── ML_Project.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── predict.py
│
├── dashboard/
│   └── Project_Dashboard.pbix
│
├── report/
│   └── Project_Report.pdf
│
├── presentation/
│   └── Client_Pitch.pptx
│
├── tests/
│   └── MLE_Sanity_Tests.ipynb
│
└── final_gradient_boosting_model.pkl
```

---

# 23. Reproducibility

The project is designed to provide a reproducible machine learning workflow.

## Step 1 — Clone the Repository

```bash
git clone https://github.com/SSRajeshwari-13/ML-Final-Lab-Group-09.git
```

Move into the project directory:

```bash
cd ML-Final-Lab-Group-09
```

---

## Step 2 — Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Step 3 — Run Data Preprocessing

Run the preprocessing script:

```bash
python src/preprocessing.py
```

This step performs data preparation, validation, encoding, scaling, and generation of processed datasets.

---

## Step 4 — Train the Models

Run:

```bash
python src/model.py
```

This step trains and evaluates the machine learning models, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

It also performs model comparison and hyperparameter tuning where implemented.

---

## Step 5 — Run Individual Predictions

Run:

```bash
python src/predict.py
```

This loads the trained model and generates credit-risk predictions for applicant-level input data.

---

## Step 6 — Run the Notebook

The complete experimental workflow can also be reproduced through:

```text
notebooks/ML_Project.ipynb
```

Open the notebook using Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Then open:

```text
notebooks/ML_Project.ipynb
```

---

# 24. Requirements

The project dependencies are maintained in:

```text
requirements.txt
```

The project uses Python and common machine learning/data-analysis libraries including:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# 25. Testing

The project includes ML sanity tests under:

```text
tests/MLE_Sanity_Tests.ipynb
```

These tests can be used to verify important properties of the machine learning pipeline, including data and model behavior.

---

# 26. Project Deliverables

The repository contains the major project deliverables:

| Deliverable | Location |
|---|---|
| Project Documentation | `README.md` |
| Requirements | `requirements.txt` |
| Raw Data | `data/raw/` |
| Processed Data | `data/processed/` |
| ML Notebook | `notebooks/ML_Project.ipynb` |
| Preprocessing Script | `src/preprocessing.py` |
| Model Training Script | `src/model.py` |
| Prediction Script | `src/predict.py` |
| Dashboard | `dashboard/Project_Dashboard.pbix` |
| Project Report | `report/Project_Report.pdf` |
| Client Presentation | `presentation/Client_Pitch.pptx` |
| ML Tests | `tests/MLE_Sanity_Tests.ipynb` |
| Trained Model | `final_gradient_boosting_model.pkl` |

---

# 27. End-to-End ML Pipeline

The complete solution follows this workflow:

```text
                 ┌───────────────────────┐
                 │   German Credit Data  │
                 └───────────┬───────────┘
                             ↓
                 ┌───────────────────────┐
                 │ Data Quality Checks   │
                 │ Missing/Duplicates    │
                 │ Schema/Validity       │
                 └───────────┬───────────┘
                             ↓
                 ┌───────────────────────┐
                 │ Exploratory Analysis  │
                 └───────────┬───────────┘
                             ↓
                 ┌───────────────────────┐
                 │ Train/Test Split      │
                 │ 800 / 200             │
                 └───────────┬───────────┘
                             ↓
             ┌───────────────────────────────┐
             │ Feature Preprocessing         │
             │                               │
             │ Numerical → StandardScaler    │
             │ Categorical → OneHotEncoder   │
             └──────────────┬────────────────┘
                            ↓
             ┌───────────────────────────────┐
             │ Model Development              │
             │                               │
             │ Logistic Regression            │
             │ Decision Tree                  │
             │ Random Forest                  │
             │ Gradient Boosting              │
             └──────────────┬────────────────┘
                            ↓
             ┌───────────────────────────────┐
             │ Cross-Validation               │
             └──────────────┬────────────────┘
                            ↓
             ┌───────────────────────────────┐
             │ Hyperparameter Tuning          │
             │ Gradient Boosting + GridSearch │
             └──────────────┬────────────────┘
                            ↓
             ┌───────────────────────────────┐
             │ Final Tuned Model              │
             └──────────────┬────────────────┘
                            ↓
             ┌───────────────────────────────┐
             │ Cost-Sensitive Evaluation      │
             │ Cost-Weighted F1               │
             │ Expected Loss                  │
             └──────────────┬────────────────┘
                            ↓
             ┌───────────────────────────────┐
             │ Applicant-Level Prediction     │
             │ Good / Bad Risk                │
             └───────────────────────────────┘
```

---

# 28. Key Project Insights

### Insight 1 — Class Imbalance

The dataset contains approximately 70% Good-risk and 30% Bad-risk applicants.

**Implication:** Accuracy should not be used as the only measure of model quality.

---

### Insight 2 — Checking Status

Checking Status shows differences in credit-risk distribution across categories.

**Implication:** It provides useful information for credit-risk classification and is retained as a model feature.

---

### Insight 3 — Loan Duration and Credit Amount

Loan Duration and Credit Amount show useful relationships with credit risk.

**Implication:** Both variables are retained during model development and their distributions are monitored.

---

# 29. Limitations

This project uses the German Credit dataset and therefore has limitations associated with:

- Dataset size
- Historical applicant population
- Dataset-specific feature definitions
- Class imbalance
- Potential differences between historical data and modern lending environments

The reported performance should therefore be interpreted within the context of the dataset and test split used in this project.

---

# 30. Important Note on Model Evaluation

The project uses both standard machine learning metrics and business-oriented cost-sensitive metrics.

Standard metrics:

```text
Accuracy
Precision
Recall
F1 Score
```

Business-oriented metrics:

```text
Cost-Weighted F1
Expected Loss
```

Because the cost of Type II errors is explicitly higher, model evaluation should not rely on accuracy alone.

---

# 31. Final Project Summary

CrediTrust Capital demonstrates an end-to-end machine learning workflow for credit-risk classification.

The project covers:

```text
Data Acquisition
      ↓
Data Quality Validation
      ↓
Exploratory Data Analysis
      ↓
Feature Preprocessing
      ↓
Model Development
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Cost-Sensitive Evaluation
      ↓
Model Saving
      ↓
Applicant-Level Prediction
```

The final tuned Gradient Boosting model achieved the following reported test-set metrics:

```text
Accuracy          : 0.7450
Macro Precision   : 0.6934
Macro Recall      : 0.6750
Macro F1          : 0.6820
Cost-Weighted F1  : 0.5877
Expected Loss     : 0.8550
```

The trained model is saved as:

```text
final_gradient_boosting_model.pkl
```

---

# 32. Team

## CrediTrust Capital — Group 09

**BCADA 5P126 · Machine Learning Lab**

| Member | Responsibility |
|---|---|
| **BATCHU LEKHYA REDDY** | Data Engineering |
| **SYED FARHAN BASITH MADNI** | Data Analysis |
| **PATRICIA F T** | Data Science |
| **RAJESHWARI S S** | Machine Learning Engineering |
| **ROOPASHREE P** | Analytics Engineering |
| **GEDELA NIKHITA** | Business Intelligence |

---

**CrediTrust Capital — Turning Applicant Data into Defensible Credit-Risk Decisions.**
