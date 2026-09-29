# ML-Final-Lab-Group-09

## CrediTrust Capital — AI-Powered Credit Risk Scoring

CrediTrust Capital — AI-powered credit risk scoring for retail lending, turning applicant data into fast, defensible Good/Bad credit-risk decisions.

---

## 1. Team

Student Name             | Role |

BATCHU LEKHYA REDDY      | Data Engineer |
SYED FARHAN BASITH MADNI | Data Analyst |
PATRICIA F T             | Data Scientist |
RAJESHWARI S S           | ML Engineer |
ROOPASHREE P             | Analytics Engineer |
GEDELA NIKHITA           | BI Developer |

---

## 2. Client Persona

**Client:** CrediTrust Capital  
**Industry:** Retail Banking  
**Business Area:** Credit Risk and Retail Lending

CrediTrust Capital needs a machine learning-based credit risk scoring solution to support lending decisions using applicant-level financial and demographic information.

---

## 3. Problem Statement

The project focuses on predicting the credit risk of loan applicants as **Good** or **Bad** based on information such as credit history, checking status, loan duration, credit amount, savings status, employment, housing, and other applicant characteristics.

The objective is to develop a reproducible machine learning pipeline that can transform applicant information into a credit-risk prediction while considering the business cost of incorrect classifications.

---

## 4. Dataset

The project uses the **German Credit dataset** for credit-risk classification.

### Target Variable

`Credit_Risk`

The model predicts the applicant's credit-risk class:

- **Good**
- **Bad**

---

## 5. Primary Evaluation Metrics

The project gives particular importance to business-oriented evaluation.

### Primary Target Metrics

- **Cost-Weighted F1**
- **Expected Loss**

These metrics are used alongside standard classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The cost-sensitive evaluation gives greater importance to errors where credit is granted to applicants who belong to the higher-risk class.

### Baseline Performance

**To be updated with the final baseline values from `ML_Project.ipynb`**

---

## 6. Project Structure

```text
ML-Final-Lab-Group-09/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
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
└── tests/
    └── MLE_Sanity_Tests.ipynb
