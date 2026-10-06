# Loan Approval AI — Professional Web Dashboard

A mini-project that predicts whether a loan application is likely to be approved using a Random Forest machine-learning pipeline and a Flask web dashboard.

## Features

- 800-row loan-application CSV included
- 13 predictive features + ID + target
- Missing-value handling
- One-hot encoding for categorical features
- Random Forest Classifier
- 200 estimators
- `random_state=42`
- `max_depth=10`
- Accuracy, precision, recall and F1 metrics
- Prediction probability
- Risk indicator
- Feature-importance analytics
- Responsive banking-style dashboard

## Project structure

```text
Loan_Approval_AI_Dashboard/
├── app.py
├── loan_approval_data.csv
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Setup

### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### Windows Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## How the prediction works

1. The CSV is loaded when the Flask server starts.
2. Missing categorical values are filled with the most frequent category.
3. Missing numerical values are filled with the median.
4. Categorical fields are one-hot encoded.
5. A Random Forest classifier learns from the training data.
6. A held-out test set is used for evaluation.
7. The web form sends a new applicant to `/api/predict`.
8. The model returns the predicted class and probabilities.

## Dataset

The included CSV is **synthetic educational data**, generated specifically for this mini-project. It is intentionally designed to resemble common loan-approval datasets while avoiding dependence on another student's repository or copying a third-party dataset.

This makes the project self-contained and suitable for a college demonstration.

## Important

This is an academic machine-learning demonstration. A real lending institution would require much more extensive validation, regulatory review, fairness testing, explainability, security, and domain-specific underwriting rules.
