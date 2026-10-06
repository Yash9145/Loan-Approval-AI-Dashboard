import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "loan_approval_data.csv"

app = Flask(__name__)

TARGET = "Loan_Status"
ID_COL = "Loan_ID"

CATEGORICAL = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area"
]
NUMERICAL = [
    "Applicant_Income", "Coapplicant_Income", "Loan_Amount",
    "Loan_Amount_Term", "Credit_Score", "Existing_Loans", "Savings"
]
FEATURES = CATEGORICAL + NUMERICAL

model = None
metrics = {}
dataset_stats = {}
approval_counts = {}
feature_importance = []


def train_model():
    global model, metrics, dataset_stats, approval_counts, feature_importance

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_pipe, CATEGORICAL),
        ("num", numerical_pipe, NUMERICAL)
    ])

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred) * 100), 2),
        "precision": round(float(precision_score(y_test, pred, zero_division=0) * 100), 2),
        "recall": round(float(recall_score(y_test, pred, zero_division=0) * 100), 2),
        "f1": round(float(f1_score(y_test, pred, zero_division=0) * 100), 2),
    }

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()

    approval_counts = {
        "approved": int((y == 1).sum()),
        "rejected": int((y == 0).sum())
    }

    dataset_stats = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "features": int(len(FEATURES)),
        "missing_cells": int(df[FEATURES].isna().sum().sum()),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test))
    }

    # Feature importance after one-hot encoding.
    clf = model.named_steps["classifier"]
    pre = model.named_steps["preprocessor"]
    names = list(pre.get_feature_names_out())
    raw_importances = clf.feature_importances_

    grouped = {}
    for name, importance in zip(names, raw_importances):
        clean = name.split("__", 1)[-1]
        base = clean
        for feature in FEATURES:
            if clean == feature or clean.startswith(feature + "_"):
                base = feature
                break
        grouped[base] = grouped.get(base, 0) + float(importance)

    feature_importance = [
        {"feature": k, "importance": round(v * 100, 2)}
        for k, v in sorted(grouped.items(), key=lambda x: x[1], reverse=True)
    ]


def clean_value(value, kind):
    if value is None or value == "":
        return None
    if kind == "int":
        return int(float(value))
    if kind == "float":
        return float(value)
    return value


def build_input(payload):
    return pd.DataFrame([{
        "Gender": payload.get("Gender"),
        "Married": payload.get("Married"),
        "Dependents": payload.get("Dependents"),
        "Education": payload.get("Education"),
        "Self_Employed": payload.get("Self_Employed"),
        "Property_Area": payload.get("Property_Area"),
        "Applicant_Income": clean_value(payload.get("Applicant_Income"), "float"),
        "Coapplicant_Income": clean_value(payload.get("Coapplicant_Income"), "float"),
        "Loan_Amount": clean_value(payload.get("Loan_Amount"), "float"),
        "Loan_Amount_Term": clean_value(payload.get("Loan_Amount_Term"), "int"),
        "Credit_Score": clean_value(payload.get("Credit_Score"), "int"),
        "Existing_Loans": clean_value(payload.get("Existing_Loans"), "int"),
        "Savings": clean_value(payload.get("Savings"), "float")
    }])


@app.route("/")
def index():
    return render_template(
        "index.html",
        metrics=metrics,
        dataset_stats=dataset_stats,
        approval_counts=approval_counts,
        feature_importance=feature_importance[:7]
    )


@app.route("/api/stats")
def stats():
    return jsonify({
        "metrics": metrics,
        "dataset": dataset_stats,
        "approval_counts": approval_counts,
        "feature_importance": feature_importance
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        X_new = build_input(payload)

        probabilities = model.predict_proba(X_new)[0]
        prediction = int(model.predict(X_new)[0])
        approval_probability = float(probabilities[1] * 100)
        rejection_probability = float(probabilities[0] * 100)

        if approval_probability >= 70:
            risk = "LOW"
        elif approval_probability >= 45:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return jsonify({
            "approved": bool(prediction),
            "label": "LOAN APPROVED" if prediction else "LOAN NOT APPROVED",
            "approval_probability": round(approval_probability, 2),
            "rejection_probability": round(rejection_probability, 2),
            "risk": risk,
            "message": (
                "The model estimates a higher likelihood of approval based on the supplied profile."
                if prediction else
                "The model estimates a higher likelihood of rejection based on the supplied profile."
            )
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


train_model()

if __name__ == "__main__":
    app.run(debug=True)
