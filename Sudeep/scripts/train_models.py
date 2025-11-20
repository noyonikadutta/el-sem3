# FarmerML/scripts/train_models.py
import os
import warnings
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_CSV = os.path.join(PROJECT_ROOT, "output", "train.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "output", "test.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Labels to train
LABELS = ["drought_label", "flood_label", "heatwave_label"]

# Features to use (these should match columns produced by the feature script)
FEATURE_COLS = [
    "imd_rain_mm", "rain_1w", "rain_4w", "rain_12w", "rain_1w_lag1", "rain_4w_lag1",
    "t2m_4w", "t2mmax_4w", "t2m_4w_lag1", "rh_4w",
    "rain_imd_vs_nasa_ratio", "rain12w_anom_ratio"
]

# Utility: scoring function
def score_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = np.nan
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except:
        pr_auc = np.nan
    return {"precision": p, "recall": r, "f1": f1, "roc_auc": roc, "pr_auc": pr_auc}

# ---------- LOAD data ----------
print("Loading train/test CSVs...")
train_df = pd.read_csv(TRAIN_CSV, parse_dates=["week_start"])
test_df = pd.read_csv(TEST_CSV, parse_dates=["week_start"])

# confirm features exist
missing_feats = [c for c in FEATURE_COLS if c not in train_df.columns]
if missing_feats:
    raise ValueError("Missing feature columns in train.csv: " + ", ".join(missing_feats))

# Ensure iso_year column exists
if "iso_year" not in train_df.columns:
    train_df["iso_year"] = train_df["week_start"].dt.isocalendar().year
if "iso_year" not in test_df.columns:
    test_df["iso_year"] = test_df["week_start"].dt.isocalendar().year

# ---------- Rolling-origin cross-validation (by year) ----------
years = sorted(train_df["iso_year"].unique())
if len(years) < 2:
    raise ValueError("Not enough years in training data to perform rolling CV; need >= 2 years.")

cv_splits = []
for i in range(len(years)-1):
    train_years = [y for y in years if y <= years[i]]
    val_year = years[i+1]
    cv_splits.append((train_years, val_year))

print(f"Rolling CV folds (train up to year -> validate on next year): {cv_splits}")

# Storage for metrics
all_metrics = []

# Train per label
final_models = {}

for label in LABELS:
    print("\n" + "="*60)
    print(f"Training label: {label}")
    # Collect per-fold metrics
    fold_metrics = []

    # For each fold: train on train_years, validate on val_year
    for idx, (train_years, val_year) in enumerate(cv_splits):
        tr_mask = train_df["iso_year"].isin(train_years)
        val_mask = train_df["iso_year"] == val_year
        X_tr = train_df.loc[tr_mask, FEATURE_COLS].fillna(0.0)
        y_tr = train_df.loc[tr_mask, label].fillna(0).astype(int)
        X_val = train_df.loc[val_mask, FEATURE_COLS].fillna(0.0)
        y_val = train_df.loc[val_mask, label].fillna(0).astype(int)

        if len(y_tr) == 0 or len(y_val) == 0:
            print(f"Skipping fold {idx}: insufficient data.")
            continue

        # Standardize for logistic regression
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        # Baseline: Logistic Regression
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr.fit(X_tr_scaled, y_tr)
        lr_prob = lr.predict_proba(X_val_scaled)[:, 1]
        lr_scores = score_binary(y_val, lr_prob)

        # Stronger model: XGBoost
        # compute scale_pos_weight to help with imbalance
        pos = max(y_tr.sum(), 1)
        neg = max(len(y_tr) - y_tr.sum(), 1)
        scale_pos_weight = neg / pos
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200, max_depth=4,
                            learning_rate=0.1, scale_pos_weight=scale_pos_weight, random_state=42)
        xgb.fit(X_tr, y_tr)
        xgb_prob = xgb.predict_proba(X_val)[:, 1]
        xgb_scores = score_binary(y_val, xgb_prob)

        fold_metrics.append({
            "label": label,
            "fold": idx,
            "train_years": train_years,
            "val_year": val_year,
            "lr_precision": lr_scores["precision"],
            "lr_recall": lr_scores["recall"],
            "lr_f1": lr_scores["f1"],
            "lr_roc_auc": lr_scores["roc_auc"],
            "lr_pr_auc": lr_scores["pr_auc"],
            "xgb_precision": xgb_scores["precision"],
            "xgb_recall": xgb_scores["recall"],
            "xgb_f1": xgb_scores["f1"],
            "xgb_roc_auc": xgb_scores["roc_auc"],
            "xgb_pr_auc": xgb_scores["pr_auc"],
        })

        print(f" Fold {idx} | val_year {val_year} | LR f1 {lr_scores['f1']:.3f} | XGB f1 {xgb_scores['f1']:.3f}")

    # Save fold metrics
    fold_df = pd.DataFrame(fold_metrics)
    if not fold_df.empty:
        all_metrics.append(fold_df)

    # ---------- Fit final models on ALL train data for this label ----------
    X_full = train_df[FEATURE_COLS].fillna(0.0)
    y_full = train_df[label].fillna(0).astype(int)

    # Final logistic pipeline (scaler + LR)
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])
    lr_pipe.fit(X_full, y_full)

    # Final XGBoost
    pos = max(y_full.sum(), 1)
    neg = max(len(y_full) - y_full.sum(), 1)
    scale_pos_weight = neg / pos
    xgb_final = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=300,
                              max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42)
    xgb_final.fit(X_full, y_full)

    # Save final models
    lr_path = os.path.join(MODELS_DIR, f"{label}_logreg.joblib")
    xgb_path = os.path.join(MODELS_DIR, f"{label}_xgb.joblib")
    joblib.dump(lr_pipe, lr_path)
    joblib.dump(xgb_final, xgb_path)
    print(f"Saved models: {lr_path}, {xgb_path}")

    final_models[label] = {"lr": lr_pipe, "xgb": xgb_final}

# Combine and persist CV metrics
if all_metrics:
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_out = os.path.join(OUT_DIR, "model_cv_metrics.csv")
    metrics_df.to_csv(metrics_out, index=False)
    print("Saved CV metrics to:", metrics_out)
else:
    print("No CV metrics to save.")

# ---------- Evaluate on TEST set and produce sample predictions ----------
print("\nEvaluating final models on TEST set...")
sample_rows = []
test_X = test_df[FEATURE_COLS].fillna(0.0)
for label in LABELS:
    models = final_models[label]
    # LR probabilities
    lr_prob = models["lr"].predict_proba(test_X)[:, 1]
    xgb_prob = models["xgb"].predict_proba(test_X)[:, 1]

    # Scores
    y_true = test_df[label].fillna(0).astype(int)
    lr_scores = score_binary(y_true, lr_prob)
    xgb_scores = score_binary(y_true, xgb_prob)

    print(f"\nLabel: {label} | Test LR F1: {lr_scores['f1']:.3f} | Test XGB F1: {xgb_scores['f1']:.3f} | XGB PR-AUC: {xgb_scores['pr_auc']:.3f}")

    # Save summary per label
    summary = {
        "label": label,
        "lr_precision_test": lr_scores["precision"],
        "lr_recall_test": lr_scores["recall"],
        "lr_f1_test": lr_scores["f1"],
        "lr_roc_auc_test": lr_scores["roc_auc"],
        "lr_pr_auc_test": lr_scores["pr_auc"],
        "xgb_precision_test": xgb_scores["precision"],
        "xgb_recall_test": xgb_scores["recall"],
        "xgb_f1_test": xgb_scores["f1"],
        "xgb_roc_auc_test": xgb_scores["roc_auc"],
        "xgb_pr_auc_test": xgb_scores["pr_auc"],
    }
    sample_rows.append(summary)

    # SHAP summary for xgb: save plot
    try:
        explainer = shap.TreeExplainer(models["xgb"])
        shap_vals = explainer.shap_values(test_X)
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_vals, test_X, show=False, plot_type="bar")
        out_shap = os.path.join(OUT_DIR, f"shap_summary_{label}.png")
        plt.tight_layout()
        plt.savefig(out_shap, dpi=150)
        plt.close()
        print("Saved SHAP summary:", out_shap)
    except Exception as e:
        print("SHAP generation failed for", label, ":", e)

# Save test summary
test_summary_df = pd.DataFrame(sample_rows)
test_summary_out = os.path.join(OUT_DIR, "model_test_metrics.csv")
test_summary_df.to_csv(test_summary_out, index=False)
print("\nSaved test metrics to:", test_summary_out)

# Also produce example predictions CSV (probs + thresholds)
print("Producing sample_predictions.csv with XGB probs and simple advisory labels...")
pred_out = test_df[["district", "week_start", "iso_year", "week_of_year"]].copy()
for label in LABELS:
    xgb = final_models[label]["xgb"]
    pred_out[f"{label}_prob_xgb"] = xgb.predict_proba(test_X)[:, 1]
    # Simple advisory levels
    pred_out[f"{label}_advisory"] = pd.cut(pred_out[f"{label}_prob_xgb"], bins=[-0.01, 0.5, 0.8, 1.0], labels=["low", "moderate", "high"]).astype(str)

sample_pred_path = os.path.join(OUT_DIR, "sample_predictions.csv")
pred_out.to_csv(sample_pred_path, index=False)
print("Saved sample predictions to:", sample_pred_path)

print("\nAll done. Models saved in:", MODELS_DIR)
