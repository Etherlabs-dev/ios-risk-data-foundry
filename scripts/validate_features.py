"""
IOS Risk — Feature Validation Script
======================================
Compares base features vs engineered features using the eval harness.
Proves whether the 11 new features improve on the XGBoost 0.869 baseline.

Run with:
    cd ~/Data-Foundry/ios-risk-data-foundry
    PYTHONPATH=../../eval-harness/ios-risk-eval-harness python3 scripts/validate_features.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Eval harness imports — available via PYTHONPATH
from ios_eval.metrics import compute_core_metrics, find_optimal_threshold

# Foundry imports
from foundry.features.tabular_features import run_full_feature_engineering


def train_and_evaluate(X_train, X_test, y_train, y_test, label: str) -> dict:
    """
    Train XGBoost on the given feature matrix and evaluate using
    the eval harness metrics. Returns the metrics dict.
    """
    print(f"\n--- {label} ---")
    print(f"  Train shape : {X_train.shape}")
    print(f"  Test shape  : {X_test.shape}")
    print(f"  Fraud rate  : {y_train.mean():.4%}")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold, best_metrics = find_optimal_threshold(y_test, y_prob, metric='f1')
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_core_metrics(y_test, y_pred, y_prob)

    print(f"  avg_precision : {metrics['avg_precision']:.4f}")
    print(f"  roc_auc       : {metrics['roc_auc']:.4f}")
    print(f"  f1            : {metrics['f1']:.4f}")
    print(f"  precision     : {metrics['precision']:.4f}")
    print(f"  recall        : {metrics['recall']:.4f}")

    return metrics


def run_validation(data_path: str = "data/raw/creditcard.csv") -> None:
    """
    Run two evaluation passes — base features vs engineered features —
    and print a side-by-side comparison against the 0.869 baseline.
    """
    print("Loading raw data...")
    df_raw = pd.read_csv(data_path)
    print(f"  {len(df_raw):,} rows loaded")

    # ── PASS 1: BASE FEATURES ──────────────────────────────────────────
    # Mirrors exactly what the eval harness does in Project 1:
    # drop Time, scale Amount, use V1-V28 + Amount
    df_base = df_raw.drop(columns=['Time']).copy()
    scaler  = StandardScaler()
    df_base['Amount'] = scaler.fit_transform(df_base[['Amount']])

    X_base = df_base.drop(columns=['Class']).values
    y_base = df_base['Class'].values

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42, stratify=y_base
    )

    base_metrics = train_and_evaluate(
        X_train_b, X_test_b, y_train_b, y_test_b,
        label="BASE FEATURES (V1-V28 + Amount)"
    )

    # ── PASS 2: ENGINEERED FEATURES ────────────────────────────────────
    # Run full feature engineering first, then split and scale
    print("\nRunning feature engineering...")
    df_eng = run_full_feature_engineering(df_raw.copy())

    # Drop non-numeric columns that can't go into XGBoost
    drop_cols = ['Class']
    X_eng = df_eng.drop(columns=drop_cols).values
    y_eng = df_eng['Class'].values

    scaler2 = StandardScaler()
    X_eng   = scaler2.fit_transform(X_eng)

    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        X_eng, y_eng, test_size=0.2, random_state=42, stratify=y_eng
    )

    eng_metrics = train_and_evaluate(
        X_train_e, X_test_e, y_train_e, y_test_e,
        label="ENGINEERED FEATURES (V1-V28 + Amount + 11 new)"
    )

    # ── COMPARISON TABLE ───────────────────────────────────────────────
    baseline = 0.869
    print("\n" + "=" * 55)
    print("RESULTS COMPARISON")
    print("=" * 55)
    print(f"{'Metric':<20} {'Baseline':>10} {'Base':>10} {'Engineered':>12}")
    print("-" * 55)

    metrics_to_show = ['avg_precision', 'roc_auc', 'f1', 'precision', 'recall']
    for metric in metrics_to_show:
        base_val = base_metrics[metric]
        eng_val  = eng_metrics[metric]
        bl       = baseline if metric == 'avg_precision' else '-'
        delta    = f"+{eng_val - base_val:.4f}" if eng_val > base_val else f"{eng_val - base_val:.4f}"
        print(f"{metric:<20} {str(bl):>10} {base_val:>10.4f} {eng_val:>10.4f}  ({delta})")

    print("=" * 55)

    if eng_metrics['avg_precision'] > base_metrics['avg_precision']:
        improvement = eng_metrics['avg_precision'] - base_metrics['avg_precision']
        print(f"\n✓ Engineered features IMPROVED avg_precision by +{improvement:.4f}")
    else:
        gap = base_metrics['avg_precision'] - eng_metrics['avg_precision']
        print(f"\n✗ Engineered features did not improve avg_precision (-{gap:.4f})")


if __name__ == "__main__":
    run_validation()
