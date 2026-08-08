"""
IOS Risk — Synthetic Transaction Generator
==========================================
Generates realistic synthetic fraud and legitimate transactions
by sampling from the statistical distributions of real data.

WHY THIS EXISTS:
  The real dataset is 99.83% legitimate. A model trained on this
  imbalance learns to predict "legitimate" always and still scores
  99.83% accuracy — useless for fraud detection. Synthetic data
  rebalances the training set so the model must learn both classes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from foundry.features.tabular_features import run_full_feature_engineering
from foundry.formatters import df_to_instruction_pairs


def _validate_generation_input(n: int, real_df: pd.DataFrame, class_label: int) -> pd.DataFrame:
    if n < 0:
        raise ValueError("Synthetic record count must be non-negative")
    required = {"Time", "Amount", "Class"}
    missing = required.difference(real_df.columns)
    if missing:
        raise ValueError(f"Real data is missing required columns: {', '.join(sorted(missing))}")
    source_rows = real_df[real_df["Class"] == class_label]
    if source_rows.empty:
        label = "fraud" if class_label == 1 else "legitimate"
        raise ValueError(f"Real data must include at least one {label} row")
    return source_rows


def generate_synthetic_fraud(n: int, real_df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """
    Generate n synthetic fraud transactions by sampling from the
    statistical profile of real fraud rows in real_df.
    """
    rng = np.random.default_rng(seed)
    fraud_rows = _validate_generation_input(n, real_df, class_label=1)

    # Sample amounts from a log-normal distribution matching real fraud
    # Log-normal because amounts can't go below 0 and are right-skewed
    mean_amt = fraud_rows["Amount"].mean()
    amounts = rng.lognormal(mean=np.log(mean_amt + 1), sigma=0.5, size=n).round(2)

    # Sample times spread across the dataset's time range
    max_time = real_df["Time"].max()
    times = rng.uniform(low=0, high=max_time, size=n)

    # Sample V1-V28 from real fraud row distributions
    v_cols = [c for c in real_df.columns if c.startswith("V")]
    v_data = {}
    for col in v_cols:
        col_mean = fraud_rows[col].mean()
        col_std = fraud_rows[col].std()
        v_data[col] = rng.normal(loc=col_mean, scale=col_std + 1e-8, size=n)

    synthetic = pd.DataFrame(v_data)
    synthetic["Time"] = times
    synthetic["Amount"] = amounts
    synthetic["Class"] = 1  # all fraud

    return synthetic[real_df.columns]  # same column order as real data


def generate_synthetic_legit(n: int, real_df: pd.DataFrame, *, seed: int = 99) -> pd.DataFrame:
    """
    Generate n synthetic legitimate transactions by sampling from the
    statistical profile of real legitimate rows in real_df.
    """
    rng = np.random.default_rng(seed)
    legit_rows = _validate_generation_input(n, real_df, class_label=0)

    # Legitimate amounts have a wider spread than fraud
    mean_amt = legit_rows["Amount"].mean()
    amounts = rng.lognormal(
        mean=np.log(mean_amt + 1),
        sigma=0.8,  # wider sigma = more variety in amounts
        size=n,
    ).round(2)

    # Times spread evenly across the full dataset range
    max_time = real_df["Time"].max()
    times = rng.uniform(low=0, high=max_time, size=n)

    # Sample V1-V28 from real legitimate row distributions
    v_cols = [c for c in real_df.columns if c.startswith("V")]
    v_data = {}
    for col in v_cols:
        col_mean = legit_rows[col].mean()
        col_std = legit_rows[col].std()
        v_data[col] = rng.normal(loc=col_mean, scale=col_std + 1e-8, size=n)

    synthetic = pd.DataFrame(v_data)
    synthetic["Time"] = times
    synthetic["Amount"] = amounts
    synthetic["Class"] = 0  # all legitimate

    return synthetic[real_df.columns]


def process_synthetic_source(config: dict, real_df: pd.DataFrame) -> None:
    """
    Generate synthetic transactions, engineer features, format as
    instruction pairs and export to JSONL.

    real_df: the raw creditcard DataFrame — used to derive statistical
             profiles for generation and column structure.
    """
    synth_cfg = config["sources"]["synthetic"]
    n_fraud = synth_cfg["n_fraud"]
    n_legit = synth_cfg["n_legit"]
    seed = synth_cfg.get("seed", 42)
    output_path = synth_cfg["output_path"]

    print(f"Generating {n_fraud:,} synthetic fraud transactions...")
    fraud_df = generate_synthetic_fraud(n_fraud, real_df, seed=seed)

    print(f"Generating {n_legit:,} synthetic legitimate transactions...")
    legit_df = generate_synthetic_legit(n_legit, real_df, seed=seed + 1)

    # Combine and shuffle so fraud/legit aren't in blocks
    combined = pd.concat([fraud_df, legit_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Combined shape: {combined.shape}")
    print(f"Fraud rate: {combined['Class'].mean():.2%}")

    # Run the same feature engineering as the real pipeline
    print("Running feature engineering on synthetic data...")
    engineered = run_full_feature_engineering(combined)

    # Format as instruction pairs
    pairs = df_to_instruction_pairs(engineered)

    # Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print("\n✓ Synthetic pipeline complete")
    print(f"  Fraud pairs : {n_fraud:,}")
    print(f"  Legit pairs : {n_legit:,}")
    print(f"  Total pairs : {len(pairs):,}")
    print(f"  Output file : {output_path}")
