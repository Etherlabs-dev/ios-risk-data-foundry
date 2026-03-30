"""
IOS Risk — Main Pipeline
========================
Entry point for the full data processing pipeline.

Run with:
    python -m foundry.pipeline

What it does:
    1. Loads raw creditcard.csv
    2. Runs feature engineering
    3. Formats as Alpaca instruction pairs
    4. Exports to JSONL for HuggingFace fine-tuning
"""


import pandas as pd
import yaml
from pathlib import Path
from foundry.features.tabular_features import run_full_feature_engineering
from foundry.formatters import df_to_instruction_pairs, export_to_jsonl


def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    """
    Load the YAML config file and return it as a Python dict.
    All pipeline parameters come from here — nothing hardcoded below.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Config loaded from {config_path}")
    return config


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """
    Load the raw CSV and print a loading summary.
    Fails early with a clear message if the file is missing.
    """
    path = Path(raw_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at '{raw_path}'.\n"
            f"Make sure creditcard.csv is in data/raw/"
        )

    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} rows from {raw_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Fraud rate: {df['Class'].mean():.4%}  ({df['Class'].sum():,} fraud / {len(df):,} total)")

    return df




def run_pipeline(config_path: str = "configs/pipeline_config.yaml") -> None:
    """
    Orchestrates the full pipeline end to end.
    Reads all parameters from config — nothing hardcoded here.
    """
    # Step 1: Load config
    config = load_config(config_path)
    raw_path     = config['sources']['fraud_transactions']['raw_path']
    export_path  = config['exports']['instruction_pairs_path']

    # Step 2: Load raw data
    print("\n--- Loading raw data ---")
    df = load_raw_data(raw_path)

    # Step 3: Feature engineering
    print("\n--- Running feature engineering ---")
    df_engineered = run_full_feature_engineering(df)

    # Step 4: Format as instruction pairs
    print("\n--- Formatting instruction pairs ---")
    pairs = df_to_instruction_pairs(df_engineered)
    print(f"Generated {len(pairs):,} instruction pairs")

    # Step 5: Export to JSONL
    print("\n--- Exporting to JSONL ---")
    export_to_jsonl(pairs, export_path)

    # Step 6: Summary
    fraud_pairs = sum(1 for p in pairs if p['output'] == 'FRAUD')
    legit_pairs = sum(1 for p in pairs if p['output'] == 'LEGITIMATE')
    print(f"\n✓ Pipeline complete")
    print(f"  Total pairs : {len(pairs):,}")
    print(f"  Fraud       : {fraud_pairs:,} ({fraud_pairs/len(pairs):.2%})")
    print(f"  Legitimate  : {legit_pairs:,} ({legit_pairs/len(pairs):.2%})")
    print(f"  Output file : {export_path}")

if __name__ == "__main__":
    run_pipeline()
