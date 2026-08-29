"""
IOS Risk — Main Pipeline
========================
Entry point for the full data processing pipeline.

Run with:
    python -m foundry.pipeline

What it does:
    1. Loads source configuration
    2. Runs enabled source processors
    3. Merges and deduplicates their instruction pairs
    4. Exports the final JSONL dataset
"""

from pathlib import Path

import pandas as pd
import yaml

from foundry.features.tabular_features import run_full_feature_engineering
from foundry.formatters import df_to_instruction_pairs, export_to_jsonl
from foundry.merger import merge_sources
from foundry.sources.bsa_regulations import build_bsa_dataset
from foundry.sources.sec_edgar import process_edgar_source
from foundry.sources.synthetic import process_synthetic_source
from foundry.sources.synthetic_generator import build_synthetic_dataset


def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    """
    Load the YAML config file and return it as a Python dict.
    All pipeline parameters come from here — nothing hardcoded below.
    """
    with open(config_path) as f:
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
            f"Raw data not found at '{raw_path}'.\nMake sure creditcard.csv is in data/raw/"
        )

    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} rows from {raw_path}")
    print(f"Columns: {list(df.columns)}")
    print(
        f"Fraud rate: {df['Class'].mean():.4%}  ({df['Class'].sum():,} fraud / {len(df):,} total)"
    )

    return df


def _enabled_sources(config: dict) -> list[str]:
    sources = config.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("Config must define a 'sources' mapping")
    enabled = [name for name, settings in sources.items() if settings.get("enabled", False)]
    if not enabled:
        raise ValueError("At least one source must be enabled")
    unknown = set(enabled).difference(
        {"fraud_transactions", "synthetic", "sec_edgar", "synthetic_scenarios", "bsa_regulations"}
    )
    if unknown:
        raise ValueError(f"Unsupported enabled sources: {', '.join(sorted(unknown))}")
    return enabled


def process_tabular_source(config: dict, df: pd.DataFrame) -> None:
    """Engineer and export instruction pairs for the tabular source."""
    output_path = config["sources"]["fraud_transactions"]["output_path"]
    print("\n--- Running tabular feature engineering ---")
    engineered = run_full_feature_engineering(df)
    pairs = df_to_instruction_pairs(engineered)
    export_to_jsonl(pairs, output_path)


def run_pipeline(config_path: str = "configs/pipeline_config.yaml") -> list[dict]:
    """
    Orchestrates the full pipeline end to end.
    Reads all parameters from config — nothing hardcoded here.
    """
    # Step 1: Load config
    config = load_config(config_path)
    enabled = _enabled_sources(config)
    real_df = None

    if {"fraud_transactions", "synthetic"}.intersection(enabled):
        raw_path = config["sources"]["fraud_transactions"]["raw_path"]
        print("\n--- Loading raw transaction data ---")
        real_df = load_raw_data(raw_path)

    if "fraud_transactions" in enabled:
        process_tabular_source(config, real_df)
    if "synthetic" in enabled:
        print("\n--- Running synthetic source ---")
        process_synthetic_source(config, real_df)
    if "sec_edgar" in enabled:
        print("\n--- Running SEC EDGAR source ---")
        process_edgar_source(config)
    if "synthetic_scenarios" in enabled:
        print("\n--- Running fraud scenario source ---")
        settings = config["sources"]["synthetic_scenarios"]
        build_synthetic_dataset(
            n_fraud=settings["n_fraud"],
            n_legit=settings["n_legit"],
            output_path=settings["output_path"],
            seed=settings.get("seed", config.get("pipeline", {}).get("seed", 42)),
        )
    if "bsa_regulations" in enabled:
        print("\n--- Running BSA regulation source ---")
        settings = config["sources"]["bsa_regulations"]
        build_bsa_dataset(
            date=settings.get("date", "2026-01-01"),
            output_path=settings["output_path"],
            variants_per_section=settings.get("variants_per_section", 4),
        )

    print("\n--- Merging enabled sources ---")
    pairs = merge_sources(config, source_names=enabled)
    print(f"\n✓ Pipeline complete: {len(pairs):,} merged pairs")
    return pairs


if __name__ == "__main__":
    run_pipeline()
