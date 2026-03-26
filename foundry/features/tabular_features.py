"""
IOS Risk — Tabular Feature Engineering
=======================================
Takes the raw creditcard.csv and adds domain-aware features.

WHY THIS FILE EXISTS:
  The raw dataset has V1-V28 (PCA columns), Time, Amount, and Class.
  The PCA columns are already encoded by whoever made the dataset — we
  can't interpret them. But Time and Amount are raw numbers that hide
  real fraud signals. This file extracts those signals explicitly.

  A model trained on raw numbers doesn't know that "17 transactions in
  1 hour" is suspicious. If we add a column txn_count_1h, it can learn
  that pattern directly. Feature engineering = encoding domain knowledge
  as numbers the model can use.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def engineer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many transactions occurred in rolling time windows.

    FRAUD SIGNAL: Card testing shows as many small transactions in a
    short window. Account takeover shows as burst activity after dormancy.

    NOTE: The Time column is seconds elapsed since the first transaction
    in the dataset — not a real timestamp. We use it for relative
    comparisons, which is all we need.
    """
    df = df.sort_values('Time').copy()

    # For each transaction, count how many others fall within ±1 hour
    # This is O(n²) on the full dataset — fine for 284k rows, but for
    # production you'd use a sliding window approach instead.
    df['txn_count_1h'] = df['Time'].transform(
        lambda t: ((df['Time'] - t).abs() < 3600).sum()
    )
    df['txn_count_24h'] = df['Time'].transform(
        lambda t: ((df['Time'] - t).abs() < 86400).sum()
    )

    # Is the amount escalating over time? (bust-out fraud pattern)
    df['amount_rolling_mean'] = df['Amount'].expanding().mean()
    df['amount_vs_mean'] = df['Amount'] / (df['amount_rolling_mean'] + 1e-8)

    return df


def engineer_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capture anomalies in transaction amounts.

    FRAUD SIGNALS:
    - Card testing: very small amounts ($0.01–$1) to check if card works
    - Authorization limit probing: round numbers just under limits ($99, $499)
    - Account takeover: unusually large amounts relative to history
    """
    df = df.copy()

    # Z-score: how far is this amount from the mean, in standard deviations?
    # A z-score of 3+ means the amount is extremely unusual.
    # We add 1e-8 to prevent division by zero on edge cases.
    mean_amt = df['Amount'].mean()
    std_amt  = df['Amount'].std()
    df['amount_zscore'] = (df['Amount'] - mean_amt) / (std_amt + 1e-8)

    # Round-number flag: fraud often uses round amounts for structuring
    df['is_round_amount'] = (df['Amount'] % 10 == 0).astype(int)

    # Micro-transaction flag: card testing pattern
    df['is_micro_txn'] = (df['Amount'] < 1.0).astype(int)

    # Large transaction flag: above the 95th percentile of amounts
    p95 = df['Amount'].quantile(0.95)
    df['is_large_txn'] = (df['Amount'] > p95).astype(int)

    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract meaningful signals from the Time column.

    FRAUD SIGNAL: Fraud disproportionately happens at 2–5am. Two reasons:
    1. Fraud detection teams are understaffed overnight.
    2. Victims are asleep and won't notice alerts immediately.

    Since Time is seconds-since-start (not a real clock), we convert it
    to a proxy hour-of-day by assuming the dataset starts at midnight.
    """
    df = df.copy()

    # Modulo 86400 gives seconds-within-a-day, divide by 3600 for hours
    df['hour_of_day'] = (df['Time'] % 86400 / 3600).astype(int)

    # Off-hours flag: 11pm–6am (hours 23, 0, 1, 2, 3, 4, 5, 6)
    off_hours = list(range(0, 7)) + [23]
    df['is_off_hours'] = df['hour_of_day'].isin(off_hours).astype(int)

    # Day of week proxy (86400 seconds/day, 7 days/week)
    df['day_of_week'] = (df['Time'] // 86400 % 7).astype(int)

    return df


def run_full_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering steps in the correct order.

    ORDER MATTERS: velocity features need the original Time column,
    so we run them before dropping it at the end.
    """
    print(f'Input shape: {df.shape}')
    print(f'Fraud rate before engineering: {df.Class.mean():.4%}')

    df = engineer_velocity_features(df)
    df = engineer_amount_features(df)
    df = engineer_time_features(df)

    # Drop original Time: it leaks ordering information that won't exist
    # at inference time (you won't know where in the dataset a new
    # transaction falls). The engineered features capture what mattered.
    df = df.drop(columns=['Time'])

    print(f'Output shape: {df.shape}')
    print(f'New features added: velocity (4), amount (4), time (3) = 11 total')
    print(f'Fraud rate after engineering: {df.Class.mean():.4%}  ← must be unchanged')

    return df


if __name__ == '__main__':
    # Quick smoke test — run this directly to verify it works
    raw_path = Path('data/raw/creditcard.csv')
    print(f'Loading {raw_path}...')
    df = pd.read_csv(raw_path)

    # Use a small sample first so this runs fast during development
    sample = df.sample(n=5000, random_state=42)
    engineered = run_full_feature_engineering(sample)

    print(f'\nSample of new columns:')
    new_cols = ['amount_zscore', 'is_round_amount', 'is_micro_txn',
                'is_large_txn', 'txn_count_1h', 'hour_of_day', 'is_off_hours']
    print(engineered[new_cols].describe())
