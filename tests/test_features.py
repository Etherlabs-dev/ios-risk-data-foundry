import pandas as pd
import pytest

from foundry.features.tabular_features import (
    engineer_amount_features,
    engineer_time_features,
    engineer_velocity_features,
    run_full_feature_engineering,
)
from foundry.formatters import df_to_instruction_pairs
from foundry.sources.sec_edgar import chunk_text


@pytest.fixture
def sample_df():
    """
    A tiny, controlled dataset we build ourselves.
    We don't use the real CSV in tests — too slow, and we want
    predictable numbers we can assert against exactly.
    """
    return pd.DataFrame(
        {
            "Time": [0, 3600, 7200, 86400, 90000],
            "Amount": [0.50, 100.0, 200.0, 10.0, 500.0],
            "V1": [0.1, -0.2, 0.3, 0.0, 1.0],  # dummy PCA columns
            "Class": [1, 0, 0, 0, 1],  # 2 fraud, 3 legit = 40% fraud
        }
    )


def test_fraud_rate_preserved(sample_df):
    """Fraud rate must be identical before and after engineering."""
    fraud_rate_before = sample_df["Class"].mean()
    result = run_full_feature_engineering(sample_df)
    fraud_rate_after = result["Class"].mean()
    assert fraud_rate_before == fraud_rate_after, (
        f"Fraud rate changed: {fraud_rate_before:.4f} → {fraud_rate_after:.4f}"
    )


def test_amount_zscore_shape(sample_df):
    """
    engineer_amount_features must add exactly 4 new columns — no more, no fewer.
    The original columns must still be present.
    """
    original_cols = set(sample_df.columns)
    result = engineer_amount_features(sample_df)

    expected_new_cols = {"amount_zscore", "is_round_amount", "is_micro_txn", "is_large_txn"}
    actual_new_cols = set(result.columns) - original_cols

    assert actual_new_cols == expected_new_cols, (
        f"Expected new columns {expected_new_cols}, got {actual_new_cols}"
    )


def test_amount_zscore_is_finite_for_constant_values():
    frame = pd.DataFrame({"Amount": [5.0, 5.0]})
    result = engineer_amount_features(frame)

    assert result["amount_zscore"].tolist() == [0.0, 0.0]


def test_velocity_counts_are_past_only_with_open_lower_boundary():
    frame = pd.DataFrame(
        {
            "Time": [7200, 0, 3600, 3599],
            "Amount": [1.0, 1.0, 1.0, 1.0],
            "Class": [0, 0, 1, 0],
        }
    )

    result = engineer_velocity_features(frame)

    assert result["Time"].tolist() == [0, 3599, 3600, 7200]
    assert result["txn_count_1h"].tolist() == [1, 2, 2, 1]
    assert result["txn_count_24h"].tolist() == [1, 2, 3, 4]


def test_round_amount_flag(sample_df):
    """
    is_round_amount must be 1 for amounts divisible by 10, 0 otherwise.
    With our fixture: [0.50, 100.0, 200.0, 10.0, 500.0] → [0, 1, 1, 1, 1]
    """
    result = engineer_amount_features(sample_df)
    expected = [0, 1, 1, 1, 1]
    actual = result["is_round_amount"].tolist()

    assert actual == expected, f"Round amount flags wrong. Expected {expected}, got {actual}"


def test_time_features_off_hours(sample_df):
    """
    All 5 fixture rows fall in hours 0-2, which are off-hours (11pm-6am).
    is_off_hours must be 1 for all rows.
    """
    result = engineer_time_features(sample_df)
    expected = [1, 1, 1, 1, 1]
    actual = result["is_off_hours"].tolist()

    assert actual == expected, f"Off-hours flags wrong. Expected {expected}, got {actual}"


def test_formatter_output(sample_df):
    """
    Full pipeline: engineer features → format as instruction pairs.
    Checks count, required keys, and valid label values.
    """
    engineered = run_full_feature_engineering(sample_df)
    pairs = df_to_instruction_pairs(engineered)

    # One pair per row
    assert len(pairs) == len(sample_df)

    # Every pair has the three required keys
    for pair in pairs:
        assert set(pair.keys()) == {"instruction", "input", "output"}

    # Labels are always one of two valid values
    valid_labels = {"FRAUD", "LEGITIMATE"}
    for pair in pairs:
        assert pair["output"] in valid_labels, f"Unexpected label: {pair['output']}"


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(), "missing required columns"),
        (pd.DataFrame({"Time": [0], "Amount": [1.0]}), "Class"),
        (
            pd.DataFrame({"Time": [0], "Amount": [None], "Class": [0]}),
            "contains missing",
        ),
    ],
)
def test_feature_engineering_rejects_malformed_input(frame, message):
    with pytest.raises(ValueError, match=message):
        run_full_feature_engineering(frame)


def test_chunk_text():
    """
    chunk_text must:
    1. Split text into chunks of the right word count
    2. Create overlap between consecutive chunks
    3. Return empty list for empty input
    """
    # Build a predictable 20-word string
    words = [f"word{i}" for i in range(20)]
    text = " ".join(words)

    # chunk_size=10, overlap=2 → step=8
    chunks = chunk_text(text, chunk_size=10, overlap=2)

    # First chunk: words 0-9
    assert chunks[0] == " ".join(words[0:10])

    # Second chunk: words 8-17 — words 8,9 are the overlap
    assert chunks[1] == " ".join(words[8:18])

    # Empty input returns empty list
    assert chunk_text("") == []
    assert chunk_text("   ") == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunk_text_rejects_invalid_windows(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some filing text", chunk_size=chunk_size, overlap=overlap)
