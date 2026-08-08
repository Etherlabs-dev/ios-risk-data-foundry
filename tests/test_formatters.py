import json

import pandas as pd
import pytest

from foundry.formatters import export_to_jsonl, row_to_instruction_pair


def formatted_row(**overrides):
    values = {
        "Amount": 12.345,
        "Class": 1,
        "amount_zscore": 1.23456,
        "hour_of_day": 3,
        "is_large_txn": 0,
        "is_micro_txn": 0,
        "is_off_hours": 1,
        "is_round_amount": 0,
        "txn_count_1h": 7,
    }
    values.update(overrides)
    return pd.Series(values)


def test_instruction_format_is_stable_and_explicit():
    pair = row_to_instruction_pair(formatted_row())

    assert pair == {
        "instruction": (
            "Classify this financial transaction as FRAUD or LEGITIMATE based on the "
            "features provided."
        ),
        "input": (
            "Amount: $12.35 | Hour: 3 | OffHours: 1 | MicroTxn: 0 | RoundAmt: 0 | "
            "LargeTxn: 0 | AmtZscore: 1.235 | TxnCount1h: 7"
        ),
        "output": "FRAUD",
    }


@pytest.mark.parametrize("class_value", [-1, 2, "fraud"])
def test_instruction_formatter_rejects_invalid_labels(class_value):
    with pytest.raises(ValueError, match="Class must be either"):
        row_to_instruction_pair(formatted_row(Class=class_value))


def test_instruction_formatter_rejects_missing_or_null_values():
    with pytest.raises(ValueError, match="missing columns"):
        row_to_instruction_pair(formatted_row().drop(labels=["txn_count_1h"]))
    with pytest.raises(ValueError, match="missing values"):
        row_to_instruction_pair(formatted_row(Amount=None))


def test_export_to_jsonl_round_trips_utf8(tmp_path):
    output = tmp_path / "pairs.jsonl"
    pairs = [{"instruction": "Analyse", "input": "£10", "output": "LEGITIMATE"}]

    export_to_jsonl(pairs, output)

    assert json.loads(output.read_text(encoding="utf-8")) == pairs[0]
