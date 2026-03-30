import pandas as pd
import json
from pathlib import Path


def row_to_instruction_pair(row: pd.Series) -> dict:
    """
    Convert one engineered transaction row into an instruction-tuning dict.

    The Alpaca format (instruction / input / output) is the standard
    format for fine-tuning open-source LLMs like LLaMA and Mistral.
    """
    label = "FRAUD" if row['Class'] == 1 else "LEGITIMATE"

    input_text = (
        f"Amount: ${row['Amount']:.2f} | "
        f"Hour: {int(row['hour_of_day'])} | "
        f"OffHours: {int(row['is_off_hours'])} | "
        f"MicroTxn: {int(row['is_micro_txn'])} | "
        f"RoundAmt: {int(row['is_round_amount'])} | "
        f"LargeTxn: {int(row['is_large_txn'])} | "
        f"AmtZscore: {row['amount_zscore']:.3f} | "
        f"TxnCount1h: {int(row['txn_count_1h'])}"
    )

    return {
        "instruction": "Classify this financial transaction as FRAUD or LEGITIMATE based on the features provided.",
        "input": input_text,
        "output": label,
    }

def df_to_instruction_pairs(df: pd.DataFrame) -> list[dict]:
    """
    Apply row_to_instruction_pair across the full DataFrame.
    Returns a list of Alpaca-format dicts ready for JSONL export.
    """
    pairs = []
    for _, row in df.iterrows():
        pairs.append(row_to_instruction_pair(row))
    return pairs


def export_to_jsonl(pairs: list[dict], output_path: str) -> None:
    """
    Write instruction pairs to a JSONL file.
    Each line is one valid JSON object — the format HuggingFace expects.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"Exported {len(pairs)} records to {output_path}")
