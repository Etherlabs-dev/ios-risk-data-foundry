"""
IOS Risk — Dataset Merger
=========================
Combines all source JSONL files into one final training dataset.

WHY THIS EXISTS:
  Each source pipeline writes its own JSONL file independently.
  The merger is the final step — it loads all sources, removes
  duplicates, shuffles, and writes the single file that gets
  uploaded to HuggingFace for fine-tuning.
"""
import json
import random
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    """
    Load a JSONL file into a list of dicts.
    Returns empty list if the file doesn't exist — so the merger
    can run even if some source pipelines haven't been run yet.
    """
    p = Path(path)

    if not p.exists():
        print(f"  Skipping {path} — file not found")
        return []

    pairs = []
    with open(p, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    print(f"  Loaded {len(pairs):,} records from {path}")
    return pairs


def deduplicate(pairs: list[dict]) -> list[dict]:
    """
    Remove exact duplicate records using the full JSON string as the key.
    Order is preserved — first occurrence of each record is kept.
    """
    seen    = set()
    unique  = []

    for pair in pairs:
        key = json.dumps(pair, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    removed = len(pairs) - len(unique)
    print(f"  Deduplicated: {len(pairs):,} → {len(unique):,} ({removed:,} duplicates removed)")
    return unique


def merge_sources(config: dict) -> None:
    """
    Load all source JSONL files, deduplicate, shuffle and write
    the final combined training dataset.
    """
    export_path = config['exports']['instruction_pairs_path']

    print("\n--- Loading sources ---")
    all_pairs = []
    all_pairs += load_jsonl(config['sources']['fraud_transactions']['output_path'])
    all_pairs += load_jsonl(config['sources']['synthetic']['output_path'])
    all_pairs += load_jsonl(config['sources']['sec_edgar']['output_path'])

    print(f"\n  Total before dedup : {len(all_pairs):,}")

    print("\n--- Deduplicating ---")
    all_pairs = deduplicate(all_pairs)

    print("\n--- Shuffling ---")
    random.seed(42)
    random.shuffle(all_pairs)
    print(f"  {len(all_pairs):,} records shuffled")

    print("\n--- Exporting ---")
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + '\n')

    # Summary
    fraud = sum(1 for p in all_pairs if p.get('output') == 'FRAUD')
    legit = sum(1 for p in all_pairs if p.get('output') == 'LEGITIMATE')
    text  = len(all_pairs) - fraud - legit

    print(f"\n✓ Merge complete")
    print(f"  Total records  : {len(all_pairs):,}")
    print(f"  FRAUD          : {fraud:,}")
    print(f"  LEGITIMATE     : {legit:,}")
    print(f"  Text/EDGAR     : {text:,}")
    print(f"  Output file    : {export_path}")
