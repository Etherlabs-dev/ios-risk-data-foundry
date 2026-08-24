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
    with open(p, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON in {path} at line {line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
                pairs.append(record)

    print(f"  Loaded {len(pairs):,} records from {path}")
    return pairs


def deduplicate(pairs: list[dict]) -> list[dict]:
    """
    Remove exact duplicate records using the full JSON string as the key.
    Order is preserved — first occurrence of each record is kept.
    """
    seen = set()
    unique = []

    for pair in pairs:
        key = json.dumps(pair, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    removed = len(pairs) - len(unique)
    print(f"  Deduplicated: {len(pairs):,} → {len(unique):,} ({removed:,} duplicates removed)")
    return unique


def merge_sources(config: dict, source_names: list[str] | None = None) -> list[dict]:
    """
    Load all source JSONL files, deduplicate, shuffle and write
    the final combined training dataset.
    """
    export_path = config["exports"]["instruction_pairs_path"]
    if source_names is None:
        source_names = ["fraud_transactions", "synthetic_scenarios", "bsa_regulations", "synthetic"]

    print("\n--- Loading sources ---")
    all_pairs = []
    for source_name in source_names:
        try:
            source_path = config["sources"][source_name]["output_path"]
        except KeyError as exc:
            raise ValueError(
                f"Missing output_path configuration for source: {source_name}"
            ) from exc
        all_pairs.extend(load_jsonl(source_path))

    print(f"\n  Total before dedup : {len(all_pairs):,}")

    print("\n--- Deduplicating ---")
    all_pairs = deduplicate(all_pairs)

    print("\n--- Shuffling ---")
    seed = config.get("pipeline", {}).get("seed", 42)
    random.Random(seed).shuffle(all_pairs)
    print(f"  {len(all_pairs):,} records shuffled")

    print("\n--- Exporting ---")
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # Summary
    fraud = sum(1 for p in all_pairs if p.get("output") == "FRAUD")
    legit = sum(1 for p in all_pairs if p.get("output") == "LEGITIMATE")
    text = len(all_pairs) - fraud - legit

    print("\n✓ Merge complete")
    print(f"  Total records  : {len(all_pairs):,}")
    print(f"  FRAUD          : {fraud:,}")
    print(f"  LEGITIMATE     : {legit:,}")
    print(f"  Text/EDGAR     : {text:,}")
    print(f"  Output file    : {export_path}")
    return all_pairs
