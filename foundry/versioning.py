"""
IOS Risk — Dataset Versioning
==============================
Generates a manifest of all dataset files with record counts,
sizes and SHA256 hashes for reproducibility tracking.

WHY THIS EXISTS:
  Without versioning, you cannot answer: "which dataset produced
  which model?" The manifest is the provenance record that ties
  every training run to an exact dataset state. DVC then makes
  that state reproducible with a single command.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime


DATASET_FILES = [
    "data/exports/ios_risk_finetune_v1.jsonl",
    "data/processed/synthetic_transactions.jsonl",
    "data/processed/synthetic_scenario_pairs.jsonl",
    "data/processed/edgar_risk_chunks.jsonl",
]


def sha256_of_file(path: Path) -> str:
    """
    Compute the SHA256 hash of a file by reading it in chunks.
    Reading in chunks handles large files without loading them
    entirely into memory.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):   # 64KB at a time
            h.update(chunk)
    return h.hexdigest()


def count_records(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def generate_manifest(output_path: str = "dataset_manifest.json") -> None:
    """
    Scan all dataset files, compute hashes and record counts,
    write to dataset_manifest.json.
    """
    manifest = {
        "version":    "foundry-v1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "files":      {},
    }

    for file_path in DATASET_FILES:
        p = Path(file_path)

        if not p.exists():
            print(f"  Skipping {file_path} — not found")
            manifest["files"][file_path] = {"status": "missing"}
            continue

        print(f"  Hashing {file_path}...")
        sha256  = sha256_of_file(p)
        records = count_records(p)
        size_mb = p.stat().st_size / (1024 * 1024)

        manifest["files"][file_path] = {
            "records":    records,
            "size_mb":    round(size_mb, 3),
            "sha256":     sha256,
            "status":     "ok",
        }

        print(f"    {records:,} records | {size_mb:.1f} MB | {sha256[:12]}...")

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Manifest written to {output_path}")


if __name__ == "__main__":
    generate_manifest()
