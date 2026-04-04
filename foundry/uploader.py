"""
IOS Risk — HuggingFace Dataset Uploader
========================================
Uploads the final merged JSONL to HuggingFace Hub as a versioned dataset.

WHY THIS EXISTS:
  The merged JSONL on disk is only useful locally. Uploading to HuggingFace
  makes the dataset versioned, shareable, and loadable anywhere with:
      from datasets import load_dataset
      ds = load_dataset("Etherlabs/ios-risk-finetune-v1")
"""
from datasets import Dataset
from huggingface_hub import HfApi
from pathlib import Path
import json


def build_dataset_card(config: dict) -> str:
    """
    Generate a HuggingFace dataset card (README.md content).
    The YAML frontmatter is parsed by HuggingFace to populate
    search filters and dataset metadata on the Hub.
    """
    hf_name = config['exports']['hf_dataset_name']

    card = f"""---
language:
- en
license: apache-2.0
task_categories:
- text-classification
tags:
- finance
- fraud-detection
- risk
- instruction-tuning
- alpaca
size_categories:
- 100K<n<1M
---

# IOS Risk Fine-Tune Dataset v1

Instruction-tuning dataset for financial risk and fraud detection.
Built by the IOS Risk Data Foundry pipeline.

## Sources

| Source | Records | Description |
|--------|---------|-------------|
| Credit Card Fraud (real) | ~284k | Kaggle creditcard.csv with engineered features |
| Synthetic Transactions | ~10k | Statistically sampled from real distributions |
| SEC EDGAR 10-K Filings | varies | Regulatory risk language from public filings |

## Format

Alpaca instruction format — three fields per record:

```json
{{
  "instruction": "Classify this financial transaction as FRAUD or LEGITIMATE based on the features provided.",
  "input": "Amount: $149.62 | Hour: 0 | OffHours: 1 | ...",
  "output": "LEGITIMATE"
}}
```

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Etherlabs/{hf_name}")
```

## Features Engineered

- `amount_zscore` — how unusual the amount is vs dataset mean
- `is_round_amount` — round number flag (structuring signal)
- `is_micro_txn` — micro transaction flag (card testing signal)
- `is_large_txn` — above 95th percentile flag
- `txn_count_1h` / `txn_count_24h` — velocity features
- `hour_of_day` / `is_off_hours` — time-based fraud signals
"""
    return card


def upload_to_huggingface(config: dict) -> None:
    """
    Load the final merged JSONL, convert to HuggingFace Dataset,
    push to Hub, then upload the dataset card.
    """
    hf_name     = config['exports']['hf_dataset_name']
    repo_id     = f"Etherlabs/{hf_name}"
    export_path = config['exports']['instruction_pairs_path']

    # Step 1: load the merged JSONL into memory
    print(f"Loading {export_path}...")
    records = []
    with open(export_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records):,} records")

    # Step 2: convert to HuggingFace Dataset
    print("Converting to HuggingFace Dataset format...")
    dataset = Dataset.from_list(records)
    print(f"Dataset schema: {dataset.features}")

    # Step 3: push dataset to Hub
    print(f"Pushing to {repo_id}...")
    dataset.push_to_hub(
        repo_id,
        private=False,
        token=None,
    )
    print(f"Dataset pushed successfully")

    # Step 4: upload the dataset card
    print("Uploading dataset card...")
    api  = HfApi()
    card = build_dataset_card(config)
    api.upload_file(
        path_or_fileobj=card.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"\n✓ Upload complete")
    print(f"  Dataset : https://huggingface.co/datasets/{repo_id}")
    print(f"  Records : {len(records):,}")
    print(f"  Load with: dataset = load_dataset('{repo_id}')")


if __name__ == "__main__":
    import yaml
    config = yaml.safe_load(open('configs/pipeline_config.yaml'))
    upload_to_huggingface(config)
