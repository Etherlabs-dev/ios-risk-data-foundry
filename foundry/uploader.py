"""Publish a validated Foundry export and its provenance card to Hugging Face."""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi


def build_dataset_card(config: dict) -> str:
    """Return the public, evidence-bounded v3 dataset card."""
    hf_name = config["exports"]["hf_dataset_name"]
    return f"""---
language:
- en
license: apache-2.0
task_categories:
- text-classification
- question-answering
tags:
- finance
- fraud-detection
- aml
- risk
- instruction-tuning
size_categories:
- 10K<n<100K
---

# IOS Risk Fine-Tune Dataset v3

Quality-gated instruction-tuning data for financial fraud, AML typologies, and
Bank Secrecy Act regulatory recall. This is a research dataset assembled from
public data, official public regulations, deterministic synthetic scenarios,
and validated model-assisted rewrites. It is not production transaction evidence.

## Composition

| Source | Records | Description |
|---|---:|---|
| Public tabular benchmark | 9,242 | ULB/Kaggle credit-card examples; record holdout enforced |
| AML typology cases | 6,000 | Six typologies plus documented benign counterexamples |
| Fraud scenarios | 5,000 | Card testing, takeover, mule, bust-out, and benign cases |
| eCFR BSA regulations | 364 | Official 31 CFR Chapter X text; whole sections held out |

The AML set includes 2,462 strictly validated rewrites produced with
`nvidia/nemotron-3-super-120b-a12b`; the remaining AML answers use deterministic
templates. Generated records were validated before inclusion.

## Format

```json
{{
  "instruction": "Classify this transaction as FRAUD or LEGITIMATE.",
  "input": "Amount: $149.62 | Hour: 0 | OffHours: 1 | ...",
  "output": "LEGITIMATE",
  "source": "ulb_tabular_train"
}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("Etherlabs/{hf_name}")
```

## Quality and leakage controls

- 20,606 unique instruction/input pairs
- source-record hash partition for tabular train/test separation
- whole-section partition for regulatory train/test separation
- malformed, duplicate, fallback, and unsupported-number rewrite rejection
- independently authored counterfactual risk evaluation cases kept outside training

## Limitations

- Most risk scenarios are synthetic and cover a bounded set of patterns.
- Regulatory pairs teach recall of text and citations, not legal judgement.
- The tabular benchmark lacks rich device, merchant, customer-history, and graph features.
- Do not use a downstream model as an autonomous fraud, account-restriction, or
  regulatory-filing decision-maker.
- Compare any fine-tune with its untouched base model on the held-out Project 03
  evaluation before making an improvement claim.
"""


def normalized_records(export_path: str) -> list[dict[str, str]]:
    """Keep a stable Hub schema even though local provenance fields vary by source."""
    records: list[dict[str, str]] = []
    with open(export_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                {
                    "instruction": row["instruction"],
                    "input": row["input"],
                    "output": row["output"],
                    "source": row.get("source", "unknown"),
                }
            )
    return records


def upload_to_huggingface(config: dict) -> None:
    """Upload the v3 export, card, and exact manifest using HF_TOKEN."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; refusing an unauthenticated publication")

    hf_name = config["exports"]["hf_dataset_name"]
    repo_id = f"Etherlabs/{hf_name}"
    export_path = config["exports"]["instruction_pairs_path"]
    records = normalized_records(export_path)
    print(f"Loaded {len(records):,} records from {export_path}")

    dataset = Dataset.from_list(records)
    dataset.push_to_hub(repo_id, private=False, token=token)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=build_dataset_card(config).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    manifest_path = Path(f"{export_path}.manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing export manifest: {manifest_path}")
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="dataset_manifest.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Published and documented https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import yaml

    with open("configs/pipeline_config.yaml", encoding="utf-8") as config_file:
        upload_to_huggingface(yaml.safe_load(config_file))
