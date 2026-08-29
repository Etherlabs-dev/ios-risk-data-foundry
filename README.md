# IOS Risk Data Foundry

> **Status: validated research prototype and reproducible public-data benchmark.**
> This repository does not claim production fraud-loss reduction, client impact,
> or autonomous compliance decision-making.

The Foundry is Project 02 of IntelligenceOS. It turns public transaction data,
official BSA regulations, and deterministic synthetic scenarios into versioned
instruction data for Project 03 model adaptation.

## Verified artifacts

| Artifact | Verified state |
|---|---|
| Feature benchmark | AP `0.8510 → 0.8588`; recall `0.7857 → 0.8367` |
| Legacy dataset | `Etherlabs/ios-risk-finetune-v1`; retained for lineage, unsuitable for Project 03 reasoning |
| Current dataset | [`Etherlabs/ios-risk-finetune-v3`](https://huggingface.co/datasets/Etherlabs/ios-risk-finetune-v3); 20,606 quality-gated pairs, published and independently re-downloaded |
| v3 fingerprint | `485f02df11b2e1dd4b1dbe0bb4dd9a68615735bbcf64cc7fbbb08933008ca075` |
| Automated checks | Ruff, pytest, manifest/hash verification, duplication and unsupported-claim gates |

## v3 composition

| Source | Records | Purpose |
|---|---:|---|
| ULB/Kaggle tabular benchmark | 9,242 | Exact FRAUD/LEGITIMATE classification |
| AML typology cases | 6,000 | Account-level reasoning and documented hard negatives |
| Fraud scenarios | 5,000 | Card testing, takeover, money mule, bust-out, and benign cases |
| eCFR BSA regulations | 364 | Recall of official 31 CFR Chapter X language and citations |

Of the AML cases, 2,462 use strictly validated rewrites produced with
`nvidia/nemotron-3-super-120b-a12b`; 3,538 use deterministic templates. The
original 3,355-record generation checkpoint is preserved. Malformed results,
fallbacks, duplicates, unsupported numbers, and legacy prompts with missing
evidence were rejected before inclusion.

## Leakage controls

- Every training instruction/input pair is unique.
- Tabular evaluation records are selected from a SHA-256 source-record partition
  that is excluded from training.
- Regulatory evaluation holds out complete CFR sections, not alternative
  phrasings of sections seen during training.
- Counterfactual risk-assessment cases are authored independently of the
  training generators.
- The committed manifest records the exact dataset fingerprint and composition.

## Sources and boundaries

- **Transaction classification:** the public, anonymized ULB/Kaggle credit-card
  fraud benchmark. It does not contain merchant identity, customer history,
  device fingerprints, or account-network graphs.
- **Regulatory recall:** official eCFR text for 31 CFR Chapter X. This teaches
  what a cited rule says; it does not confer legal judgement.
- **Risk scenarios and AML cases:** synthetic cases with the evidence needed by
  their expected answer included in the prompt. These are controlled research
  fixtures, not real customer transactions.
- **SEC EDGAR:** disabled. The old path attached query-templated answers to
  retrieved text and was not safe training evidence.

## Reproduce v3

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

PYTHONPATH=. python -m foundry.sources.bsa_regulations
PYTHONPATH=. python scripts/audit_distillation.py \
  --source data/processed/aml_typology_pairs.jsonl \
  --checkpoint data/processed/aml_sample_distilled.jsonl \
  --out data/processed/aml_distilled_validated.jsonl
PYTHONPATH=. python scripts/build_finetune_v3.py
PYTHONPATH=. python scripts/validate_finetune_v3.py
```

The tabular build reads `Etherlabs/ios-risk-finetune-v1` only to recover the
public benchmark records and applies a deterministic train/test partition. Raw,
processed, and exported data remain outside Git; the exact compact manifest is
committed as [`dataset_manifest_v3.json`](dataset_manifest_v3.json).

## Tests

```bash
ruff check .
ruff format --check .
pytest -q
```

## Repository map

```text
foundry/
  sources/aml_typologies.py       account-level AML cases
  sources/bsa_regulations.py      official eCFR regulatory pairs
  sources/synthetic_generator.py  grounded fraud scenarios
  pipeline.py                     config-driven source orchestration
  uploader.py                     Hugging Face publication and card
scripts/
  audit_distillation.py           strict checkpoint compaction
  build_finetune_v3.py            deterministic v3 build and holdouts
  validate_finetune_v3.py         release-blocking content/hash checks
docs/                              architecture, evidence, and failure modes
tests/                             deterministic unit and regression tests
```

See [`docs/evidence.md`](docs/evidence.md),
[`docs/benchmark-reproduction.md`](docs/benchmark-reproduction.md), and
[`docs/failure-modes.md`](docs/failure-modes.md) for the evidence policy and
benchmark limitations.

## What v3 does not prove

- that a fine-tuned LLM beats its untouched base model;
- that an LLM beats the Project 01 XGBoost classifier;
- robust generalization to unseen fraud families or institutions;
- calibrated fraud probabilities or production decision thresholds;
- production compliance, financial, or customer impact.

Those questions belong to Project 03's held-out base-versus-tuned evaluation.
