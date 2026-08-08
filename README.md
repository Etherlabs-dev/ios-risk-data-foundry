# IOS Risk Data Foundry

> **Status: Validated Prototype + Reproducible Benchmark**  
> Domain-data engineering pipeline for finance/risk model development. Uses public datasets, public regulatory filings, and synthetic scenarios. This repository is **not presented as a production client deployment**.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Tests](https://img.shields.io/badge/tests-pytest-informational)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

The Data Foundry is the data infrastructure layer of **IntelligenceOS**, a domain AI system for Finance & Risk. It prepares structured and unstructured financial-risk data for evaluation, model adaptation, and downstream fine-tuning experiments.

It is designed to answer a specific engineering question:

> **Can domain data be prepared, versioned, tested, and measured well enough that model improvements are reproducible rather than anecdotal?**

---

## Evidence Summary

| Claim | Evidence type | Current evidence |
|---|---|---|
| Feature engineering improves fraud-detection recall | **Benchmarked** | Recall `0.7857 → 0.8367` on the reproduced validation run |
| Engineered features improve average precision vs. the base feature set | **Benchmarked** | `0.8510 → 0.8588` (+0.0079) |
| Pipeline produces training-ready instruction pairs | **Implemented / tested** | Formatter + pipeline code with pytest coverage |
| Dataset outputs are versionable and fingerprinted | **Implemented** | DVC configuration + SHA256 dataset manifest |
| Domain dataset is published for reuse | **Published artifact** | Hugging Face dataset: `Etherlabs/ios-risk-finetune-v1` |
| Production financial impact | **Not claimed** | No client revenue, loss-prevention, or production ROI claim is made here |

See [`docs/evidence.md`](docs/evidence.md) for the evidence policy and interpretation of these results.

---

## What the System Does

The repository contains three domain-data preparation paths:

1. **Transaction data** — feature engineering over the public Kaggle credit-card fraud dataset.
2. **Regulatory text** — SEC EDGAR 10-K retrieval and chunking for finance/risk language.
3. **Synthetic risk scenarios** — controlled fraud scenarios with explicit risk labels and natural-language explanations.

The outputs are prepared for evaluation and model-adaptation work inside IntelligenceOS.

### Current implemented components

- Config-driven tabular ingestion pipeline
- Financial/fraud feature engineering
- Alpaca-style instruction formatting
- SEC EDGAR text ingestion/chunking utilities
- Synthetic fraud scenario generation
- Dataset merge/deduplication utilities
- SHA256 dataset manifest/versioning
- Hugging Face upload tooling
- Feature validation against the IOS Risk evaluation harness
- Pytest unit tests for core feature and formatting behavior

---

## Architecture

```text
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ Public transaction   │   │ SEC EDGAR filings    │   │ Synthetic scenarios  │
│ data                 │   │ public risk language │   │ controlled labels    │
└──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
           │                          │                          │
           └──────────────┬───────────┴──────────────┬───────────┘
                          │                          │
                 ┌────────▼─────────┐       ┌────────▼─────────┐
                 │ Cleaning /       │       │ Feature / text   │
                 │ normalization    │       │ transformation   │
                 └────────┬─────────┘       └────────┬─────────┘
                          └────────────┬─────────────┘
                                       │
                              ┌────────▼────────┐
                              │ Instruction     │
                              │ pair formatting │
                              └────────┬────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │ Merge · dedupe · version │
                         │ manifest / DVC           │
                         └─────────────┬─────────────┘
                                       │
                      ┌────────────────▼────────────────┐
                      │ Evaluation / model adaptation   │
                      │ IntelligenceOS downstream work  │
                      └─────────────────────────────────┘
```

A more detailed engineering view is in [`docs/architecture.md`](docs/architecture.md).

---

## Dataset Sources

### 1. Tabular — Kaggle Credit Card Fraud

The tabular path uses the public credit-card fraud dataset containing **284,807 transactions**. The pipeline adds domain-oriented features such as:

- trailing dataset-density proxies (`txn_count_1h`, `txn_count_24h`)
- amount anomaly (`amount_zscore`)
- round-amount detection
- micro-transaction detection
- large-transaction detection
- hour/day encodings
- off-hours activity
- rolling amount behavior

The purpose is not to claim that these features are universally optimal. Their value is measured against the existing baseline and kept only when the evaluation evidence supports them.

### 2. Unstructured — SEC EDGAR 10-K filings

The repository includes utilities to retrieve and chunk public regulatory filings around finance/risk topics such as fraud, AML, credit default, cybersecurity, and operational risk.

### 3. Synthetic — Fraud Scenario Factory

Synthetic examples provide controlled coverage for fraud patterns that are useful for training/evaluation experiments, including card testing, account takeover, money-mule patterns, and bust-out behavior.

Synthetic data is always treated as **synthetic evidence**, not as production transaction history.

---

## Reproducible Benchmark

A documented validation run compared the base feature set with the engineered feature set using the IOS Risk evaluation harness.

| Metric | Base | Engineered | Delta |
|---|---:|---:|---:|
| Average precision | 0.8510 | **0.8588** | **+0.0079** |
| ROC AUC | 0.9747 | **0.9803** | **+0.0055** |
| F1 | 0.8415 | **0.8586** | **+0.0171** |
| Precision | **0.9059** | 0.8817 | -0.0242 |
| Recall | 0.7857 | **0.8367** | **+0.0510** |

The important trade-off is explicit: recall improved while precision declined. In a fraud-risk context, whether that trade-off is acceptable depends on the operational cost of false positives versus missed fraud. This repository does **not** treat a single metric increase as proof of production superiority.

These figures were independently rerun on 2026-08-08 with the public Kaggle dataset,
the sibling evaluation harness at commit `5144f8c`, and the pinned validation dependencies.
See [`docs/benchmark-reproduction.md`](docs/benchmark-reproduction.md) for dataset fingerprints,
commands, environment, and methodological limitations.

---

## Repository Structure

```text
ios-risk-data-foundry/
├── .github/workflows/        # CI checks
├── configs/
│   └── pipeline_config.yaml
├── docs/
│   ├── architecture.md
│   ├── evidence.md
│   └── failure-modes.md
├── foundry/
│   ├── features/
│   ├── sources/
│   │   ├── sec_edgar.py
│   │   ├── synthetic.py
│   │   └── synthetic_generator.py
│   ├── formatters.py
│   ├── merger.py
│   ├── pipeline.py
│   ├── uploader.py
│   └── versioning.py
├── scripts/
│   └── validate_features.py
├── tests/
│   └── test_features.py
├── dataset_manifest.json
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and create an environment

```bash
git clone https://github.com/Etherlabs-dev/ios-risk-data-foundry.git
cd ios-risk-data-foundry
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add the public fraud dataset

Place `creditcard.csv` in:

```text
data/raw/creditcard.csv
```

The raw dataset is intentionally not committed to Git.

### 3. Run the tabular pipeline

```bash
python -m foundry.pipeline
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

### 5. Validate engineered features against the evaluation harness

Clone `eval-harness` alongside this repository and run:

```bash
pip install -r requirements-validation.txt
PYTHONPATH=../eval-harness/ios-risk-eval-harness:. python scripts/validate_features.py
```

---

## Docker

The included Dockerfile provides a reproducible Python environment for tests and development checks.

```bash
docker build -t ios-risk-data-foundry .
docker run --rm ios-risk-data-foundry
```

The default container command runs the unit tests. Running the full data pipeline still requires the external dataset to be mounted or provided separately.

---

## CI

GitHub Actions runs Ruff lint/format checks, the unit test suite, and a Docker image build plus
the container test command on pushes and pull requests. CI intentionally avoids external dataset
downloads and SEC network calls so the required jobs remain deterministic.

---

## Reliability & Failure Modes

The pipeline is intentionally explicit about where reproducibility can break:

- upstream public-source availability
- schema drift in source data
- distribution shift between public/synthetic data and real production traffic
- duplicate or malformed records
- non-deterministic synthetic generation if seeds are not fixed
- label leakage or class imbalance
- model-selection bias from repeatedly optimizing against one validation set

See [`docs/failure-modes.md`](docs/failure-modes.md) for the mitigation checklist.

---

## Data Governance & Security

This repository uses public datasets, public filings, and generated synthetic records. It does not require client PII.

For production adaptation, the design should be extended with:

- explicit data provenance and consent/permission records
- PII classification and minimization
- encryption at rest/in transit
- access controls and least privilege
- dataset retention/deletion policy
- reproducible train/validation/test splits
- model and dataset lineage
- audit logs for data transformations

These controls are part of the production standard for IntelligenceOS but should not be inferred as implemented here unless they are present in code/configuration.

---

## Published Dataset

The current dataset artifact is published on Hugging Face as:

**`Etherlabs/ios-risk-finetune-v1`**

Example:

```python
from datasets import load_dataset

ds = load_dataset("Etherlabs/ios-risk-finetune-v1")
```

The repository's `dataset_manifest.json` provides content fingerprints for versioned outputs.

---

## Limitations

This project does **not** prove:

- production fraud-loss reduction
- production model lift on a financial institution's live traffic
- client ROI
- generalization to every fraud/risk domain
- that an LLM fine-tuned on these outputs will outperform simpler models or RAG

Those require separate held-out evaluation and production validation.

---

## IntelligenceOS Context

| Project | Component | Status |
|---|---|---|
| Project 01 | Eval Harness | Complete |
| **Project 02** | **Data Foundry v1** | **Complete / validated prototype** |
| Project 03 | Domain Intelligence Core / model adaptation | Next stage |

The next model-adaptation stage should compare base models, retrieval, and tuned variants on a held-out evaluation set rather than assuming fine-tuning is the best option.

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

---

**Built by Ugo Chukwu · Etherlabs**  
Part of the public IntelligenceOS engineering portfolio for production AI + financial systems.
