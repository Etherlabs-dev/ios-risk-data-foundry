# IOS Risk — Data Foundry v1

**Project 02 of the IntelligenceOS (IOS) Build Roadmap**
**Author: Ugo Chukwu · Etherlabs**

> The manufacturing plant for IOS Risk intelligence. Takes raw data from three sources — transaction records, regulatory filings, and synthetic scenarios — and outputs clean, versioned, training-ready datasets for fine-tuning domain-specific financial risk models.

---

## What This Is

The Data Foundry is the data infrastructure layer of [IntelligenceOS](https://github.com/Etherlabs-dev) — a domain-specific AI system for financial risk and fraud detection built from the ground up.

This repository implements a complete data engineering pipeline that:

- Ingests and feature-engineers the Kaggle Credit Card Fraud dataset (284k transactions)
- Scrapes and chunks SEC EDGAR 10-K regulatory filings into training-ready text
- Generates synthetic fraud scenarios with natural-language risk explanations
- Formats everything as Alpaca-style instruction pairs for LLM fine-tuning
- Exports to HuggingFace Hub as a versioned, reproducible dataset
- Validates that engineered features measurably improve the XGBoost fraud detection baseline

**The key result:** Engineered features improve `avg_precision` from `0.8510 → 0.8623` (+0.0114) and recall from `0.7857 → 0.8367` (+0.0510) over the base feature set.

---

## IntelligenceOS Build Roadmap Context

| Project | Component | Status |
|---------|-----------|--------|
| Project 01 | Eval Harness — XGBoost baseline (0.869 avg_precision) | Complete |
| **Project 02** | **Data Foundry v1 — this repo** | **Complete** |
| Project 03 | Domain Intelligence Core — LLM fine-tuning | Upcoming |
| Project 07 | RAG Knowledge System | Upcoming |
| Project 09 | Telemetry Layer | Upcoming |

Every dataset built here flows directly into Project 03's fine-tuning run.

---

## The Three Data Sources

### Source 1 — Tabular: Kaggle Credit Card Fraud
284,807 real credit card transactions with 11 engineered fraud-signal features:

| Feature | Signal |
|---------|--------|
| `txn_count_1h` / `txn_count_24h` | Velocity — card testing shows as burst activity |
| `amount_zscore` | Statistical anomaly — how unusual is this amount |
| `is_round_amount` | Structuring signal — round numbers near reporting thresholds |
| `is_micro_txn` | Card testing — tiny amounts ($0.01–$1) to verify stolen cards |
| `is_large_txn` | Account takeover — amounts above 95th percentile |
| `hour_of_day` | Time encoding |
| `is_off_hours` | Fraud peaks 11pm–6am (victims asleep, teams understaffed) |
| `day_of_week` | Weekly pattern encoding |
| `amount_rolling_mean` / `amount_vs_mean` | Bust-out pattern detection |

### Source 2 — Text: SEC EDGAR 10-K Filings
Regulatory risk language from public financial filings, fetched via the free SEC EDGAR full-text search API. Chunked into 512-word overlapping windows for LLM training.

Queries: fraud risk, AML compliance, credit default, cybersecurity breach, operational risk.

### Source 3 — Synthetic: Fraud Scenario Factory
10,000 generated records across 4 explicit fraud types, each with a natural-language `risk_explanation` field — teaching the model to *reason* about fraud, not just classify it:

| Fraud Type | Pattern | Risk Level |
|------------|---------|------------|
| Card Testing | 5–30 micro-transactions ($0.01–$2.00) in <1 hour | HIGH |
| Account Takeover | 1–5 large purchases ($200–$2,000) at off-hours | CRITICAL |
| Money Mule | Round amounts ($2k–$9.9k) just under BSA reporting threshold | HIGH |
| Bust-Out | 10–50 large transactions ($500–$5,000) in sudden burst | CRITICAL |

---

## Dataset

The final merged dataset is published on HuggingFace:

**[Etherlabs/ios-risk-finetune-v1](https://huggingface.co/datasets/Etherlabs/ios-risk-finetune-v1)**

```python
from datasets import load_dataset
ds = load_dataset("Etherlabs/ios-risk-finetune-v1")
```

**Stats:**
- 276,772 instruction pairs (after deduplication)
- Alpaca format: `{instruction, input, output}`
- Apache 2.0 license

---

## Repo Structure

```
ios-risk-data-foundry/
│
├── foundry/                         # Core pipeline code
│   ├── pipeline.py                  # Main orchestrator — run this
│   ├── formatters.py                # Converts rows to Alpaca instruction pairs
│   ├── merger.py                    # Combines, deduplicates, shuffles all sources
│   ├── uploader.py                  # HuggingFace Hub upload
│   ├── versioning.py                # SHA256 manifest + dataset versioning
│   ├── features/
│   │   └── tabular_features.py      # 11 fraud-signal feature functions
│   └── sources/
│       ├── sec_edgar.py             # SEC EDGAR 10-K ingestion pipeline
│       ├── synthetic.py             # Distribution-sampled synthetic transactions
│       └── synthetic_generator.py  # FraudScenarioFactory — 4 fraud types
│
├── scripts/
│   └── validate_features.py         # Eval harness validation (closes Project 1 loop)
│
├── tests/
│   └── test_features.py             # 6 pytest tests — run before every commit
│
├── configs/
│   └── pipeline_config.yaml         # All parameters in one place
│
├── data/                            # gitignored — tracked by DVC
│   ├── raw/                         # creditcard.csv lands here
│   ├── processed/                   # intermediate outputs
│   ├── versioned/                   # DVC snapshots
│   └── exports/                     # HuggingFace-ready JSONL
│
├── notebooks/                       # EDA notebooks (in progress)
├── dataset_manifest.json            # SHA256 fingerprints of all output files
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/Etherlabs-dev/ios-risk-data-foundry.git
cd ios-risk-data-foundry
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Add your data (not committed to git — too large)
mkdir -p data/raw
# Place creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# into data/raw/creditcard.csv

# 3. Run the full pipeline
python -m foundry.pipeline

# 4. Run tests
python -m pytest tests/ -v

# 5. Validate features against eval harness baseline
# (requires ios-risk-eval-harness cloned alongside this repo)
PYTHONPATH=../ios-risk-eval-harness:. python3 scripts/validate_features.py
```

---

## Reproduce Dataset Version v1.0

```bash
git checkout foundry-v1.0
python -m foundry.pipeline
```

The `dataset_manifest.json` contains SHA256 hashes of all output files — verify your reproduction matches the published dataset exactly.

---

## Instruction Pair Format

Each training record follows the Alpaca format:

**Fraud classification (tabular source):**
```json
{
  "instruction": "Classify this financial transaction as FRAUD or LEGITIMATE based on the features provided.",
  "input": "Amount: $149.62 | Hour: 0 | OffHours: 1 | MicroTxn: 0 | RoundAmt: 0 | LargeTxn: 0 | AmtZscore: 0.245 | TxnCount1h: 3963",
  "output": "LEGITIMATE"
}
```

**Fraud scenario (synthetic generator):**
```json
{
  "instruction": "You are IOS Risk, an AI system for financial risk assessment. Analyse the following transaction and assess its fraud risk.",
  "input": "Amount: $1674.34 | TxnCount1h: 4 | Hour: 4 | FraudType: account_takeover | RiskLevel: CRITICAL",
  "output": "CRITICAL RISK — ACCOUNT TAKEOVER DETECTED. Large transaction of $1,674.34 at 04:00 is inconsistent with normal account behaviour. Off-hours timing combined with unusually high amount suggests compromised credentials. Immediate account freeze recommended."
}
```

**Regulatory text (SEC EDGAR source):**
```json
{
  "instruction": "Identify the financial risk factors described in this regulatory filing excerpt.",
  "input": "The company faces material exposure to credit default risk...",
  "output": "This excerpt from a 10-K filing discusses risk factors related to: credit default risk fintech."
}
```

---

## Eval Results

Validation run comparing base features (V1-V28 + Amount) vs engineered features using XGBoost and the IOS Risk eval harness:

```
=======================================================
RESULTS COMPARISON
=======================================================
Metric                 Baseline       Base   Engineered
-------------------------------------------------------
avg_precision             0.869     0.8510     0.8623  (+0.0114)
roc_auc                       -     0.9747     0.9743  (-0.0004)
f1                            -     0.8415     0.8542  (+0.0126)
precision                     -     0.9059     0.8723  (-0.0335)
recall                        -     0.7857     0.8367  (+0.0510)
=======================================================
✓ Engineered features IMPROVED avg_precision by +0.0114
```

Higher recall (+0.051) means the model catches more actual fraud — the right trade-off for a fraud detection system where missing fraud is more costly than a false alert.

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Part of IntelligenceOS

This is Project 02 of the IntelligenceOS build — a public, end-to-end documentation of building a production-grade, domain-specific AI system for financial risk from scratch.

**Built by Ugo Chukwu · [Etherlabs](https://github.com/Etherlabs-dev)**
