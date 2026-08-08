# Architecture

## Purpose

IOS Risk Data Foundry prepares finance/risk domain data for reproducible evaluation and model-adaptation experiments. The architecture separates source acquisition, transformation, instruction formatting, dataset assembly, and evidence generation so each layer can be inspected independently.

## High-level flow

```text
Public transaction data      SEC EDGAR text       Synthetic risk scenarios
          │                       │                         │
          └───────────────┬───────┴───────────────┬─────────┘
                          │                       │
                  Source-specific parsing / normalization
                          │
                  Feature & text transformation
                          │
                  Instruction-pair formatting
                          │
                   Merge / deduplicate / shuffle
                          │
               Versioned output + SHA256 manifest
                          │
             Evaluation / model adaptation downstream
```

## Components

### `foundry/pipeline.py`

The config-driven entry point for enabled sources. It keeps SEC network access opt-in,
loads the raw transaction dataset only when required, invokes each enabled source processor,
and passes only those source outputs to the deterministic merger.

### `foundry/features/`

Domain feature engineering for transaction data. This layer is intentionally deterministic so the same input produces the same transformed representation.

### `foundry/sources/sec_edgar.py`

Utilities for acquiring and chunking public SEC filing text for finance/risk language tasks.

### `foundry/sources/synthetic.py` and `synthetic_generator.py`

Synthetic data generation for controlled fraud scenarios. These records provide coverage for explicit patterns but must never be treated as observed production traffic.

### `foundry/formatters.py`

Transforms structured rows into task-specific instruction/input/output records suitable for downstream supervised experiments.

### `foundry/merger.py`

Dataset assembly and deduplication logic.

### `foundry/versioning.py`

Produces fingerprints and version metadata for generated artifacts.

### `foundry/uploader.py`

Publishes prepared artifacts to the configured Hugging Face dataset repository.

### `scripts/validate_features.py`

Connects the data layer back to the evaluation harness to measure whether engineered features improve the selected baseline metrics.

## Design principles

### 1. Evidence before tuning

The pipeline exists to make downstream model experiments measurable. Fine-tuning is not assumed to be useful; base models, simpler ML systems, retrieval, and tuned variants should be compared on held-out evaluations.

### 2. Deterministic transformations where possible

Feature engineering, formatting, and versioning should be reproducible. Synthetic generation should use explicit seeds when exact reproduction matters.

### 3. Preserve provenance

Every final record should be traceable to a source class: public transaction data, public regulatory text, or synthetic generation. Production implementations should extend this with stronger lineage metadata at the individual-record level.

### 4. Separate public benchmark evidence from production claims

The architecture is designed for experimentation and reproducible evidence. Deployment claims require a separate production system, target data, access controls, and online monitoring.

## Multi-source orchestration boundary

The repository now has a small orchestration layer for the tabular, SEC, and statistical
synthetic paths. Each source is enabled independently in `configs/pipeline_config.yaml`.
The default remains tabular-only so a normal run does not unexpectedly make network requests
or mix synthetic examples into an export. The merger receives the enabled source names, which
prevents stale output from a disabled source from leaking into a new dataset.

The rule-based `synthetic_generator.py` scenario factory remains an explicit standalone artifact.
It emits explanatory scenarios rather than the same statistical source contract, so forcing it
into the main pipeline would obscure that evidence boundary. A future extension can add it as a
fourth named source once its distinct provenance is represented in each output record.
