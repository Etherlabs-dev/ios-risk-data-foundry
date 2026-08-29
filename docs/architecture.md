# Architecture

## Purpose

IOS Risk Data Foundry prepares finance/risk domain data for reproducible evaluation and model-adaptation experiments. The architecture separates source acquisition, transformation, instruction formatting, dataset assembly, and evidence generation so each layer can be inspected independently.

## High-level flow

```text
Public transaction data      eCFR BSA text        Synthetic risk and AML cases
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

The config-driven entry point for enabled sources. It loads transaction data only when
required, invokes the official eCFR and synthetic sources, and passes only enabled outputs
to deterministic assembly. EDGAR remains explicitly disabled because its legacy answers
were not grounded in the retrieved filing text.

### `foundry/features/`

Domain feature engineering for transaction data. This layer is intentionally deterministic so the same input produces the same transformed representation.

### `foundry/sources/bsa_regulations.py`

Fetches or reuses cached official 31 CFR Chapter X XML. Answers are regulatory text with
citations; complete CFR sections are assigned wholly to train or evaluation.

### `foundry/sources/aml_typologies.py`

Produces account-level AML cases and documented benign counterexamples. Observable evidence
is rendered in the input; typology, tier, and action remain in the answer.

### `foundry/sources/synthetic.py` and `synthetic_generator.py`

Synthetic data generation for controlled fraud scenarios. These records provide coverage for explicit patterns but must never be treated as observed production traffic.

### `foundry/formatters.py`

Transforms structured rows into task-specific instruction/input/output records suitable for downstream supervised experiments.

### `foundry/merger.py`

Dataset assembly and deduplication logic.

### `foundry/versioning.py`

Produces fingerprints and version metadata for generated artifacts.

### `foundry/uploader.py`

Normalizes the public schema and publishes the exact export, dataset card, and manifest to
the configured Hugging Face dataset repository. It refuses unauthenticated publication.

### `scripts/build_finetune_v3.py` and `validate_finetune_v3.py`

Assemble the deterministic Project 03 export, create source-level holdouts, and fail closed
on malformed rows, duplicate prompts, unsupported model-score claims, or manifest drift.

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

Each source is enabled independently in `configs/pipeline_config.yaml`. Source provenance is
preserved on every v3 record. The final v3 assembly is deliberately separate from the legacy
merger because it also creates tabular source-record and regulatory section holdouts. A stale
output from a disabled source cannot be silently included.
