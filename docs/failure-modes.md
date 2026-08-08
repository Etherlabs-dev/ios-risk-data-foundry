# Failure Modes & Reliability Checklist

This document records the main ways a domain-data pipeline like IOS Risk Data Foundry can produce misleading, incomplete, or non-reproducible outputs.

## Source failures

### Upstream availability

**Risk:** SEC endpoints, external dataset hosts, or other public sources can be unavailable or rate-limited.

**Mitigation:**
- cache source artifacts when licensing permits;
- retry with bounded backoff;
- fail loudly rather than silently dropping a source;
- record acquisition timestamps and source versions.

### Schema drift

**Risk:** upstream fields or formats change.

**Mitigation:**
- validate required columns/keys before transformation;
- add schema-contract tests;
- version parsers independently when source contracts change.

## Data-quality failures

### Duplicate records

**Risk:** repeated records contaminate train/evaluation splits and inflate apparent performance.

**Mitigation:**
- deterministic deduplication;
- unique content fingerprints;
- split after deduplication;
- test for cross-split overlap.

### Label leakage

**Risk:** engineered features or text directly expose the target label.

**Mitigation:**
- review feature provenance;
- remove post-outcome information;
- run leakage checks before benchmarking;
- document every engineered feature.

### Class imbalance

**Risk:** high headline accuracy can hide poor fraud recall.

**Mitigation:**
- prefer precision/recall, average precision, F1, and cost-aware metrics over accuracy alone;
- inspect per-class errors;
- avoid resampling evaluation data.

## Synthetic-data failures

### Unrealistic synthetic patterns

**Risk:** generated scenarios are easier or more stereotyped than real-world events.

**Mitigation:**
- label synthetic records explicitly;
- measure performance separately by source;
- never use synthetic-only performance as production evidence;
- validate against real public or permissioned data.

### Non-reproducible generation

**Risk:** repeated runs produce materially different datasets.

**Mitigation:**
- expose random seeds in configuration;
- record generator configuration;
- fingerprint outputs.

## Evaluation failures

### Validation-set overfitting

**Risk:** repeatedly tuning features against the same validation set produces optimistic results.

**Mitigation:**
- freeze a final held-out test set;
- keep experiment logs;
- minimize repeated decisions using the final test split.

The current feature-validation script is exploratory: it chooses the F1 threshold on the test split
and derives some unsupervised statistics before splitting. A production-grade benchmark should add
a validation split, fit preprocessing on training data only, freeze the threshold, and report the
test split once.

### Metric selection bias

**Risk:** one improved metric is presented as overall system improvement while another degrades.

**Mitigation:**
- report the full metric set;
- document trade-offs;
- tie thresholds to operational costs.

The current documented benchmark is a good example: recall increases while precision decreases. Both must be visible.

## Distribution shift

**Risk:** public fraud data, SEC language, or synthetic examples do not match a target institution's production traffic.

**Mitigation:**
- evaluate on representative target-domain data before deployment;
- compare feature distributions;
- monitor drift online;
- maintain rollback thresholds.

## Privacy and governance

**Risk:** future production datasets may contain PII or regulated financial data.

**Mitigation for production extensions:**
- classify data before ingestion;
- minimize retained identifiers;
- encrypt data at rest and in transit;
- enforce least-privilege access;
- define retention/deletion rules;
- record dataset and transformation lineage.

## CI boundary

The baseline GitHub Actions job should remain deterministic and must not depend on:
- downloading the full Kaggle dataset;
- live SEC network access;
- Hugging Face credentials;
- external model services.

Networked integration tests should be separated from the fast unit-test job and run only when their dependencies and credentials are explicitly configured.
