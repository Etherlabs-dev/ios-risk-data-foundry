# Benchmark Reproduction

## Status

The feature-validation workflow was independently rerun on 2026-08-08 using the public source
dataset and public sibling evaluation harness. The reproduced metrics differ from the historical
README values, so the README now reports the reproduced run rather than preserving unsupported
numbers.

## Inputs

- Dataset: Kaggle `mlg-ulb/creditcardfraud`, 284,807 rows and 31 columns
- Downloaded archive SHA256: `a0360ce715992212e9ac72d8ccdca97f4be87dc1fdf2bed011358f7ab409a28a`
- Extracted CSV SHA256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`
- Evaluation harness: `Etherlabs-dev/eval-harness` commit `5144f8c`
- Python: 3.11.15
- NumPy: 2.0.2
- pandas: 2.3.3
- scikit-learn: 1.6.1
- XGBoost: 3.2.0

The validation-only XGBoost dependency is pinned in `requirements-validation.txt`. On macOS,
XGBoost also requires an OpenMP runtime; Linux wheels used in CI/container environments supply
their corresponding runtime dependency differently.

## Command

```bash
pip install -r requirements-validation.txt
PYTHONPATH=../eval-harness/ios-risk-eval-harness:. \
  python scripts/validate_features.py
```

## Reproduced results

| Metric | Base | Engineered | Delta |
|---|---:|---:|---:|
| Average precision | 0.8510 | 0.8588 | +0.0079 |
| ROC AUC | 0.9747 | 0.9803 | +0.0055 |
| F1 | 0.8415 | 0.8586 | +0.0171 |
| Precision | 0.9059 | 0.8817 | -0.0242 |
| Recall | 0.7857 | 0.8367 | +0.0510 |

These are public-dataset benchmark results. They are not production outcomes, client results, or
evidence of financial impact.

## Methodological limitations

- The script selects the F1 threshold on the same test split used to report final metrics.
- Amount scaling and some engineered statistics are fit before the train/test split.
- The public dataset has no account identifier. `txn_count_1h` and `txn_count_24h` are causal,
  trailing dataset-density proxies, not per-account velocity features.
- `Time` is elapsed time from dataset collection, not an anchored local timestamp; derived
  hour/day fields are proxies.
- Repeated feature decisions against this split can overfit the benchmark.

A stronger next benchmark should use train/validation/test partitions, fit preprocessing on the
training partition only, select a threshold on validation data, and evaluate the frozen pipeline
once on a held-out test set.
