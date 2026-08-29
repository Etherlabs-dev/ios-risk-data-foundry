# Evidence Standard

This repository is part of a public engineering portfolio. Claims are separated by evidence type so benchmark results, simulated scenarios, and production outcomes are not conflated.

## Evidence labels

- **Implemented** — behavior exists in code/configuration in this repository.
- **Tested** — behavior is covered by automated or documented repeatable tests.
- **Benchmarked** — a metric was produced by a documented evaluation run.
- **Published artifact** — an output is available as a versioned external artifact.
- **Simulated** — result comes from synthetic or test data.
- **Projected** — estimated business impact; not an observed production result.
- **Production** — reserved for systems verified as operating in a live environment.

## Current evidence table

| Area | Status | Evidence |
|---|---|---|
| Feature engineering | Implemented + tested | `foundry/features/` and `tests/test_features.py` |
| Instruction formatting | Implemented + tested | `foundry/formatters.py` and formatter tests |
| eCFR regulatory preparation | Implemented + tested | Cached official XML parsing, cited answers, and whole-section holdout |
| AML distillation audit | Implemented + tested | 3,355 checkpoint rows preserved; 2,462 strict-valid rewrites included |
| Synthetic scenario generation | Implemented + tested | Explicit seeds, deterministic output, class/count validation |
| Dataset merge/versioning | Implemented + tested | Deterministic deduplication, JSONL validation, SHA256 tests, manifest |
| Multi-source orchestration | Implemented + tested | Tabular, scenario, AML, and eCFR paths with explicit provenance |
| Feature uplift | Benchmarked | Reproduced run documented in `docs/benchmark-reproduction.md` |
| Hugging Face v1 dataset | Published legacy artifact | Retained for lineage; unsuitable as Project 03 reasoning data |
| v3 dataset | Published artifact + independently verified | 20,606 unique pairs; public repository `Etherlabs/ios-risk-finetune-v3`; downloaded SHA-256 matches `dataset_manifest_v3.json` |
| Project 03 model improvement | Not yet evidenced | Requires base-versus-tuned run on the frozen v3 evaluation set |
| Client financial impact | Not claimed | No verified production client outcome in this repo |

## Benchmark interpretation

The reproduced validation run shows engineered features improved average precision, ROC AUC,
F1, and recall while reducing precision. This is evidence of a measurable trade-off on one
public dataset, not proof that the feature set is optimal for a production fraud system.

The validation script currently selects its classification threshold on the same test split used
for final metrics and computes some unsupervised feature statistics before splitting. The result is
therefore an exploratory benchmark, not a clean final held-out estimate. This limitation is explicit
in the reproduction record and remains technical debt.

A production decision would require:

1. a frozen held-out test set;
2. cost-weighted error analysis;
3. distribution-shift checks against target traffic;
4. threshold selection tied to operational capacity;
5. online monitoring after deployment.

## Rules for future claims

Before adding a performance or business-result claim to the README:

1. label the evidence type;
2. link to the code, test, report, or artifact that produced it;
3. state the dataset/source used;
4. state important assumptions and limitations;
5. do not convert modeled ROI into a client result without verifiable evidence and permission.
