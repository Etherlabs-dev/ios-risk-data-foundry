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
| SEC text preparation | Implemented + partially tested | `foundry/sources/sec_edgar.py`; chunking test |
| Synthetic scenario generation | Implemented | `foundry/sources/synthetic*.py` |
| Dataset versioning | Implemented | DVC config, `foundry/versioning.py`, `dataset_manifest.json` |
| Feature uplift | Benchmarked | `scripts/validate_features.py` and documented comparison |
| Hugging Face dataset | Published artifact | `Etherlabs/ios-risk-finetune-v1` |
| Client financial impact | Not claimed | No verified production client outcome in this repo |

## Benchmark interpretation

The documented validation run shows engineered features improved average precision, F1, and recall while reducing precision slightly. This is evidence of a measurable trade-off on the evaluation dataset, not proof that the feature set is optimal for every production fraud system.

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
