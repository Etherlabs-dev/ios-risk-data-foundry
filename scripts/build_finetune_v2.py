"""
Build ios-risk-finetune-v2
==========================
Rebuilds the fine-tuning dataset after the v1 defect.

WHAT WENT WRONG IN v1
  `Etherlabs/ios-risk-finetune-v1` shipped 276,772 rows with ONE unique
  instruction and TWO unique outputs (LEGITIMATE 99.1% / FRAUD 0.9%). Only
  the tabular path ran: `synthetic` and `sec_edgar` were `enabled: false`,
  and `synthetic_scenario_pairs.jsonl` was never a registered source at all.
  A model trained on that can only emit one of two words, which cannot meet
  Project 03's success criteria (risk tier + reasoning + recommended action).

WHAT v2 CHANGES
  1. Scenario pairs are included — the only source with risk TIERS
     (LOW / HIGH / CRITICAL) and written reasoning.
  2. Label leakage is fixed upstream in synthetic_generator.py: fraud_type
     and risk_level no longer appear in the model's input.
  3. The tabular path is RESAMPLED rather than dumped whole. v1's 99.1/0.9
     split makes "always answer LEGITIMATE" a near-optimal strategy.
  4. EDGAR stays excluded. Its output is templated from the search query
     rather than the filing text, so every chunk from one query shares an
     identical answer. That path needs redesign, not enabling.

USAGE
  python -m scripts.build_finetune_v2 --out data/exports/ios_risk_finetune_v2.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from foundry.sources.synthetic_generator import build_synthetic_dataset  # noqa: E402

V1_DATASET = "Etherlabs/ios-risk-finetune-v1"

# Tabular resampling target. v1 is 0.89% FRAUD; at that ratio the model learns
# to ignore the input. 25% is a deliberate compromise: enough signal to learn
# the minority class, while still teaching that most traffic is legitimate.
# NOTE: this means predicted FRAUD rates are NOT calibrated to real-world
# prevalence. Any downstream threshold must be re-tuned on real data.
TABULAR_TOTAL = 10_000
TABULAR_FRAUD_RATIO = 0.25

SCENARIO_FRAUD = 2_000
SCENARIO_LEGIT = 8_000

SEED = 42


def load_v1_tabular() -> list[dict]:
    """Pull v1 from the Hub. It is the tabular path, already cleaned and formatted."""
    from datasets import load_dataset

    print(f"Loading {V1_DATASET} from HuggingFace...")
    ds = load_dataset(V1_DATASET, split="train")
    rows = [
        {"instruction": r["instruction"], "input": r["input"], "output": r["output"]} for r in ds
    ]
    print(f"  {len(rows):,} tabular rows")
    return rows


def resample_tabular(rows: list[dict], rng: random.Random) -> list[dict]:
    """Rebalance the tabular pairs toward the minority class."""
    fraud = [r for r in rows if r["output"] == "FRAUD"]
    legit = [r for r in rows if r["output"] == "LEGITIMATE"]
    print(f"  source balance: {len(fraud):,} FRAUD / {len(legit):,} LEGITIMATE")

    want_fraud = min(len(fraud), int(TABULAR_TOTAL * TABULAR_FRAUD_RATIO))
    want_legit = min(len(legit), TABULAR_TOTAL - want_fraud)

    picked = rng.sample(fraud, want_fraud) + rng.sample(legit, want_legit)
    print(f"  resampled to : {want_fraud:,} FRAUD / {want_legit:,} LEGITIMATE")
    return picked


def summarise(pairs: list[dict]) -> dict:
    instructions = collections.Counter(p["instruction"] for p in pairs)
    tiers = collections.Counter()
    for p in pairs:
        head = p["output"].split("—")[0].strip()
        tiers[head if head in {"LOW RISK", "HIGH RISK", "CRITICAL RISK"} else p["output"][:20]] += 1
    return {
        "records": len(pairs),
        "unique_instructions": len(instructions),
        "unique_outputs": len({p["output"] for p in pairs}),
        "output_classes": dict(tiers.most_common(8)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the ios-risk-finetune-v2 dataset")
    ap.add_argument("--out", default="data/exports/ios_risk_finetune_v2.jsonl")
    ap.add_argument(
        "--scenarios-only",
        action="store_true",
        help="Skip the tabular path (no network / no v1 download)",
    )
    args = ap.parse_args()

    rng = random.Random(SEED)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Reasoning pairs — regenerated so the leakage fix is applied.
    print("\n--- Fraud Scenario Factory ---")
    scenario_path = REPO_ROOT / "data/processed/synthetic_scenario_pairs.jsonl"
    build_synthetic_dataset(
        n_fraud=SCENARIO_FRAUD,
        n_legit=SCENARIO_LEGIT,
        output_path=str(scenario_path),
        seed=SEED,
    )
    scenarios = [json.loads(line) for line in scenario_path.read_text().splitlines() if line]

    # 2. Tabular pairs — resampled.
    tabular: list[dict] = []
    if not args.scenarios_only:
        print("\n--- Tabular (resampled from v1) ---")
        tabular = resample_tabular(load_v1_tabular(), rng)

    # 3. Merge.
    print("\n--- Merging ---")
    pairs = scenarios + tabular
    rng.shuffle(pairs)

    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    stats = summarise(pairs)

    manifest = {
        "version": "foundry-v2.0",
        "dataset": "ios-risk-finetune-v2",
        "supersedes": "ios-risk-finetune-v1",
        "reason": (
            "v1 contained only the tabular path: 1 unique instruction, 2 unique "
            "outputs, 99.1% class imbalance. Scenario pairs were generated but "
            "never merged; EDGAR never ran."
        ),
        "composition": {
            "synthetic_scenarios": len(scenarios),
            "tabular_resampled": len(tabular),
        },
        "excluded": {
            "sec_edgar": "output templated from search query, not filing text",
        },
        "sha256": digest,
        "seed": SEED,
        **stats,
    }
    manifest_path = out_path.parent / "ios_risk_finetune_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n✓ Build complete")
    print(json.dumps(manifest, indent=2))
    print(f"\n  Dataset  : {out_path}")
    print(f"  Manifest : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
