"""Build the provenance-aware, leakage-controlled Project 03 v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from foundry.sources.aml_typologies import build_aml_dataset
from foundry.sources.synthetic_generator import build_synthetic_dataset

SEED = 42
V1_DATASET = "Etherlabs/ios-risk-finetune-v1"
TARGET_AML = {
    "structuring": 500,
    "smurfing": 500,
    "layering": 500,
    "mule_network": 500,
    "trade_based": 500,
    "funnel_account": 500,
    "legitimate": 3000,
}


def content_id(record: dict[str, Any]) -> str:
    raw = "\x00".join(str(record.get(key, "")) for key in ("instruction", "input", "output"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_aml(templates_path: Path, distilled_path: Path) -> list[dict[str, Any]]:
    templates = build_aml_dataset(
        n_per_typology=500,
        n_legitimate=3000,
        output_path=str(templates_path),
        seed=SEED,
    )
    distilled = read_jsonl(distilled_path) if distilled_path.exists() else []

    selected: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    counts = Counter()
    for row in distilled:
        typology = row.get("typology")
        if typology not in TARGET_AML or counts[typology] >= TARGET_AML[typology]:
            continue
        if row["input"] in seen_inputs:
            continue
        selected.append({**row, "source": "aml_typology_distilled"})
        seen_inputs.add(row["input"])
        counts[typology] += 1

    for row in templates:
        typology = row["typology"]
        if counts[typology] >= TARGET_AML[typology] or row["input"] in seen_inputs:
            continue
        selected.append({**row, "distilled": False, "source": "aml_typology_template"})
        seen_inputs.add(row["input"])
        counts[typology] += 1

    if dict(counts) != TARGET_AML:
        raise ValueError(f"AML target not met: {dict(counts)}")
    return selected


def split_regulations(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Hold out whole CFR sections, not alternate phrasings of trained answers."""
    citations = sorted({row["citation"] for row in rows})
    heldout = {
        citation
        for citation in citations
        if int(hashlib.sha256(citation.encode()).hexdigest(), 16) % 5 == 0
    }
    train = [{**row, "source": "ecfr_bsa_train"} for row in rows if row["citation"] not in heldout]
    test = [{**row, "source": "ecfr_bsa_holdout"} for row in rows if row["citation"] in heldout]
    if {row["citation"] for row in train}.intersection(row["citation"] for row in test):
        raise AssertionError("Regulatory citation leaked across train and holdout")
    return train, test


def load_tabular_train(limit_legitimate: int = 7000) -> tuple[list[dict], list[dict], dict]:
    from datasets import load_dataset

    dataset = load_dataset(V1_DATASET, split="train")
    fraud: list[dict] = []
    legitimate: list[dict] = []
    heldout: list[dict] = []
    for source_index, row in enumerate(dataset):
        clean = {key: row[key] for key in ("instruction", "input", "output")}
        rid = content_id(clean)
        bucket = int(rid[:8], 16) % 10
        if bucket == 0:
            heldout.append({**clean, "source_record_id": rid, "source_index": source_index})
            continue
        enriched = {
            **clean,
            "source": "ulb_tabular_train",
            "source_record_id": rid,
            "source_index": source_index,
        }
        (fraud if clean["output"] == "FRAUD" else legitimate).append(enriched)

    rng = random.Random(SEED)
    rng.shuffle(legitimate)
    picked = fraud + legitimate[:limit_legitimate]
    return (
        picked,
        heldout,
        {
            "train_fraud": len(fraud),
            "train_legitimate": min(limit_legitimate, len(legitimate)),
            "heldout_records": len(heldout),
            "partition": "sha256(content) modulo 10; bucket 0 held out",
        },
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/exports/ios_risk_finetune_v3.jsonl")
    parser.add_argument("--distilled", default="data/processed/aml_distilled_validated.jsonl")
    parser.add_argument("--bsa", default="data/processed/bsa_regulation_pairs.jsonl")
    parser.add_argument("--skip-tabular", action="store_true")
    args = parser.parse_args()

    processed = Path("data/processed")
    aml = build_aml(processed / "aml_typology_pairs_v3.jsonl", Path(args.distilled))

    bsa_rows = read_jsonl(args.bsa)
    bsa_train, bsa_holdout = split_regulations(bsa_rows)
    write_jsonl(processed / "bsa_regulation_holdout_v3.jsonl", bsa_holdout)

    scenario_path = processed / "synthetic_scenario_pairs_v3.jsonl"
    build_synthetic_dataset(n_fraud=2000, n_legit=3000, output_path=str(scenario_path), seed=SEED)
    scenarios = [{**row, "source": "fraud_scenario_v3"} for row in read_jsonl(scenario_path)]

    tabular: list[dict] = []
    tabular_holdout: list[dict] = []
    tabular_meta: dict[str, Any] = {"skipped": True}
    if not args.skip_tabular:
        tabular, tabular_holdout, tabular_meta = load_tabular_train()
        holdout_path = processed / "tabular_holdout_v3.jsonl"
        holdout_sha = write_jsonl(holdout_path, tabular_holdout)
        tabular_meta.update({"holdout_path": str(holdout_path), "holdout_sha256": holdout_sha})

    rows = aml + bsa_train + scenarios + tabular
    input_keys = [(row["instruction"], row["input"]) for row in rows]
    if len(input_keys) != len(set(input_keys)):
        raise ValueError("Duplicate instruction/input pairs remain in the v3 training set")
    random.Random(SEED).shuffle(rows)

    out = Path(args.out)
    digest = write_jsonl(out, rows)
    composition = Counter(row["source"] for row in rows)
    manifest = {
        "version": "foundry-v3.0",
        "dataset": "ios-risk-finetune-v3",
        "records": len(rows),
        "sha256": digest,
        "seed": SEED,
        "unique_instruction_input": len(set(input_keys)),
        "unique_outputs": len({row["output"] for row in rows}),
        "composition": dict(sorted(composition.items())),
        "regulatory_holdout": {
            "records": len(bsa_holdout),
            "citations": sorted({row["citation"] for row in bsa_holdout}),
        },
        "tabular_partition": tabular_meta,
        "claim_boundary": "Synthetic/public-data research artifact; not production evidence.",
    }
    Path(f"{args.out}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
