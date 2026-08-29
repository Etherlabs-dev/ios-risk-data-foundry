"""Fail-closed validation for the exact Project 03 training export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

UNSUPPORTED_METRIC = re.compile(
    r"\b(?:probability of fraud|confidence score|risk score|fraud score|model score)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate(export_path: Path, manifest_path: Path) -> dict:
    rows = read_jsonl(export_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"instruction", "input", "output", "source"}
    malformed = [index for index, row in enumerate(rows) if not required.issubset(row)]
    empty = [
        index
        for index, row in enumerate(rows)
        if any(not str(row.get(key, "")).strip() for key in ("instruction", "output"))
    ]
    keys = [(row.get("instruction"), row.get("input")) for row in rows]
    forbidden_metrics = [
        index for index, row in enumerate(rows) if UNSUPPORTED_METRIC.search(row["output"])
    ]
    digest = hashlib.sha256(export_path.read_bytes()).hexdigest()
    sources = Counter(row["source"] for row in rows)

    failures = {
        "malformed_rows": malformed[:20],
        "empty_required_text": empty[:20],
        "duplicate_instruction_input": len(keys) - len(set(keys)),
        "unsupported_metric_claims": forbidden_metrics[:20],
        "manifest_record_match": len(rows) == manifest["records"],
        "manifest_sha_match": digest == manifest["sha256"],
        "manifest_composition_match": dict(sorted(sources.items())) == manifest["composition"],
    }
    if malformed or empty or failures["duplicate_instruction_input"] or forbidden_metrics:
        raise ValueError(f"Dataset content validation failed: {failures}")
    if not all(
        failures[key]
        for key in ("manifest_record_match", "manifest_sha_match", "manifest_composition_match")
    ):
        raise ValueError(f"Dataset manifest validation failed: {failures}")

    report = {
        "records": len(rows),
        "sha256": digest,
        "unique_instruction_input": len(set(keys)),
        "unique_outputs": len({row["output"] for row in rows}),
        "composition": dict(sorted(sources.items())),
        "checks": failures,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", default="data/exports/ios_risk_finetune_v3.jsonl")
    parser.add_argument(
        "--manifest", default="data/exports/ios_risk_finetune_v3.jsonl.manifest.json"
    )
    args = parser.parse_args()
    validate(Path(args.export), Path(args.manifest))
