"""Revalidate and compact a resumable distillation checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.distill_reasoning import record_key, validate

UNSAFE_LEGACY_TYPOLOGIES = {"structuring", "smurfing", "mule_network"}


def audit(source_path: str, checkpoint_path: str, output_path: str) -> dict:
    source_rows = [
        json.loads(line)
        for line in Path(source_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_by_key = {record_key(row): row for row in source_rows}

    accepted: list[dict] = []
    seen_keys: set[str] = set()
    seen_outputs: set[str] = set()
    rejected = Counter()

    for line in Path(checkpoint_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rejected["malformed_json"] += 1
            continue
        key = record_key(row)
        if key in seen_keys:
            rejected["duplicate_source_key"] += 1
            continue
        seen_keys.add(key)
        original = source_by_key.get(key)
        if original is None:
            rejected["missing_source"] += 1
            continue
        if not row.get("distilled"):
            rejected["template_fallback"] += 1
            continue
        if row.get("typology") in UNSAFE_LEGACY_TYPOLOGIES:
            rejected["legacy_source_claim_gap"] += 1
            continue
        ok, reason = validate(original, row.get("output", ""))
        if not ok:
            rejected[reason.split(":", 1)[0]] += 1
            continue
        normalized = row["output"].strip()
        if normalized in seen_outputs:
            rejected["duplicate_output"] += 1
            continue
        seen_outputs.add(normalized)
        accepted.append(row)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row) + "\n")

    report = {
        "source_rows": len(source_rows),
        "source_unique_keys": len(source_by_key),
        "accepted": len(accepted),
        "rejected": dict(sorted(rejected.items())),
        "output": str(out),
    }
    Path(f"{output_path}.audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.checkpoint, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
