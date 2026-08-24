"""
IOS Risk — Reasoning Distillation
=================================
Rewrites templated scenario explanations as varied, analyst-quality prose
using a larger open-weights model.

WHY
  aml_typologies.py and synthetic_generator.py produce correct labels with
  templated explanations. A model trained on templates learns the template.
  This pass keeps every verdict, tier and typology exactly as generated and
  regenerates only the PROSE, so the training signal gains linguistic variety
  without gaining factual noise.

MODEL CHOICE — read before running
  Use an open-weights model whose licence places no restriction on using its
  outputs to train other models:

    Qwen2.5-72B-Instruct   Apache 2.0   no output restrictions   (recommended)
    Mixtral-8x22B-Instruct Apache 2.0   no output restrictions
    DeepSeek-V3            MIT          no output restrictions
    Llama-3.3-70B-Instruct Llama CCL    permitted, but the derived model name
                                        must begin with "Llama" — which already
                                        binds this project, base model is Llama

  Do NOT use a hosted frontier model whose terms restrict training use.

PROVIDERS
  Any OpenAI-compatible endpoint works. Set BASE_URL and API_KEY:

    Together   https://api.together.xyz/v1
    DeepInfra  https://api.deepinfra.com/v1/openai
    OpenRouter https://openrouter.ai/api/v1

USAGE
    export DISTILL_API_KEY=...
    python -m scripts.distill_reasoning \
        --in  data/processed/aml_typology_pairs.jsonl \
        --out data/processed/aml_typology_pairs_distilled.jsonl \
        --base-url https://api.together.xyz/v1 \
        --model Qwen/Qwen2.5-72B-Instruct \
        --limit 200          # start small, inspect, then run the full set

COST
  ~250 output tokens per record. 6,000 records is roughly 1.5M tokens,
  which on a 72B open model runs about USD 1-3. Start with --limit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

SYSTEM = (
    "You are a senior AML analyst writing case notes. You rewrite draft "
    "assessments into natural analyst prose. You never change the verdict, the "
    "risk tier, the typology, or any figure. You vary sentence structure and "
    "phrasing so that no two notes read alike."
)

USER_TEMPLATE = """Rewrite the assessment below as a senior analyst's case note.

HARD CONSTRAINTS — violating any of these makes the output unusable:
- Keep the risk tier exactly: {tier}
- Keep the typology exactly: {typology}
- Keep every figure exactly as written. Invent no new numbers.
- Reference only facts present in the account activity. Add no new evidence.
- Begin with "{tier} RISK — {typology_display} DETECTED." (or, for LOW,
  "LOW RISK — NO TYPOLOGY DETECTED.")
- End with a recommended action consistent with the tier.
- 90 to 160 words. Vary your phrasing from any standard template.

ACCOUNT ACTIVITY:
{features}

DRAFT ASSESSMENT:
{draft}

Rewritten case note:"""


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    tier = record["risk_tier"]
    typology = record.get("typology", "unknown")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(
            tier=tier,
            typology=typology,
            typology_display=typology.upper().replace("_", " "),
            features=record["input"],
            draft=record["output"],
        )},
    ]


def numbers_in(text: str) -> set[str]:
    """Figures the rewrite must preserve. Commas stripped so $9,500 == $9500."""
    return set(re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")))


def validate(original: dict[str, Any], rewritten: str) -> tuple[bool, str]:
    """
    Reject rewrites that drift. Cheaper to discard and keep the template than
    to train on a confident-sounding fabrication.
    """
    tier = original["risk_tier"]
    if not rewritten.upper().startswith(f"{tier} RISK"):
        return False, f"does not open with {tier} RISK"

    typology = original.get("typology", "")
    if typology != "legitimate":
        token = typology.upper().replace("_", " ")
        if token not in rewritten.upper():
            return False, f"typology '{token}' missing"

    # Every figure in the rewrite must have existed in the source material.
    allowed = numbers_in(original["input"]) | numbers_in(original["output"])
    invented = numbers_in(rewritten) - allowed
    if invented:
        return False, f"invented figures: {sorted(invented)[:5]}"

    words = len(rewritten.split())
    if not 60 <= words <= 220:
        return False, f"length {words} words out of range"

    return True, "ok"


def call_model(client: Any, model: str, messages: list[dict[str, str]],
               retries: int = 4) -> str | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.9,       # high: variety is the entire point
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001 - provider errors vary widely
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{retries} after {type(exc).__name__} ({wait}s)")
            time.sleep(wait)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Distil varied reasoning into scenario pairs")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = os.environ.get("DISTILL_API_KEY")
    if not api_key:
        print("DISTILL_API_KEY is not set.", file=sys.stderr)
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    records = [json.loads(x) for x in Path(args.inp).read_text().splitlines() if x.strip()]
    if args.limit:
        random.Random(args.seed).shuffle(records)
        records = records[: args.limit]
    print(f"Distilling {len(records):,} records via {args.model}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = rejected = failed = 0
    reasons: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            text = call_model(client, args.model, build_messages(rec))
            if text is None:
                failed += 1
                out = rec["output"]           # provider gave up: keep the template
            else:
                ok, reason = validate(rec, text)
                if ok:
                    kept += 1
                    out = text
                else:
                    rejected += 1
                    reasons[reason.split(":")[0]] = reasons.get(reason.split(":")[0], 0) + 1
                    out = rec["output"]       # drifted: keep the template
            f.write(json.dumps({
                "instruction": rec["instruction"],
                "input": rec["input"],
                "output": out,
                "typology": rec.get("typology"),
                "risk_tier": rec.get("risk_tier"),
                "distilled": out != rec["output"],
            }) + "\n")

            if i % 50 == 0:
                print(f"  {i}/{len(records)}  kept={kept} rejected={rejected} failed={failed}")

    total = kept + rejected + failed
    print("\n✓ Distillation complete")
    print(f"  distilled : {kept:,} ({100 * kept / max(total, 1):.1f}%)")
    print(f"  rejected  : {rejected:,}  {reasons}")
    print(f"  api failed: {failed:,}")
    print(f"  output    : {out_path}")
    print("\nRejected and failed records keep their template text, so the file is")
    print("always complete. Inspect a sample before training on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
