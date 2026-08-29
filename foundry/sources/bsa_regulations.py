"""
IOS Risk — BSA Regulation Source
================================
Builds instruction pairs from 31 CFR Chapter X, the Bank Secrecy Act
regulations, via the public eCFR API.

WHY THIS REPLACES sec_edgar.py
  The EDGAR path produced an answer templated from the SEARCH QUERY:

      "output": f"This excerpt ... discusses risk factors related to: {query}."

  Every chunk retrieved for one query shared an identical answer that was
  never derived from the text. Training on it teaches the model to ignore
  its input and emit a canned sentence. That is the same defect that made
  the v1 dataset unusable, and enabling the source would have propagated it.

  The direction is reversed here. The ANSWER is verbatim regulatory text
  with its citation; only the QUESTION is templated. A templated question
  with a real answer teaches real content. A real question with a templated
  answer teaches fabrication. If exactly one half has to be synthetic, it
  must be the question.

SOURCE
  eCFR API — https://www.ecfr.gov/api — public, versioned, no auth, and
  authoritative. 31 CFR Chapter X is the BSA/AML rulebook itself:

    1010  General Provisions (CTR, SAR, recordkeeping, definitions)
    1020  Rules for Banks
    1021  Rules for Casinos and Card Clubs
    1022  Rules for Money Services Businesses
    1023  Rules for Brokers or Dealers in Securities
    1025  Rules for Insurance Companies

LIMITATION
  These pairs teach regulatory language and citation, not judgement. The
  model learns to state what a rule requires; it does not learn when a rule
  applies to an ambiguous fact pattern. That is what the AML typology cases
  are for.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ECFR_FULL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-31.xml?part={part}"

CHAPTER_X_PARTS = {
    "1010": "General Provisions",
    "1020": "Rules for Banks",
    "1021": "Rules for Casinos and Card Clubs",
    "1022": "Rules for Money Services Businesses",
    "1023": "Rules for Brokers or Dealers in Securities",
    "1025": "Rules for Insurance Companies",
}

# Several phrasings per section so the model does not bind the answer to one
# question form. Distillation widens this further.
QUESTION_FORMS = [
    "What does the Bank Secrecy Act require regarding {topic}?",
    "Under 31 CFR Chapter X, what are the obligations concerning {topic}?",
    "A compliance officer asks about {topic}. What does the regulation say?",
    "Summarise the BSA requirement for {topic}, citing the controlling section.",
    "Which BSA provision governs {topic}, and what does it require?",
]

INSTRUCTION = (
    "You are IOS Risk, a financial compliance assistant. Answer the question "
    "using the Bank Secrecy Act regulations and cite the controlling section."
)

# Trained at max_seq_length 1024, so answers are trimmed rather than truncated
# mid-sentence by the tokenizer.
MAX_ANSWER_CHARS = 1_400
MIN_ANSWER_CHARS = 200


def fetch(url: str, retries: int = 5, backoff: float = 4.0) -> bytes | None:
    """
    eCFR rate-limits and returns 503 or an HTML "queue full" page under load.
    Both are transient, so retry with backoff rather than failing the build.

    Uses requests rather than urllib: urllib relies on the system trust store,
    which is not configured on some macOS Python installs and fails with an
    opaque URLError that looks like the remote service being down. requests
    bundles certifi and gives an accurate error.
    """
    import requests

    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                headers={"Accept": "application/xml", "User-Agent": "ios-risk-foundry/1.0"},
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            body = resp.content
            if b"queue full" in body[:400].lower():
                raise RuntimeError("eCFR queue full")
            return body
        except Exception as exc:  # noqa: BLE001 - network errors vary widely
            wait = backoff * (attempt + 1)
            # Print the message, not just the class: "URLError" alone hides
            # whether the cause is the remote service or the local TLS store.
            print(f"    {type(exc).__name__}: {exc}; retry {attempt + 1}/{retries} in {wait:.0f}s")
            time.sleep(wait)
    return None


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_sections(xml_bytes: bytes) -> list[dict[str, str]]:
    """Extract (citation, heading, body) for each section in a CFR part."""
    root = ET.fromstring(xml_bytes)
    out: list[dict[str, str]] = []

    for div in root.iter():
        if div.tag != "DIV8":  # DIV8 is a section in the eCFR schema
            continue
        number = div.get("N") or ""
        head_el = div.find("HEAD")
        heading = clean("".join(head_el.itertext())) if head_el is not None else ""

        paras = []
        for p in div.findall(".//P"):
            t = clean("".join(p.itertext()))
            if t:
                paras.append(t)
        body = " ".join(paras)
        if body:
            out.append({"citation": number, "heading": heading, "body": body})
    return out


def topic_from_heading(heading: str) -> str:
    """'§ 1020.320 Reports by banks of suspicious transactions.' -> the topic."""
    topic = re.sub(r"^§?\s*[\d.]+\s*", "", heading).strip().rstrip(".")
    return topic[0].lower() + topic[1:] if topic else "this requirement"


def build_pairs(
    sections: list[dict[str, str]], part: str, variants_per_section: int = 4
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for sec in sections:
        body = sec["body"]
        if len(body) < MIN_ANSWER_CHARS:
            continue  # cross-references and reserved sections carry no content
        if len(body) > MAX_ANSWER_CHARS:
            cut = body[:MAX_ANSWER_CHARS]
            body = cut[: cut.rfind(". ") + 1] if ". " in cut else cut

        citation = f"31 CFR § {sec['citation']}"
        topic = f"{topic_from_heading(sec['heading'])} under {citation}"
        # The eCFR heading already carries the section number; repeating it
        # would render as "Under 31 CFR § 1020.320 (§ 1020.320 Reports ...)".
        title = re.sub(r"^§?\s*[\d.]+\s*", "", sec["heading"]).strip().rstrip(".")

        # Chapter X has a fixed number of sections — roughly 120 with substantive
        # text — so volume cannot come from more regulation. It comes from asking
        # each rule several ways. Repeating one answer across phrasings teaches
        # robustness to how a compliance question is worded, which is the actual
        # skill; it does not invent regulatory content.
        for form in QUESTION_FORMS[:variants_per_section]:
            pairs.append(
                {
                    "instruction": INSTRUCTION,
                    "input": form.format(topic=topic),
                    "output": f"Under {citation} ({title}): {body}",
                    "citation": citation,
                    "part": part,
                    "source": "eCFR 31 CFR Chapter X",
                }
            )
    return pairs


def build_bsa_dataset(
    parts: list[str] | None = None,
    date: str = "2026-01-01",
    output_path: str = "data/processed/bsa_regulation_pairs.jsonl",
    cache_dir: str = "data/raw/ecfr",
    variants_per_section: int = 4,
) -> list[dict[str, Any]]:
    parts = parts or list(CHAPTER_X_PARTS)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict[str, Any]] = []
    for part in parts:
        cache = Path(cache_dir) / f"title31-part{part}-{date}.xml"
        if cache.exists():
            print(f"Part {part} ({CHAPTER_X_PARTS.get(part, '')}) — cached")
            raw = cache.read_bytes()
        else:
            print(f"Part {part} ({CHAPTER_X_PARTS.get(part, '')}) — fetching")
            raw = fetch(ECFR_FULL.format(date=date, part=part))
            if raw is None:
                print("  skipped: eCFR unavailable after retries")
                continue
            cache.write_bytes(raw)

        try:
            sections = parse_sections(raw)
        except ET.ParseError as exc:
            print(f"  skipped: unparseable XML ({exc})")
            cache.unlink(missing_ok=True)
            continue

        pairs = build_pairs(sections, part, variants_per_section=variants_per_section)
        print(f"  {len(sections)} sections -> {len(pairs)} pairs")
        all_pairs.extend(pairs)
        time.sleep(1)  # be polite to a public service

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n✓ BSA regulation pairs: {len(all_pairs):,} -> {output_path}")
    if all_pairs:
        print(f"  unique citations: {len({p['citation'] for p in all_pairs}):,}")
        print(f"  unique answers  : {len({p['output'] for p in all_pairs}):,}")
    return all_pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build BSA regulation instruction pairs")
    ap.add_argument("--parts", nargs="*", default=None, help="CFR parts, default all of Chapter X")
    ap.add_argument("--date", default="2026-01-01")
    ap.add_argument("--out", default="data/processed/bsa_regulation_pairs.jsonl")
    args = ap.parse_args()
    build_bsa_dataset(parts=args.parts, date=args.date, output_path=args.out)
