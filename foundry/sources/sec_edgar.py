"""
IOS Risk — SEC EDGAR Source
============================
Fetches 10-K risk filings from SEC EDGAR's public full-text search API.
Chunks the text and formats it as instruction pairs for LLM fine-tuning.

WHY THIS SOURCE EXISTS:
  The fraud transaction dataset teaches the model numerical patterns.
  EDGAR filings teach it the language of financial risk — how regulators,
  lawyers and risk officers actually describe fraud, AML failures,
  credit defaults and operational risk events.
"""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_BASE_URL = "https://www.sec.gov"
HEADERS = {"User-Agent": "IOS-Risk-DataFoundry research@etherlabs.dev"}


def fetch_filing_urls(query: str, limit: int = 10) -> list[str]:
    """
    Search EDGAR full-text search for 10-K filings matching the query.
    Returns a list of direct document URLs.

    The API returns _id in format "accession_number:filename" and
    CIK in _source.ciks[0] — we combine these to build the direct URL.
    """
    params = {
        "q": query,
        "forms": "10-K",
    }

    try:
        response = requests.get(EDGAR_SEARCH_URL, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"EDGAR search failed for '{query}': {e}")
        return []

    hits = data.get("hits", {}).get("hits", [])
    urls = []

    for hit in hits[:limit]:
        hit_id = hit.get("_id", "")
        source = hit.get("_source", {})

        # _id format is "accession_number:filename" e.g.
        # "0000950133-07-002009:w33365exv99w6.htm"
        if ":" not in hit_id:
            continue

        accession, filename = hit_id.split(":", 1)
        accession_nodashes = accession.replace("-", "")

        ciks = source.get("ciks", [])
        if not ciks:
            continue

        # CIK arrives with leading zeros ("0000310522") — strip for URL.
        # Ignore malformed search hits instead of aborting the full query.
        try:
            cik = str(int(ciks[0]))
        except (TypeError, ValueError):
            continue

        url = f"{EDGAR_BASE_URL}/Archives/edgar/data/{cik}/{accession_nodashes}/{filename}"
        urls.append(url)

    print(f"Found {len(urls)} filings for query: '{query}'")
    return urls


def fetch_filing_text(doc_url: str) -> str:
    """
    Fetch a direct document URL and return clean plain text.
    Now that fetch_filing_urls returns direct document URLs, we no
    longer need to parse an index page — we go straight to the document.
    Returns empty string if anything fails.
    """
    try:
        time.sleep(0.5)  # be polite — SEC rate limits aggressive scrapers
        resp = requests.get(doc_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        # Extract plain text, strip all HTML tags
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        return text

    except requests.RequestException as e:
        print(f"Failed to fetch filing text from {doc_url}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    chunk_size: number of words per chunk
    overlap:    number of words shared between consecutive chunks
                so meaning at chunk boundaries is never lost
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return []

    words = text.split()
    chunks = []
    step = chunk_size - overlap  # how far to advance each iteration
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += step

    return chunks


def process_edgar_source(config: dict) -> None:
    """
    Full EDGAR pipeline: search → fetch → chunk → format → export.
    Reads all parameters from the config dict passed in.
    """
    edgar_cfg = config["sources"]["sec_edgar"]
    queries = edgar_cfg["queries"]
    limit = edgar_cfg["limit_per_query"]
    chunk_size = edgar_cfg["chunk_size"]
    chunk_overlap = edgar_cfg["chunk_overlap"]
    output_path = edgar_cfg["output_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    total_docs = 0

    for query in queries:
        print(f"\nProcessing query: '{query}'")
        urls = fetch_filing_urls(query, limit=limit)

        for url in urls:
            text = fetch_filing_text(url)
            if not text:
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            total_docs += 1

            for chunk in chunks:
                pair = {
                    "instruction": (
                        "Identify the financial risk factors described in this regulatory filing "
                        "excerpt."
                    ),
                    "input": chunk,
                    "output": (
                        "This excerpt from a 10-K filing discusses risk factors related to: "
                        f"{query}."
                    ),
                }
                all_pairs.append(pair)

        # Polite pause between queries
        time.sleep(1)

    # Export to JSONL
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print("\n✓ EDGAR pipeline complete")
    print(f"  Queries processed : {len(queries)}")
    print(f"  Documents fetched : {total_docs}")
    print(f"  Chunks generated  : {len(all_pairs)}")
    print(f"  Output file       : {output_path}")
