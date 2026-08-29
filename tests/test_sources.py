import json

import pandas as pd
import pytest

from foundry.sources import sec_edgar
from foundry.sources.aml_typologies import build_aml_dataset
from foundry.sources.sec_edgar import fetch_filing_text, fetch_filing_urls, process_edgar_source
from foundry.sources.synthetic import generate_synthetic_fraud, generate_synthetic_legit
from foundry.sources.synthetic_generator import build_synthetic_dataset


class FakeResponse:
    def __init__(self, *, payload=None, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def real_transactions():
    return pd.DataFrame(
        {
            "Time": [0.0, 10.0, 20.0, 30.0],
            "V1": [-2.0, -1.0, 0.1, 0.2],
            "V2": [-1.0, -0.5, 0.5, 1.0],
            "Amount": [1.0, 3.0, 10.0, 20.0],
            "Class": [1, 1, 0, 0],
        }
    )


def test_statistical_synthetic_generation_is_seeded(real_transactions):
    first_fraud = generate_synthetic_fraud(3, real_transactions, seed=7)
    second_fraud = generate_synthetic_fraud(3, real_transactions, seed=7)
    other_fraud = generate_synthetic_fraud(3, real_transactions, seed=8)
    first_legit = generate_synthetic_legit(3, real_transactions, seed=7)

    pd.testing.assert_frame_equal(first_fraud, second_fraud)
    assert not first_fraud.equals(other_fraud)
    assert first_fraud["Class"].eq(1).all()
    assert first_legit["Class"].eq(0).all()
    assert list(first_fraud.columns) == list(real_transactions.columns)


def test_statistical_synthetic_generation_rejects_missing_class(real_transactions):
    no_fraud = real_transactions.assign(Class=0)
    with pytest.raises(ValueError, match="at least one fraud row"):
        generate_synthetic_fraud(1, no_fraud)
    with pytest.raises(ValueError, match="non-negative"):
        generate_synthetic_legit(-1, real_transactions)


def test_scenario_generation_preserves_requested_count_and_seed(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    build_synthetic_dataset(n_fraud=5, n_legit=3, output_path=str(first), seed=17)
    build_synthetic_dataset(n_fraud=5, n_legit=3, output_path=str(second), seed=17)

    assert first.read_bytes() == second.read_bytes()
    records = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 8
    assert all("RiskLevel" not in record["input"] for record in records)
    assert all("fraud_type" not in record["input"] for record in records)


def test_aml_generation_produces_unique_inputs(tmp_path):
    output = tmp_path / "aml.jsonl"
    records = build_aml_dataset(n_per_typology=3, n_legitimate=7, output_path=str(output), seed=19)
    assert len(records) == 25
    assert len({record["input"] for record in records}) == 25


def test_fetch_filing_urls_parses_valid_hits_and_skips_malformed(monkeypatch):
    payload = {
        "hits": {
            "hits": [
                {
                    "_id": "0000950133-07-002009:risk.htm",
                    "_source": {"ciks": ["0000310522"]},
                },
                {"_id": "missing-separator", "_source": {"ciks": ["1"]}},
                {"_id": "0001-01-000001:file.htm", "_source": {"ciks": ["bad"]}},
            ]
        }
    }
    monkeypatch.setattr(
        sec_edgar.requests, "get", lambda *args, **kwargs: FakeResponse(payload=payload)
    )

    assert fetch_filing_urls("fraud", limit=10) == [
        "https://www.sec.gov/Archives/edgar/data/310522/000095013307002009/risk.htm"
    ]


def test_fetch_filing_text_strips_html_without_live_network(monkeypatch):
    monkeypatch.setattr(sec_edgar.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        sec_edgar.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            text="<html><body><h1>Risk</h1><p>Credit loss</p></body></html>"
        ),
    )

    assert fetch_filing_text("https://www.sec.gov/example.htm") == "Risk Credit loss"


def test_process_edgar_source_uses_injected_public_responses(monkeypatch, tmp_path):
    output = tmp_path / "edgar.jsonl"
    config = {
        "sources": {
            "sec_edgar": {
                "queries": ["operational risk"],
                "limit_per_query": 1,
                "chunk_size": 3,
                "chunk_overlap": 1,
                "output_path": str(output),
            }
        }
    }
    monkeypatch.setattr(sec_edgar, "fetch_filing_urls", lambda query, limit: ["document"])
    monkeypatch.setattr(sec_edgar, "fetch_filing_text", lambda url: "one two three four")
    monkeypatch.setattr(sec_edgar.time, "sleep", lambda *_: None)

    process_edgar_source(config)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["input"] for record in records] == ["one two three", "three four"]
    assert all("operational risk" in record["output"] for record in records)
