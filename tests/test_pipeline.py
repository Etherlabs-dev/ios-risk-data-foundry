import json

import pandas as pd
import pytest
import yaml

import foundry.pipeline as pipeline


def test_pipeline_orchestrates_only_enabled_sources(monkeypatch, tmp_path):
    raw_path = tmp_path / "raw.csv"
    source_output = tmp_path / "processed" / "tabular.jsonl"
    export_output = tmp_path / "exports" / "merged.jsonl"
    config_path = tmp_path / "config.yaml"
    pd.DataFrame(
        {
            "Time": [0, 1200, 7200],
            "V1": [0.1, -0.2, 0.3],
            "Amount": [0.5, 20.0, 100.0],
            "Class": [1, 0, 0],
        }
    ).to_csv(raw_path, index=False)
    config = {
        "pipeline": {"seed": 42},
        "sources": {
            "fraud_transactions": {
                "enabled": True,
                "raw_path": str(raw_path),
                "output_path": str(source_output),
            },
            "sec_edgar": {"enabled": False, "output_path": str(tmp_path / "stale.jsonl")},
        },
        "exports": {"instruction_pairs_path": str(export_output)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "process_edgar_source",
        lambda _: pytest.fail("disabled SEC source should not execute"),
    )

    pairs = pipeline.run_pipeline(str(config_path))

    assert len(pairs) == 3
    assert source_output.exists()
    assert export_output.exists()
    assert [json.loads(line) for line in export_output.read_text().splitlines()] == pairs


def test_pipeline_requires_an_enabled_source(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("sources:\n  sec_edgar:\n    enabled: false\n", encoding="utf-8")

    with pytest.raises(ValueError, match="At least one source"):
        pipeline.run_pipeline(str(config_path))
