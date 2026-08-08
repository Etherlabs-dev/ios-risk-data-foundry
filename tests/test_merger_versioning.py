import hashlib
import json

import pytest

from foundry.merger import deduplicate, load_jsonl, merge_sources
from foundry.versioning import count_records, generate_manifest, sha256_of_file


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def test_deduplicate_preserves_first_occurrence_and_ignores_key_order():
    first = {"instruction": "one", "input": "x", "output": "FRAUD"}
    reordered = {"output": "FRAUD", "input": "x", "instruction": "one"}
    second = {"instruction": "two", "input": "y", "output": "LEGITIMATE"}

    assert deduplicate([first, reordered, second]) == [first, second]


def test_load_jsonl_handles_missing_and_reports_malformed_lines(tmp_path):
    assert load_jsonl(tmp_path / "missing.jsonl") == []

    malformed = tmp_path / "bad.jsonl"
    malformed.write_text('{"valid": true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl at line 2"):
        load_jsonl(malformed)

    non_object = tmp_path / "list.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_jsonl(non_object)


def test_merge_uses_only_selected_sources_and_is_deterministic(tmp_path):
    fraud_path = tmp_path / "fraud.jsonl"
    stale_path = tmp_path / "stale.jsonl"
    output_path = tmp_path / "merged.jsonl"
    records = [
        {"instruction": "one", "input": "x", "output": "FRAUD"},
        {"instruction": "two", "input": "y", "output": "LEGITIMATE"},
    ]
    write_jsonl(fraud_path, records + [records[0]])
    write_jsonl(stale_path, [{"instruction": "stale", "input": "z", "output": "FRAUD"}])
    config = {
        "pipeline": {"seed": 9},
        "sources": {
            "fraud_transactions": {"output_path": str(fraud_path)},
            "synthetic": {"output_path": str(stale_path)},
        },
        "exports": {"instruction_pairs_path": str(output_path)},
    }

    first = merge_sources(config, source_names=["fraud_transactions"])
    first_bytes = output_path.read_bytes()
    second = merge_sources(config, source_names=["fraud_transactions"])

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert {record["instruction"] for record in first} == {"one", "two"}


def test_fingerprint_and_manifest_are_content_based(tmp_path):
    dataset = tmp_path / "records.jsonl"
    dataset.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    expected_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()

    assert sha256_of_file(dataset) == expected_hash
    assert count_records(dataset) == 2

    output = tmp_path / "manifest.json"
    manifest = generate_manifest(
        output_path=str(output),
        dataset_files=[str(dataset), str(tmp_path / "missing.jsonl")],
        version="test-v1",
    )

    assert manifest["version"] == "test-v1"
    assert manifest["files"][str(dataset)]["sha256"] == expected_hash
    assert manifest["files"][str(dataset)]["records"] == 2
    assert manifest["files"][str(tmp_path / "missing.jsonl")] == {"status": "missing"}
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
