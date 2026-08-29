from foundry.uploader import build_dataset_card, normalized_records


def test_normalized_records_produces_stable_schema(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text('{"instruction":"i","input":"x","output":"o","source":"s","citation":"c"}\n')
    assert normalized_records(str(source)) == [
        {"instruction": "i", "input": "x", "output": "o", "source": "s"}
    ]


def test_v3_card_states_provenance_and_limitations():
    config = {"exports": {"hf_dataset_name": "ios-risk-finetune-v3"}}
    card = build_dataset_card(config)
    assert "nvidia/nemotron-3-super-120b-a12b" in card
    assert "not production transaction evidence" in card
    assert "20,606 unique" in card
