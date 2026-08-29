from scripts.distill_reasoning import record_key, validate


def test_validator_rejects_invented_derived_number():
    source = {
        "input": "DeclaredAnnualIncome: $18000 | TotalCashDeposited: $27600",
        "output": "HIGH RISK — STRUCTURING DETECTED. Review the activity.",
        "risk_tier": "HIGH",
        "typology": "structuring",
    }
    ok, reason = validate(
        source,
        "HIGH RISK — STRUCTURING DETECTED. Deposits equal 153% of income. "
        "Recommended action: escalate this case for review. " * 4,
    )
    assert not ok
    assert "invented figures" in reason


def test_record_key_is_stable_and_ignores_output():
    left = {"instruction": "i", "input": "x", "output": "one"}
    right = {"instruction": "i", "input": "x", "output": "two"}
    assert record_key(left) == record_key(right)
