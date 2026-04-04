"""
IOS Risk — Fraud Scenario Factory
===================================
Generates explicit fraud scenario records with natural language
risk explanations — not just labels.

WHY THIS IS DIFFERENT FROM synthetic.py:
  synthetic.py samples from real data distributions to balance class counts.
  This factory generates from explicit fraud behaviour rules and produces
  a risk_explanation field — teaching the LLM to reason about fraud,
  not just classify it.
"""
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path


class FraudScenarioFactory:
    """
    Generates realistic fraud and legitimate transaction scenarios.
    Each scenario includes all feature fields plus a natural-language
    risk explanation that becomes the LLM's target output.
    """

    # Fraud types and their risk levels
    FRAUD_TYPES = {
        "card_testing":      "HIGH",
        "account_takeover":  "CRITICAL",
        "money_mule":        "HIGH",
        "bust_out":          "CRITICAL",
    }

    def __init__(self, seed: int = 42):
        self.rng    = np.random.default_rng(seed)
        self.random = random.Random(seed)


    def generate_card_testing(self) -> dict:
        """
        Card testing: many micro-transactions in a short burst.
        Fraudsters verify stolen card numbers with small amounts
        before escalating to large purchases.
        """
        txn_count = int(self.rng.integers(5, 30))
        amount    = round(float(self.rng.uniform(0.01, 2.00)), 2)
        hour      = int(self.rng.integers(0, 23))
        time_sec  = hour * 3600 + int(self.rng.integers(0, 3600))

        # V1-V28: card testing shows anomalous PCA patterns
        v_features = {
            f"V{i}": round(float(self.rng.normal(loc=-1.5, scale=1.2)), 4)
            for i in range(1, 29)
        }

        explanation = (
            f"HIGH RISK — CARD TESTING DETECTED. "
            f"{txn_count} micro-transactions in under 1 hour suggests automated "
            f"card validation. Amount of ${amount:.2f} is characteristic of testing "
            f"whether a stolen card is active before escalating to larger purchases. "
            f"Velocity of {txn_count} transactions/hour far exceeds normal behaviour."
        )

        return {
            **v_features,
            "Time":           time_sec,
            "Amount":         amount,
            "Class":          1,
            "fraud_type":     "card_testing",
            "risk_level":     "HIGH",
            "txn_count_1h":   txn_count,
            "risk_explanation": explanation,
        }


    def generate_account_takeover(self) -> dict:
        """
        Account takeover: large purchases at unusual hours.
        Fraudster has compromised legitimate credentials and makes
        high-value purchases before the victim notices.
        """
        txn_count = int(self.rng.integers(1, 5))
        amount    = round(float(self.rng.uniform(200.00, 2000.00)), 2)

        # Heavily skewed to off-hours — victims are asleep
        off_hours = list(range(0, 7)) + [23]
        hour      = int(self.random.choice(off_hours))
        time_sec  = hour * 3600 + int(self.rng.integers(0, 3600))

        # V features: account takeover has distinct PCA signature
        v_features = {
            f"V{i}": round(float(self.rng.normal(loc=-2.5, scale=1.5)), 4)
            for i in range(1, 29)
        }

        explanation = (
            f"CRITICAL RISK — ACCOUNT TAKEOVER DETECTED. "
            f"Large transaction of ${amount:,.2f} at {hour:02d}:00 is inconsistent "
            f"with normal account behaviour. Off-hours timing combined with "
            f"unusually high amount suggests compromised credentials. "
            f"Fraudsters target sleeping victims to maximise time before detection. "
            f"Immediate account freeze recommended."
        )

        return {
            **v_features,
            "Time":             time_sec,
            "Amount":           amount,
            "Class":            1,
            "fraud_type":       "account_takeover",
            "risk_level":       "CRITICAL",
            "txn_count_1h":     txn_count,
            "risk_explanation": explanation,
        }


    def generate_money_mule(self) -> dict:
        """
        Money mule: round-number amounts just under reporting thresholds.
        Structured to avoid triggering Bank Secrecy Act reporting
        requirements ($10,000 threshold).
        """
        txn_count = int(self.rng.integers(2, 8))
        hour      = int(self.rng.integers(0, 23))
        time_sec  = hour * 3600 + int(self.rng.integers(0, 3600))

        # Round numbers just under $10k — structuring pattern
        base   = int(self.rng.integers(20, 99)) * 100   # $2,000–$9,900
        amount = float(base)

        # V features: money mule has moderate anomaly signal
        v_features = {
            f"V{i}": round(float(self.rng.normal(loc=-1.0, scale=1.0)), 4)
            for i in range(1, 29)
        }

        explanation = (
            f"HIGH RISK — MONEY MULE / STRUCTURING DETECTED. "
            f"Round-number amount of ${amount:,.0f} is consistent with structuring "
            f"transactions to stay below the $10,000 Bank Secrecy Act reporting "
            f"threshold. {txn_count} transactions of similar size suggest deliberate "
            f"layering to obscure the origin of funds. "
            f"AML review and SAR filing may be required."
        )

        return {
            **v_features,
            "Time":             time_sec,
            "Amount":           amount,
            "Class":            1,
            "fraud_type":       "money_mule",
            "risk_level":       "HIGH",
            "txn_count_1h":     txn_count,
            "risk_explanation": explanation,
        }


    def generate_bust_out(self) -> dict:
        """
        Bust-out fraud: sudden high-velocity large purchases after
        a period of normal behaviour. Fraudster maxes out credit
        lines and disappears.
        """
        txn_count = int(self.rng.integers(10, 50))
        amount    = round(float(self.rng.uniform(500.00, 5000.00)), 2)
        hour      = int(self.rng.integers(0, 23))
        time_sec  = hour * 3600 + int(self.rng.integers(0, 3600))

        # V features: bust-out has the most extreme anomaly signal
        v_features = {
            f"V{i}": round(float(self.rng.normal(loc=-3.0, scale=2.0)), 4)
            for i in range(1, 29)
        }

        explanation = (
            f"CRITICAL RISK — BUST-OUT FRAUD DETECTED. "
            f"Burst of {txn_count} large transactions averaging ${amount:,.2f} "
            f"is the hallmark of bust-out fraud. Account history shows normal "
            f"behaviour followed by sudden maxing of credit lines. "
            f"Total exposure in this burst: ${txn_count * amount:,.2f}. "
            f"Account should be frozen immediately and referred to fraud investigations."
        )

        return {
            **v_features,
            "Time":             time_sec,
            "Amount":           amount,
            "Class":            1,
            "fraud_type":       "bust_out",
            "risk_level":       "CRITICAL",
            "txn_count_1h":     txn_count,
            "risk_explanation": explanation,
        }



def build_synthetic_dataset(
    n_fraud: int = 2000,
    n_legit: int = 8000,
    output_path: str = "data/processed/synthetic_scenario_pairs.jsonl",
) -> None:
    """
    Generate a balanced dataset of fraud scenarios and legitimate
    transactions, formatted as instruction pairs with risk explanations.
    """
    factory   = FraudScenarioFactory(seed=42)
    fraud_types = list(FraudScenarioFactory.FRAUD_TYPES.keys())
    per_type  = n_fraud // len(fraud_types)   # 500 per fraud type

    generators = {
        "card_testing":     factory.generate_card_testing,
        "account_takeover": factory.generate_account_takeover,
        "money_mule":       factory.generate_money_mule,
        "bust_out":         factory.generate_bust_out,
    }

    all_pairs = []

    # Generate fraud scenarios — equal split across all 4 types
    print(f"Generating {n_fraud:,} fraud scenarios ({per_type} per type)...")
    for fraud_type, generator in generators.items():
        for _ in range(per_type):
            record = generator()
            pair = {
                "instruction": (
                    "You are IOS Risk, an AI system for financial risk assessment. "
                    "Analyse the following transaction and assess its fraud risk."
                ),
                "input": (
                    f"Amount: ${record['Amount']:.2f} | "
                    f"TxnCount1h: {record['txn_count_1h']} | "
                    f"Hour: {int(record['Time']) // 3600 % 24} | "
                    f"FraudType: {record['fraud_type']} | "
                    f"RiskLevel: {record['risk_level']}"
                ),
                "output": record["risk_explanation"],
            }
            all_pairs.append(pair)

    # Generate legitimate transactions
    print(f"Generating {n_legit:,} legitimate transactions...")
    rng = np.random.default_rng(seed=99)
    for _ in range(n_legit):
        amount = round(float(rng.lognormal(mean=3.5, sigma=1.2)), 2)
        hour   = int(rng.integers(0, 23))
        txn_count = int(rng.integers(1, 5))
        pair = {
            "instruction": (
                "You are IOS Risk, an AI system for financial risk assessment. "
                "Analyse the following transaction and assess its fraud risk."
            ),
            "input": (
                f"Amount: ${amount:.2f} | "
                f"TxnCount1h: {txn_count} | "
                f"Hour: {hour} | "
                f"FraudType: none | "
                f"RiskLevel: LOW"
            ),
            "output": (
                f"LOW RISK — LEGITIMATE TRANSACTION. "
                f"Amount of ${amount:.2f} with {txn_count} transactions in the past hour "
                f"is consistent with normal spending behaviour. "
                f"No anomalous velocity, amount, or timing signals detected."
            ),
        }
        all_pairs.append(pair)

    # Shuffle and export
    random.Random(42).shuffle(all_pairs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"\n✓ Scenario dataset complete")
    print(f"  Fraud scenarios : {n_fraud:,} ({per_type} × {len(fraud_types)} types)")
    print(f"  Legit records   : {n_legit:,}")
    print(f"  Total pairs     : {len(all_pairs):,}")
    print(f"  Output          : {output_path}")


if __name__ == "__main__":
    build_synthetic_dataset()
