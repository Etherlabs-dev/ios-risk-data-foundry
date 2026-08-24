"""
IOS Risk — AML Typology Factory
===============================
Generates money-laundering cases, not card-fraud transactions.

WHY THIS EXISTS
  synthetic_generator.py covers four CARD FRAUD patterns from seven
  transaction features. The Project 03 brief also promises AML typologies
  and regulatory language, and nothing in the foundry produced either.
  A single transaction cannot express laundering: structuring is only
  visible across deposits, layering only across accounts, mule activity
  only across counterparties. So these cases carry ACCOUNT-LEVEL context.

TYPOLOGIES COVERED
  Drawn from FATF typology reports and FinCEN advisory patterns:
    structuring        sub-threshold cash to evade CTR reporting
    smurfing           structuring distributed across multiple people
    layering           rapid movement through accounts to break the trail
    mule_network       third-party account receiving and forwarding funds
    trade_based        invoice mismatch used to move value across borders
    funnel_account     many-to-one deposits, withdrawn in another region
    legitimate         benign activity that superficially resembles the above

WHAT THIS DOES NOT DO
  The explanations are still generated from templates. That gives the model
  correct structure, correct typology naming and correct escalation, but not
  novel reasoning. Diversity comes from the distillation pass
  (scripts/distill_reasoning.py), which rewrites these outputs using a
  larger model. Templates are the scaffold, not the destination.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# CTR threshold in the US. Structuring is defined relative to it, so the
# generator places deposits just underneath.
CTR_THRESHOLD = 10_000

TYPOLOGY_TIERS = {
    "structuring": "HIGH",
    "smurfing": "HIGH",
    "layering": "CRITICAL",
    "mule_network": "HIGH",
    "trade_based": "CRITICAL",
    "funnel_account": "CRITICAL",
    "legitimate": "LOW",
}

# Recommended action per tier. SAR is a regulatory filing, EDD is enhanced
# due diligence — the distinction matters and the model should learn it.
TIER_ACTIONS = {
    "CRITICAL": "File SAR within 30 days and freeze pending investigation",
    "HIGH": "File SAR within 30 days and apply enhanced due diligence",
    "LOW": "No action. Continue standard monitoring",
}

OCCUPATIONS = [
    ("student", 18_000), ("retail associate", 32_000), ("rideshare driver", 41_000),
    ("teacher", 55_000), ("nurse", 78_000), ("software engineer", 130_000),
    ("unemployed", 0), ("restaurant server", 29_000),
]
CORRIDORS = ["UAE", "Hong Kong", "Cyprus", "Panama", "Turkey", "Singapore", "Nigeria"]

INSTRUCTION = (
    "You are IOS Risk, an AML analyst. Review the account activity below, "
    "identify any money laundering typology present, assign a risk tier, "
    "and state the required action."
)


class AMLTypologyFactory:
    """Generates account-level AML cases with narrative context."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    # ── input rendering ────────────────────────────────────────────────
    def _profile(self) -> dict[str, Any]:
        occupation, income = self.rng.choice(OCCUPATIONS)
        return {
            "occupation": occupation,
            "declared_income": income,
            "account_age_months": self.rng.randint(1, 84),
        }

    @staticmethod
    def _render(profile: dict[str, Any], activity: dict[str, Any]) -> str:
        """
        Render observable account facts. No typology name, no risk tier —
        those are the answer. Same discipline as format_feature_line().
        """
        lines = [
            f"AccountAgeMonths: {profile['account_age_months']}",
            f"StatedOccupation: {profile['occupation']}",
            f"DeclaredAnnualIncome: ${profile['declared_income']:,}",
        ]
        lines += [f"{k}: {v}" for k, v in activity.items()]
        return " | ".join(lines)

    # ── typologies ─────────────────────────────────────────────────────
    def structuring(self) -> tuple[str, str, str]:
        p = self._profile()
        n = self.rng.randint(3, 9)
        amounts = [self.rng.randrange(8_200, 9_900, 50) for _ in range(n)]
        total = sum(amounts)
        days = self.rng.randint(2, 10)
        branches = self.rng.randint(2, 5)
        activity = {
            "CashDeposits": n,
            "DepositWindowDays": days,
            "LargestDeposit": f"${max(amounts):,}",
            "TotalCashDeposited": f"${total:,}",
            "DistinctBranches": branches,
            "WiresOut": self.rng.randint(0, 2),
            "PriorSARs": 0,
        }
        # Only assert an income mismatch when the numbers actually show one.
        # Claiming it unconditionally teaches the model to justify a verdict
        # with a fact its own input contradicts.
        income_clause = ""
        if p["declared_income"] == 0:
            income_clause = (
                f" The holder declares no income, so ${total:,} in cash has no "
                f"evidenced legitimate source."
            )
        elif total > 0.5 * p["declared_income"]:
            multiple = total / max(p["declared_income"], 1)
            income_clause = (
                f" Cash volume represents {multiple:.1f}x declared annual income of "
                f"${p['declared_income']:,} for a {p['occupation']}, with no documented source."
            )
        why = (
            f"{n} cash deposits over {days} days, every one between $8,200 and $9,900 — "
            f"consistently below the ${CTR_THRESHOLD:,} Currency Transaction Report "
            f"threshold. Total of ${total:,} would have triggered a CTR as a single "
            f"deposit. Use of {branches} branches spreads activity to avoid teller "
            f"familiarity.{income_clause}"
        )
        return self._render(p, activity), why, "structuring"

    def smurfing(self) -> tuple[str, str, str]:
        p = self._profile()
        depositors = self.rng.randint(4, 12)
        per = self.rng.randrange(3_000, 9_000, 100)
        activity = {
            "CashDeposits": depositors * self.rng.randint(1, 2),
            "DistinctDepositors": depositors,
            "AvgDepositAmount": f"${per:,}",
            "TotalCashDeposited": f"${per * depositors:,}",
            "DepositWindowDays": self.rng.randint(1, 5),
            "RelationshipToDepositors": "undocumented",
            "PriorSARs": 0,
        }
        why = (
            f"Cash arriving from {depositors} distinct depositors with no documented "
            f"relationship to the account holder, averaging ${per:,} each. This is "
            f"structuring distributed across multiple individuals — each deposit is "
            f"unremarkable alone, but the aggregate of ${per * depositors:,} into one "
            f"account within days indicates coordinated placement."
        )
        return self._render(p, activity), why, "smurfing"

    def layering(self) -> tuple[str, str, str]:
        p = self._profile()
        hops = self.rng.randint(3, 8)
        amount = self.rng.randrange(40_000, 400_000, 1_000)
        hours = self.rng.randint(2, 48)
        activity = {
            "InboundWires": 1,
            "InboundAmount": f"${amount:,}",
            "OutboundTransfers": hops,
            "DistinctBeneficiaryAccounts": hops,
            "TimeToFullyDisburseHours": hours,
            "ResidualBalance": f"${self.rng.randint(50, 900):,}",
            "CrossBorderLegs": self.rng.randint(1, 3),
            "PriorSARs": 0,
        }
        why = (
            f"${amount:,} received and dispersed across {hops} beneficiary accounts "
            f"within {hours} hours, leaving a negligible residual balance. Rapid "
            f"in-and-out movement with no economic purpose is classic layering — the "
            f"intent is to break the audit trail between the funds and their origin. "
            f"Cross-border legs compound the tracing difficulty."
        )
        return self._render(p, activity), why, "layering"

    def mule_network(self) -> tuple[str, str, str]:
        p = self._profile()
        p["account_age_months"] = self.rng.randint(1, 6)
        inbound = self.rng.randint(5, 20)
        amount = self.rng.randrange(15_000, 90_000, 500)
        activity = {
            "InboundP2PTransfers": inbound,
            "DistinctSenders": inbound,
            "TotalInbound": f"${amount:,}",
            "OutboundCashWithdrawals": self.rng.randint(3, 12),
            "PctForwardedWithin48h": self.rng.randint(85, 99),
            "DeviceSharedWithOtherAccounts": self.rng.randint(2, 9),
            "PriorSARs": 0,
        }
        why = (
            f"A {p['account_age_months']}-month-old account receiving ${amount:,} from "
            f"{inbound} unrelated senders and forwarding almost all of it within 48 "
            f"hours. Device fingerprint shared with multiple other accounts indicates "
            f"coordinated control. The holder is functioning as a money mule — the "
            f"account is a pass-through, not a destination."
        )
        return self._render(p, activity), why, "mule_network"

    def trade_based(self) -> tuple[str, str, str]:
        p = self._profile()
        declared = self.rng.randrange(80_000, 900_000, 1_000)
        market = int(declared * self.rng.uniform(0.10, 0.35))
        corridor = self.rng.choice(CORRIDORS)
        activity = {
            "TradeInvoices": self.rng.randint(2, 9),
            "DeclaredInvoiceValue": f"${declared:,}",
            "AssessedMarketValue": f"${market:,}",
            "Corridor": corridor,
            "GoodsCategory": self.rng.choice(
                ["electronics components", "textiles", "scrap metal", "machinery parts"]
            ),
            "ShippingDocsProvided": self.rng.choice(["partial", "none"]),
            "PriorSARs": 0,
        }
        why = (
            f"Invoices declare ${declared:,} against an assessed market value of "
            f"${market:,} — an over-invoicing gap of roughly "
            f"{100 - int(100 * market / declared)}%. Mispricing of this magnitude on a "
            f"{corridor} corridor, with incomplete shipping documentation, is trade-based "
            f"money laundering: value is moved across borders disguised as commerce."
        )
        return self._render(p, activity), why, "trade_based"

    def funnel_account(self) -> tuple[str, str, str]:
        p = self._profile()
        states = self.rng.randint(4, 14)
        total = self.rng.randrange(30_000, 250_000, 500)
        activity = {
            "CashDeposits": self.rng.randint(10, 40),
            "DepositOriginStates": states,
            "TotalCashDeposited": f"${total:,}",
            "WithdrawalLocation": f"single branch, {self.rng.choice(CORRIDORS)} corridor",
            "PctWithdrawnAsCash": self.rng.randint(80, 100),
            "DepositWindowDays": self.rng.randint(3, 21),
            "PriorSARs": 0,
        }
        why = (
            f"Deposits made in {states} different states totalling ${total:,}, withdrawn "
            f"almost entirely as cash from a single location. Geographic separation "
            f"between deposit and withdrawal, with no business rationale, is the defining "
            f"signature of a funnel account used to consolidate and move illicit proceeds."
        )
        return self._render(p, activity), why, "funnel_account"

    def legitimate(self) -> tuple[str, str, str]:
        """
        Benign activity that superficially resembles a typology. Without these
        the model learns "unusual number = file SAR" and over-escalates.
        """
        p = self._profile()
        variant = self.rng.choice(["bonus", "property", "business", "routine"])
        if variant == "bonus":
            amt = self.rng.randrange(8_000, 30_000, 500)
            activity = {"InboundWires": 1, "InboundAmount": f"${amt:,}",
                        "Source": "employer payroll (documented)",
                        "OutboundTransfers": 1, "Destination": "own brokerage account",
                        "PriorSARs": 0}
            why = (f"Single documented payroll wire of ${amt:,} moved to the holder's own "
                   f"brokerage account. Source, purpose and beneficiary are all verified, "
                   f"and the amount is consistent with a {p['occupation']} receiving an "
                   f"annual bonus. No layering: funds moved once, to a self-owned account.")
        elif variant == "property":
            amt = self.rng.randrange(60_000, 400_000, 5_000)
            activity = {"InboundWires": 1, "InboundAmount": f"${amt:,}",
                        "Source": "title company escrow (documented)",
                        "OutboundTransfers": 1, "Destination": "mortgage lender",
                        "PriorSARs": 0}
            why = (f"${amt:,} received from a title company and disbursed to a mortgage "
                   f"lender. Large and fast, but every leg is documented and the "
                   f"counterparties are regulated institutions. Rapid movement alone is "
                   f"not layering when the economic purpose is evidenced.")
        elif variant == "business":
            n = self.rng.randint(8, 25)
            amt = self.rng.randrange(20_000, 120_000, 500)
            activity = {"CashDeposits": n, "TotalCashDeposited": f"${amt:,}",
                        "BusinessType": "registered cash-intensive retail",
                        "DepositWindowDays": 30, "DistinctBranches": 1,
                        "DepositPatternVsPrior12m": "consistent", "PriorSARs": 0}
            why = (f"{n} cash deposits totalling ${amt:,} over 30 days from a registered "
                   f"cash-intensive retail business, all at one branch, consistent with the "
                   f"prior 12 months. Cash volume alone is not suspicious where the business "
                   f"model explains it and the pattern is stable.")
        else:
            amt = self.rng.randrange(500, 6_000, 50)
            activity = {"InboundTransfers": self.rng.randint(1, 3),
                        "TotalInbound": f"${amt:,}", "OutboundTransfers": self.rng.randint(1, 4),
                        "DepositWindowDays": 30, "PriorSARs": 0}
            why = (f"Routine activity totalling ${amt:,} over 30 days, proportionate to a "
                   f"declared income of ${p['declared_income']:,}. No velocity, threshold, "
                   f"or counterparty anomalies present.")
        return self._render(p, activity), why, "legitimate"


def build_aml_dataset(
    n_per_typology: int = 500,
    n_legitimate: int = 3_000,
    output_path: str = "data/processed/aml_typology_pairs.jsonl",
    seed: int = 42,
) -> list[dict[str, str]]:
    """Generate AML instruction pairs and write them as JSONL."""
    factory = AMLTypologyFactory(seed=seed)
    generators = {
        "structuring": factory.structuring,
        "smurfing": factory.smurfing,
        "layering": factory.layering,
        "mule_network": factory.mule_network,
        "trade_based": factory.trade_based,
        "funnel_account": factory.funnel_account,
    }

    pairs: list[dict[str, str]] = []
    for name, gen in generators.items():
        for _ in range(n_per_typology):
            features, why, typology = gen()
            tier = TYPOLOGY_TIERS[typology]
            pairs.append({
                "instruction": INSTRUCTION,
                "input": features,
                "output": (
                    f"{tier} RISK — {typology.upper().replace('_', ' ')} DETECTED. "
                    f"{why} Recommended action: {TIER_ACTIONS[tier]}."
                ),
                "typology": typology,
                "risk_tier": tier,
            })

    for _ in range(n_legitimate):
        features, why, typology = factory.legitimate()
        pairs.append({
            "instruction": INSTRUCTION,
            "input": features,
            "output": (
                f"LOW RISK — NO TYPOLOGY DETECTED. {why} "
                f"Recommended action: {TIER_ACTIONS['LOW']}."
            ),
            "typology": typology,
            "risk_tier": "LOW",
        })

    random.Random(seed).shuffle(pairs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"✓ AML typology dataset: {len(pairs):,} pairs -> {output_path}")
    return pairs


if __name__ == "__main__":
    build_aml_dataset()
