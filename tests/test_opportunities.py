import pandas as pd
import pytest

from spendintel.compliance import leakage
from spendintel.opportunities import (
    duplicates,
    invoice_no_similar,
    item_signature,
    match_contracts,
    maverick,
    norm_invoice_no,
    payment_terms,
    price_variance,
)


def _lines(rows, **defaults):
    df = pd.DataFrame(rows)
    base = {
        "bu": "X",
        "supplier_id": "P1",
        "supplier": "A",
        "l3": "MR-FAS-BLT",
        "has_po": True,
        "erp": "A",
        "invoice_date": pd.Timestamp("2025-03-01"),
        "paid_date": pd.Timestamp("2025-03-31"),
        "net_days": 30,
        "disc_pct": 0.0,
        "disc_days": 0,
        "description": "bolt m8",
        "item_group": "MR-FAS-BLT|bolt m8",
    }
    base.update(defaults)
    for k, v in base.items():
        if k not in df:
            df[k] = v
    if "line_id" not in df:
        df["line_id"] = range(len(df))
    if "invoice_no" not in df:
        df["invoice_no"] = [f"INV{100 + i}" for i in range(len(df))]
    if "amount" not in df:
        df["amount"] = df["qty"] * df["unit_price"]
    return df


def test_item_signature_normalizes():
    assert item_signature("SS bolt M8x30 qty 500") == item_signature("stainless bolt m8x30")
    assert item_signature("LTL FRT PRO# 88812") == item_signature("ltl freight pro")
    assert item_signature("misc supplies per quote 4471") == "4471"


def test_price_variance_hand_worked():
    # Six lines of one item. Prices 1,1,2,2,3,5 (median 2, 10th percentile 1), qty 100 each.
    # Pay-the-median saving: (3-2)*100 + (5-2)*100 = 400. Best price saving: (1+1+2+4)*100 = 800.
    prices = [1, 1, 2, 2, 3, 5]
    df = _lines({"qty": [100] * 6, "unit_price": prices, "supplier": list("AABBCC"), "bu": list("XYXYXY")})
    sink = {}
    (o,) = price_variance(df, min_lines=6, sink=sink)
    assert o["value"] == pytest.approx(400)
    assert o["upside"] == pytest.approx(800)
    assert o["details"]["median_price"] == 2
    assert sink[5] == pytest.approx(300)
    assert o["evidence"][0]["line_id"] == 5


def test_price_variance_skips_unit_of_measure_outliers():
    df = _lines({"qty": [100] * 7, "unit_price": [2, 2, 2, 2, 2, 3, 100], "supplier": list("AABBCCD")})
    (o,) = price_variance(df, min_lines=6)
    assert o["value"] == pytest.approx(100)  # the $100 line is excluded as a 50x outlier


def test_price_variance_needs_enough_lines():
    df = _lines({"qty": [10] * 3, "unit_price": [1, 2, 9], "supplier": list("ABC")})
    assert price_variance(df, min_lines=6) == []


def test_payment_terms_discount_missed():
    # 2/10 net 30, paid on day 25: missed 2% of $10,000 = $200.
    df = _lines(
        {"qty": [1], "unit_price": [10_000.0]},
        net_days=30,
        disc_pct=2.0,
        disc_days=10,
        paid_date=pd.Timestamp("2025-03-26"),
    )
    opps = {o["details"]["kind"]: o for o in payment_terms(df, policy_days=30)}
    assert opps["discount_missed"]["value"] == pytest.approx(200)


def test_payment_terms_short_terms_math():
    # Net 15 against a 60 day policy at 8%: 1,000,000 x 45/365 x 0.08 = 9,863.01
    df = _lines({"qty": [1], "unit_price": [1_000_000.0]}, net_days=15, paid_date=pd.Timestamp("2025-03-16"))
    opps = {o["details"]["kind"]: o for o in payment_terms(df, policy_days=60, cost_of_capital=0.08)}
    assert opps["short_terms"]["value"] == pytest.approx(1_000_000 * 45 / 365 * 0.08, rel=1e-6)
    assert "paid_early" not in opps


def test_payment_terms_paid_early():
    # Net 60, paid on day 10: 50 days early on 365,000 at 8% = 4,000.
    df = _lines({"qty": [1], "unit_price": [365_000.0]}, net_days=60, paid_date=pd.Timestamp("2025-03-11"))
    opps = {o["details"]["kind"]: o for o in payment_terms(df, policy_days=60)}
    assert opps["paid_early"]["value"] == pytest.approx(4_000)


def test_payment_terms_discount_taken_is_not_flagged():
    df = _lines(
        {"qty": [1], "unit_price": [10_000.0]},
        net_days=60,
        disc_pct=2.0,
        disc_days=10,
        paid_date=pd.Timestamp("2025-03-09"),
    )
    assert payment_terms(df, policy_days=60) == []


CONTRACTS = pd.DataFrame(
    [
        {
            "contract_id": "C1",
            "supplier_id": "P1",
            "l3": "MR-FAS-BLT",
            "item_description": "stainless steel bolt M8x30",
            "aliases": "stainless steel bolt M8x30 | SS bolt M8x30",
            "contract_price": 0.40,
        },
    ]
)


def test_contract_matching_by_alias():
    df = _lines(
        {
            "qty": [100, 100, 100],
            "unit_price": [0.40, 0.40, 0.40],
            "description": ["SS bolt M8x30 qty 500", "stainless steel bolt m8x30", "nitrile gloves L"],
            "l3": ["MR-FAS-BLT", "MR-FAS-BLT", "MR-PPE-GLV"],
        }
    )
    m = match_contracts(df, CONTRACTS)
    assert m["contract_item"].notna().tolist() == [True, True, False]
    assert m["on_contract_supplier"].tolist() == [True, True, False]


def test_maverick_detection():
    df = _lines(
        {
            "qty": [1000, 1000],
            "unit_price": [0.40, 0.55],
            "supplier_id": ["P1", "P2"],
            "supplier": ["A", "B"],
            "description": ["SS bolt M8x30", "SS bolt M8x30"],
        }
    )
    m = match_contracts(df, CONTRACTS)
    (o,) = maverick(m)
    assert o["value"] == pytest.approx((0.55 - 0.40) * 1000)
    assert o["evidence"][0]["supplier"] == "B"


def test_contract_leakage():
    df = _lines({"qty": [1000, 1000, 1000], "unit_price": [0.40, 0.44, 0.401], "description": ["SS bolt M8x30"] * 3})
    m = match_contracts(df, CONTRACTS)
    c = leakage(m)
    assert c["matched_lines"] == 3
    assert c["leaking_lines"] == 1  # 0.401 is within the 0.5% tolerance
    assert c["leakage"] == pytest.approx(40)
    assert c["by_supplier"][0]["compliance_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_invoice_number_normalization():
    assert norm_invoice_no("INV-0010234") == "10234"
    assert norm_invoice_no("inv10234") == "10234"
    assert invoice_no_similar("INV-10234", "INV10234") == 1.0
    assert invoice_no_similar("INV-10234", "INV-10234A") == 0.9
    assert invoice_no_similar("10234", "10243") == 0.9  # transposed digits
    assert invoice_no_similar("10234", "10235") == 0.0  # the next invoice, not a re-key
    assert invoice_no_similar("10234", "55555") == 0.0


def test_duplicate_detection():
    d = pd.Timestamp("2025-05-01")
    df = _lines(
        {
            "qty": [1, 1, 1, 1, 1],
            "unit_price": [812.5, 812.5, 300.0, 300.0, 812.5],
            "invoice_no": ["INV-10234", "INV10234A", "INV-20001", "INV-20002", "INV-30000"],
            "invoice_date": [d, d + pd.Timedelta(days=4), d, d + pd.Timedelta(days=3), d + pd.Timedelta(days=40)],
        }
    )
    found = duplicates(df)
    assert len(found) == 1
    assert found[0]["line_ids_duplicate"] == [1]
    assert found[0]["value"] == 812.5


def test_duplicate_needs_same_supplier():
    d = pd.Timestamp("2025-05-01")
    df = _lines(
        {
            "qty": [1, 1],
            "unit_price": [500.0, 500.0],
            "invoice_no": ["A1", "A1"],
            "supplier_id": ["P1", "P2"],
            "invoice_date": [d, d],
        }
    )
    assert duplicates(df) == []
