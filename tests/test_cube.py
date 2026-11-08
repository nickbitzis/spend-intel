import pandas as pd
import pytest

from spendintel.cube import category_profile, hhi, query, tail_spend


def test_hhi_monopoly_and_equal_split():
    assert hhi([100]) == pytest.approx(10_000)
    assert hhi([25, 25, 25, 25]) == pytest.approx(2_500)
    assert hhi([10] * 10) == pytest.approx(1_000)


def test_hhi_by_hand():
    # shares 50%, 30%, 20%: 2500 + 900 + 400
    assert hhi([500, 300, 200]) == pytest.approx(3_800)


def test_hhi_ignores_zero_and_empty():
    assert hhi([0, 0]) == 0.0
    assert hhi([100, 0]) == pytest.approx(10_000)


def test_tail_spend_by_hand():
    # total 1000. cumulative before each: A 0, B .5, C .7, D .8, E .9
    s = pd.Series({"A": 500, "B": 200, "C": 100, "D": 100, "E": 100})
    t = tail_spend(s)
    assert t["tail_suppliers"] == 2  # D and E start at or after 80%
    assert t["tail_spend"] == 200
    assert t["tail_spend_share"] == 0.2
    assert t["names"] == ["D", "E"]


def test_tail_spend_single_supplier_has_no_tail():
    assert tail_spend(pd.Series({"A": 10.0}))["tail_suppliers"] == 0


@pytest.fixture
def mini():
    return pd.DataFrame(
        {
            "l1": ["MR", "MR", "MR", "LG"],
            "l2": ["MR-FAS", "MR-FAS", "MR-PPE", "LG-FRT"],
            "l3": ["MR-FAS-BLT", "MR-FAS-BLT", "MR-PPE-GLV", "LG-FRT-LTL"],
            "supplier": ["A", "B", "A", "C"],
            "bu": ["X", "Y", "X", "X"],
            "month": ["2025-01", "2025-01", "2025-02", "2025-02"],
            "erp": ["A", "B", "A", "C"],
            "amount": [100.0, 300.0, 50.0, 25.0],
            "has_po": [True, False, True, True],
        }
    )


def test_query_group_and_filter(mini):
    rows = query(mini, ["l1"])
    assert rows[0]["l1"] == "MR" and rows[0]["spend"] == 450 and rows[0]["suppliers"] == 2
    rows = query(mini, ["supplier"], {"l2": "MR-FAS"})
    assert [r["supplier"] for r in rows] == ["B", "A"]
    assert query(mini, [])[0]["spend"] == 475


def test_query_rejects_unknown_dimension(mini):
    with pytest.raises(ValueError):
        query(mini, ["color"])


def test_category_profile(mini):
    p = {r["code"]: r for r in category_profile(mini, "l3")}
    blt = p["MR-FAS-BLT"]
    assert blt["suppliers"] == 2
    assert blt["hhi"] == pytest.approx(25**2 + 75**2)
    assert blt["po_share"] == 0.25
    assert blt["top_suppliers"][0]["supplier"] == "B"
