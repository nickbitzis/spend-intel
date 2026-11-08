"""Spend cube: totals by category, supplier, business unit, and month, plus concentration math."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .taxonomy import is_addressable, name_of

DIMS = ("l1", "l2", "l3", "supplier", "bu", "month", "erp")


def hhi(spend: pd.Series | list[float] | np.ndarray) -> float:
    """Herfindahl-Hirschman index on a 0 to 10,000 scale: the sum of squared percentage shares.
    One supplier is 10,000. Ten equal suppliers is 1,000. Under 1,500 is usually called fragmented."""
    s = np.asarray(spend, dtype=float)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    shares = s / s.sum() * 100
    return float((shares**2).sum())


def tail_spend(spend_by_supplier: pd.Series, cutoff: float = 0.8) -> dict:
    """Tail suppliers: the ones that together make up the last (1 - cutoff) of spend.

    Sort suppliers by spend, largest first. A supplier is in the tail if the cumulative share of
    every supplier above it is already at or past the cutoff."""
    s = spend_by_supplier[spend_by_supplier > 0].sort_values(ascending=False)
    total = float(s.sum())
    if total == 0:
        return {"tail_suppliers": 0, "tail_spend": 0.0, "suppliers": 0, "tail_supplier_share": 0.0, "names": []}
    before = s.cumsum().shift(1, fill_value=0.0) / total
    tail = s[before >= cutoff - 1e-12]
    return {
        "suppliers": int(len(s)),
        "tail_suppliers": int(len(tail)),
        "tail_supplier_share": round(len(tail) / len(s), 4),
        "tail_spend": round(float(tail.sum()), 2),
        "tail_spend_share": round(float(tail.sum()) / total, 4),
        "names": list(tail.index[:50]),
    }


def query(lines: pd.DataFrame, group_by: list[str], filters: dict | None = None, limit: int = 200) -> list[dict]:
    """Slice the cube. `lines` needs amount plus the DIMS columns."""
    bad = [d for d in group_by if d not in DIMS]
    if bad:
        raise ValueError(f"unknown dimension(s): {bad}")
    df = lines
    for k, v in (filters or {}).items():
        if k not in DIMS:
            raise ValueError(f"unknown filter: {k}")
        if v not in (None, ""):
            df = df[df[k] == v]
    if not group_by:
        return [{"spend": round(float(df["amount"].sum()), 2), "lines": int(len(df))}]
    g = df.groupby(group_by, observed=True).agg(
        spend=("amount", "sum"), lines=("amount", "size"), suppliers=("supplier", "nunique")
    )
    g = g.sort_values("spend", ascending=False).head(limit).reset_index()
    rows = g.to_dict("records")
    for r in rows:
        r["spend"] = round(float(r["spend"]), 2)
        for d in ("l1", "l2", "l3"):
            if d in r:
                r[f"{d}_name"] = name_of(r[d])
    return rows


def category_profile(lines: pd.DataFrame, level: str = "l3", code: str | None = None) -> list[dict]:
    """Per category at one level: spend, suppliers, HHI, top suppliers, tail, PO share."""
    df = lines if code is None else lines[lines[level] == code]
    out = []
    for cat, sub in df.groupby(level):
        by_sup = sub.groupby("supplier")["amount"].sum().sort_values(ascending=False)
        total = float(by_sup.sum())
        t = tail_spend(by_sup)
        out.append(
            {
                "code": cat,
                "name": name_of(cat),
                "spend": round(total, 2),
                "lines": int(len(sub)),
                "suppliers": int(len(by_sup)),
                "hhi": round(hhi(by_sup), 1),
                "top_suppliers": [
                    {"supplier": k, "spend": round(float(v), 2), "share": round(float(v) / total, 4)}
                    for k, v in by_sup.head(5).items()
                ],
                "tail_suppliers": t["tail_suppliers"],
                "tail_spend": t["tail_spend"],
                "po_share": round(float(sub.loc[sub["has_po"], "amount"].sum()) / total, 4) if total else 0.0,
                "addressable": is_addressable(cat),
            }
        )
    out.sort(key=lambda r: -r["spend"])
    return out


def summary(lines: pd.DataFrame) -> dict:
    total = float(lines["amount"].sum())
    addr = float(lines.loc[lines["addressable"], "amount"].sum())
    by_sup = lines.groupby("supplier")["amount"].sum()
    t = tail_spend(by_sup)
    return {
        "total_spend": round(total, 2),
        "addressable_spend": round(addr, 2),
        "addressable_share": round(addr / total, 4) if total else 0.0,
        "lines": int(len(lines)),
        "suppliers": int(by_sup.size),
        "tail": {k: v for k, v in t.items() if k != "names"},
        "po_share": round(float(lines.loc[lines["has_po"], "amount"].sum()) / total, 4) if total else 0.0,
        "by_l1": query(lines, ["l1"]),
        "by_bu": query(lines, ["bu"]),
        "by_month": sorted(query(lines, ["month"]), key=lambda r: r["month"]),
    }
