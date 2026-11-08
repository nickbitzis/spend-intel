"""Contract compliance: on lines that went to the contracted supplier for a contracted item, did
the invoice price match the contract price? Leakage is the amount paid above contract."""

from __future__ import annotations

import pandas as pd

from .taxonomy import name_of


def leakage(lines: pd.DataFrame, tolerance: float = 0.005, sink: dict | None = None) -> dict:
    """`lines` must come from opportunities.match_contracts. A line leaks if its unit price is more
    than `tolerance` (0.5%) above the contract price; leakage = (price - contract) x qty."""
    on = lines[lines["on_contract_supplier"]].copy()
    on["over"] = on["unit_price"] - on["contract_price"]
    on["leak"] = (on["over"] * on["qty"]).where(on["over"] > on["contract_price"] * tolerance, 0.0)
    if sink is not None:
        sink.update(dict(zip(on["line_id"], on["leak"], strict=True)))
    total_on = float(on["amount"].sum())
    leaking = on[on["leak"] > 0]
    contracted_items = lines[lines["contract_item"].notna()]

    def by(col: str) -> list[dict]:
        g = on.groupby(col).agg(
            spend=("amount", "sum"),
            leakage=("leak", "sum"),
            lines=("leak", "size"),
            leaking_lines=("leak", lambda s: int((s > 0).sum())),
        )
        g = g.sort_values("leakage", ascending=False).reset_index()
        rows = g.to_dict("records")
        for r in rows:
            r["spend"] = round(float(r["spend"]), 2)
            r["leakage"] = round(float(r["leakage"]), 2)
            r["compliance_rate"] = round(1 - r["leaking_lines"] / r["lines"], 4) if r["lines"] else 1.0
            if col == "l3":
                r["name"] = name_of(r["l3"])
        return rows

    ev = leaking.sort_values("leak", ascending=False).head(25)
    return {
        "matched_lines": int(len(on)),
        "matched_spend": round(total_on, 2),
        "contracted_item_spend": round(float(contracted_items["amount"].sum()), 2),
        "on_contract_share": round(total_on / float(contracted_items["amount"].sum()), 4)
        if len(contracted_items)
        else 0.0,
        "leaking_lines": int(len(leaking)),
        "price_compliance_rate": round(1 - len(leaking) / len(on), 4) if len(on) else 1.0,
        "leakage": round(float(on["leak"].sum()), 2),
        "by_supplier": by("supplier")[:25],
        "by_category": by("l3")[:25],
        "evidence": [
            {
                "line_id": int(r.line_id),
                "supplier": r.supplier,
                "description": r.description,
                "qty": float(r.qty),
                "unit_price": round(float(r.unit_price), 4),
                "contract_price": round(float(r.contract_price), 4),
                "leak": round(float(r.leak), 2),
            }
            for r in ev.itertuples()
        ],
    }
