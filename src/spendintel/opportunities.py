"""Savings levers. Each opportunity carries an estimated annual value and the invoice lines behind it.

Levers:
- price_variance: the same item bought at different prices. Value if every line above the median
  had paid the median (headline) or the 10th percentile price (upside).
- consolidation: fragmented categories. Value is an assumed rate on the spend outside the top
  suppliers, the standard planning number before a sourcing event.
- payment_terms: early payment discounts not taken, plus working capital from terms shorter than
  policy and from paying before the due date.
- maverick: a contract exists for the item, but the line went to a different supplier. Value is
  what the line would have saved at the contract price.
- duplicates: the same invoice keyed twice. Value is the duplicate amount (a recovery, not a saving).

Contract compliance (price leakage on lines that did go to the contracted supplier) is in
`compliance.py`, which uses the contract matching here.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .suppliers import within_one_edit
from .taxonomy import name_of

EVIDENCE_COLS = [
    "line_id",
    "invoice_no",
    "invoice_date",
    "supplier",
    "bu",
    "description",
    "qty",
    "unit_price",
    "amount",
]

ITEM_CANON = {
    "ss": "stainless",
    "zn": "zinc",
    "hd": "head",
    "glv": "glove",
    "gloves": "glove",
    "lg": "large",
    "l": "large",
    "wshr": "washer",
    "hyd": "hydraulic",
    "res": "resistant",
    "clr": "clear",
    "sht": "sheet",
    "rd": "round",
    "al": "aluminum",
    "alum": "aluminum",
    "aluminium": "aluminum",
    "frt": "freight",
    "svc": "service",
    "svcs": "service",
    "elec": "electric",
    "nat": "natural",
    "plt": "pallet",
    "conc": "concentrate",
    "whl": "wheel",
    "blk": "black",
    "mm": "",
    "x": "",
}
GENERIC = {
    "item",
    "pn",
    "per",
    "po",
    "svc",
    "chg",
    "ref",
    "misc",
    "supplies",
    "materials",
    "agreement",
    "see",
    "attached",
    "replacement",
    "parts",
    "order",
    "line",
    "monthly",
    "charges",
    "restock",
    "req",
    "quote",
    "rush",
    "urgent",
    "backorder",
    "qty",
    "1",
    "2",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "w",
    "and",
    "of",
    "the",
}


def item_signature(desc: str) -> str:
    """A normalized token signature for 'same item' matching. Drops quantities, reference numbers,
    and order chatter; maps abbreviations to one form. 'SS bolt M8x30 qty 500' and
    'stainless bolt m8x30' get the same signature."""
    s = desc.lower()
    s = re.sub(r"\bqty\s*\d+", " ", s)
    s = re.sub(r"[#:/,()]+", " ", s)
    toks = []
    for t in s.split():
        t = ITEM_CANON.get(t, t)
        if not t or t in GENERIC or (t.isdigit() and len(t) >= 5):
            continue
        toks.append(t)
    return " ".join(sorted(set(toks)))


def assign_item_groups(lines: pd.DataFrame, min_tokens: int = 2) -> pd.Series:
    """Item group = predicted L3 + signature, with one-typo tokens folded into the most common
    spelling seen in that category. Signatures with too few real tokens get no group."""
    sigs = lines["description"].map(item_signature)
    groups = pd.Series(index=lines.index, dtype=object)
    for l3, idx in lines.groupby("l3").groups.items():
        sub = sigs.loc[idx]
        freq = Counter(t for s in sub for t in s.split())
        common = [t for t, c in freq.most_common() if c >= 8]
        fix = {}
        for t, c in freq.items():
            if c < 8 and len(t) >= 4:
                for u in common:
                    if len(u) >= 4 and within_one_edit(t, u):
                        fix[t] = u
                        break
        fixed = sub.map(lambda s, fix=fix: " ".join(sorted({fix.get(t, t) for t in s.split()})))
        ok = fixed.map(lambda s: len(s.split()) >= min_tokens)
        groups.loc[idx] = np.where(ok, l3 + "|" + fixed, None)
    return groups


def _evidence(df: pd.DataFrame, value_col: str | None = None, n: int = 8) -> list[dict]:
    d = df.sort_values(value_col, ascending=False) if value_col else df
    cols = [c for c in EVIDENCE_COLS if c in d.columns]
    if value_col and value_col not in cols:
        cols.append(value_col)
    out = d.head(n)[cols].copy()
    if "invoice_date" in out:
        out["invoice_date"] = out["invoice_date"].dt.strftime("%Y-%m-%d")
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in out.to_dict("records")]


def price_variance(
    lines: pd.DataFrame, min_lines: int = 6, outlier: float = 3.0, sink: dict | None = None
) -> list[dict]:
    """Per item group with at least `min_lines` lines from 2+ suppliers or business units.
    `sink`, if given, collects the saving per line_id so overlapping levers can be netted."""
    df = lines[lines["item_group"].notna() & (lines["unit_price"] > 0)]
    opps = []
    for grp, sub in df.groupby("item_group"):
        if len(sub) < min_lines or (sub["supplier"].nunique() < 2 and sub["bu"].nunique() < 2):
            continue
        med = float(sub["unit_price"].median())
        sub = sub[sub["unit_price"] <= outlier * med]  # a 3x price is a unit of measure problem, not a saving
        p10 = float(sub["unit_price"].quantile(0.10))
        over_med = (sub["unit_price"] - med).clip(lower=0) * sub["qty"]
        over_best = (sub["unit_price"] - p10).clip(lower=0) * sub["qty"]
        value = float(over_med.sum())
        if value < 50:
            continue
        if sink is not None:
            sink.update(dict(zip(sub["line_id"], over_med, strict=True)))
        ev = sub.assign(excess=over_med)
        by_bu = ev.groupby("bu")["excess"].sum().sort_values(ascending=False)
        by_sup = sub.groupby("supplier")["unit_price"].median().sort_values()
        l3 = grp.split("|")[0]
        opps.append(
            {
                "lever": "price_variance",
                "title": f"{grp.split('|')[1]}: unit prices from {sub['unit_price'].min():,.2f} "
                f"to {sub['unit_price'].max():,.2f}",
                "category": l3,
                "category_name": name_of(l3),
                "value": round(value, 2),
                "upside": round(float(over_best.sum()), 2),
                "spend": round(float(sub["amount"].sum()), 2),
                "details": {
                    "median_price": round(med, 4),
                    "p10_price": round(p10, 4),
                    "lines": int(len(sub)),
                    "excess_by_bu": {k: round(float(v), 2) for k, v in by_bu.items()},
                    "median_price_by_supplier": {k: round(float(v), 4) for k, v in by_sup.head(6).items()},
                },
                "evidence": _evidence(ev, "excess"),
            }
        )
    return opps


def consolidation(
    lines: pd.DataFrame,
    keep: int = 3,
    rate: float = 0.06,
    min_suppliers: int = 6,
    max_hhi: float = 2500.0,
    addressable_only: bool = True,
) -> list[dict]:
    """Fragmented L3 categories: value = rate x spend outside the top `keep` suppliers."""
    from .cube import hhi

    df = lines[lines["addressable"]] if addressable_only else lines
    opps = []
    for l3, sub in df.groupby("l3"):
        by_sup = sub.groupby("supplier")["amount"].sum().sort_values(ascending=False)
        h = hhi(by_sup)
        if len(by_sup) < min_suppliers or h > max_hhi:
            continue
        moved = float(by_sup.iloc[keep:].sum())
        value = moved * rate
        if value < 100:
            continue
        movers = by_sup.iloc[keep:]
        ev = sub[sub["supplier"].isin(movers.index)]
        opps.append(
            {
                "lever": "consolidation",
                "title": f"{name_of(l3)}: {len(by_sup)} suppliers, HHI {h:,.0f}",
                "category": l3,
                "category_name": name_of(l3),
                "value": round(value, 2),
                "spend": round(float(by_sup.sum()), 2),
                "details": {
                    "suppliers": int(len(by_sup)),
                    "hhi": round(h, 1),
                    "keep": [{"supplier": k, "spend": round(float(v), 2)} for k, v in by_sup.head(keep).items()],
                    "move": [{"supplier": k, "spend": round(float(v), 2)} for k, v in movers.head(15).items()],
                    "moved_spend": round(moved, 2),
                    "assumed_rate": rate,
                },
                "evidence": _evidence(ev, "amount"),
            }
        )
    return opps


def invoices(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per invoice: supplier cluster, number, dates, terms, total."""
    return lines.groupby(["supplier_id", "invoice_no", "erp"], as_index=False).agg(
        supplier=("supplier", "first"),
        bu=("bu", "first"),
        invoice_date=("invoice_date", "first"),
        paid_date=("paid_date", "first"),
        net_days=("net_days", "first"),
        disc_pct=("disc_pct", "first"),
        disc_days=("disc_days", "first"),
        amount=("amount", "sum"),
        lines=("line_id", "size"),
        line_ids=("line_id", list),
        l3=("l3", "first"),
    )


def payment_terms(
    lines: pd.DataFrame, policy_days: int = 60, cost_of_capital: float = 0.08, early_grace: int = 5
) -> list[dict]:
    """Three findings per supplier, summed across invoices:
    - discount_missed: terms offered a discount and the invoice was paid after the discount window.
      Value = discount % x invoice amount.
    - short_terms: net days below policy. Value = amount x (policy - net) / 365 x cost of capital.
    - paid_early: paid more than `early_grace` days before due with no discount taken.
      Value = amount x days early / 365 x cost of capital."""
    inv = invoices(lines)
    inv["days_to_pay"] = (inv["paid_date"] - inv["invoice_date"]).dt.days
    disc = inv[(inv["disc_pct"] > 0) & (inv["days_to_pay"] > inv["disc_days"])].copy()
    disc["value"] = disc["amount"] * disc["disc_pct"] / 100
    short = inv[inv["net_days"] < policy_days].copy()
    short["value"] = short["amount"] * (policy_days - short["net_days"]) / 365 * cost_of_capital
    early = inv[
        (inv["days_to_pay"] < inv["net_days"] - early_grace)
        & ~((inv["disc_pct"] > 0) & (inv["days_to_pay"] <= inv["disc_days"]))
    ].copy()
    early["value"] = early["amount"] * (early["net_days"] - early["days_to_pay"]) / 365 * cost_of_capital
    opps = []
    for kind, df, label in (
        ("discount_missed", disc, "early payment discount not taken"),
        ("short_terms", short, f"terms shorter than the {policy_days} day policy"),
        ("paid_early", early, "paid before due date with no discount"),
    ):
        for sup, sub in df.groupby("supplier"):
            v = float(sub["value"].sum())
            if v < 50:
                continue
            ev = sub.sort_values("value", ascending=False).head(8)
            opps.append(
                {
                    "lever": "payment_terms",
                    "title": f"{sup}: {label}",
                    "category": sub["l3"].mode().iloc[0],
                    "category_name": name_of(sub["l3"].mode().iloc[0]),
                    "supplier": sup,
                    "value": round(v, 2),
                    "spend": round(float(sub["amount"].sum()), 2),
                    "details": {
                        "kind": kind,
                        "invoices": int(len(sub)),
                        "terms": f"{int(sub['disc_pct'].iloc[0])}/{int(sub['disc_days'].iloc[0])} net "
                        f"{int(sub['net_days'].iloc[0])}"
                        if kind == "discount_missed"
                        else f"net {int(sub['net_days'].iloc[0])}",
                        "avg_days_to_pay": round(float(sub["days_to_pay"].mean()), 1),
                    },
                    "evidence": [
                        {
                            "invoice_no": r.invoice_no,
                            "invoice_date": r.invoice_date.strftime("%Y-%m-%d"),
                            "paid_date": r.paid_date.strftime("%Y-%m-%d"),
                            "days_to_pay": int(r.days_to_pay),
                            "amount": round(float(r.amount), 2),
                            "value": round(float(r.value), 2),
                            "line_ids": list(r.line_ids)[:5],
                        }
                        for r in ev.itertuples()
                    ],
                }
            )
    return opps


# ----------------------------------------------------------------- contracts


def match_contracts(lines: pd.DataFrame, contracts: pd.DataFrame, min_score: float = 0.55) -> pd.DataFrame:
    """Match each line to a contracted item by description similarity within its predicted category.

    `contracts` needs: contract_id, supplier_id (resolved cluster), l3, item_description,
    contract_price. Returns lines with contract_item (an item id shared by every contract for the
    same item), match_score, and whether the line's supplier holds a contract for that item."""
    out = lines.copy()
    out["contract_item"] = None
    out["match_score"] = 0.0
    if contracts.empty:
        out["on_contract_supplier"] = False
        out["contract_price"] = np.nan
        out["best_contract_price"] = np.nan
        return out
    items = contracts.drop_duplicates("item_description")[["item_description", "l3"]].reset_index(drop=True)
    items["contract_item"] = [f"K{i:03d}" for i in range(len(items))]
    alias_src = contracts.drop_duplicates("item_description")
    aliases = alias_src["aliases"] if "aliases" in alias_src else alias_src["item_description"]
    # One row per (contract item, alias). A line matches an item if it is close to any alias.
    alias_rows = [
        (k, l3, a.strip())
        for k, l3, al in zip(items["contract_item"], items["l3"], aliases, strict=True)
        for a in str(al).split("|")
        if a.strip()
    ]
    al = pd.DataFrame(alias_rows, columns=["contract_item", "l3", "alias"])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True)
    vec.fit(pd.concat([al["alias"].str.lower(), out["description"].str.lower()]))
    for l3, idx in out.groupby("l3").groups.items():
        cand = al[al["l3"] == l3]
        if cand.empty:
            continue
        a = vec.transform(out.loc[idx, "description"].str.lower())
        b = vec.transform(cand["alias"].str.lower())
        sim = (a @ b.T).toarray()
        best = sim.argmax(axis=1)
        score = sim[np.arange(len(best)), best]
        hit = score >= min_score
        sel = np.asarray(idx)[hit]
        out.loc[sel, "contract_item"] = cand["contract_item"].to_numpy()[best[hit]]
        out.loc[sel, "match_score"] = score[hit]
    c = contracts.merge(items[["item_description", "contract_item"]], on="item_description")
    price = {(r.contract_item, r.supplier_id): r.contract_price for r in c.itertuples()}
    best_price = c.groupby("contract_item")["contract_price"].min().to_dict()
    keys = list(zip(out["contract_item"], out["supplier_id"], strict=True))
    out["contract_price"] = [price.get(k, np.nan) for k in keys]
    out["on_contract_supplier"] = out["contract_price"].notna()
    out["best_contract_price"] = out["contract_item"].map(best_price)
    return out


def maverick(lines: pd.DataFrame, sink: dict | None = None) -> list[dict]:
    """Lines for a contracted item that went to a supplier without a contract for it."""
    off = lines[lines["contract_item"].notna() & ~lines["on_contract_supplier"]].copy()
    off["saving"] = (off["unit_price"] - off["best_contract_price"]).clip(lower=0) * off["qty"]
    if sink is not None:
        sink.update(dict(zip(off["line_id"], off["saving"], strict=True)))
    opps = []
    for (l3, item), sub in off.groupby(["l3", "contract_item"]):
        v = float(sub["saving"].sum())
        if v < 50:
            continue
        opps.append(
            {
                "lever": "maverick",
                "title": f"{name_of(l3)}: contracted item bought from {sub['supplier'].nunique()} "
                "suppliers without a contract",
                "category": l3,
                "category_name": name_of(l3),
                "value": round(v, 2),
                "spend": round(float(sub["amount"].sum()), 2),
                "details": {
                    "contract_item": item,
                    "contract_price": round(float(sub["best_contract_price"].iloc[0]), 4),
                    "lines": int(len(sub)),
                    "no_po_share": round(float((~sub["has_po"]).mean()), 4),
                    "by_bu": {k: round(float(v), 2) for k, v in sub.groupby("bu")["amount"].sum().items()},
                },
                "evidence": _evidence(sub, "saving"),
            }
        )
    return opps


# ----------------------------------------------------------------- duplicates

_PREFIX = re.compile(r"^(inv|in|no|nr)+", re.I)


def norm_invoice_no(s: str) -> str:
    """'INV-0010234' and 'inv10234' both become '10234'."""
    t = re.sub(r"[^0-9a-z]", "", str(s).lower())
    t = _PREFIX.sub("", t)
    return t.lstrip("0") or "0"


def invoice_no_similar(a: str, b: str) -> float:
    """1.0 same after normalizing, 0.9 one edit apart or a one-character suffix, else 0.
    Numbers 1 to 3 apart (10234 and 10235) score 0: that is the supplier's next invoice."""
    na, nb = norm_invoice_no(a), norm_invoice_no(b)
    if na == nb:
        return 1.0
    if (
        na.isdigit()
        and nb.isdigit()
        and len(na) == len(nb)
        and abs(int(na) - int(nb)) <= 3
        and sorted(na) != sorted(nb)
    ):
        return 0.0  # consecutive numbers: the supplier's next invoice, not a re-key
    if within_one_edit(na, nb):
        return 0.9
    return 0.0


def duplicates(lines: pd.DataFrame, window_days: int = 10, amount_tol: float = 0.01) -> list[dict]:
    """Same supplier cluster, same invoice total (to the cent), invoice dates within the window, and
    invoice numbers equal or one edit apart after normalizing. Sorting by (supplier, amount) makes the
    candidate search linear instead of all pairs."""
    inv = invoices(lines).sort_values(["supplier_id", "amount", "invoice_date"]).reset_index(drop=True)
    found = []
    by_key = defaultdict(list)
    for r in inv.itertuples():
        by_key[(r.supplier_id, round(r.amount / amount_tol))].append(r)
    for rows in by_key.values():
        if len(rows) < 2:
            continue
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                gap = abs((b.invoice_date - a.invoice_date).days)
                if gap > window_days or (a.invoice_no == b.invoice_no and a.erp == b.erp):
                    continue
                sim = invoice_no_similar(a.invoice_no, b.invoice_no)
                if sim == 0:
                    continue
                first, second = (a, b) if (a.invoice_date, a.line_ids[0]) <= (b.invoice_date, b.line_ids[0]) else (b, a)
                found.append(
                    {
                        "lever": "duplicates",
                        "title": f"{a.supplier}: {first.invoice_no} and {second.invoice_no}, {gap} days apart",
                        "category": first.l3,
                        "category_name": name_of(first.l3),
                        "supplier": a.supplier,
                        "value": round(float(second.amount), 2),
                        "spend": round(float(second.amount), 2),
                        "details": {
                            "invoice_a": first.invoice_no,
                            "invoice_b": second.invoice_no,
                            "days_apart": gap,
                            "number_similarity": sim,
                            "lines": int(second.lines),
                        },
                        "evidence": _evidence(lines[lines["line_id"].isin(first.line_ids + second.line_ids)]),
                        "line_ids_original": list(first.line_ids),
                        "line_ids_duplicate": list(second.line_ids),
                    }
                )
    return found
