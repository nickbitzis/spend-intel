"""Map three ERP extracts onto one canonical invoice line schema.

Each ERP gets a small adapter: column renames, a date parser, an amount parser, and a payment
terms parser. Anything that does not parse is counted, not silently dropped.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

CANONICAL = [
    "line_id",
    "erp",
    "source_ref",
    "supplier_raw",
    "invoice_no",
    "invoice_date",
    "paid_date",
    "description",
    "qty",
    "unit_price",
    "amount",
    "gl_account",
    "cost_center",
    "bu",
    "terms_raw",
    "net_days",
    "disc_pct",
    "disc_days",
    "has_po",
]

_TERMS_PATTERNS = [
    (re.compile(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+)\s*N\s*(\d+)$", re.I), "disc"),  # 2/10N30
    (re.compile(r"^(\d+(?:\.\d+)?)%\s*(\d+)\s*(?:net\s*|/)\s*(\d+)$", re.I), "disc"),  # 2% 10 Net 30, 1%15/45
    (re.compile(r"^(?:NT|net\s*)(\d+)$", re.I), "net"),  # NT30, Net 30
    (re.compile(r"^(\d+)\s*days$", re.I), "net"),  # 30 days
    (re.compile(r"^(due|due on receipt|immediate)$", re.I), "due"),
]


def parse_terms(text: str) -> tuple[int, float, int]:
    """Payment terms as (net days, discount percent, discount days). Unknown text returns net 30,
    the most common contract default, and is counted by the caller."""
    t = (text or "").strip()
    for pat, kind in _TERMS_PATTERNS:
        m = pat.match(t)
        if not m:
            continue
        if kind == "disc":
            return int(m.group(3)), float(m.group(1)), int(m.group(2))
        if kind == "net":
            return int(m.group(1)), 0.0, 0
        return 0, 0.0, 0
    raise ValueError(f"unparsed terms: {text!r}")


def parse_amount(text) -> float:
    """'1,234.50' -> 1234.5. Also accepts numbers."""
    if isinstance(text, int | float | np.floating | np.integer):
        return float(text)
    return float(str(text).replace(",", "").strip())


def _terms_cols(series: pd.Series) -> tuple[pd.DataFrame, int]:
    parsed, bad = {}, 0
    for t in series.unique():
        try:
            parsed[t] = parse_terms(t)
        except ValueError:
            parsed[t] = (30, 0.0, 0)
            bad += 1
    arr = np.array([parsed[t] for t in series], dtype=float).reshape(-1, 3)
    return pd.DataFrame(
        {"net_days": arr[:, 0].astype(int), "disc_pct": arr[:, 1], "disc_days": arr[:, 2].astype(int)}
    ), bad


def from_erp_a(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "erp": "A",
            "source_ref": df["LINE_REF"],
            "supplier_raw": df["VENDOR_NAME"],
            "invoice_no": df["INV_NO"].astype(str),
            "invoice_date": pd.to_datetime(df["INV_DATE"].astype(str), format="%Y%m%d"),
            "paid_date": pd.to_datetime(df["PAID_DATE"].astype(str), format="%Y%m%d"),
            "description": df["LINE_TXT"],
            "qty": df["QTY"].astype(float),
            "unit_price": df["UNIT_PRICE"].astype(float),
            "amount": df["AMOUNT"].astype(float),
            "gl_account": df["GL_ACCT"].astype(str),
            "cost_center": df["COST_CTR"],
            "bu": df["PLANT"].str.title(),
            "terms_raw": df["PAY_TERMS"],
            "has_po": df["PO_NUM"].fillna("").astype(str).str.len() > 0,
        }
    )


def from_erp_b(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "erp": "B",
            "source_ref": df["line_ref"],
            "supplier_raw": df["supplier"],
            "invoice_no": df["invoice_number"].astype(str),
            "invoice_date": pd.to_datetime(df["invoice_date"], format="%m/%d/%Y"),
            "paid_date": pd.to_datetime(df["payment_date"], format="%m/%d/%Y"),
            "description": df["description"],
            "qty": df["quantity"].astype(float),
            "unit_price": df["price_each"].astype(float),
            "amount": df["extended_amount_cents"].astype(float) / 100.0,
            "gl_account": df["gl"].astype(str),
            "cost_center": df["dept"],
            "bu": df["business_unit"].str.title(),
            "terms_raw": df["terms"],
            "has_po": df["po_flag"].astype(str).str.upper().eq("Y"),
        }
    )


def from_erp_c(df: pd.DataFrame) -> pd.DataFrame:
    amount = df["Net Value"].map(parse_amount)
    qty = df["Qty"].astype(float)
    return pd.DataFrame(
        {
            "erp": "C",
            "source_ref": df["Line Ref"],
            "supplier_raw": df["Supplier Name"],
            "invoice_no": df["Doc #"].astype(str),
            "invoice_date": pd.to_datetime(df["Posting Date"], format="%d.%m.%Y"),
            "paid_date": pd.to_datetime(df["Clearing Date"], format="%d.%m.%Y"),
            "description": df["Item Text"],
            "qty": qty,
            # ERP C stores only the line value, so unit price is derived.
            "unit_price": (amount / qty.where(qty > 0)).round(4),
            "amount": amount,
            "gl_account": df["Account"].astype(str),
            "cost_center": df["Cost Center"],
            "bu": df["Division"],
            "terms_raw": df["Payment Terms"],
            "has_po": df["Purchase Order"].fillna("").astype(str).str.len() > 0,
        }
    )


ADAPTERS = {"A": from_erp_a, "B": from_erp_b, "C": from_erp_c}


def line_id_from_ref(ref: str) -> int:
    return int(re.sub(r"\D", "", ref))


def unify(raw: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """All ERPs into one frame. Returns the frame and a small data quality report."""
    parts = []
    for erp, df in raw.items():
        if len(df):
            parts.append(ADAPTERS[erp](df))
    lines = pd.concat(parts, ignore_index=True)
    terms, bad_terms = _terms_cols(lines["terms_raw"])
    lines = pd.concat([lines, terms], axis=1)
    lines["line_id"] = lines["source_ref"].map(line_id_from_ref)
    lines = lines.sort_values("line_id").reset_index(drop=True)
    lines["month"] = lines["invoice_date"].dt.strftime("%Y-%m")
    quality = {
        "lines": int(len(lines)),
        "by_erp": {k: int(v) for k, v in lines["erp"].value_counts().sort_index().items()},
        "unparsed_terms": int(bad_terms),
        "missing_unit_price": int(lines["unit_price"].isna().sum()),
        "no_po_share": round(float((~lines["has_po"]).mean()), 4),
        "distinct_supplier_strings": int(lines["supplier_raw"].nunique()),
        "distinct_descriptions": int(lines["description"].nunique()),
        "distinct_gl_accounts": int(lines["gl_account"].nunique()),
    }
    return lines[[*CANONICAL, "month"]], quality
