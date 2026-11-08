"""Seeded synthetic accounts payable data for Harrow Valley Manufacturing, a made-up company.

One year (2025) of invoice lines from three ERPs that each write the same facts differently:

- ERP A (plants east, Fabrication and Castings): fixed upper-case columns, dates as YYYYMMDD,
  terms codes like NT30 and 2/10N30, GL accounts as 6 digit numbers.
- ERP B (plants west, Assembly and Plastics): lower-case columns, dates as MM/DD/YYYY, amounts in
  cents, terms written out ("2% 10 Net 30"), GL as "6100-200", a Y/N PO flag.
- ERP C (corporate and distribution): title-case columns, dates as DD.MM.YYYY, amounts as text with
  thousands separators, GL as mnemonic accounts ("EXP-MRO-02"), terms like "1%15/45".

The same supplier shows up under different spellings, descriptions carry typos and abbreviations,
and the GL account a line is booked to depends on the business unit and the clerk, not just on what
was bought. Ground truth (true parent supplier, true category, true item, injected duplicates) is
kept in a separate table so the pipeline never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .taxonomy import L2_NAMES, LEAVES, Leaf, l1_of

YEAR = 2025
BUS = ("Fabrication", "Castings", "Assembly", "Plastics", "Distribution", "Corporate")
BU_ERP = {"Fabrication": "A", "Castings": "A", "Assembly": "B", "Plastics": "B", "Distribution": "C", "Corporate": "C"}
BU_WEIGHT = np.array([0.24, 0.16, 0.22, 0.16, 0.12, 0.10])

# How much each business unit buys from each L1 category (relative weights).
BU_MIX = {
    "Fabrication": {"MR": 3, "RM": 5, "PK": 1, "LG": 1.5, "IT": 0.5, "PS": 0.6, "FC": 1, "EL": 1.6},
    "Castings": {"MR": 3, "RM": 5, "PK": 0.8, "LG": 1.2, "IT": 0.4, "PS": 0.5, "FC": 1.2, "EL": 1.2},
    "Assembly": {"MR": 2.5, "RM": 2, "PK": 2, "LG": 2, "IT": 0.6, "PS": 0.8, "FC": 1, "EL": 2.4},
    "Plastics": {"MR": 1.6, "RM": 5, "PK": 2.2, "LG": 1.5, "IT": 0.4, "PS": 0.5, "FC": 1, "EL": 1},
    "Distribution": {"MR": 1, "RM": 0.1, "PK": 3, "LG": 5, "IT": 0.6, "PS": 0.6, "FC": 1, "EL": 0.5},
    "Corporate": {"MR": 0.3, "RM": 0, "PK": 0.2, "LG": 0.6, "IT": 4, "PS": 4, "FC": 1.5, "EL": 0.2},
}

# Typical quantity per line by unit of measure (lognormal median).
QTY_MEDIAN = {
    "ea": 60,
    "box": 12,
    "pair": 48,
    "tube": 24,
    "pail": 4,
    "drum": 2,
    "cwt": 120,
    "lb": 2500,
    "ft": 400,
    "roll": 24,
    "coil": 4,
    "shipment": 1,
    "load": 1,
    "pkg": 6,
    "pallet-mo": 180,
    "pallet": 160,
    "seat-yr": 6,
    "seat-mo": 80,
    "yr": 1,
    "line-mo": 40,
    "mo": 1,
    "day": 4,
    "hr": 60,
    "case": 10,
    "gal": 20,
    "mwh": 300,
    "dth": 2200,
    "visit": 1,
}
# Lines whose quantity is effectively 1 (a bill, not a count of things).
UNIT_SINGLE = {"shipment", "load", "yr", "mo", "visit"}

PREFIXES = (
    "Acme",
    "Apex",
    "Brightline",
    "Calder",
    "Dunmore",
    "Eastgate",
    "Fairhaven",
    "Granite",
    "Halston",
    "Ironwood",
    "Juniper",
    "Keystone",
    "Lakeshore",
    "Meridian",
    "Northfield",
    "Oakridge",
    "Pinecrest",
    "Quarry",
    "Redstone",
    "Summit",
    "Tidewater",
    "Unity",
    "Valley",
    "Westbrook",
    "Yardley",
    "Zephyr",
    "Alder",
    "Beacon",
    "Cedar",
    "Delta",
    "Evergreen",
    "Frontier",
    "Gateway",
    "Harbor",
    "Ingram",
    "Jasper",
    "Kestrel",
    "Lincoln",
    "Maple",
    "Nordic",
    "Orchard",
    "Pioneer",
    "Riverside",
    "Sterling",
    "Tristate",
    "Vantage",
    "Whitfield",
    "Crescent",
    "Bluewater",
    "Copperfield",
    "Stonebridge",
    "Highland",
    "Midstate",
    "Prairie",
    "Silverline",
    "Trident",
)
OWNER_NAMES = ("Becker", "Morales", "Okafor", "Lindqvist", "Patel", "Novak", "Haines", "Ferreira", "Kowalski", "Tanaka")
INDUSTRY_WORDS = {
    "MR-FAS": ("Fastener", "Bolt and Nut", "Industrial Supply", "Fastening Systems"),
    "MR-PPE": ("Safety Products", "Safety Supply", "Industrial Supply", "Protective Wear"),
    "MR-CUT": ("Tool", "Cutting Tools", "Abrasives", "Industrial Supply"),
    "MR-LUB": ("Lubricants", "Oil", "Fluid Technologies", "Industrial Supply"),
    "RM-MET": ("Steel", "Metals", "Metal Service Center", "Alloys"),
    "RM-PLS": ("Polymers", "Plastics", "Resin Distribution", "Plastic Supply"),
    "PK-COR": ("Packaging", "Container", "Box", "Corrugated"),
    "PK-PAL": ("Pallet", "Pallet and Crate", "Wood Products"),
    "PK-FLM": ("Packaging Supply", "Film Products", "Packaging"),
    "LG-FRT": ("Freight", "Logistics", "Transport", "Express", "Carriers"),
    "LG-WHS": ("Warehousing", "Logistics", "Distribution Services"),
    "IT-HW": ("Technology", "Computer Systems", "Tech Solutions"),
    "IT-SW": ("Software", "Systems", "Digital"),
    "IT-TEL": ("Communications", "Telecom", "Networks"),
    "PS-CON": ("Consulting", "Advisors", "Partners", "Engineering Group"),
    "PS-LEG": ("Law Group", "and Associates", "Accounting", "CPAs"),
    "PS-STF": ("Staffing", "Workforce Solutions", "Personnel"),
    "FC-JAN": ("Janitorial Supply", "Facility Services", "Sanitation"),
    "FC-UTL": ("Power and Light", "Energy", "Gas Company", "Electric Cooperative"),
    "FC-MNT": ("Mechanical", "Building Services", "Facility Services", "HVAC"),
    "EL-CMP": ("Electric Supply", "Electrical", "Wire and Cable", "Motor and Drive"),
    "EL-AUT": ("Automation", "Controls", "Industrial Electronics"),
}
LEGAL_SUFFIXES = ("Inc", "LLC", "Co", "Corp", "Ltd", "Company", "Incorporated", "LP", "")
ABBREV = {
    "Industrial": "Ind.",
    "Supply": "Sply",
    "Company": "Co",
    "Corporation": "Corp",
    "Manufacturing": "Mfg",
    "Logistics": "Logistic",
    "Technologies": "Tech",
    "Distribution": "Dist",
    "Services": "Svcs",
    "Systems": "Sys",
    "Electrical": "Elec",
    "Electric": "Elec",
    "Products": "Prod",
    "Associates": "Assoc",
    "International": "Intl",
    "Packaging": "Pkg",
    "Center": "Ctr",
    "Engineering": "Eng",
}
DIVISIONS = ("Midwest Division", "East", "Remit Dept", "Branch 12", "Ohio", "dba Direct")

ERP_TERMS = {
    "A": {
        (30, 0, 0): "NT30",
        (45, 0, 0): "NT45",
        (60, 0, 0): "NT60",
        (15, 0, 0): "NT15",
        (30, 2, 10): "2/10N30",
        (45, 1, 15): "1/15N45",
        (0, 0, 0): "DUE",
        (90, 0, 0): "NT90",
    },
    "B": {
        (30, 0, 0): "Net 30",
        (45, 0, 0): "Net 45",
        (60, 0, 0): "Net 60",
        (15, 0, 0): "Net 15",
        (30, 2, 10): "2% 10 Net 30",
        (45, 1, 15): "1% 15 Net 45",
        (0, 0, 0): "Due on receipt",
        (90, 0, 0): "Net 90",
    },
    "C": {
        (30, 0, 0): "30 days",
        (45, 0, 0): "45 days",
        (60, 0, 0): "60 days",
        (15, 0, 0): "15 days",
        (30, 2, 10): "2%10/30",
        (45, 1, 15): "1%15/45",
        (0, 0, 0): "immediate",
        (90, 0, 0): "90 days",
    },
}
# Supplier standard terms, as (net days, discount %, discount days).
TERMS_CHOICES = [(30, 0, 0), (45, 0, 0), (60, 0, 0), (15, 0, 0), (30, 2, 10), (45, 1, 15), (0, 0, 0), (90, 0, 0)]
TERMS_P = np.array([0.34, 0.16, 0.18, 0.06, 0.12, 0.06, 0.04, 0.04])

TERSE = (
    "misc supplies per quote {q}",
    "item {q}",
    "PN {q}-{s} per PO",
    "svc chg ref {q}",
    "materials per agreement",
    "see attached {q}",
    "replacement parts",
    "order {q} line 2",
    "monthly charges",
    "restock per req {q}",
)


@dataclass
class Dataset:
    raw: dict[str, pd.DataFrame]  # ERP letter -> raw extract
    truth: pd.DataFrame  # line_id -> true parent, true item, true category, duplicate_of
    suppliers: pd.DataFrame  # true parents
    contracts: pd.DataFrame  # parent_id, item_key, contract price, terms
    labeled_ids: np.ndarray  # line_ids an analyst labeled for training


def _pick(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 4 or not word.isalpha():
        return word
    i = int(rng.integers(1, len(word) - 1))
    op = rng.integers(0, 3)
    if op == 0:
        return word[:i] + word[i + 1 :]  # drop
    if op == 1:
        return word[: i - 1] + word[i] + word[i - 1] + word[i + 1 :]  # swap
    return word[:i] + word[i] + word[i:]  # double


def make_suppliers(rng: np.random.Generator, n_parents: int) -> pd.DataFrame:
    """True parent suppliers. Prefixes are reused across industries on purpose, so 'Apex Packaging'
    and 'Apex Plastics' are different companies that a careless matcher would merge."""
    l2s = list(L2_NAMES)
    # Bigger categories (more leaves, more spend) get more suppliers, and every L2 gets at least four.
    leaf_count = {l2: sum(1 for lf in LEAVES if lf.l2 == l2) for l2 in l2s}
    w = np.array([leaf_count[l2] ** 1.2 for l2 in l2s], dtype=float)
    alloc = np.maximum(4, np.round(w / w.sum() * n_parents)).astype(int)
    rows, used = [], set()
    pid = 0
    for l2, k in zip(l2s, alloc, strict=True):
        for _ in range(k):
            for _attempt in range(50):
                if rng.random() < 0.12:
                    first = str(rng.choice(OWNER_NAMES))
                    if rng.random() < 0.5:
                        first += f" {rng.choice(OWNER_NAMES)}"
                else:
                    first = str(rng.choice(PREFIXES))
                ind = str(rng.choice(INDUSTRY_WORDS[l2]))
                name = f"{first} {ind}"
                if name not in used:
                    break
            used.add(name)
            suffix = str(rng.choice(LEGAL_SUFFIXES, p=[0.3, 0.25, 0.12, 0.1, 0.05, 0.06, 0.04, 0.03, 0.05]))
            terms = TERMS_CHOICES[min(int(np.searchsorted(np.cumsum(TERMS_P), rng.random())), len(TERMS_P) - 1)]
            # A second category for some suppliers (an industrial distributor also sells PPE, say).
            other = str(rng.choice([x for x in l2s if l1_of(x) == l1_of(l2) and x != l2] or [l2]))
            rows.append(
                {
                    "parent_id": f"S{pid:04d}",
                    "canonical_name": f"{name} {suffix}".strip(),
                    "base_name": name,
                    "primary_l2": l2,
                    "secondary_l2": other if rng.random() < 0.3 else "",
                    "size": float(rng.lognormal(0, 1.1)),
                    "price_factor": float(rng.lognormal(0, 0.07)),
                    "net_days": terms[0],
                    "disc_pct": terms[1],
                    "disc_days": terms[2],
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def name_variants(base: str, suffix_name: str, rng: np.random.Generator, k: int) -> list[str]:
    """Spellings of one supplier as different clerks and ERPs would key it in."""
    out = [suffix_name]
    tries = 0
    while len(out) < k and tries < 40:
        tries += 1
        words = base.split()
        r = rng.random()
        if r < 0.25:
            words = [ABBREV.get(w, w) for w in words]
        elif r < 0.35:
            words = ["&" if w == "and" else w for w in words]
        elif r < 0.45 and len(words) > 2 and words[1] != "and":
            words = words[:2]
        elif r < 0.55:
            j = int(rng.integers(0, len(words)))
            words[j] = _typo(words[j], rng)
        name = " ".join(words)
        suf = str(rng.choice(LEGAL_SUFFIXES))
        if suf and rng.random() < 0.5:
            suf = suf + "." if suf in ("Inc", "Co", "Corp", "Ltd") else suf
        name = f"{name}{',' if rng.random() < 0.3 and suf else ''} {suf}".strip()
        r2 = rng.random()
        if r2 < 0.3:
            name = name.upper()
        elif r2 < 0.4:
            name = name.lower()
        if rng.random() < 0.08:
            name = f"{name} - {rng.choice(DIVISIONS)}"
        if rng.random() < 0.05:
            name = f"The {name}"
        if name not in out:
            out.append(name)
    return out


def _describe(leaf: Leaf, item, rng: np.random.Generator, qty: float) -> tuple[str, float]:
    """Pick and dirty a description. Returns the text and a difficulty from 0 (clear) to 1 (terse)."""
    if rng.random() < 0.11:
        t = _pick(rng, TERSE).format(q=int(rng.integers(1000, 99999)), s=_pick(rng, "ABCDEFG"))
        return t, 1.0
    t = _pick(rng, item.texts).format(n=int(qty) if rng.random() < 0.7 else int(rng.integers(10000, 999999)), pk=100)
    diff = 0.1
    words = t.split()
    if rng.random() < 0.3:
        for _ in range(1 + int(rng.random() < 0.3)):
            j = int(rng.integers(0, len(words)))
            words[j] = _typo(words[j], rng)
        diff += 0.25
    if rng.random() < 0.12 and len(words) > 3:
        words = words[: max(2, len(words) - 2)]
        diff += 0.25
    t = " ".join(words)
    r = rng.random()
    if r < 0.25:
        t = t.upper()
    elif r < 0.4:
        t = t.lower()
    if rng.random() < 0.15:
        t = f"{t} {_pick(rng, ('per quote', 'rush', 'restock', 'urgent', 'line 1', 'backorder'))}"
    return t, min(diff, 0.8)


def _inv_variant(inv: str, rng: np.random.Generator) -> str:
    """How a duplicate invoice number gets re-keyed: a dash dropped, a suffix, a transposition."""
    digits = "".join(ch for ch in inv if ch.isdigit())
    r = rng.random()
    if r < 0.3:
        return inv.replace("-", "")
    if r < 0.5:
        return f"{inv}A"
    if r < 0.7 and len(digits) > 3:
        i = int(rng.integers(0, len(digits) - 1))
        d = list(digits)
        d[i], d[i + 1] = d[i + 1], d[i]
        return inv.replace(digits, "".join(d))
    if r < 0.85:
        return "0" + digits
    return inv.lower()


def generate(n_lines: int = 120_000, seed: int = 42, n_parents: int = 340, labeled_share: float = 0.06) -> Dataset:
    rng = np.random.default_rng(seed)
    sup = make_suppliers(rng, n_parents)

    # Variants per supplier: big suppliers used by many BUs collect more spellings.
    variants: dict[str, list[str]] = {}
    for r in sup.itertuples():
        k = 1 + int(min(5, rng.poisson(0.6 + 0.9 * min(r.size, 3))))
        variants[r.parent_id] = name_variants(r.base_name, r.canonical_name, rng, k)

    # Suppliers per L2, with spend weight by size (heavy tail) and a per-BU preference.
    by_l2: dict[str, list[int]] = {}
    for idx, r in enumerate(sup.itertuples()):
        by_l2.setdefault(r.primary_l2, []).append(idx)
        if r.secondary_l2:
            by_l2.setdefault(r.secondary_l2, []).append(idx)
    bu_pref = rng.lognormal(0, 0.9, size=(len(BUS), len(sup)))
    srec = sup.to_dict("records")
    sizes = sup["size"].to_numpy()
    cand_arr = {l2: np.array(v) for l2, v in by_l2.items()}

    # Contracts: for the biggest suppliers in each L2, about half of their items have a price.
    contracts = []
    leaf_by_l2: dict[str, list[Leaf]] = {}
    for lf in LEAVES:
        leaf_by_l2.setdefault(lf.l2, []).append(lf)
    for l2, idxs in by_l2.items():
        ranked = sorted(idxs, key=lambda i: -srec[i]["size"])[:3]
        for i in ranked:
            s = srec[i]
            if l2 not in (s["primary_l2"],) or rng.random() < 0.15:
                continue
            for lf in leaf_by_l2.get(l2, []):
                for it in lf.items:
                    if rng.random() < 0.55:
                        contracts.append(
                            {
                                "contract_id": f"C{len(contracts):04d}",
                                "parent_id": s["parent_id"],
                                "supplier_name": s["canonical_name"],
                                "l3": lf.code,
                                "item_description": it.texts[0],
                                # The contract catalog lists the item under each name the supplier uses.
                                "aliases": " | ".join(it.texts).replace(" qty {n}", "").replace(" {n}", ""),
                                "item_key": it.key,
                                "contract_price": round(it.price * s["price_factor"] * 0.92, 4),
                                "net_days": 60,
                                "disc_pct": 2 if rng.random() < 0.3 else 0,
                                "disc_days": 10,
                            }
                        )
    con = pd.DataFrame(contracts)
    con_lookup = {(c.parent_id, c.item_key): c for c in con.itertuples()}
    contracted_items: dict[str, list] = {}
    for c in con.itertuples():
        contracted_items.setdefault(c.item_key, []).append(c)

    # Leaf weights per BU.
    leaf_l1 = np.array([lf.l1 for lf in LEAVES])
    leaf_spend_w = np.array([np.mean([it.price for it in lf.items]) for lf in LEAVES])

    leaf_cum = []
    for bu in BUS:
        w = np.array([BU_MIX[bu][c] for c in leaf_l1]) / np.sqrt(leaf_spend_w)  # cheap items are bought more often
        leaf_cum.append(np.cumsum(w / w.sum()))
    bu_cum = np.cumsum(BU_WEIGHT / BU_WEIGHT.sum())

    def pick(cum: np.ndarray) -> int:
        return min(int(np.searchsorted(cum, rng.random())), len(cum) - 1)

    start = date(YEAR, 1, 1)
    rows = []
    truth = []
    line_id = 0
    inv_seq = 10000 + int(rng.integers(0, 5000))
    # Lines come in invoices of 1 to 4 lines from one supplier, one BU, one date.
    while line_id < n_lines:
        bu_i = pick(bu_cum)
        bu = BUS[bu_i]
        erp = BU_ERP[bu]
        leaf = LEAVES[pick(leaf_cum[bu_i])]
        cands = cand_arr[leaf.l2]
        sw = sizes[cands] * bu_pref[bu_i, cands]
        si = int(cands[pick(np.cumsum(sw / sw.sum()))])
        s = srec[si]
        day = int(rng.integers(0, 365))
        inv_date = start + timedelta(days=day)
        inv_seq += int(rng.integers(1, 4))
        inv_no = f"{('INV-', 'INV', '', 'IN')[si % 4]}{inv_seq}"  # each supplier has its own number format
        has_po = rng.random() < (0.78 if leaf.l1 in ("RM", "MR", "EL", "PK") else 0.45)
        variant = variants[s["parent_id"]][int(rng.integers(0, len(variants[s["parent_id"]])))]
        n_inv_lines = (1, 1, 1, 2, 2, 3, 4)[int(rng.integers(0, 7))]
        # Payment: paid on time-ish; the discount is taken only sometimes.
        terms = (int(s["net_days"]), int(s["disc_pct"]), int(s["disc_days"]))
        lines_this = []
        for _ in range(n_inv_lines):
            if line_id >= n_lines:
                break
            if rng.random() < 0.75:
                lf = leaf
            else:
                lf = LEAVES[pick(leaf_cum[bu_i])]
                if lf.l2 not in (s["primary_l2"], s["secondary_l2"]):
                    lf = leaf
            item = lf.items[int(rng.integers(0, len(lf.items)))]
            # Quantity scales so a typical line is worth a few thousand dollars; big-ticket items come in ones.
            med = max(1.0, min(QTY_MEDIAN[item.unit], 2500.0 / item.price))
            qty = (
                1.0
                if item.unit in UNIT_SINGLE or med <= 1.0
                else max(1.0, round(float(rng.lognormal(np.log(med), 0.7))))
            )
            contract = con_lookup.get((s["parent_id"], item.key))
            on_contract = contract is not None and has_po and rng.random() < 0.9
            if on_contract:
                price = contract.contract_price
            else:
                bu_factor = 1 + 0.05 * (bu_i - 2.5) / 2.5
                price = item.price * s["price_factor"] * bu_factor * float(rng.lognormal(0, 0.06))
                if not has_po:
                    price *= 1.06  # spot buys without a PO pay list price
            price = round(price, 4 if price < 1 else 2)
            desc, diff = _describe(lf, item, rng, qty)
            amount = round(price * qty, 2)
            lines_this.append((lf, item, qty, price, amount, desc, diff, contract is not None, on_contract))
        if not lines_this:
            break
        total = sum(x[4] for x in lines_this)
        took_disc = terms[1] > 0 and rng.random() < 0.35
        pay_days = terms[2] - int(rng.integers(0, 3)) if took_disc else terms[0] + int(rng.integers(-3, 12))
        pay_date = inv_date + timedelta(days=max(0, pay_days))
        gl_seed = int(rng.integers(0, 10))
        for lf, item, qty, price, amount, desc, diff, has_c, on_c in lines_this:
            rows.append(
                _render(
                    erp,
                    line_id,
                    variant,
                    inv_no,
                    inv_date,
                    pay_date,
                    desc,
                    qty,
                    price,
                    amount,
                    lf,
                    bu,
                    terms,
                    has_po,
                    gl_seed,
                    rng,
                )
            )
            truth.append(
                {
                    "line_id": line_id,
                    "parent_id": s["parent_id"],
                    "item_key": item.key,
                    "l3": lf.code,
                    "difficulty": diff,
                    "has_contract_for_pair": has_c,
                    "on_contract": on_c,
                    "invoice_total": round(total, 2),
                    "duplicate_of": -1,
                }
            )
            line_id += 1

    # Duplicate invoices: about 0.4% of invoices are keyed twice, a few days apart, with a re-typed number.
    tdf = pd.DataFrame(truth)
    rdf = pd.DataFrame(rows)
    groups = rdf.groupby("_inv_key").indices
    keys = sorted(groups)
    n_dup = max(3, int(len(keys) * 0.004))
    picks = rng.choice(len(keys), size=n_dup, replace=False)
    dup_rows, dup_truth = [], []
    for k in picks:
        orig = rdf.iloc[groups[keys[k]]]
        # Most ERPs block an exact repeat of supplier + invoice number, so the duplicates that get through
        # carry a re-keyed number and a later date.
        shift = int(rng.integers(1, 9))
        first_no = str(orig["_inv_no"].iloc[0])
        new_no = _inv_variant(first_no, rng)
        if new_no == first_no:
            new_no = first_no + "A"
        for _, o in orig.iterrows():
            t = tdf.loc[o["_line_id"]].to_dict()
            nr = _render(
                o["_erp"],
                line_id,
                o["_supplier"],
                new_no,
                o["_date"] + timedelta(days=shift),
                o["_pay"] + timedelta(days=shift),
                o["_desc"],
                o["_qty"],
                o["_price"],
                o["_amount"],
                _leaf(t["l3"]),
                o["_bu"],
                o["_terms"],
                o["_po"],
                int(rng.integers(0, 10)),
                rng,
            )
            dup_rows.append(nr)
            t.update({"line_id": line_id, "duplicate_of": int(o["_line_id"])})
            dup_truth.append(t)
            line_id += 1
    # Legitimate repeats: a second load on the same lane at the same rate a few days later, billed on
    # the next invoice number. Same supplier, same amount, close dates, numbers one digit apart. These
    # are NOT duplicates and a naive matcher flags them.
    singles = [k for k in keys if len(groups[k]) == 1]
    for k in rng.choice(len(singles), size=max(2, n_dup // 2), replace=False):
        o = rdf.iloc[groups[singles[k]][0]]
        no = str(o["_inv_no"])
        digits = "".join(ch for ch in no if ch.isdigit())
        new_no = no.replace(digits, str(int(digits) + int(rng.integers(1, 3))))
        shift = int(rng.integers(2, 7))
        t = tdf.loc[o["_line_id"]].to_dict()
        dup_rows.append(
            _render(
                o["_erp"],
                line_id,
                o["_supplier"],
                new_no,
                o["_date"] + timedelta(days=shift),
                o["_pay"] + timedelta(days=shift),
                o["_desc"],
                o["_qty"],
                o["_price"],
                o["_amount"],
                _leaf(t["l3"]),
                o["_bu"],
                o["_terms"],
                o["_po"],
                int(rng.integers(0, 10)),
                rng,
            )
        )
        t.update({"line_id": line_id, "duplicate_of": -1})
        dup_truth.append(t)
        line_id += 1
    rdf = pd.concat([rdf, pd.DataFrame(dup_rows)], ignore_index=True)
    tdf = pd.concat([tdf, pd.DataFrame(dup_truth)], ignore_index=True)

    raw = {}
    for erp in ("A", "B", "C"):
        part = rdf[rdf["_erp"] == erp]
        raw[erp] = (
            part[[c for c in part.columns if not c.startswith("_")]].dropna(axis=1, how="all").reset_index(drop=True)
        )
    labeled = rng.choice(tdf["line_id"].to_numpy(), size=int(len(tdf) * labeled_share), replace=False)
    return Dataset(raw=raw, truth=tdf, suppliers=sup, contracts=con, labeled_ids=np.sort(labeled))


def _leaf(code: str) -> Leaf:
    for lf in LEAVES:
        if lf.code == code:
            return lf
    raise KeyError(code)


# GL accounts. Each ERP has its own chart, and clerks pick from a few plausible accounts, so the GL
# account agrees with the true category only part of the time.
_GL_A = {
    "MR": "610200",
    "RM": "500100",
    "PK": "520300",
    "LG": "540100",
    "IT": "650100",
    "PS": "670100",
    "FC": "630100",
    "EL": "610400",
}
_GL_B = {
    "MR": "6100-200",
    "RM": "5000-100",
    "PK": "5200-300",
    "LG": "5400-100",
    "IT": "6500-100",
    "PS": "6700-100",
    "FC": "6300-100",
    "EL": "6100-400",
}
_GL_C = {
    "MR": "EXP-MRO-02",
    "RM": "COGS-MAT-01",
    "PK": "COGS-PKG-01",
    "LG": "EXP-FRT-01",
    "IT": "EXP-IT-01",
    "PS": "EXP-PRO-01",
    "FC": "EXP-FAC-01",
    "EL": "EXP-MRO-04",
}
_GL_CATCHALL = {"A": "699000", "B": "6999-000", "C": "EXP-MISC-99"}


def _gl(erp: str, l1: str, bu: str, gl_seed: int) -> str:
    table = {"A": _GL_A, "B": _GL_B, "C": _GL_C}[erp]
    if gl_seed < 2:
        return _GL_CATCHALL[erp]  # miscellaneous expense
    if gl_seed < 4:
        # A business unit habit: MRO, electrical, and facilities all go to one maintenance account.
        if l1 in ("MR", "EL", "FC"):
            return table["MR"]
        if l1 in ("PK", "LG"):
            return table["LG" if bu == "Distribution" else "PK"]
    return table[l1]


def _cost_center(bu: str, rng: np.random.Generator) -> str:
    return f"{bu[:3].upper()}-{int(rng.integers(100, 140))}"


def _render(
    erp, line_id, supplier, inv_no, inv_date, pay_date, desc, qty, price, amount, leaf, bu, terms, has_po, gl_seed, rng
) -> dict:
    gl = _gl(erp, leaf.l1, bu, gl_seed)
    cc = _cost_center(bu, rng)
    po = f"PO{int(rng.integers(400000, 499999))}" if has_po else ""
    base = {
        "_line_id": line_id,
        "_erp": erp,
        "_supplier": supplier,
        "_inv_no": inv_no,
        "_inv_key": f"{supplier}|{inv_no}",
        "_date": inv_date,
        "_pay": pay_date,
        "_desc": desc,
        "_qty": qty,
        "_price": price,
        "_amount": amount,
        "_bu": bu,
        "_terms": terms,
        "_po": has_po,
    }
    term_txt = ERP_TERMS[erp][terms]
    if erp == "A":
        return base | {
            "LINE_REF": f"A{line_id:07d}",
            "VENDOR_NAME": supplier,
            "INV_NO": inv_no,
            "INV_DATE": inv_date.strftime("%Y%m%d"),
            "PAID_DATE": pay_date.strftime("%Y%m%d"),
            "LINE_TXT": desc,
            "QTY": qty,
            "UNIT_PRICE": price,
            "AMOUNT": amount,
            "GL_ACCT": gl,
            "COST_CTR": cc,
            "PLANT": bu.upper(),
            "PAY_TERMS": term_txt,
            "PO_NUM": po,
        }
    if erp == "B":
        return base | {
            "line_ref": f"B-{line_id}",
            "supplier": supplier,
            "invoice_number": inv_no,
            "invoice_date": inv_date.strftime("%m/%d/%Y"),
            "payment_date": pay_date.strftime("%m/%d/%Y"),
            "description": desc,
            "quantity": qty,
            "price_each": price,
            "extended_amount_cents": int(round(amount * 100)),
            "gl": gl,
            "dept": cc,
            "business_unit": bu.lower(),
            "terms": term_txt,
            "po_flag": "Y" if has_po else "N",
        }
    return base | {
        "Line Ref": f"C/{line_id}",
        "Supplier Name": supplier,
        "Doc #": inv_no,
        "Posting Date": inv_date.strftime("%d.%m.%Y"),
        "Clearing Date": pay_date.strftime("%d.%m.%Y"),
        "Item Text": desc,
        "Qty": qty,
        "Net Value": f"{amount:,.2f}",
        "Account": gl,
        "Cost Center": cc,
        "Division": bu,
        "Payment Terms": term_txt,
        "Purchase Order": po,
    }
