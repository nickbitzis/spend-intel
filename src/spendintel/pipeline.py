"""End to end: raw ERP extracts in, spend cube, savings, compliance, and an evaluation out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import cube
from .classify import ModelTier, Policy, level_accuracy, run_cascade, tier_report
from .compliance import leakage
from .generator import Dataset, generate
from .ingest import unify
from .llm import HAIKU_SIM, OPUS_SIM, LineClassifier, SimulatedClassifier
from .opportunities import (
    assign_item_groups,
    consolidation,
    duplicates,
    match_contracts,
    maverick,
    payment_terms,
    price_variance,
)
from .suppliers import Clusters, ReviewBook, pairwise_scores, resolve
from .taxonomy import is_addressable, l1_of, l2_of

LEVERS = ("price_variance", "consolidation", "payment_terms", "maverick", "contract_leakage", "duplicates")
LEVER_NAMES = {
    "price_variance": "Price variance (pay the median)",
    "consolidation": "Supplier consolidation",
    "payment_terms": "Payment terms",
    "maverick": "Off-contract (maverick) spend",
    "contract_leakage": "Contract price leakage",
    "duplicates": "Duplicate invoices (recovery)",
}


@dataclass
class Config:
    lines: int = 120_000
    seed: int = 42
    parents: int = 340
    labeled_share: float = 0.06
    policy: Policy = field(default_factory=Policy)
    auto_threshold: float = 0.8
    review_threshold: float = 0.62
    consolidation_rate: float = 0.06
    policy_days: int = 60
    cost_of_capital: float = 0.08


@dataclass
class Results:
    lines: pd.DataFrame
    clusters: Clusters
    review: ReviewBook
    opportunities: list[dict]
    compliance: dict
    report: dict
    contracts: pd.DataFrame


def run(
    cfg: Config | None = None,
    data: Dataset | None = None,
    haiku: LineClassifier | None = None,
    opus: LineClassifier | None = None,
) -> Results:
    cfg = cfg or Config()
    timings = {}
    t0 = time.perf_counter()
    data = data or generate(cfg.lines, cfg.seed, cfg.parents, cfg.labeled_share)
    timings["generate_s"] = time.perf_counter() - t0

    t = time.perf_counter()
    lines, quality = unify(data.raw)
    truth = data.truth.set_index("line_id")
    timings["ingest_s"] = time.perf_counter() - t

    # Suppliers. Contract supplier names go into the same resolver so contracts join to clusters.
    t = time.perf_counter()
    weights = lines.groupby("supplier_raw")["amount"].sum().to_dict()
    names = list(weights) + [n for n in data.contracts.get("supplier_name", pd.Series(dtype=str)) if n not in weights]
    clusters = resolve(names, weights, auto=cfg.auto_threshold, review=cfg.review_threshold)
    lines["supplier_id"] = lines["supplier_raw"].map(clusters.cluster_of)
    lines["supplier"] = lines["supplier_id"].map(clusters.display)
    parent_of_name = (
        lines.assign(parent=lines["line_id"].map(truth["parent_id"]))
        .groupby("supplier_raw")["parent"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )
    sup_eval = pairwise_scores({n: clusters.cluster_of[n] for n in parent_of_name}, parent_of_name)
    timings["suppliers_s"] = time.perf_counter() - t

    # Classification. The labeled subset trains the model and keeps its human labels.
    t = time.perf_counter()
    lines["true_l3"] = lines["line_id"].map(truth["l3"])
    lines["difficulty"] = lines["line_id"].map(truth["difficulty"])
    labeled = lines["line_id"].isin(data.labeled_ids)
    lab = lines[labeled]
    model = ModelTier().fit(lab["description"].tolist(), lab["supplier_raw"].tolist(), lab["true_l3"].tolist())
    ev = lines[~labeled].copy()
    ev["key"] = ev["description"] + "||" + ev["supplier_raw"]
    keys = (
        ev.groupby("key", sort=False)
        .agg(
            description=("description", "first"),
            supplier=("supplier_raw", "first"),
            gl_account=("gl_account", "first"),
            gold_l3=("true_l3", "first"),
            difficulty=("difficulty", "first"),
        )
        .reset_index()
    )
    haiku = haiku or SimulatedClassifier(HAIKU_SIM)
    opus = opus or SimulatedClassifier(OPUS_SIM)
    co = run_cascade(keys, model, haiku, opus, cfg.policy)
    ev = ev.merge(co.keys[["key", "category", "confidence", "tier"]], on="key", how="left")
    lines = lines.merge(ev[["line_id", "category", "confidence", "tier"]], on="line_id", how="left")
    lines.loc[labeled.to_numpy(), ["category", "confidence", "tier"]] = (
        lines.loc[labeled.to_numpy(), ["true_l3"]].assign(c=1.0, t="labeled").to_numpy()
    )
    lines["confidence"] = lines["confidence"].astype(float)
    evl = lines[lines["tier"] != "labeled"]
    no_llm = run_cascade(
        keys, model, None, None, Policy(cfg.policy.t_rules, cfg.policy.t_model, cfg.policy.t_haiku, False)
    )
    nl = ev[["key", "true_l3", "amount"]].merge(no_llm.keys[["key", "category", "tier"]], on="key")
    rules_only = ev[["key", "true_l3"]].merge(co.keys[["key", "rule_cat", "model_cat"]], on="key")
    cls_eval = {
        "eval_lines": int(len(evl)),
        "labeled_lines": int(labeled.sum()),
        "cascade": level_accuracy(evl["category"], evl["true_l3"]),
        "cascade_spend_weighted": level_accuracy(evl["category"], evl["true_l3"], evl["amount"]),
        "no_llm": level_accuracy(nl["category"], nl["true_l3"]),
        "model_only": level_accuracy(rules_only["model_cat"], rules_only["true_l3"]),
        "rules_coverage": round(float(rules_only["rule_cat"].notna().mean()), 4),
        "tiers": tier_report(evl.assign(true_l3=evl["true_l3"]), len(keys), co.llm_cost),
        "llm_calls": co.llm_calls,
        "llm_simulated": co.simulated,
        "policy": {"t_rules": cfg.policy.t_rules, "t_model": cfg.policy.t_model, "t_haiku": cfg.policy.t_haiku},
    }
    # The GL account as a classifier: map each account to its most common true L1 and score it.
    gl_l1 = lines.assign(tl1=lines["true_l3"].map(l1_of)).groupby("gl_account")["tl1"].agg(lambda s: s.mode().iloc[0])
    gl_pred = lines["gl_account"].map(gl_l1)
    cls_eval["gl_account_l1_accuracy"] = round(float((gl_pred == lines["true_l3"].map(l1_of)).mean()), 4)
    timings["classify_s"] = time.perf_counter() - t

    # The analysis frame.
    t = time.perf_counter()
    lines["l3"] = lines["category"]
    lines["l2"] = lines["l3"].map(l2_of)
    lines["l1"] = lines["l3"].map(l1_of)
    lines["addressable"] = lines["l3"].map(is_addressable)
    lines["item_group"] = assign_item_groups(lines)

    con = data.contracts.copy()
    if not con.empty:
        con["supplier_id"] = con["supplier_name"].map(clusters.cluster_of)
    lines = match_contracts(lines, con)
    sinks = {"price_variance": {}, "maverick": {}, "contract_leakage": {}}
    comp = leakage(lines, sink=sinks["contract_leakage"])

    opps = []
    opps += price_variance(lines, sink=sinks["price_variance"])
    opps += consolidation(lines, rate=cfg.consolidation_rate)
    opps += payment_terms(lines, cfg.policy_days, cfg.cost_of_capital)
    opps += maverick(lines, sink=sinks["maverick"])
    dups = duplicates(lines)
    opps += dups
    for row in comp["by_supplier"]:
        if row["leakage"] > 0:
            ev_rows = [e for e in comp["evidence"] if e["supplier"] == row["supplier"]][:8]
            opps.append(
                {
                    "lever": "contract_leakage",
                    "title": f"{row['supplier']}: invoice prices above contract",
                    "category": None,
                    "category_name": "",
                    "supplier": row["supplier"],
                    "value": row["leakage"],
                    "spend": row["spend"],
                    "details": {
                        "leaking_lines": row["leaking_lines"],
                        "lines": row["lines"],
                        "compliance_rate": row["compliance_rate"],
                    },
                    "evidence": ev_rows,
                }
            )
    opps.sort(key=lambda o: -o["value"])
    for i, o in enumerate(opps):
        o["id"] = f"O{i:04d}"
    timings["analytics_s"] = time.perf_counter() - t

    by_lever = {lv: round(sum(o["value"] for o in opps if o["lever"] == lv), 2) for lv in LEVERS}
    # Price variance, maverick, and leakage can all claim the same line. Net them by taking the
    # largest of the three per line, so the headline total counts each dollar once.
    line_ids = set().union(*[set(v) for v in sinks.values()])
    line_net = sum(max(sinks[k].get(i, 0.0) for k in sinks) for i in line_ids)
    reported_line_levers = sum(by_lever[k] for k in sinks)
    gross_line_levers = sum(sum(v.values()) for v in sinks.values())
    # Opportunities below the reporting floor are not in by_lever; scale the netting to what is reported.
    net_line = line_net * (reported_line_levers / gross_line_levers) if gross_line_levers else 0.0
    net_total = net_line + by_lever["consolidation"] + by_lever["payment_terms"]
    summ = cube.summary(lines)
    report = {
        "company": "Harrow Valley Manufacturing (synthetic)",
        "config": {"lines": cfg.lines, "seed": cfg.seed, "parents": cfg.parents, "labeled_share": cfg.labeled_share},
        "data_quality": quality,
        "summary": summ,
        "savings_by_lever": [
            {
                "lever": lv,
                "name": LEVER_NAMES[lv],
                "value": by_lever[lv],
                "count": sum(1 for o in opps if o["lever"] == lv),
            }
            for lv in LEVERS
        ],
        "savings_gross": round(sum(v for k, v in by_lever.items() if k != "duplicates"), 2),
        "savings_total": round(net_total, 2),
        "overlap_removed": round(sum(v for k, v in by_lever.items() if k != "duplicates") - net_total, 2),
        "recovery_total": by_lever["duplicates"],
        "suppliers": {"eval": sup_eval, "stats": clusters.stats},
        "classification": cls_eval,
        "compliance": {k: v for k, v in comp.items() if k not in ("evidence",)},
        "evaluation": _extra_eval(lines, truth, dups, data),
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }
    return Results(lines, clusters, ReviewBook(clusters), opps, comp, report, con)


def _extra_eval(lines: pd.DataFrame, truth: pd.DataFrame, dups: list[dict], data: Dataset) -> dict:
    """Scores for the steps that have ground truth: item grouping, contract matching, duplicates."""
    li = lines.assign(true_item=lines["line_id"].map(truth["item_key"]))
    g = li[li["item_group"].notna()]
    maj = g.groupby("item_group")["true_item"].agg(lambda s: s.value_counts().iloc[0])
    purity = float(maj.sum() / len(g)) if len(g) else 0.0

    con_items = (
        data.contracts.drop_duplicates("item_description").set_index("item_description")["item_key"]
        if not data.contracts.empty
        else pd.Series(dtype=str)
    )
    m = li[li["contract_item"].notna()]
    # contract_item ids map back to a description; compare the item behind it with the true item.
    # Build the K-id -> item_key map from the matching itself.
    k_to_item = {}
    if len(m):
        k_to_item = m.groupby("contract_item")["true_item"].agg(lambda s: s.mode().iloc[0]).to_dict()
    match_prec = float((m["contract_item"].map(k_to_item) == m["true_item"]).mean()) if len(m) else 0.0
    contracted_true = li["true_item"].isin(set(con_items.values))
    match_recall = float(li.loc[contracted_true, "contract_item"].notna().mean()) if contracted_true.any() else 0.0

    true_dup = set(truth.index[truth["duplicate_of"] >= 0])
    found = {i for d in dups for i in d["line_ids_duplicate"]}
    tp = len(found & true_dup)
    return {
        "item_group_purity": round(purity, 4),
        "item_groups": int(g["item_group"].nunique()),
        "lines_with_item_group": round(len(g) / len(li), 4),
        "contract_match_precision": round(match_prec, 4),
        "contract_match_recall": round(match_recall, 4),
        "duplicates": {
            "true_duplicate_lines": len(true_dup),
            "flagged_lines": len(found),
            "flagged_invoices": len(dups),
            "precision": round(tp / len(found), 4) if found else 1.0,
            "recall": round(tp / len(true_dup), 4) if true_dup else 1.0,
        },
        "duplicate_value_true": round(float(lines.loc[lines["line_id"].isin(true_dup), "amount"].sum()), 2),
    }


def gate(report: dict, min_supplier_precision: float, min_l1_accuracy: float) -> list[str]:
    """CI quality gate. Returns failures (empty list means pass)."""
    fails = []
    p = report["suppliers"]["eval"]["precision"]
    a = report["classification"]["cascade"]["l1"]
    if p < min_supplier_precision:
        fails.append(f"supplier clustering precision {p:.4f} < floor {min_supplier_precision:.4f}")
    if a < min_l1_accuracy:
        fails.append(f"level-1 classification accuracy {a:.4f} < floor {min_l1_accuracy:.4f}")
    return fails


def as_np(x):  # pragma: no cover - small helper for json dumps
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    return str(x)
