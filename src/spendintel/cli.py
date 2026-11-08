"""Command line: run the analysis, gate it in CI, or serve the API and UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import LEVER_NAMES, Config, gate, run


def _money(x: float) -> str:
    return f"${x:,.0f}"


def markdown(rep: dict) -> str:
    s = rep["summary"]
    c = rep["classification"]
    out = [
        "## Spend Intel results",
        "",
        f"Total spend {_money(s['total_spend'])}, addressable {_money(s['addressable_spend'])}, "
        f"{s['suppliers']} supplier clusters, {s['lines']:,} lines.",
        "",
        "| Lever | Opportunities | Estimated value |",
        "|---|---|---|",
    ]
    for lv in rep["savings_by_lever"]:
        out.append(f"| {LEVER_NAMES[lv['lever']]} | {lv['count']} | {_money(lv['value'])} |")
    out += [
        f"| **Savings, net of overlap** | | **{_money(rep['savings_total'])}** |",
        "",
        f"Supplier clustering: precision {rep['suppliers']['eval']['precision']:.3f}, "
        f"recall {rep['suppliers']['eval']['recall']:.3f}.",
        f"Classification accuracy L1 {c['cascade']['l1']:.3f}, L2 {c['cascade']['l2']:.3f}, "
        f"L3 {c['cascade']['l3']:.3f}; "
        f"LLM cost per 10,000 lines ${c['tiers']['cost_per_10k_lines']:.2f}"
        + (" (simulated LLM tiers)." if c["llm_simulated"] else "."),
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spendintel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="run the pipeline and write report.json")
    a.add_argument("--lines", type=int, default=120_000)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--out", type=Path, default=Path("data/report.json"))
    a.add_argument("--summary", type=Path, help="append a markdown summary here (CI step summary)")
    a.add_argument("--gate", action="store_true", help="exit 1 if quality drops below the floors")
    a.add_argument("--min-supplier-precision", type=float, default=0.97)
    a.add_argument("--min-l1-accuracy", type=float, default=0.97)
    s = sub.add_parser("serve", help="run the API and web UI")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "serve":  # pragma: no cover
        from .app import serve

        serve(args.host, args.port)
        return

    r = run(Config(lines=args.lines, seed=args.seed))
    rep = r.report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=str))
    md = markdown(rep)
    print(md)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(md)
    if args.gate:
        fails = gate(rep, args.min_supplier_precision, args.min_l1_accuracy)
        for f in fails:
            print(f"GATE FAIL: {f}", file=sys.stderr)
        if fails:
            sys.exit(1)
        print("gate passed")
