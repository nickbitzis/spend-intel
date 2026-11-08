"""FastAPI service: spend cube queries, supplier clusters and review, classification results,
opportunities, duplicates, compliance, category briefs, /healthz, /metrics, and the web UI."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import cube
from .classify import TIERS
from .llm import ClaudeBriefWriter, TemplateBriefWriter
from .logs import get_logger, log_event
from .metrics import Metrics
from .pipeline import Config, Results, run
from .taxonomy import name_of, tree

STATIC = Path(__file__).parent / "static"


class ReviewAction(BaseModel):
    action: str  # confirm | reject | merge | split
    name: str | None = None


def brief_writer():
    if os.environ.get("SPENDINTEL_LLM", "offline") == "claude":
        return ClaudeBriefWriter()
    return TemplateBriefWriter()


def create_app(results: Results | None = None, cfg: Config | None = None, data_dir: str | None = None) -> FastAPI:
    log = get_logger()
    metrics = Metrics()
    state: dict = {"results": results, "ready": results is not None}

    @asynccontextmanager
    async def lifespan(_app):
        # Build the cube before the first request when running as a server.
        if os.environ.get("SPENDINTEL_WARM", "1") == "1":
            res()
        yield

    app = FastAPI(title="Spend Intel", version="1.0.0", lifespan=lifespan)
    data_dir = data_dir or os.environ.get("SPENDINTEL_DATA_DIR", "data")

    def res() -> Results:
        if state["results"] is None:
            t = time.perf_counter()
            lines = int(os.environ.get("SPENDINTEL_LINES", "120000"))
            state["results"] = run(cfg or Config(lines=lines))
            state["ready"] = True
            log_event(log, "pipeline_done", seconds=round(time.perf_counter() - t, 1), lines=lines)
            _publish_gauges()
            _save_report()
        return state["results"]

    def _publish_gauges() -> None:
        r = state["results"]
        rep = r.report
        metrics.set_gauge("spendintel_total_spend_usd", "Total spend in the cube.", rep["summary"]["total_spend"])
        metrics.set_gauge("spendintel_addressable_spend_usd", "Addressable spend.", rep["summary"]["addressable_spend"])
        for lv in rep["savings_by_lever"]:
            metrics.set_gauge(
                "spendintel_savings_identified_usd", "Identified savings by lever.", lv["value"], lever=lv["lever"]
            )
        metrics.set_gauge(
            "spendintel_supplier_precision",
            "Supplier clustering pairwise precision.",
            rep["suppliers"]["eval"]["precision"],
        )
        metrics.set_gauge(
            "spendintel_classification_accuracy",
            "Cascade accuracy by level.",
            rep["classification"]["cascade"]["l1"],
            level="l1",
        )
        for t, v in rep["classification"]["tiers"]["share"].items():
            metrics.set_gauge("spendintel_lines_by_tier_share", "Share of lines classified by each tier.", v, tier=t)
        metrics.set_gauge("spendintel_review_queue_open", "Open supplier review items.", len(r.review.open))

    def _save_report() -> None:
        try:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            (Path(data_dir) / "report.json").write_text(json.dumps(state["results"].report, indent=2, default=str))
        except OSError as exc:  # a read-only disk should not stop the API
            log_event(log, "report_write_failed", error=str(exc))

    if results is not None:
        _publish_gauges()

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        dur = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        metrics.observe_request(path, response.status_code, dur)
        response.headers["x-request-id"] = rid
        if path not in ("/healthz", "/metrics"):
            log_event(
                log,
                "request",
                request_id=rid,
                method=request.method,
                path=path,
                status=response.status_code,
                ms=round(dur * 1000, 1),
            )
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok" if state["ready"] else "starting", "ready": state["ready"]}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prom():
        return metrics.render()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/summary")
    def summary():
        r = res()
        rep = r.report
        return {
            "company": rep["company"],
            "summary": rep["summary"],
            "savings_by_lever": rep["savings_by_lever"],
            "savings_total": rep["savings_total"],
            "savings_gross": rep["savings_gross"],
            "overlap_removed": rep["overlap_removed"],
            "recovery_total": rep["recovery_total"],
            "classification": rep["classification"],
            "suppliers": rep["suppliers"],
            "compliance": {k: v for k, v in rep["compliance"].items() if k not in ("by_supplier", "by_category")},
            "evaluation": rep["evaluation"],
            "data_quality": rep["data_quality"],
        }

    @app.get("/api/report")
    def report():
        return res().report

    @app.get("/api/taxonomy")
    def taxonomy():
        return tree()

    @app.get("/api/cube")
    def cube_query(
        group_by: str = Query("l1", description="comma-separated: l1,l2,l3,supplier,bu,month,erp"),
        l1: str | None = None,
        l2: str | None = None,
        l3: str | None = None,
        supplier: str | None = None,
        bu: str | None = None,
        month: str | None = None,
        limit: int = Query(200, ge=1, le=5000),
    ):
        dims = [d for d in group_by.split(",") if d]
        try:
            rows = cube.query(
                res().lines, dims, {"l1": l1, "l2": l2, "l3": l3, "supplier": supplier, "bu": bu, "month": month}, limit
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"group_by": dims, "rows": rows}

    @app.get("/api/categories")
    def categories(level: str = Query("l1", pattern="^l[123]$"), parent: str | None = None):
        lines = res().lines
        if parent:
            parent_level = {"l2": "l1", "l3": "l2"}.get(level)
            if parent_level is None:
                raise HTTPException(400, "parent only applies to l2 and l3")
            lines = lines[lines[parent_level] == parent]
        return {
            "level": level,
            "parent": parent,
            "parent_name": name_of(parent) if parent else None,
            "rows": cube.category_profile(lines, level),
        }

    @app.get("/api/suppliers/clusters")
    def clusters(limit: int = Query(100, ge=1, le=2000), min_members: int = 1):
        r = res()
        spend = r.lines.groupby("supplier_id")["amount"].sum()
        rows = []
        for cid, members in r.clusters.members.items():
            if len(members) < min_members:
                continue
            rows.append(
                {
                    "id": cid,
                    "name": r.clusters.display.get(cid, cid),
                    "members": sorted(members),
                    "confidence": r.clusters.confidence.get(cid, 1.0),
                    "spend": round(float(spend.get(cid, 0.0)), 2),
                }
            )
        rows.sort(key=lambda x: -x["spend"])
        return {"stats": r.clusters.stats, "eval": r.report["suppliers"]["eval"], "rows": rows[:limit]}

    @app.get("/api/suppliers/review")
    def review_queue(limit: int = Query(50, ge=1, le=1000)):
        r = res()
        items = []
        for q in list(r.review.open.values())[:limit]:
            item = dict(q)
            item["a_members"] = sorted(r.clusters.members.get(q["a"], []))
            item["b_members"] = sorted(r.clusters.members.get(q["b"], [])) if q["b"] else []
            items.append(item)
        return {"open": len(r.review.open), "decided": len(r.review.decisions), "items": items}

    @app.post("/api/suppliers/review/{item_id}")
    def review_action(item_id: str, body: ReviewAction):
        r = res()
        try:
            rec = r.review.apply(item_id, body.action, body.name)
        except KeyError as exc:
            raise HTTPException(404, f"no open review item {item_id}") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        # Re-point lines at their (possibly new) cluster so cube queries reflect the decision.
        r.lines["supplier_id"] = r.lines["supplier_raw"].map(r.clusters.cluster_of)
        r.lines["supplier"] = r.lines["supplier_id"].map(r.clusters.display)
        metrics.set_gauge("spendintel_review_queue_open", "Open supplier review items.", len(r.review.open))
        log_event(log, "review_decision", **rec)
        try:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(data_dir) / "review_decisions.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        return {"decision": rec, "open": len(r.review.open)}

    @app.get("/api/classification")
    def classification(
        tier: str | None = None,
        max_confidence: float = Query(1.0, ge=0, le=1),
        l1: str | None = None,
        limit: int = Query(100, ge=1, le=2000),
    ):
        r = res()
        df = r.lines
        if tier:
            if tier not in (*TIERS, "labeled"):
                raise HTTPException(400, f"tier must be one of {TIERS}")
            df = df[df["tier"] == tier]
        if l1:
            df = df[df["l1"] == l1]
        df = df[df["confidence"] <= max_confidence].sort_values("confidence").head(limit)
        cols = ["line_id", "description", "supplier", "gl_account", "l3", "confidence", "tier", "amount"]
        rows = df[cols].to_dict("records")
        for x in rows:
            x["l3_name"] = name_of(x["l3"])
        return {"report": r.report["classification"], "rows": rows}

    @app.get("/api/opportunities")
    def opportunities(lever: str | None = None, category: str | None = None, limit: int = Query(100, ge=1, le=2000)):
        ops = res().opportunities
        if lever:
            ops = [o for o in ops if o["lever"] == lever]
        if category:
            ops = [o for o in ops if (o.get("category") or "").startswith(category)]
        slim = [
            {k: v for k, v in o.items() if k not in ("evidence", "line_ids_original", "line_ids_duplicate")}
            for o in ops[:limit]
        ]
        return {"count": len(ops), "value": round(sum(o["value"] for o in ops), 2), "rows": slim}

    @app.get("/api/opportunities/{opp_id}")
    def opportunity(opp_id: str):
        for o in res().opportunities:
            if o["id"] == opp_id:
                return o
        raise HTTPException(404, f"no opportunity {opp_id}")

    @app.get("/api/duplicates")
    def dups():
        ops = [o for o in res().opportunities if o["lever"] == "duplicates"]
        return {
            "count": len(ops),
            "value": round(sum(o["value"] for o in ops), 2),
            "eval": res().report["evaluation"]["duplicates"],
            "rows": ops,
        }

    @app.get("/api/compliance")
    def compliance():
        return res().compliance

    @app.get("/api/categories/{code}/brief")
    def brief(code: str):
        r = res()
        level = {1: "l1", 2: "l2", 3: "l3"}[min(3, code.count("-") + 1)]
        sub = r.lines[r.lines[level] == code]
        if sub.empty:
            raise HTTPException(404, f"no spend in category {code}")
        prof = cube.category_profile(sub, level)[0]
        opps = [o for o in r.opportunities if (o.get("category") or "").startswith(code)]
        by_lever = {}
        for o in opps:
            by_lever[o["lever"]] = round(by_lever.get(o["lever"], 0) + o["value"], 2)
        contracted = sub[sub["contract_item"].notna()]
        facts = {
            "code": code,
            "name": prof["name"],
            "spend": prof["spend"],
            "suppliers": prof["suppliers"],
            "hhi": prof["hhi"],
            "top_suppliers": prof["top_suppliers"][:3],
            "tail_suppliers": prof["tail_suppliers"],
            "opportunities": by_lever,
            "contract_coverage": round(float(contracted["amount"].sum() / sub["amount"].sum()), 4),
            "leakage": round(
                float(sum(x["leakage"] for x in r.compliance["by_category"] if x["l3"].startswith(code))), 2
            ),
        }
        writer = brief_writer()
        try:
            out = writer.write(facts)
        except Exception as exc:  # any provider failure falls back to the template
            out = TemplateBriefWriter().write(facts)
            out["fallback_reason"] = f"{type(exc).__name__}"
        metrics.summary(out["writer"], "fallback_reason" in out)
        return {"facts": facts, **out}

    return app


def serve(host: str | None = None, port: int | None = None) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(
        create_app(),
        host=host or os.environ.get("HOST", "127.0.0.1"),
        port=int(port or os.environ.get("PORT", "8849")),
        log_level="warning",
    )
