import pytest
from fastapi.testclient import TestClient

from spendintel.app import create_app
from spendintel.pipeline import gate


@pytest.fixture(scope="module")
def client(results, tmp_path_factory):
    return TestClient(create_app(results=results, data_dir=str(tmp_path_factory.mktemp("data"))))


def test_healthz_and_metrics(client):
    assert client.get("/healthz").json()["ready"] is True
    client.get("/api/summary")
    m = client.get("/metrics").text
    assert "spendintel_savings_identified_usd" in m
    assert 'spendintel_http_requests_total{route="/api/summary",status="200"}' in m


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Spend Intel" in r.text


def test_summary_numbers_are_consistent(client):
    s = client.get("/api/summary").json()
    total = s["summary"]["total_spend"]
    assert 0 < s["summary"]["addressable_spend"] <= total
    assert sum(r["spend"] for r in s["summary"]["by_l1"]) == pytest.approx(total, rel=1e-6)
    assert 0 < s["savings_total"] <= s["savings_gross"]
    assert {lv["lever"] for lv in s["savings_by_lever"]} >= {"price_variance", "payment_terms", "duplicates"}


def test_cube_query(client):
    r = client.get("/api/cube", params={"group_by": "l2,bu", "l1": "MR"}).json()
    assert r["rows"] and all(row["l2"].startswith("MR") for row in r["rows"])
    assert client.get("/api/cube", params={"group_by": "color"}).status_code == 400


def test_category_drilldown_and_brief(client):
    l1 = client.get("/api/categories", params={"level": "l1"}).json()["rows"]
    top = l1[0]["code"]
    l2 = client.get("/api/categories", params={"level": "l2", "parent": top}).json()["rows"]
    assert all(r["code"].startswith(top) for r in l2)
    b = client.get(f"/api/categories/{l2[0]['code']}/brief").json()
    assert b["simulated"] and b["text"]
    assert client.get("/api/categories/ZZ-NOPE/brief").status_code == 404


def test_supplier_review_flow(client):
    q = client.get("/api/suppliers/review").json()
    assert q["open"] > 0
    item = q["items"][0]
    action = "merge" if item["kind"] == "possible_merge" else "confirm"
    r = client.post(f"/api/suppliers/review/{item['id']}", json={"action": action})
    assert r.status_code == 200 and r.json()["open"] == q["open"] - 1
    assert client.post(f"/api/suppliers/review/{item['id']}", json={"action": "confirm"}).status_code == 404
    c = client.get("/api/suppliers/clusters", params={"min_members": 2}).json()
    assert c["eval"]["precision"] > 0.9 and all(len(row["members"]) >= 2 for row in c["rows"])


def test_classification_results(client):
    r = client.get("/api/classification", params={"max_confidence": 0.9, "limit": 20}).json()
    assert all(row["confidence"] <= 0.9 for row in r["rows"])
    assert set(r["report"]["tiers"]["share"]) == {"rules", "model", "haiku", "opus"}
    assert client.get("/api/classification", params={"tier": "wizard"}).status_code == 400


def test_opportunities_and_evidence(client):
    r = client.get("/api/opportunities", params={"lever": "price_variance"}).json()
    assert r["count"] > 0 and all(o["lever"] == "price_variance" for o in r["rows"])
    o = client.get(f"/api/opportunities/{r['rows'][0]['id']}").json()
    assert o["evidence"] and o["value"] > 0
    assert client.get("/api/opportunities/O9999999").status_code == 404


def test_duplicates_and_compliance(client):
    d = client.get("/api/duplicates").json()
    assert d["count"] > 0 and d["eval"]["precision"] >= 0.9
    c = client.get("/api/compliance").json()
    assert 0 <= c["price_compliance_rate"] <= 1 and c["leakage"] >= 0


def test_quality_on_small_run(results):
    rep = results.report
    assert rep["suppliers"]["eval"]["precision"] >= 0.95
    assert rep["classification"]["cascade"]["l1"] >= 0.95
    assert rep["classification"]["llm_simulated"] is True
    assert gate(rep, 0.95, 0.95) == []
    assert gate(rep, 1.01, 0.95) and gate(rep, 0.95, 1.01)
