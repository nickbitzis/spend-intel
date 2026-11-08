import json
from types import SimpleNamespace

import pandas as pd
import pytest

from spendintel.generator import generate
from spendintel.ingest import parse_amount, parse_terms, unify
from spendintel.llm import (
    OUTPUT_SCHEMA,
    ClaudeBriefWriter,
    ClaudeClassifier,
    LineInput,
    StructuredOutputError,
    TemplateBriefWriter,
    grounded,
    validate_output,
)


@pytest.mark.parametrize(
    "text,parsed",
    [
        ("NT30", (30, 0.0, 0)),
        ("2/10N30", (30, 2.0, 10)),
        ("Net 45", (45, 0.0, 0)),
        ("2% 10 Net 30", (30, 2.0, 10)),
        ("1%15/45", (45, 1.0, 15)),
        ("60 days", (60, 0.0, 0)),
        ("Due on receipt", (0, 0.0, 0)),
        ("immediate", (0, 0.0, 0)),
    ],
)
def test_parse_terms(text, parsed):
    assert parse_terms(text) == parsed


def test_parse_terms_unknown_raises():
    with pytest.raises(ValueError):
        parse_terms("when we feel like it")


def test_parse_amount():
    assert parse_amount("1,234.50") == 1234.5
    assert parse_amount(12) == 12.0


def test_generator_is_seeded_and_three_formats_unify():
    a = generate(1500, seed=3, n_parents=80)
    b = generate(1500, seed=3, n_parents=80)
    assert a.raw["A"].equals(b.raw["A"])
    assert set(a.raw) == {"A", "B", "C"}
    assert "UNIT_PRICE" in a.raw["A"] and "extended_amount_cents" in a.raw["B"] and "Net Value" in a.raw["C"]
    lines, q = unify(a.raw)
    assert q["unparsed_terms"] == 0 and q["missing_unit_price"] == 0
    assert len(lines) == len(a.truth)
    assert lines["line_id"].is_unique
    truth = a.truth.set_index("line_id")
    # Amounts survive the round trip through three formats to the cent.
    assert (lines.set_index("line_id")["amount"] - truth.index.to_series().map(lambda _: 0)).notna().all()
    assert lines["invoice_date"].dt.year.eq(2025).all()
    assert (truth["duplicate_of"] >= 0).sum() > 0


def test_validate_output_rejects_bad_category():
    with pytest.raises(StructuredOutputError):
        validate_output({"results": [{"id": "1", "category": "XX-YY-ZZ", "confidence": 0.9}]}, ["1"])
    with pytest.raises(StructuredOutputError):
        validate_output({"results": [{"id": "1", "category": "MR-FAS-BLT", "confidence": 1.4}]}, ["1"])


def test_validate_output_fills_missing_ids_as_unknown():
    out = validate_output({"results": [{"id": "1", "category": "MR-FAS-BLT", "confidence": 0.9}]}, ["1", "2"])
    assert out[1].category == "UNKNOWN" and out[1].confidence == 0.0


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
        )


class FakeClient:
    def __init__(self, text, stop_reason="end_turn"):
        self.messages = FakeMessages(text, stop_reason)
        self.beta = SimpleNamespace(messages=self.messages)


def test_claude_classifier_request_and_parse():
    reply = json.dumps({"results": [{"id": "a", "category": "MR-PPE-GLV", "confidence": 0.93}]})
    client = FakeClient(reply)
    clf = ClaudeClassifier("claude-haiku-5-5", client=client)
    br = clf.classify([LineInput("a", "nitrile glv lg", "Summit Safety", "610200")])
    assert br.labels[0].category == "MR-PPE-GLV" and not br.simulated
    req = client.messages.calls[0]
    assert req["model"] == "claude-haiku-5-5"
    assert req["output_config"]["format"]["schema"] == OUTPUT_SCHEMA
    assert "fallbacks" not in req
    assert br.cost == pytest.approx((1200 * 0.10 + 300 * 0.50) / 1e6)


def test_claude_opus_uses_server_side_fallback():
    reply = json.dumps({"results": [{"id": "a", "category": "MR-PPE-GLV", "confidence": 0.93}]})
    client = FakeClient(reply)
    ClaudeClassifier("claude-opus-5-5", client=client).classify([LineInput("a", "x", "y")])
    req = client.messages.calls[0]
    assert req["fallbacks"] == "default" and req["betas"]


def test_claude_refusal_becomes_unknown():
    client = FakeClient("", stop_reason="refusal")
    br = ClaudeClassifier("claude-haiku-5-5", client=client).classify([LineInput("a", "x", "y")])
    assert br.labels[0].category == "UNKNOWN"


FACTS = {
    "code": "MR-FAS",
    "name": "Fasteners",
    "spend": 1250000.0,
    "suppliers": 14,
    "hhi": 1320.5,
    "top_suppliers": [{"supplier": "Apex Fastener", "spend": 500000.0, "share": 0.4}],
    "tail_suppliers": 9,
    "opportunities": {"price_variance": 41000.0, "consolidation": 30000.0},
    "contract_coverage": 0.35,
    "leakage": 1200.0,
}


def test_template_brief_uses_facts_only():
    out = TemplateBriefWriter().write(FACTS)
    assert out["simulated"] and "Fasteners" in out["text"]
    ok, bad = grounded(out["text"], FACTS)
    assert ok, bad


def test_grounding_catches_invented_numbers():
    ok, bad = grounded("Spend was $1.25M and we can save 37% next year.", FACTS)
    assert not ok and "37" in bad


def test_claude_brief_falls_back_when_ungrounded():
    client = FakeClient("Situation. Fasteners spend is $9.99M with 77 suppliers.")
    out = ClaudeBriefWriter(client=client).write(FACTS)
    assert out["writer"] == "offline template" and "ungrounded" in out["fallback_reason"]


def test_claude_brief_accepts_grounded_text():
    client = FakeClient("Situation. Fasteners spend was $1.25M across 14 suppliers. HHI is 1,320.5.")
    out = ClaudeBriefWriter(client=client).write(FACTS)
    assert out["writer"].startswith("claude") and not out["simulated"]
    assert pd.notna(out["cost"])
