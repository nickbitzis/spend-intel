"""LLM tier for line classification and the category brief writer, behind small interfaces.

`ClaudeClassifier` sends a batch of up to 20 lines per request and asks for a JSON object that
matches OUTPUT_SCHEMA (structured output via `output_config.format`). The `anthropic` package is
imported only when no client is injected, so the app runs without the extra installed.

`SimulatedClassifier` is the offline default. It calls no model. Its accuracy, confidence, and
token counts come from a config so the cascade economics can be studied for free. Every number it
produces is simulated and is labeled that way in reports and the UI.

Prices are USD per million tokens.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol

from .taxonomy import BY_CODE, L1_NAMES, L2_NAMES, L3_CODES, LEAVES, l2_of

PRICES = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}
HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
UNKNOWN = "UNKNOWN"
BATCH = 20


FALLBACK_BETA = "server-side-fallback-2026-07-01"


def create_message(client, **kwargs):
    """messages.create, with server-side refusal fallback turned on for Opus.

    Opus 5.5 runs safety classifiers that can decline a request. With `fallbacks="default"` the API
    retries a declined request on a model chosen by refusal category, so a false positive on an
    invoice line does not stall a batch. Haiku has no server-side fallback and uses the plain call."""
    if kwargs.get("model", "").startswith("claude-opus"):
        return client.beta.messages.create(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
    return client.messages.create(**kwargs)


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "category": {"type": "string", "enum": [*L3_CODES, UNKNOWN]},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "category", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def system_prompt() -> str:
    lines = [
        "You classify accounts payable invoice lines for a mid-size manufacturer into a fixed category tree.",
        "Each line has a free-text description (often abbreviated or misspelled), the supplier name, and the GL",
        "account it was booked to. GL accounts are unreliable; trust the description first, then the supplier.",
        "Return one result per input id. Use the most specific category code. If nothing fits, use UNKNOWN.",
        "Confidence is your probability that the category is right.",
        "",
        "Categories (code: L1 > L2 > L3):",
    ]
    for lf in LEAVES:
        lines.append(f"- {lf.code}: {L1_NAMES[lf.l1]} > {L2_NAMES[lf.l2]} > {lf.name}")
    return "\n".join(lines)


SYSTEM_PROMPT = system_prompt()
SYSTEM_TOKENS = len(SYSTEM_PROMPT) // 4 + 80  # schema and framing overhead


@dataclass
class LineInput:
    id: str
    description: str
    supplier: str
    gl_account: str = ""
    gold_l3: str | None = None  # only used by the simulator
    difficulty: float = 0.5  # only used by the simulator
    hint: str | None = None  # best local guess, used by the simulator when there is no gold label


@dataclass
class LineLabel:
    id: str
    category: str
    confidence: float


@dataclass
class BatchResult:
    labels: list[LineLabel]
    model: str
    input_tokens: int
    output_tokens: int
    simulated: bool

    @property
    def cost(self) -> float:
        return call_cost(self.model, self.input_tokens, self.output_tokens)


class StructuredOutputError(ValueError):
    """The reply did not match the schema."""


class LineClassifier(Protocol):
    model: str
    simulated: bool

    def classify(self, lines: list[LineInput]) -> BatchResult: ...


def validate_output(obj: object, ids: list[str]) -> list[LineLabel]:
    """Check a parsed reply against OUTPUT_SCHEMA. Missing ids come back as UNKNOWN at confidence 0
    so the cascade escalates them instead of crashing."""
    if not isinstance(obj, dict) or not isinstance(obj.get("results"), list):
        raise StructuredOutputError("reply has no results list")
    got: dict[str, LineLabel] = {}
    for r in obj["results"]:
        if not isinstance(r, dict) or set(r) != {"id", "category", "confidence"}:
            raise StructuredOutputError(f"bad result object: {r!r}")
        if r["category"] not in BY_CODE and r["category"] != UNKNOWN:
            raise StructuredOutputError(f"category {r['category']!r} is not in the taxonomy")
        conf = r["confidence"]
        if isinstance(conf, bool) or not isinstance(conf, int | float) or not 0 <= conf <= 1:
            raise StructuredOutputError(f"confidence must be 0 to 1, got {conf!r}")
        got[str(r["id"])] = LineLabel(str(r["id"]), r["category"], float(conf))
    return [got.get(i, LineLabel(i, UNKNOWN, 0.0)) for i in ids]


class ClaudeClassifier:
    """Real Claude calls. Inject `client` in tests; otherwise the SDK is imported here."""

    simulated = False

    def __init__(self, model: str, client=None, effort: str = "low"):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on install extras
                raise RuntimeError("Install the Claude extra first: uv sync --extra claude") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def request(self, lines: list[LineInput]) -> dict:
        payload = [
            {"id": x.id, "description": x.description, "supplier": x.supplier, "gl_account": x.gl_account}
            for x in lines
        ]
        return {
            "model": self.model,
            "max_tokens": 200 + 60 * len(lines),
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"<lines>\n{json.dumps(payload)}\n</lines>"}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        }

    def classify(self, lines: list[LineInput]) -> BatchResult:
        resp = create_message(self.client, **self.request(lines))
        ids = [x.id for x in lines]
        if getattr(resp, "stop_reason", None) == "refusal":
            labels = [LineLabel(i, UNKNOWN, 0.0) for i in ids]
        else:
            raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            try:
                labels = validate_output(json.loads(raw), ids)
            except json.JSONDecodeError as exc:
                raise StructuredOutputError(f"reply is not JSON: {raw[:80]!r}") from exc
        return BatchResult(labels, self.model, resp.usage.input_tokens, resp.usage.output_tokens, simulated=False)


@dataclass(frozen=True)
class SimConfig:
    """Assumptions for a simulated tier, not measurements. Accuracy falls linearly from acc_easy to
    acc_hard as line difficulty goes from 0 to 1. Wrong answers usually land in a sibling category."""

    model: str
    acc_easy: float
    acc_hard: float
    conf_right: tuple[float, float]
    conf_wrong: tuple[float, float]
    tokens_per_line_in: int = 32
    tokens_per_line_out: int = 22


HAIKU_SIM = SimConfig(HAIKU, acc_easy=0.96, acc_hard=0.72, conf_right=(0.7, 0.99), conf_wrong=(0.4, 0.88))
OPUS_SIM = SimConfig(OPUS, acc_easy=0.99, acc_hard=0.86, conf_right=(0.8, 0.99), conf_wrong=(0.45, 0.85))


def _u(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


class SimulatedClassifier:
    simulated = True

    def __init__(self, config: SimConfig):
        self.config = config
        self.model = config.model

    def classify(self, lines: list[LineInput]) -> BatchResult:
        c = self.config
        labels = []
        for x in lines:
            key = f"{c.model}|{x.id}|{x.description}"
            if x.gold_l3 is None:
                cat = x.hint if x.hint in BY_CODE else UNKNOWN
                right = True
            else:
                p = c.acc_easy - (c.acc_easy - c.acc_hard) * x.difficulty
                right = _u(key, "right") < p
                if right:
                    cat = x.gold_l3
                else:
                    sibs = [code for code in L3_CODES if l2_of(code) == l2_of(x.gold_l3) and code != x.gold_l3]
                    pool = sibs if sibs and _u(key, "sib") < 0.7 else [k for k in L3_CODES if k != x.gold_l3]
                    cat = pool[int(_u(key, "pick") * len(pool))]
            lo, hi = c.conf_right if right else c.conf_wrong
            labels.append(LineLabel(x.id, cat, round(lo + (hi - lo) * _u(key, "conf"), 4)))
        n = len(lines)
        return BatchResult(
            labels, c.model, SYSTEM_TOKENS + c.tokens_per_line_in * n, 12 + c.tokens_per_line_out * n, simulated=True
        )


# ---------------------------------------------------------------- category strategy brief

BRIEF_SYSTEM = (
    "You write a one-page category strategy brief for a procurement leader. Use only the facts in the JSON "
    "you are given. Do not introduce any number that is not in the facts. Plain sentences, no marketing words. "
    "Sections: Situation, Opportunities, Recommended next steps, Risks."
)


class BriefWriter(Protocol):
    name: str

    def write(self, facts: dict) -> dict: ...


def _fmt_money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


class TemplateBriefWriter:
    """Offline default: fills a fixed template from the computed facts. No model call."""

    name = "offline template"

    def write(self, facts: dict) -> dict:
        f = facts
        top = ", ".join(f"{s['supplier']} ({s['share']:.0%})" for s in f["top_suppliers"][:3])
        levers = sorted(f["opportunities"].items(), key=lambda kv: -kv[1])
        lever_txt = "; ".join(f"{k.replace('_', ' ')} {_fmt_money(v)}" for k, v in levers if v > 0) or "none found"
        steps = []
        if f["opportunities"].get("price_variance", 0) > 0:
            steps.append("Bring the highest-paying business units to the median price with the current suppliers.")
        if f["opportunities"].get("consolidation", 0) > 0:
            steps.append(
                f"Run a sourcing event to move tail suppliers ({f['tail_suppliers']} of them) to the top three."
            )
        if f["opportunities"].get("maverick", 0) > 0 or f["leakage"] > 0:
            steps.append("Route buys of contracted items to the contracted supplier and audit invoice prices.")
        if f["opportunities"].get("payment_terms", 0) > 0:
            steps.append("Align payment terms to policy and take early payment discounts where offered.")
        text = "\n".join(
            [
                f"Situation. {f['name']} spend was {_fmt_money(f['spend'])} across {f['suppliers']} suppliers. "
                f"Top suppliers: {top}. HHI is {f['hhi']:,.0f}. Contract coverage is {f['contract_coverage']:.0%}.",
                f"Opportunities. {lever_txt}. Contract price leakage: {_fmt_money(f['leakage'])}.",
                "Recommended next steps. " + " ".join(steps or ["Hold. The category is concentrated and on contract."]),
                "Risks. Savings estimates assume volumes stay flat and quality is equivalent across suppliers.",
            ]
        )
        return {"text": text, "writer": self.name, "simulated": True, "grounded": True, "cost": 0.0}


NUM = re.compile(r"\d[\d,]*\.?\d*")


def grounded(text: str, facts: dict) -> tuple[bool, list[str]]:
    """Every number in the brief must come from the facts (allowing $M/$K and percent formatting).
    Returns (ok, unknown numbers)."""
    allowed: set[str] = set()

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, bool):
            return
        elif isinstance(v, int | float):
            for s in (
                str(v),
                str(round(v, 1)),
                f"{v:,.0f}",
                f"{v:.0f}",
                f"{v:,.2f}",
                f"{v / 1e6:,.1f}",
                f"{v / 1e6:,.2f}",
                f"{v / 1e3:,.0f}",
                f"{v * 100:.0f}",
                f"{v * 100:.1f}",
            ):
                allowed.add(s.replace(",", ""))
        elif isinstance(v, str):
            for m in NUM.findall(v):
                allowed.add(m.replace(",", "").rstrip("."))

    walk(facts)
    bad = []
    for m in NUM.findall(text):
        n = m.replace(",", "").rstrip(".")
        if n in allowed or n in {"1", "2", "3", "4", "5"}:  # small counts like "three steps" or list numbers
            continue
        bad.append(m)
    return (not bad, bad)


class ClaudeBriefWriter:
    name = "claude"

    def __init__(self, client=None, model: str = OPUS):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install the Claude extra first: uv sync --extra claude") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def write(self, facts: dict) -> dict:
        resp = create_message(
            self.client,
            model=self.model,
            max_tokens=2000,
            system=BRIEF_SYSTEM,
            messages=[{"role": "user", "content": f"<facts>\n{json.dumps(facts, default=str)}\n</facts>"}],
            output_config={"effort": "medium"},
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        ok, bad = grounded(text, facts)
        cost = call_cost(self.model, resp.usage.input_tokens, resp.usage.output_tokens)
        if getattr(resp, "stop_reason", None) == "refusal" or not text:
            out = TemplateBriefWriter().write(facts)
            out.update({"fallback_reason": "model declined or returned no text", "cost": cost})
            return out
        if not ok:
            # A number we cannot trace back to the facts is a hallucination risk. Fall back.
            out = TemplateBriefWriter().write(facts)
            out.update({"fallback_reason": f"ungrounded numbers: {bad[:5]}", "cost": cost})
            return out
        return {"text": text, "writer": f"claude ({self.model})", "simulated": False, "grounded": True, "cost": cost}
