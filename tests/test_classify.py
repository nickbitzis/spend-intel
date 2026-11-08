import pandas as pd
import pytest

from spendintel.classify import ModelTier, Policy, level_accuracy, rule_classify, run_cascade
from spendintel.llm import HAIKU_SIM, OPUS_SIM, BatchResult, LineLabel, SimulatedClassifier


@pytest.mark.parametrize(
    "text,code",
    [
        ("SS bolt M8x30 qty 500", "MR-FAS-BLT"),
        ("NITRILE GLV LG 100CT", "MR-PPE-GLV"),
        ("LTL FRT PRO# 88812", "LG-FRT-LTL"),
        ("HVAC PM QTRLY RTU", "FC-MNT-HVC"),
        ("6061 T6 aluminum round bar", "RM-MET-ALU"),
    ],
)
def test_rules_hit(text, code):
    r = rule_classify(text)
    assert r.category == code
    assert r.confidence >= 0.9


def test_rules_ambiguous_is_low_confidence():
    r = rule_classify("box of drill bits and tape")
    assert r.confidence < 0.9
    r = rule_classify("nitrile gloves L box/100")  # gloves and boxes both match
    assert r.category == "MR-PPE-GLV" and r.confidence < 0.9


def test_rules_no_match():
    r = rule_classify("misc supplies per quote 4471")
    assert r.category is None and r.confidence == 0.0


TRAIN = [
    ("hex bolt m10", "Apex Fastener", "MR-FAS-BLT"),
    ("bolt m8 stainless", "Apex Fastener", "MR-FAS-BLT"),
    ("cap screw m6", "Apex Fastener", "MR-FAS-BLT"),
    ("nitrile glove large", "Summit Safety", "MR-PPE-GLV"),
    ("cut glove a4", "Summit Safety", "MR-PPE-GLV"),
    ("glove nitrile box", "Summit Safety", "MR-PPE-GLV"),
    ("ltl freight", "Delta Freight", "LG-FRT-LTL"),
    ("ltl shipment class 70", "Delta Freight", "LG-FRT-LTL"),
    ("freight outbound ltl", "Delta Freight", "LG-FRT-LTL"),
]


@pytest.fixture(scope="module")
def tiny_model():
    d, s, y = zip(*TRAIN, strict=True)
    return ModelTier(C=10).fit(list(d), list(s), list(y))


def test_model_tier_learns_tiny_set(tiny_model):
    cats, conf = tiny_model.predict(
        ["hex bolt m12", "nitrile glve lg", "ltl frt"], ["Apex Fastener", "Summit Safety", "Delta Freight"]
    )
    assert list(cats) == ["MR-FAS-BLT", "MR-PPE-GLV", "LG-FRT-LTL"]
    assert all(0 < c <= 1 for c in conf)


def test_model_uses_supplier_for_terse_text(tiny_model):
    cats, _ = tiny_model.predict(["item 4471"], ["Summit Safety"])
    assert cats[0] == "MR-PPE-GLV"


class Recorder:
    """A fake LLM tier that records what it was sent and answers with a fixed label."""

    simulated = True

    def __init__(self, model, category, confidence):
        self.model, self.category, self.confidence, self.seen = model, category, confidence, []

    def classify(self, lines):
        self.seen += [x.id for x in lines]
        return BatchResult(
            [LineLabel(x.id, self.category, self.confidence) for x in lines], self.model, 1000, 100, True
        )


def _keys(rows):
    return pd.DataFrame(rows, columns=["key", "description", "supplier", "gl_account"])


def test_cascade_routing(tiny_model):
    keys = _keys(
        [
            ("k1", "SS HEX HD BOLT M8 X 30MM", "Apex Fastener", ""),  # rules
            ("k2", "nitril glve larg", "Summit Safety", ""),  # typos defeat the rules; the model knows
            ("k3", "misc supplies per quote 4471", "Unknown Vendor", ""),  # LLM
        ]
    )
    haiku = Recorder("claude-haiku-5-5", "MR-PPE-EYE", 0.5)  # unsure, so Opus sees it too
    opus = Recorder("claude-opus-5-5", "FC-JAN-SUP", 0.9)
    out = run_cascade(keys, tiny_model, haiku, opus, Policy(t_rules=0.9, t_model=0.75, t_haiku=0.85))
    tiers = dict(zip(out.keys["key"], out.keys["tier"], strict=True))
    assert tiers["k1"] == "rules"
    assert tiers["k2"] == "model"
    assert tiers["k3"] == "opus"
    assert haiku.seen == ["2"] and opus.seen == ["2"]
    assert out.keys.loc[2, "category"] == "FC-JAN-SUP"
    assert out.llm_cost > 0 and out.llm_calls == {"haiku": 1, "opus": 1}


def test_cascade_confident_haiku_skips_opus(tiny_model):
    keys = _keys([("k3", "misc supplies per quote 4471", "Unknown Vendor", "")])
    haiku = Recorder("claude-haiku-5-5", "FC-JAN-SUP", 0.95)
    opus = Recorder("claude-opus-5-5", "FC-JAN-SUP", 0.9)
    out = run_cascade(keys, tiny_model, haiku, opus, Policy(t_model=0.99))
    assert out.keys.loc[0, "tier"] == "haiku"
    assert opus.seen == []


def test_cascade_without_llm_falls_back_to_model(tiny_model):
    keys = _keys([("k3", "misc supplies per quote 4471", "Unknown Vendor", "")])
    out = run_cascade(keys, tiny_model, None, None, Policy(t_model=0.99, use_llm=False))
    assert out.keys.loc[0, "tier"] == "model" and out.llm_cost == 0


def test_level_accuracy_per_level():
    pred = pd.Series(["MR-FAS-BLT", "MR-FAS-NUT", "MR-PPE-GLV", "LG-FRT-LTL"])
    gold = pd.Series(["MR-FAS-BLT", "MR-FAS-BLT", "MR-FAS-BLT", "MR-FAS-BLT"])
    acc = level_accuracy(pred, gold)
    assert acc == {"l1": 0.75, "l2": 0.5, "l3": 0.25}


def test_level_accuracy_spend_weighted():
    pred = pd.Series(["MR-FAS-BLT", "LG-FRT-LTL"])
    gold = pd.Series(["MR-FAS-BLT", "MR-FAS-BLT"])
    assert level_accuracy(pred, gold, pd.Series([900.0, 100.0]))["l1"] == 0.9


def test_simulated_tier_is_deterministic_and_labeled():
    from spendintel.llm import LineInput

    sim = SimulatedClassifier(HAIKU_SIM)
    xs = [LineInput(str(i), f"bolt {i}", "Apex", gold_l3="MR-FAS-BLT", difficulty=0.2) for i in range(40)]
    a, b = sim.classify(xs), sim.classify(xs)
    assert [x.category for x in a.labels] == [x.category for x in b.labels]
    assert a.simulated and a.cost > 0
    acc = sum(x.category == "MR-FAS-BLT" for x in a.labels) / 40
    assert acc > 0.75
    assert SimulatedClassifier(OPUS_SIM).classify(xs).cost > a.cost
