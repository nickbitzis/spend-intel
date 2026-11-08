"""Line classification cascade: keyword rules, then a TF-IDF linear model, then an LLM.

Order for one distinct (description, supplier) key:
  1. Rules. Accept if exactly one category matches with confidence >= t_rules.
  2. Model. Accept if the top class probability >= t_model.
  3. Haiku, in batches of 20. Accept if its confidence >= t_haiku.
  4. Opus for whatever Haiku was unsure about. Always accepted.

Lines are deduplicated first: the same description from the same supplier is classified once. On
this data that cuts the work by about a third, and in real AP data (recurring invoices) by more.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .llm import BATCH, UNKNOWN, LineClassifier, LineInput
from .suppliers import clean_name
from .taxonomy import LEAVES, l1_of, l2_of

TIERS = ("rules", "model", "haiku", "opus")


# ------------------------------------------------------------------ tier 1: rules

_RULES = [(lf.code, [re.compile(k, re.I) for k in lf.keywords]) for lf in LEAVES]


@dataclass
class RuleResult:
    category: str | None
    confidence: float
    matched: dict[str, int]


def rule_classify(text: str) -> RuleResult:
    """Count keyword hits per category. One category with 2+ hits: 0.95. One category with one hit:
    0.9. Several categories: the leader at a confidence that shrinks with the runner-up's hits."""
    hits = {}
    for code, pats in _RULES:
        n = sum(1 for p in pats if p.search(text))
        if n:
            hits[code] = n
    if not hits:
        return RuleResult(None, 0.0, {})
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])
    best, n1 = ranked[0]
    if len(ranked) == 1:
        return RuleResult(best, 0.95 if n1 >= 2 else 0.9, hits)
    n2 = ranked[1][1]
    if n1 == n2:
        return RuleResult(best, 0.4, hits)
    return RuleResult(best, round(0.5 + 0.3 * (n1 - n2) / n1, 3), hits)


# ------------------------------------------------------------------ tier 2: model


def model_text(description: str, supplier: str) -> tuple[str, str]:
    return description.lower(), " ".join(f"s_{t}" for t in clean_name(supplier).split())


class ModelTier:
    """Character n-grams of the description (typos and abbreviations still share most n-grams) plus
    the cleaned supplier name tokens (a supplier usually sells one or two kinds of thing), into a
    multinomial logistic regression."""

    def __init__(self, C: float = 8.0, seed: int = 7):
        self.desc_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        self.sup_vec = TfidfVectorizer(analyzer="word", token_pattern=r"s_\w+", min_df=1)
        self.clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        self.classes_: np.ndarray | None = None

    def _x(self, desc: list[str], sup: list[str], fit: bool = False):
        pairs = [model_text(d, s) for d, s in zip(desc, sup, strict=True)]
        d = [p[0] for p in pairs]
        s = [p[1] for p in pairs]
        if fit:
            return hstack([self.desc_vec.fit_transform(d), 0.6 * self.sup_vec.fit_transform(s)]).tocsr()
        return hstack([self.desc_vec.transform(d), 0.6 * self.sup_vec.transform(s)]).tocsr()

    def fit(self, desc: list[str], sup: list[str], labels: list[str]) -> ModelTier:
        x = self._x(desc, sup, fit=True)
        self.clf.fit(x, labels)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, desc: list[str], sup: list[str]) -> tuple[np.ndarray, np.ndarray]:
        proba = self.clf.predict_proba(self._x(desc, sup))
        idx = proba.argmax(axis=1)
        return self.classes_[idx], proba[np.arange(len(idx)), idx]


# ------------------------------------------------------------------ cascade


@dataclass(frozen=True)
class Policy:
    t_rules: float = 0.9
    t_model: float = 0.7
    t_haiku: float = 0.85
    use_llm: bool = True


@dataclass
class CascadeOutput:
    keys: pd.DataFrame  # one row per distinct key: category, confidence, tier
    llm_cost: float
    llm_calls: dict[str, int]
    simulated: bool


def run_cascade(
    keys: pd.DataFrame,
    model: ModelTier,
    haiku: LineClassifier | None,
    opus: LineClassifier | None,
    policy: Policy | None = None,
) -> CascadeOutput:
    """`keys` columns: key, description, supplier, gl_account, and optionally gold_l3, difficulty
    (used only by simulated LLM tiers)."""
    policy = policy or Policy()
    out = keys.copy()
    rr = [rule_classify(t) for t in out["description"]]
    out["rule_cat"] = [r.category for r in rr]
    out["rule_conf"] = [r.confidence for r in rr]
    m_cat, m_conf = model.predict(out["description"].tolist(), out["supplier"].tolist())
    out["model_cat"] = m_cat
    out["model_conf"] = m_conf
    out["category"] = None
    out["confidence"] = 0.0
    out["tier"] = None

    take = out["rule_cat"].notna() & (out["rule_conf"] >= policy.t_rules)
    out.loc[take, ["category", "confidence", "tier"]] = (
        out.loc[take, ["rule_cat", "rule_conf"]].assign(tier="rules").values
    )
    take2 = out["tier"].isna() & (out["model_conf"] >= policy.t_model)
    out.loc[take2, ["category", "confidence", "tier"]] = (
        out.loc[take2, ["model_cat", "model_conf"]].assign(tier="model").values
    )

    cost = 0.0
    calls = {"haiku": 0, "opus": 0}
    simulated = True
    rest = out.index[out["tier"].isna()]
    if not policy.use_llm or haiku is None or opus is None:
        out.loc[rest, ["category", "confidence", "tier"]] = (
            out.loc[rest, ["model_cat", "model_conf"]].assign(tier="model").values
        )
        return CascadeOutput(out, 0.0, calls, simulated)

    simulated = bool(getattr(haiku, "simulated", True) and getattr(opus, "simulated", True))

    def llm_pass(client: LineClassifier, idx, name: str) -> dict:
        res = {}
        nonlocal cost
        for s in range(0, len(idx), BATCH):
            chunk = idx[s : s + BATCH]
            inputs = [
                LineInput(
                    id=str(i),
                    description=out.at[i, "description"],
                    supplier=out.at[i, "supplier"],
                    gl_account=out.at[i, "gl_account"],
                    gold_l3=out.at[i, "gold_l3"] if "gold_l3" in out else None,
                    difficulty=float(out.at[i, "difficulty"]) if "difficulty" in out else 0.5,
                    hint=out.at[i, "model_cat"],
                )
                for i in chunk
            ]
            br = client.classify(inputs)
            cost += br.cost
            calls[name] += 1
            for lab in br.labels:
                res[int(lab.id)] = lab
        return res

    h = llm_pass(haiku, list(rest), "haiku")
    unsure = []
    for i, lab in h.items():
        if lab.category != UNKNOWN and lab.confidence >= policy.t_haiku:
            out.loc[i, ["category", "confidence", "tier"]] = [lab.category, lab.confidence, "haiku"]
        else:
            unsure.append(i)
    o = llm_pass(opus, unsure, "opus")
    for i, lab in o.items():
        cat = lab.category if lab.category != UNKNOWN else out.at[i, "model_cat"]
        out.loc[i, ["category", "confidence", "tier"]] = [cat, lab.confidence, "opus"]
    return CascadeOutput(out, cost, calls, simulated)


# ------------------------------------------------------------------ metrics


def level_accuracy(pred: pd.Series, gold: pd.Series, weights: pd.Series | None = None) -> dict:
    """Accuracy at L1, L2, and L3 for L3 predictions. With weights, spend-weighted accuracy."""
    p = pred.astype(str).to_numpy()
    g = gold.astype(str).to_numpy()
    w = np.ones(len(p)) if weights is None else weights.to_numpy(dtype=float)
    tot = w.sum() or 1.0
    out = {}
    for name, fn in (("l1", l1_of), ("l2", l2_of), ("l3", lambda x: x)):
        pv = np.array([fn(x) for x in p])
        gv = np.array([fn(x) for x in g])
        out[name] = round(float(((pv == gv) * w).sum() / tot), 4)
    return out


def tier_report(lines: pd.DataFrame, n_keys: int, cost: float) -> dict:
    """Share of lines per tier, accuracy per tier, and LLM cost per 10,000 lines."""
    n = len(lines)
    share = Counter(lines["tier"])
    by_tier = {}
    for t in TIERS:
        sub = lines[lines["tier"] == t]
        by_tier[t] = {
            "lines": int(len(sub)),
            "share": round(len(sub) / n, 4) if n else 0.0,
            "accuracy": level_accuracy(sub["category"], sub["true_l3"]) if len(sub) else None,
        }
    return {
        "lines": n,
        "distinct_keys": n_keys,
        "by_tier": by_tier,
        "share": {t: round(share.get(t, 0) / n, 4) for t in TIERS} if n else {},
        "llm_cost": round(cost, 4),
        "cost_per_10k_lines": round(cost / n * 10_000, 4) if n else 0.0,
    }
