"""Supplier normalization: clean, block, score, cluster, and queue the unsure cases for review.

1. Clean: case, punctuation, '&', legal suffixes (Inc, LLC, Co...), division tails ("- Ohio"),
   a leading "The", and abbreviations mapped to one canonical token (Ind. and Industrial both
   become "industrial").
2. Block: two names are compared only if they share a 3-character prefix or suffix of a token that
   is not too common. That cuts the pair count by about 50x while keeping typo variants together.
3. Score: half IDF-weighted token Jaccard (rare words like "Halston" count more than "supply"),
   half cosine similarity of character trigrams (catches typos and spacing).
4. Cluster: union-find over pairs above the auto-merge threshold. Pairs between the review and
   auto thresholds go to a review queue instead of being merged.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

LEGAL = {"inc", "incorporated", "llc", "co", "company", "corp", "corporation", "ltd", "limited", "lp", "plc", "llp"}
CANON = {
    "ind": "industrial",
    "indus": "industrial",
    "sply": "supply",
    "supplies": "supply",
    "mfg": "manufacturing",
    "pkg": "packaging",
    "pkging": "packaging",
    "svcs": "services",
    "svc": "services",
    "service": "services",
    "sys": "systems",
    "elec": "electric",
    "electrical": "electric",
    "intl": "international",
    "assoc": "associates",
    "ctr": "center",
    "eng": "engineering",
    "tech": "technology",
    "technologies": "technology",
    "dist": "distribution",
    "prod": "products",
    "logistic": "logistics",
    "bros": "brothers",
    "natl": "national",
}
DROP_TAIL = re.compile(r"\s+(-|\u2013|/)\s+.*$|\s+dba\s+.*$|\s+c/o\s+.*$", re.I)


def clean_name(name: str) -> str:
    """'ACME IND. SUPPLY CO., INC. - Midwest Division' -> 'acme industrial supply'."""
    s = DROP_TAIL.sub("", name.strip())
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [CANON.get(t, t) for t in s.split()]
    if toks and toks[0] == "the":
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in LEGAL:
        toks = toks[:-1]
    toks = [t for t in toks if t not in LEGAL or len(toks) <= 1]
    return " ".join(toks)


def trigrams(s: str) -> Counter:
    p = f"  {s} "
    return Counter(p[i : i + 3] for i in range(len(p) - 2))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def within_one_edit(a: str, b: str) -> bool:
    """True if a and b differ by one insert, delete, substitute, or adjacent swap."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1 :]


def token_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) >= 4 and len(b) >= 4 and within_one_edit(a, b):
        return 0.9  # a typo
    return 0.0


def weighted_jaccard(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    """IDF-weighted Jaccard where a token pair one typo apart counts as 0.9 of a match."""
    if not a or not b:
        return 0.0
    matched = 0.0
    used = set()
    for t in sorted(a):
        best, best_u = 0.0, None
        for u in b:
            if u in used:
                continue
            sim = token_sim(t, u)
            if sim > best:
                best, best_u = sim, u
                if sim == 1.0:
                    break
        if best_u is not None:
            used.add(best_u)
            matched += best * (idf.get(t, 1.0) + idf.get(best_u, 1.0)) / 2
    total = sum(idf.get(t, 1.0) for t in a) + sum(idf.get(t, 1.0) for t in b)
    union = total - matched
    return matched / union if union > 0 else 0.0


def similarity(a: str, b: str, idf: dict[str, float] | None = None) -> float:
    """Score two cleaned names from 0 to 1."""
    if a == b:
        return 1.0
    idf = idf or {}
    tok = weighted_jaccard(set(a.split()), set(b.split()), idf)
    return 0.5 * tok + 0.5 * cosine(trigrams(a), trigrams(b))


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class Clusters:
    cluster_of: dict[str, str]  # raw name -> cluster id
    members: dict[str, list[str]]  # cluster id -> raw names
    display: dict[str, str]  # cluster id -> display name
    confidence: dict[str, float]  # cluster id -> min link score inside the cluster (1.0 for one name)
    review: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def block_keys(clean: str, common: set[str]) -> set[str]:
    keys = set()
    for t in clean.split():
        if len(t) < 3 or t in common:
            continue
        keys.add("p:" + t[:3])
        keys.add("s:" + t[-3:])
    if not keys:
        keys.add("w:" + clean)
    return keys


def resolve(
    names: list[str],
    weights: dict[str, float] | None = None,
    auto: float = 0.8,
    review: float = 0.62,
    common_share: float = 0.04,
) -> Clusters:
    """Cluster raw supplier names. `weights` (spend per name) picks display names and orders the
    review queue so the biggest dollars get looked at first."""
    weights = weights or {}
    names = sorted(set(names))
    clean = {n: clean_name(n) for n in names}
    uniq = sorted(set(clean.values()))
    df = Counter(t for c in uniq for t in set(c.split()))
    n_docs = max(1, len(uniq))
    idf = {t: math.log(1 + n_docs / c) for t, c in df.items()}
    common = {t for t, c in df.items() if c >= 3 and c / n_docs > common_share}

    blocks: dict[str, list[str]] = defaultdict(list)
    for c in uniq:
        for k in block_keys(c, common):
            blocks[k].append(c)
    pairs: set[tuple[str, str]] = set()
    for members in blocks.values():
        if len(members) > 400:  # a runaway block means the key is too common; skip it
            continue
        for a, b in combinations(members, 2):
            pairs.add((a, b) if a < b else (b, a))

    tri = {c: trigrams(c) for c in uniq}
    toks = {c: set(c.split()) for c in uniq}
    uf = UnionFind(uniq)
    link: dict[str, float] = {}
    gray: list[tuple[str, str, float]] = []
    for a, b in pairs:
        s = 0.5 * weighted_jaccard(toks[a], toks[b], idf) + 0.5 * cosine(tri[a], tri[b])
        if s >= auto:
            uf.union(a, b)
            link[a] = max(link.get(a, 0.0), s)  # each name's best link into its cluster
            link[b] = max(link.get(b, 0.0), s)
        elif s >= review:
            gray.append((a, b, s))

    by_root: dict[str, list[str]] = defaultdict(list)
    for c in uniq:
        by_root[uf.find(c)].append(c)
    cluster_of_clean, members, display, confidence = {}, {}, {}, {}
    ordered = sorted(by_root.values(), key=lambda cs: -sum(weights.get(n, 0) for n in names if clean[n] in cs))
    for i, cs in enumerate(ordered):
        cid = f"P{i:04d}"
        for c in cs:
            cluster_of_clean[c] = cid
        confidence[cid] = round(min((link.get(c, 1.0) for c in cs), default=1.0), 4) if len(cs) > 1 else 1.0
    for n in names:
        cid = cluster_of_clean[clean[n]]
        members.setdefault(cid, []).append(n)
    for cid, ms in members.items():
        best = max(ms, key=lambda n: (weights.get(n, 0), -len(n)))
        display[cid] = clean_display(best)
    cluster_of = {n: cluster_of_clean[clean[n]] for n in names}

    queue = []
    seen = set()
    for a, b, s in sorted(gray, key=lambda x: -x[2]):
        ca, cb = cluster_of_clean[a], cluster_of_clean[b]
        if ca == cb or (ca, cb) in seen:
            continue
        seen.add((ca, cb))
        spend = sum(weights.get(n, 0) for n in members[ca] + members[cb])
        queue.append(
            {
                "kind": "possible_merge",
                "a": ca,
                "b": cb,
                "score": round(s, 4),
                "spend": round(spend, 2),
                "a_name": display[ca],
                "b_name": display[cb],
            }
        )
    for cid, conf in confidence.items():
        if len(members[cid]) > 1 and conf < auto + 0.06:
            spend = sum(weights.get(n, 0) for n in members[cid])
            queue.append(
                {
                    "kind": "confirm_cluster",
                    "a": cid,
                    "b": None,
                    "score": conf,
                    "spend": round(spend, 2),
                    "a_name": display[cid],
                    "b_name": None,
                }
            )
    queue.sort(key=lambda q: -q["spend"])
    for i, q in enumerate(queue):
        q["id"] = f"R{i:04d}"
    total_pairs = len(uniq) * (len(uniq) - 1) // 2
    stats = {
        "raw_names": len(names),
        "clean_names": len(uniq),
        "clusters": len(members),
        "pairs_compared": len(pairs),
        "pairs_possible": total_pairs,
        "reduction_ratio": round(1 - len(pairs) / total_pairs, 4) if total_pairs else 0.0,
        "review_items": len(queue),
        "auto_threshold": auto,
        "review_threshold": review,
    }
    return Clusters(cluster_of, members, display, confidence, queue, stats)


def clean_display(raw: str) -> str:
    """A tidy display name from the most used raw spelling: drop tails, title case if shouted."""
    s = DROP_TAIL.sub("", raw.strip()).strip(" ,")
    if s.isupper() or s.islower():
        s = " ".join(w if w.upper() in {"LLC", "LP", "HVAC", "CPAS"} else w.capitalize() for w in s.split())
    return s


def pairwise_scores(pred: dict[str, str], truth: dict[str, str]) -> dict:
    """Pairwise precision and recall over distinct raw names.

    A pair of names is a predicted match if they share a cluster and a true match if they share a
    true parent. Counting pairs through the contingency table avoids building all n^2 pairs."""
    names = [n for n in pred if n in truth]
    joint = Counter((pred[n], truth[n]) for n in names)
    by_pred = Counter(pred[n] for n in names)
    by_true = Counter(truth[n] for n in names)

    def c2(k: int) -> int:
        return k * (k - 1) // 2

    tp = sum(c2(v) for v in joint.values())
    pp = sum(c2(v) for v in by_pred.values())
    tt = sum(c2(v) for v in by_true.values())
    precision = tp / pp if pp else 1.0
    recall = tp / tt if tt else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_pairs": tt,
        "predicted_pairs": pp,
        "true_positive_pairs": tp,
        "clusters": len(by_pred),
        "true_parents": len(by_true),
    }


class ReviewBook:
    """Review decisions on top of the automatic clusters. Decisions are kept in order so they can be
    replayed onto a fresh clustering (the next monthly refresh)."""

    def __init__(self, clusters: Clusters):
        self.c = clusters
        self.decisions: list[dict] = []
        self.open = {q["id"]: q for q in clusters.review}

    def apply(self, item_id: str, action: str, name: str | None = None) -> dict:
        q = self.open.get(item_id)
        if q is None:
            raise KeyError(item_id)
        if action == "merge":
            if q["b"] is None:
                raise ValueError("merge needs a pair item")
            self._merge(q["a"], q["b"])
        elif action == "split":
            if not name or self.c.cluster_of.get(name) not in (q["a"], q["b"]):
                raise ValueError("split needs a raw name from this item")
            self._split(name)
        elif action not in ("confirm", "reject"):
            raise ValueError(f"unknown action {action!r}")
        rec = {"id": item_id, "action": action, "name": name, "a": q["a"], "b": q["b"]}
        self.decisions.append(rec)
        del self.open[item_id]
        return rec

    def _merge(self, a: str, b: str) -> None:
        if a not in self.c.members or b not in self.c.members:
            return
        for n in self.c.members.pop(b):
            self.c.cluster_of[n] = a
            self.c.members[a].append(n)
        self.c.display.pop(b, None)
        self.c.confidence[a] = 1.0
        self.c.confidence.pop(b, None)

    def _split(self, name: str) -> None:
        old = self.c.cluster_of[name]
        new = f"P{len(self.c.members) + len(self.decisions) + 9000:04d}"
        self.c.members[old].remove(name)
        self.c.members[new] = [name]
        self.c.cluster_of[name] = new
        self.c.display[new] = clean_display(name)
        self.c.confidence[new] = 1.0
