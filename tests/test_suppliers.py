import pytest

from spendintel.suppliers import (
    ReviewBook,
    clean_name,
    pairwise_scores,
    resolve,
    similarity,
    weighted_jaccard,
    within_one_edit,
)

ACME = [
    "ACME Industrial Supply",
    "Acme Ind. Supply Co",
    "ACME INDUSTRIAL SUPPLY LLC",
    "The Acme Industrial Supply, Inc.",
]


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("ACME INDUSTRIAL SUPPLY LLC", "acme industrial supply"),
        ("Acme Ind. Supply Co", "acme industrial supply"),
        ("The Acme Industrial Supply, Inc.", "acme industrial supply"),
        ("Halston Pallet & Crate Co. - Midwest Division", "halston pallet and crate"),
        ("Delta Wire and Cable Corp dba Direct", "delta wire and cable"),
        ("Brightline Svcs, LLC", "brightline services"),
        ("Company", "company"),
    ],
)
def test_clean_name(raw, clean):
    assert clean_name(raw) == clean


def test_within_one_edit():
    assert within_one_edit("metals", "metlas")  # swap
    assert within_one_edit("metals", "mtals")  # delete
    assert within_one_edit("metals", "metalls")  # insert
    assert within_one_edit("metals", "metels")  # substitute
    assert not within_one_edit("metals", "plastics")


def test_typo_scores_high_and_different_company_scores_low():
    assert similarity("acme metals", "acme metlas") > 0.75
    assert similarity("apex packaging", "apex plastics") < 0.62


def test_idf_weighting_makes_rare_tokens_count_more():
    idf = {"halston": 4.0, "supply": 0.5, "industrial": 0.5, "keystone": 4.0}
    same_rare = weighted_jaccard({"halston", "supply"}, {"halston", "industrial"}, idf)
    same_common = weighted_jaccard({"halston", "supply"}, {"keystone", "supply"}, idf)
    assert same_rare > same_common


def test_variants_cluster_together():
    c = resolve(ACME + ["Apex Packaging Corp", "APEX PACKAGING LTD"])
    assert len({c.cluster_of[n] for n in ACME}) == 1
    assert c.cluster_of["Apex Packaging Corp"] == c.cluster_of["APEX PACKAGING LTD"]
    assert c.cluster_of["Apex Packaging Corp"] != c.cluster_of[ACME[0]]


def test_hard_negatives_stay_apart():
    names = ["Apex Packaging Inc", "Apex Plastics Inc", "Apex Freight LLC", "Summit Steel Co", "Summit Staffing Co"]
    c = resolve(names)
    assert len(set(c.cluster_of.values())) == 5


def test_display_name_comes_from_the_biggest_spelling():
    c = resolve(ACME, weights={"Acme Ind. Supply Co": 10.0, "ACME INDUSTRIAL SUPPLY LLC": 500.0})
    assert c.display[c.cluster_of[ACME[0]]] == "Acme Industrial Supply LLC"


def test_gray_zone_goes_to_review_queue():
    c = resolve(["Cedar Metal Service Center", "Cedar Metal Ltd"], auto=0.95, review=0.5)
    assert len(set(c.cluster_of.values())) == 2
    assert any(q["kind"] == "possible_merge" for q in c.review)


def test_blocking_skips_unrelated_pairs():
    names = [
        f"{p} {w}" for p in ("Alder", "Beacon", "Cedar", "Delta", "Eastgate") for w in ("Steel", "Freight", "Polymers")
    ]
    c = resolve(names)
    assert c.stats["pairs_compared"] < c.stats["pairs_possible"]


def test_pairwise_scores_by_hand():
    truth = {"a": "X", "b": "X", "c": "X", "d": "Y"}
    pred = {"a": "1", "b": "1", "c": "2", "d": "2"}
    s = pairwise_scores(pred, truth)
    # true pairs: ab ac bc = 3. predicted: ab, cd = 2. correct: ab = 1.
    assert s["precision"] == 0.5
    assert s["recall"] == round(1 / 3, 4)


def test_review_merge_and_split():
    c = resolve(["Cedar Metal Service Center", "Cedar Metal Ltd"], auto=0.95, review=0.5)
    book = ReviewBook(c)
    item = next(q for q in c.review if q["kind"] == "possible_merge")
    book.apply(item["id"], "merge")
    assert c.cluster_of["Cedar Metal Service Center"] == c.cluster_of["Cedar Metal Ltd"]
    assert item["id"] not in book.open
    c2 = resolve(ACME, auto=0.8)
    c2.review.append({"id": "R9", "kind": "confirm_cluster", "a": c2.cluster_of[ACME[0]], "b": None})
    book2 = ReviewBook(c2)
    book2.apply("R9", "split", ACME[1])
    assert c2.cluster_of[ACME[1]] != c2.cluster_of[ACME[0]]


def test_review_rejects_bad_actions():
    c = resolve(["Cedar Metal Service Center", "Cedar Metal Ltd"], auto=0.95, review=0.5)
    book = ReviewBook(c)
    qid = c.review[0]["id"]
    with pytest.raises(ValueError):
        book.apply(qid, "explode")
    with pytest.raises(KeyError):
        book.apply("nope", "confirm")
