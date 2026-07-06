"""Within-batch team-vote dedupe: one VLM batch returning the same team for
multiple crops (two blue crops near the parked 2992 both read "2992") must
cast at most ONE vote for that team — the strongest read wins, the rest are
discarded (qm11 forensics: L186-188 voted 2992 into B1 and B3 simultaneously,
the duplicate-label bug)."""
from web.field.identity_confidence import dedupe_batch_team_reads


def test_duplicate_team_keeps_strongest_read():
    reads = [
        ("2992", 0.76, 0.5),   # weaker: conf*weight = 0.38
        ("2992", 0.88, 1.0),   # stronger: 0.88
        ("3181", 1.00, 1.0),
    ]
    assert dedupe_batch_team_reads(reads) == [False, True, True]


def test_unique_teams_all_vote():
    reads = [("4786", 0.5, 1.0), ("3181", 0.9, 0.5), (None, 0.0, 1.0)]
    assert dedupe_batch_team_reads(reads) == [True, True, True]


def test_none_teams_never_blocked():
    reads = [(None, 0.9, 1.0), (None, 0.9, 1.0)]
    assert dedupe_batch_team_reads(reads) == [True, True]


def test_tie_keeps_first():
    reads = [("2992", 0.8, 1.0), ("2992", 0.8, 1.0)]
    assert dedupe_batch_team_reads(reads) == [True, False]


def test_three_way_duplicate():
    reads = [("6800", 0.3, 1.0), ("6800", 0.9, 0.5), ("6800", 1.0, 1.0)]
    assert dedupe_batch_team_reads(reads) == [False, False, True]


# Cross-slot tentative-claim arbitration: accumulated votes (across batches)
# can still put the same team's top vote on two robots; the HUD must award a
# tentative "TEAM?" label to only the strongest claimant (qm11: B1 2992:0.5
# vs B3 2992:2.0 both displayed "2992?").
from web.field.identity_confidence import tentative_claim_winners, best_owned_team


def test_claim_winner_is_highest_vote():
    votes = {"b1": {"2992": 0.5, "3181": 0.5}, "b3": {"2992": 2.0}}
    assert tentative_claim_winners(votes) == {"2992": "b3", "3181": "b1"}


def test_claim_tie_is_stable_first_by_key():
    votes = {"a": {"2992": 1.0}, "b": {"2992": 1.0}}
    assert tentative_claim_winners(votes)["2992"] == "a"


def test_best_owned_team_skips_lost_claims():
    winners = {"2992": "b3", "3181": "b1"}
    # b1's top vote (2992) is owned by b3 -> falls through to 3181
    assert best_owned_team({"2992": 0.5, "3181": 0.4}, "b1", winners) == ("3181", 0.4)
    # b3 owns its top claim
    assert best_owned_team({"2992": 2.0}, "b3", winners) == ("2992", 2.0)
    # nothing owned -> None
    assert best_owned_team({"2992": 0.5}, "b1", winners) == (None, 0.0)
    # empty votes
    assert best_owned_team({}, "b1", winners) == (None, 0.0)
