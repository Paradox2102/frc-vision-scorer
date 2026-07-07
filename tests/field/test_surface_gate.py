"""Surface-aware vote gating: the bumper reader reports whether digits sit on
a bumper band or are printed on equipment; the gate (default OFF) zeroes
printed-surface votes (qm11: OVERFLOW crate base read "1477" at prob 1.0)."""
from web.field.identity_confidence import normalize_surface, surface_vote_multiplier


class TestNormalizeSurface:
    def test_bumper_passes_through(self):
        assert normalize_surface("bumper") == "bumper"

    def test_printed_passes_through(self):
        assert normalize_surface("printed") == "printed"

    def test_case_and_whitespace_normalized(self):
        assert normalize_surface("  PRINTED ") == "printed"
        assert normalize_surface("Bumper") == "bumper"

    def test_garbage_is_unknown(self):
        assert normalize_surface("crate") == "unknown"
        assert normalize_surface(None) == "unknown"
        assert normalize_surface("") == "unknown"
        assert normalize_surface(42) == "unknown"


class TestSurfaceVoteMultiplier:
    def test_printed_with_gate_on_drops_vote(self):
        assert surface_vote_multiplier("printed", True) == 0.0

    def test_printed_with_gate_off_keeps_vote(self):
        assert surface_vote_multiplier("printed", False) == 1.0

    def test_bumper_with_gate_on_keeps_vote(self):
        assert surface_vote_multiplier("bumper", True) == 1.0

    def test_unknown_fails_open(self):
        # A reader that doesn't emit the field (older model, error entry)
        # must never lose votes to the gate.
        assert surface_vote_multiplier("unknown", True) == 1.0
        assert surface_vote_multiplier(None, True) == 1.0
