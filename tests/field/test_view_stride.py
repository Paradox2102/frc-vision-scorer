"""Invariants for per-view frame decimation (corner-cam stride)."""
import pytest

from web.field.view_stride import build_view_frame_plan, scale_frame_knob


def _frames(n, start=100, skip=4):
    return [start + i * skip for i in range(n)]


class TestBuildViewFramePlan:
    def test_stride_one_is_identity(self):
        frames = _frames(10)
        plan = build_view_frame_plan(frames, prompt_local_idx=3, stride=1)
        assert plan.source_frames == frames
        assert plan.prompt_idx == 3
        assert plan.source_to_local == {f: i for i, f in enumerate(frames)}

    def test_prompt_frame_always_kept(self):
        frames = _frames(20)
        for stride in (1, 2, 3, 4, 7):
            for prompt in (0, 1, 5, 19):
                plan = build_view_frame_plan(frames, prompt, stride)
                # The prompt frame survives decimation under its own index.
                assert plan.source_frames[plan.prompt_idx] == frames[prompt]

    def test_stride_two_keeps_alternating_frames(self):
        frames = _frames(8)
        plan = build_view_frame_plan(frames, prompt_local_idx=3, stride=2)
        # Phase-aligned to the prompt: odd local indices 1, 3, 5, 7.
        assert plan.source_frames == [frames[i] for i in (1, 3, 5, 7)]
        assert plan.prompt_idx == 1

    def test_source_to_local_inverts_source_frames(self):
        frames = _frames(30)
        plan = build_view_frame_plan(frames, prompt_local_idx=10, stride=3)
        for local, src in enumerate(plan.source_frames):
            assert plan.source_to_local[src] == local
        # Dropped frames have no mapping (gap, not stale reuse).
        dropped = set(frames) - set(plan.source_frames)
        assert dropped
        for src in dropped:
            assert src not in plan.source_to_local

    def test_local_indices_are_contiguous_from_zero(self):
        # SAM trackers key output by position in the sorted frames dir, so
        # the per-view local index space must be 0..N-1 with no holes.
        frames = _frames(11)
        plan = build_view_frame_plan(frames, prompt_local_idx=4, stride=3)
        assert sorted(plan.source_to_local.values()) == list(
            range(len(plan.source_frames)))

    def test_at_most_stride_minus_one_leading_frames_dropped(self):
        frames = _frames(40)
        for stride in (2, 3, 5):
            plan = build_view_frame_plan(frames, prompt_local_idx=17, stride=stride)
            first_kept_idx = frames.index(plan.source_frames[0])
            assert first_kept_idx < stride

    def test_prompt_out_of_range_raises(self):
        with pytest.raises(ValueError):
            build_view_frame_plan(_frames(5), prompt_local_idx=5, stride=2)
        with pytest.raises(ValueError):
            build_view_frame_plan(_frames(5), prompt_local_idx=-1, stride=2)

    def test_stride_below_one_clamped(self):
        frames = _frames(6)
        plan = build_view_frame_plan(frames, prompt_local_idx=2, stride=0)
        assert plan.source_frames == frames


class TestDenseHead:
    def test_dense_head_kept_in_full(self):
        frames = _frames(40)
        plan = build_view_frame_plan(frames, prompt_local_idx=3, stride=2,
                                     dense_frames=10)
        # First 10 kept indices survive untouched...
        assert plan.source_frames[:10] == frames[:10]
        # ...then the tail is strided (phase-aligned to the prompt: odd).
        assert plan.source_frames[10:] == [frames[i] for i in range(11, 40, 2)]

    def test_dense_zero_equals_pure_stride(self):
        frames = _frames(20)
        a = build_view_frame_plan(frames, 4, 3, dense_frames=0)
        b = build_view_frame_plan(frames, 4, 3)
        assert a == b

    def test_dense_covers_everything(self):
        frames = _frames(12)
        plan = build_view_frame_plan(frames, 2, 4, dense_frames=100)
        assert plan.source_frames == frames

    def test_prompt_in_dense_head_kept(self):
        frames = _frames(30)
        # Prompt at idx 5, dense head 10: kept via the head even when the
        # stride phase wouldn't align.
        plan = build_view_frame_plan(frames, 5, 4, dense_frames=10)
        assert plan.source_frames[plan.prompt_idx] == frames[5]

    def test_mapping_invariants_hold_with_dense_head(self):
        frames = _frames(50)
        plan = build_view_frame_plan(frames, 7, 3, dense_frames=15)
        assert sorted(plan.source_to_local.values()) == list(
            range(len(plan.source_frames)))
        for local, src in enumerate(plan.source_frames):
            assert plan.source_to_local[src] == local


class TestScaleFrameKnob:
    def test_stride_one_identity(self):
        assert scale_frame_knob(8, 1) == 8

    def test_halves_with_stride_two(self):
        assert scale_frame_knob(8, 2) == 4

    def test_never_below_one(self):
        assert scale_frame_knob(1, 4) == 1
        assert scale_frame_knob(2, 8) == 1

    def test_rounds_to_nearest(self):
        assert scale_frame_knob(8, 3) == 3
