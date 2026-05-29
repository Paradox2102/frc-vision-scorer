#!/usr/bin/env python3
"""Phase 1 evaluation scaffold for FRC Vision Scorer.

Metrics:
1) Per-team made-shot count MAE
2) Event-level precision / recall for made_shot_center

Expected JSONL schema per line (prediction + ground truth):
{
  "match_id": "2026miket_qm1",
  "team_number": "3255",
  "timestamp_s": 43.27,
  "event_label": "made_shot_center"
}

Optional fields are ignored.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ShotEvent:
    match_id: str
    team_number: str
    timestamp_s: float
    event_label: str


@dataclass
class EvaluationResult:
    per_team_count_mae: float
    total_gt_events: int
    total_pred_events: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 scoring outputs")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth events JSONL")
    parser.add_argument("--predictions", required=True, help="Predicted events JSONL")
    parser.add_argument(
        "--event-label",
        default="made_shot_center",
        help="Event label to evaluate (default: made_shot_center)",
    )
    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=0.6,
        help="Temporal tolerance in seconds for matching events",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path for full metric report as JSON",
    )
    return parser.parse_args()


def load_events(path: Path, event_label: str) -> list[ShotEvent]:
    events: list[ShotEvent] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event_label") != event_label:
                continue

            try:
                events.append(
                    ShotEvent(
                        match_id=str(row["match_id"]),
                        team_number=str(row["team_number"]),
                        timestamp_s=float(row["timestamp_s"]),
                        event_label=str(row["event_label"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Missing key at {path}:{line_no}: {exc}") from exc
    return events


def aggregate_team_counts(events: Iterable[ShotEvent]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for e in events:
        key = (e.match_id, e.team_number)
        counts[key] = counts.get(key, 0) + 1
    return counts


def compute_mae(
    gt_counts: dict[tuple[str, str], int],
    pred_counts: dict[tuple[str, str], int],
) -> float:
    keys = sorted(set(gt_counts) | set(pred_counts))
    if not keys:
        return 0.0

    abs_errors = [abs(gt_counts.get(k, 0) - pred_counts.get(k, 0)) for k in keys]
    return sum(abs_errors) / len(abs_errors)


def group_by_match(events: list[ShotEvent]) -> dict[str, list[ShotEvent]]:
    grouped: dict[str, list[ShotEvent]] = {}
    for event in events:
        grouped.setdefault(event.match_id, []).append(event)
    for match_id in grouped:
        grouped[match_id].sort(key=lambda e: e.timestamp_s)
    return grouped


def match_events(
    gt_events: list[ShotEvent],
    pred_events: list[ShotEvent],
    tolerance_s: float,
) -> tuple[int, int, int]:
    """Greedy one-to-one matching by team and nearest timestamp."""

    gt_by_match = group_by_match(gt_events)
    pred_by_match = group_by_match(pred_events)

    tp = 0
    fp = 0
    fn = 0

    for match_id in sorted(set(gt_by_match) | set(pred_by_match)):
        gt_match = gt_by_match.get(match_id, [])
        pred_match = pred_by_match.get(match_id, [])

        gt_used = [False] * len(gt_match)

        for pred in pred_match:
            best_idx = None
            best_dt = None
            for i, gt in enumerate(gt_match):
                if gt_used[i]:
                    continue
                if gt.team_number != pred.team_number:
                    continue
                dt = abs(gt.timestamp_s - pred.timestamp_s)
                if dt > tolerance_s:
                    continue
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_idx = i

            if best_idx is None:
                fp += 1
            else:
                gt_used[best_idx] = True
                tp += 1

        fn += sum(1 for used in gt_used if not used)

    return tp, fp, fn


def safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def evaluate(
    gt_events: list[ShotEvent],
    pred_events: list[ShotEvent],
    tolerance_s: float,
) -> EvaluationResult:
    gt_counts = aggregate_team_counts(gt_events)
    pred_counts = aggregate_team_counts(pred_events)
    mae = compute_mae(gt_counts, pred_counts)

    tp, fp, fn = match_events(gt_events, pred_events, tolerance_s=tolerance_s)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return EvaluationResult(
        per_team_count_mae=mae,
        total_gt_events=len(gt_events),
        total_pred_events=len(pred_events),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def main() -> None:
    args = parse_args()
    gt_path = Path(args.ground_truth).expanduser().resolve()
    pred_path = Path(args.predictions).expanduser().resolve()

    gt_events = load_events(gt_path, args.event_label)
    pred_events = load_events(pred_path, args.event_label)
    result = evaluate(gt_events, pred_events, tolerance_s=args.time_tolerance)

    print("=" * 70)
    print("Phase 1 evaluation summary")
    print("=" * 70)
    print(f"Event label: {args.event_label}")
    print(f"Time tolerance (s): {args.time_tolerance}")
    print(f"GT events: {result.total_gt_events}")
    print(f"Pred events: {result.total_pred_events}")
    print(f"Per-team count MAE: {result.per_team_count_mae:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall: {result.recall:.4f}")
    print(f"F1: {result.f1:.4f}")
    print(f"TP/FP/FN: {result.true_positives}/{result.false_positives}/{result.false_negatives}")

    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
        print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
