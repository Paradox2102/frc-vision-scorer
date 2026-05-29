#!/usr/bin/env python3
"""Sample diverse hub-zone crops for labeling.

Instead of labeling 3,464 near-identical frames, sample ~300 diverse ones:
- Every 30th frame (1 per second at 30fps, every 10th crop since crops are every 3rd frame)
- Ensure representation from all 3 videos and both alliances
- Prioritize frames where v5 detects balls (potential scoring moments)
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).parent.parent
FILTERED_DIR = PROJECT / "data" / "hub_zones_filtered"
SAMPLE_DIR = PROJECT / "data" / "hub_zones_sample"
METADATA = FILTERED_DIR / "filtered_metadata.csv"

def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for d in ["blue", "red"]:
        (SAMPLE_DIR / d).mkdir(exist_ok=True)

    rows = []
    with open(METADATA) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Group by video + alliance
    groups = defaultdict(list)
    for row in rows:
        key = (row["video"], row["alliance"])
        groups[key].append(row)

    sampled = []
    for (video, alliance), group_rows in sorted(groups.items()):
        # Sample every 10th crop (= every 30th frame since crops are every 3 frames)
        for i, row in enumerate(group_rows):
            if i % 10 == 0:
                sampled.append(row)

    # Copy sampled crops
    sample_meta = []
    for row in sampled:
        src = PROJECT / row["crop_filename"]
        alliance = row["alliance"]
        fname = f"{row['video']}_{Path(row['crop_filename']).name}"
        dst = SAMPLE_DIR / alliance / fname
        if src.exists():
            shutil.copy2(src, dst)
            sample_meta.append({**row, "sample_filename": str(dst.relative_to(PROJECT))})

    # Write sample metadata
    with open(SAMPLE_DIR / "sample_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_meta[0].keys()))
        writer.writeheader()
        writer.writerows(sample_meta)

    # Stats
    by_video = defaultdict(int)
    by_alliance = defaultdict(int)
    for row in sample_meta:
        by_video[row["video"]] += 1
        by_alliance[row["alliance"]] += 1

    print(f"Sampled {len(sample_meta)} crops from {len(rows)} filtered")
    print(f"By alliance: {dict(by_alliance)}")
    print(f"By video: {dict(by_video)}")
    print(f"Saved to: {SAMPLE_DIR}")

if __name__ == "__main__":
    main()
