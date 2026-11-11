"""Where the student checkpoints and the bumper detector weights live.

Data and weights sit in the main checkout even when this code runs from a git
worktree, so paths resolve against the parent of the git common dir."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main_checkout(repo: Path = REPO) -> Path:
    """The main checkout (data, weights, .env live there even when this code
    runs from a git worktree)."""
    try:
        common = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        p = Path(common)
        if not p.is_absolute():
            p = (repo / p).resolve()
        return p.parent
    except Exception:
        return repo


MAIN = main_checkout()
MODELS = MAIN / "data/teacher/models"
BUMPER_WEIGHTS = MAIN / "runs/train/frc_bumpers_2026cur_v3/weights/best.pt"
