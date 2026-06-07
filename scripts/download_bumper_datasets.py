#!/usr/bin/env python3
"""Download the two Roboflow bumper datasets used to train the FRC bumper model.

Reads ROBOFLOW_API from the project .env. Downloads in YOLOv8 format. Defaults to
each project's latest version. Override a version with --benjamin-version /
--karthik-version if the latest changes.

Usage:
    .venv/bin/python scripts/download_bumper_datasets.py
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# (workspace_slug, project_slug, local_dir_name) parsed from the Universe URLs.
BENJAMIN = ("benjamins-workspace-h5vao", "frc-bumpers-h0p1g", "bumper_benjamin")
KARTHIK = ("karthiks-workspace-gez2e", "frc-bumpers-red-blue", "bumper_karthik")


def download_one(rf, workspace, project_slug, dest_name, version_num):
    project = rf.workspace(workspace).project(project_slug)
    if version_num is None:
        versions = project.versions()
        if not versions:
            raise SystemExit(f"No versions found for {workspace}/{project_slug}")
        # Pick the highest version id to be safe (versions() ordering varies).
        version_num = max(int(str(v.version).split("/")[-1]) for v in versions)
    dest = PROJECT_ROOT / "data" / dest_name
    print(f"Downloading {workspace}/{project_slug} v{version_num} -> {dest}")
    project.version(version_num).download("yolov8", location=str(dest), overwrite=True)
    yaml = dest / "data.yaml"
    print(f"  done. data.yaml present: {yaml.exists()}")
    if yaml.exists():
        print("  --- data.yaml ---")
        print("  " + yaml.read_text().replace("\n", "\n  "))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benjamin-version", type=int, default=None)
    ap.add_argument("--karthik-version", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("ROBOFLOW_API")
    if not api_key:
        raise SystemExit("ROBOFLOW_API not set in .env")
    rf = Roboflow(api_key=api_key)

    download_one(rf, *BENJAMIN, args.benjamin_version)
    download_one(rf, *KARTHIK, args.karthik_version)
    print("\nBoth datasets downloaded. Inspect the printed data.yaml class names "
          "before running the merge.")


if __name__ == "__main__":
    main()
