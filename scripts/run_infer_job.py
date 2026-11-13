"""Headless driver for an inference job's websocket.

The server runs process_video() inside the /ws/infer/{job_id} handler, so a
client must connect and STAY connected for the whole run. This script connects,
sends the match roster (so VLM team reads bind to real team numbers, as the
frontend would), then drains messages until the job completes or errors.

Usage:
    .venv/bin/python scripts/run_infer_job.py <job_id> \
        --blue 8513 573 4174 --red 401 6328 3506

Prints a one-line progress heartbeat and the final status/identity payloads.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets


async def run(job_id: str, blue: list[str], red: list[str], host: str) -> int:
    uri = f"ws://{host}/ws/infer/{job_id}"
    last_beat = 0.0
    final_payloads = []
    # Job runs can stream for an hour with quiet stretches during SAM
    # propagation; rely on server completion rather than a recv timeout.
    async with websockets.connect(uri, max_size=None, ping_interval=20,
                                  ping_timeout=600) as ws:
        if blue or red:
            await ws.send(json.dumps({
                "type": "robot_identifications",
                "match_teams": blue + red,
                "blue_teams": blue,
                "red_teams": red,
            }))
        while True:
            try:
                raw = await ws.recv()
            except websockets.ConnectionClosed:
                print("\n[runner] websocket closed by server")
                break
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            mtype = msg.get("type")
            if mtype == "status":
                now = time.time()
                if now - last_beat >= 10:
                    last_beat = now
                    print(f"[{time.strftime('%H:%M:%S')}] "
                          f"{msg.get('progress')}% {msg.get('message')}",
                          flush=True)
                if msg.get("status") in ("completed", "error"):
                    print(json.dumps(msg, indent=2)[:2000], flush=True)
                    if msg.get("status") == "error":
                        return 1
            elif mtype == "error":
                print(f"[runner] ERROR: {msg.get('error')}", flush=True)
                return 1
            elif mtype in ("complete", "final_scores", "identity_summary"):
                final_payloads.append(msg)
                print(f"[runner] final payload: {mtype}", flush=True)
    for p in final_payloads:
        print(json.dumps(p, indent=2)[:4000], flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--blue", nargs="*", default=[])
    ap.add_argument("--red", nargs="*", default=[])
    ap.add_argument("--host", default="localhost:8004")
    args = ap.parse_args()
    return asyncio.run(run(args.job_id, args.blue, args.red, args.host))


if __name__ == "__main__":
    raise SystemExit(main())
