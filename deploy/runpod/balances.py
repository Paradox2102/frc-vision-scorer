"""Print the RunPod and vast.ai account balances as one JSON line.

    .venv/bin/python deploy/runpod/balances.py

{"runpod": {"balance": 89.48, "spend_per_hr": 0.0, "spend_limit": 80},
 "vast": {"credit": 104.73},
 "jev": {"error": "..."}}

Each provider is independent: one that has no key or does not answer gets
{"error": "..."} and the others still print. Keys come from the environment or
the main checkout's .env (runpod_cli / vast); nothing secret is printed. Jev
(TypeSafe) publishes no balance endpoint (its OpenAPI lists only /v1/models
and /v1/systemone, checked Nov 24 2026), so it is always an error entry.
The frc-servers Claude mod runs this every few minutes.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod_cli  # noqa: E402
import vast  # noqa: E402

TIMEOUT_S = 10


def _err(e: BaseException) -> dict:
    # SystemExit carries the helpers' own messages (no key, HTTP status);
    # keep them short and never echo a request.
    return {"error": (str(e) or type(e).__name__)[:200]}


def runpod_balance() -> dict:
    try:
        d = runpod_cli._request(
            "POST", runpod_cli.GRAPHQL,
            {"query": "query { myself { clientBalance currentSpendPerHr spendLimit } }"},
            timeout=TIMEOUT_S)
        if d.get("errors"):
            return {"error": "GraphQL error"}
        me = (d.get("data") or {}).get("myself") or {}
        return {"balance": me.get("clientBalance"),
                "spend_per_hr": me.get("currentSpendPerHr"),
                "spend_limit": me.get("spendLimit")}
    except BaseException as e:  # SystemExit from a missing key / HTTP error
        return _err(e)


def vast_balance() -> dict:
    try:
        u = vast._request("GET", "/users/current", timeout=TIMEOUT_S, retries=1) or {}
        return {"credit": u.get("credit")}
    except BaseException as e:
        return _err(e)


def main() -> None:
    print(json.dumps({
        "runpod": runpod_balance(),
        "vast": vast_balance(),
        "jev": {"error": "Jev has no balance API"},
    }))


if __name__ == "__main__":
    main()
