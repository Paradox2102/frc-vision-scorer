"""vast.ai provider for runpod_cli.py (stdlib only): ONE rented instance that
the existing single-pod commands (tune / pull / status / ssh / logs / stop /
terminate) drive exactly like a RunPod pod.

REST API (endpoints + field names from the vast CLI source, vastai/api/*.py):

    POST   /api/v0/bundles/              offer search (body = filter dict)
    PUT    /api/v0/asks/{offer_id}/      create an instance from an offer
    GET    /api/v0/instances/{id}/?owner=me
    PUT    /api/v0/instances/{id}/       {"state": "stopped" | "running"}
    DELETE /api/v0/instances/{id}/       destroy (instance + its disk)
    GET    /api/v0/users/current         balance / credit
    GET    /api/v0/ssh/                  the account's registered ssh keys

Auth: ``Authorization: Bearer $VASTAI_API`` (env or the main checkout's .env;
never sent to the instance, never printed).

ssh: vast replaces the image entrypoint in ssh launch mode, runs its own sshd
and injects the ACCOUNT's registered ssh public keys into authorized_keys.
Nothing secret goes into the create body (``env`` is empty); the pod .env is
installed over ssh afterwards, as on RunPod.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://console.vast.ai/api/v0"
UA = "frc-runpod-cli/1.0 (vast)"

# vast ssh launch mode: direct (host port mapped to 22) with the proxy
# (sshN.vast.ai) as fallback. Same string the vast CLI builds for --ssh --direct.
RUNTYPE = "ssh_direc ssh_proxy"
# Runs at container start (vast's entrypoint). Not secret: vast's login shells
# auto-attach tmux unless ~/.no_auto_tmux exists, and /workspace is where the
# pod layout lives (on vast it is just a directory on the container disk).
# The runpod/pytorch image ships WITHOUT ssh host keys (its own /start.sh makes
# them, but vast replaces the entrypoint): vast's sshd then exits with "no
# hostkeys available" and port 22 refuses. Generate them and start sshd here
# (measured on vast instance 60611414, Nov 14 2026).
ONSTART = ("ssh-keygen -A; mkdir -p /run/sshd; pgrep -x sshd >/dev/null || /usr/sbin/sshd; "
           "touch /root/.no_auto_tmux; mkdir -p /workspace")
# vast's own base image: its ssh launch mode works out of the box (Ubuntu +
# sshd + host keys). The RunPod image does NOT (no host keys; see ONSTART).
# bootstrap.sh needs only apt, python3 and curl, which it has; it builds
# its own uv venv (Python 3.12, torch cu128) as on RunPod. Verified Nov 14 2026
# on instance 60615714 (ssh 43 s, bootstrap ~5 min, full tune ladder ran).
DEFAULT_IMAGE = "vastai/base-image:@vastai-automatic-tag"
DEFAULT_DISK_GB = 120

OFFER_COLUMNS = (
    "id", "gpu_name", "num_gpus", "gpu_ram", "cpu_name", "cpu_ghz", "cpu_cores",
    "cpu_cores_effective", "cpu_ram", "dph_total", "inet_down", "inet_up",
    "inet_down_cost", "inet_up_cost", "storage_cost", "disk_space", "cuda_max_good",
    "driver_version", "reliability", "verified", "geolocation", "direct_port_count",
)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def api_key() -> str:
    k = os.environ.get("VASTAI_API")
    if not k:
        import runpod_cli  # lazy: runpod_cli imports this module
        k = runpod_cli.local_env().get("VASTAI_API")
    if not k:
        raise SystemExit("VASTAI_API not found in the environment or the main checkout's .env")
    return k


def _request(method: str, path: str, body=None, *, query: dict | None = None,
             timeout: float = 60, retries: int = 3):
    """JSON request to the vast REST API. 429 is retried (rate limit: the
    request was not processed); 502/503/504 only for GETs (a create must
    never be sent twice)."""
    url = API + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
            "Accept": "application/json", "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            retry = e.code == 429 or (method == "GET" and e.code in (502, 503, 504))
            if retry and attempt < retries - 1:
                try:
                    wait = float(e.headers.get("Retry-After") or 0) or 2.0 * (attempt + 1)
                except (TypeError, ValueError):
                    wait = 2.0 * (attempt + 1)
                time.sleep(min(wait, 15.0))
                continue
            raise SystemExit(f"vast {method} {path} -> HTTP {e.code}: "
                             f"{e.read()[:400].decode(errors='replace')}")
    raise SystemExit(f"vast {method} {path}: retries exhausted")


# --------------------------------------------------------------------------
# Offers
# --------------------------------------------------------------------------
def offer_query(*, gpus=("RTX 5090", "RTX 4090"), verified: bool = True, min_cuda: float = 12.8,
                min_disk: float = DEFAULT_DISK_GB, min_reliability: float = 0.97,
                min_ghz: float = 0.0, min_cores: float = 8, max_dph: float | None = None,
                limit: int = 50) -> dict:
    """POST /bundles/ body for single-GPU on-demand offers (vast CLI defaults:
    rentable, not rented, not external; ``verified`` dropped with False)."""
    q = {
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "external": {"eq": False},
        "num_gpus": {"eq": 1},
        "gpu_name": {"in": [g.replace("_", " ") for g in gpus]},
        "cuda_max_good": {"gte": float(min_cuda)},
        "direct_port_count": {"gte": 1},
        "disk_space": {"gte": float(min_disk)},
        "reliability": {"gte": float(min_reliability)},
        "cpu_cores_effective": {"gte": float(min_cores)},
    }
    if verified:
        q["verified"] = {"eq": True}
    if min_ghz:
        q["cpu_ghz"] = {"gte": float(min_ghz)}
    if max_dph:
        q["dph_total"] = {"lte": float(max_dph)}
    q["order"] = [["cpu_ghz", "desc"], ["dph_total", "asc"]]
    q["type"] = "on-demand"
    q["limit"] = int(limit)
    q["allocated_storage"] = float(min_disk)    # prices dph_total with this much disk
    return q


def sort_offers(offers: list) -> list:
    """cpu_ghz desc, then dph_total asc (the server orders too; this makes the
    table independent of how it breaks ties)."""
    return sorted(offers, key=lambda o: (-(o.get("cpu_ghz") or 0), o.get("dph_total") or 1e9))


def search_offers(query: dict) -> list:
    return sort_offers(_request("POST", "/bundles/", query).get("offers") or [])


def get_offer(offer_id: int, storage_gb: float = DEFAULT_DISK_GB) -> dict | None:
    """The offer's current listing (read-only search by id), None if it is gone."""
    q = {"id": {"eq": int(offer_id)}, "type": "on-demand", "allocated_storage": float(storage_gb)}
    for o in _request("POST", "/bundles/", q).get("offers") or []:
        if int(o.get("id", -1)) == int(offer_id):
            return o
    return None


def per_tb(cost_per_gb) -> float | None:
    """inet_*_cost is $/GB; the vast CLI shows $/TB as x1024."""
    return None if cost_per_gb is None else float(cost_per_gb) * 1024


def _f(v, fmt):
    return "-" if v is None else format(v, fmt)


def format_offers(offers: list) -> str:
    hdr = (f"{'id':>9} {'gpu':<9} {'cpu_name':<30} {'GHz':>4} {'eff':>5} {'RAM':>5} {'$/h':>6} "
           f"{'down':>6} {'$/TBd':>6} {'$/TBu':>6} {'cuda':>5} {'rel':>5} geo")
    lines = [hdr]
    for o in offers:
        ram = o.get("cpu_ram")
        lines.append(
            f"{o.get('id', '-'):>9} {str(o.get('gpu_name') or '-')[:9]:<9} "
            f"{str(o.get('cpu_name') or '-')[:30]:<30} {_f(o.get('cpu_ghz'), '.1f'):>4} "
            f"{_f(o.get('cpu_cores_effective'), '.1f'):>5} "
            f"{('-' if ram is None else f'{ram / 1000:.0f}G'):>5} {_f(o.get('dph_total'), '.3f'):>6} "
            f"{_f(o.get('inet_down'), '.0f'):>6} {_f(per_tb(o.get('inet_down_cost')), '.2f'):>6} "
            f"{_f(per_tb(o.get('inet_up_cost')), '.2f'):>6} {_f(o.get('cuda_max_good'), '.1f'):>5} "
            f"{_f(o.get('reliability'), '.3f'):>5} {o.get('geolocation') or '-'}")
    return "\n".join(lines)


def offer_summary(o: dict) -> str:
    ram = o.get("cpu_ram")
    return (f"offer {o.get('id')}: {o.get('gpu_name')} ({_f((o.get('gpu_ram') or 0) / 1000, '.0f')} GB) on "
            f"{o.get('cpu_name') or '?'} {_f(o.get('cpu_ghz'), '.1f')} GHz, "
            f"{_f(o.get('cpu_cores_effective'), '.1f')} eff cores, "
            f"{'-' if ram is None else f'{ram / 1000:.0f}'} GB RAM; ${_f(o.get('dph_total'), '.3f')}/h "
            f"(+ ${_f(per_tb(o.get('inet_down_cost')), '.2f')}/TB down, "
            f"${_f(per_tb(o.get('inet_up_cost')), '.2f')}/TB up); CUDA {o.get('cuda_max_good')}, "
            f"reliability {_f(o.get('reliability'), '.3f')}, {o.get('geolocation') or '?'}")


# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------
def build_create_body(*, image: str, disk_gb: int, label: str) -> dict:
    """PUT /asks/{offer_id}/ body (field set of the vast CLI's
    build_create_instance_payload). ``env`` is empty: no secret, and no
    PUBLIC_KEY (vast injects the account's registered ssh keys)."""
    return {
        "client_id": "me",
        "image": image,
        "env": {},
        "price": None,               # on-demand (no bid)
        "disk": int(disk_gb),
        "label": label,
        "extra": None,
        "onstart": ONSTART,
        "image_login": None,
        "python_utf8": False,
        "lang_utf8": False,
        "use_jupyter_lab": False,
        "jupyter_dir": None,
        "force": False,
        "cancel_unavail": True,      # error out if the offer is gone (no stopped placeholder)
        "template_hash_id": None,
        "user": None,
        "runtype": RUNTYPE,
    }


def create_instance(offer_id: int, *, image: str, disk_gb: int, label: str) -> dict:
    """Rents the offer. Returns {"id", "raw"} where raw has vast's response
    MINUS any instance-scoped API key (never stored or printed)."""
    r = _request("PUT", f"/asks/{int(offer_id)}/", build_create_body(image=image, disk_gb=disk_gb, label=label))
    r = {k: v for k, v in (r or {}).items() if "api_key" not in k}
    if not r.get("success") or not r.get("new_contract"):
        raise SystemExit(f"vast create failed: {json.dumps(r)[:400]}")
    return {"id": int(r["new_contract"]), "raw": r}


def show_instance(instance_id) -> dict:
    d = _request("GET", f"/instances/{int(instance_id)}/", query={"owner": "me"})
    inst = d.get("instances") if isinstance(d, dict) else None
    if not inst:
        raise SystemExit(f"vast instance {instance_id} not found (destroyed?)")
    return inst


def stop_instance(instance_id) -> dict:
    return _request("PUT", f"/instances/{int(instance_id)}/", {"state": "stopped"})


def start_instance(instance_id) -> dict:
    return _request("PUT", f"/instances/{int(instance_id)}/", {"state": "running"})


def destroy_instance(instance_id) -> dict:
    return _request("DELETE", f"/instances/{int(instance_id)}/", {})


def current_user() -> dict:
    u = dict(_request("GET", "/users/current") or {})
    for k in list(u):
        if "api_key" in k or "token" in k:
            u.pop(k)
    return u


def account_ssh_keys() -> list:
    """Public keys registered on the account (any string field that looks
    like an OpenSSH public key; the response shape is not documented)."""
    found = []

    def walk(v):
        if isinstance(v, str):
            s = v.strip()
            if s.startswith(("ssh-", "ecdsa-", "sk-")):
                found.append(s)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(_request("GET", "/ssh/"))
    return found


def key_registered(pubkey: str, keys: list) -> bool:
    """Same key type + base64 body (comments differ freely)."""
    parts = pubkey.split()
    if len(parts) < 2:
        return False
    return any(k.split()[:2] == parts[:2] for k in keys)


def map_status(actual_status) -> str:
    """vast actual_status -> RunPod desiredStatus vocabulary (what the
    existing code checks)."""
    s = (actual_status or "").strip().lower()
    if s == "running":
        return "RUNNING"
    if s in ("stopped", "exited"):
        return "EXITED"
    return s.upper() if s else "UNKNOWN"


def ssh_endpoint(inst: dict) -> tuple:
    """(host, port, kind): the DIRECT endpoint (public_ipaddr + the host
    port mapped to 22/tcp) when the instance has one, else vast's ssh proxy
    (ssh_host/ssh_port), else (None, None, None)."""
    ports = inst.get("ports") or {}
    p22 = ports.get("22/tcp") if isinstance(ports, dict) else None
    ip = (inst.get("public_ipaddr") or "").strip()
    if ip and p22:
        try:
            return ip, int(p22[0]["HostPort"]), "direct"
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    if inst.get("ssh_host") and inst.get("ssh_port"):
        return str(inst["ssh_host"]).strip(), int(inst["ssh_port"]), "proxy"
    return None, None, None


def instance_endpoint(inst: dict) -> tuple:
    """(status, cost_per_hr, ssh_host, ssh_port)."""
    host, port, _ = ssh_endpoint(inst)
    return map_status(inst.get("actual_status")), inst.get("dph_total"), host, port


def pod_view(inst: dict) -> dict:
    """The instance in the shape runpod_cli reads from a RunPod pod
    (desiredStatus, costPerHr, publicIp, portMappings["22"], gpu, image,
    lastStartedAt), with the raw instance under "vast"."""
    status, cost, host, port = instance_endpoint(inst)
    started = inst.get("start_date")
    iso = (_dt.datetime.fromtimestamp(float(started), _dt.timezone.utc).isoformat(timespec="seconds")
           if started else None)
    return {
        "id": str(inst.get("id")),
        "desiredStatus": status,
        "costPerHr": cost,
        "adjustedCostPerHr": cost,
        "publicIp": host,
        "portMappings": {"22": port} if port else {},
        "gpu": {"displayName": inst.get("gpu_name")},
        "image": inst.get("image_uuid"),
        "lastStartedAt": iso,
        "vast": inst,
    }
