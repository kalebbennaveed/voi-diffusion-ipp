"""Loopback-only interactive demo. JAX jobs run serially outside the request threads."""

import argparse
import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ..config import config_dict, load_config
from .demo_data import ROOT, export_demo

WEB = ROOT / "demo"


def validate_request(data):
    allowed = {
        "scenario",
        "seed",
        "method",
        "variant",
        "wind_std",
        "observation_std",
        "measurements",
        "diffusion_steps",
        "proposals",
    }
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Unsupported experiment options")
    method = data.get("method", "proposed")
    if method not in ("proposed", "current_belief", "cem", "mppi"):
        raise ValueError("Unsupported method")
    variant = data.get("variant", "two_stage")
    if variant not in ("two_stage", "multistage"):
        raise ValueError("Unsupported continuation")
    bounds = {
        "seed": (0, 1000000),
        "wind_std": (0.05, 1.5),
        "observation_std": (0.03, 2.0),
        "diffusion_steps": (2, 8),
        "proposals": (4, 24),
    }
    integers = {"seed", "diffusion_steps", "proposals"}
    values = {}
    for key, (lo, hi) in bounds.items():
        if key in data:
            value = data[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not lo <= value <= hi
            ):
                raise ValueError(f"{key} must be in [{lo}, {hi}]")
            if key in integers and int(value) != value:
                raise ValueError(f"{key} must be an integer")
            values[key] = int(value) if key in integers else float(value)
    if "measurements" in data and not isinstance(data["measurements"], bool):
        raise ValueError("measurements must be boolean")
    seed = values.pop("seed", 102)
    scenario = data.get("scenario", "field_vortices")
    if scenario not in ("field_vortices", "field_shear", "field_mixed"):
        raise ValueError("Select a whole-field POAM scenario")
    cfg = load_config(ROOT / "configs/field_poam.yaml")
    cfg = replace(
        cfg,
        **values,
        scenario=scenario,
        future_updates=(variant == "multistage"),
        measurements=data.get("measurements", True),
    )
    return cfg, method, seed


class State:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jax-planner")

    def catalog(self):
        file = WEB / "data/catalog.json"
        records = json.loads(file.read_text()) if file.exists() else []
        records += [
            json.loads(p.read_text())
            for p in sorted((ROOT / "results/demo/live").glob("*/record.json"))
        ]
        seen = {r["id"] for r in records}
        with self.lock:
            records += [
                j["record"]
                for j in self.jobs.values()
                if j["status"] == "complete" and j["record"]["id"] not in seen
            ]
        return records

    def run(self, cfg, method, seed):
        with self.lock:
            if any(j["status"] in ("queued", "running") for j in self.jobs.values()):
                raise RuntimeError("A JAX experiment is already running. Wait for it to finish.")
            identity = uuid.uuid4().hex
            self.jobs[identity] = {"id": identity, "status": "queued", "message": "Queued"}

        def work():
            def update(message):
                with self.lock:
                    self.jobs[identity].update(status="running", message=message)

            try:
                path = export_demo(cfg, method, seed, ROOT / "results/demo/live" / identity, update)
                numerical_source = b"".join(
                    p.read_bytes()
                    for p in sorted((ROOT / "src").rglob("*.py"))
                    if "/viz/" not in str(p)
                )
                group = hashlib.sha256(
                    json.dumps(config_dict(cfg), sort_keys=True).encode() + numerical_source
                ).hexdigest()[:16]
                record = {
                    "id": identity,
                    "label": f"Live · {method} · seed {seed}",
                    "method": method,
                    "seed": seed,
                    "variant": "multistage" if cfg.future_updates else "two_stage",
                    "group": group,
                    "file": str(path.relative_to(ROOT)),
                    "live": True,
                }
                (path.parent / "record.json").write_text(json.dumps(record, indent=2) + "\n")
                with self.lock:
                    self.jobs[identity].update(status="complete", message="Complete", record=record)
            except Exception as exc:  # noqa: BLE001 -- job boundary must report all worker failures
                import traceback

                traceback.print_exc()
                with self.lock:
                    self.jobs[identity].update(status="failed", message=str(exc))

        self.pool.submit(work)
        return identity


def handler(state):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, data, code=200):
            self.send_bytes(json.dumps(data, allow_nan=False).encode(), "application/json", code)

        def send_bytes(self, data, content_type, code=200):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/catalog":
                return self.send_json(state.catalog())
            if path.startswith("/api/job/"):
                with state.lock:
                    job = state.jobs.get(path.rsplit("/", 1)[-1])
                    return self.send_json(
                        dict(job) if job else {"error": "Unknown job"}, 200 if job else 404
                    )
            if path.startswith("/api/replay/"):
                identity = path.rsplit("/", 1)[-1]
                record = next((r for r in state.catalog() if r["id"] == identity), None)
                if not record:
                    return self.send_json({"error": "Unknown replay"}, 404)
                file = (ROOT / record["file"]).resolve()
                if not file.is_relative_to(ROOT):
                    return self.send_json({"error": "Invalid replay path"}, 400)
                return self.send_bytes(file.read_bytes(), "application/json")
            files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
                "/model.js": ("model.js", "text/javascript"),
            }
            if path not in files:
                return self.send_json({"error": "Not found"}, 404)
            name, mime = files[path]
            self.send_bytes((WEB / name).read_bytes(), mime)

        def do_POST(self):
            if self.path != "/api/run":
                return self.send_json({"error": "Not found"}, 404)
            expected = {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }
            if self.headers.get("Origin") not in expected | {None}:
                return self.send_json({"error": "Local origin required"}, 403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send_json({"error": "JSON required"}, 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 16384:
                    raise ValueError("Invalid request size")
                cfg, method, seed = validate_request(json.loads(self.rfile.read(size)))
                identity = state.run(cfg, method, seed)
                self.send_json({"id": identity}, 202)
            except (ValueError, TypeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, 409)

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    state = State()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(state))
    print(f"VoI Diffusion Lab: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
