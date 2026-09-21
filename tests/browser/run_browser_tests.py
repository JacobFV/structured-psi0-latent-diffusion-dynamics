#!/usr/bin/env python3
"""Run the Playwright workbench acceptance suite against a REAL backend.

Must itself be launched inside a broker lease, e.g.:

  PYTHONPATH=src python3 -m rrp.cli ops run --cpu 3 --mem 6G --label ui-browser-tests --max-seconds 1800 \
      --env PATH=$HOME/.local/share/vite-plus/bin:/usr/bin:/bin --env HOME=$HOME \
      -- .venv/bin/python tests/browser/run_browser_tests.py

Starts the FastAPI workbench on 127.0.0.1:<port> with a random per-run token (passed to the
tests via env, never written to logs), waits for /health, runs `playwright test` (chromium,
headless, software GL), copies the video + JSON report into artifacts/, and always stops the
backend afterwards.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UI = REPO / "ui"
CACHE = REPO / ".cache"
PORT = int(os.environ.get("RRP_BROWSER_PORT", "8791"))
BACKEND = f"""
import sys, uvicorn
sys.path.insert(0, {str(REPO / 'src')!r})
from rrp.service.app import create_app, MAX_BODY
app = create_app(token=sys.argv[1], port={PORT})
uvicorn.run(app, host="127.0.0.1", port={PORT}, log_level="warning", ws_max_size=MAX_BODY)
"""


def wait_health(url: str, timeout: float = 90.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise SystemExit(f"backend did not become healthy at {url}")


def main() -> int:
    if not (UI / "dist" / "index.html").exists():
        raise SystemExit("ui/dist missing: run `npm run build` in ui/ (inside a lease) first")
    token = secrets.token_urlsafe(24)
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")          # software rendering for /frame (no GPU inside the lease)
    env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
    out_dir = CACHE / "browser-run"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    blog = open(out_dir / "backend.log", "w")
    py = str(REPO / ".venv" / "bin" / "python")
    backend = subprocess.Popen([py, "-c", BACKEND, token], cwd=REPO, env=env, stdout=blog, stderr=subprocess.STDOUT,
                               start_new_session=True)
    rc = 1
    try:
        wait_health(f"http://127.0.0.1:{PORT}/health")
        penv = dict(env)
        penv.update({
            "RRP_BASE_URL": f"http://127.0.0.1:{PORT}", "RRP_TOKEN": token,
            "PLAYWRIGHT_BROWSERS_PATH": str(CACHE / "ms-playwright"), "npm_config_cache": str(CACHE / "npm"),
            "NODE_PATH": str(UI / "node_modules"), "RRP_PW_OUT": str(out_dir), "CI": "1",
        })
        args = ["npx", "playwright", "test", "-c", str(UI / "playwright.config.ts"), *sys.argv[1:]]
        rc = subprocess.call(args, cwd=UI, env=penv)
    finally:
        try:
            os.killpg(backend.pid, signal.SIGTERM)
            backend.wait(timeout=15)
        except Exception:  # noqa: BLE001
            os.killpg(backend.pid, signal.SIGKILL)
        blog.close()
    # collect artifacts
    rec = REPO / "artifacts" / "receipts" / "browser"
    rec.mkdir(parents=True, exist_ok=True)
    rep = out_dir / "report.json"
    if rep.exists():
        shutil.copy(rep, rec / "playwright-report.json")
        stats = json.loads(rep.read_text()).get("stats", {})
        (rec / "summary.json").write_text(json.dumps({"returncode": rc, "stats": stats,
                                                      "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    shutil.copy(out_dir / "backend.log", rec / "backend.log")
    videos = sorted((out_dir / "test-results").rglob("*.webm"), key=lambda p: p.stat().st_size, reverse=True)
    if videos:
        vdir = REPO / "artifacts" / "video"
        vdir.mkdir(parents=True, exist_ok=True)
        demo = next((v for v in videos if "demo" in str(v.parent)), videos[0])
        shutil.copy(demo, vdir / "workbench-demo.webm")
        print(f"video: {vdir / 'workbench-demo.webm'} (from {demo.relative_to(out_dir)})")
    print(f"playwright rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
