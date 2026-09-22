"""`rrp adapt grpo|expo|synthetic --config ...` (online adaptation; GPU work goes to the peer via the broker)."""
from __future__ import annotations

import json
from pathlib import Path


def _cfg(a, method):
    cfg = json.loads(open(a.config).read()) if a.config else {}
    if method == "grpo":
        cfg["method"] = "grpo_shared_prefix" if (a.shared_prefix or cfg.get("method") == "grpo_shared_prefix") \
            else "grpo"
    else:
        cfg["method"] = method
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    cfg.setdefault("out_dir", f"artifacts/runs/{cfg.get('name', 'adapt')}_{cfg['method']}")
    return cfg


def cmd_adapt(a):
    if a.method == "synthetic":
        from rrp.learning.synthetic import run_synthetic
        cfg = json.loads(open(a.config).read()) if a.config else {}
        res = run_synthetic(**cfg.get("synthetic", {}))
        if a.out_dir:
            Path(a.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(a.out_dir) / "result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    from rrp.learning.adapt import run
    print(json.dumps(run(_cfg(a, a.method)), indent=1, default=str)[:4000])


def register(sub):
    p = sub.add_parser("adapt", help="online adaptation (flow-SDE GRPO / EXPO-FT-inspired)")
    p.add_argument("method", choices=["grpo", "expo", "synthetic"])
    p.add_argument("--config")
    p.add_argument("--shared-prefix", action="store_true", help="GRPO with event-boundary shared prefix")
    p.add_argument("--out-dir")
    p.set_defaults(fn=cmd_adapt)
