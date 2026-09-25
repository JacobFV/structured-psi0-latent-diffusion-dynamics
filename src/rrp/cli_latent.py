"""CLI for the controller-facing semantic latent path (R38)."""
from __future__ import annotations

import json
from pathlib import Path


def cmd_rep(a):
    from rrp.learning.latent_train import train_representation
    cfg = json.loads(open(a.config).read())
    d = Path(cfg["out_dir"]); d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(train_representation(cfg, d), indent=1, default=str))


def register(sub):
    p = sub.add_parser("latent", help="controller-facing semantic latent (R38)").add_subparsers(dest="lat_cmd", required=True)
    r = p.add_parser("train-representation")
    r.add_argument("--config", required=True)
    r.set_defaults(fn=cmd_rep)
