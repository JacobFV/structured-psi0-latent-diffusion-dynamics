"""ML/data CLI commands."""
from __future__ import annotations

import json


def cmd_data_generate(a):
    from rrp.data.generate import generate
    cfg = json.loads(open(a.config).read())
    if a.workers:
        cfg["workers"] = a.workers
    generate(cfg)


def register(sub):
    d = sub.add_parser("data", help="dataset generation").add_subparsers(dest="data_cmd", required=True)
    g = d.add_parser("generate")
    g.add_argument("--config", required=True)
    g.add_argument("--workers", type=int)
    g.set_defaults(fn=cmd_data_generate)
    try:
        from rrp import cli_train
        cli_train.register(sub)
    except ImportError:
        pass
