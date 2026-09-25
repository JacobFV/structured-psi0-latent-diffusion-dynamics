"""ML/data CLI commands."""
from __future__ import annotations

import json
from pathlib import Path


def cmd_data_generate(a):
    from rrp.data.generate import generate
    cfg = json.loads(open(a.config).read())
    if a.workers:
        cfg["workers"] = a.workers
    generate(cfg)


def cmd_data_pack(a):
    from rrp.learning.packed import pack_dataset
    cfg = json.loads(open(a.config).read())
    meta = pack_dataset(Path(cfg["dataset"]), Path(a.out), set(cfg["train_robots"]), cfg["horizon"],
                        stride=cfg.get("stride", 1), include_dart_failures=cfg.get("include_dart_failures", False),
                        limit_per_robot=cfg.get("episodes_per_robot"))
    print(json.dumps(meta))


def register(sub):
    d = sub.add_parser("data", help="dataset generation").add_subparsers(dest="data_cmd", required=True)
    g = d.add_parser("generate")
    g.add_argument("--config", required=True)
    g.add_argument("--workers", type=int)
    g.set_defaults(fn=cmd_data_generate)
    pk = d.add_parser("pack", help="pack a training config's data into memory-mapped arrays")
    pk.add_argument("--config", required=True)
    pk.add_argument("--out", required=True)
    pk.set_defaults(fn=cmd_data_pack)
    try:
        from rrp import cli_train
        cli_train.register(sub)
    except ImportError:
        pass
    from rrp import cli_latent
    cli_latent.register(sub)
    try:
        from rrp import cli_adapt
        cli_adapt.register(sub)
    except ImportError:
        pass
