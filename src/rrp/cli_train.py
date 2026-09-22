"""train / evaluate CLI."""
from __future__ import annotations

import json
from pathlib import Path


def _run_dir(cfg, name):
    d = Path(cfg.get("out_dir", f"artifacts/runs/{name}"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    return d


def cmd_train_codec(a):
    from rrp.learning.behavior import train_codec
    cfg = json.loads(open(a.config).read())
    print(json.dumps(train_codec(cfg, _run_dir(cfg, cfg["name"])), indent=1, default=str))


def cmd_train_policy(a):
    from rrp.learning.behavior import train_policy
    cfg = json.loads(open(a.config).read())
    if a.seed is not None:
        cfg["seed"] = a.seed
        cfg["out_dir"] = cfg.get("out_dir", f"artifacts/runs/{cfg['name']}") + f"_seed{a.seed}"
    print(json.dumps(train_policy(cfg, _run_dir(cfg, cfg["name"])), indent=1, default=str))


def cmd_evaluate(a):
    import torch
    from rrp.policy.runner import LearnedPolicy
    from rrp.evaluation.runner import evaluate, summarize
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from rrp.ops.gpu import apply_cap
        apply_cap()
    pol = LearnedPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=a.nfe, execute_prefix=a.prefix)
    seeds = list(range(a.seed_start, a.seed_start + a.episodes))
    out = {}
    for robot in a.robots.split(","):
        res = evaluate(pol, robot, seeds, method=a.method, checkpoint=a.checkpoint, max_steps=a.max_steps,
                       batch=a.batch, out_path=Path(a.out))
        out[robot] = summarize(res)
        print(robot, json.dumps(out[robot]), flush=True)
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(out, indent=1))


def register(sub):
    t = sub.add_parser("train", help="training").add_subparsers(dest="train_cmd", required=True)
    c = t.add_parser("codec")
    c.add_argument("--config", required=True)
    c.set_defaults(fn=cmd_train_codec)
    p = t.add_parser("policy")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.set_defaults(fn=cmd_train_policy)
    e = sub.add_parser("evaluate", help="closed-loop evaluation")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--robots", required=True)
    e.add_argument("--episodes", type=int, default=20)
    e.add_argument("--seed-start", type=int, default=3000000)
    e.add_argument("--method", default="policy")
    e.add_argument("--max-steps", type=int, default=300)
    e.add_argument("--nfe", type=int, default=8)
    e.add_argument("--prefix", type=int, default=8)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_evaluate)
    register_campaign(sub)


def cmd_campaign_cell(a):
    from rrp.evaluation.campaign import run_cell
    protocol = json.loads(open(a.protocol).read())
    if torch_cuda():
        from rrp.ops.gpu import apply_cap
        apply_cap()
    print(json.dumps(run_cell(protocol, a.method, a.seed), indent=1))


def cmd_latency(a):
    from rrp.evaluation.latency import run_latency_suite
    cks = dict(kv.split("=", 1) for kv in a.models)
    run_latency_suite(cks, Path(a.out))


def torch_cuda():
    import torch
    return torch.cuda.is_available()


def register_campaign(sub):
    c = sub.add_parser("campaign", help="resumable campaign cells").add_subparsers(dest="camp_cmd", required=True)
    r = c.add_parser("cell")
    r.add_argument("--protocol", required=True)
    r.add_argument("--method", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.set_defaults(fn=cmd_campaign_cell)
    l = sub.add_parser("latency", help="synchronized latency suite")
    l.add_argument("--models", nargs="+", required=True, help="name=checkpoint")
    l.add_argument("--out", required=True)
    l.set_defaults(fn=cmd_latency)
