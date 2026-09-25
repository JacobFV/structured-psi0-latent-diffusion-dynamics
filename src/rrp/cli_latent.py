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
    register_more(p)
    register_probe_cmd(p)
    register_cell(p)
    register_latency(p)
    register_counterfactuals(p)


def cmd_flow(a):
    from rrp.learning.latent_train import train_latent_flow
    cfg = json.loads(open(a.config).read())
    d = Path(cfg["out_dir"]); d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(train_latent_flow(cfg, d), indent=1, default=str))


def _load(a):
    import torch
    from rrp.policy.latent_runner import LatentPolicy
    from rrp.learning.latent_train import load_representation
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from rrp.ops.gpu import apply_cap
        apply_cap()
    pol = LatentPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=a.nfe)
    from rrp.learning.checkpoint import load_checkpoint
    rep = load_checkpoint(a.checkpoint, map_location="cpu")["config"]["representation"]
    lcfg, E, R, P, _ = load_representation(Path(rep), dev)
    return pol, R, P, dev


def cmd_eval(a):
    from rrp.evaluation.latent_eval import evaluate_latent
    from rrp.evaluation.statistics import wilson
    pol, R, P, dev = _load(a)
    summ = {}
    for robot in a.robots.split(","):
        res = evaluate_latent(pol, R, P, robot, list(range(a.seed_start, a.seed_start + a.episodes)), method=a.method,
                              batch=a.batch, out_path=Path(a.out), device=dev, replan_ticks=a.replan)
        att = [r for r in res if r.outcome != "infeasible"]
        k = sum(r.privileged_success for r in att)
        probes = {}
        for r in att:
            for q, (x, n) in r.probe_counts.items():
                s_, n_ = probes.get(q, (0, 0)); probes[q] = (s_ + x, n_ + n)
        summ[robot] = dict(attempted=len(att), successes=k, wilson95=wilson(k, len(att)),
                           outcomes={o: sum(r.outcome == o for r in res) for o in {r.outcome for r in res}},
                           system_i_calls=sum(r.system_i_calls for r in att), system0_ticks=sum(r.system0_ticks for r in att),
                           free_sample_packet_probes={q: (x / n if n else None) for q, (x, n) in probes.items()})
        print(robot, json.dumps(summ[robot]), flush=True)
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))


def cmd_disturb(a):
    from rrp.evaluation.latent_eval import disturbance_test
    pol, R, P, dev = _load(a)
    rows = disturbance_test(pol, R, a.robots.split(",")[0], list(range(a.seed_start, a.seed_start + a.episodes)), device=dev)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows))
    for k in ("final_dev_closed_loop_m", "final_dev_open_loop_delta_m", "final_dev_servo_absolute_m"):
        print(k, round(float(sum(r[k] for r in rows) / len(rows)), 4))


def register_more(p):
    f = p.add_parser("train-flow")
    f.add_argument("--config", required=True)
    f.set_defaults(fn=cmd_flow)
    for name, fn in (("evaluate", cmd_eval), ("disturbance", cmd_disturb)):
        e = p.add_parser(name)
        e.add_argument("--checkpoint", required=True)
        e.add_argument("--robots", required=True)
        e.add_argument("--episodes", type=int, default=20)
        e.add_argument("--seed-start", type=int, default=3000000)
        e.add_argument("--method", default="latent")
        e.add_argument("--nfe", type=int, default=8)
        e.add_argument("--replan", type=int, default=8)
        e.add_argument("--batch", type=int, default=16)
        e.add_argument("--out", required=True)
        e.set_defaults(fn=fn)


def cmd_fit_probes(a):
    from rrp.learning.latent_train import fit_probes_on_frozen
    res = fit_probes_on_frozen(Path(a.representation), Path(a.packed_dir), Path(a.out), steps=a.steps,
                               metadata_only=a.metadata_only)
    print(json.dumps(res, indent=1))


def register_probe_cmd(p):
    f = p.add_parser("fit-probes", help="measurement probe on frozen detached z (or metadata-only control)")
    f.add_argument("--representation", required=True)
    f.add_argument("--packed-dir", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--steps", type=int, default=6000)
    f.add_argument("--metadata-only", action="store_true")
    f.set_defaults(fn=cmd_fit_probes)


def cmd_cell(a):
    import torch
    from rrp.evaluation.latent_campaign import run_latent_cell
    if torch.cuda.is_available():
        from rrp.ops.gpu import apply_cap
        apply_cap()
    proto = json.loads(open(a.protocol).read())
    print(json.dumps(run_latent_cell(proto, a.method, a.seed, base_flow_config=a.base_flow_config,
                                     target_packed=a.target_packed), indent=1))


def register_cell(p):
    c = p.add_parser("cell", help="resumable latent_slice1 campaign cell")
    c.add_argument("--protocol", default="configs/eval/latent_slice1.json")
    c.add_argument("--method", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--base-flow-config", required=True)
    c.add_argument("--target-packed", default="artifacts/packed/latent_targets")
    c.set_defaults(fn=cmd_cell)


def cmd_latency(a):
    from rrp.evaluation.latency import latent_latency_suite
    print(json.dumps(latent_latency_suite(a.checkpoint, Path(a.out), direct_ckpt=a.direct), indent=1))


def register_latency(p):
    c = p.add_parser("latency")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--direct")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_latency)


def cmd_counterfactuals(a):
    import torch
    from rrp.evaluation.latent_counterfactuals import counterexample, embodiment_swap
    from rrp.learning.latent_train import load_representation
    dev = "cuda" if a.gpu and torch.cuda.is_available() else "cpu"
    _, E, _, P, res = load_representation(Path(a.representation), dev)
    if a.probe:                                   # measurement probe fitted post hoc on frozen z (fair across variants)
        from rrp.model.latent_probes import PacketProbe
        st = torch.load(a.probe, map_location=dev, weights_only=False)
        P = PacketProbe(**st["cfg"]).to(dev).eval()
        P.load_state_dict(st["state"])
    ds = Path(a.dataset)
    out = dict(representation=a.representation, probe=a.probe or "representation (jointly trained)",
               latent_space_version=res["latent_space_version"],
               source="encoded teacher targets (not generated packets)",
               counterexample=counterexample(E, P, ds, robot=a.robot, n=a.n, dev=dev),
               embodiment_swap=embodiment_swap(E, P, ds, arm=a.arm, seeds=range(a.seed_start, a.seed_start + a.n), dev=dev))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "counterexample"} | {
        "counterexample": {k: v for k, v in out["counterexample"].items() if k != "rows"}}, indent=1))


def register_counterfactuals(p):
    c = p.add_parser("counterfactuals", help="section-6 counterexample + compatible embodiment swap on frozen E/P")
    c.add_argument("--representation", required=True)
    c.add_argument("--dataset", default="artifacts/datasets/pick_place_primary_v3dart")
    c.add_argument("--robot", default="panda_pg2")
    c.add_argument("--arm", default="parm6")
    c.add_argument("--n", type=int, default=20)
    c.add_argument("--seed-start", type=int, default=0)
    c.add_argument("--gpu", action="store_true")
    c.add_argument("--probe", help="post-hoc probe .pt (fit-probes output); default: the representation's own P")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_counterfactuals)
