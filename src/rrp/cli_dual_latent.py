"""CLI: dual-arm (multi-assembly) latent packet — packing and closed-loop evaluation."""
from __future__ import annotations

import json
from pathlib import Path


def cmd_pack(a):
    from rrp.learning.packed import pack_dataset
    from rrp.learning.dual_latent import concat_packed
    cfg = json.loads(open(a.config).read())
    out = Path(cfg["out_dir"])
    parts = []
    for k, src in enumerate(cfg["sources"]):
        d = out.parent / f"{out.name}.part{k}"
        if not (d / "meta.json").exists():
            meta = pack_dataset(Path(src["dataset"]), d, set(src["robots"]), cfg["horizon"], stride=1,
                                statuses=tuple(src.get("statuses", ["success"])),
                                include_dart_failures=src.get("include_dart_failures", False),
                                seeds=tuple(src["seeds"]) if src.get("seeds") else None,
                                limits=cfg["limits"], multi_m=cfg["multi_m"])
            print(k, json.dumps({kk: meta[kk] for kk in ("n", "robots")}), flush=True)
        parts.append(d)
    meta = concat_packed(parts, out)
    meta["config"] = cfg
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if not a.keep_parts:
        import shutil
        for d in parts:
            shutil.rmtree(d)
    print(json.dumps({k: meta[k] for k in ("n", "robots", "robot_ids")}))


def cmd_eval(a):
    import torch
    from rrp.evaluation.dual_latent_eval import DualLatentPolicy, evaluate_dual_latent
    from rrp.evaluation.statistics import wilson
    from rrp.learning.checkpoint import load_checkpoint
    from rrp.learning.latent_train import load_representation
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    if dev == "cuda":
        from rrp.ops.gpu import apply_cap
        apply_cap()
    pol = DualLatentPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=a.nfe)
    rep = load_checkpoint(a.checkpoint, map_location="cpu")["config"]["representation"]
    _, _, R, P, _ = load_representation(Path(rep), dev)
    if a.probe:                          # post-hoc measurement probe (identical procedure for sem / nosem)
        from rrp.model.latent_probes import PacketProbe
        st = torch.load(a.probe, map_location=dev, weights_only=False)
        P = PacketProbe(**st["cfg"]).to(dev).eval()
        P.load_state_dict(st["state"])
    summ = {}
    for pair in a.pairs.split(","):
        res = evaluate_dual_latent(pol, R, P, a.task, pair, list(range(a.seed_start, a.seed_start + a.episodes)),
                                   method=a.method, batch=a.batch, out_path=Path(a.out), device=dev,
                                   replan_ticks=a.replan, max_steps=a.max_steps, packet_edit=a.packet_edit)
        att = [r for r in res if r.outcome != "infeasible"]
        k = sum(r.privileged_success for r in att)
        def agg(key):
            tot = {}
            for r in att:
                for q, (x, n) in getattr(r, key).items():
                    s_, n_ = tot.get(q, (0, 0))
                    tot[q] = (s_ + x, n_ + n)
            return {q: (x / n if n else None) for q, (x, n) in tot.items()}
        ev_done = {}
        for r in att:
            for e, st_ in r.events.items():
                ev_done.setdefault(e, {}).setdefault(st_, 0)
                ev_done[e][st_] += 1
        summ[pair] = dict(task=a.task, source=f"learned:{a.checkpoint}", packet_edit=a.packet_edit, attempted=len(att), successes=k,
                          public_successes=sum(r.public_success for r in att), wilson95=wilson(k, len(att)),
                          outcomes={o: sum(r.outcome == o for r in res) for o in {r.outcome for r in res}},
                          event_final_status=ev_done, system_i_calls=sum(r.system_i_calls for r in att),
                          system0_ticks=sum(r.system0_ticks for r in att),
                          packet_rejections=sum(r.packet_rejections for r in att),
                          free_sample_packet_probes=agg("probe_counts"),
                          free_sample_packet_probes_slotswap=agg("probe_counts_slotswap"),
                          probe=a.probe or "representation (jointly trained)")
        print(pair, json.dumps({kk: summ[pair][kk] for kk in ("attempted", "successes", "outcomes")}), flush=True)
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))


def cmd_teacher_ref(a):
    from rrp.evaluation.dual_latent_eval import teacher_reference
    rows = []
    for pair in a.pairs.split(","):
        rows += teacher_reference(a.task, pair, list(range(a.seed_start, a.seed_start + a.episodes)), a.max_steps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
    for pair in a.pairs.split(","):
        rr = [r for r in rows if r["pair"] == pair]
        print(pair, {s: sum(r["status"] == s for r in rr) for s in ("success", "failure", "infeasible", "error")})


def register(p):
    k = p.add_parser("pack-dual", help="pack dual-arm datasets with multi-assembly arrays and concatenate")
    k.add_argument("--config", required=True)
    k.add_argument("--keep-parts", action="store_true")
    k.set_defaults(fn=cmd_pack)
    e = p.add_parser("evaluate-dual", help="closed-loop dual-arm eval: system i packet -> system 0 per tick")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--task", required=True, choices=["support_insert", "handover"])
    e.add_argument("--pairs", required=True)
    e.add_argument("--episodes", type=int, default=20)
    e.add_argument("--seed-start", type=int, default=3000000)
    e.add_argument("--method", default="latent")
    e.add_argument("--nfe", type=int, default=8)
    e.add_argument("--replan", type=int, default=8)
    e.add_argument("--batch", type=int, default=20)
    e.add_argument("--max-steps", type=int, default=800)
    e.add_argument("--probe")
    e.add_argument("--packet-edit", choices=["swap_slots"], help="causal intervention on the received packet")
    e.add_argument("--cpu", action="store_true")
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_eval)
    t = p.add_parser("teacher-ref-dual", help="scripted_teacher reference on the same dev scenes")
    t.add_argument("--task", required=True)
    t.add_argument("--pairs", required=True)
    t.add_argument("--episodes", type=int, default=20)
    t.add_argument("--seed-start", type=int, default=3000000)
    t.add_argument("--max-steps", type=int, default=800)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_teacher_ref)
