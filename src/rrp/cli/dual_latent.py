"""CLI: dual-arm (multi-assembly) latent packet — packing and closed-loop evaluation."""
from __future__ import annotations

import json
from pathlib import Path


def cmd_pack(a):
    from rrp.harness.data.dual_latent import pack_dual
    cfg = json.loads(open(a.config).read())
    meta = pack_dual(cfg, keep_parts=a.keep_parts, log=lambda m: print(m, flush=True))
    print(json.dumps({k: meta[k] for k in ("n", "robots", "robot_ids")}))


def cmd_eval(a):
    import torch
    from rrp.harness.eval.hooks import dual_latent_hooks
    from rrp.harness.eval.latent_eval import probe_rates
    from rrp.harness.eval.evaluate import evaluate
    from rrp.policies.latent import DualLatentPolicy, LatentStackPolicy
    from rrp.harness.eval.statistics import wilson
    from rrp.policies.nets.checkpoint import load_checkpoint
    from rrp.policies.bundles import load_representation
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    if dev == "cuda":
        from rrp.ops.workload import apply_cap
        apply_cap()
    pol = DualLatentPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=a.nfe)
    rep = load_checkpoint(a.checkpoint, map_location="cpu")["config"]["representation"]
    _, _, R, P, _ = load_representation(Path(rep), dev)
    if a.probe:                          # post-hoc measurement probe (identical procedure for sem / nosem)
        from rrp.policies.nets.latent_probes import PacketProbe
        st = torch.load(a.probe, map_location=dev, weights_only=False)
        P = PacketProbe(**st["cfg"]).to(dev).eval()
        P.load_state_dict(st["state"])
    summ = {}
    for pair in a.pairs.split(","):
        stack = LatentStackPolicy(pol, R, replan_ticks=a.replan, device=dev, name=a.method,
                                  version=f"learned:{a.checkpoint}")
        label = f"learned:{pol.name}" + (f"+edit:{a.packet_edit}" if a.packet_edit else "")
        res = evaluate(stack, "mujoco/dual", a.task, pair, list(range(a.seed_start, a.seed_start + a.episodes)),
                       batch=a.batch, max_steps=a.max_steps, out=Path(a.out),
                       hooks=dual_latent_hooks(stack, a.task, P, device=dev, packet_edit=a.packet_edit),
                       row_extra=dict(method=a.method, checkpoint=a.checkpoint, pair=pair, packet_edit=a.packet_edit,
                                      controller_source=label))
        att = [e for e in res if e.outcome != "infeasible"]
        k = sum(bool(e.success_privileged) for e in att)
        ev_done = {}
        for e in att:
            for ev, st_ in e.metrics["events"].items():
                ev_done.setdefault(ev, {}).setdefault(st_, 0)
                ev_done[ev][st_] += 1
        summ[pair] = dict(task=a.task, source=f"learned:{a.checkpoint}", packet_edit=a.packet_edit, attempted=len(att),
                          successes=k, public_successes=sum(bool(e.success_public) for e in att),
                          wilson95=wilson(k, len(att)),
                          outcomes={o: sum(e.outcome == o for e in res) for o in {e.outcome for e in res}},
                          event_final_status=ev_done, system_i_calls=sum(e.metrics["packets"] for e in att),
                          system0_ticks=sum(e.metrics["system0_ticks"] for e in att),
                          packet_rejections=sum(e.metrics["packet_rejections"] for e in att),
                          free_sample_packet_probes=probe_rates(att),
                          free_sample_packet_probes_slotswap=probe_rates(att, "probe_counts_slotswap"),
                          probe=a.probe or "representation (jointly trained)")
        print(pair, json.dumps({kk: summ[pair][kk] for kk in ("attempted", "successes", "outcomes")}), flush=True)
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))


def cmd_teacher_ref(a):
    from rrp.harness.eval.dual_latent_eval import teacher_reference
    rows = []
    for pair in a.pairs.split(","):
        rows += teacher_reference(a.task, pair, list(range(a.seed_start, a.seed_start + a.episodes)), a.max_steps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
    for pair in a.pairs.split(","):
        rr = [r for r in rows if r["pair"] == pair]
        print(pair, {s: sum(r["status"] == s for r in rr) for s in ("success", "failure", "infeasible", "error")})


def cmd_pair_index(a):
    from rrp.harness.data.dual_pairs import build_pair_index
    idx = build_pair_index(Path(a.dataset), Path(a.packed) if a.packed else None, check_scene=not a.no_scene_check)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(idx, indent=1))
    print(json.dumps(dict(summary=idx["summary"], by_group=idx["by_group"]), indent=1))


def register(p):
    pi = p.add_parser("pair-index-dual", help="pair index for the manipulator-assignment family")
    pi.add_argument("--dataset", required=True)
    pi.add_argument("--packed")
    pi.add_argument("--no-scene-check", action="store_true")
    pi.add_argument("--out", required=True)
    pi.set_defaults(fn=cmd_pair_index)
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
