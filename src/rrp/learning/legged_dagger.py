"""Stateless-expert DAgger, offline gate and system-0 refit for the legged latent route (lessons D-050..D-056).

collect  roll out a route (bc | oracle_bc = E(BC chunk) -> system 0 | generated = flow -> system 0) on seeds and
         record, at every native tick, the deployed features and the STATELESS BC expert's chunk at that state
         (label a = chunk[:, 0]; beh = the 40-tick chunk, i.e. the oracle packet source E(ctx, beh)). Shards use the
         teacher-shard format (+ `beh`), so LeggedData loads them.
gate     offline, on recorded states: system-0 error vs hold-still with the oracle packet (and generated packets if a
         flow is given); shortcut measures: qd zeroed, z zeroed, z shuffled across states, and the step gain
         <pred - hold, label - hold> / |label - hold|^2.
refit    fine-tune system 0 (E and P frozen: same latent space) on original data + DAgger buffers, with qd dropout.

usage: python -m rrp.learning.legged_dagger collect --route oracle_bc --rep R.pt --bc BC.pt --body go2 --seeds 20000-20019 --out DIR
       python -m rrp.learning.legged_dagger gate --buf DIR --rep R.pt [--realizer RZ.pt] [--flow F.pt] --out G.json
       python -m rrp.learning.legged_dagger refit --config C.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rrp.control.legged_latent import public_context, active_event, MAX_N
from rrp.learning.legged_latent_train import LeggedData, load_rep, rep_step, _dev, _save, MAX_J, H


class Recorder:
    def __init__(self, bc, dev, nfe=8, seed=0):
        self.bc, self.dev, self.nfe = bc, dev, nfe
        self.gen = torch.Generator(device=dev).manual_seed(seed + 991)
        self.reset()

    def reset(self):
        self.rows = {k: [] for k in ("q", "qd", "imu", "touch", "osc", "a", "beh", "ctx", "ev", "pose", "contact")}
        self.ctx = None

    def __call__(self, ad, fc):
        b = ad.s.binding
        bb = ad.dyn_batch()
        if self.ctx is None or ad.ticks % 5 == 0:          # step-start context, as in the teacher shards
            self.ctx = public_context(ad.s, ad.osc())
        bb["ctx"] = torch.from_numpy(self.ctx)[None].to(self.dev)
        ch = self.bc.sample(bb, nfe=self.nfe, generator=self.gen)[0, :b.n].cpu().numpy()
        N = ad.m.N
        self.rows["q"].append(bb["q"][0, :N].cpu().numpy()); self.rows["qd"].append(bb["qd"][0, :N].cpu().numpy())
        self.rows["imu"].append(bb["imu"][0].cpu().numpy()); self.rows["touch"].append(bb["asm_touch"][0, :ad.m.nf].cpu().numpy())
        self.rows["osc"].append(float(bb["osc"][0])); self.rows["a"].append(ch[:, 0].astype(np.float32))
        self.rows["beh"].append(ch.astype(np.float16)); self.rows["ctx"].append(self.ctx.copy())
        self.rows["ev"].append(active_event(ad.s.runtime)); self.rows["pose"].append(ad.s.base_pose_truth().astype(np.float32))
        self.rows["contact"].append(np.asarray(fc, bool))


def collect(a):
    from rrp.evaluation.legged_latent_eval import (BCController, LatentLeggedController, run_episode, _seeds)
    from rrp.learning.legged_bc import load_bc
    dev = torch.device("cpu")
    torch.set_num_threads(1)
    bc, _ = load_bc(a.bc, dev)
    out = Path(a.out) / a.body
    out.mkdir(parents=True, exist_ok=True)
    seeds = _seeds(a.seeds)
    f = out / f"s{seeds[0]}-{seeds[-1]}.npz"
    eps, metas, morph = [], [], None
    for sd in seeds:
        if a.route == "bc":
            ctl = BCController(Path(a.bc), dev, nfe=a.nfe, seed=sd)
        else:
            ctl = LatentLeggedController(Path(a.flow) if a.route == "generated" else None, dev, nfe=a.nfe, seed=sd,
                                         rep=a.rep, realizer=a.realizer)
            if a.route == "oracle_bc":
                ctl.bc, ctl.bc_version = bc, Path(a.bc).name
        rec = Recorder(bc, dev, a.nfe, sd)
        ctl.recorder = rec
        row, _ = run_episode(ctl, a.body, sd, a.max_s)
        arr = {k: np.asarray(v) for k, v in rec.rows.items()}
        if len(arr["a"]) < 25:
            continue
        eps.append(arr)
        metas.append(dict(seed=sd, status="success" if row["success"] else ("fell" if row["fell"] else "failure"),
                          waypoints=row["waypoints"], route=a.route, route_source=row["source"],
                          label_source=f"learned:{Path(a.bc).parent.name}/{Path(a.bc).name} (stateless BC expert)",
                          failure_stage=row["failure_stage"], ticks=len(arr["a"])))
        print(json.dumps(dict(seed=sd, route=a.route, success=row["success"], stage=row["failure_stage"],
                              ticks=len(arr["a"]))), flush=True)
        if morph is None:
            from rrp.control.legged_latent import LeggedMorph
            from rrp.sim.legged import LeggedSession, build_waypoint_contact
            sc = build_waypoint_contact(a.body, sd)
            s = LeggedSession(sc, tracker_kind="cpg", seed=sd) if a.body not in ("go2", "t1", "g1") else \
                LeggedSession(sc, seed=sd)
            morph = LeggedMorph(s.model, s.binding, sc.robots[0].robot_spec.spec_hash)
    cat = {k: np.concatenate([e[k] for e in eps]) for k in eps[0]}
    cat["ep"] = np.concatenate([np.full(len(e["a"]), i) for i, e in enumerate(eps)])
    np.savez_compressed(f, **cat, node_static=morph.node_static, node_asm=morph.node_asm, asm_static=morph.asm_static,
                        q0_all=morph.q0_all, n_policy=morph.n_policy, nf=morph.nf)
    f.with_suffix(".json").write_text(json.dumps(dict(body=a.body, episodes=metas, route=a.route), indent=1))
    ok = sum(m["status"] == "success" for m in metas)
    print(json.dumps(dict(out=str(f), n=len(metas), success=ok, ticks=int(len(cat["a"])))))


# ------------------------------------------------------------------ gate
@torch.no_grad()
def gate_metrics(E, R, data, n_batches=20, B=256, seed=3, F_=None, realizer_label="rep", nfe=8):
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=data.dev).manual_seed(seed)
    acc = {k: [0.0, 0.0] for k in ("R_oracle", "hold", "zero_action", "R_oracle_qd0", "R_z0", "R_zshuf", "R_gen",
                                   "gain_num", "gain_den", "gain_qd0_num", "gain_z0_num", "gain_zshuf_num", "zgap")}
    acc_j0 = {k: [0.0, 0.0] for k in ("R_oracle", "hold")}
    for _ in range(n_batches):
        i = data.sample(B, rng)
        j = torch.from_numpy(rng.integers(0, 20, B)).to(data.dev)
        b = data.ctx_batch(i)
        z, _ = E(b, data.beh(i))
        br, ph, a1, am = data.realizer_batch(i, j)
        tj = torch.minimum(i + j, data.A["ep_end"][i])
        hold = data.hold_still(tj)
        m = am.float()
        preds = dict(R_oracle=R(z, br, ph))
        b0 = dict(br); b0["qd"] = torch.zeros_like(br["qd"])
        preds["R_oracle_qd0"] = R(z, b0, ph)
        preds["R_z0"] = R(torch.zeros_like(z), br, ph)
        preds["R_zshuf"] = R(z[torch.randperm(B, device=z.device)], br, ph)
        if F_ is not None:
            zg = F_.sample(b, nfe=nfe, generator=g)
            preds["R_gen"] = R(zg, br, ph)
            am_ = b["asm_mask"][:, None, :, None].float()
            acc["zgap"][0] += float(((zg - z) ** 2 * am_).sum()); acc["zgap"][1] += float((z ** 2 * am_).sum())
        for k, p in preds.items():
            acc[k][0] += float((((p - a1) ** 2) * m).sum()); acc[k][1] += float(m.sum())
        acc["hold"][0] += float((((hold - a1) ** 2) * m).sum()); acc["hold"][1] += float(m.sum())
        acc["zero_action"][0] += float(((a1 ** 2) * m).sum()); acc["zero_action"][1] += float(m.sum())
        dl = (a1 - hold) * m
        den = float((dl ** 2).sum())
        for k, pk in (("gain_num", "R_oracle"), ("gain_qd0_num", "R_oracle_qd0"), ("gain_z0_num", "R_z0"),
                      ("gain_zshuf_num", "R_zshuf")):
            acc[k][0] += float(((preds[pk] - hold) * dl).sum()); acc[k][1] += den
        j0 = (j == 0)
        if j0.any():
            mm = m[j0]
            acc_j0["R_oracle"][0] += float((((preds["R_oracle"][j0] - a1[j0]) ** 2) * mm).sum()); acc_j0["R_oracle"][1] += float(mm.sum())
            acc_j0["hold"][0] += float((((hold[j0] - a1[j0]) ** 2) * mm).sum()); acc_j0["hold"][1] += float(mm.sum())
    r = {k: (v[0] / v[1] if v[1] else None) for k, v in acc.items()}
    h = r["hold"]
    out = dict(mse={k: r[k] for k in ("R_oracle", "hold", "zero_action", "R_oracle_qd0", "R_z0", "R_zshuf", "R_gen")},
               ratio_to_hold={k: (r[k] / h if r[k] is not None else None) for k in
                              ("R_oracle", "R_oracle_qd0", "R_z0", "R_zshuf", "R_gen", "zero_action")},
               step_gain=dict(normal=r["gain_num"], qd_zeroed=r["gain_qd0_num"], z_zeroed=r["gain_z0_num"],
                              z_shuffled=r["gain_zshuf_num"]),
               j0_ratio_to_hold=(acc_j0["R_oracle"][0] / max(acc_j0["hold"][0], 1e-9)),
               packet_gap_rel=(r["zgap"] ** 0.5 if r["zgap"] is not None else None))
    out["gate_pass_R_oracle_le_0.2_hold"] = bool(out["ratio_to_hold"]["R_oracle"] <= 0.2)
    return out


def gate(a):
    dev = _dev()
    rcfg, E, R, P, rres = load_rep(Path(a.rep), dev)
    if a.realizer:
        R.load_state_dict(torch.load(a.realizer, map_location=dev, weights_only=False)["R"])
    F_ = None
    if a.flow:
        from rrp.model.legged_latent import LeggedFlow
        st = torch.load(a.flow, map_location=dev, weights_only=False)
        F_ = LeggedFlow(dz=rcfg["latent"]["dz"], D=st["cfg"].get("width", 256), layers=st["cfg"].get("layers", 4)).to(dev)
        F_.load_state_dict(st["flow"]); F_.eval()
    res = dict(rep=a.rep, realizer=a.realizer, flow=a.flow, label_source="stateless BC expert chunk (see buffer json)")
    for name, root in (("bc_visited", a.buf), ("teacher_heldout", a.teacher_data)):
        if not root:
            continue
        data = LeggedData(Path(root), [a.body], dev)
        if name == "teacher_heldout":
            data.train_idx = data.test_idx
        res[name] = gate_metrics(E, R, data, F_=F_)
        print(name, json.dumps(res[name]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ refit
def refit(cfg, out: Path):
    dev = _dev()
    rcfg, E, R, P, rres = load_rep(Path(cfg["representation"]), dev)
    if cfg.get("init_realizer"):
        R.load_state_dict(torch.load(cfg["init_realizer"], map_location=dev, weights_only=False)["R"])
    R.train()
    for p in R.parameters():
        p.requires_grad_(True)
    base = LeggedData(Path(rcfg["data"]), cfg.get("bodies", rcfg["bodies"]), dev)
    dag = [LeggedData(Path(r), cfg.get("bodies", rcfg["bodies"]), dev) for r in cfg["dagger"]]
    steps, lr = cfg["steps"], cfg.get("lr", 3e-4)
    opt = torch.optim.AdamW(R.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "train_log.jsonl", "w")
    B, fr = cfg.get("batch_size", 256), cfg.get("dagger_frac", 0.5)
    qd_drop = cfg.get("qd_dropout", 0.5)
    t0 = time.time()
    for step in range(1, steps + 1):
        loss = 0.0
        parts = [(base, int(B * (1 - fr)), MAX_J)] + [(d, int(B * fr / len(dag)), 19) for d in dag]
        logs = {}
        for k, (d, nb, mj) in enumerate(parts):
            if nb == 0:
                continue
            i = d.sample(nb, rng)
            j = torch.from_numpy(rng.integers(0, mj + 1, nb)).to(dev)
            l, lg, _ = rep_step(E, R, P, d, i, j, 0.0, 0.0, train=True, qd_drop=qd_drop)
            loss = loss + l * nb / B
            logs[f"real_{k}"] = lg["real"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 200 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=float(loss.detach()), **logs)) + "\n"); log.flush()
        if step % 1000 == 0 or step == steps:
            _save(out / "realizer.pt", R=R.state_dict(), cfg=cfg, step=step, representation=cfg["representation"])
    R.eval()
    res = dict(steps=steps, wall_s=time.time() - t0)
    for k, d in enumerate(dag):
        res[f"gate_dagger{k}"] = gate_metrics(E, R, d)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "gate", "refit"])
    ap.add_argument("--route", default="oracle_bc", choices=["bc", "oracle_bc", "generated"])
    ap.add_argument("--rep"); ap.add_argument("--bc"); ap.add_argument("--flow"); ap.add_argument("--realizer")
    ap.add_argument("--body", default="go2"); ap.add_argument("--seeds")
    ap.add_argument("--nfe", type=int, default=8); ap.add_argument("--max-s", type=float, default=40.0)
    ap.add_argument("--buf"); ap.add_argument("--teacher-data", default=None)
    ap.add_argument("--config"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.stage == "collect":
        collect(a)
    elif a.stage == "gate":
        gate(a)
    else:
        print(json.dumps(refit(json.loads(Path(a.config).read_text()), Path(a.out)), indent=1)[:3000])


if __name__ == "__main__":
    main()
