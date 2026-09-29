"""Closed-loop evaluation of the legged latent-packet policy (system i flow -> LatentActionChunk -> system 0).

Controller sources (labelled in every output row):
  learned:<flow ckpt>      system i samples z from PUBLIC context every 0.4 s; system 0 realizes it every 20 ms
  scripted_teacher         privileged WaypointTeacher -> frozen body tracker (reference only)
Packet interventions (causal edits of the RECEIVED packet, everything else unchanged):
  none | mirror_goal (packet generated from the context with lateral waypoint estimates mirrored)
  | halt (packet generated from the context with the task view set to `halt`) | probe_yaw:+X / probe_yaw:-X
  (gradient edit of z so the frozen probe reads a desired yaw displacement X rad over the horizon)
  | freeze (the first packet is held for the whole episode, validity extended — declared diagnostic)
  | zero (z = 0).  Edits start at t_edit and are applied to every later packet.

The system-0 adapter replaces the body tracker inside LeggedSession (the native joint-target layer is unchanged);
held upper-body actuators remain servoed to the default pose. Truth (base pose, contacts) is logged per tick
for evaluation only.

usage: python -m rrp.harness.eval.legged_latent_eval --flow artifacts/runs/X/policy.pt --bodies go2 --seeds 10000-10019
       --out artifacts/runs/X/eval_dev.jsonl [--edit mirror_goal --t-edit 1.0] [--video-dir artifacts/video --video-n 2]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from rrp.core.action import NativeCommand
from rrp.policies.features.legged import LeggedMorph, KNOT_TIMES
from rrp.policies.bundles import _dev
from rrp.policies.legged import BCAdapter, BCController, LatentLeggedController, OracleShadow, System0Adapter
from rrp.envs.mujoco.legged import LeggedSession, build_waypoint_contact
from rrp.bodies.contact import model_contact_version
from rrp.core.runs import parse_seed_spec

def failure_stage(row):
    """success | fell | stall (base moved < 0.3 m over the episode) | drift_a (never reached waypoint a) |
    drift_b (reached a, not b) | halt (reached b, halt not completed)."""
    if row["success"]:
        return "success"
    if row["fell"]:
        return "fell"
    tr = row.get("trace") or []
    path = sum(math.hypot(b["pose"][0] - a["pose"][0], b["pose"][1] - a["pose"][1]) for a, b in zip(tr, tr[1:]))
    ev = row["events"]
    if ev.get("walk_to_a") not in ("succeeded", "completed"):
        return "stall" if path < 0.3 else "drift_a"
    if ev.get("walk_to_b") not in ("succeeded", "completed"):
        return "drift_b"
    return "halt"


from rrp.envs.mujoco.perturb import PhysicsPerturbation as _PP
_NOMINAL = _PP()

LEARNED_TRACKER_BODIES = ("go2", "t1", "g1", "h1", "anymal_c")   # frozen learned trackers; procedural bodies use CPG


def default_tracker_kind(body):
    return "auto" if body in LEARNED_TRACKER_BODIES else "cpg"


def _limits(model):
    from rrp.bodies.actuator import model_actuator_limits
    return model_actuator_limits(model)


def run_episode(ctl, body, seed, max_s=60.0, video=None, oracle=False, scenario=None, arc_only=False, frame_every=2,
                cam_scale=1.0, size=(368, 480), perturb=None, deploy=None):
    """perturb: rrp.envs.perturb.PhysicsPerturbation (W6 robustness sweeps; None = nominal, unchanged behaviour).
    Every row carries `motion` (rrp.evaluation.motion_quality, read-only recording) and, if perturbed, `perturbation`.
    deploy: rrp.evaluation.deploy_eval.DeployOptions (D-126: estimator, packet OOD, safety, long runs, latency;
    None or the default instance = unchanged behaviour and rows)."""
    if deploy is not None and deploy.is_default():
        deploy = None
    if deploy is not None and deploy.eval_mode == "long":
        max_s = deploy.long_s
    from rrp.envs.mujoco.perturb import apply_model, install_legged
    from rrp.envs.mujoco.motion_quality import LeggedMotionRecorder
    if scenario is not None:
        sc = scenario
    elif perturb is not None and perturb.terrain_amp_m > 0:
        sc = build_waypoint_contact(body, seed, terrain=perturb.terrain(seed))
    else:
        sc = build_waypoint_contact(body, seed)
    s = LeggedSession(sc, tracker_kind=default_tracker_kind(body), seed=seed,
                      **({} if deploy is None else dict(base_state_source=deploy.base_state_source)))
    trk_sha = getattr(s.tracker, "sha256", None)   # the body tracker (drives the teacher route; label source for ours)
    pert_rec = None
    if perturb is not None:
        b_ = s.binding
        pert_rec = apply_model(s.model, perturb, robot_bodies=b_.robot_bodies, com_body=b_.root_bid, act_ids=b_.pol_act)
    from rrp.harness.data.contact_metrics import contact_metrics_enabled
    cfm = contact_metrics_enabled()             # W12 stance-drift keys (RRP_CONTACT_METRICS=1; off = rows unchanged)
    mrec = LeggedMotionRecorder(s, record_stance=cfm)
    pst = install_legged(s, perturb if perturb is not None else _NOMINAL, seed, on_substep=mrec.on_substep,
                         on_tick=mrec.on_tick, on_reset=mrec.on_reset)
    morph = LeggedMorph(s.model, s.binding, sc.robots[0].robot_spec.spec_hash)
    is_bc = isinstance(ctl, BCController)
    ad = (BCAdapter if is_bc else System0Adapter)(ctl, s, morph) if ctl is not None else None
    teacher = None
    if ctl is not None:
        ctl.bind(s, morph)
        if not is_bc:
            ctl.oracle = OracleShadow(ctl, s, s.tracker) if oracle else None
        s.tracker = ad
    dep = _deploy_setup(s, ctl, ad, deploy) if deploy is not None else None
    s.reset()
    if dep is not None and deploy.measure_latency and ad is not None:
        from rrp.harness.eval.deploy_eval import instrument_latency
        dep["timers"] = instrument_latency(ctl, s.tracker)
    if ctl is None:
        from rrp.policies.teachers.legged import WaypointTeacher
        teacher = WaypointTeacher(s, arc_only=arc_only)
    else:
        ad.armed = True
    frames = []
    rend = None
    if video is not None:
        import mujoco
        rend = mujoco.Renderer(s.model, *size)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = s.binding.root_bid
        cam.distance, cam.elevation, cam.azimuth = cam_scale * 3.0 * max(0.5, s.binding.nominal_height() / 0.35) ** 0.5, -25, 135
    zero = NativeCommand(controller_version=s.controller_version(), groups={"base_velocity": [0.0, 0.0, 0.0]},
                         source="learned")
    t0 = time.time()
    steps = 0
    while s.data.time < max_s:
        cmd = teacher.act() if teacher else zero
        s.step(cmd)
        steps += 1
        if dep is not None and dep["rec"] is not None:
            dep["rec"].on_step()
        if rend is not None and steps % frame_every == 0:
            rend.update_scene(s.data, camera=cam)
            st = " ".join(f"{e}:{v.status}" for e, v in s.runtime.instances.items())
            frames.append((rend.render().copy(), f"t={s.data.time:.1f}s {st}"))
        if s.fell or ((s.runtime.succeeded() or (teacher is not None and teacher.done))
                      and not (dep is not None and deploy.eval_mode == "long")):
            break
    ok = bool(s.privileged_success() and not s.fell)
    src = ("scripted_teacher" + (":arc_only" if arc_only else "")) if ctl is None else (
        f"privileged_oracle_packet:{ctl.lsv}" if oracle else (
            f"oracle_diagnostic:E(bc_chunk)|{ctl.policy_version}|bc={ctl.bc_version}" if getattr(ctl, "bc", None) is not None
            else ctl.policy_version))
    row = dict(body=body, seed=seed, source=src,
               edit=(ctl.edit if ctl else "none"), t_edit=(ctl.t_edit if ctl else None), success=ok, fell=bool(s.fell),
               public_success=bool(s.runtime.succeeded()), sim_time=float(s.data.time), wall_s=time.time() - t0,
               events={e: v.status for e, v in s.runtime.instances.items()},
               final_pose=s.base_pose_truth().tolist(), waypoints=sc.meta["waypoints"],
               tracker=getattr(s, "tracker_version_str", None), n_steps=steps,
               contact_version=model_contact_version(s.model) or sc.meta.get("contact_model"),
               tracker_sha256=trk_sha, actuator_limits=_limits(s.model))
    from rrp.core.provenance import Source, parse_legacy_source, stamp_source_label
    lab = parse_legacy_source(src)                  # sl-1 canonical label: a no-op unless RRP_SOURCE_LABELS=canonical
    stamp_source_label(row, lab.kind, f"privileged_packet:{ctl.lsv}" if src.startswith("privileged_oracle_packet")
                       else lab.detail)
    if ctl is not None:
        row.update(stats=ad.stats, packets=ctl.packets, packet_log=ad.log[:20],
                   latent_space_version=getattr(ctl, "lsv", None), realizer_compat_version=getattr(ctl, "rcv", None),
                   checkpoint_provenance=getattr(ctl, "checkpoint_provenance", None))
        row["trace"] = ctl.trace[::5]
        if getattr(ctl, "zero_qd", False):
            row["zero_qd"] = True
    row["failure_stage"] = failure_stage(row)
    ar = s.actuator_record() if hasattr(s, "actuator_record") else None
    if ar is not None:                 # D-126 #14: only non-ideal actuator modes add the key (default rows unchanged)
        row["actuator_mode"] = ar
    if dep is not None:
        row.update(_deploy_row(dep, deploy, ctl, ad))
    row["motion"] = mrec.summary()
    if cfm:
        from rrp.harness.data.contact_metrics import legged_contact_motion
        row["motion"].update(legged_contact_motion(mrec.stance_trace()))
    if perturb is not None:
        row["perturbation"] = dict(perturb.to_dict(), applied=pert_rec, terrain=sc.meta.get("terrain"),
                                   actuator=(pst["actuator"].params if pst["actuator"] is not None else None),
                                   push=(pst["push"].record() if pst["push"] is not None else None))
    return row, frames


def _deploy_setup(s, ctl, ad, deploy) -> dict:
    """D-126: attach the non-default deployment options to a built session (before reset)."""
    from rrp.harness.eval.deploy_eval import LongRunRecorder, SafeTracker, make_safety
    dep = dict(ood=None, safety=None, rec=None, timers=None)
    if deploy.packet_ood != "off" and isinstance(ad, System0Adapter):
        from rrp.policies.packet_ood import PacketOODModel, PacketOODMonitor
        model = PacketOODModel.load(Path(deploy.ood_model), expect_lsv=ctl.lsv)
        if model.n_assemblies is not None and model.feature == "packet" and model.n_assemblies != ad.m.M:
            raise ValueError(f"OOD model for {model.n_assemblies} assemblies, body has {ad.m.M}")
        dep["ood"] = ad.ood = PacketOODMonitor(model, deploy.packet_ood, deploy.ood_fallback)
        ad.fallback_mode = deploy.ood_fallback
    layer = make_safety(s, deploy)
    if layer is not None:
        dep["safety"] = layer
        s.tracker = SafeTracker(s.tracker, s, layer)
        if ad is not None:
            ad.safety = layer
    if deploy.eval_mode == "long" or deploy.base_state_source == "estimator":
        dep["rec"] = LongRunRecorder(s, deploy.window_s)
    if deploy.record_packets and ctl is not None and hasattr(ctl, "record_z"):
        ctl.record_z = []
    return dep


def _deploy_row(dep, deploy, ctl, ad) -> dict:
    out = dict(deploy=deploy.record())
    if dep["ood"] is not None:
        out["packet_ood"] = dep["ood"].summary()
    if dep["safety"] is not None:
        out["safety"] = dep["safety"].summary()
    if dep["rec"] is not None:
        out["long_run" if deploy.eval_mode == "long" else "base_state"] = dep["rec"].summary()
    if dep["timers"] is not None:
        from rrp.harness.eval.deploy_eval import latency_summary
        out["latency"] = latency_summary(dep["timers"])
    if getattr(ctl, "record_z", None) is not None:
        out["_packet_z"] = ctl.record_z                  # popped by main() into <out>.packets/*.npz
    return out


def packet_probe_accuracy(rows):
    """Closed-loop packet probes vs truth (generated packets, privileged truth used only for scoring)."""
    agg = dict(subtask=[0, 0], contact=[0, 0], goal_err=[0.0, 0], disp_yaw_err=[0.0, 0], disp_xy_err=[0.0, 0])
    for r in rows:
        tr = r.get("trace") or []
        if not tr:
            continue
        ts = np.array([x["t"] for x in tr])
        for p in r["packets"]:
            if p["edit"] != "none":
                continue
            agg["subtask"][0] += int(p["probe"]["subtask"] == min(p["ev"], 3)); agg["subtask"][1] += 1
            x, y, yaw = p["pose"]
            wp = r["waypoints"]["a" if p["ev"] == 0 else "b"]
            c, s_ = math.cos(yaw), math.sin(yaw)
            gx, gy = (c * (wp[0] - x) + s_ * (wp[1] - y)) / 2, (-s_ * (wp[0] - x) + c * (wp[1] - y)) / 2
            if p["ev"] < 3:
                agg["goal_err"][0] += 2 * math.hypot(p["probe"]["goal"][0] - gx, p["probe"]["goal"][1] - gy)
                agg["goal_err"][1] += 1
            k2 = int(np.searchsorted(ts, p["t"] + 0.8))
            if k2 < len(tr):
                x2, y2, yaw2 = tr[k2]["pose"]
                ex, ey = (c * (x2 - x) + s_ * (y2 - y)), (-s_ * (x2 - x) + c * (y2 - y))
                dyaw = (yaw2 - yaw + math.pi) % (2 * math.pi) - math.pi
                agg["disp_yaw_err"][0] += abs(p["probe"]["disp"][2] - dyaw); agg["disp_yaw_err"][1] += 1
                agg["disp_xy_err"][0] += math.hypot(p["probe"]["disp"][0] * 0.5 - ex, p["probe"]["disp"][1] * 0.5 - ey)
                agg["disp_xy_err"][1] += 1
            for kk, kt in enumerate(KNOT_TIMES):
                k3 = int(np.searchsorted(ts, p["t"] + kt))
                if k3 < len(tr):
                    for m, c_ in enumerate(tr[k3]["contact"]):
                        agg["contact"][0] += int(p["probe"]["contact"][kk][m] == c_); agg["contact"][1] += 1
    return {k: (v[0] / v[1] if v[1] else None) for k, v in agg.items()}


def _caption(img, lines):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14 * len(lines) + 6], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((6, 3 + 14 * i), t, fill=(255, 255, 255))
    return np.asarray(im)


def save_video(frames, row, video_dir: Path, label: str):
    import imageio
    video_dir.mkdir(parents=True, exist_ok=True)
    tag = "success" if row["success"] else ("fell" if row["fell"] else "failure")
    import re
    src = re.sub(r"[^A-Za-z0-9_.+-]+", "-", row["source"])[:80]
    ed = "" if row["edit"] == "none" else f"_edit-{row['edit'].replace(':', '')}"
    name = f"{dt.date.today()}_legged_{src}{ed}_{row['body']}_waypoint_contact_s{row['seed']}_{tag}.mp4"
    imgs = [_caption(f, [f"{label} | {row['body']} | waypoint_contact | seed {row['seed']}" + (
        f" | EDIT {row['edit']}@{row['t_edit']}s" if row['edit'] != 'none' else ""), st]) for f, st in frames]
    imageio.mimsave(video_dir / name, imgs, fps=25, quality=6)
    with open(video_dir / "INDEX.md", "a") as f:
        f.write(f"- `{name}` — source={row['source']} robot={row['body']} task=waypoint_contact seed={row['seed']} "
                f"edit={row['edit']} outcome={tag} (privileged evaluator)\n")
    return name


def _save_packets(zs, out_dir: Path, row):
    """D-126 #29: one .npz per episode: t [P], edit [P], z [P, K, M, dz] (+ outcome, for OOD scoring)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{row['body']}_s{row['seed']}_{str(row['edit']).replace(':', '')}.npz"
    np.savez_compressed(out_dir / name, t=np.array([x[0] for x in zs]), edit=np.array([x[1] for x in zs]),
                        z=np.stack([x[2] for x in zs]) if zs else np.zeros((0,)), fell=row["fell"],
                        success=row["success"], source=row["source"], lsv=str(row.get("latent_space_version")))
    row["packets_file"] = str(out_dir / name)




def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default=None, help="flow policy.pt; omit for the scripted_teacher reference")
    ap.add_argument("--bodies", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--edit", default="none")
    ap.add_argument("--t-edit", type=float, default=1.0)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--posthoc-probe", default=None)
    ap.add_argument("--oracle", action="store_true", help="DIAGNOSTIC: E-encoded shadow teacher rollouts as packets")
    ap.add_argument("--bc", default=None, help="BC policy.pt: alone = BC route (positive control); with --rep or "
                    "--flow and --oracle-bc = stateless oracle packets E(BC chunk) (DIAGNOSTIC)")
    ap.add_argument("--oracle-bc", action="store_true")
    ap.add_argument("--rep", default=None, help="representation.pt (oracle routes without a flow)")
    ap.add_argument("--realizer", default=None, help="refit system-0 state (R) to use instead of the rep's R")
    ap.add_argument("--zero-qd", action="store_true", help="DIAGNOSTIC: system 0 sees qd = 0")
    ap.add_argument("--replan", type=int, default=5, help="BC replan period (ticks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--video-n", type=int, default=0)
    ap.add_argument("--arc-only", default="none", help="teacher arc_only variant: none | all | comma list of bodies")
    ap.add_argument("--device", default="cpu", help="cpu (default; eval runs in CPU leases) or cuda")
    from rrp.harness.eval.deploy_eval import DeployOptions, add_deploy_args
    add_deploy_args(ap)
    ap.add_argument("--system2", default="off", help="D-126 #31 system II harness (rrp.evaluation.system2): off (default) "
                    "| oracle (DIAGNOSTIC) | default | mock[:name] | vlm[:<weights dir>]; instruction -> target -> context")
    a = ap.parse_args(argv)
    from rrp.harness.eval.system2 import make_system2
    sys2 = make_system2(a.system2)
    deploy = DeployOptions.from_args(a)
    deploy = None if deploy.is_default() else deploy
    dev = _dev() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(2)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "a") as f:
        for body in a.bodies.split(","):
            nv = 0
            for sd in parse_seed_spec(a.seeds):
                if a.bc and not a.oracle_bc:
                    ctl = BCController(Path(a.bc), dev, nfe=a.nfe, replan=a.replan, seed=sd)
                elif a.flow or a.rep:
                    ctl = LatentLeggedController(Path(a.flow) if a.flow else None, dev, nfe=a.nfe, edit=a.edit,
                                                 t_edit=a.t_edit, seed=sd, posthoc_probe=a.posthoc_probe, rep=a.rep,
                                                 realizer=a.realizer, zero_qd=a.zero_qd)
                    if a.oracle_bc:
                        from rrp.policies.nets.legged_bc import load_bc
                        ctl.bc, _ = load_bc(a.bc, dev)
                        ctl.bc_version = f"{Path(a.bc).parent.name}/{Path(a.bc).name}"
                else:
                    ctl = None
                want = a.video_dir is not None and nv < a.video_n
                s2kw, s2rec = {}, None
                if sys2 is not None:
                    from rrp.harness.eval.system2 import _attach_provider, system2_episode
                    sc2, prov, s2rec = system2_episode(sys2, body, sd)
                    s2kw["scenario"] = sc2
                    if ctl is not None and not isinstance(ctl, BCController):
                        _attach_provider(ctl, prov)
                row, frames = run_episode(ctl, body, sd, a.max_s, video=True if want else None, oracle=a.oracle,
                                         arc_only=(a.flow is None and (a.arc_only == "all" or body in a.arc_only.split(","))),
                                         **({} if deploy is None else dict(deploy=deploy)), **s2kw)
                if s2rec is not None:
                    row["system2"] = s2rec
                if "_packet_z" in row:
                    _save_packets(row.pop("_packet_z"), Path(a.record_packets), row)
                if want:
                    lab = ("SCRIPTED TEACHER (privileged)" if ctl is None else (
                        "PRIVILEGED ORACLE packets (E on shadow teacher) + LEARNED sys-0" if a.oracle
                        else f"LEARNED BC (no packet; positive control) {row['source']}" if isinstance(ctl, BCController)
                        else f"ORACLE DIAGNOSTIC packets E(BC chunk) + LEARNED sys-0" if a.oracle_bc
                        else f"LEARNED latent sys-i+sys-0 {row['source']}"))
                    row["video"] = save_video(frames, row, Path(a.video_dir), lab)
                    nv += 1
                rows.append(row)
                f.write(json.dumps(row) + "\n"); f.flush()
                del frames, ctl
                gc.collect()      # W8: sessions sit in reference cycles (~0.45 GB each in contact_v2); free per episode
                print(json.dumps({k: row[k] for k in ("body", "seed", "source", "edit", "success", "fell", "sim_time",
                                                      "failure_stage")}),
                      flush=True)
    summ = {}
    for body in a.bodies.split(","):
        rs = [r for r in rows if r["body"] == body]
        stages = {}
        for r in rs:
            stages[r["failure_stage"]] = stages.get(r["failure_stage"], 0) + 1
        summ[body] = dict(n=len(rs), success=sum(r["success"] for r in rs), fell=sum(r["fell"] for r in rs),
                          stages=stages, seeds=a.seeds,
                          mean_sim_time=float(np.mean([r["sim_time"] for r in rs])),
                          packet_probes=packet_probe_accuracy(rs) if (a.flow or a.rep) else None)
    out.with_suffix(".summary.json").write_text(json.dumps(dict(source=rows[0]["source"], edit=a.edit, per_body=summ),
                                                           indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
