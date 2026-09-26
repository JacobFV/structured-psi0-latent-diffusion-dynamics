"""Side-by-side video of a causal edit of the RECEIVED latent packet: left = unedited packets (control),
right = the same seed with every received packet transformed (same flow noise per system-i call).
System i and system 0 are frozen; only the packet handed to system 0 differs.
Usage (peer GPU lease):
  render_causal_edit.py --checkpoint artifacts/runs/flow_latent_sem_v2/policy.pt --robot panda_pg2 --seed 3000001 \
      --condition focus_swap --out artifacts/video
Appends a line to <out>/INDEX.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np


def main_semantic(a):
    """Semantic / arm edits (rrp.evaluation.latent_semantic_edits): the SAME runner as the measured suite, with a
    frame observer. Left = unedited control, right = edited context (same seed / scene / flow-noise keys)."""
    import torch
    from rrp.evaluation import latent_semantic_edits as se
    from rrp.evaluation import latent_causal as lc
    from rrp.learning.latent_train import load_representation
    sys_path = Path(__file__).resolve().parent
    import sys
    sys.path.insert(0, str(sys_path))
    from render_episode import caption
    dev = "cpu"
    arm = a.suite == "arm"
    if a.route == "bc":
        from rrp.policy.runner import LearnedPolicy
        src = se.BCSource(LearnedPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=8, execute_prefix=8), a.checkpoint)
        R = P = rep = None
        srclab = f"LEARNED BC reference {Path(a.checkpoint).stem} (not latent)"
        tag = f"learned_bc_{Path(a.checkpoint).stem}"
    elif a.route == "generated":
        from rrp.learning.checkpoint import load_checkpoint
        rep = Path(load_checkpoint(a.checkpoint, map_location="cpu")["config"]["representation"])
        _, _, R, P, _ = load_representation(rep, dev)
        if arm:
            from rrp.evaluation.dual_latent_eval import DualLatentPolicy as Pol
        else:
            from rrp.policy.latent_runner import LatentPolicy as Pol
        src = se.GeneratedSource(Pol.from_checkpoint(a.checkpoint, device=dev, nfe=8))
        srclab = f"LEARNED {Path(a.checkpoint).parent.name}"
        tag = f"learned_{Path(a.checkpoint).parent.name}"
    else:
        rep = Path(a.representation)
        lcfg, E, R, P, res = load_representation(rep, dev)
        if a.route == "oracle" and a.oracle_expert == "bc":
            from rrp.policy.runner import LearnedPolicy
            src = se.OracleSource(E, lcfg, res, rep, dev, expert="bc", bc=LearnedPolicy.from_checkpoint(
                a.checkpoint, device=dev, nfe=8, execute_prefix=8))
            srclab = f"ORACLE DIAGNOSTIC E({rep.parent.name})+stateless BC demo"
            tag = f"oracle_bcexpert_{rep.parent.name}"
        elif a.route == "oracle":
            src = se.OracleSource(E, lcfg, res, rep, dev)
            srclab = f"ORACLE DIAGNOSTIC E({rep.parent.name})+teacher demo"
            tag = f"oracle_{rep.parent.name}"
        else:
            src = se.TeacherSource()
            srclab = "SCRIPTED TEACHER (privileged)"
            tag = "scripted_teacher"
    if rep is not None:
        probe = a.probe or (str(rep.parent / "probe_posthoc.pt") if (rep.parent / "probe_posthoc.pt").exists() else None)
        P = lc.load_probe(probe, P, dev)
    rend, frames, rows = {}, {}, {}
    for c in ("control", a.condition):
        fr = frames.setdefault(c, [])

        def obs(s, step, fr=fr):
            if step % a.every:
                return
            r = rend.get(id(s))
            if r is None:
                r = rend[id(s)] = mujoco.Renderer(s.model, a.height, a.width)
            r.update_scene(s.data, camera=a.camera)
            fr.append((r.render().copy(), float(s.data.time)))
        if arm:
            rows[c] = se.run_arm_condition(src, R, P, a.pair, a.seed, c, max_steps=a.max_steps, dev=dev, on_step=obs)
        else:
            from rrp.morphology.catalog import workbench_robots
            robot = workbench_robots()[a.robot]()
            rows[c] = se.run_condition(src, R, P, robot, a.robot, a.seed, c, max_steps=a.max_steps, dev=dev,
                                       scene=a.scene, on_step=obs)
    def desc(r):
        if "skipped" in r:
            return [f"skipped: {r['skipped']}"]
        if arm:
            return [f"assigned={r['assigned']} -> expected mover={r['expected_arm']}",
                    f"first touch={r['first_contact_arm']} min dist L/R={r['min_dist_bar_m']['left']:.2f}/"
                    f"{r['min_dist_bar_m']['right']:.2f} m"]
        return [f"first approached={r['approached_first']} first touch={r['first_contact']}",
                f"min dist old/new={r['min_tcp_dist']['cube']:.2f}/{r['min_tcp_dist']['distractor0']:.2f} m"]
    what = (lambda r: f"task: {r['assigned']} arm") if arm else (
        lambda r: f"task object: {r.get('patient_color', 'cube')}")
    edit_lab = {"rebind_desc": "EDIT: task rebound to " + str(rows["control"].get("rebind_color", "distractor0")),
                "rebind_obj": "EDIT: task rebound to " + str(rows["control"].get("rebind_color", "distractor0")),
                "swap_arm": "EDIT: actor rebound to " + str(rows["control"].get("edited_to", "")) + " arm",
                "goal_shift": "EDIT: goal moved 12 cm", "irrelevant_distractor": "CONTROL EDIT: unbound object belief moved",
                "orthogonal_matched": "CONTROL EDIT: probe-orthogonal z, matched norm",
                "swap_slots": "EDIT: packet slots exchanged"}.get(a.condition, a.condition)
    n = max(len(frames["control"]), len(frames[a.condition]))
    out_frames = []
    for k in range(n):
        imgs = []
        for c, lab in (("control", "UNEDITED (control)"), (a.condition, edit_lab)):
            fr = frames[c]
            img, t = fr[min(k, len(fr) - 1)]
            imgs.append(caption(img, [f"{srclab} | {a.pair if arm else a.robot} | key {a.seed}",
                                      f"{lab} | {what(rows['control'])}", f"t={t:.1f}s"] + desc(rows[c])))
        out_frames.append(np.concatenate(imgs, axis=1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    body = a.pair if arm else a.robot
    name = f"{dt.date.today()}_semantic_edit_{a.condition}_{tag}_{body}_k{a.seed}.mp4"
    imageio.mimsave(out / name, out_frames, fps=a.fps, quality=6)
    rc, re_ = rows["control"], rows[a.condition]
    if arm:
        res = (f"first touch control={rc.get('first_contact_arm')} edited={re_.get('first_contact_arm')}; "
               f"expected mover edited={re_.get('expected_arm')}")
    else:
        res = (f"first approached control={rc.get('approached_first')} edited={re_.get('approached_first')}; "
               f"first touch control={rc.get('first_contact')} edited={re_.get('first_contact')}")
    with open(out / "INDEX.md", "a") as f:
        f.write(f"- `{name}` — semantic edit side by side (left unedited, right {a.condition}); source="
                f"{srclab}; {'dual-arm assign_pick_place ' + a.pair if arm else a.scene + ' ' + a.robot} key={a.seed}; "
                f"{res} (privileged measurement; old object = 'cube', new = 'distractor0')\n")
    print(name, res)


def main(a):
    if a.suite in ("semantic", "arm"):
        return main_semantic(a)
    import torch
    from rrp.policy.latent_runner import LatentPolicy
    from rrp.learning.checkpoint import load_checkpoint
    from rrp.learning.latent_train import load_representation
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.evaluation import latent_causal as lc
    sys_path = Path(__file__).resolve().parent
    import sys
    sys.path.insert(0, str(sys_path))
    from render_episode import caption
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from rrp.ops.gpu import apply_cap
        apply_cap()
    from rrp.evaluation import latent_semantic_edits as se
    if a.route == "oracle":                 # ORACLE DIAGNOSTIC: E(context, scripted-teacher demo) -> system 0
        rep = Path(a.representation)
        lcfg, E, R, P, res = load_representation(rep, dev)
        pol = se.OracleSource(E, lcfg, res, rep, dev)
        pol.lsv, pol.rcv = res["latent_space_version"], res["realizer_compat_version"]
    else:
        pol = LatentPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=8)
        rep = Path(load_checkpoint(a.checkpoint, map_location="cpu")["config"]["representation"])
        _, _, R, P, _ = load_representation(rep, dev)
    probe = a.probe or (str(rep.parent / "probe_posthoc.pt") if (rep.parent / "probe_posthoc.pt").exists() else None)
    P = lc.load_probe(probe, P, dev)
    robot = workbench_robots()[a.robot]()
    nd = max(1, a.seed % 3) if a.condition in se.CONDITIONS else a.seed % 3
    S = [Session(BUILDERS["pick_place"](robot, a.seed, n_distractors=nd), seed=a.seed) for _ in range(2)]
    goff = se.goal_offset(S[0])
    teach = [se.make_teacher("control", S[0], goff), se.make_teacher(a.condition, S[1], goff)] \
        if a.route == "oracle" else [None, None]
    s0 = [lc._s0(pol, R, s, dev) for s in S]
    rend = [mujoco.Renderer(s.model, a.height, a.width) for s in S]
    ck = "ORACLE-DIAGNOSTIC E(" + rep.parent.name + ")+teacher demo" if a.route == "oracle" else Path(a.checkpoint).parent.name
    frames, calls, done = [], 0, [False, False]
    d = a.delta_cm / 100
    for step in range(a.max_steps):
        if step % 8 == 0:
            key = lc._key(a.seed, calls)
            calls += 1
            for i, s in enumerate(S):
                if done[i]:
                    continue
                if a.route == "oracle" or a.condition in se.CONDITIONS:
                    c = "control" if i == 0 else a.condition
                    srcobj = pol if a.route == "oracle" else se.GeneratedSource(pol)
                    q = lc.deliver(srcobj.packet(s, c, teach[i], goff, key), s)
                    lc._recv(s0[i], q, s)
                    continue
                p = pol.packets([s], noise_keys=[key])[0]
                q = lc.deliver(p, s)
                if i == 1:
                    c = a.condition
                    if c in lc.DIRS:
                        zs, _, _ = lc.probe_edits(P, [p.z], [s.entity_slots.get("cube", lc._slot(s, "cube"))],
                                                  [len(s.detectables)], [np.array(lc.DIRS[c]) * d], dev=dev)
                        q = lc.deliver(p, s, zs[0], c)
                    elif c in lc.CF_DIRS and not lc._held(s):
                        cf = lc.counterfactual_packets(pol, [s], [key], lambda ss: lc.shift_state(
                            ss, np.array(lc.CF_DIRS[c]) * d))[0]
                        q = lc.deliver(cf, s, cf.z, c)
                    elif c == "focus_swap" and not lc._held(s):
                        cf = lc.counterfactual_packets(pol, [s], [key])[0]
                        q = lc.deliver(cf, s, cf.z, c)
                    elif c == "zero":
                        q = lc.deliver(p, s, np.zeros_like(p.z), c)
                lc._recv(s0[i], q, s)
        imgs = []
        for i, s in enumerate(S):
            if not done[i]:
                s.step(s0[i].tick(s, s.controller_version()))
                if teach[i] is not None:
                    teach[i].act()                  # shadow expert follows the real state (never executed)
                done[i] = bool(s.runtime.succeeded() or lc._body_pos(s, "cube")[2] < -0.05)
            if step % a.every == 0:
                rend[i].update_scene(s.data, camera=a.camera)
                tag = "UNEDITED packet (control)" if i == 0 else f"EDITED packet: {a.condition}"
                imgs.append(caption(rend[i].render().copy(), [
                    f"LEARNED latent {ck} | {a.robot} | seed {a.seed}", tag,
                    f"t={s.data.time:.1f}s tcp=({lc._tcp(s)[0]:.2f},{lc._tcp(s)[1]:.2f},{lc._tcp(s)[2]:.2f})"]))
        if imgs:
            frames.append(np.concatenate(imgs, axis=1))
        if all(done):
            break
    ok = [bool(s.privileged_success()) for s in S]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tagsrc = f"oracle_{rep.parent.name}" if a.route == "oracle" else f"learned_{ck}"
    name = f"{dt.date.today()}_causal_edit_{a.condition.replace('+', 'p').replace('-', 'm')}_{tagsrc}_{a.robot}_s{a.seed}.mp4"
    imageio.mimsave(out / name, frames, fps=a.fps, quality=6)
    tcp_gap = float(np.linalg.norm(lc._tcp(S[0]) - lc._tcp(S[1])))
    with open(out / "INDEX.md", "a") as f:
        f.write(f"- `{name}` — causal edit of the RECEIVED packet, side by side (left unedited control, right "
                f"{a.condition}); source={'ORACLE DIAGNOSTIC target_encoder_oracle E(' + str(rep) + ') + scripted_teacher demo' if a.route == 'oracle' else 'learned:' + str(a.checkpoint)} (system 0 frozen) robot={a.robot} "
                f"task=pick_place seed={a.seed} outcome control={'success' if ok[0] else 'failure'} "
                f"edited={'success' if ok[1] else 'failure'} (privileged evaluator); final TCP gap {tcp_gap:.3f} m\n")
    print(name, ok, tcp_gap)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--route", choices=["generated", "oracle", "teacher", "bc"], default="generated")
    ap.add_argument("--representation", help="oracle route: frozen representation.pt")
    ap.add_argument("--robot", default="panda_pg2")
    ap.add_argument("--seed", type=int, default=3000001)
    ap.add_argument("--condition", default="focus_swap")
    ap.add_argument("--probe")
    ap.add_argument("--delta-cm", type=float, default=5.0)
    ap.add_argument("--out", default="artifacts/video")
    ap.add_argument("--camera", default="front")
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=300)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=240)
    ap.add_argument("--suite", choices=["causal", "semantic", "arm"], default="causal",
                    help="semantic/arm: videos of rrp.evaluation.latent_semantic_edits conditions (same runner)")
    ap.add_argument("--scene", default="paired")
    ap.add_argument("--oracle-expert", choices=["teacher", "bc"], default="teacher")
    ap.add_argument("--pair", default="panda_pg2__ur5e_pg2")
    main(ap.parse_args())
