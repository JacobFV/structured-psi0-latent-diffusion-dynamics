"""PEER-ONLY replay recorder for the live visualization room (viz/CONTRACT.md "replay file", D-131).

    python -m rrp.viz.record --spec viz/specs/<name>.yaml --out <dir> [--only ID[,ID]] [--shard i/n] [--list]

Each spec entry re-runs chosen episodes through EXACTLY the harness that produced a recorded result (same module
functions, seeds, batching, checkpoints, physics env vars, thread counts, CPU device), and records them by OBSERVING
that harness from outside: existing read-only callbacks (ladder frame_cb, run_condition on_step), or class-level
wrappers around read-only recorder hooks (LeggedMotionRecorder.on_tick, DualQualityRecorder.after_step) or around
`mujoco.mj_step` in the harness module namespace (grasp rig, tracker validation). The wrappers call the original first
and then only READ mjData; no harness file is edited and no RNG is consumed. tests/unit/test_viz_record.py checks
recorded == unrecorded actions on a tiny case.

Every replay carries the original eval row's outcome (meta.recorded_success) next to the re-run's (meta.success) and
meta.reproduced; a mismatch is reported, never hidden. Each entry runs in its own subprocess with the entry's env
(e.g. RRP_GRASP_CONTACT, RRP_CONTACT_MODEL) and CUDA hidden. D-115/D-127: never run this on the host; the CLI refuses
unless RRP_NODE=peer (tests call the library functions on tiny cases only).

Spec (YAML subset, rrp.orchestration.yamlmini):
  entries:
    - id: arm-r2-semfix_s1-parm6_tf3-gv2        # replay id = <id>-s<seed>
      harness: ladder | arm_teacher | arm_edit | legged | legged_robust | tracker_val | dual_teacher | grasp_rig
      family / task / body / route / variant / condition / source_label / decision_refs / caveat
      env: {RRP_GRASP_CONTACT: v2}
      threads: 2
      args: {...harness arguments...}
      seeds: [..]                                # episodes to record
      recorded: {file: <rows file>, ...}         # the original eval rows (reproduction check)
      pca: {...}                                 # latent routes: training packets for the per-bundle PCA basis
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rrp.viz import replay as RP

REPO = Path.cwd()


# ------------------------------------------------------------------ small helpers
def _sha(p) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git_sha() -> str | None:
    rev = REPO / ".rrp_revision"
    if rev.exists():
        try:
            d = json.loads(rev.read_text())
            return (d.get("git_sha") or "")[:40] + ("+dirty" if d.get("dirty") else "") or None
        except ValueError:
            return rev.read_text().strip()[:40]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip() or None
    except OSError:
        return None


def _read_rows(path: str) -> list[dict]:
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as fh:
        txt = fh.read()
    if p.name.endswith(".json") or p.name.endswith(".json.gz"):
        return [json.loads(txt)]
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def find_recorded(rec: dict, seed: int, extra_match: dict | None = None) -> dict | None:
    """The original eval row for `seed` (and extra key=value matches, e.g. condition). rec = {file, seed_key?,
    match?: {...}} or {file, path: [trial], index_from: 1000} for JSON documents with per-trial episode lists."""
    if not rec or not rec.get("file") or not Path(rec["file"]).exists():
        return None
    rows = _read_rows(rec["file"])
    if "path" in rec:                                  # JSON document: episodes list under a key path
        doc = rows[0]
        for k in rec["path"]:
            doc = doc[k]
        i = seed - int(rec.get("index_from", 0))
        return doc[i] if 0 <= i < len(doc) else None
    sk = rec.get("seed_key", "seed")
    want = dict(rec.get("match") or {}, **(extra_match or {}))
    for r in rows:
        if r.get(sk) != seed:
            continue
        if all(_get(r, k) == v for k, v in want.items()):
            return r
    return None


def _get(r, dotted):
    for k in dotted.split("."):
        if not isinstance(r, dict):
            return None
        r = r.get(k)
    return r


def physics_meta(model, *, family: str) -> dict:
    from rrp.physics.contact import model_contact_version
    from rrp.physics.grasp_contact import model_grasp_version
    from rrp.physics.actuator import model_actuator_limits, resolve_mode
    arm = family in ("arm", "dual", "grasp_rig")
    return dict(contact_version=None if arm else (model_contact_version(model) or "contact_v1"),
                grasp_contact_version=(model_grasp_version(model) or "grasp_v1") if arm else None,
                actuator_limits_version=model_actuator_limits(model), actuator_mode=resolve_mode())


def _stride_for(control_hz: float) -> int:
    return max(1, math.ceil(control_hz / RP.MAX_FPS - 1e-9))


class _Proxy:
    """Stands in for the `mujoco` module inside ONE harness module: forwards everything, wraps mj_step to call the
    original and then the (read-only) observer."""

    def __init__(self, mod, on_step):
        self._mod, self._on_step = mod, on_step

    def __getattr__(self, k):
        return getattr(self._mod, k)

    def mj_step(self, m, d, *a, **k):
        r = self._mod.mj_step(m, d, *a, **k)
        self._on_step(m, d)
        return r


@contextlib.contextmanager
def _patched(obj, name, new):
    old = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield old
    finally:
        setattr(obj, name, old)


# ------------------------------------------------------------------ geometry + common per-frame reads
class Episode:
    """One replay being recorded: static geometry on first sight of the model, frames at <= 30 fps."""

    def __init__(self, control_hz: float, *, bodies_filter=None):
        self.control_hz = control_hz
        self.col: RP.FrameCollector | None = None
        self.geoms = None
        self.model = None
        self.bodies_filter = bodies_filter
        self.extra: dict = {}

    def ensure(self, model):
        if self.col is None:
            self.model = model
            self.geoms, bodies = RP.export_geoms(model)
            self.col = RP.FrameCollector(model, bodies, 1.0 / self.control_hz)
        return self.col

    def reset_frames(self):
        if self.col is not None:
            m = self.model
            self.col = None
            self.ensure(m)


def _penetration_mm(m, d, geoms_a: set | None = None, bodies_b: set | None = None) -> float:
    """Max penetration (mm) over contacts that touch geoms_a (or any, if None) and bodies_b (or any)."""
    pen = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if geoms_a is not None and not (c.geom1 in geoms_a or c.geom2 in geoms_a):
            continue
        if bodies_b is not None:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if not (b1 in bodies_b or b2 in bodies_b):
                continue
        pen = max(pen, -float(c.dist))
    return pen * 1000.0


def _statuses(runtime) -> dict:
    return {e: str(v.status) for e, v in runtime.instances.items()}


# ------------------------------------------------------------------ packet PCA bases (per bundle)
def pca_basis(spec: dict | None, cache: dict) -> dict | None:
    if not spec:
        return None
    key = json.dumps(spec, sort_keys=True)
    if key in cache:
        return cache[key]
    kind = spec["kind"]
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    n = int(spec.get("n", 4000))
    if kind == "arm_npz":                               # system-0 DAgger buffers (the bundle's training packets)
        import glob
        files = sorted(glob.glob(spec["glob"]))
        if not files:
            raise FileNotFoundError(spec["glob"])
        Z = [np.load(f)["mu"].astype(np.float32).reshape(-1, np.prod(np.load(f)["mu"].shape[1:])) for f in files]
        dims = {z.shape[1] for z in Z}
        if len(dims) != 1:
            raise ValueError(f"packet dims differ across {spec['glob']}: {dims}")
        Z = np.concatenate(Z)
        Z = Z[rng.choice(len(Z), min(n, len(Z)), replace=False)]
        b = RP.fit_pca(Z)
        b.update(fit_source=dict(kind=kind, glob=spec["glob"], files=[Path(f).name for f in files],
                                 note="mu of the system-0 refit DAgger buffers (packets system 0 was trained on)"))
    elif kind == "legged_rep":                          # E(public ctx, demonstrated targets) on the training data
        import torch
        from rrp.controllers.bundles import load_rep
        from rrp.training.legged_latent_train import LeggedData
        rcfg, E, _R, _P, _res = load_rep(Path(spec["rep"]), torch.device("cpu"))
        data = LeggedData(Path(rcfg["data"]), [spec["body"]], torch.device("cpu"))
        M = int(spec["M"])
        zs = []
        with torch.no_grad():
            for _ in range(max(1, n // 256)):
                i = torch.from_numpy(rng.choice(data.train_idx, 256)).to(torch.device("cpu"))
                mu, _ = E(data.ctx_batch(i), data.beh(i))
                zs.append(mu[:, :, :M].reshape(len(i), -1).numpy())
        b = RP.fit_pca(np.concatenate(zs))
        b.update(fit_source=dict(kind=kind, rep=spec["rep"], data=str(rcfg["data"]), body=spec["body"],
                                 note="mu = E(ctx, demonstrated targets) on training rows (the flow's targets)"))
        del data
    else:
        raise ValueError(f"unknown pca kind {kind}")
    cache[key] = b
    return b


# ------------------------------------------------------------------ ARM: ladder (R2 generated / BC learned / teacher route)
class _ArmSignals:
    def __init__(self, s, P=None, dev="cpu"):
        import mujoco
        m = s.model
        self.s, self.P, self.dev = s, P, dev
        self.r = s.robots[0]
        self.cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.cube}
        names = set()
        for r in s.robots:
            for l in r.spec.links:
                names |= {l.name, (r.prefix or "") + l.name}
        self.robot_bodies = {b for b in range(m.nbody) if m.body(b).name in names}
        grip = next((a for a in self.r.spec.assemblies if a.kind in ("gripper", "hand")), None)
        self.tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, grip.frame.site) if grip else -1
        self.cube_slot = next((i for i, o in enumerate(s.detectables) if o.sim_body == "cube"), 0)
        self.prev_rel, self._pcache = None, {}

    def joint_target(self):
        c = self.r.controller
        tg = c.target or {}
        return np.concatenate([np.atleast_1d(np.asarray(tg[g], float)) for g in c.groups if g in tg]) if tg else None

    def frame(self, packet=None, basis=None, phase=None, dt=0.05, edit_active=None) -> dict:
        s, d, m = self.s, self.s.data, self.s.model
        pose = np.concatenate([d.xpos[self.cube], d.xquat[self.cube]])
        rel = None
        if self.tcp >= 0:
            R = d.site_xmat[self.tcp].reshape(3, 3)
            rel = R.T @ (d.xpos[self.cube] - d.site_xpos[self.tcp])
        tv = s._touch_values(self.r)
        touching = bool(len(tv) and tv.max() >= 0.2)
        near = rel is not None and np.linalg.norm(rel) < 0.08
        held = touching and near
        slip = None
        if held and self.prev_rel is not None and rel is not None:
            slip = float(np.linalg.norm(rel - self.prev_rel)) * 1000.0 / dt     # mm/s relative to the TCP
        self.prev_rel = rel if held else None
        out = dict(joint_target=self.joint_target(), joint_pos=d.qpos[self.r.qadr].copy(),
                   contacts=[bool(v >= 0.2) for v in tv], object_pose=pose,
                   penetration_mm=_penetration_mm(m, d, self.cube_geoms, self.robot_bodies), slip=slip)
        if phase is not None:
            out["phase"] = str(phase)
        if edit_active is not None:
            out["edit_active"] = bool(edit_active)
        if packet is not None:
            z = np.asarray(packet.z, np.float32)
            if basis is not None and z.size == basis["dim"]:
                out["packet_pca"] = RP.project_pca(basis, z)
            if self.P is not None:
                out["probe"] = self.probe(packet, z)
        return out

    def probe(self, packet, z):
        key = id(packet)
        if key not in self._pcache:
            import torch
            with torch.no_grad():
                zt = torch.as_tensor(z[None], dtype=torch.float32, device=self.dev)
                o = self.P(zt, torch.ones(1, z.shape[1], dtype=torch.bool, device=self.dev), len(self.s.detectables))
                sig = torch.sigmoid
                g = o.get("goal_effect", o["observed_effect"])
                self._pcache = {key: dict(
                    held_by=sig(o["held_by"][0, :, 0, 0]).cpu().numpy(),
                    contact=sig(o["acting_on"][0, :, 0, 0]).cpu().numpy(),
                    subtask=int(o["subtask"][0, 0].argmax()),
                    goal=g[0, self.cube_slot, :2].cpu().numpy())}
        return self._pcache[key]


def _install_packet_tap():
    """Remember the last packet each LatentSystem0 received (keyed by its featurizer object; read-only)."""
    from rrp.controllers.latent_realizer import LatentSystem0
    tap: dict = {}
    orig = LatentSystem0.receive

    def receive(self, packet, *a, **k):
        r = orig(self, packet, *a, **k)
        tap[id(self.f)] = packet
        return r
    return tap, (LatentSystem0, "receive", receive)


def run_ladder(e: dict, out: Path, pcache: dict) -> list[dict]:
    """Arm R2 / BC / teacher-route episodes, re-run exactly as rrp.evaluation.ladder_cli does: the full feasible seed
    list, batches of `batch` in order (the flow/BC noise generator is shared across batches), CPU."""
    import torch
    from rrp.evaluation.ladder import LadderConfig, load_models, run_ladder as _run
    from rrp.evaluation.robustness import feasible_arm_seeds
    a = e["args"]
    seeds_all = feasible_arm_seeds(a["robot"], int(a["seed_start"]), int(a.get("n", 30)))
    want = [int(x) for x in e["seeds"]]
    miss = [x for x in want if x not in seeds_all]
    if miss:
        raise ValueError(f"{e['id']}: seeds {miss} not in the feasible eval set")
    cfg = LadderConfig(route=a["route"], robot=a["robot"], seeds=seeds_all, representation=a.get("rep"), flow=a.get("flow"),
                       policy=a.get("policy"), policy_label=a.get("policy_label"), oracle_expert=a.get("oracle_expert", "teacher"),
                       noise_scale=float(a.get("noise_scale", 1.0)), replan_ticks=int(a.get("replan", 8)),
                       max_steps=int(a.get("max_steps", 300)), nfe=int(a.get("nfe", 8)),
                       compare_oracle=not a.get("no_compare", False), device="cpu", prev_action=a.get("prev_action", "zero"))
    models, ids = load_models(cfg) if (a["route"] != "teacher" or a.get("rep")) else (
        dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None), {})
    basis = pca_basis(e.get("pca"), pcache)
    P = models.get("P") if a["route"] == "generated" else None
    tap, patch = _install_packet_tap()
    batch = int(a.get("batch", 16))
    eps: dict[int, tuple] = {}
    results = []
    last = max(seeds_all.index(x) for x in want)
    with _patched(*patch[:2], patch[2]):
        for i in range(0, last + 1, batch):
            cfg.seeds = seeds_all[i:i + batch]
            tk = {k: sd for k, sd in enumerate(cfg.seeds) if sd in want}

            def cb(k, s, step, phase, tk=tk):
                if k not in tk:
                    return
                if k not in eps:
                    eps[k] = (Episode(1.0 / s.dt), _ArmSignals(s, P))
                ep, sig = eps[k]
                col = ep.ensure(s.model)
                if col.due():
                    pk = tap.get(id(getattr(s, "_rrp_featurizer", None))) if P is not None or basis is not None else None
                    col.frame(s.data, **sig.frame(packet=pk, basis=basis, phase=phase, dt=s.dt * col.stride))
                    col.events(s.data.time, _statuses(s.runtime))
                col.tick()
            with torch.no_grad():
                rows = _run(cfg, None, models, ids, frame_cb=cb, cmd_log=e.get("_cmd_log"))
            for k, sd in tk.items():
                ep, _ = eps.pop(k)
                row = rows[k]
                rec = find_recorded(e.get("recorded"), sd)
                meta = _meta(e, sd, ep.model, success=bool(row["privileged_success"]), failure_stage=row.get("failed_stage"),
                             ckpt_sha=_ckpt_shas(ids), recorded=rec, rec_key="privileged_success",
                             compare=dict(outcome=(row.get("outcome"), rec.get("outcome") if rec else None),
                                          steps=(row.get("steps"), rec.get("steps") if rec else None)))
                meta["eval_config"] = dict(route=cfg.route, robot=cfg.robot, seed_list=[seeds_all[0], seeds_all[-1], len(seeds_all)],
                                           batch=batch, batch_index=i // batch, position_in_batch=k, device="cpu",
                                           replan=cfg.replan_ticks, nfe=cfg.nfe, compare_oracle=cfg.compare_oracle,
                                           prev_action=cfg.prev_action, max_steps=cfg.max_steps)
                meta["checkpoints"] = ids
                if P is not None:
                    meta["probe_source"] = dict(path=cfg.representation, note="packet probe P of the system-0 bundle")
                if basis is not None:
                    meta["packet_pca_basis"] = basis
                meta["signal_notes"] = ARM_NOTES
                results.append(_finish(e, sd, meta, ep, out))
    return results


ARM_NOTES = dict(
    joint_target="executed joint-target command (controller target after the tick), arm then gripper groups",
    joint_pos="measured arm+gripper joint positions (rad / m)",
    contacts="finger touch sensors >= 0.2 (public sensors)",
    object_pose="cube position xyz + quaternion wxyz (privileged, display only)",
    penetration_mm="max cube-robot penetration (privileged)",
    slip="cube speed relative to the TCP frame while touching and within 8 cm (mm/s; privileged); null otherwise",
    phase="shadow scripted-teacher FSM phase at the real state (privileged diagnostic label; never controls)",
    packet_pca="the packet system 0 currently executes, projected on the bundle PCA basis (meta.packet_pca_basis)",
    probe="bundle packet probe on that packet: held_by/contact = sigmoid per detector slot, subtask = argmax operator, "
          "goal = goal-effect head xy for the cube slot (decimetre units as trained)")


def _ckpt_shas(ids: dict) -> dict:
    out = {}
    for k, v in (ids or {}).items():
        if isinstance(v, dict) and v.get("sha256"):
            out[k] = v["sha256"]
    return out


# ------------------------------------------------------------------ ARM: scripted teacher v2 (rrp.evaluation.teacher_quality)
def run_arm_teacher(e: dict, out: Path, pcache: dict) -> list[dict]:
    from rrp.envs import native
    from rrp.evaluation.teacher_quality import run_quality_episode
    import rrp.teachers.arm_smooth as AS
    a = e["args"]
    results = []
    for sd in e["seeds"]:
        sd = int(sd)
        st = dict(ep=None, sig=None, armed=False, depth=0, teacher=None)
        orig_step = native.Session.step
        orig_make = AS.make_arm_teacher

        def make(s, version, *aa, **kk):
            t = orig_make(s, version, *aa, **kk)
            st.update(armed=True, teacher=t, ep=Episode(1.0 / s.dt), sig=_ArmSignals(s))
            return t

        def step(self, *aa, **kk):
            st["depth"] += 1
            try:
                r = orig_step(self, *aa, **kk)
            finally:
                st["depth"] -= 1
            if st["armed"] and st["depth"] == 0:
                col = st["ep"].ensure(self.model)
                if col.due():
                    col.frame(self.data, **st["sig"].frame(phase=getattr(st["teacher"], "phase", None), dt=self.dt * col.stride))
                    col.events(self.data.time, _statuses(self.runtime))
                col.tick()
            return r
        with _patched(native.Session, "step", step), _patched(AS, "make_arm_teacher", make):
            row = run_quality_episode(a["robot"], sd, a.get("version", "v2"), max_steps=int(a.get("max_steps", 600)))
        if st["ep"] is None:
            raise RuntimeError(f"{e['id']} seed {sd}: episode not run ({row.get('outcome')})")
        rec = find_recorded(e.get("recorded"), sd, {"robot": a["robot"], "version": a.get("version", "v2")})
        meta = _meta(e, sd, st["ep"].model, success=bool(row["success"]), failure_stage=row.get("failure_stage"),
                     ckpt_sha=None, recorded=rec, rec_key="success",
                     compare=dict(outcome=(row.get("outcome"), rec.get("outcome") if rec else None),
                                  steps=(row.get("steps"), rec.get("steps") if rec else None)))
        meta["signal_notes"] = dict(ARM_NOTES, phase="scripted teacher v2 phase (the controller itself; privileged)")
        results.append(_finish(e, sd, meta, st["ep"], out))
    return results


# ------------------------------------------------------------------ ARM: semantic edits (rrp.cli latent semantic-edits)
def run_arm_edit(e: dict, out: Path, pcache: dict) -> list[dict]:
    import torch
    from rrp.bodies.catalog import workbench_robots
    from rrp.controllers.bundles import load_representation
    from rrp.controllers.latent_runner import LatentPolicy
    from rrp.evaluation import latent_causal as lc
    from rrp.evaluation import latent_semantic_edits as se
    from rrp.models.checkpoint import load_checkpoint
    a = e["args"]
    dev = "cpu"
    if a.get("route", "generated") != "generated":
        raise ValueError("arm_edit supports the generated route (the D-074/D-095 suites)")
    rep = Path(a.get("rep") or load_checkpoint(a["flow"], map_location="cpu")["config"]["representation"])
    _, _, R, P, res = load_representation(rep, dev)
    pol = LatentPolicy.from_checkpoint(a["flow"], device=dev, nfe=int(a.get("nfe", 8)))
    if pol.lsv != res["latent_space_version"]:
        raise SystemExit(f"flow latent space {pol.lsv} != system-0 bundle {res['latent_space_version']}")
    if pol.rcv != res["realizer_compat_version"]:
        pol.rcv = res["realizer_compat_version"]
    src = se.GeneratedSource(pol)
    probe = a.get("probe") or (str(rep.parent / "probe_posthoc.pt") if (rep.parent / "probe_posthoc.pt").exists() else None)
    P = lc.load_probe(probe, P, dev)
    robot = workbench_robots()[a["robot"]]()
    basis = pca_basis(e.get("pca"), pcache)
    tap, patch = _install_packet_tap()
    ids = dict(flow=dict(path=a["flow"], sha256=_sha(a["flow"])), representation=dict(path=str(rep), sha256=_sha(rep)),
               probe=dict(path=probe, sha256=_sha(probe) if probe else None))
    results = []
    for sd in e["seeds"]:
        sd = int(sd)
        for cond in e.get("conditions", ["control"]):
            ep = [None, None]

            def on_step(s, step):
                if ep[0] is None:
                    ep[0], ep[1] = Episode(1.0 / s.dt), _ArmSignals(s, P)
                col = ep[0].ensure(s.model)
                if col.due():
                    pk = tap.get(id(src.featurizer(s)))
                    col.frame(s.data, **ep[1].frame(packet=pk, basis=basis, dt=s.dt * col.stride,
                                                   edit_active=cond not in ("control",)))
                    col.events(s.data.time, _statuses(s.runtime))
                col.tick()
            with _patched(*patch[:2], patch[2]), torch.no_grad():
                row = se.run_condition(src, R, P, robot, a["robot"], sd, cond, max_steps=int(a.get("max_steps", 400)),
                                       dev=dev, scene=a.get("scene", "pick_place"), on_step=on_step)
            if row.get("skipped"):
                results.append(dict(id=f"{e['id']}-{cond}-s{sd}", skipped=row["skipped"]))
                continue
            rec = find_recorded(e.get("recorded"), sd, {"condition": cond})
            e2 = dict(e, id=f"{e['id']}-{cond}", condition=cond)
            meta = _meta(e2, sd, ep[0].model, success=bool(row["followed"]), failure_stage=None, ckpt_sha=_ckpt_shas(ids),
                         recorded=rec, rec_key="followed",
                         compare=dict(privileged_success=(row["privileged_success"], rec.get("privileged_success") if rec else None),
                                      first_contact=(row.get("first_contact"), rec.get("first_contact") if rec else None),
                                      lifted=(row.get("lifted"), rec.get("lifted") if rec else None)))
            meta.update(edit=cond, success_definition="followed (rebind: lifted the new cube and not the old; goal_shift: "
                        "placed at the shifted goal; control: placed in the zone)", privileged_success=row["privileged_success"],
                        edit_row={k: row.get(k) for k in ("first_contact", "approached_first", "lifted", "placed_in_zone",
                                                          "placed_at_shifted_goal", "goal_offset", "patient_color", "rebind_color")},
                        checkpoints=ids, signal_notes=dict(ARM_NOTES, edit_active="the edited task context is in force (from t=0)"))
            if basis is not None:
                meta["packet_pca_basis"] = basis
            results.append(_finish(e2, sd, meta, ep[0], out))
    return results


# ------------------------------------------------------------------ LEGGED (rrp.evaluation.legged_latent_eval / robustness)
class _LeggedSignals:
    def __init__(self, s, b, *, ctl=None, basis=None, t_edit=None, edit="none"):
        self.s, self.b, self.ctl, self.basis = s, b, ctl, basis
        self.t_edit, self.edit = t_edit, edit
        self.p0, self.yaw0 = None, None
        self._fg = set(g for g in (b.floor, getattr(b, "floor2", -1)) if g is not None and g >= 0)

    def start(self, d):
        from rrp.envs.legged_core import yaw_of
        b = self.b
        self.p0 = d.qpos[b.qa:b.qa + 2].copy()
        self.yaw0 = yaw_of(d.qpos[b.qa + 3:b.qa + 7])

    def frame(self, d) -> dict:
        b, m = self.b, self.b.model
        if self.p0 is None:
            self.start(d)
        fc, _fn, slip, _bad = b.stance(d)
        dp = d.qpos[b.qa:b.qa + 2] - self.p0
        fwd = math.cos(self.yaw0) * dp[0] + math.sin(self.yaw0) * dp[1]
        out = dict(joint_target=d.ctrl[b.pol_act].copy(), joint_pos=d.qpos[b.pol_qadr].copy(), contacts=[bool(x) for x in fc],
                   slip=slip, penetration_mm=_penetration_mm(m, d, self._fg), forward_progress=fwd)
        s = self.s
        if s is not None:
            from rrp.features.legged import EVENTS, active_event
            ev = active_event(s.runtime)
            out["phase"] = EVENTS[ev] if ev < len(EVENTS) else "done"
            if self.edit and self.edit != "none" and self.t_edit is not None:
                out["edit_active"] = bool(float(d.time) >= self.t_edit)
        ctl = self.ctl
        if ctl is not None and getattr(ctl, "packets", None):
            pk = ctl.packets[-1]
            pr = pk["probe"]
            from rrp.features.legged import EVENTS
            out["probe"] = dict(contact=[bool(x) for x in (pr["contact"][0] if pr.get("contact") else [])],
                                halt=bool(pr.get("subtask") == EVENTS.index("halt")), goal=pr.get("goal"),
                                subtask=pr.get("subtask"), fall=pr.get("fall"))
            zr = getattr(ctl, "record_z", None)
            if zr and self.basis is not None:
                z = zr[-1][2]
                if z.size == self.basis["dim"]:
                    out["packet_pca"] = RP.project_pca(self.basis, z)
        return out


LEGGED_NOTES = dict(
    joint_target="tracker joint-position targets actually applied (ctrl of the policy actuators)",
    joint_pos="measured policy-joint positions", contacts="foot-floor contact flags (privileged)",
    slip="per-foot contact-point horizontal speed (m/s, normal-force weighted; privileged)",
    penetration_mm="max foot-floor penetration (privileged)",
    forward_progress="base displacement along the heading at the start of the episode (m; privileged)",
    phase="active task event (public runtime)", edit_active="the task-context / packet edit is in force (t >= t_edit)",
    packet_pca="the latest packet (after any edit) on the bundle PCA basis",
    probe="the controller's own probe readout of that packet: contact per assembly at knot 0, halt = subtask==halt, "
          "goal (body frame, /2 m units as trained), subtask index, fall probability")


def _legged_ctl(a: dict, seed: int):
    import torch
    from rrp.evaluation.legged_latent_eval import BCController, LatentLeggedController
    dev = torch.device("cpu")
    if a.get("bc"):
        return BCController(Path(a["bc"]), dev, nfe=int(a.get("nfe", 8)), replan=int(a.get("replan", 5)), seed=seed)
    if a.get("flow") or a.get("rep"):
        return LatentLeggedController(Path(a["flow"]) if a.get("flow") else None, dev, nfe=int(a.get("nfe", 8)),
                                      edit=a.get("edit", "none"), t_edit=float(a.get("t_edit", 1.0)), seed=seed,
                                      posthoc_probe=a.get("posthoc_probe"), rep=a.get("rep"), realizer=a.get("realizer"))
    return None


def _run_legged(e: dict, out: Path, pcache: dict, robust: bool) -> list[dict]:
    import torch
    import rrp.envs.motion_quality as MQ
    from rrp.evaluation.legged_latent_eval import run_episode
    a = e["args"]
    torch.set_num_threads(int(e.get("threads", 1 if robust else 2)))
    basis = pca_basis(e.get("pca"), pcache)
    results = []
    for sd in e["seeds"]:
        sd = int(sd)
        ctl = _legged_ctl(a, sd)
        if ctl is not None and hasattr(ctl, "record_z") and basis is not None:
            ctl.record_z = []                            # generate() only appends (t, edit, z) here
        st = dict(ep=None, sig=None)
        o_init, o_tick, o_reset = MQ.LeggedMotionRecorder.__init__, MQ.LeggedMotionRecorder.on_tick, MQ.LeggedMotionRecorder.on_reset

        def init(self, session, *aa, **kk):
            o_init(self, session, *aa, **kk)
            st["ep"] = Episode(float(TRACKER_HZ()))
            st["sig"] = _LeggedSignals(session, session.binding, ctl=ctl, basis=basis,
                                       t_edit=float(a.get("t_edit", 1.0)), edit=a.get("edit", "none"))

        def on_reset(self, done):
            o_reset(self, done)
            if done and st["ep"] is not None:
                st["ep"].reset_frames()
                st["sig"].p0 = None

        def on_tick(self, d):
            o_tick(self, d)
            if not self.on or st["ep"] is None:
                return
            col = st["ep"].ensure(self.s.model)
            if col.due():
                col.frame(d, **st["sig"].frame(d))
                col.events(d.time, _statuses(self.s.runtime))
            col.tick()
        kw = dict(max_s=float(a.get("max_s", 60.0)))
        if robust:
            from rrp.evaluation.robustness import _cond_by_key
            cond = _cond_by_key("legged", e["body"], a["condition_key"])
            kw["perturb"] = cond["pert"]
        with _patched(MQ.LeggedMotionRecorder, "__init__", init), _patched(MQ.LeggedMotionRecorder, "on_tick", on_tick), \
                _patched(MQ.LeggedMotionRecorder, "on_reset", on_reset), torch.no_grad():
            row, _frames = run_episode(ctl, e["body"], sd, **kw)
        match = dict(e.get("recorded", {}).get("match_row") or {})
        rec = find_recorded(e.get("recorded"), sd, match)
        ep = st["ep"]
        meta = _meta(e, sd, ep.model, success=bool(row["success"]), failure_stage=row.get("failure_stage"),
                     ckpt_sha=_legged_shas(a), recorded=rec, rec_key="success",
                     compare=dict(fell=(row.get("fell"), rec.get("fell") if rec else None),
                                  sim_time=(round(row.get("sim_time", 0), 2), round(rec["sim_time"], 2) if rec and "sim_time" in rec else None),
                                  final_pose=(np.round(row["final_pose"], 3).tolist(),
                                              np.round(rec["final_pose"], 3).tolist() if rec and rec.get("final_pose") else None)))
        meta.update(edit=a.get("edit", "none"), t_edit=float(a.get("t_edit", 1.0)) if a.get("edit", "none") != "none" else None,
                    tracker=row.get("tracker"), tracker_sha256=row.get("tracker_sha256"), sim_time=row.get("sim_time"),
                    fell=row.get("fell"), row_source=row.get("source"), signal_notes=LEGGED_NOTES,
                    waypoints=row.get("waypoints"))
        if robust:
            meta["perturbation"] = row.get("perturbation")
        if basis is not None:
            meta["packet_pca_basis"] = basis
        ep.col.annotate(0.0, f"{e.get('source_label')} | {e['body']} seed {sd} | {e.get('condition')}")
        if meta.get("t_edit"):
            ep.col.annotate(meta["t_edit"], f"edit '{meta['edit']}' starts")
        results.append(_finish(e, sd, meta, ep, out))
        del ctl
    return results


def TRACKER_HZ():
    from rrp.envs.legged import TRACKER_HZ as T
    return T


def _legged_shas(a: dict) -> dict:
    return {k: _sha(a[k]) for k in ("flow", "bc", "rep", "posthoc_probe", "realizer") if a.get(k)}


def run_legged(e, out, pcache):
    return _run_legged(e, out, pcache, robust=False)


def run_legged_robust(e, out, pcache):
    return _run_legged(e, out, pcache, robust=True)


# ------------------------------------------------------------------ tracker validation (own mj_step loop)
def run_tracker_val(e: dict, out: Path, pcache: dict) -> list[dict]:
    import rrp.evaluation.tracker_validation as TV
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.legged_core import LeggedBinding
    from rrp.envs.legged_tracker import LearnedTracker, CPGTracker, tracker_path
    import mujoco
    a = e["args"]
    body = e["body"]
    model, _, meta_b = standalone_model(legged_body(body), contact=a.get("contact", "v1"))
    b = LeggedBinding(model, meta_b)
    actor = a.get("actor")
    if a.get("kind", "learned") == "learned":
        path = Path(actor) if actor else tracker_path(body, a.get("contact"))
        tracker = LearnedTracker(path, b, body)
        tsha = _sha(path)
    else:
        tracker, tsha = CPGTracker(b, meta_b), "scripted"
    sc = TV.scripts(b)[a["trial"]]
    results = []
    for s_i in e["seeds"]:
        sd = int(s_i)
        ep = Episode(50.0)                                   # tracker ticks at 50 Hz (dt 0.02)
        sig = _LeggedSignals(None, b)
        sub = max(1, int(round(0.02 / model.opt.timestep)))
        cnt = dict(n=0)

        def on_step(m, d):
            cnt["n"] += 1
            if cnt["n"] % sub:
                return
            col = ep.ensure(m)
            if col.due():
                col.frame(d, **sig.frame(d))
            col.tick()
        with _patched(TV, "mujoco", _Proxy(mujoco, on_step)):
            res = TV.run_episode(model, b, tracker, sc, sd, act=None)
        rec = find_recorded(e.get("recorded"), sd)
        e2 = dict(e)
        meta = _meta(e2, sd, model, success=not res["fell"], failure_stage="fell" if res["fell"] else None,
                     ckpt_sha=dict(tracker=tsha), recorded=rec, rec_key=None,
                     rec_success=(not rec["fell"]) if rec else None,
                     compare=dict(fell=(res["fell"], rec.get("fell") if rec else None),
                                  dist_m=(round(res["dist_m"], 3), round(rec["dist_m"], 3) if rec else None),
                                  slip_mps=(_r3(res.get("slip_mps")), _r3(rec.get("slip_mps")) if rec else None)))
        meta.update(trial=a["trial"], command=res["cmd"], tracker_source=tracker.source, tracker_version=tracker.version,
                    success_definition="no fall during the scripted command trial", result={k: res[k] for k in (
                        "dist_m", "mean_vx", "mean_wz", "slip_mps", "slip_ratio", "cot", "duty_factor", "swing_apex_m")},
                    signal_notes={k: v for k, v in LEGGED_NOTES.items() if k not in ("phase", "probe", "packet_pca",
                                                                                    "edit_active")})
        results.append(_finish(e2, sd, meta, ep, out))
    return results


def _r3(x):
    return None if x is None else round(float(x), 3)


# ------------------------------------------------------------------ DUAL teacher (rrp.evaluation.dual_teacher_quality)
def run_dual_teacher(e: dict, out: Path, pcache: dict) -> list[dict]:
    import rrp.data.dual_quality as DQ
    from rrp.evaluation.dual_teacher_quality import run_audit_episode
    a = e["args"]
    results = []
    for sd in e["seeds"]:
        sd = int(sd)
        st = dict(ep=None, objs=None, rb=None)
        o_init, o_after = DQ.DualQualityRecorder.__init__, DQ.DualQualityRecorder.after_step

        def init(self, session, teacher=None, *aa, **kk):
            o_init(self, session, teacher, *aa, **kk)
            st["ep"] = Episode(1.0 / session.dt)
            st["objs"] = sorted(self.obj_bodies)
            st["rb"] = set(self.body_robot)

        def after(self, *aa, **kk):
            r = o_after(self, *aa, **kk)
            s, d, m = self.s, self.s.data, self.s.model
            col = st["ep"].ensure(m)
            if col.due():
                tg, jp, tv = [], [], []
                for rr in s.robots:
                    c = rr.controller
                    if c.target:
                        tg += [float(x) for g in c.groups if g in c.target for x in np.atleast_1d(c.target[g])]
                    jp += d.qpos[rr.qadr].tolist()
                    tv += [bool(v >= 0.2) for v in s._touch_values(rr)]
                ob = st["objs"]
                pose = np.concatenate([np.concatenate([d.xpos[b], d.xquat[b]]) for b in ob]) if ob else None
                t = self.teacher
                col.frame(d, joint_target=tg, joint_pos=jp, contacts=tv, object_pose=pose,
                          penetration_mm=_penetration_mm(m, d, {g for g in range(m.ngeom) if m.geom_bodyid[g] in set(ob)},
                                                         st["rb"]),
                          phase=getattr(t, "phase_label", None) if t is not None else None)
                col.events(d.time, _statuses(s.runtime))
            col.tick()
            return r
        with _patched(DQ.DualQualityRecorder, "__init__", init), _patched(DQ.DualQualityRecorder, "after_step", after):
            row = run_audit_episode(a["task"], a["pair"], sd, max_steps=int(a.get("max_steps", 1200)),
                                    teacher_version=a.get("teacher_version"), noise=float(a.get("noise", 0.0)))
        if st["ep"] is None:
            results.append(dict(id=f"{e['id']}-s{sd}", skipped=row.get("status")))
            continue
        rec = find_recorded(e.get("recorded"), sd, {"pair": a["pair"]})
        meta = _meta(e, sd, st["ep"].model, success=row["status"] == "success", failure_stage=row.get("failure_phase"),
                     ckpt_sha=None, recorded=rec, rec_key=None,
                     rec_success=(rec.get("status") == "success") if rec else None,
                     compare=dict(status=(row.get("status"), rec.get("status") if rec else None),
                                  steps=(row.get("steps"), rec.get("steps") if rec else None),
                                  failure_phase=(row.get("failure_phase"), rec.get("failure_phase") if rec else None)))
        meta.update(pair=a["pair"], teacher_version=row.get("teacher_version") or "v2", gate=row.get("gate"),
                    penetration_max_m=row.get("penetration_max_m"), objects=[st["ep"].model.body(b).name for b in st["objs"]],
                    signal_notes=dict(joint_target="both robots' controller targets (all groups, robot order)",
                                      joint_pos="both robots' joint positions", contacts="finger touch sensors >= 0.2, robot order",
                                      object_pose="task objects (meta.objects), xyz + quat wxyz each (privileged)",
                                      penetration_mm="max object-robot penetration (privileged)",
                                      phase="scripted teacher phase label L:<left>|R:<right>"))
        results.append(_finish(e, sd, meta, st["ep"], out))
    return results


# ------------------------------------------------------------------ GRASP RIG (rrp.evaluation.grasp_rig, own mj_step loop)
def run_grasp_rig(e: dict, out: Path, pcache: dict) -> list[dict]:
    import mujoco
    import rrp.evaluation.grasp_rig as GR
    a = e["args"]
    st = dict(ep=None, sig=None, n=0, mass=None)

    def on_step(m, d):
        if st["ep"] is None:
            st["ep"] = Episode(1.0 / m.opt.timestep)
            cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
            palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r0_palm")
            cg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
            pads = [g for g in range(m.ngeom) if GR.GC.PAD_RE.search(m.geom(g).name or "")]
            jn = [j for j in range(m.njnt) if m.jnt_type[j] in (2, 3)]
            st["sig"] = dict(cube=cube, palm=palm, cg=cg, pads=pads, qadr=[int(m.jnt_qposadr[j]) for j in jn], rel0=None)
        col = st["ep"].ensure(m)
        g = st["sig"]
        if col.due():
            rel = d.xpos[g["cube"]] - d.xpos[g["palm"]]
            if g["rel0"] is None:
                g["rel0"] = rel.copy()
            touch = set()
            for i in range(d.ncon):
                c = d.contact[i]
                if g["cg"] in (c.geom1, c.geom2):
                    touch.add(c.geom1 if c.geom2 == g["cg"] else c.geom2)
            t = float(d.time)
            ph = "settle" if t < 0.2 else "close" if t < 1.0 else "lift" if t < 1.5 else "hold" if t < 2.0 else "mass_ramp"
            col.frame(d, joint_target=d.ctrl.copy(), joint_pos=d.qpos[g["qadr"]], contacts=[p in touch for p in g["pads"]],
                      object_pose=np.concatenate([d.xpos[g["cube"]], d.xquat[g["cube"]]]),
                      penetration_mm=_penetration_mm(m, d, {g["cg"]}, {int(m.geom_bodyid[p]) for p in g["pads"]}),
                      slip=float(np.linalg.norm(rel - g["rel0"])) * 1000.0, phase=ph)
            mass = round(float(m.body_mass[g["cube"]]), 4)
            if mass != st["mass"]:
                col.annotate(t, f"cube mass {mass:.3f} kg")
                st["mass"] = mass
        col.tick()
    with _patched(GR, "mujoco", _Proxy(mujoco, on_step)):
        res = GR.run(a["version"], a["gripper"], float(a.get("friction_scale", 1.0)), yaw=math.radians(float(a.get("yaw_deg", 0.0))))
    rec = _rig_recorded(e.get("recorded"), a)
    sd = 0
    meta = _meta(e, sd, st["ep"].model, success=bool(res["lift_held"]), failure_stage=None if res["lift_held"] else "lift",
                 ckpt_sha=None, recorded=rec, rec_key="lift_held",
                 compare=dict(lift_held=(res["lift_held"], rec.get("lift_held") if rec else None),
                              slip_mass_kg=(_r3(res.get("slip_mass_kg")), _r3(rec.get("slip_mass_kg")) if rec else None),
                              pen_hold_mm=(_r3(res.get("pen_hold_mm")), _r3(rec.get("pen_hold_mm")) if rec else None)))
    meta.update(result=res, success_definition="cube held through the 10 cm lift", signal_notes=dict(
        joint_target="rig actuator controls (lift, grip)", joint_pos="rig joints (lift slide, finger joints)",
        contacts="pad geoms touching the cube (rig pad order)", object_pose="cube xyz + quat wxyz",
        penetration_mm="max pad-cube penetration", slip="cube displacement relative to the palm since the first frame (mm)",
        phase="rig schedule: settle 0.2 s, close 0.8 s, lift 0.5 s, hold 0.5 s, then mass ramp x1.08 / 0.25 s to slip"))
    return [_finish(e, sd, meta, st["ep"], out)]


def _rig_recorded(rec, a):
    if not rec or not Path(rec.get("file", "")).exists():
        return None
    for r in _read_rows(rec["file"]):
        if (r.get("version") in (a["version"], f"grasp_{a['version']}") and r.get("gripper") == a["gripper"]
                and abs(float(r.get("friction_scale", 1.0)) - float(a.get("friction_scale", 1.0))) < 1e-9
                and abs(float(r.get("cube_yaw_deg", 0.0) or 0.0) - float(a.get("yaw_deg", 0.0))) < 1e-6):
            return r
    return None


# ------------------------------------------------------------------ shared: meta, finish
def _meta(e: dict, seed: int, model, *, success, failure_stage, ckpt_sha, recorded, rec_key, compare, rec_success=None) -> dict:
    fam = e["family"]
    if recorded is not None and rec_key is not None:
        rec_success = bool(recorded.get(rec_key))
    reproduced = None if rec_success is None else bool(rec_success == success)
    m = dict(family=fam, task=e.get("task"), body=e.get("body"), route=e.get("route"), source_label=e["source_label"],
             ckpt_sha=ckpt_sha, variant=e.get("variant"), seed=seed, condition=e.get("condition"), success=bool(success),
             failure_stage=failure_stage, physics=physics_meta(model, family=fam), decision_refs=list(e["decision_refs"]),
             recorded_success=rec_success, reproduced=reproduced,
             reproduction=dict(recorded_file=(e.get("recorded") or {}).get("file"), found=recorded is not None,
                               compared={k: dict(rerun=v[0], recorded=v[1]) for k, v in (compare or {}).items()}),
             harness=e["harness"], args=e.get("args"), env={k: v for k, v in (e.get("env") or {}).items()},
             recorded_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), git_sha=_git_sha(),
             recorder="rrp.viz.record/v1")
    for k in ("caveat", "group", "label"):
        if e.get(k):
            m[k] = e[k]
    return m


def _finish(e: dict, seed: int, meta: dict, ep: Episode, out: Path) -> dict:
    rid = f"{e['id']}-s{seed}"
    doc = RP.build_replay(rid, meta, ep.geoms, ep.col.result())
    p = RP.write_replay(doc, out / e["family"])
    return dict(id=rid, file=str(p), bytes=p.stat().st_size, n_frames=doc["n_frames"], success=meta["success"],
                recorded_success=meta["recorded_success"], reproduced=meta["reproduced"])


HARNESSES = dict(ladder=run_ladder, arm_teacher=run_arm_teacher, arm_edit=run_arm_edit, legged=run_legged,
                 legged_robust=run_legged_robust, tracker_val=run_tracker_val, dual_teacher=run_dual_teacher,
                 grasp_rig=run_grasp_rig)
REQUIRED = ("id", "harness", "family", "source_label", "decision_refs", "seeds")


# ------------------------------------------------------------------ spec + CLI
def load_spec(path) -> list[dict]:
    """One spec file, or several comma-separated (entries concatenated; ids must stay unique)."""
    from rrp.orchestration.yamlmini import load
    out = []
    for one in str(path).split(","):
        out += _load_one(load(Path(one).read_text()))
    ids = [e["id"] for e in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate entry ids")
    return out


def _load_one(doc: dict) -> list[dict]:
    defaults = doc.get("defaults") or {}
    out = []
    for e in doc["entries"]:
        e = dict(defaults, **e)
        e["env"] = dict(defaults.get("env") or {}, **(e.get("env") or {}))
        miss = [k for k in REQUIRED if k not in e]
        if miss:
            raise ValueError(f"spec entry {e.get('id')}: missing {miss}")
        if e["harness"] not in HARNESSES:
            raise ValueError(f"spec entry {e['id']}: unknown harness {e['harness']}")
        out.append(e)
    return out


def run_entry(e: dict, out: Path) -> list[dict]:
    t0 = time.time()
    res = HARNESSES[e["harness"]](e, out, {})
    for r in res:
        r["wall_s"] = round(time.time() - t0, 1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma list of entry ids")
    ap.add_argument("--shard", default=None, help="i/n: every n-th entry starting at i")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inproc", action="store_true", help="run the selected entries in this process (child mode)")
    ap.add_argument("--skip-done", action="store_true", help="skip entries whose ids are already in <out>/record_log.jsonl")
    ap.add_argument("--index-only", action="store_true", help="just (re)write <out>/index.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.index_only:
        idx = RP.write_index(out, _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), _git_sha())
        print(json.dumps(dict(n=idx["n"], total_bytes=idx["total_bytes"])))
        return 0
    entries = load_spec(a.spec)
    if a.only:
        keep = set(a.only.split(","))
        entries = [e for e in entries if e["id"] in keep]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        entries = entries[i::n]
    if a.list:
        for e in entries:
            print(e["id"], e["harness"], e.get("seeds"))
        return 0
    if os.environ.get("RRP_NODE") != "peer" and os.environ.get("RRP_VIZ_RECORD_ALLOW_HOST") != "1":
        sys.exit("rrp.viz.record simulates: run it on the PEER through the broker (scripts/peer_run.sh; D-115/D-127)")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "record_log.jsonl"
    done = set()
    if a.skip_done and log.exists():
        done = {json.loads(l)["entry"] for l in log.read_text().splitlines() if l.strip() and json.loads(l).get("ok")}
    if a.inproc:
        rc = 0
        for e in entries:
            try:
                res = run_entry(e, out)
                rec = dict(entry=e["id"], ok=True, replays=res)
            except Exception as ex:  # noqa: BLE001 - recorded, the next entry still runs
                import traceback
                rec = dict(entry=e["id"], ok=False, error=repr(ex)[:500], tb=traceback.format_exc()[-2000:])
                rc = 1
            with open(log, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            print(json.dumps({k: v for k, v in rec.items() if k != "tb"}, default=str), flush=True)
        return rc
    rc = 0
    for e in entries:
        if e["id"] in done:
            print(f"skip {e['id']} (done)", flush=True)
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(e.get("threads", 2)),
                   MKL_NUM_THREADS=str(e.get("threads", 2)), **{k: str(v) for k, v in (e.get("env") or {}).items()})
        cmd = [sys.executable, "-m", "rrp.viz.record", "--spec", a.spec, "--out", a.out, "--only", e["id"], "--inproc"]
        t0 = time.time()
        r = subprocess.run(cmd, env=env)
        print(f"[record] {e['id']} rc={r.returncode} {time.time() - t0:.0f}s", flush=True)
        rc = rc or r.returncode
    RP.write_index(out, _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), _git_sha())
    return rc


if __name__ == "__main__":
    sys.exit(main())
