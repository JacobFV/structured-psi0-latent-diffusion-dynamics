"""PEER-ONLY replay recorder for the live visualization room (viz/CONTRACT.md "replay file", D-131).

    python -m rrp.cli viz record --spec viz/specs/<name>.yaml --out <dir> [--only ID[,ID]] [--shard i/n] [--list]

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
    from rrp.bodies.contact import model_contact_version
    from rrp.bodies.grasp_contact import model_grasp_version
    from rrp.bodies.actuator import model_actuator_limits, resolve_mode
    arm = family in ("arm", "dual", "grasp_rig")
    return dict(contact_version=None if arm else (model_contact_version(model) or "contact_v1"),
                grasp_contact_version=(model_grasp_version(model) or "grasp_v1") if arm else None,
                actuator_limits_version=model_actuator_limits(model), actuator_mode=resolve_mode())


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
            import mujoco
            self.model = model
            self.geoms, bodies = RP.export_geoms(model)
            ex = self.extra or {}
            want = list(ex.get("contact_bodies") or []) + [ex.get("base_body"), ex.get("object_body")]
            for n in want:                    # bodies named in meta get frames even without visible geoms
                if n and n not in bodies and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0:
                    bodies.append(n)
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


def _sensor_bodies(m, names) -> list:
    """Body of each (site-attached) sensor, for meta.contact_bodies (contract v1.1)."""
    import mujoco
    out = []
    for n in names:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
        if sid < 0:
            out.append(None)
            continue
        oid, ot = int(m.sensor_objid[sid]), int(m.sensor_objtype[sid])
        b = int(m.site_bodyid[oid]) if ot == int(mujoco.mjtObj.mjOBJ_SITE) else (oid if ot == int(mujoco.mjtObj.mjOBJ_BODY) else -1)
        out.append(m.body(b).name if b >= 0 else None)
    return out


def _qadr_joint_names(m, qadr) -> list:
    by = {int(m.jnt_qposadr[j]): m.joint(j).name for j in range(m.njnt)}
    return [by.get(int(a)) for a in qadr]


# ------------------------------------------------------------------ rich signals (v1.2): pure reads of mjData
def _subtrees(m, roots) -> list[set]:
    """Body id sets: each root with all its descendants."""
    par = [int(x) for x in m.body_parentid]
    out = []
    for r in roots:
        g = set()
        for b in range(m.nbody):
            p = b
            while True:
                if p == r:
                    g.add(b)
                    break
                if p == 0:
                    break
                p = par[p]
        out.append(g)
    return out


def _contact_groups(m, d, groups, *, other=None, exclude=None):
    """Per body group: [normal N, tangential N] summed over contacts between a group body and a body in `other`
    (or any body not in `exclude`), and the normal-force-weighted contact point (None without contact)."""
    import mujoco
    n = len(groups)
    fn, ft, pw = np.zeros(n), np.zeros(n), np.zeros((n, 3))
    owner = {}
    for k, g in enumerate(groups):
        for b in g:
            owner.setdefault(b, k)
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        for a, o in ((b1, b2), (b2, b1)):
            k = owner.get(a)
            if k is None or (other is not None and o not in other) or (exclude is not None and o in exclude):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            nn = abs(float(f6[0]))
            fn[k] += nn
            ft[k] += math.hypot(f6[1], f6[2])
            pw[k] += nn * np.asarray(c.pos)
            break
    return ([[fn[k], ft[k]] for k in range(n)],
            [(pw[k] / fn[k]) if fn[k] > 1e-9 else None for k in range(n)])


def _body_vel(m, d, b) -> np.ndarray:
    """[vx, vy, vz, wx, wy, wz] of body b's frame origin, world orientation."""
    import mujoco
    v = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(b), v, 0)
    return np.concatenate([v[3:], v[:3]])


def _power(d, act_ids) -> float:
    """Mechanical power (W) = sum |actuator force x actuator velocity| over the given actuators."""
    a = np.asarray(act_ids, int)
    return float(np.sum(np.abs(d.actuator_force[a] * d.actuator_velocity[a]))) if len(a) else 0.0


class _Energy:
    """Integrates |mechanical power|; cost of transport = energy / (m g horizontal path)."""

    def __init__(self, act_ids, mass=None, base_qadr=None):
        self.act, self.mass, self.qa = np.asarray(act_ids, int), mass, base_qadr
        self.reset()

    def reset(self):
        self.e, self.path, self.p_last, self.last_p = 0.0, 0.0, None, 0.0

    def step(self, d, dt):
        self.last_p = _power(d, self.act)
        self.e += self.last_p * dt

    def move(self, d):
        if self.qa is None:
            return
        p = d.qpos[self.qa:self.qa + 2].copy()
        if self.p_last is not None:
            self.path += float(np.linalg.norm(p - self.p_last))
        self.p_last = p

    def cot(self):
        if self.mass is None or self.path < 0.2:
            return None
        return self.e / (self.mass * 9.81 * self.path)


def _packet_summary(z, head: int = 8):
    """z [knots, assemblies, dz] -> (per knot x assembly L2 norm, the first `head` dims of each)."""
    z = np.asarray(z, np.float32)
    if z.ndim != 3:
        z = z.reshape(z.shape[0], -1, z.shape[-1])
    return np.linalg.norm(z, axis=-1), z[..., :head]


def _robot_act_ids(r) -> np.ndarray:
    c = r.controller
    return np.concatenate([np.atleast_1d(np.asarray(c.act_ids[g], int)) for g in c.groups if g in c.act_ids]) \
        if c.act_ids else np.zeros(0, int)


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
        from rrp.policies.bundles import load_rep
        from rrp.harness.train.legged_latent_train import LeggedData
        rcfg, E, _R, _P, _res = load_rep(Path(spec["rep"]), torch.device("cpu"))
        data = LeggedData(Path(rcfg["data"]), [spec["body"]], torch.device("cpu"))
        M = int(spec.get("M") or data.S["asm_mask"][0].sum())            # the body's packet assemblies
        zs = []
        with torch.no_grad():
            for _ in range(max(1, n // 256)):
                i = torch.from_numpy(rng.choice(data.train_idx, 256)).to(torch.device("cpu"))
                mu, _ = E(data.ctx_batch(i), data.beh(i))
                zs.append(mu[:, :, :M].reshape(len(i), -1).numpy())
        b = RP.fit_pca(np.concatenate(zs))
        b.update(fit_source=dict(kind=kind, rep=spec["rep"], data=str(rcfg["data"]), body=spec["body"], M=M,
                                 note="mu = E(ctx, demonstrated targets) on training rows (the flow's targets)"))
        del data
    else:
        raise ValueError(f"unknown pca kind {kind}")
    cache[key] = b
    return b


# ------------------------------------------------------------------ relation-factor maps (D-144 R21, viz/CONTRACT.md)
FACTORMAP_SCHEMA = "rrp-viz/factormap/v1"


def record_factor_maps(replay_id: str, entries: list[dict], specs=()) -> dict:
    """`FactorSite.contributions(rc, xq, xk)` (per-factor `[B,H,Q,K]` logit terms, `relations/ops.py` 3.5) at chosen
    steps, into one factor-map record: its own schema, not the replay v1 schema (a per-factor per-head logit map is
    not a fixed-shape per-frame signal like the others in `FRAME_SIGNALS`). A net that runs its attention through
    `FactorSite` calls this with the steps it chose to keep (host-light: this function does not subsample on its own
    — a full-episode recording at every tick would be large; callers keep a stride, e.g. every packet).

    `entries`: `[{t, site, contributions: {factor_name: array-like [B,H,Q,K] or [H,Q,K]}}]`, one per recorded step
    (`site` = the `FactorSite`'s `"q>k"`, e.g. `"ctx>ctx"`). `specs`: the run's resolved `FactorSpec`s (its
    `factors:` list), turned into `rrp.policies.relations.base.provenance` (name, op, form, source, control,
    privileged) so the room can badge each map by its source without re-deriving it. Only batch element 0 is kept
    (one recorded episode); values rounded to 4 dp, like every other recorded signal (`rrp.viz.replay.r4`)."""
    from rrp.policies.relations.base import provenance as factor_provenance
    steps = []
    for e in entries:
        factors = {}
        for name, arr in e["contributions"].items():
            a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
            if a.ndim == 4:                                      # [B,H,Q,K] -> the recorded episode's one batch element
                a = a[0]
            factors[name] = RP.r4(a)                              # [H,Q,K]
        steps.append(dict(t=round(float(e["t"]), 4), site=str(e["site"]), factors=factors))
    return dict(schema=FACTORMAP_SCHEMA, id=replay_id, provenance=factor_provenance(specs), steps=steps)


def write_factor_map(doc: dict, out_dir: Path) -> Path:
    """`<out_dir>/<id>.factormap.json.gz`, alongside the replay of the same id; the room's generic `/api/replay/<id>`
    file walk (viz/CONTRACT.md, `rrp-api-plugin.ts`) serves it at `/api/replay/<id>.factormap` with no plugin change."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{doc['id']}.factormap.json.gz"
    raw = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
    with gzip.open(p, "wb", compresslevel=9) as fh:
        fh.write(raw)
    return p


def read_factor_map(p: Path) -> dict:
    p = Path(p)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as fh:
        return json.load(fh)


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
        self.act_ids = _robot_act_ids(self.r)
        tb = [b for b in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) if n else -1
                          for n in _sensor_bodies(m, self.r.touch))]
        self.finger_groups = _subtrees(m, [b for b in tb if b >= 0]) if tb and all(b >= 0 for b in tb) else []
        self.energy = _Energy(self.act_ids)
        self.slot_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o.sim_body) for o in s.detectables]
        self.last_pk, self.cf_dz = None, None

    def meta(self) -> dict:
        m, r = self.s.model, self.r
        c = r.controller
        tg = c.target or {}
        tnames = [f"{g}[{i}]" for g in c.groups if g in tg for i in range(np.atleast_1d(np.asarray(tg[g])).size)]
        base = next((n for l in r.spec.links for n in ((r.prefix or "") + l.name, l.name)
                     if n in {m.body(b).name for b in self.robot_bodies}), None)
        return dict(contact_bodies=_sensor_bodies(m, r.touch), joint_names=_qadr_joint_names(m, r.qadr),
                    joint_target_names=tnames, object_body="cube", base_body=base,
                    joint_torque_names=[m.actuator(int(i)).name for i in self.act_ids])

    def joint_target(self):
        c = self.r.controller
        tg = c.target or {}
        return np.concatenate([np.atleast_1d(np.asarray(tg[g], float)) for g in c.groups if g in tg]) if tg else None

    def frame(self, packet=None, basis=None, phase=None, dt=0.05, edit_active=None, col=None) -> dict:
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
        # v1.2 rich signals
        self.energy.step(d, dt)                       # sampled at recorded frames (control-tick resolution)
        out.update(joint_vel=d.qvel[self.r.dadr].copy(), actuator_force=d.actuator_force[self.act_ids].copy(),
                   object_vel=_body_vel(m, d, self.cube), power_w=self.energy.last_p, energy_j=self.energy.e)
        if self.finger_groups:
            f, pos = _contact_groups(m, d, self.finger_groups, exclude=self.robot_bodies)
            out.update(contact_force=f, contact_pos=pos)
        w = s._grip_width(self.r)
        if w is not None:
            out["gripper_aperture"] = float(w)
        held_by = s.truth().held_by                   # privileged (display only)
        held = {b for v in held_by.values() for b in v}
        touch_cube = self._touching(self.cube)
        out["grasp_state"] = "held" if "cube" in held else ("contact" if touch_cube else "free")
        if packet is not None:
            z = np.asarray(packet.z, np.float32)
            if basis is not None and z.size == basis["dim"]:
                out["packet_pca"] = RP.project_pca(basis, z)
            nrm, headv = _packet_summary(z)
            out["packet_norm"] = nrm
            out["packet_z"] = headv.reshape(-1)
            if edit_active is not None:
                out["edit_dz_norm"] = self.cf_dz if edit_active else None
            if col is not None and packet is not self.last_pk:
                col.add_sparse("packet_events", d.time, z_norm=nrm, source=getattr(packet, "source", None),
                               edit_dz_norm=self.cf_dz if edit_active else None)
            self.last_pk = packet
            if self.P is not None:
                out["probe"] = self.probe(packet, z)
                names = [m.body(b).name if b >= 0 else None for b in self.slot_bodies]
                out["probe_truth"] = dict(held_by=[n in held for n in names],
                                          contact=[self._touching(b) if b >= 0 else False for b in self.slot_bodies])
        return out

    def _touching(self, body) -> bool:
        d, m = self.s.data, self.s.model
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 == body and b2 in self.robot_bodies) or (b2 == body and b1 in self.robot_bodies):
                return True
        return False

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
    from rrp.policies.system0 import LatentSystem0
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
    from rrp.harness.eval.ladder import LadderConfig, load_models, run_ladder as _run
    from rrp.harness.eval.robustness import feasible_arm_seeds
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
                    eps[k][0].extra = eps[k][1].meta()
                ep, sig = eps[k]
                col = ep.ensure(s.model)
                if col.due():
                    pk = tap.get(id(getattr(s, "_rrp_featurizer", None))) if P is not None or basis is not None else None
                    col.frame(s.data, **sig.frame(packet=pk, basis=basis, phase=phase, dt=s.dt * col.stride, col=col))
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
          "goal = goal-effect head xy for the cube slot (decimetre units as trained)",
    probe_truth="PRIVILEGED, DISPLAY ONLY: per detector slot, held_by = the object is held (>= 2 hand bodies touching), "
                "contact = any robot body touches it (aligned with probe held_by / contact)",
    joint_vel="arm+gripper joint velocities", actuator_force="actuator forces of the controller groups (arm, gripper)",
    contact_force="per finger (touch-sensor body subtree, meta.contact_bodies) [normal, tangential] N against non-robot "
                  "bodies (privileged)", contact_pos="per finger force-weighted contact point (world m; privileged)",
    object_vel="cube [vx, vy, vz, wx, wy, wz] world (privileged)",
    power_w="sum |actuator force x actuator velocity| sampled at recorded frames (W)",
    energy_j="frame-sampled integral of power_w (J; control-tick resolution, an approximation)",
    gripper_aperture="public gripper width sensor (m), where the gripper has one",
    grasp_state="PRIVILEGED: held (held truth) / contact (robot touches the cube) / free",
    packet_norm="L2 norm of the executed packet per knot x assembly",
    edit_dz_norm="|edited packet - unedited packet| with the same noise key at the same state (edit replays)")


def _ckpt_shas(ids: dict) -> dict:
    out = {}
    for k, v in (ids or {}).items():
        if isinstance(v, dict) and v.get("sha256"):
            out[k] = v["sha256"]
    return out


# ------------------------------------------------------------------ ARM: scripted teacher v2 (rrp.evaluation.teacher_quality)
def run_arm_teacher(e: dict, out: Path, pcache: dict) -> list[dict]:
    from rrp.envs.mujoco import session as native
    from rrp.harness.eval.teacher_quality import run_quality_episode
    import rrp.policies.teachers.arm_smooth as AS
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
            st["ep"].extra = st["sig"].meta()
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
                    col.frame(self.data, **st["sig"].frame(phase=getattr(st["teacher"], "phase", None), dt=self.dt * col.stride,
                                                           col=col))
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
    from rrp.policies.bundles import load_representation
    from rrp.policies.latent import LatentPolicy
    from rrp.harness.eval import latent_causal as lc
    from rrp.harness.eval import latent_semantic_edits as se
    from rrp.policies.nets.checkpoint import load_checkpoint
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
    cf: dict = {}
    o_packet = se.GeneratedSource.packet

    def packet(self, s, cond, teacher, goal_off, key=None):
        """The edited packet as the harness computes it, plus (for display) the UNEDITED packet with the same noise
        key at the same state: orig() snapshots, edits the context, samples with noise_keys=[key] and restores."""
        p = o_packet(self, s, cond, teacher, goal_off, key)
        cf["dz"] = None
        if cond not in ("control", "control_replay") and key is not None:
            p0 = o_packet(self, s, "control", None, goal_off, key)
            cf["dz"] = float(np.linalg.norm(np.asarray(p.z) - np.asarray(p0.z)))
        return p
    for sd in e["seeds"]:
        sd = int(sd)
        for cond in e.get("conditions", ["control"]):
            ep = [None, None]

            def on_step(s, step):
                if ep[0] is None:
                    ep[0], ep[1] = Episode(1.0 / s.dt), _ArmSignals(s, P)
                    ep[0].extra = ep[1].meta()
                col = ep[0].ensure(s.model)
                if col.due():
                    pk = tap.get(id(src.featurizer(s)))
                    ep[1].cf_dz = cf.get("dz")
                    col.frame(s.data, **ep[1].frame(packet=pk, basis=basis, dt=s.dt * col.stride,
                                                   edit_active=cond not in ("control",), col=col))
                    col.events(s.data.time, _statuses(s.runtime))
                col.tick()
            with _patched(*patch[:2], patch[2]), _patched(se.GeneratedSource, "packet", packet), torch.no_grad():
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
                        checkpoints=ids, signal_notes=dict(ARM_NOTES, edit_active="the edited task context is in force (from t=0)"),
                        edit_onset_t=0.0 if cond != "control" else None)
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
        m = b.model
        self.floor_bodies = {int(m.geom_bodyid[g]) for g in self._fg}
        self.feet = _subtrees(m, [int(x) for x in b.foot_bids])
        self.energy = _Energy(b.pol_act, mass=float(m.body_subtreemass[b.root_bid]), base_qadr=b.qa)
        self.packets: list = []            # (t, edit, z [K,M,dz] after any edit, |edited - unedited| or None)
        self._n_pk = 0

    def meta(self) -> dict:
        b, m = self.b, self.b.model
        names = [m.joint(int(j)).name for j in m.actuator_trnid[b.pol_act, 0]]
        return dict(contact_bodies=[m.body(int(x)).name for x in b.foot_bids], base_body=m.body(int(b.root_bid)).name,
                    joint_names=names, joint_target_names=names,
                    joint_torque_names=[m.actuator(int(i)).name for i in b.pol_act])

    def start(self, d):
        from rrp.envs.mujoco.legged_core import yaw_of
        b = self.b
        self.p0 = d.qpos[b.qa:b.qa + 2].copy()
        self.yaw0 = yaw_of(d.qpos[b.qa + 3:b.qa + 7])

    def frame(self, d, col=None) -> dict:
        b, m = self.b, self.b.model
        if self.p0 is None:
            self.start(d)
        fc, _fn, slip, _bad = b.stance(d)
        dp = d.qpos[b.qa:b.qa + 2] - self.p0
        fwd = math.cos(self.yaw0) * dp[0] + math.sin(self.yaw0) * dp[1]
        out = dict(joint_target=d.ctrl[b.pol_act].copy(), joint_pos=d.qpos[b.pol_qadr].copy(), contacts=[bool(x) for x in fc],
                   slip=slip, penetration_mm=_penetration_mm(m, d, self._fg), forward_progress=fwd)
        f, pos = _contact_groups(m, d, self.feet, other=self.floor_bodies)
        out.update(joint_vel=d.qvel[b.pol_dadr].copy(), actuator_force=d.actuator_force[b.pol_act].copy(),
                   contact_force=f, contact_pos=pos, base_vel=_body_vel(m, d, b.root_bid),
                   power_w=_power(d, b.pol_act), energy_j=self.energy.e, cot=self.energy.cot())
        s = self.s
        ev = None
        if s is not None:
            from rrp.policies.features.legged import EVENTS, active_event
            ev = active_event(s.runtime)
            out["phase"] = EVENTS[ev] if ev < len(EVENTS) else "done"
            if self.edit and self.edit != "none" and self.t_edit is not None:
                out["edit_active"] = bool(float(d.time) >= self.t_edit)
        ctl = self.ctl
        if ctl is not None and getattr(ctl, "packets", None):
            pk = ctl.packets[-1]
            pr = pk["probe"]
            from rrp.policies.features.legged import EVENTS
            out["probe"] = dict(contact=[bool(x) for x in (pr["contact"][0] if pr.get("contact") else [])],
                                halt=bool(pr.get("subtask") == EVENTS.index("halt")), goal=pr.get("goal"),
                                subtask=pr.get("subtask"), fall=pr.get("fall"))
            if ev is not None:                      # privileged truth for the same readouts (display only)
                from rrp.envs.mujoco.legged_core import yaw_of
                wps = s.scenario.meta.get("waypoints") or {}
                q = d.qpos
                x, y, yaw = q[b.qa], q[b.qa + 1], yaw_of(q[b.qa + 3:b.qa + 7])
                goal = None
                if ev < 2 and wps:
                    wx, wy = wps["a" if ev == 0 else "b"][:2]
                    c_, s_ = math.cos(yaw), math.sin(yaw)
                    goal = [(c_ * (wx - x) + s_ * (wy - y)) / 2.0, (-s_ * (wx - x) + c_ * (wy - y)) / 2.0]
                out["probe_truth"] = dict(contact=[bool(x_) for x_ in fc], halt=bool(ev == EVENTS.index("halt")),
                                          goal=goal, subtask=int(min(ev, 3)))
        if self.packets:
            t_, ed, z, dz = self.packets[-1]
            if self.basis is not None and z.size == self.basis["dim"]:
                out["packet_pca"] = RP.project_pca(self.basis, z)
            nrm, headv = _packet_summary(z)
            out["packet_norm"] = nrm
            out["packet_z"] = headv.reshape(-1)
            if self.edit and self.edit != "none":
                out["edit_dz_norm"] = dz
            if col is not None:
                for t_, ed, z, dz in self.packets[self._n_pk:]:
                    nrm, headv = _packet_summary(z)
                    col.add_sparse("packet_events", t_, z_norm=nrm, edit=ed, edit_dz_norm=dz)
                self._n_pk = len(self.packets)
        return out


LEGGED_NOTES = dict(
    joint_target="tracker joint-position targets actually applied (ctrl of the policy actuators)",
    joint_pos="measured policy-joint positions", contacts="foot-floor contact flags (privileged)",
    slip="per-foot contact-point horizontal speed (m/s, normal-force weighted; privileged)",
    penetration_mm="max foot-floor penetration (privileged)",
    forward_progress="base displacement along the heading at the start of the episode (m; privileged)",
    phase="active task event (public runtime)", edit_active="the task-context / packet edit is in force (t >= t_edit)",
    packet_pca="the latest packet (after any edit) on the bundle PCA basis",
    joint_vel="policy-joint velocities (rad/s)", actuator_force="policy actuator forces/torques (N m)",
    contact_force="per foot [normal, tangential] contact force with the floor (N; privileged)",
    contact_pos="per foot normal-force-weighted floor contact point (world m; null in swing; privileged)",
    base_vel="root body [vx, vy, vz, wx, wy, wz] (world frame; privileged)",
    power_w="sum |actuator force x actuator velocity| at the frame (W)",
    energy_j="integral of that power over every physics substep since the episode reset (J)",
    cot="cost of transport energy / (m g horizontal path), null until 0.2 m travelled",
    packet_norm="L2 norm of the latest packet per knot x assembly (heatmap)",
    edit_dz_norm="|edited packet - unedited packet| with the SAME flow noise at the same state (context edits; "
                 "for z edits the controller's own dz_norm); null before the edit",
    probe_truth="PRIVILEGED, DISPLAY ONLY: true foot contact (feet in assembly order), true halt (active event is halt), "
                "true body-frame goal to the active waypoint (/2 m as the probe), true subtask index",
    probe="the controller's own probe readout of that packet: contact per assembly at knot 0, halt = subtask==halt, "
          "goal (body frame, /2 m units as trained), subtask index, fall probability")


def _legged_ctl(a: dict, seed: int):
    import torch
    from rrp.policies.legged import BCController, LatentLeggedController
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
    import rrp.envs.mujoco.motion_quality as MQ
    from rrp.harness.eval.legged_latent_eval import run_episode
    a = e["args"]
    torch.set_num_threads(int(e.get("threads", 1 if robust else 2)))
    basis = pca_basis(e.get("pca"), pcache)
    results = []
    for sd in e["seeds"]:
        sd = int(sd)
        ctl = _legged_ctl(a, sd)
        st = dict(ep=None, sig=None)
        o_init, o_tick, o_reset = MQ.LeggedMotionRecorder.__init__, MQ.LeggedMotionRecorder.on_tick, MQ.LeggedMotionRecorder.on_reset
        o_sub = MQ.LeggedMotionRecorder.on_substep
        from rrp.policies.legged import LatentLeggedController as LLC
        o_gen = LLC.generate
        CTX = ("mirror_goal", "halt", "mirror_active", "mirror_inactive")

        def generate(self, ad, now):
            """The harness's generate(); for context edits also the UNEDITED packet with a copy of the generator
            state taken before the call (same noise), for |edited - unedited| (display only; the harness's own
            generator is untouched)."""
            edit = self.edit if now >= self.t_edit else "none"
            need = (edit in CTX and self.F is not None and self.bc is None and getattr(self, "oracle", None) is None)
            if need:
                gs = self.gen.get_state()
                b0 = ad.dyn_batch()
            p = o_gen(self, ad, now)
            dz = None
            if need:
                g2 = torch.Generator(device=self.dev)
                g2.set_state(gs)
                b0["ctx"] = self._ctx(ad)
                z0 = self.F.sample(b0, nfe=self.nfe, generator=g2)[0, :, :self.morph.M].cpu().numpy()
                dz = float(np.linalg.norm(np.asarray(p.z) - z0))
            elif edit != "none" and self.packets:
                dz = self.packets[-1].get("dz_norm")
            if st["sig"] is not None:
                st["sig"].packets.append((float(now), edit, np.asarray(p.z, np.float32), dz))
            return p

        def on_substep(self, d, dt):
            o_sub(self, d, dt)
            if self.on and st["sig"] is not None:
                st["sig"].energy.step(d, dt)

        def init(self, session, *aa, **kk):
            o_init(self, session, *aa, **kk)
            st["ep"] = Episode(float(TRACKER_HZ()))
            st["sig"] = _LeggedSignals(session, session.binding, ctl=ctl, basis=basis,
                                       t_edit=float(a.get("t_edit", 1.0)), edit=a.get("edit", "none"))
            st["ep"].extra = st["sig"].meta()

        def on_reset(self, done):
            o_reset(self, done)
            if done and st["ep"] is not None:
                st["ep"].reset_frames()
                st["sig"].p0 = None
                st["sig"].energy.reset()

        def on_tick(self, d):
            o_tick(self, d)
            if not self.on or st["ep"] is None:
                return
            col = st["ep"].ensure(self.s.model)
            st["sig"].energy.move(d)
            if col.due():
                col.frame(d, **st["sig"].frame(d, col=col))
                col.events(d.time, _statuses(self.s.runtime))
            col.tick()
        kw = dict(max_s=float(a.get("max_s", 60.0)))
        if robust:
            from rrp.harness.eval.robustness import _cond_by_key
            cond = _cond_by_key("legged", e["body"], a["condition_key"])
            kw["perturb"] = cond["pert"]
        with _patched(MQ.LeggedMotionRecorder, "__init__", init), _patched(MQ.LeggedMotionRecorder, "on_tick", on_tick), \
                _patched(MQ.LeggedMotionRecorder, "on_reset", on_reset), _patched(MQ.LeggedMotionRecorder, "on_substep", on_substep), \
                _patched(LLC, "generate", generate), torch.no_grad():
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
        max_s = float(a.get("max_s", 60.0))
        meta["success_definition"] = (
            "privileged end check (both waypoints reached) and not fallen" if max_s >= 30 else
            f"privileged end check and not fallen; this is a {max_s:g} s context-edit window episode (D-090 protocol, "
            "t_edit 2 s): the two-waypoint task cannot finish in it, so success is false by construction and the "
            "measured quantity is forward progress after t_edit (edited minus unedited, paired by seed)")
        meta.update(edit=a.get("edit", "none"), t_edit=float(a.get("t_edit", 1.0)) if a.get("edit", "none") != "none" else None,
                    tracker=row.get("tracker"), tracker_sha256=row.get("tracker_sha256"), sim_time=row.get("sim_time"),
                    fell=row.get("fell"), row_source=row.get("source"), signal_notes=LEGGED_NOTES,
                    waypoints=row.get("waypoints"))
        if a.get("edit", "none") != "none":
            first = next((t_ for t_, ed, _z, _dz in st["sig"].packets if ed != "none"), None)
            meta.update(edit_onset_t=float(a.get("t_edit", 1.0)), edit_first_packet_t=None if first is None else round(first, 4))
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
    from rrp.envs.mujoco.legged import TRACKER_HZ as T
    return T


def _legged_shas(a: dict) -> dict:
    return {k: _sha(a[k]) for k in ("flow", "bc", "rep", "posthoc_probe", "realizer") if a.get(k)}


def run_legged(e, out, pcache):
    return _run_legged(e, out, pcache, robust=False)


def run_legged_robust(e, out, pcache):
    return _run_legged(e, out, pcache, robust=True)


# ------------------------------------------------------------------ tracker validation (own mj_step loop)
def run_tracker_val(e: dict, out: Path, pcache: dict) -> list[dict]:
    import rrp.harness.eval.tracker_validation as TV
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.mujoco.legged_core import LeggedBinding
    from rrp.envs.mujoco.legged_tracker import LearnedTracker, CPGTracker, tracker_path
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
        ep.extra = sig.meta()
        sub = max(1, int(round(0.02 / model.opt.timestep)))
        cnt = dict(n=0)

        def on_step(m, d):
            cnt["n"] += 1
            sig.energy.step(d, float(m.opt.timestep))
            if cnt["n"] % sub:
                return
            sig.energy.move(d)
            col = ep.ensure(m)
            if col.due():
                col.frame(d, **sig.frame(d, col=col))
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
    import mujoco
    import rrp.harness.data.dual_quality as DQ
    from rrp.harness.eval.dual_teacher_quality import run_audit_episode
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
            m = session.model
            st["ep"].extra = dict(contact_bodies=[b for rr in session.robots for b in _sensor_bodies(m, rr.touch)],
                                  joint_names=[n for rr in session.robots for n in _qadr_joint_names(m, rr.qadr)],
                                  object_body=m.body(st["objs"][0]).name if st["objs"] else None,
                                  hands=list(session.handles))
            st["acts"] = np.concatenate([_robot_act_ids(rr) for rr in session.robots])
            st["ep"].extra["joint_torque_names"] = [m.actuator(int(i)).name for i in st["acts"]]
            st["energy"] = _Energy(st["acts"])
            groups = []
            for rr in session.robots:
                tb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) if n else -1 for n in _sensor_bodies(m, rr.touch)]
                groups += _subtrees(m, [x for x in tb if x >= 0])
            st["groups"] = groups
            st["grip0"] = {}

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
                st["energy"].step(d, s.dt * col.stride)
                rich = dict(joint_vel=np.concatenate([d.qvel[rr.dadr] for rr in s.robots]),
                            actuator_force=d.actuator_force[st["acts"]].copy(),
                            power_w=st["energy"].last_p, energy_j=st["energy"].e,
                            object_vel=_body_vel(m, d, ob[0]) if ob else None)
                if st["groups"]:
                    rich["contact_force"], rich["contact_pos"] = _contact_groups(m, d, st["groups"], exclude=st["rb"])
                held_by = s.truth().held_by
                hc, drift = [], []
                for ent, h in s.handles.items():
                    tvh = list(s._touch_values(s.robots[h.robot]))
                    hc.append(bool(len(tvh) and max(tvh) >= 0.2))
                    objs = held_by.get(ent) or []
                    if not objs:
                        st["grip0"].pop(ent, None)
                        drift.append(None)
                        continue
                    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, objs[0])
                    tcp, R = s.tcp_pose(ent)
                    pr = R.T @ (d.xpos[bid] - tcp)
                    Rr = R.T @ d.xmat[bid].reshape(3, 3)
                    if ent not in st["grip0"] or st["grip0"][ent][0] != bid:
                        st["grip0"][ent] = (bid, pr.copy(), Rr.copy())
                    _, p0_, R0_ = st["grip0"][ent]
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0_.T @ Rr) - 1) / 2))))
                    drift.append([float(np.linalg.norm(pr - p0_)) * 1000.0, ang])
                rich.update(hand_contact=hc, grip_drift=drift)
                col.frame(d, joint_target=tg, joint_pos=jp, contacts=tv, object_pose=pose,
                          penetration_mm=_penetration_mm(m, d, {g for g in range(m.ngeom) if m.geom_bodyid[g] in set(ob)},
                                                         st["rb"]),
                          phase=getattr(t, "phase_label", None) if t is not None else None, **rich)
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
                                      joint_vel="both robots' joint velocities", actuator_force="both robots' actuator forces",
                                      contact_force="per finger (touch-sensor body subtree, meta.contact_bodies) [normal, "
                                                    "tangential] N against non-robot bodies (privileged)",
                                      contact_pos="per finger force-weighted contact point (world m; privileged)",
                                      power_w="sum |actuator force x velocity| sampled at control ticks (W)",
                                      energy_j="tick-sampled integral of power_w (J)",
                                      object_vel="first task object [v, w] world (privileged)",
                                      hand_contact="per hand (meta.hands): any touch sensor >= 0.2 (public)",
                                      grip_drift="per hand: held object's [position mm, rotation deg] drift in the TCP frame "
                                                 "since the grip formed (privileged held truth); null when not holding",
                                      penetration_mm="max object-robot penetration (privileged)",
                                      phase="scripted teacher phase label L:<left>|R:<right>"))
        results.append(_finish(e, sd, meta, st["ep"], out))
    return results


# ------------------------------------------------------------------ GRASP RIG (rrp.evaluation.grasp_rig, own mj_step loop)
def run_grasp_rig(e: dict, out: Path, pcache: dict) -> list[dict]:
    import mujoco
    import rrp.harness.eval.grasp_rig as GR
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
            st["sig"] = dict(cube=cube, palm=palm, cg=cg, pads=pads, qadr=[int(m.jnt_qposadr[j]) for j in jn], rel0=None,
                             dadr=[int(m.jnt_dofadr[j]) for j in jn], energy=_Energy(np.arange(m.nu)),
                             groups=[{int(m.geom_bodyid[p])} for p in pads])
            st["ep"].extra = dict(contact_bodies=[m.body(int(m.geom_bodyid[p])).name for p in pads], object_body="cube",
                                  base_body="carriage", joint_names=[m.joint(j).name for j in jn],
                                  joint_target_names=[m.actuator(i).name for i in range(m.nu)],
                                  joint_torque_names=[m.actuator(i).name for i in range(m.nu)])
        col = st["ep"].ensure(m)
        g = st["sig"]
        g["energy"].step(d, float(m.opt.timestep))
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
                      slip=float(np.linalg.norm(rel - g["rel0"])) * 1000.0, phase=ph,
                      joint_vel=d.qvel[g["dadr"]], actuator_force=d.actuator_force.copy(), object_vel=_body_vel(m, d, g["cube"]),
                      power_w=g["energy"].last_p, energy_j=g["energy"].e,
                      **dict(zip(("contact_force", "contact_pos"), _contact_groups(m, d, g["groups"], other={g["cube"]}))))
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
        phase="rig schedule: settle 0.2 s, close 0.8 s, lift 0.5 s, hold 0.5 s, then mass ramp x1.08 / 0.25 s to slip",
        joint_vel="rig joint velocities", actuator_force="rig actuator forces (lift, grip)",
        contact_force="per pad [normal, tangential] force on the cube (N)", contact_pos="per pad force-weighted contact point",
        object_vel="cube [v, w] world", power_w="sum |actuator force x velocity| (W)", energy_j="integral over every step (J)"))
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


RENAMES_V12 = {"actuator_force": "joint_torque", "power_w": "power"}


def _align_v12(col, meta: dict):
    """Contract v1.2 names/shapes (frontend proposal): joint_torque, contact_force = normal only (+ _tangential),
    power [W] + energy [J per frame interval] (+ meta.energy_total_j), linear base/object velocity (+ angular)."""
    sg = col.signals
    for a, b in RENAMES_V12.items():
        if a in sg:
            sg[b] = sg.pop(a)
    if "contact_force" in sg:
        cf = sg.pop("contact_force")
        sg["contact_force"] = [None if f is None else [x[0] for x in f] for f in cf]
        sg["contact_force_tangential"] = [None if f is None else [x[1] for x in f] for f in cf]
    if "energy_j" in sg:
        cum = sg.pop("energy_j")
        prev, per = 0.0, []
        for v in cum:
            per.append(None if v is None else round(v - prev, 4))
            prev = v if v is not None else prev
        sg["energy"] = per
        meta["energy_total_j"] = next((v for v in reversed(cum) if v is not None), None)
    for k in ("base_vel", "object_vel"):
        if k in sg:
            v6 = sg.pop(k)
            sg[k] = [None if v is None else v[:3] for v in v6]
            sg[k.replace("_vel", "_ang_vel")] = [None if v is None else v[3:] for v in v6]
    notes = meta.get("signal_notes")
    if isinstance(notes, dict):
        notes = dict(notes)
        for a, b in RENAMES_V12.items():
            if a in notes:
                notes[b] = notes.pop(a)
        if "energy_j" in notes:
            notes["energy"] = notes.pop("energy_j").replace("since the episode reset", "") + \
                " -> reported per frame interval (J); meta.energy_total_j = the episode total"
        if "contact_force" in notes:
            notes["contact_force_tangential"] = "tangential part of the same contacts (N)"
        for k in ("base_vel", "object_vel"):
            if k in notes:
                notes[k.replace("_vel", "_ang_vel")] = "angular part (rad/s, world)"
        notes["packet_z"] = ("the executed packet downsampled to the first 8 latent dims per knot x assembly, flattened in "
                             "meta.packet_shape order")
        meta["signal_notes"] = notes
    if "packet_z" in sg:
        zs = next((z for z in sg["packet_z"] if z is not None), None)
        pn = next((z for z in sg.get("packet_norm", []) if z is not None), None)
        if zs is not None and pn is not None:
            meta["packet_shape"] = [len(pn), len(pn[0]), len(zs) // (len(pn) * len(pn[0]))]
            meta["packet_z_downsample"] = "first 8 of dz latent dims per knot x assembly (packet_norm has the full-dz norm)"


def _finish(e: dict, seed: int, meta: dict, ep: Episode, out: Path) -> dict:
    rid = f"{e['id']}-s{seed}"
    _align_v12(ep.col, meta)
    for k, v in (ep.extra or {}).items():
        if v is not None and k not in meta:
            meta[k] = v
    b = meta.get("packet_pca_basis")
    if b is not None:                                   # contract v1.1 provenance summary of the basis
        fs = b.get("fit_source", {})
        meta["packet_pca"] = dict(fit_on=f"{fs.get('kind')}: {fs.get('glob') or fs.get('data')} ({b.get('n_fit')} packets); "
                                  f"{fs.get('note')}", explained_variance=b.get("explained_variance_ratio"))
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
    from rrp.harness.yamlmini import load
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
        sys.exit("rrp.viz.record simulates: run it on the PEER through the broker (ops/bin/peer_run.sh; D-115/D-127)")
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
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(e.get("threads", 2)), MKL_NUM_THREADS=str(e.get("threads", 2)))
        env.update({k: str(v) for k, v in (e.get("env") or {}).items()})       # the entry's recorded env wins
        cmd = [sys.executable, "-m", "rrp.cli", "viz", "record", "--spec", a.spec, "--out", a.out, "--only", e["id"], "--inproc"]
        t0 = time.time()
        r = subprocess.run(cmd, env=env)
        print(f"[record] {e['id']} rc={r.returncode} {time.time() - t0:.0f}s", flush=True)
        rc = rc or r.returncode
    RP.write_index(out, _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), _git_sha())
    return rc
