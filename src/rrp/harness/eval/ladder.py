"""Closed-loop failure-localization ladder (correction 2026-09-25, item 3).

Matched scenes (same feasible seeds, same n_distractors = seed % 3), frozen checkpoints (sha256 recorded), three rungs:
  R0 `teacher`  : scripted teacher (PRIVILEGED planner, source=scripted_teacher) -> native joint-target tracker.
                   Upper bound; shows whether tracker + sim are fine.
  R1 `oracle`   : ORACLE DIAGNOSTIC. Frozen Stage-A encoder E encodes the teacher's demonstrated chunk a[t:t+H]
                   (teacher rolled forward H ticks from the CURRENT closed-loop state via snapshot/restore, normalized
                   with q0 at t exactly as training builds targets) into z = E mean; frozen system 0 realizes it online.
                   Uses privileged future teacher actions: NOT deployable.
  R2 `generated`: deployable. System i flow samples z from public observations -> the same system 0.
  `learned`     : deployable positive control. A plain FlowPolicy (direct-action / codec baseline, LearnedPolicy)
                   emits H-step joint-target chunks from public observations; the first replan_ticks rows execute.
In every rung a SHADOW teacher is advanced once per tick at the executed state (as in DART collection: its FSM
follows its own references while the arm executes other commands). It supplies (i) the phase label per tick,
(ii) the relabelled teacher command (DAgger-style label) to measure system-0 action error, and (iii) in R2 the oracle
packet at each replan tick for a same-state oracle-vs-generated comparison. It NEVER feeds control in R1/R2 except
through E in R1 (which is the declared oracle route).
Failure stages from privileged geometry (evaluation only): approach -> grasp -> lift -> transport -> place.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from rrp.core.provenance import stamp_source_label
from rrp.core.errors import ControllerRejection, StaleActionError
from rrp.harness.eval.statistics import wilson as _stats_wilson
from rrp.policies.features.featurizer import cached_featurizer
from rrp.policies.oracle import BCLookahead, OraclePacketPolicy, ShadowTeacher, make_packet

STAGES = ["approach", "grasp", "lift", "transport", "place"]


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Ladder convention of rrp.evaluation.statistics.wilson (same formula): z=1.96 and (0, 1) for n == 0 (W4 dedup)."""
    if n == 0:
        return (0.0, 1.0)
    return _stats_wilson(k, n, z)


class PrevActionFeaturizer:
    """Deployment-side reproduction of the TRAINING input: the packed datasets carry the previous 1-step command
    (normalized with the q0 of its own tick) in node-feature column static_dim+2 (and the same morph-bank rows),
    because the D-021 load-time zeroing hit column 2 instead (see archived research/tracks/ladder.md, bug B-1). The current
    featurizer always writes 0 there. mode 'own': inject the previously EXECUTED command (public: the controller's own
    last command); 'zero': current deployment behaviour."""

    def __init__(self, base, mode: str = "own"):
        self.base, self.mode = base, mode
        self.prev = None
        self.col = base.static_dim + 2

    def __getattr__(self, k):
        return getattr(self.base, k)

    def __call__(self, obs, prev_action=None):
        pi = self.base(obs)
        if self.mode == "own" and self.prev is not None:
            n = pi.act_node_feats.shape[0]
            pi.act_node_feats = pi.act_node_feats.copy()
            pi.act_node_feats[:, self.col] = self.prev[:n]
            pi.tokens = dict(pi.tokens)
            pi.tokens["morph"] = pi.tokens["morph"].copy()
            pi.tokens["morph"][:n, self.col] = self.prev[:n]
        return pi

    def record(self, cmd, q0):
        if cmd is not None and self.mode == "own" and q0 is not None:
            self.prev = self.base.aspace.normalize([cmd.groups], q0)[0].astype(np.float32)


def install_prev_action(s, mode: str):
    """Install PrevActionFeaturizer on a session and make it follow every executed command and snapshot/restore
    (for code paths that step the session themselves, e.g. latent_eval.disturbance_test)."""
    f = s._rrp_featurizer = PrevActionFeaturizer(cached_featurizer(s), mode)
    step, snap, restore = s.step, s.snapshot, s.restore
    saved = {}

    def step_(cmd=None, robot=0):
        if f.mode == "own" and cmd is not None and not isinstance(cmd, dict):
            f.record(cmd, f.base(s.observe()).q0)
        return step(cmd, robot)

    def snap_():
        sn = snap()
        saved[id(sn)] = None if f.prev is None else f.prev.copy()
        return sn

    def restore_(sn):
        f.prev = saved.get(id(sn))
        return restore(sn)
    s.step, s.snapshot, s.restore = step_, snap_, restore_
    return f




# ------------------------------------------------------------------ privileged measurement helpers
class Meter:
    """Evaluation-only privileged measurements per session: milestones, tracking error, label error."""

    def __init__(self, s):
        self.s = s
        m = s.model
        self.cube = m.body("cube").id
        self.zone = m.body("target_zone").id
        r = s.robots[0]
        self.r = r
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE,
                                      r.tcp_sites[next(a.id for a in r.spec.assemblies if a.kind in ("gripper", "hand"))])
        self.narm = len(r.arm_joints)
        self.qadr = np.asarray(r.qadr[:self.narm])
        self.fk = mujoco.MjData(m)
        self.z0 = float(s.data.xpos[self.cube][2])
        self.reached = {k: None for k in STAGES}
        self.min_d = float("inf")
        self.rows = []

    def tcp(self):
        return self.s.data.site_xpos[self.site].copy()

    def tcp_of(self, q_arm):
        self.fk.qpos[:] = self.s.data.qpos
        self.fk.qpos[self.qadr] = q_arm
        mujoco.mj_kinematics(self.s.model, self.fk)
        return self.fk.site_xpos[self.site].copy()

    def held(self):
        h = self.s.truth().held_by
        return any("cube" in v for v in h.values())

    def update(self, k):
        s, d = self.s, self.s.data
        cube, zone, tcp = d.xpos[self.cube], d.xpos[self.zone], self.tcp()
        dist = float(np.linalg.norm(tcp - cube))
        self.min_d = min(self.min_d, dist)
        held = self.held()
        mark = lambda st: self.reached.__setitem__(st, k) if self.reached[st] is None else None
        if dist < 0.025:
            mark("approach")
        if held:
            mark("approach"); mark("grasp")
            if cube[2] > self.z0 + 0.04:
                mark("lift")
                if np.linalg.norm(cube[:2] - zone[:2]) < 0.04:
                    mark("transport")
        return held

    def stage_failed(self, success: bool) -> str | None:
        if success:
            return None
        for st in STAGES:
            if self.reached[st] is None:
                return st
        return "place"


def _arm(groups):
    return np.asarray(groups["arm"], float)


# ------------------------------------------------------------------ the ladder
@dataclass
class LadderConfig:
    route: str                       # teacher | oracle | generated
    robot: str
    seeds: list
    representation: str | None = None
    flow: str | None = None
    replan_ticks: int = 8
    max_steps: int = 300
    nfe: int = 8
    compare_oracle: bool = True      # R2: oracle z at the same state at each replan (diagnostic, never controls)
    device: str = "cpu"
    object_shift: tuple | None = None   # (tick, dx, dy): teleport cube mid-episode (intervention; labelled)
    task: str = "pick_place"
    flow_seed: int = 0
    noise_scale: float = 1.0         # R2: initial flow-noise scale (0 = deterministic mode-seeking sample)
    keep_ticks: bool = False         # store per-tick rows in the output (diagnostics)
    oracle_reanchor: bool = False    # R1: re-anchor the expert reference to the measured arm at each replan
    prev_action: str = "zero"        # zero (current deployment) | own (training-consistent input, bug B-1)
    policy: str | None = None        # route learned: LearnedPolicy checkpoint (baseline FlowPolicy)
    policy_label: str | None = None  # label for the source string (default: checkpoint path)
    perturb: object = None           # W6: rrp.envs.perturb.PhysicsPerturbation (None = nominal physics, unchanged)
    chunk_blend: str = "none"        # D-126 #7: none | crossfade | ensemble (rrp.controllers.chunk_blend; none = unchanged)
    blend_ticks: int = 4
    blend_decay: float = 0.0
    oracle_expert: str = "teacher"   # R1 packet source: teacher (shadow FSM look-ahead) | bc (stateless: E(chunk the
                                     # learned BC policy `policy` would execute from the current state); ORACLE DIAGNOSTIC
    source_labels: bool | None = None  # D-126: also write a canonical `source_label` (None = $RRP_SOURCE_LABELS, default off)


def route_source(cfg: LadderConfig) -> tuple[str, str, str | None]:
    """(legacy row `source` string, canonical kind, detail) of a ladder route. The legacy string is exactly what rows
    have always carried; the canonical pair feeds `source_label` when RRP_SOURCE_LABELS=canonical (D-126, sl-1)."""
    pl = cfg.policy_label or cfg.policy
    legacy = dict(teacher="scripted_teacher(privileged)", oracle="target_encoder_oracle(ORACLE DIAGNOSTIC: "
                  + ("teacher future actions)" if cfg.oracle_expert == "teacher" else
                     f"chunk of learned:{pl} at the current state)"), generated="learned(system-i flow)",
                  learned=f"learned:{pl}")[cfg.route]
    kind, detail = dict(teacher=("scripted_teacher", "privileged"),
                        oracle=("oracle", "teacher_future" if cfg.oracle_expert == "teacher" else f"learned_chunk:{pl}"),
                        generated=("learned", cfg.flow),
                        learned=("learned", pl))[cfg.route]
    return legacy, kind, detail


def load_models(cfg: LadderConfig):
    from rrp.policies.bundles import load_representation
    from rrp.policies.nets.checkpoint import load_checkpoint
    rep = cfg.representation
    if cfg.flow and not rep:
        rep = load_checkpoint(cfg.flow, map_location="cpu")["config"]["representation"]
    ids = {}
    out = dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None, learned=None)
    if cfg.policy:
        from rrp.policies.bc import LearnedPolicy
        bkw = dict(chunk_blend=cfg.chunk_blend, blend_ticks=cfg.blend_ticks, blend_decay=cfg.blend_decay) \
            if cfg.chunk_blend != "none" else {}
        out["learned"] = LearnedPolicy.from_checkpoint(cfg.policy, device=cfg.device, nfe=cfg.nfe,
                                                       execute_prefix=cfg.replan_ticks, seed=cfg.flow_seed, **bkw)
        ids["policy"] = dict(path=str(cfg.policy), sha256=sha256_file(cfg.policy), nfe=cfg.nfe,
                             execute_prefix=cfg.replan_ticks, label=cfg.policy_label)
    if cfg.chunk_blend != "none":
        from rrp.policies.chunk_blend import BlendConfig
        ids["chunk_blend"] = BlendConfig(cfg.chunk_blend, cfg.blend_ticks, cfg.blend_decay).record()
    if rep:
        lcfg, E, R, P, res = load_representation(Path(rep), cfg.device)
        out.update(E=E, R=R, P=P, lcfg=lcfg, res=res)
        ids["representation"] = dict(path=str(rep), sha256=sha256_file(rep), latent_space_version=res["latent_space_version"],
                                     realizer_compat_version=res["realizer_compat_version"])
    if cfg.flow:
        from rrp.policies.latent import LatentPolicy
        pol = LatentPolicy.from_checkpoint(cfg.flow, device=cfg.device, nfe=cfg.nfe, seed=cfg.flow_seed)
        pol.noise_scale = cfg.noise_scale
        if pol.lsv != out["res"]["latent_space_version"]:
            raise ValueError(f"flow latent space {pol.lsv} != representation {out['res']['latent_space_version']}")
        if out["res"]["realizer_compat_version"] != pol.rcv:
            # same latent space, refit realizer (e.g. bug B-1 fix): packets are addressed to THIS realizer; logged
            ids["realizer_override"] = dict(flow_rcv=pol.rcv, used_rcv=out["res"]["realizer_compat_version"])
            pol.rcv = out["res"]["realizer_compat_version"]
        out["flow"] = pol
        ids["flow"] = dict(path=str(cfg.flow), sha256=sha256_file(cfg.flow), nfe=cfg.nfe, sampler="euler-ode",
                           noise_scale=cfg.noise_scale)
    return out, ids


@torch.no_grad()
def run_ladder(cfg: LadderConfig, out_path: Path | None = None, models=None, ids=None, frame_cb=None,
               collect: dict | None = None, cmd_log: dict | None = None) -> list[dict]:
    """cmd_log: if given, cmd_log[k] = list of the EXACT executed command groups per tick (None = hold), for replay.
    frame_cb(k, session, step, shadow_phase): optional per-tick callback after each executed step (rendering).
    collect: DAgger buffer for system 0 (route oracle): at each replan the oracle posterior (mu, logvar) of the teacher
    chunk; at each executed tick with phase j <= collect['max_j'] the LEARNER-visited state (node features, local
    sensors) and the shadow teacher's command there (normalized with that tick's q0) -> see save_dagger/refit."""
    if collect is not None:
        collect.setdefault("mu", []); collect.setdefault("lv", []); collect.setdefault("rows", [])
        collect["cur"] = {}; collect.setdefault("max_j", 12)
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.envs.mujoco.session import Session
    from rrp.policies.system0 import LatentSystem0
    from rrp.policies.system0 import batched_ticks
    if models is None:
        models, ids = load_models(cfg)
    robot = workbench_robots()[cfg.robot]()
    oracle = OraclePacketPolicy(models["E"], models["lcfg"], models["res"], cfg.device,
                                reanchor=cfg.oracle_reanchor) if models["E"] is not None else None
    gen = models["flow"]
    lp = models.get("learned")
    if cfg.route == "learned" and lp is None:
        raise ValueError("route learned needs cfg.policy")
    from rrp.envs.mujoco.motion_quality import ArmMotionRecorder
    if cfg.perturb is not None and cfg.perturb.step_hooks and (cfg.route == "oracle" or (cfg.route == "generated"
                                                                                            and cfg.compare_oracle)):
        # the oracle look-ahead rolls the real session forward and restores it; the step hooks (ctrl-delay FIFO, push
        # bookkeeping) are not part of the snapshot, so they would be corrupted by the look-ahead
        raise ValueError("step-level perturbations need a route without look-ahead rollouts (compare_oracle=False)")
    S, s0, meters, shadows, meta, mrecs, perts = [], [], [], [], [], [], []
    from rrp.harness.data.contact_metrics import contact_metrics_enabled
    cfrecs = [] if contact_metrics_enabled() else None     # W12 held-object drift keys (RRP_CONTACT_METRICS=1)
    for sd in cfg.seeds:
        s = Session(BUILDERS[cfg.task](robot, sd, n_distractors=sd % 3), seed=sd)
        if cfg.perturb is not None:
            from rrp.envs.mujoco.perturb import apply_model, install_arm, arm_parts
            ap_ = arm_parts(s.model, s.robots[0])
            cube_ = [mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "cube")]
            rec_ = apply_model(s.model, cfg.perturb, robot_bodies=ap_["bodies"], com_body=ap_["last_link"],
                               act_ids=ap_["act_ids"], object_bodies=cube_, object_contact_geoms=ap_["finger_geoms"])
            perts.append((rec_, install_arm(s, cfg.perturb, sd)))
        mrecs.append(ArmMotionRecorder(s))
        if cfrecs is not None:
            from rrp.harness.data.contact_labels import ContactFrameRecorder
            cfrecs.append(ContactFrameRecorder(s))
        f = s._rrp_featurizer = PrevActionFeaturizer(cached_featurizer(s), cfg.prev_action)
        S.append(s)
        meters.append(Meter(s))
        if cfg.oracle_expert == "bc":
            if lp is None:
                raise ValueError("oracle_expert bc needs cfg.policy")
            oracle.shadows[id(s)] = BCLookahead(lp)
            shadows.append(ShadowTeacher(s))            # labels/phase only
        else:
            shadows.append(oracle.shadow(s) if oracle else ShadowTeacher(s))
        if cfg.route not in ("teacher", "learned") or models["R"] is not None:     # teacher route + R: shadow system 0 (not executed)
            s0.append(LatentSystem0(models["R"], f, latent_space_version=models["res"]["latent_space_version"],
                                    realizer_compat_version=models["res"]["realizer_compat_version"], device=cfg.device))
            if cfg.chunk_blend != "none" and cfg.route in ("oracle", "generated"):     # D-126 #7 (executed system 0)
                s0[-1].configure_blend(cfg.chunk_blend, cfg.blend_ticks, cfg.blend_decay)
        meta.append(dict(done=False, outcome=None, steps=0, calls=0, t0=time.time(), ticks=[], replans=[],
                         teacher_done_tick=None))
    for step in range(cfg.max_steps):
        act = [k for k, m in enumerate(meta) if not m["done"]]
        if not act:
            break
        if cfg.object_shift and step == cfg.object_shift[0]:
            for k in act:
                p = S[k].data.xpos[meters[k].cube].copy()
                p[0] += cfg.object_shift[1]; p[1] += cfg.object_shift[2]
                S[k].teleport_object("cube", p, source="ladder_disturbance")
        need = [] if not s0 else [k for k in act if step % cfg.replan_ticks == 0 or s0[k].packet is None]
        if need:
            zo = oracle.encode([S[k] for k in need]) if (cfg.route != "generated" or (cfg.compare_oracle and oracle)) else None
            if cfg.route != "generated":
                pk = [make_packet(S[k], cached_featurizer(S[k]), z, models["lcfg"].knot_times, oracle.lsv, oracle.rcv,
                                  oracle.validity, source="target_encoder_oracle", policy_version="oracle_diagnostic")
                      for k, z in zip(need, zo)]
            else:
                pk = gen.packets([S[k] for k in need])
            if collect is not None and cfg.route == "oracle":
                for j, k in enumerate(need):
                    collect["cur"][k] = len(collect["mu"])
                    collect["mu"].append(zo[j].astype(np.float16)); collect["lv"].append(oracle.last_logvar[j].astype(np.float16))
            elif collect is not None and cfg.route == "generated" and cfg.oracle_expert == "bc":
                # system-0 DAgger on system i's OWN packets: z = generated packet (deterministic, tiny logvar),
                # label = the stateless BC expert's plan row j from the replan state (oracle.last_cmds via zo above)
                for j, k in enumerate(need):
                    collect["cur"][k] = len(collect["mu"])
                    zg = np.asarray(pk[j].z, np.float32)
                    collect["mu"].append(zg.astype(np.float16)); collect["lv"].append(np.full_like(zg, -8.0).astype(np.float16))
                    if collect.get("gen_ctx") is not None:   # generator DAgger: (public context at the learner state, z*)
                        collect["gen_ctx"].append((cached_featurizer(S[k])(S[k].observe()), np.asarray(zo[j], np.float32)))
            for j, (k, p) in enumerate(zip(need, pk)):
                meta[k]["calls"] += 1
                rec = dict(t=step)
                if zo is not None and cfg.route == "generated":
                    rec.update(_compare(models, S[k], p.z, zo[j], cfg.device))
                meta[k]["replans"].append(rec)
                try:
                    s0[k].receive(p, now=float(S[k].data.time), graph_version=S[k].runtime.graph_version)
                except (ControllerRejection, StaleActionError):
                    rec["rejected"] = True
        needl = []
        if cfg.route == "learned":
            needl = [k for k in act if step % cfg.replan_ticks == 0 or not S[k].executor.queue]
            if needl:
                for k, ch in zip(needl, lp.chunks([S[k] for k in needl])):
                    meta[k]["calls"] += 1
                    meta[k]["replans"].append(dict(t=step))
                    try:
                        S[k].submit_chunk(ch, execute_prefix=cfg.replan_ticks)
                    except (ControllerRejection, StaleActionError):
                        meta[k]["replans"][-1]["rejected"] = True
        bnd = set(need) | (set(needl) if cfg.route == "learned" else set())
        labels = {k: shadows[k].label(S[k]) for k in act}
        sys0 = batched_ticks([s0[k] for k in act], [S[k] for k in act]) if s0 else [None] * len(act)
        if cfg.route == "learned":
            from rrp.core.action import NativeCommand
            sys0 = []
            for k in act:
                row_ = S[k].executor.pop()
                sys0.append(None if row_ is None else NativeCommand(controller_version=S[k].controller_version(),
                                                                    groups=row_, source="learned"))
        cmds = [labels[k] for k in act] if cfg.route == "teacher" else sys0
        for k, cmd, c0 in zip(act, cmds, sys0):
            s, mt = S[k], meters[k]
            if cmd_log is not None:
                cmd_log.setdefault(k, []).append(None if cmd is None else {g: np.array(v, copy=True) if not np.isscalar(v)
                                                                           else v for g, v in cmd.groups.items()})
            f = cached_featurizer(s)
            q_meas = s.data.qpos[mt.qadr].copy()
            lab = labels[k]
            row = dict(t=step, phase=shadows[k].t.phase)
            if cfg.keep_ticks:
                xm = s.data.site_xmat[mt.site].reshape(3, 3)
                row.update(q=q_meas.round(4).tolist(), tcp=mt.tcp().round(4).tolist(),
                           cube=s.data.xpos[mt.cube].round(4).tolist(), tool_z=xm[:, 2].round(3).tolist(),
                           lab=np.round(lab.groups["arm"], 4).tolist(), lab_g=lab.groups.get("gripper"),
                           cmd=np.round(c0.groups["arm"], 4).tolist() if c0 is not None else None,
                           cmd_g=c0.groups.get("gripper") if c0 is not None else None)
            if s0 and s0[k].packet is not None:
                row["j"] = int(round((float(s.data.time) - s0[k].packet.valid_from) / s.dt))
                if cfg.keep_ticks and oracle is not None and getattr(oracle, "last_cmds", None) and id(s) in oracle.last_cmds:
                    pl = oracle.last_cmds[id(s)]                    # the packet's encoded plan row j (diagnostic)
                    pr = pl[min(row["j"], len(pl) - 1)]
                    row["plan"] = np.round(pr["arm"], 4).tolist(); row["plan_g"] = pr.get("gripper")
                    row["plan_tcp"] = mt.tcp_of(np.asarray(pr["arm"], float)).round(4).tolist()
                    if c0 is not None:
                        row["cmd_tcp"] = mt.tcp_of(_arm(c0.groups)).round(4).tolist()
            if collect is not None and c0 is not None and k in collect["cur"] and row.get("j", 99) <= collect["max_j"]:
                pi_c = f.base(s.observe())
                n_ = pi_c.act_node_feats.shape[0]
                nd = np.zeros((12, pi_c.act_node_feats.shape[1]), np.float16); nd[:n_] = pi_c.act_node_feats
                lg = lab.groups
                if cfg.oracle_expert == "bc":           # label = the packet's own plan row j (consistent with z)
                    plan = oracle.last_cmds.get(id(s))
                    lg = plan[min(row["j"], len(plan) - 1)] if plan else None
                if lg is not None:
                    a1 = np.zeros(12, np.float32); a1[:n_] = f.aspace.normalize([lg], pi_c.q0)[0]
                    from rrp.policies.features.derived import local_sensors
                    collect["rows"].append((collect["cur"][k], row["j"], nd, n_, local_sensors(pi_c).astype(np.float16), a1))
            if c0 is not None:              # system-0 output (executed in R1/R2; shadow-only in R0) vs teacher label
                q0z = np.zeros(len(f.aspace.node_group))       # the q0 offset cancels in the difference
                la = f.aspace.normalize([lab.groups], q0z)[0]
                ca = f.aspace.normalize([c0.groups], q0z)[0]
                arm_n = [i for i, g in enumerate(f.aspace.is_gripper) if not g]
                grip_n = [i for i, g in enumerate(f.aspace.is_gripper) if g]
                row["lab_err_arm"] = float(np.mean((la[arm_n] - ca[arm_n]) ** 2))
                row["lab_err_grip"] = float(np.mean((la[grip_n] - ca[grip_n]) ** 2)) if grip_n else 0.0
                row["lab_step_arm"] = float(np.mean(((la[arm_n] - f.aspace.normalize([dict(lab.groups, arm=q_meas.tolist())], q0z)[0][arm_n])) ** 2))
            if cmd is not None:
                qc = _arm(cmd.groups)
                row["cmd_step"] = float(np.abs(qc - q_meas).max())
                tcp_cmd = mt.tcp_of(qc)
            f.record(cmd, f.base(s.observe()).q0 if cmd is not None and f.mode == "own" else None)
            s.step(cmd)
            meta[k]["steps"] += 1
            mrecs[k].tick(None if cmd is None else cmd.groups, k in bnd)
            if cfrecs is not None:
                cfrecs[k].tick()
            if cmd is not None:
                row["track_q"] = float(np.abs(s.data.qpos[mt.qadr] - qc).mean())
                row["track_tcp"] = float(np.linalg.norm(mt.tcp() - tcp_cmd))
            row["held"] = mt.update(step)
            meta[k]["ticks"].append(row)
            if frame_cb is not None:
                frame_cb(k, s, step, shadows[k].t.phase)
            if cfg.route == "teacher" and shadows[k].t.done:
                meta[k]["done"] = True
            if s.data.xpos[mt.cube][2] < -0.05:
                meta[k].update(done=True, outcome="dropped_off_table")
            elif s.runtime.succeeded() and cfg.route != "teacher":
                meta[k]["done"] = True
    out = []
    for k, s in enumerate(S):
        m, mt = meta[k], meters[k]
        if cfg.route == "teacher":
            s.step(None)
        priv = bool(s.privileged_success())
        if m["outcome"] is None:
            m["outcome"] = "success" if priv else ("timeout" if m["steps"] >= cfg.max_steps else "failure")
        T = m["ticks"]
        agg = lambda key, rows: float(np.mean([r[key] for r in rows if key in r])) if any(key in r for r in rows) else None
        by_phase = {}
        for ph in sorted({r["phase"] for r in T}):
            rows = [r for r in T if r["phase"] == ph]
            by_phase[ph] = dict(n=len(rows), track_q=agg("track_q", rows), track_tcp=agg("track_tcp", rows),
                                lab_err_arm=agg("lab_err_arm", rows), lab_err_grip=agg("lab_err_grip", rows))
        by_j = {}
        for r in T:
            if "j" in r and "lab_err_arm" in r:
                by_j.setdefault(r["j"], []).append(r["lab_err_arm"])
        rp = [r for r in m["replans"] if "z_dist_rel" in r]
        out.append(dict(
            route=cfg.route, robot=cfg.robot, seed=cfg.seeds[k], outcome=m["outcome"], privileged_success=priv,
            public_success=bool(s.runtime.succeeded()), steps=m["steps"], packets=m["calls"],
            rejected=s0[k].stats.rejected if s0 else 0, fallback_holds=s0[k].stats.fallback_holds if s0 else 0,
            events={e: v.status for e, v in s.runtime.instances.items()}, stage_reached=mt.reached,
            failed_stage=mt.stage_failed(priv), min_tcp_cube_m=mt.min_d, final_teacher_phase=shadows[k].t.phase,
            track_q_rad=agg("track_q", T), track_tcp_m=agg("track_tcp", T), cmd_step_rad=agg("cmd_step", T),
            lab_err_arm=agg("lab_err_arm", T), lab_step_arm=agg("lab_step_arm", T), lab_err_grip=agg("lab_err_grip", T), by_phase=by_phase,
            lab_err_by_j={j: float(np.mean(v)) for j, v in sorted(by_j.items())},
            oracle_cmp=({key: float(np.mean([r[key] for r in rp if key in r])) for key in sorted({x for r in rp for x in r})
                         if key not in ("t", "rejected")} if rp else None),
            oracle_cmp_by_phase=_cmp_by_phase(rp, T) if rp else None,
            ticks=T if cfg.keep_ticks else None, replans=m["replans"] if cfg.keep_ticks else None,
            interventions=s.intervention_log, wall_s=time.time() - m["t0"],
            source=route_source(cfg)[0],
            checkpoints=ids, motion=mrecs[k].summary()))
        stamp_source_label(out[-1], *route_source(cfg)[1:], enabled=cfg.source_labels)
        if cfrecs is not None:
            from rrp.harness.data.contact_metrics import arm_contact_motion
            out[-1]["motion"].update(arm_contact_motion(cfrecs[k].recording()))
        if cfg.perturb is not None:
            rec_, st_ = perts[k]
            out[-1]["perturbation"] = dict(cfg.perturb.to_dict(), applied=rec_,
                                           ctrl_delay=(dict(version=st_["delay"].version, substeps=st_["delay"].n)
                                                       if st_["delay"] is not None else None),
                                           push=(st_["push"].record() if st_["push"] is not None else None))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as fh:
            for r in out:
                fh.write(json.dumps(r, default=str) + "\n")
    return out


def _cmp_by_phase(rp, T):
    ph = {r["t"]: r["phase"] for r in T}
    out = {}
    for r in rp:
        out.setdefault(ph.get(r["t"], "?"), []).append(r)
    return {p: {k: float(np.mean([x[k] for x in v if k in x])) for k in sorted({y for x in v for y in x})
                if k not in ("t", "rejected")} | {"n": len(v)} for p, v in out.items()}


@torch.no_grad()
def _compare(models, s, zg, zo, device) -> dict:
    """Same state: generated z vs oracle z (distance), system-0 first-tick action from each vs the teacher label,
    and packet-probe readouts of each against privileged labels (diagnostic)."""
    from rrp.policies.features.derived import local_sensors
    from rrp.harness.eval.latent_eval import packet_labels, readout_metrics
    f = cached_featurizer(s)
    pi = f(s.observe())
    kt = torch.tensor(models["lcfg"].knot_times, dtype=torch.float32, device=device)
    nf_ = pi.act_node_feats.astype(np.float32).copy()
    if getattr(models["R"], "drop_qd", False):
        nf_[:, 27] = 0
    nf = torch.from_numpy(nf_)[None].to(device)
    nm = torch.ones(1, nf.shape[1], dtype=torch.bool, device=device)
    lc = torch.from_numpy(local_sensors(pi))[None].to(device)
    ph = torch.zeros(1, device=device)
    zm = torch.ones(1, zg.shape[1], dtype=torch.bool, device=device)
    tz = lambda z: torch.from_numpy(np.asarray(z, np.float32))[None].to(device)
    ag = models["R"](tz(zg), zm, kt, ph, nf, nm, lc)[0].cpu().numpy()
    ao = models["R"](tz(zo), zm, kt, ph, nf, nm, lc)[0].cpu().numpy()
    arm = [i for i, g in enumerate(f.aspace.is_gripper) if not g]
    grip = [i for i, g in enumerate(f.aspace.is_gripper) if g]
    out = dict(z_dist_rel=float(np.linalg.norm(zg - zo) / (np.linalg.norm(zo) + 1e-6)),
               z_mse=float(np.mean((zg - zo) ** 2)),
               act_gen_vs_oracle_arm=float(np.mean((ag[arm] - ao[arm]) ** 2)),
               act_gen_vs_oracle_grip=float(np.mean((ag[grip] - ao[grip]) ** 2)) if grip else 0.0)
    if models["P"] is not None:
        lab = packet_labels(s, pi)
        Sn = lab["held"].shape[1]
        lab = {k: v.to(device) for k, v in lab.items()}
        msk = torch.ones(1, Sn, dtype=torch.bool, device=device)
        for name, z in (("gen", zg), ("oracle", zo)):
            for q, (x, n) in readout_metrics(models["P"](tz(z), zm, Sn), lab, msk).items():
                if n:
                    out[f"probe_{name}_{q}"] = float(x / n)
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["privileged_success"] for r in rows)
    fs = {}
    for r in rows:
        fs[r["failed_stage"] or "success"] = fs.get(r["failed_stage"] or "success", 0) + 1
    g = lambda key: float(np.mean([r[key] for r in rows if r.get(key) is not None])) if any(r.get(key) is not None for r in rows) else None
    return dict(n=n, success=k, rate=k / n if n else None, wilson95=wilson(k, n), failed_stage=fs,
                track_q_rad=g("track_q_rad"), track_tcp_m=g("track_tcp_m"), lab_err_arm=g("lab_err_arm"),
                lab_err_grip=g("lab_err_grip"), min_tcp_cube_m=g("min_tcp_cube_m"), cmd_step_rad=g("cmd_step_rad"))


def _gripper_mask(robot_key: str, N: int) -> np.ndarray:
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.envs.mujoco.session import Session
    s = Session(BUILDERS["pick_place"](workbench_robots()[robot_key](), 3_000_000, n_distractors=0), seed=3_000_000)
    g = np.zeros(N, bool)
    isg = np.asarray(cached_featurizer(s).aspace.is_gripper, bool)
    g[:len(isg)] = isg
    return g


def save_dagger(collect: dict, path: Path, meta: dict):
    rows = collect["rows"]
    np.savez_compressed(path, mu=np.stack(collect["mu"]), lv=np.stack(collect["lv"]),
                        rp=np.array([r[0] for r in rows], np.int32), j=np.array([r[1] for r in rows], np.int8),
                        node=np.stack([r[2] for r in rows]), n_nodes=np.array([r[3] for r in rows], np.int16),
                        local=np.stack([r[4] for r in rows]), a1=np.stack([r[5] for r in rows]),
                        meta=np.array(json.dumps(meta)))
    if collect.get("gen_ctx"):
        import pickle
        with open(str(path) + ".genctx.pkl", "wb") as fh:
            pickle.dump(dict(items=collect["gen_ctx"], meta=meta), fh)
