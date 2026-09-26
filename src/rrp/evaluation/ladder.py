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
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from rrp.contracts.errors import ControllerRejection, StaleActionError

STAGES = ["approach", "grasp", "lift", "transport", "place"]


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


class PrevActionFeaturizer:
    """Deployment-side reproduction of the TRAINING input: the packed datasets carry the previous 1-step command
    (normalized with the q0 of its own tick) in node-feature column static_dim+2 (and the same morph-bank rows),
    because the D-021 load-time zeroing hit column 2 instead (see research/tracks/ladder.md, bug B-1). The current
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
    f = s._rrp_featurizer = PrevActionFeaturizer(_featurizer(s), mode)
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


def _featurizer(s):
    from rrp.data.collect import featurizer_for
    f = getattr(s, "_rrp_featurizer", None)
    if f is None:
        f = s._rrp_featurizer = featurizer_for(s)
    return f


class ShadowTeacher:
    """Per-session scripted teacher (PRIVILEGED). label(s) advances it once at the current state; lookahead(s, H)
    returns the teacher's next H commands executed from the current state, then restores session + teacher."""

    def __init__(self, s, reanchor: bool = False):
        from rrp.control.teachers import PickPlaceTeacher
        self.t = PickPlaceTeacher(s)
        self.synced = s.step_count
        self.reanchor = reanchor

    def reanchor_now(self, s):
        """Re-anchor the expert's internal TCP reference (and IK seed) to the MEASURED arm, keeping its phase: the
        expert then demonstrates a smooth continuation from where the arm actually is (as in clean demonstrations)
        instead of a jump back to its own run-away reference."""
        t = self.t
        t.tcp_cmd = s._fk_site(t.r, t.tcp_site)[0].copy()
        t.q_arm = s.data.qpos[t.r.qadr[:len(t.q_arm)]].copy()

    def catch_up(self, s):
        while self.synced < s.step_count:       # missed ticks (e.g. disturbance_test warmup): advance at current state
            self.t.act()
            self.synced += 1

    def label(self, s):
        self.catch_up(s)
        c = self.t.act()
        self.synced = s.step_count + 1
        return c

    def lookahead(self, s, H: int):
        self.catch_up(s)
        if self.reanchor:
            self.reanchor_now(s)
        snap, st = s.snapshot(), self.t.state()
        cmds = []
        for _ in range(H):
            c = self.t.act()
            cmds.append(c.groups)
            s.step(c)
            if self.t.done:
                break
        s.restore(snap)
        self.t.load(st)
        return cmds


class BCLookahead:
    """Stateless expert for the oracle route: the H-step chunk a learned BC policy emits at the CURRENT state (no
    stepping, no FSM). Valid off the teacher trajectory, unlike the shadow teacher (see sprint_bc / D-050)."""

    def __init__(self, lp):
        self.lp = lp

    def lookahead(self, s, H: int):
        return self.lookahead_batch([s], H)[0]

    def lookahead_batch(self, sessions, H: int):
        return [[{g.group: np.asarray(g.values[t]).tolist() for g in ch.command_groups} for t in range(min(H, ch.horizon))]
                for ch in self.lp.chunks(sessions)]


class OraclePacketPolicy:
    """ORACLE DIAGNOSTIC: z = E(public context at t, teacher chunk a[t:t+H]) (posterior mean). Same interface as
    LatentPolicy (packets/featurizer/lsv/rcv/calls) so evaluate/disturbance code can drive it."""
    name = "target_encoder_oracle"

    def __init__(self, E, cfg, res, device, validity_s: float = 0.8, reanchor: bool = False):
        self.reanchor = reanchor
        self.E, self.cfg, self.device, self.validity = E, cfg, device, validity_s
        self.lsv, self.rcv = res["latent_space_version"], res["realizer_compat_version"]
        self.shadows: dict[int, ShadowTeacher] = {}
        self.calls = 0

    def featurizer(self, s):
        return _featurizer(s)

    def shadow(self, s) -> ShadowTeacher:
        k = id(s)
        if k not in self.shadows:
            self.shadows[k] = ShadowTeacher(s, reanchor=self.reanchor)
        return self.shadows[k]

    @torch.no_grad()
    def encode(self, sessions) -> list[np.ndarray]:
        from rrp.model.batch import collate_inputs
        from rrp.model.semantic_latent import assembly_tokens
        H = self.cfg.horizon
        feats, A, V = [], [], []
        pre = {}
        bcs = [s for s in sessions if isinstance(self.shadow(s), BCLookahead)]
        if bcs:                                           # one batched BC call for all stateless-expert sessions
            for s, rows in zip(bcs, self.shadow(bcs[0]).lookahead_batch(bcs, H)):
                pre[id(s)] = rows
        self.last_cmds = getattr(self, "last_cmds", {})
        for s in sessions:
            pi = self.featurizer(s)(s.observe())
            cmds = pre[id(s)] if id(s) in pre else self.shadow(s).lookahead(s, H)
            self.last_cmds[id(s)] = cmds
            n = len(cmds)
            seq = cmds + [cmds[-1]] * (H - n)
            a = self.featurizer(s).aspace.normalize(seq, pi.q0).astype(np.float16).astype(np.float32)  # packed fp16
            v = np.zeros_like(a, bool)
            v[:n] = True
            feats.append(pi); A.append(a); V.append(v)
        b = collate_inputs(feats).to(self.device)
        N = b.node_feats.shape[1]
        a = np.zeros((len(A), H, N), np.float32); v = np.zeros((len(A), H, N), bool)
        for i in range(len(A)):
            a[i, :, :A[i].shape[1]] = A[i]; v[i, :, :V[i].shape[1]] = V[i]
        af, am, ai = assembly_tokens(b)
        mu, lv = self.E(b, torch.from_numpy(a).to(self.device), torch.from_numpy(v).to(self.device), af, am, ai)
        mu, lv = mu.float().cpu().numpy(), lv.float().cpu().numpy()
        self.last_logvar = [lv[i][:, :int(am[i].sum())] for i in range(len(sessions))]
        return [mu[i][:, :int(am[i].sum())] for i in range(len(sessions))]

    def packets(self, sessions):
        zs = self.encode(sessions)
        self.calls += len(sessions)
        return [make_packet(s, self.featurizer(s), z, self.cfg.knot_times, self.lsv, self.rcv, self.validity,
                            source="target_encoder_oracle", policy_version=self.name) for s, z in zip(sessions, zs)]


def make_packet(s, f, z, knot_times, lsv, rcv, validity, *, source, policy_version):
    from rrp.contracts.latent_action import LatentActionChunk, AssemblyHandle, EntityHandle
    o = s.observe()
    M = z.shape[1]
    gasms = [a for a in f.spec.assemblies if a.kind in ("gripper", "hand")][:M]
    now = float(s.data.time)
    return LatentActionChunk(
        latent_space_version=lsv, realizer_compat_version=rcv, z=np.asarray(z, np.float32), knot_times=list(knot_times),
        assemblies=[AssemblyHandle(handle=f"asm:{f.spec.spec_hash}:{a.frame.link}", robot_index=0) for a in gasms],
        assembly_mask=[True] * M, entity_registry=[EntityHandle(handle=f"ent:{d.slot}") for d in o.object_descriptors],
        observation_id=o.observation_id, graph_version=s.runtime.graph_version, runtime_version=s.runtime.runtime_version,
        robot_spec_hash=f.spec.spec_hash, generated_at=time.time(), valid_from=now, valid_until=now + validity,
        source=source, policy_version=policy_version)


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
    keep_ticks: bool = False         # store per-tick rows in the output (diagnostics)
    oracle_reanchor: bool = False    # R1: re-anchor the expert reference to the measured arm at each replan
    prev_action: str = "zero"        # zero (current deployment) | own (training-consistent input, bug B-1)
    policy: str | None = None        # route learned: LearnedPolicy checkpoint (baseline FlowPolicy)
    policy_label: str | None = None  # label for the source string (default: checkpoint path)
    oracle_expert: str = "teacher"   # R1 packet source: teacher (shadow FSM look-ahead) | bc (stateless: E(chunk the
                                     # learned BC policy `policy` would execute from the current state); ORACLE DIAGNOSTIC


def load_models(cfg: LadderConfig):
    from rrp.learning.latent_train import load_representation
    from rrp.learning.checkpoint import load_checkpoint
    rep = cfg.representation
    if cfg.flow and not rep:
        rep = load_checkpoint(cfg.flow, map_location="cpu")["config"]["representation"]
    ids = {}
    out = dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None, learned=None)
    if cfg.policy:
        from rrp.policy.runner import LearnedPolicy
        out["learned"] = LearnedPolicy.from_checkpoint(cfg.policy, device=cfg.device, nfe=cfg.nfe,
                                                       execute_prefix=cfg.replan_ticks, seed=cfg.flow_seed)
        ids["policy"] = dict(path=str(cfg.policy), sha256=sha256_file(cfg.policy), nfe=cfg.nfe,
                             execute_prefix=cfg.replan_ticks, label=cfg.policy_label)
    if rep:
        lcfg, E, R, P, res = load_representation(Path(rep), cfg.device)
        out.update(E=E, R=R, P=P, lcfg=lcfg, res=res)
        ids["representation"] = dict(path=str(rep), sha256=sha256_file(rep), latent_space_version=res["latent_space_version"],
                                     realizer_compat_version=res["realizer_compat_version"])
    if cfg.flow:
        from rrp.policy.latent_runner import LatentPolicy
        pol = LatentPolicy.from_checkpoint(cfg.flow, device=cfg.device, nfe=cfg.nfe, seed=cfg.flow_seed)
        if pol.lsv != out["res"]["latent_space_version"]:
            raise ValueError(f"flow latent space {pol.lsv} != representation {out['res']['latent_space_version']}")
        if out["res"]["realizer_compat_version"] != pol.rcv:
            # same latent space, refit realizer (e.g. bug B-1 fix): packets are addressed to THIS realizer; logged
            ids["realizer_override"] = dict(flow_rcv=pol.rcv, used_rcv=out["res"]["realizer_compat_version"])
            pol.rcv = out["res"]["realizer_compat_version"]
        out["flow"] = pol
        ids["flow"] = dict(path=str(cfg.flow), sha256=sha256_file(cfg.flow), nfe=cfg.nfe, sampler="euler-ode")
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
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.control.latent_realizer import LatentSystem0
    from rrp.learning.latent_grpo import batched_ticks
    if models is None:
        models, ids = load_models(cfg)
    robot = workbench_robots()[cfg.robot]()
    oracle = OraclePacketPolicy(models["E"], models["lcfg"], models["res"], cfg.device,
                                reanchor=cfg.oracle_reanchor) if models["E"] is not None else None
    gen = models["flow"]
    lp = models.get("learned")
    if cfg.route == "learned" and lp is None:
        raise ValueError("route learned needs cfg.policy")
    S, s0, meters, shadows, meta = [], [], [], [], []
    for sd in cfg.seeds:
        s = Session(BUILDERS[cfg.task](robot, sd, n_distractors=sd % 3), seed=sd)
        f = s._rrp_featurizer = PrevActionFeaturizer(_featurizer(s), cfg.prev_action)
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
                pk = [make_packet(S[k], _featurizer(S[k]), z, models["lcfg"].knot_times, oracle.lsv, oracle.rcv,
                                  oracle.validity, source="target_encoder_oracle", policy_version="oracle_diagnostic")
                      for k, z in zip(need, zo)]
            else:
                pk = gen.packets([S[k] for k in need])
            if collect is not None and cfg.route == "oracle":
                for j, k in enumerate(need):
                    collect["cur"][k] = len(collect["mu"])
                    collect["mu"].append(zo[j].astype(np.float16)); collect["lv"].append(oracle.last_logvar[j].astype(np.float16))
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
        labels = {k: shadows[k].label(S[k]) for k in act}
        sys0 = batched_ticks([s0[k] for k in act], [S[k] for k in act]) if s0 else [None] * len(act)
        if cfg.route == "learned":
            from rrp.contracts.action import NativeCommand
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
            f = _featurizer(s)
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
                    from rrp.learning.packed import local_sensors
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
            source=dict(teacher="scripted_teacher(privileged)", oracle="target_encoder_oracle(ORACLE DIAGNOSTIC: "
                        + ("teacher future actions)" if cfg.oracle_expert == "teacher" else
                           f"chunk of learned:{cfg.policy_label or cfg.policy} at the current state)"), generated="learned(system-i flow)",
                        learned=f"learned:{cfg.policy_label or cfg.policy}")[cfg.route],
            checkpoints=ids))
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
    from rrp.learning.packed import local_sensors
    from rrp.evaluation.latent_eval import packet_labels
    from rrp.model.latent_probes import probe_metrics
    f = _featurizer(s)
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
            for q, (x, n) in probe_metrics(models["P"](tz(z), zm, Sn), lab, msk).items():
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


@torch.no_grad()
def packed_realization_check(rep_path: str, packed_dir: str, robot_key: str | None, device, n_batches: int = 20,
                             seed: int = 3, max_j: int = 7, zero_prev_action: bool = False) -> dict:
    """Stage-A realization error on the TRAINING pack split into arm / gripper nodes and by phase j
    (encoded-target oracle, the same quantity the ladder measures online as lab_err_*)."""
    import random
    from rrp.learning.latent_train import load_representation, LatentData
    from rrp.model.semantic_latent import assembly_tokens
    lcfg, E, R, P, res = load_representation(Path(rep_path), device)
    data = LatentData(Path(packed_dir), zero_prev_action=zero_prev_action, anchor=getattr(R, "anchor", False),
                      drop_qd=getattr(R, "drop_qd", False))
    rid = data.ds.meta["robot_ids"].get(robot_key) if robot_key else None
    pool = None
    if rid is not None:
        pool = [int(i) for i in np.nonzero(np.asarray(data.ds.arr["robot_id"]) == rid)[0]]
    rng = random.Random(seed)
    acc = {}
    for _ in range(n_batches):
        if pool is not None:
            sel = np.array(sorted(rng.sample(pool, 128)))
            j = np.array([rng.randint(0, max_j) for _ in range(128)])
            tgt = np.minimum(sel + j, data.n - 1)
            same = data.ep[tgt] == data.ep[sel]
            tgt = np.where(same, tgt, sel); j = np.where(same, j, 0)
        else:
            sel, tgt, j = data.sample(128, rng, max_j)
        batch, a, v, lab, r = data.fetch(sel, tgt, device)
        af, am, ai = assembly_tokens(batch)
        mu, _ = E(batch, a, v, af, am, ai)
        kt = torch.tensor(lcfg.knot_times, device=device)
        pred = R(mu, am, kt, torch.as_tensor(j * lcfg.control_dt, dtype=mu.dtype, device=device), r["node"],
                 r["node_mask"], r["local"])
        # gripper nodes: action-node feature flags differ; use the target distribution: |a| saturates at 1 for grippers
        m = (r["v1"] & r["node_mask"]).cpu().numpy()
        err = ((pred - r["a1"]) ** 2).cpu().numpy()
        zero = (r["a1"] ** 2).cpu().numpy()
        isg = np.broadcast_to(_gripper_mask(robot_key, m.shape[1]), m.shape) if robot_key else np.zeros_like(m)
        for b in range(len(sel)):
            for key, msk in (("arm", m[b] & ~isg[b]), ("grip", m[b] & isg[b])):
                if msk.any():
                    d = acc.setdefault((key, int(j[b])), [[], []])
                    d[0].append(err[b][msk].mean()); d[1].append(zero[b][msk].mean())
    out = {}
    for (key, jj), (e, z) in sorted(acc.items()):
        out.setdefault(key, {})[jj] = dict(n=len(e), err=float(np.mean(e)), zero_action=float(np.mean(z)))
    return dict(robot=robot_key, rep=rep_path, by_node_and_j=out)


def _gripper_mask(robot_key: str, N: int) -> np.ndarray:
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    s = Session(BUILDERS["pick_place"](workbench_robots()[robot_key](), 3_000_000, n_distractors=0), seed=3_000_000)
    g = np.zeros(N, bool)
    isg = np.asarray(_featurizer(s).aspace.is_gripper, bool)
    g[:len(isg)] = isg
    return g


def save_dagger(collect: dict, path: Path, meta: dict):
    rows = collect["rows"]
    np.savez_compressed(path, mu=np.stack(collect["mu"]), lv=np.stack(collect["lv"]),
                        rp=np.array([r[0] for r in rows], np.int32), j=np.array([r[1] for r in rows], np.int8),
                        node=np.stack([r[2] for r in rows]), n_nodes=np.array([r[3] for r in rows], np.int16),
                        local=np.stack([r[4] for r in rows]), a1=np.stack([r[5] for r in rows]),
                        meta=np.array(json.dumps(meta)))
