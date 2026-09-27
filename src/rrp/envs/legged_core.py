"""Legged tracker core shared by training (vectorised envs) and deployment (LeggedSession).

Policy-facing command: base-velocity group [vx (m/s, body frame), vy (m/s), wz (rad/s)].
Tracker output: joint position targets for the body's `policy_actuators` (held actuators are
servoed to the default pose).

PUBLIC tracker observation (deployable; identical in training and deployment):
    gyro (IMU, body frame) * 0.25, gravity direction in IMU frame (from IMU orientation, roll/pitch
    only), command * scales, q - q_default, qdot * 0.05, previous action, gait clock sin/cos.
PRIVILEGED critic extras (training only, never fed to the actor):
    base linear velocity (body frame), base height, foot contact flags, friction scale, push flag.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import mujoco
import numpy as np

CMD_SCALE = np.array([2.0, 2.0, 0.25])
# default swing-apex targets (m) when the body meta has no `swing_height` (gait_v2 clearance term)
SWING_HEIGHT = {"humanoid": 0.08, "biped": 0.08, "quadruped": 0.06, "hexapod": 0.03, "multipod": 0.03}


def quat_rotate_inv(q, v):
    """Rotate world vector v into the frame of unit quaternion q (w,x,y,z)."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ v


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class LeggedBinding:
    """Resolves a legged body's indices inside a compiled model (robot namespaced by prefix)."""

    def __init__(self, model: mujoco.MjModel, meta: dict, prefix: str = "r0_"):
        L = meta["legged"]
        self.model, self.meta, self.L, self.prefix = model, meta, L, prefix
        nid = lambda t, n: mujoco.mj_name2id(model, t, prefix + n)
        O = mujoco.mjtObj
        self.root_bid = nid(O.mjOBJ_BODY, L["root_body"])
        fj = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
              and model.jnt_bodyid[j] == self.root_bid]
        assert len(fj) == 1, "root free joint not found"
        self.qa = int(model.jnt_qposadr[fj[0]])
        self.da = int(model.jnt_dofadr[fj[0]])
        self.pol_act = np.array([nid(O.mjOBJ_ACTUATOR, a) for a in L["policy_actuators"]])
        self.held_act = np.array([nid(O.mjOBJ_ACTUATOR, a) for a in L["held_actuators"]], dtype=int)
        assert (self.pol_act >= 0).all() and (self.held_act >= 0).all()
        pj = model.actuator_trnid[self.pol_act, 0]
        self.pol_qadr = model.jnt_qposadr[pj]
        self.pol_dadr = model.jnt_dofadr[pj]
        hj = model.actuator_trnid[self.held_act, 0] if len(self.held_act) else np.zeros(0, int)
        self.held_qadr = model.jnt_qposadr[hj] if len(hj) else np.zeros(0, int)
        jn = lambda j: model.joint(j).name[len(prefix):]
        self.q0 = np.array([L["default_pose"][jn(j)] for j in pj])
        self.q0_held = np.array([L["default_pose"][jn(j)] for j in hj]) if len(hj) else np.zeros(0)
        self.lo = model.actuator_ctrlrange[self.pol_act, 0].copy()
        self.hi = model.actuator_ctrlrange[self.pol_act, 1].copy()
        self.effort = model.actuator_forcerange[self.pol_act, 1].copy()
        self.jlo = model.jnt_range[pj, 0].copy()
        self.jhi = model.jnt_range[pj, 1].copy()
        self.n = len(self.pol_act)
        self.foot_bids = [nid(O.mjOBJ_BODY, f) for f in L["foot_bodies"]]
        self.nf = len(self.foot_bids)
        self.floor = mujoco.mj_name2id(model, O.mjOBJ_GEOM, "floor")
        self.body_is_foot = np.full(model.nbody, -1)
        for i, b in enumerate(self.foot_bids):
            self.body_is_foot[b] = i
        # robot bodies (for "non-foot body touches floor" terminations)
        self.robot_bodies = np.array([b for b in range(model.nbody) if model.body(b).name.startswith(prefix)])
        self.is_robot_body = np.zeros(model.nbody, bool)
        self.is_robot_body[self.robot_bodies] = True
        # bodies allowed to touch the ground: feet and their descendants (toes etc.)
        self.allowed_ground = np.zeros(model.nbody, bool)
        for b in range(model.nbody):
            p = b
            while p > 0:
                if p in self.foot_bids:
                    self.allowed_ground[b] = True
                    break
                p = model.body_parentid[p]
        self.kind = L["kind"]
        self.biped = self.kind in ("humanoid", "biped")
        self.action_scale = float(L["action_scale"])
        self.period = float(L["gait_period"])
        self.cmd_ranges = L["command_ranges"]
        self.min_h = float(L["min_height_frac"]) * self.nominal_height()
        self.tilt_limit = float(L["tilt_limit"])
        self.foot_sids = [nid(O.mjOBJ_SITE, f) for f in L["foot_sites"]]
        self.swing_height = float(L.get("swing_height") or SWING_HEIGHT.get(self.kind, 0.05))
        self._foot_z0 = None
        imu = L["imu"]
        self.imu_quat = self._sensor_adr(prefix + imu["quat"])
        self.imu_gyro = self._sensor_adr(prefix + imu["gyro"])

    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid]) if sid >= 0 else None

    def nominal_height(self) -> float:
        L = self.L
        if L.get("nominal_height") is not None:
            return float(L["nominal_height"])
        d = mujoco.MjData(self.model)
        self.set_default(d, z=1.0)
        mujoco.mj_kinematics(self.model, d)
        fb = set(self.foot_bids)
        low = min(d.geom_xpos[g][2] - self.model.geom_rbound[g] for g in range(self.model.ngeom)
                  if self.model.geom_bodyid[g] in fb)
        L["nominal_height"] = float(1.0 - low)
        return L["nominal_height"]

    obs_dim = property(lambda self: 3 + 3 + 3 + 3 * self.n + 2)
    priv_dim = property(lambda self: 3 + 1 + self.nf + 2)

    def set_default(self, d: mujoco.MjData, z: float | None = None, xy=(0.0, 0.0), yaw: float = 0.0,
                    noise: float = 0.0, rng=None):
        d.qpos[self.qa:self.qa + 2] = xy
        d.qpos[self.qa + 2] = self.nominal_height() + 0.005 if z is None else z
        d.qpos[self.qa + 3:self.qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        q = self.q0 + (rng.uniform(-noise, noise, self.n) if (noise and rng is not None) else 0)
        d.qpos[self.pol_qadr] = np.clip(q, self.jlo, self.jhi)
        if len(self.held_qadr):
            d.qpos[self.held_qadr] = self.q0_held
        d.qvel[:] = 0
        d.ctrl[self.pol_act] = np.clip(self.q0, self.lo, self.hi)
        if len(self.held_act):
            d.ctrl[self.held_act] = self.q0_held

    # ---------------------------------------------------------------- public observation
    def imu(self, d: mujoco.MjData):
        """(quat, gyro) as the IMU reports them. IMU frame == root frame (identity site), so these
        equal the free-joint quaternion and local angular velocity (asserted in tests)."""
        return d.qpos[self.qa + 3:self.qa + 7], d.qvel[self.da + 3:self.da + 6]

    def public_obs(self, d: mujoco.MjData, cmd, last_action, phase) -> np.ndarray:
        quat, gyro = self.imu(d)
        g = quat_rotate_inv(quat, np.array([0, 0, -1.0]))
        return np.concatenate([gyro * 0.25, g, np.asarray(cmd) * CMD_SCALE, d.qpos[self.pol_qadr] - self.q0,
                               d.qvel[self.pol_dadr] * 0.05, last_action,
                               [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]]).astype(np.float32)

    def targets(self, action) -> np.ndarray:
        return np.clip(self.q0 + self.action_scale * np.asarray(action), self.lo, self.hi)

    # ---------------------------------------------------------------- privileged quantities
    def base_lin_vel_body(self, d):
        return quat_rotate_inv(d.qpos[self.qa + 3:self.qa + 7], d.qvel[self.da:self.da + 3])

    def contacts(self, d) -> tuple[np.ndarray, bool]:
        """(foot contact flags, bad_contact: a non-foot robot body touches the floor)."""
        fc = np.zeros(self.nf, bool)
        bad = False
        m = self.model
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self.floor:
                b = m.geom_bodyid[c.geom2]
            elif c.geom2 == self.floor:
                b = m.geom_bodyid[c.geom1]
            else:
                continue
            f = self.body_is_foot[b]
            if f >= 0:
                fc[f] = True
            elif self.allowed_ground[b]:
                p = b
                while self.body_is_foot[p] < 0:
                    p = m.body_parentid[p]
                fc[self.body_is_foot[p]] = True
            elif self.is_robot_body[b]:
                bad = True
        return fc, bad

    def stance(self, d):
        """Contact-point stance measurements (privileged; training reward + validation).

        Returns (fc, fn, slip, bad): per-foot contact flag, total normal force (N), normal-force-weighted
        horizontal speed of the foot AT ITS FLOOR CONTACT POINTS (m/s; true slip, not the ankle-origin speed),
        and whether a non-foot robot body touches the floor."""
        m = self.model
        fc = np.zeros(self.nf, bool)
        fn = np.zeros(self.nf)
        sv = np.zeros(self.nf)
        bad = False
        f6 = np.zeros(6)
        vel = {}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self.floor:
                b = m.geom_bodyid[c.geom2]
            elif c.geom2 == self.floor:
                b = m.geom_bodyid[c.geom1]
            else:
                continue
            if not self.allowed_ground[b]:
                if self.is_robot_body[b]:
                    bad = True
                continue
            p = b
            while self.body_is_foot[p] < 0:
                p = m.body_parentid[p]
            k = self.body_is_foot[p]
            fc[k] = True
            mujoco.mj_contactForce(m, d, i, f6)
            n = max(float(f6[0]), 0.0)
            if b not in vel:
                v6 = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, v6, 0)
                vel[b] = v6
            v6 = vel[b]
            vp = v6[3:6] + np.cross(v6[0:3], c.pos - d.xpos[b])
            fn[k] += n
            sv[k] += n * math.hypot(vp[0], vp[1])
        slip = np.where(fn > 1e-9, sv / np.maximum(fn, 1e-9), 0.0)
        return fc, fn, slip, bad

    def foot_clearance(self, d) -> np.ndarray:
        """Foot-site height above its standing height (m), per foot."""
        if self._foot_z0 is None:
            d0 = mujoco.MjData(self.model)
            self.set_default(d0)
            mujoco.mj_kinematics(self.model, d0)
            self._foot_z0 = np.array([d0.site_xpos[s][2] for s in self.foot_sids]) - 0.005
        return np.array([d.site_xpos[s][2] for s in self.foot_sids]) - self._foot_z0

    def tilt(self, d) -> float:
        g = quat_rotate_inv(d.qpos[self.qa + 3:self.qa + 7], np.array([0, 0, -1.0]))
        return float(math.acos(max(-1.0, min(1.0, -g[2]))))


PRIOR_TERMS = ("air_time", "clearance", "contact_phase", "stand_contact")        # gait-shaping priors: decay
NATURAL_TERMS = ("torque", "action_rate", "smooth", "power", "impact")           # natural objectives: ramp up
# everything else is PERMANENT (tracking, termination, orientation/height, lin_z/ang_xy, slip, limits, alive,
# stand_still): the stance-slip penalty never decays, otherwise skating returns.


@dataclass
class RewardCfg:
    track_lin: float = 1.5
    track_ang: float = 0.75
    lin_z: float = -2.0
    ang_xy: float = -0.05
    orient: float = -2.0
    torque: float = -0.02
    action_rate: float = -0.02
    limits: float = -2.0
    air_time: float = 1.0
    stand_still: float = -0.3
    alive: float = 0.2
    contact_phase: float = 0.0
    height: float = 0.0
    feet_slip: float = -0.05
    termination: float = -5.0
    stand_contact: float = 0.5     # zero command: all feet down (stance transition)
    sigma: float = 0.25
    # gait_v2 terms (0 in the legacy config)
    slip: float = 0.0              # x sum over feet of contact-point slip speed |v_xy| (m/s), loaded feet
    clearance: float = 0.0         # x sum over swinging feet of min(foot height / swing target, 1) (moving only)
    smooth: float = 0.0            # x second difference of actions (a - 2a1 + a2)^2, like action_rate
    power: float = 0.0             # x mechanical power / (m g max(|cmd_xy|, 0.25)): a per-step cost of transport
    impact: float = 0.0            # x clip(touchdown normal force / (m g) - 1, 0, 3), per touchdown
    # PERMANENT swing-clearance hinge floor (W1 task 1, 2026-09-27): at each touchdown during a moving command,
    # x clip((floor - swing apex) / floor, 0, 1)^2 per foot, floor = floor_frac * body swing_height. Never decays.
    clearance_floor: float = 0.0
    floor_frac: float = 0.6
    # turn-in-place terms (W1 task 2), PERMANENT: yaw_slip x sum over stance feet of |foot yaw rate| (rad/s), which forbids
    # pivoting on a planted foot; turn_step x agreement of foot contacts with the alternating gait clock during
    # pure-turn commands (a stepping turn), independent of the decaying contact_phase prior.
    yaw_slip: float = 0.0
    turn_step: float = 0.0
    # PERMANENT stepping floor (W1, D-103 task 2): during any moving/turning command, each foot that has stayed in contact longer than
    # stance_cap_frac x gait period is penalised: x clip((stance time - cap) / period, 0, 1) per foot. Forces a minimum swing frequency.
    stance_cap: float = 0.0
    stance_cap_frac: float = 0.75
    # clock-driven reference stepping motion (bipeds; humanoid-gym style "joint_pos" reward): during moving/turning commands the swing leg
    # of the gait clock follows hip_pitch -a, knee +2a, ankle_pitch -a (a = ref_amp x |sin 2 pi phase|). Reward x (exp(-2|q - q_ref|) -
    # 0.2 min(|q - q_ref|, 0.5)) over the six pitch joints. Gives the MEAN policy a stepping target (noise-only stepping was the h1/g1 failure).
    ref_step: float = 0.0
    ref_amp: float = 0.2
    turn_lin: float = 0.0          # dense yaw progress during pure-turn commands: x clip(w_z sign(c)/|c|, -0.5, 1.2)
    sigma_ang: float = 0.0         # yaw-rate tracking kernel width; 0 -> sigma (a sharper kernel keeps small turn commands informative)
    version: str = "gait_v1"
    # schedule (gait_v2): alpha in [0,1]; priors w0*(floor + (1-floor)(1-alpha)); natural w_min + alpha(w_max-w_min)
    alpha: float = 0.0
    prior_floor: float = 0.1
    natural_max: dict = field(default_factory=dict)

    def effective(self, alpha: float) -> "RewardCfg":
        """Weights at schedule position alpha (self holds the alpha=0 weights; natural_max the alpha=1 ones)."""
        a = float(min(1.0, max(0.0, alpha)))
        e = replace(self, alpha=a)
        for t in PRIOR_TERMS:
            setattr(e, t, getattr(self, t) * (self.prior_floor + (1 - self.prior_floor) * (1 - a)))
        for t, wmax in self.natural_max.items():
            setattr(e, t, getattr(self, t) + a * (wmax - getattr(self, t)))
        return e

    def weights(self) -> dict:
        return {k: v for k, v in asdict(self).items() if isinstance(v, float) and k not in ("alpha", "prior_floor")}

    @staticmethod
    def for_kind(kind: str, version: str = "gait_v1") -> "RewardCfg":
        if version == "gait_v2":
            # gait_v2 (contact track, 2026-09-26): v1 skated (stance-foot slip ~ body speed). Slip is now a
            # PERMANENT linear contact-point speed penalty (x20-40 stronger at 0.3 m/s than v1's -0.05 v^2);
            # swing-apex clearance, air time, contact phase and stand_contact are decaying priors; torque,
            # action rate, jerk, power (CoT) and impact are natural objectives that ramp with alpha.
            # Reference values: research/tracks/contact.md (legged_gym, humanoid-gym, unitree_rl_gym).
            nat = dict(torque=-0.1, action_rate=-0.08, smooth=-0.04, power=-0.3, impact=-0.2)
            if kind in ("humanoid", "biped"):
                return RewardCfg(orient=-5.0, alive=0.3, contact_phase=1.0, height=-20.0, air_time=1.0,
                                 stand_still=-0.5, termination=-10.0, sigma=0.1, track_lin=2.5, track_ang=2.0,
                                 feet_slip=0.0, slip=-1.0, clearance=1.0, torque=-0.02, action_rate=-0.02,
                                 smooth=-0.01, power=-0.02, impact=0.0, version=version, natural_max=nat)
            return RewardCfg(feet_slip=0.0, slip=-0.5, clearance=0.5, torque=-0.02, action_rate=-0.02, smooth=-0.01,
                             power=-0.02, impact=0.0, version=version, natural_max=nat)
        if kind in ("humanoid", "biped"):
            # v3: track_ang 1.0 -> 2.0 + turn-in-place commands (v2 walked but never turned)
            # v2 (after v1 converged to a stable non-walking stander): sharper tracking kernel, more
            # tracking weight, less alive bonus so standing still under a walk command is not optimal
            return RewardCfg(orient=-5.0, alive=0.3, contact_phase=0.4, height=-20.0, air_time=1.0,
                             stand_still=-0.5, termination=-10.0, sigma=0.1, track_lin=2.5, track_ang=2.0)
        return RewardCfg()


class LeggedEnv:
    """N independent MjData sharing one model; 50 Hz tracker rate, PD on physics substeps."""

    def __init__(self, module_factory, n_envs: int, seed: int, *, control_dt: float = 0.02,
                 episode_s: float = 20.0, friction_scale: float = 1.0, push: bool = True, obs_noise: float = 1.0,
                 contact: str = "v1", reward: str | None = None, randomize: bool = True,
                 reward_overrides: dict | None = None, actuator: str = "v1"):
        from rrp.physics.contact import ContactRandomizer, resolve
        from rrp.bodies.legged import standalone_model
        mod = module_factory()
        self.contact = resolve(contact)
        self.model, _, self.meta = standalone_model(mod, contact=self.contact)
        self.b = LeggedBinding(self.model, self.meta)
        self.cdr = None
        if self.contact == "v1":
            if friction_scale != 1.0:
                self.model.geom_friction[:, 0] *= friction_scale
            self.friction_scale = friction_scale
        else:   # contact_v2: per-worker contact/mass/CoM randomisation + per-episode latency
            self.cdr = ContactRandomizer(self.model, self.b.root_bid, np.random.default_rng([seed, 77])) \
                if randomize else None
            self.friction_scale = (self.cdr.mu / self.cdr.nominal_mu) if self.cdr else 1.0
        self.cfg0 = RewardCfg.for_kind(self.b.kind, reward or ("gait_v2" if self.contact == "v2" else "gait_v1"))
        if self.cfg0.version == "gait_v2" and not self.b.biped:
            # tracking kernel scaled to the body's command range: with sigma 0.25 a small robot (hexapod6, vx_max 0.3)
            # loses only ~0.18/step by standing still, less than the stance-slip cost, so gait_v2 converged to standing
            vmax = float(self.b.cmd_ranges["vx"][1])
            self.cfg0 = replace(self.cfg0, sigma=min(self.cfg0.sigma, (0.5 * vmax) ** 2))
        if reward_overrides:   # e.g. {"clearance_floor": -2.0}; base (alpha=0) weights, recorded in the train meta
            for k, v in reward_overrides.items():
                if not hasattr(self.cfg0, k):
                    raise KeyError(f"unknown reward term {k}")
            self.cfg0 = replace(self.cfg0, **{k: type(getattr(self.cfg0, k))(v) for k, v in reward_overrides.items()})
        self.cfg = self.cfg0.effective(0.0) if self.cfg0.version == "gait_v2" else self.cfg0
        self.sched = self.cfg0.version == "gait_v2"      # alpha schedule (critic sees alpha; the actor never does)
        self.priv_dim = self.b.priv_dim + (1 if self.sched else 0)
        self.act = None
        if actuator in ("v2", "v1lat"):   # rrp.physics.actuator (v1lat: ideal joints + sourced limits + 0-30 ms latency)
            from rrp.physics.actuator import ActuatorModel
            self.act = ActuatorModel(self.model, self.b, n_envs, np.random.default_rng([seed, 91]), name=self.meta["name"],
                                     mode=actuator)
        self.actuator = actuator
        self.turn_vx = 0.0
        self.slow_frac = 0.0      # bipeds: fraction of translational commands rescaled to 0.05-0.2 m/s (slow-gait mix)
        self.stance_t = np.zeros((n_envs, self.b.nf))
        import re as _re
        acts = self.meta["legged"]["policy_actuators"]
        def _idx(side, pat):
            return [k for k, a in enumerate(acts) if _re.search(side, a, _re.I) and _re.search(pat, a, _re.I)]
        self.ref_idx = None
        if self.b.biped:
            try:
                L = [_idx("left", "hip_pitch")[0], _idx("left", "knee")[0], _idx("left", "ankle_pitch|ankle$")[0]]
                Rr = [_idx("right", "hip_pitch")[0], _idx("right", "knee")[0], _idx("right", "ankle_pitch|ankle$")[0]]
                self.ref_idx = (np.array(L), np.array(Rr))
            except IndexError:
                self.ref_idx = None
        self.turn_cmd = np.zeros(n_envs, bool)
        self.turn_scale = 1.0     # curriculum (set_turn_scale); 1.0 = the v3 sampler unchanged
        self.turn_frac = 0.25
        self.mass = float(self.model.body_subtreemass[self.b.root_bid])
        self._gm_reset()
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.data = [mujoco.MjData(self.model) for _ in range(n_envs)]
        self.dt = control_dt
        self.substeps = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.max_steps = int(episode_s / control_dt)
        self.push = push
        self.obs_noise = obs_noise
        nA, nf = self.b.n, self.b.nf
        self.cmd = np.zeros((n_envs, 3))
        self.last_a = np.zeros((n_envs, nA))
        self.last_a2 = np.zeros((n_envs, nA))
        self.latency = np.zeros(n_envs, int)
        self.apex = np.zeros((n_envs, nf))
        self.ticks = 0
        self.phase = np.zeros(n_envs)
        self.t = np.zeros(n_envs, int)
        self.air = np.zeros((n_envs, nf))
        self.cmd_timer = np.zeros(n_envs, int)
        self.push_flag = np.zeros(n_envs)
        self.ep_ret = np.zeros(n_envs)
        self.stats = []
        for i in range(n_envs):
            self._reset(i)

    def set_alpha(self, alpha: float):
        if self.sched:
            self.cfg = self.cfg0.effective(alpha)

    def _gm_reset(self):
        # gate metrics accumulated over steps with a translational command (read by the trainer's alpha gate)
        self.gm = dict(steps=0, track_err=0.0, cmd=0.0, slip=0.0, speed=0.0, power=0.0, cot_den=0.0,
                       turn_steps=0, turn_cmd=0.0, turn_w=0.0)

    def set_turn_scale(self, scale: float):
        """Turn-in-place curriculum: pure-turn yaw-rate commands are drawn from +-[0.3, 1.0] x scale x wz_max."""
        self.turn_scale = float(min(1.0, max(0.05, scale)))

    def _sample_cmd(self, i):
        r = self.b.cmd_ranges
        c = np.array([self.rng.uniform(*r["vx"]), self.rng.uniform(*r["vy"]), self.rng.uniform(*r["wz"])])
        u = self.rng.random()
        if u < 0.15:
            c[:] = 0
        elif u < 0.45:        # forward + turn (the waypoint-teacher regime)
            c[1] = 0
        elif self.b.biped and u < 0.45 + self.turn_frac:   # v3 (bipeds): pure turn-in-place commands
            c[:2] = 0
            c[0] = self.turn_vx        # arc-to-in-place curriculum: a forward component that shrinks to 0
            c[2] = self.rng.choice([-1, 1]) * self.rng.uniform(0.3, 1.0) * self.turn_scale * self.b.cmd_ranges["wz"][1]
        if np.linalg.norm(c[:2]) < 0.05:
            c[:2] = 0
        if abs(c[2]) < 0.05:
            c[2] = 0
        if self.slow_frac and self.b.biped and np.linalg.norm(c[:2]) > 0.05 and self.rng.random() < self.slow_frac:
            c[:2] *= self.rng.uniform(0.05, 0.2) / np.linalg.norm(c[:2])
            c[2] *= 0.3
        self.turn_cmd[i] = bool(self.b.biped and 0.45 <= u < 0.45 + self.turn_frac and abs(c[2]) > 0)
        self.cmd[i] = c
        self.cmd_timer[i] = int(self.rng.integers(150, 300))

    def _reset(self, i):
        d = self.data[i]
        mujoco.mj_resetData(self.model, d)
        self.b.set_default(d, yaw=self.rng.uniform(-math.pi, math.pi), noise=0.05, rng=self.rng)
        mujoco.mj_forward(self.model, d)
        self.last_a[i] = 0
        self.last_a2[i] = 0
        self.apex[i] = 0
        self.latency[i] = self.cdr.latency() if (self.cdr and self.act is None) else 0
        self.stance_t[i] = 0
        if self.act is not None:
            self.act.reset(i, d.ctrl[self.b.pol_act].copy())
        self.phase[i] = self.rng.random()
        self.t[i] = 0
        self.air[i] = 0
        self.push_flag[i] = 0
        self.ep_ret[i] = 0
        self._sample_cmd(i)

    def obs(self, i):
        d = self.data[i]
        o = self.b.public_obs(d, self.cmd[i], self.last_a[i], self.phase[i])
        if self.obs_noise:
            nz = np.concatenate([np.full(3, 0.05), np.full(3, 0.03), np.zeros(3), np.full(self.b.n, 0.01),
                                 np.full(self.b.n, 0.05), np.zeros(self.b.n), np.zeros(2)]) * self.obs_noise
            o = o + (self.rng.standard_normal(o.shape) * nz).astype(np.float32)
        return o

    def priv(self, i, fc=None):
        d = self.data[i]
        if fc is None:
            fc, _ = self.b.contacts(d)
        return np.concatenate([self.b.base_lin_vel_body(d), [d.qpos[self.b.qa + 2] - self.b.nominal_height()],
                               fc.astype(float), [self.friction_scale - 1.0, self.push_flag[i]]]
                              + ([[self.cfg.alpha]] if self.sched else [])).astype(np.float32)

    def observe_all(self):
        return np.stack([self.obs(i) for i in range(self.n)]), np.stack([self.priv(i) for i in range(self.n)])

    def step(self, actions: np.ndarray):
        b, cfg, m = self.b, self.cfg, self.model
        O, P = [], []
        R = np.zeros(self.n)
        D = np.zeros(self.n, bool)
        T = np.zeros(self.n, bool)   # time-out (bootstrap)
        self.ticks += 1
        if self.cdr and self.ticks % self.cdr.R["resample_steps"] == 0:
            self.cdr.resample()
            self.friction_scale = self.cdr.mu / self.cdr.nominal_mu
        v2 = cfg.version == "gait_v2"
        for i in range(self.n):
            d = self.data[i]
            a = np.clip(actions[i], -5, 5)
            new_t = b.targets(a)
            lat = int(self.latency[i])
            if self.act is not None:
                self.act.command(i, new_t)
            elif lat == 0:
                d.ctrl[b.pol_act] = new_t
            pw = 0.0
            for k in range(self.substeps):
                if self.act is not None:
                    d.ctrl[b.pol_act] = self.act.substep_ctrl(i, d)
                elif lat and k == lat:
                    d.ctrl[b.pol_act] = new_t      # actuator/PD latency: previous targets for `lat` substeps
                mujoco.mj_step(m, d)
                if v2:
                    pw += float(np.sum(np.abs(d.actuator_force[b.pol_act] * d.qvel[b.pol_dadr])))
            pw /= self.substeps                    # mean mechanical power over the tick (W)
            tau2 = float(np.sum((d.actuator_force[b.pol_act] / b.effort) ** 2))
            self.phase[i] = (self.phase[i] + self.dt / b.period) % 1.0
            self.t[i] += 1
            self.cmd_timer[i] -= 1
            if self.push and self.rng.random() < self.dt / 5.0:
                kick = self.rng.uniform(-0.4, 0.4, 2) * (0.5 if b.biped else 1.0)
                d.qvel[b.da:b.da + 2] += kick
                self.push_flag[i] = 1.0
            else:
                self.push_flag[i] *= 0.9
            if v2:
                fc, fn, slip_v, bad = b.stance(d)
            else:
                fc, bad = b.contacts(d)
            v = b.base_lin_vel_body(d)
            quat, w = b.imu(d)
            g = quat_rotate_inv(quat, np.array([0, 0, -1.0]))
            c = self.cmd[i]
            r = cfg.track_lin * math.exp(-float(np.sum((c[:2] - v[:2]) ** 2)) / cfg.sigma)
            r += cfg.track_ang * math.exp(-float((c[2] - w[2]) ** 2) / (cfg.sigma_ang or cfg.sigma))
            r += cfg.lin_z * v[2] ** 2 + cfg.ang_xy * float(np.sum(w[:2] ** 2))
            r += cfg.orient * float(np.sum(g[:2] ** 2))
            r += cfg.torque * tau2 / b.n
            r += cfg.action_rate * float(np.sum((a - self.last_a[i]) ** 2)) / b.n * 4
            if cfg.smooth:
                r += cfg.smooth * float(np.sum((a - 2 * self.last_a[i] + self.last_a2[i]) ** 2)) / b.n * 4
            q = d.qpos[b.pol_qadr]
            span = b.jhi - b.jlo
            soft_lo, soft_hi = b.jlo + 0.05 * span, b.jhi - 0.05 * span
            r += cfg.limits * float(np.sum(np.clip(soft_lo - q, 0, None) + np.clip(q - soft_hi, 0, None)))
            moving = np.linalg.norm(c[:2]) > 0.05 or abs(c[2]) > 0.05
            first = fc & (self.air[i] > 0)
            r += cfg.air_time * float(np.sum((self.air[i] - 0.5 * b.period) * first)) * moving
            if cfg.clearance and moving:
                # swing-height reward (humanoid-gym style, positive): each foot in a bounded swing (off the floor
                # for < 0.6 gait period) earns min(height / target, 1). v2a's touchdown-apex PENALTY made never
                # lifting a foot optimal early in training (stander basin), so it was replaced.
                clr = np.clip(b.foot_clearance(d) / b.swing_height, 0.0, 1.0)
                swing = (~fc) & (self.air[i] < 0.6 * b.period)
                if b.biped:
                    r += cfg.clearance * float(np.sum(clr * swing))
                elif swing.any():
                    # gait_v2c (non-bipeds): mean over swinging feet, i.e. swing quality, not the number of feet in the air
                    # (the sum rewarded lifting more legs: anymal_c kicked its shanks, hexapod6 held legs up)
                    r += cfg.clearance * float(np.mean(clr[swing]))
            if cfg.clearance_floor:
                clr_f = b.foot_clearance(d)
                self.apex[i] = np.where(fc, self.apex[i], np.maximum(self.apex[i], clr_f))
                if moving:
                    fl = cfg.floor_frac * b.swing_height
                    short = np.clip((fl - self.apex[i]) / fl, 0.0, 1.0) ** 2
                    r += cfg.clearance_floor * float(np.sum(short * first))
                self.apex[i] = np.where(fc, 0.0, self.apex[i])
            if cfg.slip:
                # bipeds: sum over feet; gait_v2c non-bipeds: mean over feet in contact, so lifting legs does not reduce it
                r += cfg.slip * (float(np.sum(slip_v)) if b.biped else (float(np.mean(slip_v[fc])) * 2.0 if fc.any() else 0.0))
            if v2:
                cspd = float(np.linalg.norm(c[:2]))
                r += cfg.power * pw / (self.mass * 9.81 * max(cspd, 0.25))
                if cfg.impact:
                    r += cfg.impact * float(np.sum(np.clip(fn / (self.mass * 9.81) - 1.0, 0, 3) * first))
                if cspd > 0.05:
                    gm = self.gm
                    gm["steps"] += 1
                    gm["track_err"] += float(np.linalg.norm(c[:2] - v[:2]))
                    gm["cmd"] += cspd
                    gm["slip"] += float(np.mean(slip_v[fc])) if fc.any() else 0.0
                    gm["speed"] += float(np.linalg.norm(v[:2]))
                    gm["power"] += pw
                    gm["cot_den"] += self.mass * 9.81 * float(np.linalg.norm(v[:2]))
            self.air[i] = np.where(fc, 0.0, self.air[i] + self.dt)
            if not moving:
                r += cfg.stand_still * float(np.sum(np.abs(q - b.q0))) / b.n * 4
                r += cfg.stand_contact * float(np.mean(fc))
            pure_turn = bool(self.turn_cmd[i]) or (np.linalg.norm(c[:2]) < 0.05 and abs(c[2]) > 0.05)
            if cfg.yaw_slip:
                ys = 0.0
                for k, fb in enumerate(b.foot_bids):
                    if fc[k]:
                        v6 = np.zeros(6)
                        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, fb, v6, 0)
                        ys += abs(float(v6[2]))
                r += cfg.yaw_slip * ys
            self.stance_t[i] = np.where(fc, self.stance_t[i] + self.dt, 0.0)
            if cfg.ref_step and moving and self.ref_idx is not None:
                sp = math.sin(2 * math.pi * self.phase[i])
                aL, aR = cfg.ref_amp * max(-sp, 0.0), cfg.ref_amp * max(sp, 0.0)   # left swings when phase >= 0.5
                qr = b.q0.copy()
                for idx, a_ in ((self.ref_idx[0], aL), (self.ref_idx[1], aR)):
                    qr[idx] += np.array([-a_, 2 * a_, -a_])
                sel = np.concatenate(self.ref_idx)
                dn = float(np.linalg.norm(q[sel] - qr[sel]))
                r += cfg.ref_step * (math.exp(-2 * dn) - 0.2 * min(dn, 0.5))
            if cfg.stance_cap and moving:
                cap = cfg.stance_cap_frac * b.period
                r += cfg.stance_cap * float(np.sum(np.clip((self.stance_t[i] - cap) / b.period, 0.0, 1.0)))
            if cfg.turn_lin and pure_turn:
                r += cfg.turn_lin * float(np.clip(w[2] * np.sign(c[2]) / abs(c[2]), -0.5, 1.2))
            if cfg.turn_step and pure_turn and b.nf == 2:
                want = np.array([self.phase[i] < 0.55, self.phase[i] >= 0.45])
                r += cfg.turn_step * float(np.mean(fc == want))
            if v2 and pure_turn:
                self.gm["turn_steps"] += 1
                self.gm["turn_cmd"] += abs(float(c[2]))
                self.gm["turn_w"] += float(w[2]) * float(np.sign(c[2]))
            if cfg.contact_phase and b.nf == 2:
                if moving:
                    want = np.array([self.phase[i] < 0.55, self.phase[i] >= 0.45])  # left stance / right stance
                else:
                    want = np.array([True, True])
                r += cfg.contact_phase * float(np.mean(fc == want))
            if cfg.height:
                h = d.qpos[b.qa + 2]
                r += cfg.height * max(0.0, 0.93 * b.nominal_height() - h) ** 2
            # foot slip: feet in contact should not slide (privileged foot velocity)
            if cfg.feet_slip:
                slip = 0.0
                for k, fb in enumerate(b.foot_bids):
                    if fc[k]:
                        vel = np.zeros(6)
                        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, fb, vel, 0)
                        slip += float(np.sum(vel[3:5] ** 2))
                r += cfg.feet_slip * slip
            r += cfg.alive
            h = d.qpos[b.qa + 2]
            fell = bad or h < b.min_h or b.tilt(d) > b.tilt_limit or not np.isfinite(d.qpos).all()
            if fell:
                r += cfg.termination
            self.last_a2[i] = self.last_a[i]
            self.last_a[i] = a
            self.ep_ret[i] += r
            R[i] = r
            timeout = self.t[i] >= self.max_steps
            if fell or timeout:
                self.stats.append(dict(ret=float(self.ep_ret[i]), len=int(self.t[i]), fell=bool(fell)))
                D[i] = True
                T[i] = timeout and not fell
                self._reset(i)
                fc = None
            elif self.cmd_timer[i] <= 0:
                self._sample_cmd(i)
            O.append(self.obs(i))
            P.append(self.priv(i, fc))
        return np.stack(O), np.stack(P), R, D, T

    def pop_stats(self):
        s, self.stats = self.stats, []
        if self.sched and self.gm["steps"]:
            s.append(dict(gm=self.gm))
            self._gm_reset()
        return s
