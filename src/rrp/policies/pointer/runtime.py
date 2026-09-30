"""Pointer runtime (docs/architecture.md sections 3 and 5): packet sources, learned system 0 and the `Policy` factories.

`TeacherOracleSource` (ORACLE DIAGNOSTIC, privileged) encodes the scripted teacher's next 7 commands: it runs the
teacher on a shadow twin of each env (same task, seed and frame; CW is deterministic) ahead of the real episode and
counts shadow/real state-hash divergences at every packet. `make_pointer_oracle` = oracle packets -> engineered
system 0 behind the Policy interface (rrp.policies.latent.LatentStackPolicy). `PointerSystemI` / `make_pointer_latent`
(learned system i -> learned or engineered system 0) and `PointerBCPolicy` read only `features.public_features`.
"""
from __future__ import annotations

import numpy as np

from rrp.core.action import NativeCommand
from rrp.core.latent_action import LatentActionChunk
from rrp.core.system0 import System0Base
from rrp.policies.pointer.checkpoint import load_pointer_bundle
from rrp.policies.pointer.features import EventHistory, collate_public, public_features, screen_half
from rrp.policies.pointer.packet import (EngineeredSystem0, cmd_source, encode_commands, packet_ticks, pointer_packet,
                                         tick_slot)
from rrp.policies.relations.base import assert_deployable
from rrp.policies.pointer.spec import ENG_VERSION, KNOT_TIMES, POINTER_KINDS, PointerGeometry, VALIDITY_S, eng_layout


def _twin(env):
    from rrp.envs.computerworld import ComputerWorldEnv
    f = env.frame
    return ComputerWorldEnv(env.task_name, seed=env.seed, width=f.width, height=f.height, m_per_px=f.m_per_px,
                            depth=f.depth, dz=f.dz, control_hz=env.control_hz, theme=env.spec.provenance["theme"],
                            strings=env.strings)


def target_depth(env, x: float, y: float) -> float:
    """Stack depth (m) of the topmost visible widget whose box contains (x, y); 0 if none."""
    from rrp.envs.computerworld import scene_widgets, widget_position
    u, v = env.frame.m_to_px(x, y)
    hits = [w for w in scene_widgets(env.scene()) if w["visible"] and w["box"]
            and w["box"][0] <= u < w["box"][2] and w["box"][1] <= v < w["box"][3]]
    if not hits:
        return 0.0
    return float(widget_position(max(hits, key=lambda w: (w["layer"], w["order"])), env.frame)[2])


def env_widget_table(env) -> list[dict | None]:
    """The env's current slot-indexed `scene_widgets` table (`SlotRegistry.assign`, `env.slots`): 1:1 with
    `obs.object_descriptors` when called right after `env.observe()` with no `env.step()` in between, matching
    `widget_features`' `table` argument (D-144 R20 follow-up). Used only at LIVE rollout call sites
    (`TeacherOracleSource._encode`, `PointerSystemI.packets`, `PointerBCPolicy.act`) AND by training collect
    (`harness.train.pointer.collect_episode`): one featurizer path, so the stored UI fields are exactly what a rollout
    feeds the net."""
    from rrp.envs.computerworld import scene_widgets
    return env.slots.assign(scene_widgets(env.scene()))


class TeacherOracleSource:
    """ORACLE DIAGNOSTIC packet source (privileged): the scripted teacher's next ticks, run on a shadow twin env, encoded
    with the engineered encoding, or (`encoder` = a frozen PointerEncoder with its versions) as z = E(public features
    at t, teacher chunk) for the LEARNED system 0 (the arm's target-encoder oracle). Metric: shadow/real divergences
    (state hash at the packet tick)."""
    name = "cw_teacher_oracle"

    def __init__(self, validity: float = VALIDITY_S, encoder=None, versions: dict | None = None, device="cpu",
                 eng_version: str = ENG_VERSION):
        self.validity, self.E, self.device = validity, encoder, device
        self.eng_version = eng_layout(eng_version).version
        self.lsv = versions["latent_space_version"] if versions else self.eng_version
        self.rcv = versions["realizer_compat_version"] if versions else self.eng_version
        self.calls = 0

    def reset(self, envs):
        from rrp.policies.teachers.computerworld import CWTeacher
        self.st, self.hist = {}, {}
        for e in envs:
            sh = _twin(e)
            self.st[id(e)] = dict(shadow=sh, teacher=CWTeacher(sh, e.task_name), cmds=[], depths=[], ptrs=[],
                                  btns=[], hashes=[sh.state_hash()], done=False, divergences=0)
            self.hist[id(e)] = EventHistory()

    def featurizer(self, env):
        return None

    def executed(self, env, command) -> None:
        self.hist[id(env)].push(env.steps, None if command is None else command.groups, screen_half(env.spec))

    def _advance(self, st, upto: int):
        sh = st["shadow"]
        while len(st["cmds"]) < upto:
            c = None if st["done"] else st["teacher"].act()
            st["ptrs"].append(sh._qpos()[:2].copy())
            st["btns"].append(float(sh.pointer.button))
            if c is None:
                st["done"] = True
                st["cmds"].append(None)
                st["depths"].append(0.0)
                continue
            g = c.groups
            st["depths"].append(target_depth(sh, *g["pointer"]))
            sh.step(c)
            st["cmds"].append(g)
            st["hashes"].append(sh.state_hash())

    def _encode(self, e, obs, st, n, H):
        import torch
        geom = PointerGeometry.from_spec(e.spec)
        half = geom.half
        f = public_features(obs, half, self.hist[id(e)], n, table=env_widget_table(e))
        cmds, ptrs, btns = st["cmds"][n:n + H], st["ptrs"][n:n + H], st["btns"][n:n + H]
        a = dict(dxy=np.zeros((H, 2), np.float32), xy=np.zeros((H, 2), np.float32), btn=np.zeros(H, np.float32),
                 key=np.zeros(H, np.int64), valid=np.zeros(H, bool))
        for j, g in enumerate(cmds):
            if g is None:
                continue
            xy = np.asarray(g["pointer"], np.float32)
            a["dxy"][j] = (xy - ptrs[j]) / geom.step_m
            a["xy"][j] = xy / half
            a["btn"][j] = float(g["button"][0] >= 0.5)
            a["key"][j] = int(round(g.get("key", [-1])[0])) + 1
            a["valid"][j] = True
        b = collate_public([f], self.device)
        at = {k: torch.from_numpy(v)[None].to(self.device) for k, v in a.items()}
        with torch.no_grad():
            return self.E(b, at)[0][0].float().cpu().numpy()

    def packets(self, envs) -> list[LatentActionChunk]:
        out = []
        for e in envs:
            st, n = self.st[id(e)], e.steps
            dt = 1.0 / e.spec.control_hz
            H = len(packet_ticks(dt))
            self._advance(st, n + H)
            if n < len(st["hashes"]) and st["hashes"][n] != e.state_hash():
                st["divergences"] += 1
            obs = e.observe()
            if self.E is None:
                z = encode_commands(st["cmds"][n:n + H], dt, len(e.spec.space("key").vocab), st["depths"][n:n + H],
                                    version=self.eng_version)
            else:
                z = self._encode(e, obs, st, n, H)
            out.append(pointer_packet(e, obs, z, lsv=self.lsv, rcv=self.rcv,
                                      source="oracle" if self.E is None else "target_encoder_oracle", name=self.name,
                                      validity=self.validity))
        self.calls += len(envs)
        return out


def pointer_requirements(*, privileged: bool = False, tasks=None):
    from rrp.policies.base import Requirements
    return Requirements(POINTER_KINDS, groups=frozenset({"pointer", "button", "key"}),
                        observations=frozenset({"proprio", "object_descriptors", "language"}),
                        body_families=frozenset({"pointer"}), tasks=frozenset(tasks) if tasks else None,
                        privileged=privileged)


def make_pointer_oracle(*, replan_ticks: int = 4, name: str = "pointer_oracle", representation: str | None = None,
                        device: str = "cpu", eng_version: str = ENG_VERSION):
    """ORACLE DIAGNOSTIC: scripted-teacher look-ahead -> engineered packet -> SCRIPTED engineered system 0; with
    `representation` (a pointer rep checkpoint): -> z = E(public features, teacher chunk) -> its LEARNED system 0."""
    from rrp.policies.latent import LatentStackPolicy
    if representation is None:
        return LatentStackPolicy(TeacherOracleSource(eng_version=eng_version), None, replan_ticks=replan_ticks, name=name,
                                 source="oracle",
                                 version=f"oracle:teacher->{eng_version}|system0={EngineeredSystem0.label_for(eng_version)}",
                                 make_s0=lambda e: EngineeredSystem0(e, version=eng_version),
                                 requires=pointer_requirements(privileged=True))
    rb = load_pointer_bundle(representation, device)
    v = rb["versions"]
    R = rb["modules"]["R"]
    return LatentStackPolicy(TeacherOracleSource(encoder=rb["modules"]["E"], versions=v, device=device), R,
                             replan_ticks=replan_ticks, name=name, source="oracle", variant=rb["config"]["variant"],
                             version=f"oracle:E({representation})|system0=learned:{v['realizer_compat_version']}",
                             make_s0=lambda e: LearnedSystem0(R, e, latent_space_version=v["latent_space_version"],
                                                              realizer_compat_version=v["realizer_compat_version"],
                                                              device=device),
                             requires=pointer_requirements(privileged=True))




class LearnedSystem0(System0Base):
    """Learned pointer system 0 (PointerRealizer): z + phase + measured pointer/button -> this tick's command."""

    def __init__(self, R, env, *, latent_space_version: str, realizer_compat_version: str, device="cpu"):
        super().__init__(latent_space_version=latent_space_version, realizer_compat_version=realizer_compat_version)
        self.R, self.device = R.eval(), device
        self.spec_hash = env.body.spec_hash
        geom = PointerGeometry.from_spec(env.spec)
        self.half, self.dt, self.step_m = geom.half, geom.dt, geom.step_m

    @property
    def robot_spec_hash(self) -> str:
        return self.spec_hash

    def tick(self, env, controller_version=None) -> NativeCommand | None:
        import torch
        now = float(env.time)
        p = self.packet
        if p is None or now > p.valid_until - 1e-9:
            if p is not None:
                self.invalidate("expired", now)
            self.stats.fallback_holds += 1
            return None
        ph = now - p.valid_from
        if tick_slot(ph, self.dt, p.knot_times) is None:
            self.stats.fallback_holds += 1
            return None
        obs = env.observe()
        q = np.asarray(obs.measured_node_state.qpos[:2], np.float32)
        btn = float(obs.declared_sensor_channels[0].values[0])
        dev = self.device
        with torch.no_grad():
            dxy, bl, kl = self.R(torch.from_numpy(np.asarray(p.z, np.float32))[None].to(dev),
                                 torch.tensor([ph], dtype=torch.float32, device=dev),
                                 torch.from_numpy(q / self.half)[None].to(dev), torch.tensor([btn], device=dev))
        d = np.clip(dxy[0].float().cpu().numpy(), -1.0, 1.0) * self.step_m
        g = {"pointer": [float(q[0] + d[0]), float(q[1] + d[1])], "button": [1.0 if float(bl[0]) > 0 else 0.0]}
        k = int(kl[0].argmax())
        if k > 0:
            g["key"] = [float(k - 1)]
        self.stats.ticks += 1
        return NativeCommand(controller_version=controller_version or "cw_pointer.v1", groups=g, source=cmd_source(p))


def flow_noise(policy_seed: int, env_seed: int, packet_idx: int, shape) -> np.ndarray:
    """Initial flow noise of ONE packet, a pure function of (policy seed, env seed, packet index): the same episode
    gets the same noise whatever else is in the batch or was rolled out before it (paired interventions, batch-1 vs
    batch-N equality). Seeds a fresh CPU generator from a hash of the triple (never a shared, order-dependent one)."""
    import hashlib

    import torch
    key = int.from_bytes(hashlib.sha256(f"pointer-flow-noise|{policy_seed}|{env_seed}|{packet_idx}".encode())
                         .digest()[:8], "little") % (2 ** 63)
    return torch.randn(tuple(shape), generator=torch.Generator().manual_seed(key)).numpy()


class PointerSystemI:
    """Learned system i: public features (+ own event history) -> flow sample -> packet. `target="latent"` emits the
    learned latent (for LearnedSystem0); `target="eng"` emits the engineered encoding (for EngineeredSystem0).
    Flow noise per packet is `flow_noise(seed, env.seed, packet index of that env)`; `reset` restarts the counters."""

    def __init__(self, flow, *, lsv: str, rcv: str, device="cpu", nfe=8, seed=0, name="pointer_system_i",
                 validity=VALIDITY_S):
        self.flow, self.lsv, self.rcv, self.device, self.nfe = flow.eval(), lsv, rcv, device, nfe
        assert_deployable(flow.ctx.specs)              # a learned system i never runs on privileged (gt) factor sources
        self.seed = int(seed)
        self.name, self.validity = name, validity
        self.calls = 0
        self.hist: dict[int, EventHistory] = {}
        self.n_packets: dict[int, int] = {}

    def reset(self, envs):
        self.hist = {id(e): EventHistory() for e in envs}
        self.n_packets = {id(e): 0 for e in envs}

    def featurizer(self, env):
        return None

    def executed(self, env, command) -> None:
        self.hist[id(env)].push(env.steps, None if command is None else command.groups, screen_half(env.spec))

    def packets(self, envs) -> list[LatentActionChunk]:
        import torch
        obs = [e.observe() for e in envs]
        fs = [public_features(o, screen_half(e.spec), self.hist[id(e)], e.steps, table=env_widget_table(e))
              for e, o in zip(envs, obs)]
        shape = (len(KNOT_TIMES), 1, self.flow.dz)
        noise = np.stack([flow_noise(self.seed, e.seed, self.n_packets[id(e)], shape) for e in envs])
        for e in envs:
            self.n_packets[id(e)] += 1
        with torch.no_grad():
            z = self.flow.sample(collate_public(fs, self.device), nfe=self.nfe,
                                 noise=torch.from_numpy(noise)).float().cpu().numpy()
        self.calls += len(envs)
        return [pointer_packet(e, o, z[i], lsv=self.lsv, rcv=self.rcv, source="learned", name=self.name,
                               sampling=dict(nfe=self.nfe, sampler="euler"), validity=self.validity)
                for i, (e, o) in enumerate(zip(envs, obs))]


def make_pointer_latent(*, flow: str, representation: str | None = None, device: str = "cpu", replan_ticks: int = 4,
                        nfe: int = 8, seed: int = 0, name: str = "pointer_latent"):
    """Learned route: system i (PointerFlow checkpoint `flow`) -> packet -> learned system 0 (the realizer of the flow's
    frozen representation), or, for a flow trained on the engineered encoding, -> the SCRIPTED engineered system 0."""
    from rrp.policies.latent import LatentStackPolicy
    fb = load_pointer_bundle(flow, device)
    variant, target = fb["config"]["variant"], fb["config"]["target"]
    tasks = fb["config"].get("tasks")
    if target == "eng":
        ev = fb["versions"]["latent_space_version"]          # the checkpoint's packet version tag (v1 / v2)
        si = PointerSystemI(fb["modules"]["S"], lsv=ev, rcv=ev, device=device, nfe=nfe, seed=seed,
                            name=f"{name}:{flow}")
        return LatentStackPolicy(si, None, replan_ticks=replan_ticks, name=name, source="learned", variant=variant,
                                 version=f"learned:{flow}|system0={EngineeredSystem0.label_for(ev)}",
                                 make_s0=lambda e: EngineeredSystem0(e, version=ev),
                                 requires=pointer_requirements(tasks=tasks))
    rb = load_pointer_bundle(representation or fb["config"]["representation"], device)
    lsv, rcv = rb["versions"]["latent_space_version"], rb["versions"]["realizer_compat_version"]
    if fb["versions"]["latent_space_version"] != lsv:
        from rrp.core.errors import ControllerRejection
        raise ControllerRejection(f"flow {flow} was trained on latent space {fb['versions']['latent_space_version']}, "
                                  f"representation is {lsv}", code="latent_space_mismatch")
    si = PointerSystemI(fb["modules"]["S"], lsv=lsv, rcv=rcv, device=device, nfe=nfe, seed=seed, name=f"{name}:{flow}")
    R = rb["modules"]["R"]
    return LatentStackPolicy(si, R, replan_ticks=replan_ticks, name=name, source="learned", variant=variant,
                             version=f"learned:{flow}|system0=learned:{rcv}",
                             make_s0=lambda e: LearnedSystem0(R, e, latent_space_version=lsv,
                                                              realizer_compat_version=rcv, device=device),
                             requires=pointer_requirements(tasks=tasks))


class PointerBCPolicy:
    """BC baseline (source bc): the same public features; every `replan_ticks` ticks a 7-tick chunk of absolute
    pointer / button / key commands, executed tick by tick (CW has no chunk executor)."""

    def __init__(self, bundle: dict, *, path: str, replan_ticks: int = 4, device: str = "cpu", name: str = "pointer_bc"):
        from rrp.policies.base import PolicyInfo
        self.net, self.replan, self.device = bundle["modules"]["BC"], int(replan_ticks), device
        assert_deployable(self.net.ctx.specs)
        self.info = PolicyInfo(name, "bc", f"bc:{path}",
                               pointer_requirements(tasks=bundle["config"].get("tasks")), "bc")

    def reset(self, spec, task, seeds, *, envs=None):
        self.envs = list(envs)
        self.hist = [EventHistory() for _ in self.envs]
        self.chunk = [None] * len(self.envs)
        self.t = [0] * len(self.envs)
        self.t0 = [0] * len(self.envs)

    def act(self, obs):
        import torch
        from rrp.policies.base import Act
        idx = sorted(obs)
        need = [i for i in idx if self.t[i] % self.replan == 0 or self.chunk[i] is None]
        if need:
            fs = [public_features(obs[i], screen_half(self.envs[i].spec), self.hist[i], self.envs[i].steps,
                                  table=env_widget_table(self.envs[i])) for i in need]
            with torch.no_grad():
                xy, bl, kl = self.net(collate_public(fs, self.device))
            for n, i in enumerate(need):
                half = screen_half(self.envs[i].spec)
                self.chunk[i] = [(xy[n, j].float().cpu().numpy() * half, float(bl[n, j]), int(kl[n, j].argmax()))
                                 for j in range(xy.shape[1])]
                self.t0[i] = self.t[i]
        out = {}
        for i in idx:
            j = self.t[i] - self.t0[i]
            (x, y), b, k = self.chunk[i][min(j, len(self.chunk[i]) - 1)]
            g = {"pointer": [float(x), float(y)], "button": [1.0 if b > 0 else 0.0]}
            if k > 0:
                g["key"] = [float(k - 1)]
            self.hist[i].push(self.envs[i].steps, g, screen_half(self.envs[i].spec))
            out[i] = Act(NativeCommand(controller_version="cw_pointer.v1", groups=g, source="bc"))
            self.t[i] += 1
        return out


def make_pointer_bc(*, checkpoint: str, device: str = "cpu", replan_ticks: int = 4, name: str = "pointer_bc"):
    return PointerBCPolicy(load_pointer_bundle(checkpoint, device), path=checkpoint, replan_ticks=replan_ticks,
                           device=device, name=name)
