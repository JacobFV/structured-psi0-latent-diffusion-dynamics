"""Batched GPU physics for harness evaluation (`eval_backend: warp`; docs/architecture.md section 14.8).

The episodes stay ordinary CPU `Session` objects: tasks, judges, hooks, detectors, controllers, trackers and policies are the CPU
code. What moves to MuJoCo Warp is the physics integration of a control step, for all running episodes at once:

  `BatchStepper.step({i: command})` drives each env's `_step_gen` (envs/mujoco/session.py `Integrate`): the generators yield their
  substeps, `WarpPool.integrate` uploads each episode's state (qpos, qvel, qacc_warmstart, act, mocap) and per-substep ctrl, advances
  substeps 0..n-2 on the GPU (one CUDA graph launch per substep), downloads the state and runs the LAST substep with CPU `mj_step`.
  That last CPU substep keeps the derived quantities (xpos, contacts, sensors, actuator forces) the ones CPU `mj_step` leaves, so
  everything the Sessions read after the step has CPU semantics; only the state trajectory carries Warp's float32 / solver differences.
  Because the state is re-uploaded every round, CPU-side edits between steps (resets, teleports, hooks) stay coherent.

Episodes are grouped by model: identical up to the per-world batchable Model fields (the arm's object placement changes
`body_pos / body_quat / qpos0` and derived inertia terms, which become batched model fields). Different structure (the arm's distractor
count) -> another group. Anything Warp cannot build raises `BatchUnsupported` with the reason; the caller (`harness.rollout`) falls back to CPU
and records the reason in the episode rows.
"""

import dataclasses  # no `from __future__ import annotations`: warp kernels resolve their annotations from the defining scope

import mujoco
import numpy as np

from rrp.envs.mujoco.session import Integrate
from rrp.envs.warp.model import _wp


class BatchUnsupported(RuntimeError):
    """The batched Warp stepper cannot run these envs (the reason is recorded; the rollout falls back to CPU)."""


def env_reason(env) -> str | None:
    """None when `env`'s physics can be batched, else why not."""
    if not hasattr(env, "_step_gen"):
        return f"{type(env).__name__} has no step generator (not a MuJoCo Session)"
    return env.batch_reason()


def _batched_fields() -> frozenset:
    _wp()
    from mujoco_warp._src import types, warp_util
    return frozenset(f.name for f in dataclasses.fields(types.Model)
                     if warp_util.is_array_spec(f.type) and getattr(f.type, "shape", ()) and f.type.shape[0] == "*")


def _arrays(m) -> dict[str, np.ndarray]:
    out = {}
    for k in dir(m):
        if k.startswith("_"):
            continue
        try:
            v = getattr(m, k)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, np.ndarray):
            out[k] = v
    return out


def _opt_key(m) -> tuple:
    o = m.opt
    return (float(o.timestep), int(o.cone), int(o.solver), int(o.iterations), int(o.ls_iterations), int(o.integrator),
            float(o.impratio), float(o.tolerance), int(o.disableflags), int(o.enableflags), tuple(map(float, o.gravity)))


def group_models(models: list, *, atol: float = 1e-8) -> list[list[int]]:
    """Indices of `models` grouped so a group differs only in Warp-batchable Model fields (or float noise below `atol`)."""
    batched = _batched_fields()
    arrs = [_arrays(m) for m in models]
    groups: list[list[int]] = []
    for i, a in enumerate(arrs):
        for g in groups:
            r = arrs[g[0]]
            if _opt_key(models[g[0]]) == _opt_key(models[i]) and r.keys() == a.keys() and all(
                    r[k].shape == a[k].shape and (k in batched or (np.allclose(r[k], a[k], rtol=atol, atol=atol)
                                                                    if r[k].dtype.kind in "fc" else np.array_equal(r[k], a[k])))
                    for k in a):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


class _Group:
    """One structure group: a batched Warp model + data over its worlds, with a CUDA-graphed substep."""

    def __init__(self, models, device, nconmax, njmax):
        wp, mjw = _wp()
        self.wp, self.mjw, self.device, self.models = wp, mjw, device, models
        self.nw, m0 = len(models), models[0]
        self.m0, self.dt = m0, float(m0.opt.timestep)
        batched = _batched_fields()
        arrs = [_arrays(m) for m in models]
        self.batched_fields = sorted(k for k in arrs[0] if k in batched and any(not np.array_equal(arrs[0][k], a[k]) for a in arrs[1:]))
        try:
            with wp.ScopedDevice(device):
                self.mw = mjw.put_model(m0, batch_sizes={k: self.nw for k in self.batched_fields})
                for k in self.batched_fields:
                    arr = getattr(self.mw, k)
                    stack = np.stack([a[k] for a in arrs])
                    arr.assign(wp.array(stack.astype(np.float32) if stack.dtype.kind == "f" else stack, dtype=arr.dtype, device=device))
                self.dw = mjw.make_data(m0, nworld=self.nw, nconmax=nconmax, njmax=njmax)
        except NotImplementedError as e:
            raise BatchUnsupported(f"mujoco_warp does not support this model: {e}") from e
        self.nu = m0.nu
        self.k = wp.zeros(1, dtype=wp.int32, device=device)
        self.energy = wp.zeros(self.nw, dtype=wp.float32, device=device)
        self.eidx = wp.zeros(max(self.nu, 1), dtype=wp.int32, device=device)       # actuator ids of the energy sum
        self.sched = None                                                         # (world, substep, nu), allocated per round length
        self._cap = None                                                          # (n_gpu_substeps, neidx) the graph was captured for
        self.graph = None
        self.peak_contacts = 0
        self._kernels()
        self.host = dict(qpos=np.zeros((self.nw, m0.nq), np.float32), qvel=np.zeros((self.nw, m0.nv), np.float32),
                         warm=np.zeros((self.nw, m0.nv), np.float32), act=np.zeros((self.nw, m0.na), np.float32),
                         mpos=np.zeros((self.nw, m0.nmocap, 3), np.float32), mquat=np.zeros((self.nw, m0.nmocap, 4), np.float32))

    def _kernels(self):
        wp = self.wp

        @wp.kernel
        def set_ctrl(sched: wp.array3d(dtype=wp.float32), k: wp.array(dtype=wp.int32), ctrl: wp.array2d(dtype=wp.float32)):
            w, u = wp.tid()
            ctrl[w, u] = sched[w, k[0], u]

        @wp.kernel
        def tick(k: wp.array(dtype=wp.int32)):
            k[0] = k[0] + 1

        @wp.kernel
        def add_energy(f: wp.array2d(dtype=wp.float32), v: wp.array2d(dtype=wp.float32), idx: wp.array(dtype=wp.int32),
                       n: int, dt: float, e: wp.array(dtype=wp.float32)):
            w = wp.tid()
            s = float(0.0)
            for j in range(n):
                a = idx[j]
                s += wp.abs(f[w, a] * v[w, a])
            e[w] = e[w] + s * dt

        self._k_set, self._k_tick, self._k_energy = set_ctrl, tick, add_energy

    def _substep(self, neidx):
        wp, d, dev = self.wp, self.dw, self.device
        if self.nu:
            wp.launch(self._k_set, dim=(self.nw, self.nu), inputs=[self.sched, self.k, d.ctrl], device=dev)
        self.mjw.step(self.mw, d)
        if neidx:
            wp.launch(self._k_energy, dim=self.nw, inputs=[d.actuator_force, d.actuator_velocity, self.eidx, neidx, self.dt, self.energy],
                      device=dev)
        wp.launch(self._k_tick, dim=1, inputs=[self.k], device=dev)

    def _prepare(self, ngpu, neidx):
        """(Re)allocate the schedule buffer and (re)capture the one-substep graph when the round length / energy sum changed."""
        wp = self.wp
        if self._cap == (ngpu, neidx):
            return
        self.sched = wp.zeros((self.nw, ngpu, max(self.nu, 1)), dtype=wp.float32, device=self.device)
        self.graph = None
        if wp.get_device(self.device).is_cuda:
            self._substep(neidx)                                  # module load / warm-up before capture
            with wp.ScopedCapture(device=self.device) as cap:
                self._substep(neidx)
            self.graph = cap.graph
        self._cap = (ngpu, neidx)

    def integrate(self, reqs: dict[int, Integrate], datas: list) -> None:
        """Advance the worlds in `reqs` (world index -> Integrate); `datas[j]` is world j's CPU MjData (state in, state out)."""
        wp, d, h, nu = self.wp, self.dw, self.host, self.nu
        ns = {r.n for r in reqs.values()}
        if len(ns) != 1:
            raise BatchUnsupported(f"substep counts differ inside one model group: {sorted(ns)}")
        n = ns.pop()
        ids = next((np.atleast_1d(np.asarray(r.energy_act, np.int32)) for r in reqs.values() if r.energy_act is not None), None)
        neidx = 0 if ids is None else len(ids)
        t0 = {j: float(datas[j].time) for j in reqs}
        if n > 1:
            for j, dj in enumerate(datas):                        # every world's state goes up; idle worlds simply hold
                h["qpos"][j], h["qvel"][j], h["warm"][j] = dj.qpos, dj.qvel, dj.qacc_warmstart
                if h["act"].shape[1]:
                    h["act"][j] = dj.act
                if h["mpos"].shape[1]:
                    h["mpos"][j], h["mquat"][j] = dj.mocap_pos, dj.mocap_quat
            if any(np.any(dj.xfrc_applied) or np.any(dj.qfrc_applied) for dj in datas):
                raise BatchUnsupported("externally applied forces (xfrc_applied / qfrc_applied) are not uploaded to Warp")
            self._prepare(n - 1, neidx)
            sched = np.empty((self.nw, n - 1, max(nu, 1)), np.float32)
            for j, dj in enumerate(datas):
                sched[j, :, :nu] = dj.ctrl
            for j, r in reqs.items():
                sched[j, :, :nu] = r.ctrl[:n - 1] if r.ctrl.ndim == 2 else r.ctrl
            self.sched.assign(sched)
            if neidx:
                self.eidx.assign(np.concatenate([ids, np.zeros(self.eidx.shape[0] - neidx, np.int32)]))
            d.qpos.assign(h["qpos"]); d.qvel.assign(h["qvel"]); d.qacc_warmstart.assign(h["warm"])
            if h["act"].shape[1]:
                d.act.assign(h["act"])
            if h["mpos"].shape[1]:
                d.mocap_pos.assign(h["mpos"]); d.mocap_quat.assign(h["mquat"])
            self.k.zero_(); self.energy.zero_()
            for _ in range(n - 1):
                if self.graph is not None:
                    wp.capture_launch(self.graph)
                else:
                    self._substep(neidx)
            wp.synchronize_device(self.device)
            q, v, w_ = d.qpos.numpy(), d.qvel.numpy(), d.qacc_warmstart.numpy()
            a = d.act.numpy() if h["act"].shape[1] else None
            mp = d.mocap_pos.numpy() if h["mpos"].shape[1] else None
            e = self.energy.numpy() if neidx else None
            self.peak_contacts = max(self.peak_contacts, int(d.nacon.numpy()[0]))
        for j, r in reqs.items():
            dj = datas[j]
            if n > 1:
                dj.qpos[:], dj.qvel[:], dj.qacc_warmstart[:] = q[j], v[j], w_[j]
                if a is not None:
                    dj.act[:] = a[j]
                dj.time = t0[j] + (n - 1) * self.dt
                r.energy = float(e[j]) if (neidx and r.energy_act is not None) else 0.0
            dj.ctrl[:] = r.ctrl_at(n - 1)
            mujoco.mj_step(self.models[j], dj)
            r.accumulate(dj, self.dt)


class WarpPool:
    """The Warp side for a list of envs: structure groups, each batched; `integrate({env index: Integrate})`."""

    def __init__(self, envs, device: str | None = None, *, nconmax: int = 96, njmax: int = 768):
        wp, _ = _wp()
        self.device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
        self.envs = envs
        models = [e.model for e in envs]
        self.groups: list[tuple[_Group, list[int]]] = []
        self.where: dict[int, tuple[int, int]] = {}
        for gi, idx in enumerate(group_models(models)):
            self.groups.append((_Group([models[i] for i in idx], self.device, nconmax, njmax), idx))
            for j, i in enumerate(idx):
                self.where[i] = (gi, j)

    def integrate(self, reqs: dict[int, Integrate], datas: dict | None = None) -> None:
        """`datas` (env index -> MjData) replaces an env's own data (the parity runner integrates a copy, leaving the env alone)."""
        by: dict[int, dict[int, Integrate]] = {}
        for i, r in reqs.items():
            gi, j = self.where[i]
            by.setdefault(gi, {})[j] = r
        for gi, rs in by.items():
            g, idx = self.groups[gi]
            g.integrate(rs, [(datas or {}).get(i, self.envs[i].data) for i in idx])

    def describe(self) -> dict:
        return dict(device=self.device, groups=[dict(worlds=g.nw, batched_fields=g.batched_fields, peak_contacts=g.peak_contacts)
                                                for g, _ in self.groups])


class BatchStepper:
    """`step({i: command}) -> {i: StepResult | Exception}`: one control step of the envs in `commands`, physics batched on Warp."""

    def __init__(self, envs, device: str | None = None, **kw):
        for i, e in enumerate(envs):
            if (why := env_reason(e)) is not None:
                raise BatchUnsupported(f"env {i}: {why}")
        self.envs = envs
        self.pool = WarpPool(envs, device, **kw)

    def step(self, commands: dict) -> dict:
        out: dict = {}
        live: dict = {}

        def advance(i, g):
            try:
                live[i] = (g, g.send(None))
            except StopIteration as e:
                out[i] = e.value
            except Exception as ex:  # noqa: BLE001  (per-episode: the rollout turns it into that episode's crash)
                out[i] = ex

        for i, c in commands.items():
            advance(i, self.envs[i]._step_gen(c))
        while live:
            cur, live = live, {}
            self.pool.integrate({i: r for i, (_, r) in cur.items()})
            for i, (g, _) in cur.items():
                advance(i, g)
        return out
