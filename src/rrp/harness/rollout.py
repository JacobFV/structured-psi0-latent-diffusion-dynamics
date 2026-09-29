"""The one episode loop (docs/architecture.md section 6): any Policy x Env x Task that `negotiate` accepts.

Episodes run in lock-step batches. Each tick the policy sees only the running episodes (a dict keyed by episode index,
so per-episode state and random streams do not depend on which other episodes are still running), its commands are
applied, the task's judge decides termination, and hooks observe or intervene. Every special case of the old
per-family loops (packet edits, perturbations, recorders, DAgger collection, video) is a Hook.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

from rrp.core.errors import ControllerRejection, StaleActionError
from rrp.policies.base import Act, Policy, negotiate
from rrp.tasks.spec import Judgement, TaskSpec


class Hook(Protocol):
    """All methods optional (rollout calls the ones a hook defines)."""

    def on_reset(self, i: int, env, obs) -> None: ...

    def on_act(self, i: int, obs, act: Act) -> Act: ...          # packet / chunk / command edits

    def on_step(self, i: int, env, act: Act, step) -> None: ...  # recorders, perturbations, DAgger collection

    def on_end(self, i: int, env, ep: "Episode") -> dict: ...    # metrics merged into Episode.metrics


@dataclass
class Episode:
    seed: int
    task: str
    env_id: str
    body: str
    policy: str
    source: str
    outcome: str
    failure_reason: str | None
    success_public: bool | None
    success_privileged: bool | None
    steps: int
    time: float
    wall_s: float
    metrics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def row(self) -> dict:
        return asdict(self)


class Incompatible(RuntimeError):
    def __init__(self, reasons):
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


def _call(hooks, name, *a):
    out = None
    for h in hooks:
        f = getattr(h, name, None)
        if f is not None:
            r = f(*a)
            out = r if r is not None else out
            if name == "on_act" and r is not None:
                a = (a[0], a[1], r)
    return out


def rollout(make_env: Callable[[int], object], policy: Policy, task: TaskSpec, seeds: Sequence[int], *,
            batch: int = 8, max_seconds: float | None = None, max_steps: int | None = None,
            hooks: Sequence = ()) -> list[Episode]:
    """make_env(seed) returns an env already reset to `seed` (every registered factory does).

    Budget: `max_seconds` (default task.max_seconds) of env time; `max_steps` additionally ends an episode after that
    many control ticks (the task judge is then asked with the budget spent, so its timeout rule applies exactly).
    A hook's on_reset may return a done Judgement (e.g. "infeasible") to end that episode before its first tick."""
    max_s = task.max_seconds if max_seconds is None else max_seconds
    info = policy.info
    episodes: list[Episode] = []
    for b0 in range(0, len(seeds), batch):
        group = list(seeds[b0:b0 + batch])
        envs = [make_env(sd) for sd in group]
        spec = envs[0].spec
        c = negotiate(info, spec, task)
        if not c.ok:
            raise Incompatible(c.reasons)
        policy.reset(spec, task, group, envs=envs)
        obs = {i: e.observe() for i, e in enumerate(envs)}
        t0 = {i: float(getattr(o, "sensor_time", getattr(o, "time", 0.0))) for i, o in obs.items()}
        state = {i: dict(steps=0, rej=0, chunk_rej=0, chunks=0, packets=0, wall=time.time(), judgement=None, note="")
                 for i in obs}
        running = set(obs)
        for i, e in enumerate(envs):
            j = _call(hooks, "on_reset", i, e, obs[i])
            if isinstance(j, Judgement) and j.done:
                state[i]["judgement"] = j
                running.discard(i)
        while running:
            idx = sorted(running)
            try:
                acts = policy.act({i: obs[i] for i in idx})
            except Exception as ex:  # noqa: BLE001  (a policy crash ends its episodes, recorded)
                for i in idx:
                    state[i].update(judgement=Judgement(True, "crash", "policy_exception"), note=repr(ex)[:300])
                running.clear()
                break
            for i in idx:
                env, st = envs[i], state[i]
                a = _call(hooks, "on_act", i, obs[i], acts[i]) or acts[i]
                st["chunks"] += a.chunk is not None
                st["packets"] += a.packet is not None
                try:
                    if a.chunk is not None:
                        try:
                            env.submit_chunk(a.chunk, execute_prefix=a.info.get("execute_prefix"))
                        except (StaleActionError, ControllerRejection):
                            st["chunk_rej"] += 1
                    step = env.step(a.command)
                except FloatingPointError as ex:
                    st.update(judgement=Judgement(True, "crash", "physics_divergence"), note=str(ex)[:300])
                    running.discard(i)
                    continue
                except Exception as ex:  # noqa: BLE001
                    st.update(judgement=Judgement(True, "crash", "env_exception"),
                              note=traceback.format_exc(limit=3)[-300:] or repr(ex))
                    running.discard(i)
                    continue
                st["steps"] += 1
                st["rej"] += bool(step.rejected)
                obs[i] = step.observation
                _call(hooks, "on_step", i, env, a, step)
                j = task.judge(env, step.time - t0[i], max_s)
                if not j.done and max_steps is not None and st["steps"] >= max_steps:
                    j = task.judge(env, max_s, max_s)
                if j.done:
                    st["judgement"] = j
                    running.discard(i)
        for i, env in enumerate(envs):
            st, j = state[i], state[i]["judgement"]
            ep = Episode(seed=group[i], task=task.name, env_id=spec.env_id, body="+".join(b.key for b in spec.bodies),
                         policy=info.name, source=info.source, outcome=j.outcome, failure_reason=j.failure_reason,
                         success_public=j.success_public, success_privileged=j.success_privileged, steps=st["steps"],
                         time=float(getattr(obs[i], "sensor_time", getattr(obs[i], "time", 0.0))) - t0[i],
                         wall_s=time.time() - st["wall"],
                         metrics=dict(command_rejections=st["rej"], chunk_rejections=st["chunk_rej"],
                                      chunks=st["chunks"], packets=st["packets"],
                                      **({"note": st["note"]} if st["note"] else {})),
                         provenance=dict(env=spec.provenance, policy_version=info.version, variant=info.variant))
            for h in hooks:
                if hasattr(h, "on_end"):
                    ep.metrics.update(h.on_end(i, env, ep) or {})
            episodes.append(ep)
        for e in envs:
            e.close()
    return episodes


# ------------------------------------------------------------------------------------------------ evaluate / matrix
def summarize(episodes: Sequence[Episode]) -> dict:
    """Success over attempted (non-infeasible) episodes, Wilson 95% interval, outcome counts, public/privileged
    agreement, policy calls (chunks + packets emitted) and control steps."""
    from rrp.harness.eval.statistics import wilson
    att = [e for e in episodes if e.outcome != "infeasible"]
    n, k = len(att), sum(bool(e.success_privileged) for e in att)
    lo, hi = wilson(k, n)
    return dict(attempted=n, successes=k, success_rate=k / n if n else None, wilson95=[lo, hi],
                infeasible=len(episodes) - n,
                outcomes={o: sum(e.outcome == o for e in episodes) for o in sorted({e.outcome for e in episodes})},
                public_private_agreement=(sum(bool(e.success_privileged) == bool(e.success_public) for e in att) / n
                                          if n else None),
                policy_calls=sum(e.metrics.get("chunks", 0) + e.metrics.get("packets", 0) for e in att),
                control_steps=sum(e.steps for e in att))


def _scene_fn(scene):
    return scene if callable(scene) else (lambda seed: dict(scene or {}))


def evaluate(policy, env_id: str, task: str, body, seeds: Sequence[int], *, scene=None, batch: int = 8,
             max_seconds: float | None = None, max_steps: int | None = None, hooks: Sequence = (),
             out: Path | None = None, env_kw: dict | None = None, row_extra: dict | None = None) -> list[Episode]:
    """rollout of `policy` (a Policy, or a registry name for make_policy) on make_env(env_id, task, body, seed) with
    scene kwargs `scene` (a dict, or seed -> dict). Rows (Episode.row() + row_extra) are appended to `out` (JSONL)."""
    from rrp.envs.base import make_env
    from rrp.policies.base import make_policy
    from rrp.tasks.spec import get_task
    pol = make_policy(policy) if isinstance(policy, str) else policy
    sf = _scene_fn(scene)
    eps = rollout(lambda sd: make_env(env_id, task=task, body=body, seed=sd, scene=sf(sd), **(env_kw or {})), pol,
                  get_task(task), list(seeds), batch=batch, max_seconds=max_seconds, max_steps=max_steps, hooks=hooks)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as fh:
            for e in eps:
                fh.write(json.dumps({**e.row(), **(row_extra or {})}, default=_json_default) + "\n")
    return eps


def _json_default(x):
    import numpy as np
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


HEAVY_ENVS = frozenset({"simple"})     # building these starts an external simulator (Isaac Sim): never implicitly


def env_spec(env_id: str, *, task: str, body, seed: int = 0, build_heavy: bool = False):
    """The EnvSpec of env_id for (task, body): the env module's static `env_spec(task=, body=)` when it has one (no
    simulator), else a built env's spec (refused for HEAVY_ENVS unless build_heavy)."""
    import importlib
    from rrp.envs.base import ENVS, make_env
    if env_id not in ENVS:
        raise KeyError(f"unknown env {env_id!r}; registered: {sorted(ENVS)}")
    mod = importlib.import_module(ENVS[env_id].split(":")[0])
    if hasattr(mod, "env_spec"):
        return mod.env_spec(task=task, body=body)
    if env_id in HEAVY_ENVS and not build_heavy:
        raise RuntimeError(f"{env_id} has no static env_spec and building it starts its simulator "
                           "(run on the peer with --build-heavy)")
    env = make_env(env_id, task=task, body=body, seed=seed)
    try:
        return env.spec
    finally:
        env.close()


def matrix(policies: Sequence, envs: Sequence[tuple[str, object]], tasks: Sequence[str], *, seeds: Sequence[int] = (),
           batch: int = 8, out: Path | None = None, build_heavy: bool = False) -> list[dict]:
    """negotiate() every policy x (env_id, body) x task cell. A cell is "n/a" with reasons when the task does not exist
    in the env, the env or policy cannot be built here, or negotiate declines; accepted cells are rolled out on `seeds`
    (none: negotiation only). policies: registry names, (name, make_policy kwargs) pairs, or Policy objects."""
    from rrp.envs.base import make_env
    from rrp.policies.base import make_policy, negotiate
    from rrp.tasks.spec import get_task
    pols: list[tuple[str, object]] = []
    for p in policies:
        name, kw = (p, {}) if isinstance(p, str) else (p if isinstance(p, tuple) else (p.info.name, None))
        try:
            pols.append((name, make_policy(name, **kw) if kw is not None else p))
        except Exception as ex:  # noqa: BLE001  (recorded as n/a, never skipped)
            pols.append((name, f"policy unavailable: {type(ex).__name__}: {ex}"))
    rows = []
    for env_id, body in envs:
        for tname in tasks:
            t = get_task(tname)
            spec, why = None, None
            if env_id not in t.envs:
                why = f"task {tname!r} does not exist in {env_id}"
            else:
                try:
                    spec = env_spec(env_id, task=tname, body=body, seed=seeds[0] if seeds else 0,
                                    build_heavy=build_heavy)
                except Exception as ex:  # noqa: BLE001
                    why = f"env unavailable: {type(ex).__name__}: {ex}"
            for name, pol in pols:
                row = dict(policy=name, env_id=env_id, body=body if isinstance(body, str) else "+".join(body),
                           task=tname, status="n/a", reasons=[])
                if why is not None or isinstance(pol, str):
                    row["reasons"] = [why if why is not None else pol]
                else:
                    c = negotiate(pol.info, spec, t)
                    row.update(status="accepted" if c.ok else "n/a", reasons=list(c.reasons), source=pol.info.source)
                    if c.ok and seeds:
                        eps = rollout(lambda sd: make_env(env_id, task=tname, body=body, seed=sd), pol, t, list(seeds),
                                      batch=batch)
                        row["summary"] = summarize(eps)
                rows.append(row)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, default=_json_default) + "\n" for r in rows))
    return rows
