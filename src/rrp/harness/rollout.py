"""The one episode loop (docs/architecture.md section 6): any Policy x Env x Task that `negotiate` accepts.

Episodes run in lock-step batches. Each tick the policy sees only the running episodes (a dict keyed by episode index,
so per-episode state and random streams do not depend on which other episodes are still running), its commands are
applied, the task's judge decides termination, and hooks observe or intervene. Every special case of the old
per-family loops (packet edits, perturbations, recorders, DAgger collection, video) is a Hook.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
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
            batch: int = 8, max_seconds: float | None = None, hooks: Sequence = ()) -> list[Episode]:
    """make_env(seed) returns an env already reset to `seed` (every registered factory does)."""
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
        for i, e in enumerate(envs):
            _call(hooks, "on_reset", i, e, obs[i])
        state = {i: dict(steps=0, rej=0, chunk_rej=0, wall=time.time(), judgement=None, note="") for i in obs}
        running = set(obs)
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
                                      **({"note": st["note"]} if st["note"] else {})),
                         provenance=dict(env=spec.provenance, policy_version=info.version, variant=info.variant))
            for h in hooks:
                if hasattr(h, "on_end"):
                    ep.metrics.update(h.on_end(i, env, ep) or {})
            episodes.append(ep)
        for e in envs:
            e.close()
    return episodes
