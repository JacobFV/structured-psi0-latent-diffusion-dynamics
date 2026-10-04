"""The one episode loop (docs/architecture.md section 6): any Policy x Env x Task that `negotiate` accepts.

Episodes run in lock-step batches. Each tick the policy sees only the running episodes (a dict keyed by episode index,
so per-episode state and random streams do not depend on which other episodes are still running), its commands are
applied, the task's judge decides termination, and hooks observe or intervene. Every special case of the old
per-family loops (packet edits, perturbations, recorders, DAgger collection, video) is a Hook.
"""
from __future__ import annotations

import gc
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol, Sequence

from rrp.core.errors import ControllerRejection, StaleActionError
from rrp.envs.base import env_failure_reason
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


ROLLOUT_REASONS = ("timeout", "policy_exception", "physics_divergence", "env_exception")   # rollout itself: budget spent, crash


class UndeclaredFailureReason(AssertionError):
    """An episode ended with a failure reason outside the vocabulary the task (or an episode-ending hook) declares."""


def declared_reasons(task: TaskSpec, hooks: Sequence = ()) -> frozenset[str]:
    """Every failure reason an episode of `task` may carry: `TaskSpec.failure_reasons`, rollout's own reasons (budget spent, crash) and the
    `failure_reasons` tuple of each hook that can end an episode. A reason `code:detail` is declared by its `code`, or by the full string
    (cw/fill_form declares `wrong_value:name`)."""
    out = set(task.failure_reasons) | set(ROLLOUT_REASONS)
    for h in hooks:
        out.update(getattr(h, "failure_reasons", ()))
    return frozenset(out)


def check_failure_reason(reason: str | None, task: TaskSpec, hooks: Sequence = ()) -> None:
    declared = declared_reasons(task, hooks)
    if reason is not None and reason not in declared and reason.split(":", 1)[0] not in declared:
        raise UndeclaredFailureReason(
            f"task {task.name!r}: failure reason {reason!r} is not in its declared vocabulary "
            f"{sorted(declared)} (TaskSpec.failure_reasons)")


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


def batch_reason(envs, policy, hooks) -> str | None:
    """Why `eval_backend=warp` cannot run this group on the batched stepper (None: it can). Recorded, then the group runs on CPU."""
    from rrp.envs.warp.batch_sim import env_reason
    for i, e in enumerate(envs):
        if (why := env_reason(e)) is not None:
            return f"env {i}: {why}"
    for h in hooks:
        if why := getattr(h, "batch_unsafe", None):
            return f"hook {type(h).__name__}: {why}"
    if why := getattr(policy, "batch_unsafe", None):
        return f"policy {policy.info.name}: {why}"
    return None


def rollout(make_env: Callable[[int], object], policy: Policy, task: TaskSpec, seeds: Sequence[int], *,
            batch: int = 8, max_seconds: float | None = None, max_steps: int | None = None,
            hooks: Sequence = (), eval_backend: str | None = None, device: str | None = None) -> list[Episode]:
    """make_env(seed) returns an env already reset to `seed` (every registered factory does).

    Budget: `max_seconds` (default task.max_seconds) of env time; `max_steps` (default task.max_steps) additionally ends
    an episode after that many control ticks (the task judge is then asked with the budget spent, so its timeout rule
    applies exactly). Every env of a group is negotiated (specs may differ per env: body, level, scene).
    A hook's on_reset may return a done Judgement (e.g. "infeasible") to end that episode before its first tick; an
    on_step hook may return one to end it after a tick (used only when the task judge has not ended it).

    `eval_backend="warp"` (docs/architecture.md section 14.8): the physics of every control step of a group runs batched on MuJoCo Warp
    (`rrp.envs.warp.batch_sim.BatchStepper`); everything else (policy, hooks, judge, sensing) is unchanged. A group that cannot be batched
    (a perturbation hook, an actuator model, a dual session, a policy limited to one episode, a model Warp cannot build) runs on CPU and
    says why in every row of the group (`provenance.eval_backend`). The default `cpu` adds nothing to the rows.
    `eval_backend=None` takes the ambient `core.compute.current().eval_backend` (the RunConfig `compute:` block; default `cpu`)."""
    if eval_backend is None:
        from rrp.core import compute
        eval_backend = compute.current().eval_backend
    max_s = task.max_seconds if max_seconds is None else max_seconds
    max_steps = task.max_steps if max_steps is None else max_steps
    info = policy.info
    episodes: list[Episode] = []
    for b0 in range(0, len(seeds), batch):
        group = list(seeds[b0:b0 + batch])
        envs = [make_env(sd) for sd in group]
        spec = envs[0].spec
        reasons: list[str] = []
        for e in envs:
            reasons += [r for r in negotiate(info, e.spec, task).reasons if r not in reasons]
        if reasons:
            for e in envs:
                e.close()
            raise Incompatible(reasons)
        stepper, backend_note = None, None
        if eval_backend == "warp":
            from rrp.envs.warp.batch_sim import BatchStepper, BatchUnsupported
            backend_note = dict(requested="warp", effective="warp", fallback=None)
            if (why := batch_reason(envs, policy, hooks)) is None:
                try:
                    stepper = BatchStepper(envs, device)
                except BatchUnsupported as ex:
                    why = str(ex)
            if why is not None:
                backend_note.update(effective="cpu", fallback=why)
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
        def after(i, a, step):
            env, st = envs[i], state[i]
            st["steps"] += 1
            st["rej"] += bool(step.rejected)
            obs[i] = step.observation
            hj = _call(hooks, "on_step", i, env, a, step)
            j = task.judge(env, step.time - t0[i], max_s)
            if not j.done and isinstance(hj, Judgement) and hj.done:
                j = hj                           # a hook ended the episode (e.g. a teacher reference finished)
            if not j.done and max_steps is not None and st["steps"] >= max_steps:
                j = task.judge(env, max_s, max_s)
            if j.done:
                st["judgement"] = j
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
            todo = {}
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
                            a.info["chunk_rejected"] = True          # hooks see which submissions were refused
                    if stepper is None:
                        step = env.step(a.command)
                    else:
                        todo[i] = a                                  # batched: every env's step runs in one physics call below
                        continue
                except FloatingPointError as ex:
                    st.update(judgement=Judgement(True, "crash", "physics_divergence"), note=str(ex)[:300])
                    running.discard(i)
                    continue
                except Exception as ex:  # noqa: BLE001
                    st.update(judgement=Judgement(True, "crash", "env_exception"),
                              note=traceback.format_exc(limit=3)[-300:] or repr(ex))
                    running.discard(i)
                    continue
                after(i, a, step)
            if todo:
                results = stepper.step({i: a.command for i, a in todo.items()})
                for i, a in todo.items():
                    step, st = results[i], state[i]
                    if isinstance(step, FloatingPointError):
                        st.update(judgement=Judgement(True, "crash", "physics_divergence"), note=str(step)[:300])
                        running.discard(i)
                    elif isinstance(step, BaseException):
                        st.update(judgement=Judgement(True, "crash", "env_exception"),
                                  note="".join(traceback.format_exception(step, limit=3))[-300:] or repr(step))
                        running.discard(i)
                    else:
                        after(i, a, step)
        for i, env in enumerate(envs):
            st, j = state[i], state[i]["judgement"]
            try:
                check_failure_reason(j.failure_reason, task, hooks)
            except UndeclaredFailureReason:
                for e in envs:
                    e.close()
                raise
            ep = Episode(seed=group[i], task=task.name, env_id=spec.env_id, body="+".join(b.key for b in spec.bodies),
                         policy=info.name, source=info.source, outcome=j.outcome, failure_reason=j.failure_reason,
                         success_public=j.success_public, success_privileged=j.success_privileged, steps=st["steps"],
                         time=float(getattr(obs[i], "sensor_time", getattr(obs[i], "time", 0.0))) - t0[i],
                         wall_s=time.time() - st["wall"],
                         metrics=dict(command_rejections=st["rej"], chunk_rejections=st["chunk_rej"],
                                      chunks=st["chunks"], packets=st["packets"],
                                      **({"note": st["note"]} if st["note"] else {})),
                         provenance=dict(env=spec.provenance, policy_version=info.version, variant=info.variant,
                                         **({} if backend_note is None else dict(eval_backend=backend_note))))
            if (why := env_failure_reason(env)) is not None:
                ep.metrics["env_failure_reason"] = why
            for h in hooks:
                if hasattr(h, "on_end"):
                    ep.metrics.update(h.on_end(i, env, ep) or {})
            episodes.append(ep)
        for e in envs:
            e.close()
        # Closed envs sit in reference cycles (Session.runtime -> bound methods / clock closure -> Session, and others) whose
        # ~100 MB of MjData per env is invisible to the gc trigger: they piled up until a rare full collection (D-147 10-04:
        # ~0.42 GiB per group of 4 humanoid episodes, 10 GB per 100-scene cell). Free each group before the next.
        envs = None
        gc.collect()
    return episodes
