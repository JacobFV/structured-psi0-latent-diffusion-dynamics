"""Evaluation front-ends over harness.rollout (docs/architecture.md section 6): `evaluate` (one policy on env x task x
body, JSONL rows), `summarize` (Wilson), `matrix` (every policy x env x task cell: accepted | n/a with reasons) and
`env_spec` (static specs; simulators such as SIMPLE/Isaac are never started implicitly). CLI: `rrp eval`, `rrp matrix`.

Nothing here compares env ids or task names: the scene an env gets is the task's (`TaskSpec.scene`, or the caller's), the
default hooks are the task's by name (`TaskSpec.hooks` -> `harness.hooks.TASK_HOOKS`). A scene for an env whose factory takes none is a
negotiation failure (`Incompatible` with the reason text, an "n/a" cell in `matrix`), never a TypeError."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from rrp.harness.hooks import TASK_HOOKS
from rrp.harness.rollout import Episode, Incompatible, rollout


def task_hooks(task: str, env_id: str) -> list:
    """The default eval hooks the task declares (`TaskSpec.hooks`) that apply to `env_id`."""
    from rrp.tasks.spec import get_task
    t = get_task(task)
    out: list = []
    for name in t.hooks:
        prefix, make = TASK_HOOKS[name]
        if env_id.startswith(prefix):
            out += make(t)
    return out

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


def env_factory(env_id: str, task: str, body, *, scene=None, env_kw: dict | None = None) -> Callable[[int], object]:
    """seed -> a built env of (env_id, task, body). `scene` (a dict, or seed -> dict) overrides the task's own scene
    (`TaskSpec.scene`); a scene is passed to the factory only when there is one. Raises Incompatible when a scene is
    requested from an env whose factory takes none."""
    from rrp.envs.base import make_env, scene_reasons
    from rrp.tasks.spec import get_task
    sc = scene if scene is not None else get_task(task).scene
    if reasons := scene_reasons(env_id, task, sc):
        raise Incompatible(reasons)
    sf = sc if callable(sc) or sc is None else (lambda seed: dict(sc))
    return lambda sd: make_env(env_id, task=task, body=body, seed=sd, scene=None if sf is None else sf(sd),
                               **(env_kw or {}))


def evaluate(policy, env_id: str, task: str, body, seeds: Sequence[int], *, scene=None, batch: int = 8,
             max_seconds: float | None = None, max_steps: int | None = None, hooks: Sequence = (),
             out: Path | None = None, env_kw: dict | None = None, row_extra: dict | None = None) -> list[Episode]:
    """rollout of `policy` (a Policy, or a registry name for make_policy) on make_env(env_id, task, body, seed). Scene:
    `scene` (a dict, or seed -> dict) else the task's own, passed only when there is one; `env_kw` are extra env
    kwargs (level, split, render profile). Rows (Episode.row() + row_extra) are appended to `out` (JSONL)."""
    from rrp.policies.base import make_policy
    from rrp.tasks.spec import get_task
    pol = make_policy(policy) if isinstance(policy, str) else policy
    eps = rollout(env_factory(env_id, task, body, scene=scene, env_kw=env_kw), pol, get_task(task), list(seeds),
                  batch=batch, max_seconds=max_seconds, max_steps=max_steps, hooks=hooks)
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
    from rrp.envs.base import scene_reasons
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
            elif reasons := scene_reasons(env_id, tname, t.scene):
                why = reasons[0]
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
                        eps = rollout(env_factory(env_id, tname, body), pol, t, list(seeds), batch=batch)
                        row["summary"] = summarize(eps)
                rows.append(row)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, default=_json_default) + "\n" for r in rows))
    return rows
