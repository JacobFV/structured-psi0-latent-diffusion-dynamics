"""`rrp eval` (any Policy x Env x Task through harness.rollout) and `rrp matrix` (negotiate every cell, n/a with reasons).
Standard library only at import time; heavy imports live inside the commands."""
from __future__ import annotations

import json
from pathlib import Path


def _seeds(s: str) -> list[int]:
    """"0:10" (range), "3,4,50" (list) or "" (none)."""
    if not s:
        return []
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a), int(b)))
    return [int(x) for x in s.split(",")]


def _policy_arg(s: str):
    """NAME or NAME=JSON-kwargs for make_policy (e.g. 'bc={"checkpoint": "artifacts/runs/x/policy.pt"}')."""
    name, _, kw = s.partition("=")
    return (name, json.loads(kw)) if kw else name


def _body(s: str):
    return s.split("+") if "+" in s else s


def default_hooks(env_id: str, task: str) -> list:
    """The per-family conventions of the former eval loops: arm feasibility + session record; dual adds settling."""
    from rrp.harness import hooks as H
    if env_id == "mujoco/arm" and task == "pick_place":
        return H.arm_hooks()
    if env_id == "mujoco/dual":
        return H.dual_hooks(task)
    if env_id.startswith("mujoco/"):
        return [H.SessionRecord()]
    return []


def cmd_eval(a):
    from rrp.harness.rollout import evaluate, summarize
    from rrp.policies.base import make_policy
    p = _policy_arg(a.policy)
    pol = make_policy(p[0], **p[1]) if isinstance(p, tuple) else make_policy(p)
    scene = json.loads(a.scene) if a.scene else None
    if a.env == "mujoco/arm" and a.task == "pick_place" and scene is None:
        from rrp.harness.hooks import arm_scene
        scene = arm_scene
    eps = evaluate(pol, a.env, a.task, _body(a.body), _seeds(a.seeds), scene=scene, batch=a.batch,
                   max_seconds=a.max_seconds, max_steps=a.max_steps,
                   hooks=[] if a.no_hooks else default_hooks(a.env, a.task), out=Path(a.out))
    summ = dict(policy=pol.info.name, source=pol.info.source, env=a.env, task=a.task, body=a.body, **summarize(eps))
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def format_matrix(rows: list[dict]) -> str:
    lines = [f"{'policy':28s} {'env':16s} {'body':18s} {'task':20s} status    reasons"]
    for r in rows:
        extra = "; ".join(r["reasons"])
        if r.get("summary"):
            s = r["summary"]
            extra = f"{s['successes']}/{s['attempted']} success" + (f"; {extra}" if extra else "")
        lines.append(f"{r['policy'][:28]:28s} {r['env_id'][:16]:16s} {r['body'][:18]:18s} {r['task'][:20]:20s} "
                     f"{r['status']:9s} {extra}")
    return "\n".join(lines)


def cmd_matrix(a):
    from rrp.harness.rollout import matrix
    envs = []
    for e in a.env:
        env_id, _, body = e.partition("=")
        envs.append((env_id, _body(body)))
    rows = matrix([_policy_arg(p) for p in a.policy], envs, a.task, seeds=_seeds(a.seeds), batch=a.batch,
                  out=Path(a.out) if a.out else None, build_heavy=a.build_heavy)
    print(format_matrix(rows))


def register(sub):
    e = sub.add_parser("eval", help="evaluate a policy on env x task x body through harness.rollout (JSONL rows + summary)")
    e.add_argument("--policy", required=True, help='registry name, or NAME=JSON kwargs for make_policy')
    e.add_argument("--env", required=True, help="env registry id (mujoco/arm, mujoco/dual, mujoco/legged, computerworld, ...)")
    e.add_argument("--task", required=True)
    e.add_argument("--body", required=True, help="body key; dual pairs as a+b or a pair key")
    e.add_argument("--seeds", required=True, help='"start:stop" or "a,b,c"')
    e.add_argument("--scene", help="JSON scene kwargs for the env factory (default: the family's eval convention)")
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--max-seconds", type=float)
    e.add_argument("--max-steps", type=int)
    e.add_argument("--no-hooks", action="store_true", help="skip the family's default hooks (feasibility, settle, ...)")
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_eval)
    m = sub.add_parser("matrix", help="policy x env x task compatibility (negotiate reasons; n/a shown), optional rollouts")
    m.add_argument("--policy", action="append", required=True, help="repeatable; NAME or NAME=JSON kwargs")
    m.add_argument("--env", action="append", required=True, help="repeatable; ENV_ID=BODY (e.g. computerworld=cw_pointer)")
    m.add_argument("--task", action="append", required=True, help="repeatable")
    m.add_argument("--seeds", default="", help="rollout accepted cells on these seeds (default: negotiation only)")
    m.add_argument("--batch", type=int, default=8)
    m.add_argument("--out", help="JSONL of the cells")
    m.add_argument("--build-heavy", action="store_true",
                   help="build envs that start an external simulator (simple/Isaac) to read their spec: peer only")
    m.set_defaults(fn=cmd_matrix)
