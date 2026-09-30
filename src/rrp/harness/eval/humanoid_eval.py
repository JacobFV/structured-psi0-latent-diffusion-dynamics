"""Humanoid evaluation in C MuJoCo (W13, D-138; full self-collision, contact_v2), over the one evaluation loop (`evaluate`,
docs/architecture.md sections 6 and 14.4). Two parts:

1. Tools (`rrp suite humanoid-steps | humanoid-gap | humanoid-gap-smoke | contact-waypoint`; the recipe stage
   `legged/eval_tracker` runs them): an h_steps / h_gap expert or blind tracker under the scripted command layer, a tracker
   under the WaypointTeacher. Their argv and result JSON are unchanged (the old one-off loops that patched `load_tracker` are
   gone): the actor is resolved to a tracker-registry entry and the env is built with `tracker=<spec>`.
   A legacy PRIVILEGED-input expert (extra block that is not the public D-146 terrain scan) is refused: nothing supplies its
   input any more; retrain it on the public scan. Rendering moved to `rrp video legged`.
2. `rrp eval humanoid-transfer` (`transfer_main`): the config-driven transfer driver of the pre-registered matrix
   (research/tracks/humanoid.md section 4): sealed bodies x methods x demo budgets x training seeds. Level 1 (existing-controller
   transfer) and Level 2 (new controller training) are separate tables and never pooled; every adapting cell of a group has
   the SAME acquisition record (demos, teacher ticks, env samples, updates), checked against the pack that produced the data;
   a cell without a run or accounting is reported (`missing_run` / `unaccounted`; `pending` = the run is planned: the method declares
   the stage that produces it, `producer`), never skipped; sealed cells run once through `SealedSplit.sealed_eval` with SL's native
   cell (body, method, train seed, scenes, task, budget, adaptation); tables (Wilson per cell, Newcombe method - reference) come from `statistics.py`.
* lab_gate: the W13 lab gate over a tracker_validation JSON (no-fall 1.0, forward >= 0.8, turn >= 0.5, slip < 0.15).
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from rrp.core.provenance import file_digest

SRC_EXPERT = "privileged_teacher:rl_expert + scripted_teacher command"
SRC_LEARNED = "learned:rl_expert + scripted_teacher command"
SRC_BLIND = "learned_tracker (blind) + scripted_teacher command"


# --------------------------------------------------------------------------------------------- tool plumbing
def resolve_actor(body: str, actor: str) -> str:
    """The tracker-registry spec `<body>:<version>` of `actor`: a spec already in the registry as is, or an actor FILE registered
    for this process under a temporary store (`<body>:file_<sha12>`, sha256 pinned, decision `unregistered`). An actor whose extra
    input block is privileged (not the public D-146 terrain scan) is refused."""
    from rrp.envs.mujoco import legged_tracker as LT
    p = Path(actor)
    if not p.is_file():
        e = LT.get_entry(actor)
        if e.extra_obs == "privileged":
            raise SystemExit(_privileged_msg(actor))
        return e.spec
    import torch
    meta = torch.load(str(p), map_location="cpu", weights_only=False)["meta"]
    if LT.extra_kind(meta) == "privileged":
        raise SystemExit(_privileged_msg(str(p)))
    sha = file_digest(p, length=None)
    version = f"file_{sha[:12]}"
    store = Path(tempfile.mkdtemp(prefix="rrp_actor_")) / body / version
    store.mkdir(parents=True)
    (store / "actor.pt").symlink_to(p.resolve())
    (store / "meta.json").write_text(json.dumps({**meta, "sha256": sha, "decision": "unregistered"}, default=str))
    LT.TRACKERS.update(LT.scan_trackers(store.parent.parent))
    return f"{body}:{version}"


def _privileged_msg(what: str) -> str:
    return (f"{what}: the actor takes a PRIVILEGED extra input block that nothing supplies any more (D-146: the steps expert reads the "
            "public terrain_scan_v1 sensor; the gap expert is blind to walls). Retrain it on the public sensor; it is not evaluated.")


class SceneRecord:
    """on_end: the scene facts the tool rows carry (final base x, x_end, scene meta) and the physics declaration."""

    def __init__(self, keys=()):
        self.keys = tuple(keys)

    def on_end(self, i, env, ep):
        m = env.scenario.meta
        return dict(x=float(env.data.qpos[env.binding.qa]), scene={k: m.get(k) for k in self.keys},
                    contact_model=m.get("contact_model"), actuator_limits=env.binding.meta.get("actuator_limits"))


class CommandLog:
    """on_act: the base_velocity command of every tick (for `pure_turn_frac`); on_end reports it per episode."""

    def __init__(self):
        self.cmds: dict[int, list] = {}

    def on_act(self, i, obs, act):
        c = getattr(act, "command", None)
        if c is not None and "base_velocity" in c.groups:
            self.cmds.setdefault(i, []).append(list(c.groups["base_velocity"]))

    def on_end(self, i, env, ep):
        c = np.array(self.cmds.pop(i, [[1.0, 0.0, 0.0]]))
        return dict(pure_turn_frac=float(np.mean((c[:, 0] < 0.05) & (np.abs(c[:, 2]) > 0.05))))


def _status(ep) -> str:
    return "success" if ep.outcome == "success" else (ep.failure_reason or ep.outcome)


def _expert_source(pol) -> str:
    return SRC_EXPERT if pol.info.source == "privileged_teacher" else (SRC_LEARNED if pol.entry.extra_obs != "none" else SRC_BLIND)


def _run_tool(a, task: str, scene: dict | None, scene_keys, row_keys, policy_of, extra_hooks=(), **ev_kw):
    """The shared body of the steps / gap / waypoint tools: evaluate() over seeds [seed0, seed0 + n), rows in the legacy shape."""
    from rrp.harness.eval.evaluate import evaluate, task_hooks
    spec = resolve_actor(a.body, a.actor)
    pol = policy_of(spec)
    rec = SceneRecord(scene_keys)
    eps = evaluate(pol, "mujoco/legged", task, a.body, range(a.seed0, a.seed0 + a.n), scene=scene, batch=4,
                   hooks=[*task_hooks(task, "mujoco/legged"), rec, *extra_hooks], env_kw=dict(tracker=spec), **ev_kw)
    rows = []
    for e in eps:
        m = e.metrics
        r = dict(seed=e.seed, status=_status(e), **{k: m["scene"][k] for k in row_keys}, x=m["x"], sim_s=m.get("sim_time", e.time),
                 wall_s=e.wall_s)
        rows.append(r)
        print(r, flush=True)
    return pol, eps, rows


def steps_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp suite humanoid-steps", description="C-MuJoCo h_steps evaluation of an expert or blind tracker.")
    ap.add_argument("body")
    ap.add_argument("actor")
    ap.add_argument("--seed0", type=int, default=7200)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h-frac", type=float, default=None, help="step height / leg length (default: the scene's own draw)")
    a = ap.parse_args(argv)
    from rrp.policies.teachers.humanoid import make_rl_expert
    pol, eps, rows = _run_tool(a, "h_steps", None if a.h_frac is None else dict(h_frac=a.h_frac), ("h_frac", "x_end"), ("h_frac", "x_end"),
                               lambda spec: make_rl_expert(arg=spec))
    rows = [dict(seed=r["seed"], status=r["status"], h_frac=r["h_frac"], x=r["x"], x_end=r["x_end"], sim_s=r["sim_s"], wall_s=r["wall_s"])
            for r in rows]
    summ = dict(body=a.body, actor=a.actor, n=a.n, source=_expert_source(pol), success=sum(r["status"] == "success" for r in rows),
                fell=sum(r["status"] == "fell" for r in rows), rows=rows)
    Path(a.out).write_text(json.dumps(summ, indent=1))
    print({k: v for k, v in summ.items() if k != "rows"})
    return 0


def gap_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp suite humanoid-gap", description="C-MuJoCo h_gap_sidestep evaluation of an expert or blind tracker.")
    ap.add_argument("body")
    ap.add_argument("actor")
    ap.add_argument("--seed0", type=int, default=7200)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=float, default=1.0)
    a = ap.parse_args(argv)
    from rrp.policies.teachers.humanoid import make_rl_expert
    pol, eps, rows = _run_tool(a, "h_gap", dict(level=a.level), ("gap_ratio", "y_c", "psi_f"), ("gap_ratio", "y_c", "psi_f"),
                               lambda spec: make_rl_expert(arg=spec))
    summ = dict(body=a.body, actor=a.actor, n=a.n, level=a.level, source=_expert_source(pol),
                success=sum(r["status"] == "success" for r in rows), status=dict(Counter(r["status"] for r in rows)), rows=rows)
    Path(a.out).write_text(json.dumps(summ, indent=1))
    print({k: v for k, v in summ.items() if k != "rows"})
    return 0


def gap_smoke_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp suite humanoid-gap-smoke", description="WarpGapEnv smoke with a tracker actor (peer GPU).")
    ap.add_argument("body")
    ap.add_argument("actor")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch
    from rrp.envs.mujoco.tracker_nets import mlp
    from rrp.envs.warp.task_env import WarpGapEnv
    st = torch.load(a.actor, map_location="cuda", weights_only=False)
    m = st["meta"]
    net = mlp(m["obs_dim"], tuple(m["hidden"]), m["act_dim"]).cuda()
    net.load_state_dict(st["actor"])
    mean, std = st["obs_mean"].cuda(), (st["obs_var"].cuda() + 1e-8).sqrt()
    out = {}
    for lv in (0.0, 1.0):
        e = WarpGapEnv(a.body, 256, seed=5, level=lv, clock_gate=bool(m.get("clock_gate")), push=False)
        for _ in range(1000):
            o = e.observe()[:, :m["obs_dim"]]
            e.step(net(((o - mean) / std).clamp(-5, 5)))
        s = e.pop_stats()
        out[f"level{lv}"] = dict(episodes=s["episodes"], falls=s["falls"], successes=s["successes"], obs_dim=e.obs_dim,
                                 gap_w_over_bw=float((e.gap_w / e.bw).mean()), finite=bool(torch.isfinite(e.observe()).all()))
    print(json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def waypoint_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp suite contact-waypoint",
                                 description="WaypointTeacher (scripted_teacher, privileged) driving a given tracker on waypoint_contact "
                                             "(env RRP_CONTACT_MODEL=v2 [RRP_ACTUATOR_LIMITS=...]).")
    ap.add_argument("body")
    ap.add_argument("actor")
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from rrp.policies.base import make_policy
    log = CommandLog()
    pol, eps, rows = _run_tool(a, "waypoint_contact", None, (), (), lambda spec: make_policy("teacher:waypoint_contact"),
                               extra_hooks=[log], max_seconds=130.0, max_steps=1300)
    out = []
    for e, r in zip(eps, rows):
        m = e.metrics
        out.append(dict(seed=e.seed, status=r["status"], steps=e.steps, sim_s=r["sim_s"], pure_turn_frac=m["pure_turn_frac"],
                        contact_model=m["contact_model"], actuator_limits=m["actuator_limits"]))
    summ = dict(body=a.body, actor=a.actor, n=a.n, success=sum(r["status"] == "success" for r in out),
                fell=sum(r["status"] == "fell" for r in out),
                mean_sim_s_success=float(np.mean([r["sim_s"] for r in out if r["status"] == "success"] or [0])), rows=out)
    Path(a.out).write_text(json.dumps(summ, indent=1))
    print({k: v for k, v in summ.items() if k != "rows"})
    return 0


def lab_gate(validation: dict, gate_report: dict | None = None, waypoint: dict | None = None) -> dict:
    """The W13 lab gate over a tracker_validation result: no-fall 1.0, forward >= 0.8, turn >= 0.5, slip < 0.15 (plus the D-112
    verdict and the waypoint_contact success when given). Verdicts are data; the caller decides what fails."""
    g = validation["gate"]
    lab = dict(no_fall=g.get("no_fall_rate"), forward=g.get("forward_ratio"), turn=g.get("turn_ratio"),
               slip=(g.get("contact_gate") or {}).get("slip_ratio"))
    lab["passed"] = bool(lab["no_fall"] == 1.0 and (lab["forward"] or 0) >= 0.8 and (lab["turn"] or 0) >= 0.5
                         and (lab["slip"] if lab["slip"] is not None else 1) < 0.15)
    if gate_report is not None:
        lab["d112_verdict"] = gate_report.get("verdict")
        lab["d112_criteria"] = {c["name"]: [c.get("value"), c.get("status")] for c in gate_report.get("criteria", [])}
    if waypoint is not None:
        lab.update(waypoint_success=waypoint["success"], waypoint_fell=waypoint["fell"], waypoint_n=waypoint["n"])
    return lab


# ================================================================================================== transfer driver
ZERO_ACQ = dict(demos=0, teacher_ticks=0, env_samples=0, updates=0)
ACQ_KEYS = tuple(ZERO_ACQ)
LEVEL_NAMES = {"1": "Level 1: existing-controller transfer", "2": "Level 2: new controller training",
               "0": "Reference: scripted / privileged teacher (not a learned method)"}
SOURCES = ("learned", "privileged_teacher", "scripted_teacher", "bc", "privileged", "oracle", "random", "mock")
RESULTS = "results.jsonl"
# the stage that creates a method's run: the adapting stages of humanoid/adapt_* (research/tracks/humanoid.md section 4), the zero-shot
# trainers, the tracker trainer and the tracker registry install. A method that declares one has its missing runs `pending` (planned),
# never `missing_run`: that word is for a run nothing is going to produce.
PRODUCERS = ("adapt_refit", "adapt_flow", "adapt_bc", "adapt_ppo", "train_rep", "train_flow", "train_bc", "train_tracker", "tracker-install")


class MissingRun(RuntimeError):
    """A cell whose run (checkpoint, registry actor) does not exist: reported as `missing_run`, never skipped."""


def fmt(x, **kv):
    """Replace `<key>` tokens (the config's own placeholders; `{}` belongs to the recipe renderer) in strings, recursively."""
    import re
    if isinstance(x, str):
        return re.sub(r"<(\w+)>", lambda m: str(kv[m.group(1)]) if m.group(1) in kv else m.group(0), x)
    if isinstance(x, dict):
        return {k: fmt(v, **kv) for k, v in x.items()}
    if isinstance(x, list):
        return [fmt(v, **kv) for v in x]
    return x


def load_config(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text()
    cfg = json.loads(text) if p.suffix == ".json" else __import__("rrp.harness.yamlmini", fromlist=["load"]).load(text)
    return validate_config(cfg)


def _flat(x) -> list:
    """`bodies` / `methods` may be a list or a {group: list} mapping (recipes keep the groups apart); returns one list."""
    return [b for v in x.values() for b in v] if isinstance(x, dict) else list(x)


def validate_config(cfg: dict) -> dict:
    for k in ("name", "task", "bodies", "train_seeds", "levels", "methods"):
        if k not in cfg:
            raise ValueError(f"transfer config: missing {k!r}")
    cfg = dict(cfg)
    cfg["bodies"] = _flat(cfg["bodies"])
    cfg["methods"] = [dict(m, group=g) for g, ms in (cfg["methods"].items() if isinstance(cfg["methods"], dict) else [("", cfg["methods"])])
                      for m in ms]
    cfg.setdefault("env", "mujoco/legged")
    cfg.setdefault("scenes", 100)
    cfg["levels"] = {str(k): dict(v) for k, v in cfg["levels"].items()}
    names = set()
    for m in cfg["methods"]:
        lv = str(m.get("level"))
        if lv not in ("0", "1", "2") or (lv != "0" and lv not in cfg["levels"]):
            raise ValueError(f"method {m.get('name')!r}: level {m.get('level')!r} is not declared in levels")
        if m.get("kind") not in ("expert", "teacher", "latent", "bc"):
            raise ValueError(f"method {m.get('name')!r}: kind {m.get('kind')!r} (expert | teacher | latent | bc)")
        if m.get("source") not in SOURCES:
            raise ValueError(f"method {m.get('name')!r}: source label {m.get('source')!r} must be one of {SOURCES}")
        if m["name"] in names:
            raise ValueError(f"duplicate method {m['name']!r}")
        names.add(m["name"])
        if m.get("budgeted") and lv == "0":
            raise ValueError(f"method {m['name']!r}: a reference method has no budget")
        if m.get("producer") is not None and m["producer"] not in PRODUCERS:
            raise ValueError(f"method {m['name']!r}: producer {m['producer']!r} is not one of {PRODUCERS}")
    for lv, d in cfg["levels"].items():
        if d.get("unit") not in ("demos", "samples"):
            raise ValueError(f"level {lv}: unit must be demos | samples")
        if d.get("reference") and d["reference"] not in names:
            raise ValueError(f"level {lv}: reference {d['reference']!r} is not a method")
        if d["unit"] == "demos" and {str(b) for b in d["budgets"]} != {str(k) for k in d.get("updates", {})}:
            raise ValueError(f"level {lv}: `updates` must give the matched update count of every demo budget")
    return cfg


def method_of(cfg: dict, name: str) -> dict:
    return next(m for m in cfg["methods"] if m["name"] == name)


def cell_key(c: dict) -> str:
    return f"{c['task']}|{c['body']}|{c['method']}|n{c['budget'] if c['budget'] is not None else 0}|s{c['train_seed']}"


def sealed_cell(cfg: dict, c: dict, scenes) -> dict:
    """The SealedSplit cell of a matrix cell, in SL's native keys (core.sealed.CELL_KEYS): its id is
    body|method|task|n<budget>|adaptation|s<train seed>|evaluation, so two cells that differ only in task, demo / sample budget or
    adaptation are different run-once cells. `adaptation` is the stage that produced the run (the method's `producer`; none for a
    zero-shot, reference or registry cell)."""
    m = method_of(cfg, c["method"])
    return dict(body=c["body"], method=c["method"], train_seed=c["train_seed"], scenes=list(scenes), task=c["task"], budget=c["budget"],
                adaptation=m.get("producer") if c["budget"] is not None else None)


def expand_cells(cfg: dict, bodies=None) -> list[dict]:
    """Every cell of the matrix: bodies x methods x (budgets of the method's level, or none) x (training seeds, or one)."""
    out = []
    for body in (bodies if bodies is not None else cfg["bodies"]):
        for m in cfg["methods"]:
            lv = str(m["level"])
            budgets = list(cfg["levels"][lv]["budgets"]) if m.get("budgeted") else [None]
            seeds = list(cfg["train_seeds"]) if m.get("trained") else [0]
            for b in budgets:
                for s in seeds:
                    out.append(dict(task=cfg["task"], body=body, method=m["name"], level=lv, budget=b, train_seed=s))
    return out


def select_cells(cfg: dict, cells: list[dict], variant: str | None = None, train_seed: int | None = None) -> list[dict]:
    """A recipe point evaluates only its own cells: methods whose `variant` is `variant` (`-` = methods without a variant: the
    existing-controller and teacher references) and, for trained methods, `train_seed`. None keeps everything."""
    out = []
    for c in cells:
        m = method_of(cfg, c["method"])
        if variant is not None and (m.get("variant") or "-") != variant:
            continue
        if train_seed is not None and m.get("trained") and c["train_seed"] != train_seed:
            continue
        out.append(c)
    return out


def scenes_for(split, sealed: bool, n: int = 100) -> list[int]:
    lo, _ = split.ranges["evaluation" if sealed else "development"]
    return list(range(lo, lo + n))


# ---- acquisition accounting (equal for every adapting method of a group; zero for zero-shot)
def acquisition_of_pack(pack: dict, cfg: dict, cell: dict) -> dict:
    """The acquisition of the cell: what its adapting data cost. Level unit `demos`: the pack record of (task, body, budget);
    unit `samples`: the env samples of the budget; not adapting: zero."""
    if cell["budget"] is None:
        return dict(ZERO_ACQ)
    d = cfg["levels"][cell["level"]]
    if d["unit"] == "samples":
        return dict(ZERO_ACQ, env_samples=int(cell["budget"]))
    rec = (pack.get("records") or {}).get(f"{cell['task']}|{cell['body']}|n{cell['budget']}")
    if rec is None:
        raise MissingRun(f"pack has no record for {cell['task']}|{cell['body']}|n{cell['budget']}")
    if int(rec["updates"]) != int(d["updates"][str(cell["budget"])]):
        raise ValueError(f"pack updates {rec['updates']} != the matched update count {d['updates'][str(cell['budget'])]} of budget {cell['budget']}")
    return {k: int(rec[k]) for k in ACQ_KEYS if k in rec} | {k: 0 for k in ACQ_KEYS if k not in rec}


def check_equal_acquisition(rows: list[dict]) -> list[str]:
    """rows: dicts with level, task, body, budget, method, train_seed, acquisition. Every adapting row of (level, task, body,
    budget) must carry ONE acquisition record (same demos, teacher ticks, env samples, updates); a row with no budget carries
    zeros. Returns the violations (empty = fair)."""
    bad, groups = [], {}
    for r in rows:
        acq = {k: int(r["acquisition"].get(k, 0)) for k in ACQ_KEYS}
        if r["budget"] is None:
            if acq != ZERO_ACQ:
                bad.append(f"{cell_key(r)}: zero-shot cell carries acquisition {acq}")
            continue
        groups.setdefault((r["level"], r["task"], r["body"], r["budget"]), []).append((cell_key(r), acq))
    for g, items in groups.items():
        if len({json.dumps(a, sort_keys=True) for _, a in items}) > 1:
            bad.append(f"unequal acquisition in {g}: " + "; ".join(f"{k} {a}" for k, a in items))
    return bad


def read_json(p: Path):
    return json.loads(p.read_text())


def _absent(m: dict, status: str, reason: str, expect: dict) -> dict:
    """A run that does not exist yet: `pending` when the method names the stage that produces it, else `status`."""
    if m.get("producer"):
        return dict(status="pending", reason=f"{reason}; produced by {m['producer']}", acquisition=expect)
    return dict(status=status, reason=reason, acquisition=expect)


def _is_actor_file(spec: str) -> bool:
    return str(spec).endswith(".pt")


def cell_state(cfg: dict, cell: dict, root: Path, pack: dict) -> dict:
    """Pre-flight of one cell without simulating: status ready | pending | missing_run | unaccounted, the reason and the acquisition.
    A budgeted cell needs the run's `acquisition.json` (path template `acquisition` of the method, next to its checkpoint) and it
    must equal the pack's record (a mismatch is `unaccounted` whatever the method declares); a trained/expert cell needs its
    checkpoints / its tracker (a registry spec or an actor file). A run that does not exist yet is `pending` when the method has a
    `producer` and `missing_run` / `unaccounted` when it has none."""
    m = method_of(cfg, cell["method"])
    kv = dict(body=cell["body"], task=cell["task"], budget=cell["budget"], seed=cell["train_seed"])
    try:
        expect = acquisition_of_pack(pack, cfg, cell)
    except MissingRun as e:
        if m.get("producer") and not pack.get("records"):             # no pack exists at all: its stage has not run
            return _absent(m, "unaccounted", "the pack is not produced yet", dict(ZERO_ACQ))
        return dict(status="unaccounted", reason=str(e), acquisition=dict(ZERO_ACQ))
    if m["kind"] == "expert":
        spec = fmt(m["tracker"], **kv)
        if _is_actor_file(spec):
            if not (root / spec).exists():
                return _absent(m, "missing_run", f"tracker actor {spec} does not exist", expect)
        else:
            from rrp.envs.mujoco.legged_tracker import get_entry
            try:
                get_entry(spec)
            except (KeyError, FileNotFoundError, ValueError) as e:
                return _absent(m, "missing_run", str(e)[:200], expect)
    for k, v in (fmt(m.get("kw") or {}, **kv)).items():
        if isinstance(v, str) and v.endswith(".pt") and not (root / v).exists():
            return _absent(m, "missing_run", f"{k}: {v} does not exist", expect)
    if m.get("budgeted"):
        ap = fmt(m.get("acquisition", ""), **kv)
        if not ap or not (root / ap).exists():
            return _absent(m, "unaccounted", f"no acquisition.json ({ap or 'method declares none'})", expect)
        got = read_json(root / ap)
        if {k: int(got.get(k, 0)) for k in ACQ_KEYS} != expect:
            return dict(status="unaccounted", reason=f"{ap} records {got}, the pack says {expect}", acquisition=expect)
    return dict(status="ready", reason="", acquisition=expect)


def build_policy(cfg: dict, cell: dict, root: Path):
    """(policy, env_kw): the cell's policy over the env the method declares. A source label that does not match the policy's
    own is an error (teacher / scripted / privileged / learned / bc are never mislabelled)."""
    from rrp.policies.base import make_policy
    m = method_of(cfg, cell["method"])
    kv = dict(body=cell["body"], task=cell["task"], budget=cell["budget"], seed=cell["train_seed"])
    env_kw = fmt(m.get("env_kw") or {}, **kv)
    if m["kind"] == "expert":
        from rrp.policies.teachers.humanoid import make_rl_expert
        spec = fmt(m["tracker"], **kv)
        if _is_actor_file(spec):                       # an adapted actor (adapt_ppo output): registered for this process, sha pinned
            spec = resolve_actor(cell["body"], str(root / spec))
        pol, env_kw = make_rl_expert(arg=spec), dict(env_kw, tracker=spec)
    else:
        kw = fmt(m.get("kw") or {}, **kv)
        if m["policy"] == "legged_latent":
            kw = {**fmt(cfg.get("latent_kw") or {}, **kv), **kw}          # config-level realizer options (wholebody: upper=True)
        for k in ("flow", "rep", "realizer", "checkpoint"):
            if k in kw:
                kw[k] = str(root / kw[k])
        pol = make_policy(fmt(m["policy"], **kv), **kw)
    actual = str(getattr(pol.info.source, "value", pol.info.source))
    if actual != m["source"]:
        raise ValueError(f"method {m['name']}: declared source {m['source']!r} but the policy is {actual!r}")
    return pol, env_kw


def run_cell(cfg: dict, cell: dict, state: dict, scenes, root: Path, out: Path, batch: int = 4) -> dict:
    from rrp.harness.eval.evaluate import evaluate, task_hooks
    pol, env_kw = build_policy(cfg, cell, root)
    key = cell_key(cell)
    eps = evaluate(pol, cfg["env"], cell["task"], cell["body"], scenes, scene=cfg.get("scene"), batch=batch,
                   hooks=task_hooks(cell["task"], cfg["env"]), env_kw=env_kw,
                   out=out / "episodes" / (key.replace("|", "__").replace("/", "_") + ".jsonl"),
                   row_extra=dict(cell=key, method=cell["method"], budget=cell["budget"], train_seed=cell["train_seed"]))
    k = sum(bool(e.success_privileged) for e in eps)
    fails = Counter((e.failure_reason or e.outcome) for e in eps if not e.success_privileged)
    return dict(k=k, n=len(eps), failures=dict(fails), source=pol.info.source, policy=pol.info.name,
                policy_version=pol.info.version, acquisition=state["acquisition"])


def append_result(out: Path, rec: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / RESULTS, "a") as fh:
        fh.write(json.dumps(rec, default=str, sort_keys=True) + "\n")


def read_results(out: Path) -> list[dict]:
    p = out / RESULTS
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def merged_results(cfg: dict, root: Path, out: Path) -> list[dict]:
    """Results of this out dir plus those of every dir matched by the config's `results_glob` (one dir per recipe point)."""
    dirs = [out] + sorted({p for g in cfg.get("results_glob") or [] for p in root.glob(fmt(g, task=cfg["task"]))
                           if p.is_dir() and p.resolve() != out.resolve()})
    return [r for d in dirs for r in read_results(d)]


def run_matrix(cfg: dict, *, root: Path, out: Path, scope: str, sealed_flag: bool, run: bool, pack: dict, split=None,
               log=print, variant: str | None = None, train_seed: int | None = None) -> dict:
    """Plan or run the cells of `scope` (dev: non-sealed bodies on development scenes; sealed: sealed bodies on evaluation scenes,
    once each, `--sealed` required). Returns {cells: [...states], deferred_bodies}. Cells already done in results.jsonl are not
    rerun (dev); a sealed cell that is not ready never enters `sealed_eval` (the run-once budget is not spent on a missing run)."""
    from rrp.core.sealed import SealedSplit
    split = split or SealedSplit.load()
    want_sealed = scope == "sealed"
    if want_sealed and run and not sealed_flag:
        raise SystemExit("--scope sealed --run needs --sealed (sealed cells run once and are logged)")
    bodies = [b for b in cfg["bodies"] if split.is_sealed_body(b) == want_sealed]
    deferred = [b for b in cfg["bodies"] if b not in bodies]
    scenes = scenes_for(split, want_sealed, int(cfg["scenes"]))
    done = {r["key"] for r in read_results(out) if r.get("status") == "done" and r.get("scope") == scope}
    planned = []
    for cell in select_cells(cfg, expand_cells(cfg, bodies), variant, train_seed):
        st = cell_state(cfg, cell, root, pack)
        key = cell_key(cell)
        rec = dict(cell, key=key, scope=scope, **{k: st[k] for k in ("status", "reason", "acquisition")})
        planned.append(rec)
        if not run:
            continue
        if key in done:
            rec["status"] = "done_earlier"
            continue
        if st["status"] != "ready":
            log(f"[transfer] {key}: {st['status']} ({st['reason']})")
            append_result(out, rec)
            continue
        try:
            if want_sealed:
                with split.sealed_eval(sealed_cell(cfg, cell, scenes)):
                    res = run_cell(cfg, cell, st, scenes, root, out)
            else:
                res = run_cell(cfg, cell, st, scenes, root, out)
            rec.update(res, status="done")
            log(f"[transfer] {key}: {res['k']}/{res['n']} ({res['source']})")
        except Exception as ex:  # noqa: BLE001  (recorded; a sealed cell stays OPEN in the run-once log by design)
            rec.update(status="error", reason=f"{type(ex).__name__}: {str(ex)[:300]}")
            log(f"[transfer] {key}: error {rec['reason']}")
        append_result(out, rec)
    return dict(cells=planned, deferred_bodies=deferred, scenes=[scenes[0], scenes[-1]])


# ---- tables
def _row(recs: list[dict]) -> dict:
    from rrp.harness.eval.statistics import wilson
    k, n = sum(r["k"] for r in recs), sum(r["n"] for r in recs)
    lo, hi = wilson(k, n)
    fails = Counter()
    for r in recs:
        fails.update(r.get("failures") or {})
    r0 = recs[0]
    return dict(task=r0["task"], body=r0["body"], method=r0["method"], budget=r0["budget"], k=k, n=n, rate=(k / n if n else None),
                wilson95=[lo, hi], per_seed={str(r["train_seed"]): [r["k"], r["n"]] for r in sorted(recs, key=lambda r: r["train_seed"])},
                source=r0["source"], failures=dict(fails), acquisition=r0["acquisition"])


def make_tables(cfg: dict, results: list[dict]) -> dict:
    """Pooled k/n per (scope, level, body, method, budget) with Wilson, and method - reference per (body, budget) with Newcombe.
    Levels are separate sections and never pooled. Later records of a cell replace earlier ones (a rerun after an error)."""
    from rrp.harness.eval.statistics import newcombe_diff
    last = {}
    for r in results:
        last[r["key"]] = r
    done = [r for r in last.values() if r["status"] == "done"]
    viol = check_equal_acquisition([r for r in last.values() if r["status"] == "done"])
    tables: dict = {}
    for scope in sorted({r["scope"] for r in last.values()}):
        for lv in ("1", "2", "0"):
            recs = [r for r in done if r["scope"] == scope and r["level"] == lv]
            if not recs:
                continue
            groups: dict = {}
            for r in recs:
                groups.setdefault((r["body"], r["method"], r["budget"]), []).append(r)
            rows = [_row(g) for _, g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], -1 if kv[0][2] is None else kv[0][2]))]
            ref = (cfg["levels"].get(lv) or {}).get("reference")
            diffs = []
            for r in rows:
                if not ref or r["method"] == ref:
                    continue
                for rr in rows:
                    if rr["method"] == ref and rr["body"] == r["body"] and (r["budget"] is None or r["budget"] == rr["budget"]):
                        lo, hi = newcombe_diff(r["k"], r["n"], rr["k"], rr["n"])
                        diffs.append(dict(body=r["body"], method=r["method"], budget=rr["budget"], ref=ref, k=r["k"], n=r["n"],
                                          k_ref=rr["k"], n_ref=rr["n"], diff=r["k"] / r["n"] - rr["k"] / rr["n"], newcombe95=[lo, hi]))
            tables.setdefault(scope, {})[lv] = dict(title=LEVEL_NAMES[lv], unit=(cfg["levels"].get(lv) or {}).get("unit"), reference=ref,
                                                    rows=rows, diffs=diffs)
    cover = Counter((r["scope"], r["status"]) for r in last.values())
    missing = sorted(r["key"] for r in last.values() if r["status"] not in ("done",))
    return dict(name=cfg["name"], task=cfg["task"], tables=tables, coverage={f"{s}/{st}": n for (s, st), n in sorted(cover.items())},
                not_done=missing, accounting_violations=viol)


def _pct(x):
    return "-" if x is None else f"{x:.3f}"


def tables_markdown(t: dict) -> str:
    out = [f"# {t['name']} ({t['task']})", ""]
    out.append(f"coverage: {t['coverage']}")
    if t["accounting_violations"]:
        out += ["", "**ACQUISITION ACCOUNTING VIOLATIONS**"] + [f"- {v}" for v in t["accounting_violations"]]
    for scope, lvls in t["tables"].items():
        for lv, d in lvls.items():
            out += ["", f"## {scope}: {d['title']} (budget unit: {d['unit'] or 'none'}; reference: {d['reference'] or 'none'})", "",
                    "| body | method (source) | budget | k/n | rate | Wilson 95% | per training seed | failures |", "|---|---|---|---|---|---|---|---|"]
            for r in d["rows"]:
                out.append(f"| {r['body']} | {r['method']} ({r['source']}) | {r['budget'] if r['budget'] is not None else '-'} | {r['k']}/{r['n']} | "
                           f"{_pct(r['rate'])} | [{_pct(r['wilson95'][0])}, {_pct(r['wilson95'][1])}] | "
                           f"{', '.join(f'{s}: {a}/{b}' for s, (a, b) in r['per_seed'].items())} | {r['failures'] or '-'} |")
            if d["diffs"]:
                out += ["", f"method - {d['reference']} (Newcombe 95%)", "", "| body | method | budget | diff | 95% CI |", "|---|---|---|---|---|"]
                out += [f"| {x['body']} | {x['method']} | {x['budget']} | {x['diff']:+.3f} | [{_pct(x['newcombe95'][0])}, {_pct(x['newcombe95'][1])}] |"
                        for x in d["diffs"]]
    if t["not_done"]:
        out += ["", "not done (reported, not skipped):"] + [f"- {k}" for k in t["not_done"]]
    return "\n".join(out) + "\n"


def load_packs(cfg: dict, root: Path) -> dict:
    """The acquisition records of every pack the config names (`pack`: a path or a list of paths, relative to the root; a pack that
    does not exist yet contributes nothing, so its cells report `unaccounted`)."""
    paths = cfg.get("pack") or []
    paths = [paths] if isinstance(paths, str) else list(paths)
    recs: dict = {}
    for p in paths:
        pk = root / fmt(p, task=cfg.get("task", ""))
        if pk.exists():
            recs.update(read_json(pk).get("records") or {})
    return dict(records=recs)


def transfer_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rrp eval humanoid-transfer",
                                 description="Config-driven humanoid transfer evaluation: bodies x methods x demo budgets x training seeds "
                                             "(research/tracks/humanoid.md section 4). Default: plan (no simulation).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=".", help="repository root the config's run paths are relative to")
    ap.add_argument("--out", default=None, help="results directory (default artifacts/runs/humanoid/transfer/<name>)")
    ap.add_argument("--scope", choices=("dev", "sealed"), default="dev")
    ap.add_argument("--sealed", action="store_true", help="required to run sealed cells (run once, logged)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tables-only", action="store_true", help="only merge results (this dir + the config's results_glob) into tables")
    ap.add_argument("--variant", default=None, help="only methods with this `variant` ('-' = methods without one)")
    ap.add_argument("--train-seed", type=int, default=None, help="only this training seed of the trained methods")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    root = Path(a.root).resolve()
    out = Path(a.out) if a.out else root / "artifacts/runs/humanoid/transfer" / cfg["name"]
    if not a.tables_only:
        pack = load_packs(cfg, root)
        plan = run_matrix(cfg, root=root, out=out, scope=a.scope, sealed_flag=a.sealed, run=a.run, pack=pack,
                          variant=a.variant, train_seed=a.train_seed)
        c = Counter(x["status"] for x in plan["cells"])
        print(json.dumps(dict(scope=a.scope, cells=len(plan["cells"]), status=dict(c), deferred_bodies=plan["deferred_bodies"],
                              scenes=plan["scenes"], run=a.run)))
        if not a.run:
            for x in plan["cells"]:
                if x["status"] != "ready":
                    print(f"  {x['key']}: {x['status']} ({x['reason']})")
    t = make_tables(cfg, merged_results(cfg, root, out) if a.tables_only else read_results(out))
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.json").write_text(json.dumps(t, indent=1, default=str))
    (out / "tables.md").write_text(tables_markdown(t))
    print(f"tables: {out / 'tables.md'}  coverage {t['coverage']}  accounting violations {len(t['accounting_violations'])}")
    return 1 if t["accounting_violations"] else 0
