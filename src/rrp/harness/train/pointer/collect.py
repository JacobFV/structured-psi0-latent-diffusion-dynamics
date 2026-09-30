"""Scripted-teacher demo collection (`rrp train pointer collect`): DART-perturbed episodes on `harness.rollout` -> one .npz per
task (D-146 round 2 PC: the private tick loop is gone; the teacher is a `Policy`, the recorder a `Hook`)."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from rrp.harness.data.relgen.ui import teacher_drag_target
from rrp.harness.train.pointer.data import TABLE_KEYS, stack_tables
from rrp.harness.train.pointer.split import excluded_seeds, heldout_goal, load_split, make_split_env
from rrp.policies.pointer import LI, NW, PointerGeometry, codes, public_features
from rrp.tasks.spec import Judgement


def phase_of(groups: dict, prev_btn: bool, prev_xy) -> int:
    """PHASES index of a teacher tick: type, press, release, drag, move, idle."""
    if "key" in groups and groups["key"][0] >= 0:
        return 5
    b = groups["button"][0] >= 0.5
    if b and not prev_btn:
        return 2
    if prev_btn and not b:
        return 3
    moved = abs(groups["pointer"][0] - prev_xy[0]) + abs(groups["pointer"][1] - prev_xy[1]) > 1e-9
    if b:
        return 4
    return 1 if moved else 0


class DemoTeacher:
    """The scripted ComputerWorld teacher as a `Policy` (source `scripted_teacher`, privileged: it reads the env's widget
    slots and goal) that EXECUTES DART-perturbed commands and reports the teacher's clean command as the label.

    Labels are the teacher's clean commands; executed pointer commands get N(0, dart_px) noise on intermediate move
    ticks (the teacher's goto corrects from wherever the pointer is; the arriving tick is never perturbed, else the goto
    would never terminate). Each tick's public input comes from `public_features` (the live rollout's featurizer path) and
    is handed to `DemoRecorder` in `Act.info["demo_row"]`; the episode's own history is the policy's efference copy."""

    def __init__(self, task: str, *, dart_px: float, rng: random.Random):
        from rrp.policies.base import PolicyInfo, Requirements
        from rrp.policies.teachers.computerworld import TEACHER_VERSION
        self.task, self.dart_px, self.rng = task, dart_px, rng
        self.info = PolicyInfo(f"pointer_demo:{task}", "scripted_teacher", TEACHER_VERSION,
                               Requirements(frozenset({"cartesian_position", "button", "discrete"}),
                                            observations=frozenset(), tasks=frozenset({task}), privileged=True))

    def reset(self, spec, task, seeds, *, envs=None):
        from rrp.policies.pointer import EventHistory, screen_half
        from rrp.policies.teachers.computerworld import CWTeacher
        self.envs = list(envs)
        self.half = screen_half(spec)
        self.tt = [CWTeacher(e, self.task) for e in self.envs]
        self.hist = [EventHistory() for _ in self.envs]
        self.tick = [0] * len(self.envs)
        self.tabs: list[dict] = [{} for _ in self.envs]          # per episode: table key -> table index (insertion order)

    def act(self, obs):
        from rrp.core.action import NativeCommand
        from rrp.policies.base import Act
        from rrp.policies.pointer import env_widget_table
        out = {}
        for i, o in obs.items():
            env, tt, hist, half, tick = self.envs[i], self.tt[i], self.hist[i], self.half, self.tick[i]
            f = public_features(o, half, hist, tick, table=env_widget_table(env))
            c = tt.act()
            if c is None:                                          # the plan finished without the task judging success
                out[i] = Act(None, info=dict(teacher_done=True))
                continue
            g = {k: list(v) for k, v in c.groups.items()}
            ph = phase_of(g, hist.button, o.measured_node_state.qpos[:2])
            ex = {k: list(v) for k, v in g.items()}
            if self.dart_px > 0 and ph == 1 and tt.target_px is not None and \
                    env.frame.m_to_px(*g["pointer"]) != tuple(tt.target_px):     # intermediate move ticks only
                ex["pointer"] = [g["pointer"][0] + self.rng.gauss(0, self.dart_px) * env.frame.m_per_px,
                                 g["pointer"][1] + self.rng.gauss(0, self.dart_px) * env.frame.m_per_px]
            wkey = tuple(f[k].tobytes() for k in TABLE_KEYS)
            tabs = self.tabs[i]
            new_table = wkey not in tabs
            if new_table:
                tabs[wkey] = len(tabs)
            tslot = env.slots.slots.get(tt.target, -1) if tt.target else -1
            tpx = tt.target_px
            txy = (np.array(env.frame.px_to_m(*tpx)) / half) if tpx is not None else np.array([np.nan, np.nan])
            key = int(round(g.get("key", [-1])[0]))
            row = dict(tab=tabs[wkey], ptr=f["ptr"], btn=f["btn"], hist=f["hist"], cmd_xy=np.array(g["pointer"]) / half,
                       cmd_btn=float(g["button"][0] >= 0.5), cmd_key=key + 1, slot=tslot if tslot < NW else -1, txy=txy,
                       phase=ph)
            hist.push(tick, ex, half)
            self.tick[i] = tick + 1
            out[i] = Act(NativeCommand(controller_version="cw_pointer.v1", groups=ex, source="scripted_teacher"),
                         info=dict(demo_row=row, demo_table=({k: f[k] for k in TABLE_KEYS} if new_table else None)))
        return out


class DemoRecorder:
    """Rollout hook: the rows and widget tables a `DemoTeacher` reports, per episode; `on_end` keeps the goal and the
    instruction (read from the env after the last tick, as the old collector did)."""

    def __init__(self):
        self.rows: dict[int, list] = {}
        self.tabs: dict[int, list] = {}
        self.goal: dict[int, dict] = {}
        self.drag: dict[int, dict | None] = {}
        self.instr: dict[int, str] = {}
        self._obs: dict[int, object] = {}

    def on_reset(self, i, env, obs):
        self.rows[i], self.tabs[i] = [], []
        self.drag[i] = teacher_drag_target(env)          # the teacher's drag (handle, drop target) as widget slots

    def on_step(self, i, env, act, step):
        self._obs[i] = step.observation
        if act.info.get("teacher_done"):
            return Judgement(True, "failure", "teacher_plan_ended")      # the old collector stopped at the plan's end
        self.rows[i].append(act.info["demo_row"])
        if act.info["demo_table"] is not None:
            self.tabs[i].append(act.info["demo_table"])

    def on_end(self, i, env, ep):
        self.goal[i], self.instr[i] = dict(env.goal), (self._obs[i].instruction if i in self._obs else "")
        return {}


def collect_episode(task: str, seed: int, *, dart_px: float, rng: random.Random, max_ticks: int = 400,
                    env_kw: dict | None = None) -> dict | None:
    """One scripted-teacher episode on `harness.rollout` (a successful episode -> rows, widget tables, goal, instruction;
    a failed one -> None). The DART stream `rng` is consumed exactly as the tick loop consumed it."""
    from rrp.envs.base import make_env
    from rrp.harness.rollout import rollout
    from rrp.tasks.spec import get_task
    rec = DemoRecorder()
    T = get_task(task)
    eps = rollout(lambda sd: make_env("computerworld", task=task, body="cw_pointer", seed=sd, **(env_kw or {})),
                  DemoTeacher(task, dart_px=dart_px, rng=rng), T, [seed], batch=1, max_steps=max_ticks, hooks=[rec])
    ep = eps[0]
    if ep.outcome == "crash":
        raise RuntimeError(f"{task} seed {seed}: {ep.failure_reason}: {ep.metrics.get('note', '')}")
    if ep.outcome != "success":
        return None
    return dict(rows=rec.rows[0], tabs=rec.tabs[0], goal=rec.goal[0], instr=rec.instr[0], drag=rec.drag[0])


def write_pack(path, eps: list[dict], task: str, geom: PointerGeometry) -> tuple[int, int]:
    """Flatten collected episodes (`collect_episode` dicts + `seed`) into one .npz pack: tables, ticks, episodes.
    -> (ticks, tables)."""
    tabs, ticks, ep_rows = [], [], []
    for e_i, ep in enumerate(eps):
        base = len(tabs)
        tabs += ep["tabs"]
        t_start = len(ticks)
        for r in ep["rows"]:
            ticks.append(dict(r, tab=r["tab"] + base, ep=e_i))
        d = ep["drag"]
        ep_rows.append((ep["seed"], t_start, len(ticks), codes(ep["instr"], LI), json.dumps(ep["goal"]),
                        (-1, -1) if d is None or d["dst_slot"] is None else (d["src_slot"], d["dst_slot"]),
                        0 if d is None else 1))
    out = dict(
        **stack_tables(tabs),
        tab=np.array([t["tab"] for t in ticks], np.int32), ep=np.array([t["ep"] for t in ticks], np.int32),
        ptr=np.stack([t["ptr"] for t in ticks]).astype(np.float32), btn=np.array([t["btn"] for t in ticks], np.float32),
        hist=np.stack([t["hist"] for t in ticks]).astype(np.float32),
        cmd_xy=np.stack([t["cmd_xy"] for t in ticks]).astype(np.float32),
        cmd_btn=np.array([t["cmd_btn"] for t in ticks], np.float32), cmd_key=np.array([t["cmd_key"] for t in ticks],
                                                                                     np.int16),
        slot=np.array([t["slot"] for t in ticks], np.int16), txy=np.stack([t["txy"] for t in ticks]).astype(np.float32),
        phase=np.array([t["phase"] for t in ticks], np.int8),
        ep_seed=np.array([e[0] for e in ep_rows], np.int64), ep_start=np.array([e[1] for e in ep_rows], np.int64),
        ep_end=np.array([e[2] for e in ep_rows], np.int64), ep_instr=np.stack([e[3] for e in ep_rows]),
        ep_goal=np.array([e[4] for e in ep_rows]),
        ep_drag=np.array([e[5] for e in ep_rows], np.int16).reshape(-1, 2), ep_drags=np.array([e[6] for e in ep_rows], bool),
        task=np.array(task),
        geom=np.array(json.dumps(geom.as_dict())))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return len(ticks), len(tabs)


def cmd_collect(a):
    split = load_split(a.split)
    lo, hi = split["seed_ranges"]["train"]
    excl = excluded_seeds(split, a.task)
    rng = random.Random(a.seed)
    eps, s, skipped, failed = [], a.start, 0, 0
    t0 = time.time()
    probe = make_split_env(split, a.task)
    while len(eps) < a.episodes:
        assert lo <= s < hi, "ran out of training seeds"
        probe.reset(s)
        probe._initial.clear()                    # the env caches one snapshot per seed; do not grow it
        if s in excl or heldout_goal(a.task, probe.goal, split):
            skipped += 1
            s += 1
            continue
        ep = collect_episode(a.task, s, dart_px=a.dart_px if rng.random() < a.dart_frac else 0.0, rng=rng,
                             env_kw=split.get("env_kw"))
        if ep is None:
            failed += 1
        else:
            ep["seed"] = s
            eps.append(ep)
        s += 1
    geom = PointerGeometry.from_spec(probe.spec)     # stamped into the pack; `Demos` refuses a mismatch
    probe.close()
    n_ticks, n_tabs = write_pack(a.out, eps, a.task, geom)
    meta = dict(task=a.task, episodes=len(eps), ticks=n_ticks, tables=n_tabs, seeds=[int(eps[0]["seed"]),
                int(eps[-1]["seed"])], skipped_heldout_or_eval=skipped, teacher_failures=failed, dart_px=a.dart_px,
                dart_frac=a.dart_frac, source="scripted_teacher", wall_s=round(time.time() - t0, 1))
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
