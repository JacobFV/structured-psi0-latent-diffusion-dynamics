"""Scripted-teacher demo collection (`rrp train pointer collect`): DART-perturbed episodes -> one .npz per task."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from rrp.harness.train.pointer.data import TABLE_KEYS, stack_tables
from rrp.harness.train.pointer.split import excluded_seeds, heldout_goal, load_split
from rrp.policies.pointer import LI, NW, PointerGeometry, codes


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


def collect_episode(task: str, seed: int, *, dart_px: float, rng: random.Random, max_ticks: int = 400) -> dict | None:
    """One scripted-teacher episode. Labels are the teacher's clean commands; executed pointer commands get
    N(0, dart_px) noise on intermediate move ticks (DART; the teacher's goto corrects from wherever the pointer is; the
    arriving tick is never perturbed, else the goto would never terminate)."""
    from rrp.core.action import NativeCommand
    from rrp.envs.base import make_env
    from rrp.policies.pointer import EventHistory, env_widget_table, public_features, screen_half
    from rrp.policies.teachers.computerworld import CWTeacher
    from rrp.tasks.spec import get_task
    env = make_env("computerworld", task=task, body="cw_pointer", seed=seed)
    T = get_task(task)
    half = screen_half(env.spec)
    tt = CWTeacher(env, task)
    hist = EventHistory()
    rows, tabs, tab_keys = [], [], {}
    obs = env.observe()
    ok = False
    for tick in range(max_ticks):
        f = public_features(obs, half, hist, tick, table=env_widget_table(env))    # the live rollout's featurizer path
        c = tt.act()
        if c is None:
            break
        g = {k: list(v) for k, v in c.groups.items()}
        q = obs.measured_node_state.qpos[:2]
        ph = phase_of(g, hist.button, q)
        ex = {k: list(v) for k, v in g.items()}
        if dart_px > 0 and ph == 1 and tt.target_px is not None and \
                env.frame.m_to_px(*g["pointer"]) != tuple(tt.target_px):     # intermediate move ticks only
            ex["pointer"] = [g["pointer"][0] + rng.gauss(0, dart_px) * env.frame.m_per_px,
                             g["pointer"][1] + rng.gauss(0, dart_px) * env.frame.m_per_px]
        wkey = tuple(f[k].tobytes() for k in TABLE_KEYS)
        if wkey not in tab_keys:
            tab_keys[wkey] = len(tabs)
            tabs.append({k: f[k] for k in TABLE_KEYS})
        tslot = env.slots.slots.get(tt.target, -1) if tt.target else -1
        tpx = tt.target_px
        txy = (np.array(env.frame.px_to_m(*tpx)) / half) if tpx is not None else np.array([np.nan, np.nan])
        key = int(round(g.get("key", [-1])[0]))
        rows.append(dict(tab=tab_keys[wkey], ptr=f["ptr"], btn=f["btn"], hist=f["hist"],
                         cmd_xy=np.array(g["pointer"]) / half, cmd_btn=float(g["button"][0] >= 0.5), cmd_key=key + 1,
                         slot=tslot if tslot < NW else -1, txy=txy, phase=ph))
        hist.push(tick, ex, half)
        st = env.step(NativeCommand(controller_version="cw_pointer.v1", groups=ex, source="scripted_teacher"))
        obs = st.observation
        j = T.judge(env, st.time, T.max_seconds)
        if j.done:
            ok = j.outcome == "success"
            break
    goal, instr = dict(env.goal), obs.instruction
    env.close()
    if not ok:
        return None
    return dict(rows=rows, tabs=tabs, goal=goal, instr=instr)


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
        ep_rows.append((ep["seed"], t_start, len(ticks), codes(ep["instr"], LI), json.dumps(ep["goal"])))
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
        ep_goal=np.array([e[4] for e in ep_rows]), task=np.array(task),
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
    from rrp.envs.base import make_env
    probe = make_env("computerworld", task=a.task, body="cw_pointer", seed=0)
    while len(eps) < a.episodes:
        assert lo <= s < hi, "ran out of training seeds"
        probe.reset(s)
        probe._initial.clear()                    # the env caches one snapshot per seed; do not grow it
        if s in excl or heldout_goal(a.task, probe.goal, split):
            skipped += 1
            s += 1
            continue
        ep = collect_episode(a.task, s, dart_px=a.dart_px if rng.random() < a.dart_frac else 0.0, rng=rng)
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
