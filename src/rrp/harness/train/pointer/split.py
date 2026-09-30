"""Pointer split (research/splits/cworld_pointer_v1.json): seed lists, held-out variants and the no-leak guard."""
from __future__ import annotations

import json
from pathlib import Path

SPLIT_PATH = "research/splits/cworld_pointer_v1.json"
TASKS = ("cw/calc_sum", "cw/open_type", "cw/drag_window", "cw/fill_form")


def load_split(path: str = SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text())


def heldout_goal(task: str, goal: dict, split: dict) -> bool:
    h = split["heldout_variants"]
    if task == "cw/calc_sum":
        return [goal["a"], goal["b"]] in h["calc_pairs"]
    if task == "cw/open_type":
        return goal["text"] in h["words"]
    if task == "cw/fill_form":
        return goal["name"] in h["names"]
    return False


def excluded_seeds(split: dict, task: str) -> set[int]:
    s = split["seeds"][task]
    return set(s["dev"]) | set(s["sealed_id"]) | set(s.get("sealed_heldout", []))


def cmd_split(a):
    """Fill the seed lists: per task the first N seeds from each declared range whose goal is in/out of the held-out
    variants (goals are sampled at env reset, so this needs the ComputerWorld wheel)."""
    from rrp.envs.base import make_env
    split = load_split(a.split)
    rng = split["seed_ranges"]
    split["seeds"] = {}
    for task in TASKS:
        env = make_env("computerworld", task=task, body="cw_pointer", seed=0)
        out = {}
        for key, start, n, want_heldout in (("dev", rng["dev"][0], a.n_dev, False),
                                            ("sealed_id", rng["sealed_id"][0], a.n_sealed, False),
                                            ("sealed_heldout", rng["sealed_heldout"][0], a.n_heldout, True)):
            if want_heldout and task == "cw/drag_window":
                continue
            got, s = [], start
            while len(got) < n:
                env.reset(s)
                env._initial.clear()
                if heldout_goal(task, env.goal, split) == want_heldout:
                    got.append(s)
                s += 1
            out[key] = got
        split["seeds"][task] = out
        env.close()
    Path(a.split).write_text(json.dumps(split, indent=1) + "\n")
    print(json.dumps({t: {k: len(v) for k, v in d.items()} for t, d in split["seeds"].items()}))


def check_no_leak(data, split: dict) -> None:
    """Split guard: every training episode comes from the train seed range, is not a dev/sealed seed and does not use
    a held-out variant."""
    lo, hi = split["seed_ranges"]["train"]
    for seed, ti, g in zip(data.ep_seed, data.ep_task, data.goals):
        task = TASKS[int(ti)]
        if not lo <= int(seed) < hi or int(seed) in excluded_seeds(split, task) or heldout_goal(task, json.loads(str(g)),
                                                                                                split):
            raise RuntimeError(f"split leak: {task} seed {seed} goal {g}")
