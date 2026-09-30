"""Pointer split (research/splits/cworld_pointer_v1.json, sealed; v2 = D-146 C2): seed lists, held-out variants and the
no-leak guard.

v1 holds out listed pool words / names. v2 (`cworld_pointer_v2.json`, procedural strings) holds out by RULE, because
a procedural string has no finite pool to list: `heldout_variants.string_rule` = {hash: crc32, modulus, buckets}; a
string (the lowercased word / name) is held out iff `crc32(string) % modulus` is in `buckets`. The rule is a pure
function of the string, so train / dev / sealed disjointness holds for every string the generator can ever draw.
`split["env_kw"]` (v2: strings = procedural) are the env kwargs every consumer builds its envs with (`make_split_env`)."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

SPLIT_PATH = "research/splits/cworld_pointer_v1.json"
SPLIT_PATH_V2 = "research/splits/cworld_pointer_v2.json"
TASKS = ("cw/calc_sum", "cw/open_type", "cw/drag_window", "cw/fill_form")


def load_split(path: str = SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text())


def make_split_env(split: dict, task: str, seed: int = 0, **kw):
    """The pointer env of `task` built with the split's declared env kwargs (v1 declares none)."""
    from rrp.envs.base import make_env
    return make_env("computerworld", task=task, body="cw_pointer", seed=seed, **{**split.get("env_kw", {}), **kw})


def string_heldout(text: str, rule: dict) -> bool:
    """`rule` = the split's `string_rule`: is this string in a held-out hash bucket?"""
    if rule["hash"] != "crc32":
        raise ValueError(f"string_rule hash {rule['hash']!r}: only crc32")
    return zlib.crc32(text.lower().encode()) % rule["modulus"] in rule["buckets"]


def heldout_goal(task: str, goal: dict, split: dict) -> bool:
    h = split["heldout_variants"]
    rule = h.get("string_rule")
    if task == "cw/calc_sum":
        return [goal["a"], goal["b"]] in h["calc_pairs"]
    if task == "cw/open_type":
        return goal["text"] in h.get("words", []) or (rule is not None and string_heldout(goal["text"], rule))
    if task == "cw/fill_form":
        return goal["name"] in h.get("names", []) or (rule is not None and string_heldout(goal["name"], rule))
    return False


def excluded_seeds(split: dict, task: str) -> set[int]:
    s = split["seeds"][task]
    return set(s["dev"]) | set(s["sealed_id"]) | set(s.get("sealed_heldout", []))


def cmd_split(a):
    """Fill the seed lists: per task the first N seeds from each declared range whose goal is in/out of the held-out
    variants. Goals are a pure function of (task, seed, strings) (`computerworld.goal_for`; cw/drag_window has no
    held-out variant, its seeds are consecutive), so no simulation runs."""
    from rrp.envs.computerworld import SEED_GOALS, goal_for
    split = load_split(a.split)
    rng, strings = split["seed_ranges"], split.get("env_kw", {}).get("strings", "pool")
    split["seeds"] = {}
    for task in TASKS:
        out = {}
        for key, start, n, want_heldout in (("dev", rng["dev"][0], a.n_dev, False),
                                            ("sealed_id", rng["sealed_id"][0], a.n_sealed, False),
                                            ("sealed_heldout", rng["sealed_heldout"][0], a.n_heldout, True)):
            if want_heldout and task == "cw/drag_window":
                continue
            got, s = [], start
            while len(got) < n:
                goal = goal_for(task, s, strings) if task in SEED_GOALS else {}
                if heldout_goal(task, goal, split) == want_heldout:
                    got.append(s)
                s += 1
            out[key] = got
        split["seeds"][task] = out
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
