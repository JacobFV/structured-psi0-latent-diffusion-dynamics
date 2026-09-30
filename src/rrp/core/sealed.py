"""Sealed split guard (D-138, D-146 item 2 and round 2 SL): the hash-pinned split files of `SPLITS` enforced in code.
`SealedSplit.load(name)` serves `humanoid_v1`, `armdiv_v1` and `cworld_pointer_v2`; each is pinned by sha256 (a changed file
is a new split id and a decision) and carries its own run-once log.

Sealed bodies. humanoid_v1: the target groups (g1_hands, n1, berkeley, toddlerbot_2xc / _2xm) plus the procedural phum
sealed seeds (generator seeds >= sealed_seed_range[0]; the generator rejects sealed-region samples for every training
seed, so the seed test is complete). armdiv_v1: every target robot, every key of an excluded-from-training family
(`gen3_*`, `rizon4_*`, `iiwa14_*`, `xarm7_*`, `lite6_*`, `fr3_*`, `panda_tf3`) and every procedural `pa2s<seed>_*` arm with
seed >= 900000 (`rrp.bodies.armdiv.is_sealed_target` reads this class: one rule). cworld_pointer_v2 has no sealed bodies:
its sealed material is the `sealed_id` / `sealed_heldout` seed lists of each task. Rules:

- `assert_train_allowed(bodies, seeds)`: no training or data collection on a sealed body except on target-adaptation
  demo seeds; evaluation and development seeds never enter any training or data collection.
- `assert_dataset_allowed(root, bodies)`: the same rule applied to the episode seeds recorded in the shard manifests
  a trainer is about to read.
- `assert_eval_allowed(body, seeds)`: a sealed body is evaluated only through `sealed_eval`.
- `sealed_eval(cell)`: a sealed cell runs ONCE. `cell` = dict(body, method, train_seed, scenes, [task, budget, adaptation,
  seed_set]); its id is body|method|task|n<budget>|adaptation|s<train_seed>|seed_set (absent fields are `-`; seed_set
  defaults to `evaluation`), so two cells that differ only in task, demo budget or adaptation are different cells. The
  run-once log (`SPLITS[name]["log"]`, e.g. `artifacts/runs/humanoid/sealed_log.jsonl`) records start / done /
  infrastructure_failure; a second start is refused unless the previous attempt has a recorded infrastructure failure
  (`record_infrastructure_failure`, with a reason; CLI `rrp suite sealed-log {list,infra-failure}`). An exception inside
  the block leaves the attempt open on purpose: only an explicit record releases the cell.

Non-sealed bodies keep whatever seeds their recipes declare (legacy seed conventions are not policed here).
Stdlib only (contracts layer).
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import fnmatch
import hashlib
import json
import re
import time
from pathlib import Path

from rrp.core.errors import RRPError
from rrp.core.paths import rrp_home

SCENES_PER_CELL = 100
# name -> pinned file, run-once log and how the split's fields map to the guard's four seed kinds. `ranges`: kind -> the
# `seed_ranges` key(s) of the spec (several keys = their union hull; None = the split has no such range). `bodies`: does the
# split protect bodies. `procedural`: (regex of a procedural body key, sealed seed minimum or None = read the spec).
SPLITS: dict[str, dict] = {
    "humanoid_v1": dict(
        path="research/splits/humanoid_v1.json", sha256="b3f591705b6eebf1815f1bc4d33b7e7072154bc9c88095a3e7695c9cf65aadb0",
        log="artifacts/runs/humanoid/sealed_log.jsonl", bodies=True, procedural=(r"phum_(\d+)", None),
        ranges=dict(source="source_train_demos", adaptation="target_adaptation_demos", evaluation="evaluation_scenes",
                    development="development")),
    "armdiv_v1": dict(
        path="research/splits/armdiv_v1.json", sha256="26448f52e7a2325ac74c0bb123616eda57362fdee142cf5defd914cec0cef414",
        log="artifacts/runs/armdiv/sealed_log.jsonl", bodies=True, procedural=(r"pa2s(\d+)_\w+", 900000),
        ranges=dict(source="source_train_demos", adaptation="target_adaptation_demos", evaluation="evaluation_scenes",
                    development="development_eval_scenes")),
    "cworld_pointer_v2": dict(
        path="research/splits/cworld_pointer_v2.json", sha256="eff81371c24cd0a299b4e6763bd0d3db3b83d2257bbd867a59515f595e2cc805",
        log="artifacts/runs/pointer/sealed_log.jsonl", bodies=False, procedural=None,
        ranges=dict(source="train", adaptation=None, evaluation=("sealed_id", "sealed_heldout"), development="dev")),
}
DEFAULT_SPLIT = "humanoid_v1"
CELL_KEYS = frozenset({"body", "method", "train_seed", "scenes", "task", "budget", "adaptation", "seed_set"})
_LOADED: dict[tuple[str, str], "SealedSplit"] = {}


class SealedSplitError(RRPError):
    code = "sealed_split_violation"


class SealedSplit:
    def __init__(self, spec: dict, name: str = DEFAULT_SPLIT):
        self.name, self.spec, self.cfg = name, spec, SPLITS[name]
        r = spec["seed_ranges"]

        def rng(keys):
            if keys is None:
                return None
            ks = [keys] if isinstance(keys, str) else list(keys)
            return (min(r[k][0] for k in ks), max(r[k][1] for k in ks))
        self.ranges = {kind: rng(k) for kind, k in self.cfg["ranges"].items()}
        proc = self.cfg["procedural"]
        self._proc_re = re.compile(proc[0]) if proc else None
        pf = spec.get("procedural_family")
        self.proc_sealed_min = (proc[1] if proc[1] is not None else int(pf["sealed_seed_range"][0])) if proc else None
        self.proc_train_max = int(pf["train_seed_range"][1]) if pf else None
        self.target_bodies = frozenset(
            b for t in spec.get("targets", {}).values() for b in (t["bodies"] if isinstance(t.get("bodies"), list) else t.get("robots", []))
        ) if self.cfg["bodies"] else frozenset()
        skip = proc[0].split("(")[0].replace("_", "") if proc else None          # armdiv `pa2s9*` is superseded by the numeric rule
        self._patterns = [p for p in spec.get("excluded_from_training", [])
                          if " " not in p and not (skip and p.startswith(skip))] if self.cfg["bodies"] else []

    @classmethod
    def load(cls, name: str = DEFAULT_SPLIT, *, path: str | Path | None = None) -> "SealedSplit":
        """The pinned split `name`. `path` reads another file under the same pin (tests of the pin itself)."""
        if name not in SPLITS:
            raise SealedSplitError(f"unknown sealed split {name!r} (known: {', '.join(SPLITS)})", code="sealed_split_unknown")
        p = Path(path) if path else rrp_home() / SPLITS[name]["path"]
        key = (name, str(p))
        if key in _LOADED:
            return _LOADED[key]
        try:
            raw = p.read_bytes()
        except OSError as e:
            raise SealedSplitError(f"sealed split file unreadable: {p} ({e})", code="sealed_split_missing") from e
        got, want = hashlib.sha256(raw).hexdigest(), SPLITS[name]["sha256"]
        if got != want:
            raise SealedSplitError(f"{p} sha256 {got[:16]} != pinned {want[:16]}: the split is frozen; a change "
                                   "needs a new split id and a decision", code="sealed_split_hash")
        _LOADED[key] = cls(json.loads(raw), name)
        return _LOADED[key]

    # ------------------------------------------------------------------ classification
    @staticmethod
    def _in(rng: tuple[int, int] | None, seed: int) -> bool:
        return rng is not None and rng[0] <= seed <= rng[1]

    def procedural_seed(self, body: str) -> int | None:
        m = self._proc_re.fullmatch(body) if self._proc_re else None
        return int(m.group(1)) if m else None

    def is_sealed_body(self, body: str) -> bool:
        s = self.procedural_seed(body)
        if s is not None:
            return s >= self.proc_sealed_min
        return body in self.target_bodies or any(fnmatch.fnmatch(body, p) for p in self._patterns)

    def seed_kind(self, seed: int) -> str | None:
        for k in ("evaluation", "development", "adaptation", "source"):
            if self._in(self.ranges[k], seed):
                return k
        return None

    # ------------------------------------------------------------------ guards
    def assert_train_allowed(self, bodies, seeds=(), *, what: str = "training") -> None:
        """Refuse `what` on `bodies` with demo/scene `seeds` when it would touch sealed material."""
        seeds = [int(s) for s in seeds]
        bad_eval = [s for s in seeds if self.seed_kind(s) in ("evaluation", "development")]
        if bad_eval:
            raise SealedSplitError(f"{what}: seeds {bad_eval[:3]}... lie in the evaluation/development ranges "
                                   f"{self.ranges['evaluation']} / {self.ranges['development']} of {self.name}; "
                                   "they never enter training", code="sealed_eval_seed_in_training")
        for b in bodies:
            s = self.procedural_seed(b)
            if s is not None and self.proc_train_max is not None and self.proc_train_max <= s < self.proc_sealed_min:
                raise SealedSplitError(f"{what}: {b} is outside the declared procedural training seeds (< {self.proc_train_max})",
                                       code="sealed_phum_seed_range")
            if not self.is_sealed_body(b):
                continue
            off = [s for s in seeds if not self._in(self.ranges["adaptation"], s)]
            if not seeds or off:
                raise SealedSplitError(f"{what} on sealed body {b} refused: only target-adaptation demo seeds "
                                       f"{self.ranges['adaptation']} are allowed"
                                       + (f" (got {off[:3]}...)" if off else " (no seeds given)"),
                                       code="sealed_body_in_training")

    def assert_dataset_allowed(self, root: str | Path, bodies) -> None:
        """Apply `assert_train_allowed` to the episode seeds recorded in `<root>/<body>/s*.json` shard manifests."""
        for b in bodies:
            seeds = [int(e["seed"]) for sh in sorted((Path(root) / b).glob("s*.json"))
                     for e in json.loads(sh.read_text()).get("episodes", [])]
            self.assert_train_allowed([b], seeds, what=f"dataset {Path(root) / b}")

    def assert_eval_allowed(self, body: str) -> None:
        if self.is_sealed_body(body):
            raise SealedSplitError(f"{body} is sealed: it is evaluated only through sealed_eval(cell) (run once)",
                                   code="sealed_body_eval")

    # ------------------------------------------------------------------ run-once cells
    @staticmethod
    def cell_id(cell: dict) -> str:
        """body|method|task|n<budget>|adaptation|s<train_seed>|seed_set; absent task / budget / adaptation are `-`, the seed
        set defaults to `evaluation`. Unknown keys are refused (a silently ignored field would merge two cells)."""
        extra = sorted(set(cell) - CELL_KEYS)
        if extra:
            raise SealedSplitError(f"sealed cell keys {extra} are not part of the cell identity {sorted(CELL_KEYS)}",
                                   code="sealed_cell_keys")
        b = cell.get("budget")
        parts = [cell["body"], cell["method"], cell.get("task") or "-", "-" if b is None else f"n{int(b)}",
                 cell.get("adaptation") or "-", f"s{int(cell['train_seed'])}", cell.get("seed_set") or "evaluation"]
        if any("|" in str(x) for x in parts):
            raise SealedSplitError(f"a sealed cell field contains '|': {parts}", code="sealed_cell_keys")
        return "|".join(str(x) for x in parts)

    def check_cell(self, cell: dict) -> None:
        self.cell_id(cell)
        scenes = [int(s) for s in cell["scenes"]]
        if self.cfg["bodies"]:
            if not self.is_sealed_body(cell["body"]):
                raise SealedSplitError(f"{cell['body']} is not a sealed body of {self.name}", code="sealed_cell_body")
            if len(scenes) != SCENES_PER_CELL or len(set(scenes)) != SCENES_PER_CELL:
                raise SealedSplitError(f"a sealed cell evaluates exactly {SCENES_PER_CELL} distinct scenes (got {len(scenes)})",
                                       code="sealed_cell_scenes")
            off = [s for s in scenes if not self._in(self.ranges["evaluation"], s)]
            if off:
                raise SealedSplitError(f"sealed scenes must lie in the evaluation range {self.ranges['evaluation']} "
                                       f"(got {off[:3]}...)", code="sealed_cell_scene_range")
            return
        # a split without sealed bodies seals seed lists: the cell names (task, seed set) and evaluates exactly that list
        task, sset = cell.get("task"), cell.get("seed_set")
        want = (self.spec.get("seeds", {}).get(task) or {}).get(sset) if sset in ("sealed_id", "sealed_heldout") else None
        if want is None:
            raise SealedSplitError(f"{self.name}: a sealed cell needs a task with a sealed seed list and seed_set "
                                   f"sealed_id | sealed_heldout (got task={task!r}, seed_set={sset!r})", code="sealed_cell_seed_set")
        if scenes != [int(x) for x in want]:
            raise SealedSplitError(f"{self.name}: cell {task}/{sset} must evaluate exactly the declared {len(want)} seeds "
                                   f"(got {len(scenes)})", code="sealed_cell_scenes")

    def _log_path(self, log: str | Path | None) -> Path:
        return Path(log) if log else rrp_home() / self.cfg["log"]

    @staticmethod
    def cell_state(cid: str, log: str | Path) -> str:
        """'none' | 'started' | 'done' | 'infrastructure_failure' (the last recorded event of the cell)."""
        p, state = Path(log), "none"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["cell"] == cid:
                        state = {"start": "started", "done": "done", "infrastructure_failure": "infrastructure_failure"}[r["event"]]
        return state

    @staticmethod
    def _append(p: Path, cid: str, event: str, guard_states: tuple[str, ...], **extra) -> None:
        """Append one event under an exclusive lock, only when the cell's current state is one of `guard_states`."""
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            state = SealedSplit.cell_state(cid, p)
            if state not in guard_states:
                raise SealedSplitError(f"sealed cell {cid} is '{state}' in {p}: " + (
                    "it already ran (a rerun needs a recorded infrastructure failure)" if state in ("done", "started") else
                    "no open attempt to close"), code="sealed_cell_rerun")
            f.write(json.dumps(dict(cell=cid, event=event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra),
                               sort_keys=True) + "\n")

    @contextlib.contextmanager
    def sealed_eval(self, cell: dict, log: str | Path | None = None):
        """Run a sealed cell once: `with split.sealed_eval(cell): evaluate(...)`. cell = dict(body, method, train_seed, scenes
        [, task, budget, adaptation, seed_set]). The start is logged before the block; `done` after a clean exit; an exception
        leaves the attempt open."""
        self.check_cell(cell)
        cid, p = self.cell_id(cell), self._log_path(log)
        self._append(p, cid, "start", ("none", "infrastructure_failure"), scene_range=[min(cell["scenes"]), max(cell["scenes"])])
        yield cid
        self._append(p, cid, "done", ("started",))

    def record_infrastructure_failure(self, cell: dict | str, reason: str, log: str | Path | None = None) -> None:
        """Release an open sealed cell after an infrastructure failure (node lost, OOM kill, broken mount...), never after a
        bad result. The reason is required and stored."""
        if not str(reason).strip():
            raise SealedSplitError("an infrastructure failure needs a recorded reason", code="sealed_cell_reason")
        cid = cell if isinstance(cell, str) else self.cell_id(cell)
        self._append(self._log_path(log), cid, "infrastructure_failure", ("started",), reason=str(reason))


# ---------------------------------------------------------------------------------------------- CLI: rrp suite sealed-log
def log_rows(log: str | Path) -> list[dict]:
    p = Path(log)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def cell_table(log: str | Path) -> list[dict]:
    """One row per cell of a run-once log: id, state, number of starts, last event time and the last failure reason."""
    rows: dict[str, dict] = {}
    for r in log_rows(log):
        c = rows.setdefault(r["cell"], dict(cell=r["cell"], state="none", starts=0, last_utc=None, reason=None))
        c["state"] = {"start": "started", "done": "done", "infrastructure_failure": "infrastructure_failure"}[r["event"]]
        c["starts"] += r["event"] == "start"
        c["last_utc"] = r["utc"]
        if r["event"] == "infrastructure_failure":
            c["reason"] = r["reason"]
    return list(rows.values())


def main(argv=None) -> int:
    """`rrp suite sealed-log list [--split NAME] [--state S]` and `rrp suite sealed-log infra-failure --split NAME --cell ID
    --reason TEXT` (the only way to release an open sealed cell; never after a bad result)."""
    ap = argparse.ArgumentParser(prog="rrp suite sealed-log", description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="cells of the run-once log(s) with their state")
    ls.add_argument("--split", choices=sorted(SPLITS), help="one split (default: every split's log)")
    ls.add_argument("--state", choices=["started", "done", "infrastructure_failure"], help="only cells in this state")
    ls.add_argument("--log", help="read this log file instead of the split's (needs --split)")
    inf = sub.add_parser("infra-failure", help="release an OPEN cell after an infrastructure failure")
    inf.add_argument("--split", required=True, choices=sorted(SPLITS))
    inf.add_argument("--cell", required=True, help="cell id exactly as `list` prints it")
    inf.add_argument("--reason", required=True, help="what failed (node lost, OOM kill, broken mount, ...): required and stored")
    inf.add_argument("--log", help="log file (default: the split's)")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        if a.log and not a.split:
            ap.error("--log needs --split")
        for name in ([a.split] if a.split else sorted(SPLITS)):
            log = Path(a.log) if a.log else rrp_home() / SPLITS[name]["log"]
            table = [c for c in cell_table(log) if not a.state or c["state"] == a.state]
            print(f"{name}  {log}  ({len(table)} cells)")
            for c in table:
                print(f"  {c['state']:<22} starts={c['starts']}  {c['last_utc']}  {c['cell']}" + (f"  reason: {c['reason']}" if c["reason"] else ""))
        return 0
    try:
        SealedSplit.load(a.split).record_infrastructure_failure(a.cell, a.reason, a.log)
    except SealedSplitError as e:
        print(f"refused: {e}")
        return 2
    print(f"released {a.cell} ({a.split}): infrastructure failure recorded")
    return 0
