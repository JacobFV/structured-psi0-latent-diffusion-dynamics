"""Sealed humanoid split guard (D-138, D-146 item 2): the hash-pinned `research/splits/humanoid_v1.json` enforced in code.

Sealed bodies = the target groups of the split (g1_hands, n1, berkeley, toddlerbot_2xc / _2xm) plus the procedural
phum sealed seeds (generator seeds >= sealed_seed_range[0]; the generator rejects sealed-region samples for every
training seed, so the seed test is complete). Rules:

- `assert_train_allowed(bodies, seeds)`: no training or data collection on a sealed body except on target-adaptation
  demo seeds; evaluation and development seeds never enter any training or data collection.
- `assert_dataset_allowed(root, bodies)`: the same rule applied to the episode seeds recorded in the shard manifests
  a trainer is about to read.
- `assert_eval_allowed(body, seeds)`: a sealed body is evaluated only through `sealed_eval`.
- `sealed_eval(cell)`: a sealed cell (body x method x training seed, 100 evaluation scenes) runs ONCE. The run-once log
  `artifacts/runs/humanoid/sealed_log.jsonl` records start / done / infrastructure_failure; a second start is refused
  unless the previous attempt has a recorded infrastructure failure (`record_infrastructure_failure`, with a reason).
  An exception inside the block leaves the attempt open on purpose: only an explicit record releases the cell.

Non-sealed bodies keep whatever seeds their recipes declare (legacy seed conventions are not policed here).
Stdlib only (contracts layer).
"""
from __future__ import annotations

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

SPLIT_PATH = "research/splits/humanoid_v1.json"
SPLIT_SHA256 = "b3f591705b6eebf1815f1bc4d33b7e7072154bc9c88095a3e7695c9cf65aadb0"
SEALED_LOG = "artifacts/runs/humanoid/sealed_log.jsonl"
SCENES_PER_CELL = 100


class SealedSplitError(RRPError):
    code = "sealed_split_violation"


class SealedSplit:
    def __init__(self, spec: dict):
        self.spec = spec
        r = spec["seed_ranges"]
        self.ranges = dict(adaptation=tuple(r["target_adaptation_demos"]), evaluation=tuple(r["evaluation_scenes"]),
                           development=tuple(r["development"]), source=tuple(r["source_train_demos"]))
        self.phum_sealed_min = int(spec["procedural_family"]["sealed_seed_range"][0])
        self.phum_train_max = int(spec["procedural_family"]["train_seed_range"][1])
        self.target_bodies = frozenset(b for t in spec["targets"].values() if isinstance(t["bodies"], list)
                                       for b in t["bodies"])
        self._patterns = [p for p in spec["excluded_from_training"] if " " not in p]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SealedSplit":
        p = Path(path) if path else rrp_home() / SPLIT_PATH
        try:
            raw = p.read_bytes()
        except OSError as e:
            raise SealedSplitError(f"sealed split file unreadable: {p} ({e})", code="sealed_split_missing") from e
        got = hashlib.sha256(raw).hexdigest()
        if got != SPLIT_SHA256:
            raise SealedSplitError(f"{p} sha256 {got[:16]} != pinned {SPLIT_SHA256[:16]}: the split is frozen; a change "
                                   "needs a new split id and a decision", code="sealed_split_hash")
        return cls(json.loads(raw))

    # ------------------------------------------------------------------ classification
    @staticmethod
    def _in(rng: tuple[int, int], seed: int) -> bool:
        return rng[0] <= seed <= rng[1]

    def phum_seed(self, body: str) -> int | None:
        m = re.fullmatch(r"phum_(\d+)", body)
        return int(m.group(1)) if m else None

    def is_sealed_body(self, body: str) -> bool:
        s = self.phum_seed(body)
        if s is not None:
            return s >= self.phum_sealed_min
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
                                   f"{self.ranges['evaluation']} / {self.ranges['development']}; they never enter training",
                                   code="sealed_eval_seed_in_training")
        for b in bodies:
            s = self.phum_seed(b)
            if s is not None and self.phum_train_max <= s < self.phum_sealed_min:
                raise SealedSplitError(f"{what}: {b} is outside the declared phum training seeds (< {self.phum_train_max})",
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
        return f"{cell['body']}|{cell['method']}|s{int(cell['train_seed'])}"

    def check_cell(self, cell: dict) -> None:
        if not self.is_sealed_body(cell["body"]):
            raise SealedSplitError(f"{cell['body']} is not a sealed body", code="sealed_cell_body")
        scenes = [int(s) for s in cell["scenes"]]
        if len(scenes) != SCENES_PER_CELL or len(set(scenes)) != SCENES_PER_CELL:
            raise SealedSplitError(f"a sealed cell evaluates exactly {SCENES_PER_CELL} distinct scenes (got {len(scenes)})",
                                   code="sealed_cell_scenes")
        off = [s for s in scenes if not self._in(self.ranges["evaluation"], s)]
        if off:
            raise SealedSplitError(f"sealed scenes must lie in the evaluation range {self.ranges['evaluation']} "
                                   f"(got {off[:3]}...)", code="sealed_cell_scene_range")

    @staticmethod
    def _log_path(log: str | Path | None) -> Path:
        return Path(log) if log else rrp_home() / SEALED_LOG

    @staticmethod
    def cell_state(cid: str, log: str | Path | None = None) -> str:
        """'none' | 'started' | 'done' | 'infrastructure_failure' (the last recorded event of the cell)."""
        p = SealedSplit._log_path(log)
        state = "none"
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
        """Run a sealed cell once: `with split.sealed_eval(cell): evaluate(...)`. cell = dict(body, method, train_seed,
        scenes). The start is logged before the block; `done` after a clean exit; an exception leaves the attempt open."""
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
