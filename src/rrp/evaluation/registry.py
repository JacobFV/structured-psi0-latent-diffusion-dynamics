"""Append-only experiment registry. Sealed (registered-before-results) runs cannot be mutated;
state transitions and results are appended as new rows referencing the run id."""
from __future__ import annotations

import fcntl
import hashlib
import json
import time
from pathlib import Path


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ExperimentRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def rows(self) -> list[dict]:
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def _append(self, row: dict):
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def get(self, run_id: str) -> dict | None:
        reg = [r for r in self.rows() if r.get("run_id") == run_id and r.get("kind") == "registration"]
        return reg[0] if reg else None

    def register(self, run_id: str, config: dict, *, sealed: bool = False, stage: str = "development",
                 question: str = "") -> dict:
        h = config_hash(config)
        prev = self.get(run_id)
        if prev is not None:
            if prev["config_hash"] == h:
                return prev          # idempotent re-registration of identical protocol
            if prev.get("sealed") or sealed:
                raise ValueError(f"run {run_id} is sealed; protocol cannot be mutated (register a new run id)")
            raise ValueError(f"run {run_id} already registered with a different config")
        row = dict(kind="registration", run_id=run_id, config=config, config_hash=h, sealed=sealed, stage=stage,
                   question=question, t=time.time())
        self._append(row)
        return row

    def update(self, run_id: str, state: str, **info):
        if self.get(run_id) is None:
            raise KeyError(run_id)
        self._append(dict(kind="state", run_id=run_id, state=state, t=time.time(), **info))

    def state(self, run_id: str) -> str | None:
        st = [r for r in self.rows() if r.get("run_id") == run_id and r.get("kind") == "state"]
        return st[-1]["state"] if st else None
