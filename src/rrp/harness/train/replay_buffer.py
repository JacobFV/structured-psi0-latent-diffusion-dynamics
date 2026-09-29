"""Versioned chunk-level replay for the EXPO-FT-inspired comparator.

A record is one decision point (one submitted chunk = SMDP option of `n_steps` control periods). It keeps
separately: the base proposal, the edit, the executed (post-bound-clip) native command rows and their
normalized form, the N base candidates sampled at that state (reused as next-state candidates in the TD
backup), a frozen public-observation embedding, versions (robot spec, controller, task graph/runtime,
base policy, edit policy, codec) and termination (done vs truncated). Records whose controller or robot
version differs from the buffer's are REJECTED, never silently mixed.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReplayRecord:
    episode_id: str
    step: int
    state_emb: np.ndarray                # [D] frozen public-observation embedding
    base_proposal: np.ndarray            # [C, n] normalized base rows the executed candidate came from
    edit: np.ndarray                     # [C, n] edit added (zeros when the unedited base won selection)
    executed_norm: np.ndarray            # [C, n] normalized rows actually submitted (after joint-bound clip)
    executed_native: list                # [C] {group: [values]} native command rows submitted
    base_candidates: np.ndarray          # [N, C, n] base samples at this state
    reward: float = 0.0                  # discounted within-chunk reward sum
    n_steps: int = 0                     # control periods executed until next decision / termination
    done: bool = False                   # true terminal (success / object fell): no bootstrap
    truncated: bool = False              # time limit: bootstrap from next_state_emb
    next_state_emb: np.ndarray | None = None
    next_base_candidates: np.ndarray | None = None
    selected: str = "base"               # base | edited
    robot_spec_hash: str = ""
    controller_version: str = ""
    graph_version: int = 0
    runtime_version: int = 0
    policy_version: str = ""
    edit_version: str = ""
    codec_version: str | None = None
    source: str = "learned"              # learned base + learned edit (never teacher)
    reward_label: str = "privileged_sim_success"

    @classmethod
    def fixture(cls, controller_version: str = "c1", robot_spec_hash: str = "r1", C=2, n=3, N=2, D=4):
        z = np.zeros((C, n), np.float32)
        return cls("ep0", 0, np.zeros(D, np.float32), z, z, z, [{"arm": [0.0] * n}] * C,
                   np.zeros((N, C, n), np.float32), controller_version=controller_version,
                   robot_spec_hash=robot_spec_hash)

    @property
    def ready(self) -> bool:
        return self.done or self.next_state_emb is not None


class ReplayBuffer:
    def __init__(self, controller_version: str, capacity: int, robot_spec_hash: str | None = None,
                 bc_capacity: int = 20000):
        self.controller_version = controller_version
        self.robot_spec_hash = robot_spec_hash
        self.capacity = capacity
        self.records: deque = deque(maxlen=capacity)
        self.bc: deque = deque(maxlen=bc_capacity)      # base flow-matching windows (successful episodes)
        self.appended = 0
        self.evicted = 0

    def check(self, rec: ReplayRecord):
        if rec.controller_version != self.controller_version:
            raise ValueError(f"replay controller version {rec.controller_version!r} != {self.controller_version!r}")
        if self.robot_spec_hash is not None and rec.robot_spec_hash != self.robot_spec_hash:
            raise ValueError("replay robot spec mismatch")
        if rec.executed_norm.shape != rec.base_proposal.shape or rec.edit.shape != rec.base_proposal.shape:
            raise ValueError("replay record action shapes disagree")
        if not (np.isfinite(rec.executed_norm).all() and np.isfinite(rec.state_emb).all()):
            raise ValueError("non-finite replay record")

    def append(self, rec: ReplayRecord):
        self.check(rec)
        if len(self.records) == self.capacity:
            self.evicted += 1
        self.records.append(rec)
        self.appended += 1

    def extend_episode(self, recs: list[ReplayRecord]):
        for r in recs:
            self.check(r)
        for r in recs:
            self.append(r)

    def add_bc(self, sample: dict):
        self.bc.append(sample)

    def __len__(self):
        return len(self.records)

    def sample(self, batch: int, rng: random.Random) -> list[ReplayRecord]:
        ready = [r for r in self.records if r.ready]
        if not ready:
            return []
        return [ready[rng.randrange(len(ready))] for _ in range(batch)]

    def sample_bc(self, batch: int, rng: random.Random) -> list[dict]:
        if not self.bc:
            return []
        return [self.bc[rng.randrange(len(self.bc))] for _ in range(batch)]
