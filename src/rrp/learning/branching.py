"""Group collection for GRPO: independent full rollouts vs. shared-prefix branching.

Shared prefix (Z-1 §3.2 with an EVENT-defined branch point instead of a fixed chunk window): one leader
executes from reset until a registered PUBLIC runtime boundary (default: `grasp` status == succeeded).
The complete continuation state (physics, controller, task runtime, tracker/belief, command queue,
env/sampler RNG) is snapshotted via Session.snapshot(); G branch sessions restore it and roll out
independent SDE suffixes. The prefix is physically executed once, its chunks are excluded from every
member's trainable set, and its transitions are counted once as new experience (the (G-1) reuses are
reported separately). The branch point depends only on the public runtime, never on final outcomes.
If the leader terminates before the boundary, the group has no suffixes (no gradient) but its steps
still count as new experience.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from rrp.learning.rollout import EpisodeState, drive, finalize, event_boundary


@dataclass
class GroupResult:
    seed: int
    mode: str
    returns: list = field(default_factory=list)
    members: list = field(default_factory=list)     # trainable chunk records per member
    outcomes: list = field(default_factory=list)
    new_transitions: int = 0                         # physically executed control steps
    prefix_steps: int = 0
    reused_prefix_transitions: int = 0               # prefix steps NOT re-executed thanks to sharing
    prefix_chunks_excluded: int = 0
    reached_boundary: bool | None = None
    snapshot_s: float = 0.0
    restore_s: float = 0.0
    suffix_steps: list = field(default_factory=list)

    def summary(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "members"}
        d["trainable_chunks"] = sum(len(m) for m in self.members)
        return d


def reward_of(fin: dict, shaping_grasp: float = 0.0) -> float:
    """Sparse success reward (privileged simulator evaluator, sim training only) + optional documented
    shaping bonus for the PUBLIC grasp event (0 by default)."""
    return float(fin["privileged_success"]) + shaping_grasp * float(fin["grasp_public"])


def _make_sessions(make_scenario, seed, n):
    from rrp.sim.native import Session
    sc = make_scenario(seed)
    return [Session(sc, seed=seed) for _ in range(n)]


def collect_plain(policy, make_scenario, seeds: list[int], G: int, max_steps: int, shaping_grasp=0.0):
    """G independent full episodes per seed (same initial state, independent SDE noise)."""
    states = []
    for sd in seeds:
        for j, s in enumerate(_make_sessions(make_scenario, sd, G)):
            states.append(EpisodeState(s, sd, max_steps, tag=dict(seed=sd, member=j)))
    drive(policy, states)
    out = []
    for sd in seeds:
        gr = GroupResult(sd, "plain")
        for st in [x for x in states if x.seed == sd]:
            fin = finalize(st)
            gr.returns.append(reward_of(fin, shaping_grasp))
            gr.outcomes.append(fin["outcome"])
            gr.members.append(st.chunks)
            gr.new_transitions += st.steps
            gr.suffix_steps.append(st.steps)
        out.append(gr)
    policy.forget([st.session for st in states])
    return out


def collect_shared_prefix(policy, make_scenario, seeds: list[int], G: int, max_steps: int, event: str = "grasp",
                          shaping_grasp=0.0):
    boundary = event_boundary(event)
    leaders = [EpisodeState(_make_sessions(make_scenario, sd, 1)[0], sd, max_steps, tag=dict(seed=sd, leader=True))
               for sd in seeds]
    drive(policy, leaders, stop_fn=boundary)
    out, branches = [], []
    for ld in leaders:
        gr = GroupResult(ld.seed, "shared_prefix", new_transitions=ld.steps, prefix_steps=ld.steps,
                         prefix_chunks_excluded=len(ld.chunks), reached_boundary=ld.paused)
        out.append(gr)
        if not ld.paused:
            fin = finalize(ld)
            gr.outcomes.append("leader_" + fin["outcome"])
            continue
        t0 = time.perf_counter()
        snap = ld.session.snapshot()
        gr.snapshot_s = time.perf_counter() - t0
        from rrp.sim.native import Session
        t0 = time.perf_counter()
        for j in range(G):
            s = Session(ld.session.scenario, seed=ld.seed)
            s.restore(snap)
            policy.copy_prev(ld.session, s)
            branches.append((gr, EpisodeState(s, ld.seed, max_steps - ld.steps, tag=dict(seed=ld.seed, member=j))))
        gr.restore_s = time.perf_counter() - t0
        gr.reused_prefix_transitions = (G - 1) * ld.steps
    drive(policy, [b for _, b in branches])
    for gr, st in branches:
        fin = finalize(st)
        gr.returns.append(reward_of(fin, shaping_grasp))
        gr.outcomes.append(fin["outcome"])
        gr.members.append(st.chunks)            # suffix chunks only
        gr.new_transitions += st.steps
        gr.suffix_steps.append(st.steps)
    policy.forget([ld.session for ld in leaders] + [b.session for _, b in branches])
    return out
