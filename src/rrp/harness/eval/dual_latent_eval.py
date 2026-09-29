"""Closed-loop evaluation of the corrected latent path on the dual-arm scenarios (support_insert, handover).

System i (flow) emits ONE LatentActionChunk per replan period for the whole multi-robot scene: z[K, M=2, dz] with
packet slots in declared role order (slot 0 = the assembly bound to `left`, slot 1 = `right` for the two-body pairs;
a body carrying both grippers uses its declared assembly order). Each slot carries an opaque AssemblyHandle
(per-robot spec hash + TCP link, robot index); a missing assembly is an explicit null slot (mask False, null handle).
System 0 realizes the received packet every control tick: every action node attends ONLY to its own assembly's
slot, with its own assembly's touch/width sensors, current proprio and phase; the flat `r<i>:<group>` output is
split into per-robot native commands.

Diagnostics (never fed back into control): packet-only probes on the ACTUAL noise-started packets, scored per slot
against privileged labels permuted into slot order (held_by / acting_on / rel_pos) and public per-slot subtask
labels; slot-swap control (the same packet with slots exchanged) tests that meaning is slot-addressed.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from rrp.core.errors import ControllerRejection, StaleActionError
from rrp.core.provenance import stamp_source_label
from rrp.harness.data import dual_latent as DL
from rrp.harness.data.packed import _focus
from rrp.policies.features.derived import OPERATORS
from rrp.policies.features.multi import (MultiFeaturizer, assembly_operators, multi_featurizer,  # noqa: F401
                                         slot_assemblies)
from rrp.policies.latent import DualLatentPolicy
from rrp.policies.system0 import DualLatentSystem0


def slot_label_columns(session, f: MultiFeaturizer, M: int) -> list[int]:
    roles = {ent: dict(robot=h.robot, assembly=h.assembly) for ent, h in session.handles.items()}
    order = [a.id for _, _, a in slot_assemblies(f, M)]
    return DL.label_columns_for_slots(dict(roles=roles), list(session.manip_map), M,
                                      spec_grip_order=order if len(f.subs) == 1 else None)


def dual_packet_labels(session, pi, M: int = 2) -> dict:
    """Probe targets at packet time, in packet slot order. Privileged (held/contact/rel) = diagnostics only."""
    from rrp.harness.data.collect import privileged_labels
    f = session._rrp_featurizer
    lab = privileged_labels(session, f)
    S = lab["slot_pos"].shape[0]
    ml = DL.multi_labels(lab, slot_label_columns(session, f, M), S, S, M)
    t = lambda x: torch.as_tensor(np.asarray(x))[None]
    return dict(visible=t(lab["slot_visible"][:S]), focus=t(_focus(pi, S)), gaze=t(lab["slot_gaze_angle"][:S]).float(),
                future_disp=torch.zeros(1, S, 3), held_m=t(ml["held_m"]), contact_m=t(ml["contact_m"]),
                rel_tcp_m=t(ml["rel_tcp_m"].astype(np.float32)),
                subtask_m=t(assembly_operators(pi, OPERATORS, M)))


def probe_readout(out: dict, S: int, slot_names=("L", "R"), ent_names=None) -> list[str]:
    """Human-readable per-slot probe answers for captions (diagnostic)."""
    lines = []
    for m, nm in enumerate(slot_names[:out["subtask"].shape[1]]):
        op = OPERATORS[int(out["subtask"][0, m].argmax())]
        held = [ent_names[e] if ent_names else str(e) for e in range(S) if out["held_by"][0, e, m, 0] > 0]
        act = [ent_names[e] if ent_names else str(e) for e in range(S) if out["acting_on"][0, e, m, 0] > 0]
        lines.append(f"{nm}: {op} holds[{','.join(held) or '-'}] touches[{','.join(act) or '-'}]")
    return lines


@dataclass
class DualLatentEpisode:
    task: str
    pair: str
    seed: int
    method: str
    outcome: str
    privileged_success: bool
    public_success: bool
    steps: int
    system_i_calls: int
    system0_ticks: int
    packet_rejections: int
    fallback_holds: int
    sim_time: float
    wall_s: float
    events: dict
    probe_counts: dict = field(default_factory=dict)
    probe_counts_slotswap: dict = field(default_factory=dict)
    source: str = "learned"


def _acc(d, res):
    for q, (x, n) in res.items():
        a_, b_ = d.get(q, (0, 0))
        d[q] = (a_ + x, b_ + n)


def evaluate_dual_latent(policy: DualLatentPolicy, realizer, probe, task: str, pair: str, seeds: list[int], *,
                         method: str, replan_ticks: int = 8, max_steps: int = 800, batch: int = 20,
                         out_path: Path | None = None, device="cpu", packet_edit: str | None = None,
                         source_labels: bool | None = None) -> list[DualLatentEpisode]:
    """packet_edit='swap_slots' (causal intervention): system 0 receives each packet with its two slots' z values
    exchanged (handles unchanged), i.e. the left arm is driven by the right arm's latent and vice versa."""
    from rrp.policies.teachers.dual_validate import make_session
    from rrp.policies.teachers.dual import TEACHERS
    from rrp.policies.nets.latent_probes import probe_metrics
    results = []
    for i in range(0, len(seeds), batch):
        group = seeds[i:i + batch]
        S, meta, s0 = [], [], []
        for sd in group:
            s = make_session(task, pair, sd)
            feas = TEACHERS[task](s).feasibility()["feasible"]   # same layout filter as teacher validation
            f = policy.featurizer(s)
            S.append(s)
            s0.append(DualLatentSystem0(realizer, f, latent_space_version=policy.lsv,
                                        realizer_compat_version=policy.rcv, device=device))
            meta.append(dict(t0=time.time(), done=not feas, outcome=None if feas else "infeasible", steps=0, calls=0,
                             probes={}, probes_swap={}))
        for step in range(max_steps):
            act = [k for k, m in enumerate(meta) if not m["done"]]
            if not act:
                break
            need = [k for k in act if step % replan_ticks == 0 or s0[k].packet is None]
            if need:
                pk = policy.packets([S[k] for k in need])
                for k, p in zip(need, pk):
                    meta[k]["calls"] += 1
                    if packet_edit == "swap_slots":
                        p = p.model_copy(update={"z": np.ascontiguousarray(p.z[:, ::-1]), "source": "debug"})
                    try:
                        s0[k].receive(p, now=float(S[k].data.time), graph_version=S[k].runtime.graph_version)
                    except (ControllerRejection, StaleActionError):
                        pass
                    if probe is not None:            # diagnostic only
                        with torch.no_grad():
                            pi = S[k]._rrp_featurizer(S[k].observe())
                            M = p.z.shape[1]
                            lab = {kk: vv.to(device) for kk, vv in dual_packet_labels(S[k], pi, M).items()}
                            z = torch.from_numpy(p.z)[None].to(device)
                            am = torch.tensor([p.assembly_mask], device=device)
                            Sn = lab["visible"].shape[1]
                            sm = torch.ones(1, Sn, dtype=torch.bool, device=device)
                            _acc(meta[k]["probes"], probe_metrics(probe(z, am, Sn), lab, sm))
                            _acc(meta[k]["probes_swap"], probe_metrics(probe(z.flip(2), am.flip(1), Sn), lab, sm))
            for k in act:
                s = S[k]
                s.step(s0[k].tick(s))
                meta[k]["steps"] += 1
                if s.runtime.succeeded():
                    meta[k]["done"] = True
        for k, s in enumerate(S):
            m = meta[k]
            for _ in range(5 if m["outcome"] != "infeasible" else 0):
                s.step(None)                                     # settle, as in data collection
            priv = bool(s.privileged_success()) if m["outcome"] != "infeasible" else False
            if m["outcome"] is None:
                m["outcome"] = "success" if priv else ("timeout" if m["steps"] >= max_steps else "failure")
            results.append(DualLatentEpisode(task, pair, group[k], method, m["outcome"], priv, bool(s.runtime.succeeded()),
                                             m["steps"], m["calls"], s0[k].stats.ticks, s0[k].stats.rejected,
                                             s0[k].stats.fallback_holds, float(s.data.time), time.time() - m["t0"],
                                             {e: v.status for e, v in s.runtime.instances.items()}, m["probes"],
                                             m["probes_swap"], source=f"learned:{policy.name}" + (f"+edit:{packet_edit}" if packet_edit else "")))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as fh:
            for r in results:
                row = stamp_source_label(asdict(r), "learned", policy.name + (f"+edit:{packet_edit}" if packet_edit else ""),
                                         enabled=source_labels)      # D-126 sl-1; default off = unchanged rows
                fh.write(json.dumps(row) + "\n")
    return results


def teacher_reference(task: str, pair: str, seeds: list[int], max_steps: int = 800) -> list[dict]:
    """scripted_teacher (privileged) on the SAME development scenes, for reference."""
    from rrp.policies.teachers.dual_validate import run_one
    return [run_one(task, pair, sd, max_steps=max_steps) for sd in seeds]
