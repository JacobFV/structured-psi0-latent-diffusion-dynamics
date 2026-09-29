"""Dual-arm latent evaluation conventions for harness.rollout (the loop is rollout + policies.latent.LatentStackPolicy
over DualLatentPolicy / DualLatentSystem0) on the dual-arm scenarios (support_insert, handover).

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

import numpy as np
import torch

from rrp.harness.data import dual_latent as DL
from rrp.harness.data.packed import _focus
from rrp.policies.features.derived import OPERATORS
from rrp.policies.features.multi import (MultiFeaturizer, assembly_operators, multi_featurizer,  # noqa: F401
                                         slot_assemblies)


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


def swap_slots(i: int, p):
    """Causal packet edit (LatentStackPolicy.packet_hook): system 0 receives the packet with its two slots' z values
    exchanged (handles unchanged), i.e. the left arm is driven by the right arm's latent and vice versa."""
    return p.model_copy(update={"z": np.ascontiguousarray(p.z[:, ::-1]), "source": "debug"})


PACKET_EDITS = {"swap_slots": swap_slots}


class DualPacketProbeHook:
    """on_act: packet-only probes on each packet system i emitted (after any edit), scored per slot against labels
    permuted into slot order at packet time, plus the slot-swapped control (probe on z with slots exchanged).
    on_end: probe_counts, probe_counts_slotswap. Diagnostics only."""

    def __init__(self, probe, device="cpu"):
        self.probe, self.device = probe, device
        self.envs, self.counts, self.swap = {}, {}, {}

    def on_reset(self, i, env, obs):
        self.envs[i], self.counts[i], self.swap[i] = env, {}, {}

    def on_act(self, i, obs, act):
        if act.packet is not None:
            self._score(i, act.packet)
        return act

    @torch.no_grad()
    def _score(self, i, p):
        from rrp.harness.eval.latent_eval import acc_probe_counts
        from rrp.policies.nets.latent_probes import probe_metrics
        env, dev = self.envs[i], self.device
        pi = env._rrp_featurizer(env.observe())
        lab = {k: v.to(dev) for k, v in dual_packet_labels(env, pi, p.z.shape[1]).items()}
        z = torch.from_numpy(p.z)[None].to(dev)
        am = torch.tensor([p.assembly_mask], device=dev)
        Sn = lab["visible"].shape[1]
        sm = torch.ones(1, Sn, dtype=torch.bool, device=dev)
        acc_probe_counts(self.counts[i], probe_metrics(self.probe(z, am, Sn), lab, sm))
        acc_probe_counts(self.swap[i], probe_metrics(self.probe(z.flip(2), am.flip(1), Sn), lab, sm))

    def on_end(self, i, env, ep):
        return dict(probe_counts=self.counts.pop(i), probe_counts_slotswap=self.swap.pop(i))


def teacher_reference(task: str, pair: str, seeds: list[int], max_steps: int = 800) -> list[dict]:
    """scripted_teacher (privileged) on the SAME development scenes, for reference."""
    from rrp.policies.teachers.dual_validate import run_one
    return [run_one(task, pair, sd, max_steps=max_steps) for sd in seeds]
