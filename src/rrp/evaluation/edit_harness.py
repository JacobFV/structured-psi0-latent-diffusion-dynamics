"""Body-agnostic packet-edit / causal-test harness interfaces (W11 stable core API).

The arm and legged suites (rrp.evaluation.latent_causal, latent_semantic_edits, legged_latent_eval) are built on
MuJoCo sessions of rrp's own bodies. This module holds the parts that do not depend on a simulator, so an external
package (psi1z: G1 in SIMPLE / Isaac Sim) runs the SAME protocol with its own rollout function:

1. Conditions are declared up front (`EditCondition`: name, kind, prediction). Kinds: `control` (unedited),
   `replay` (control re-run: noise floor), `semantic` (a valid change of the task context or a probe-defined edit of
   z; behaviour should follow), `irrelevant_control` (matched-size edit that should NOT change behaviour),
   `negative_control` (z = 0, shuffled packet: a system 0 that ignores z would not change).
2. A frozen `PacketSource` produces the packet for (state, condition, noise key); control and edited runs share the
   noise key, so every edited run has an exactly paired control.
3. Edited packets are re-stamped with `restamp` and labelled source="debug" + sampling.intervention=<condition>
   (the policy itself stays labelled learned:<ckpt> in the rows).
4. `run_suite` runs every (seed, condition) through the caller's `run_one`, appends JSONL rows (resumable log), and
   `paired_effect` compares a condition with its control over paired units (bootstrap CI + sign-flip permutation p).
5. `matched_random`, `orthogonal_matched` and `probe_guided_edit` build z edits: random / probe-orthogonal directions
   of a matched norm, and gradient edits that move a probe readout while anchoring the others.

Probes only DEFINE edit directions; the evidence is closed-loop behaviour under the edited packet.
numpy (+ torch lazily, only in probe_guided_edit / probe_jacobian).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from rrp.evaluation.statistics import paired_bootstrap_ci, paired_permutation_test

EditKind = Literal["control", "replay", "semantic", "irrelevant_control", "negative_control"]
EDIT_KINDS = ("control", "replay", "semantic", "irrelevant_control", "negative_control")


@dataclass(frozen=True)
class EditCondition:
    name: str
    kind: EditKind
    prediction: str = ""                           # the declared, falsifiable expectation (written before running)
    params: dict = field(default_factory=dict)     # e.g. {"delta_m": [0.05, 0, 0]} or {"goal_offset_m": 0.12}

    def __post_init__(self):
        if self.kind not in EDIT_KINDS:
            raise ValueError(f"{self.name}: kind {self.kind!r} not in {EDIT_KINDS}")


@runtime_checkable
class PacketSource(Protocol):
    """A frozen producer of packets. `label` is one of the rrp source kinds (learned / oracle / bc / ...)."""
    label: str
    lsv: str
    rcv: str

    def packet(self, session: Any, condition: EditCondition, key: int) -> Any: ...


def restamp(packet, *, now: float, z=None, intervention: str | None = None, graph_version: int | None = None,
            runtime_version: int | None = None):
    """Hand-over transform of a packet: validity re-stamped to `now` (same interval length); optionally a new z.
    Any edit (`intervention`) relabels the packet source="debug" with sampling.intervention (same rule as
    rrp.evaluation.latent_causal.deliver, without a MuJoCo session)."""
    upd: dict = dict(valid_from=now, valid_until=now + (packet.valid_until - packet.valid_from))
    if graph_version is not None:
        upd["graph_version"] = graph_version
    if runtime_version is not None:
        upd["runtime_version"] = runtime_version
    if z is not None:
        upd["z"] = np.ascontiguousarray(np.asarray(z, np.float32))
    if intervention:
        upd.update(source="debug", sampling=dict(packet.sampling, intervention=intervention))
    return packet.model_copy(update=upd)


def run_suite(seeds: Iterable[int], conditions: Sequence[EditCondition],
              run_one: Callable[[int, EditCondition], dict], *, out_path: Path | None = None,
              log: Callable[[str], None] = print, done: set | None = None) -> list[dict]:
    """Every seed x condition through `run_one(seed, condition) -> row` (a row may carry "skipped": reason).
    Rows get seed, condition, kind, wall_s; each is appended to `out_path` (JSONL) as soon as it exists.
    `done`: {(seed, condition)} already present (resume), skipped here."""
    rows = []
    for sd in seeds:
        for c in conditions:
            if done and (sd, c.name) in done:
                continue
            t0 = time.time()
            r = dict(run_one(sd, c))
            r.setdefault("seed", sd)
            r.setdefault("condition", c.name)
            r.setdefault("kind", c.kind)
            r["wall_s"] = round(time.time() - t0, 3)
            rows.append(r)
            if out_path:
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(r, default=_json_default) + "\n")
            log(f"[edits] seed {sd} {c.name}: " + (r.get("skipped") or "ok"))
    return rows


def read_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def paired_effect(rows: Sequence[dict], metric: str | Callable[[dict], float | None], condition: str,
                  control: str = "control", unit: Sequence[str] = ("seed",), n_boot: int = 4000,
                  n_perm: int = 10000, seed: int = 0) -> dict:
    """Effect of `condition` vs `control` on `metric` over paired units (rows matched on `unit` keys; skipped rows
    and missing metrics dropped). Returns the paired bootstrap CI of mean(cond - control) and a sign-flip
    permutation p-value."""
    f = (lambda r: r.get(metric)) if isinstance(metric, str) else metric
    by: dict = {}
    for r in rows:
        if r.get("skipped") or r.get("condition") not in (condition, control):
            continue
        v = f(r)
        if v is None:
            continue
        by.setdefault(tuple(r.get(k) for k in unit), {})[r["condition"]] = float(v)
    pairs = [(d[condition], d[control]) for d in by.values() if condition in d and control in d]
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    ci = paired_bootstrap_ci(a, b, n=n_boot, seed=seed)
    pt = paired_permutation_test(a, b, n=n_perm, seed=seed)
    return dict(condition=condition, control=control, n_pairs=len(pairs), mean_diff=ci["mean"], lo=ci["lo"],
                hi=ci["hi"], p=pt["p"], exact_p=pt["exact"])


# ------------------------------------------------------------------------------------------------ z edits
def matched_random(delta: np.ndarray, key: int) -> np.ndarray:
    """Random direction with the norm of `delta` (specificity control; same as latent_causal._rand_like)."""
    g = np.random.default_rng(key)
    r = g.standard_normal(delta.shape).astype(np.float32)
    return r / np.linalg.norm(r) * np.linalg.norm(delta)


def orthogonal_matched(shape: tuple, norm: float, jacobian: np.ndarray, key: int) -> tuple[np.ndarray, dict]:
    """Random direction orthogonal to every row of `jacobian` ([readouts, prod(shape)]: gradients of the probe's
    semantic readouts w.r.t. z), scaled to `norm` (irrelevant-edit control of matched size)."""
    J = np.asarray(jacobian, np.float64).reshape(len(jacobian), -1)
    Q, _ = np.linalg.qr(J.T)
    r = np.random.default_rng(key).standard_normal(int(np.prod(shape)))
    r = r - Q @ (Q.T @ r)
    r = r / max(np.linalg.norm(r), 1e-12) * norm
    return r.reshape(shape).astype(np.float32), dict(probe_span_dim=int(np.linalg.matrix_rank(J)),
                                                     z_dim=int(np.prod(shape)))


def probe_jacobian(readout: Callable, z: np.ndarray) -> np.ndarray:
    """Rows = d(readout(z)[i]) / dz for every scalar of the flattened readout (torch autograd)."""
    import torch
    zt = torch.tensor(np.asarray(z, np.float32), requires_grad=True)
    y = readout(zt).reshape(-1)
    rows = []
    for i in range(len(y)):
        g, = torch.autograd.grad(y[i], zt, retain_graph=True)
        rows.append(g.reshape(-1).detach().cpu().numpy())
    return np.stack(rows)


def probe_guided_edit(z0: np.ndarray, target_loss: Callable, *, anchor_loss: Callable | None = None,
                      anchor: float = 1.0, l2: float = 1e-3, steps: int = 80, lr: float = 0.05,
                      device: str = "cpu") -> tuple[np.ndarray, dict]:
    """Find delta so that target_loss(z0 + delta) is small while anchor_loss(z0 + delta, z0) keeps every other
    readout where it was (Adam, fixed steps; the generic form of latent_causal.probe_edits). Both losses take
    torch tensors on `device` and return scalars. Returns (edited z, info)."""
    import torch
    z = torch.tensor(np.asarray(z0, np.float32), device=device)
    d = torch.zeros_like(z, requires_grad=True)
    opt = torch.optim.Adam([d], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            L = target_loss(z + d) + l2 * (d ** 2).mean()
            if anchor_loss is not None:
                L = L + anchor * anchor_loss(z + d, z)
            opt.zero_grad()
            L.backward()
            opt.step()
    with torch.no_grad():
        info = dict(target_loss=float(target_loss(z + d)), delta_norm=float(d.norm()),
                    delta_norm_ratio=float(d.norm() / max(float(z.norm()), 1e-9)))
        if anchor_loss is not None:
            info["anchor_residual"] = float(anchor_loss(z + d, z))
    return (z + d).detach().cpu().numpy(), info


# ------------------------------------------------------------------------------------ W12 contact / anchor causal specs
# Declared before running (predictions are falsifiable). Units: metres, packet slot order (left, right).
# The "anchor" of a hand is its contact_anchor receipt (rrp.envs.dual); the anchor-relative packet (anchor-targets-v1)
# plans hand poses relative to it. Two ways to shift it:
#   context edit  : the runtime receipt value seen by system i (and by the anchor-input system 0, rz-anchor-in-v1)
#                   is shifted by delta (shift_receipt_value); this is a valid, deploy-time information change
#   packet edit   : probe-guided z edit moving the anchor-relative readout of the dependent hand by -delta
#                   (anchor_shift_target), all other anchor/semantic readouts anchored
def contact_edit_conditions(delta_m=(0.03, 0.0, 0.0), task: str = "support_insert") -> list[EditCondition]:
    d = [float(x) for x in delta_m]
    hand = "right (inserting) hand relative to the support anchor" if task == "support_insert" else \
        "right (receiving) hand relative to the giver's anchor"
    return [
        EditCondition("control", "control", "reference"),
        EditCondition("replay", "replay", "noise floor: same seed, same packet source"),
        EditCondition("anchor_shift_context", "semantic",
                      f"re-anchoring: the {hand} follows the shifted anchor; its contact / final position moves by "
                      f"~delta (projected gain in [0.5, 1.5]) and the task still succeeds", dict(delta_m=d, level="context")),
        EditCondition("anchor_shift_packet", "semantic",
                      f"same prediction as the context shift, through a probe-guided z edit of the {hand}",
                      dict(delta_m=d, level="packet")),
        EditCondition("contact_order_swap", "semantic",
                      "the other hand makes its first task contact first: the observed sequence satisfies the swapped "
                      "spec (swap_hands) better than the original (order_error drops vs control)",
                      dict(level="context_or_packet")),
        EditCondition("anchor_shift_matched_random", "irrelevant_control",
                      "matched-norm random z direction: projected shift gain within the replay noise floor",
                      dict(delta_m=d, level="packet", like="anchor_shift_packet")),
        EditCondition("anchor_shift_probe_orthogonal", "irrelevant_control",
                      "matched-norm z direction orthogonal to the anchor-probe Jacobian: no re-anchoring",
                      dict(delta_m=d, level="packet", like="anchor_shift_packet")),
        EditCondition("irrelevant_anchor_shift", "irrelevant_control",
                      "shift an anchor no active event consumes (a completed event's receipt, or the other hand's "
                      "released contact) by the same delta: no behaviour change", dict(delta_m=d, level="context")),
        EditCondition("zero_packet", "negative_control", "z = 0: a system 0 that ignores z would not change"),
    ]


def shift_receipt_value(value: dict, delta_m) -> dict:
    """Copy of a contact_anchor / frame_estimate receipt value with its position shifted (context-level edit)."""
    v = dict(value)
    v["pos"] = [float(a) + float(b) for a, b in zip(value["pos"], delta_m)]
    return v


def anchor_shift_target(readout: Callable, z0: np.ndarray, delta_anchor_frame, slot: int,
                        query: str = "tcp_in_support") -> Callable:
    """Target loss for probe_guided_edit: move the anchor-relative position readout of `slot` (all knots) by
    -delta (expressed in the anchor frame, metres) -- "the anchor is delta further along". readout(z) returns the
    AnchorProbe output dict for a single packet (batch 1)."""
    import torch
    with torch.no_grad():
        r0 = readout(torch.as_tensor(np.asarray(z0, np.float32)))[query][0, :, slot, :3].clone()
    tgt = r0 - torch.as_tensor(np.asarray(delta_anchor_frame, np.float32) * 10.0)          # probe positions in dm
    return lambda z: ((readout(z)[query][0, :, slot, :3] - tgt) ** 2).mean()


def projected_shift(key: str, delta_m) -> Callable[[dict], float | None]:
    """Metric for paired_effect: row[key] (a 3-vector, e.g. the hand's contact point at insertion) projected on the unit
    edit direction; paired_effect(...).mean_diff / |delta| is the re-anchoring gain."""
    u = np.asarray(delta_m, float)
    u = u / max(np.linalg.norm(u), 1e-12)
    return lambda r: None if r.get(key) is None else float(np.dot(np.asarray(r[key], float), u))
