"""Pointer trainer losses and validation metrics: packet probe, tick action (smooth L1 / BCE / CE), (sum, count) pairs."""
from __future__ import annotations

import torch

from rrp.policies.pointer.spec import MAX_STEP_PX, PointerGeometry


def probe_loss(out, lab, specs):
    """Probe objectives on the packet (docs/relations.md 4, D-144 R6): target-slot CE, relative-target Gaussian NLL
    (log-variance floor `params.lv_min`, former CLI `--lv-min`, now on `specs` -- `pointer.pointer_probe_specs` /
    `ReadoutProbe.specs`), phase CE; ignored where the label is missing (-1 / not ok). `out`:
    `pointer.run_pointer_probe`'s dict (keys `slot` / `rel` / `phase`, unchanged from the pre-R6 probe)."""
    from rrp.policies.nets.probes import readout_loss
    out = {k: v.float() for k, v in out.items()}
    lab2 = dict(slot=lab["slot"].clamp(min=0), rel=lab["rel"], phase=lab["phase"].clamp(min=0))
    masks = dict(slot=lab["slot"] >= 0, rel=lab["rel_ok"], phase=lab["phase"] >= 0)
    total, logs = readout_loss(out, lab2, specs, masks)
    return total, {k[len("probe_"):]: v for k, v in logs.items()}


@torch.no_grad()
def probe_metrics(out, lab, specs, geom: PointerGeometry) -> dict:
    """(sum, count) pairs: slot top-1 accuracy, relative-position error (px, kept in pixel units -- the generic
    `readout_metrics` MAE is in normalized screen units -- for continuity with existing eval consumers), phase
    accuracy. `slot_acc` / `phase_acc` come from the foundation's `readout_metrics` (docs/relations.md 4): same
    masked-argmax-equality formula the pre-R6 bespoke code used, so the numbers are unchanged."""
    from rrp.policies.nets.probes import readout_metrics
    lab2 = dict(slot=lab["slot"].clamp(min=0), rel=lab["rel"], phase=lab["phase"].clamp(min=0))
    masks = dict(slot=lab["slot"] >= 0, rel=lab["rel_ok"], phase=lab["phase"] >= 0)
    m = readout_metrics(out, lab2, specs, masks)
    px = torch.tensor(geom.half_px, device=lab["rel"].device)         # normalized screen units -> px
    err = ((out["rel"][..., :2] - lab["rel"]) * px).norm(dim=-1)
    ro = lab["rel_ok"]
    return dict(slot_acc=m["slot_acc"], phase_acc=m["phase_acc"],
                rel_err=(float((err * ro.float()).sum()), int(ro.sum())))


def action_loss(xy_pred_steps, bl, kl, a):
    """Tick losses in pointer-step units (smooth L1), button BCE, key CE; masked by a['valid']."""
    import torch.nn.functional as F
    xy_pred_steps, bl, kl = xy_pred_steps.float(), bl.float(), kl.float()
    v = a["valid"].float()
    n = v.sum().clamp(min=1)
    Lxy = (F.smooth_l1_loss(xy_pred_steps, a["dxy_target"], beta=0.05, reduction="none").sum(-1) * v).sum() / n
    Lb = (F.binary_cross_entropy_with_logits(bl, a["btn"], reduction="none") * v).sum() / n
    Lk = (F.cross_entropy(kl.reshape(-1, kl.shape[-1]), a["key"].reshape(-1), reduction="none") * v.reshape(-1)).sum() / n
    return Lxy, Lb, Lk


@torch.no_grad()
def action_metrics(xy_pred_steps, bl, kl, a) -> dict:
    v = a["valid"]
    e = (xy_pred_steps - a["dxy_target"]).norm(dim=-1) * MAX_STEP_PX
    return dict(xy_px=(float((e * v).sum()), int(v.sum())),
                btn_acc=(int((((bl > 0).float() == a["btn"]) & v).sum()), int(v.sum())),
                key_acc=(int(((kl.argmax(-1) == a["key"]) & v).sum()), int(v.sum())),
                keypress_acc=(int(((kl.argmax(-1) == a["key"]) & v & (a["key"] > 0)).sum()), int((v & (a["key"] > 0)).sum())))


def _agg(acc: dict, m: dict):
    for k, (s, c) in m.items():
        s0, c0 = acc.get(k, (0.0, 0))
        acc[k] = (s0 + s, c0 + c)


def _fin(acc: dict) -> dict:
    return {k: (s / c if c else None) for k, (s, c) in acc.items()}
