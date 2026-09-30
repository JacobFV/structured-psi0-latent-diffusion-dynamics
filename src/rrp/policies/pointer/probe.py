"""Pointer packet probe (D-144 R6): `nets.probes.ReadoutProbe` configured by `probes:pointer-v1` (catalog.py)."""
from __future__ import annotations

from rrp.policies.pointer.spec import KNOT_TIMES

# ------------------------------------------------------------------------------------------- D-144 R6: relation-factor probe
# Former `PointerProbe` (content-keyed: cross-attended real widget label/role/geometry) -> `nets.probes.ReadoutProbe`
# configured by `probes:pointer-v1` (catalog.py). M = 1 (the pointer body's single "tool" assembly), so every query
# addresses `knot×asm` (psi0/legged precedent, docs/relations.md 4): the generic head reads only z's own K*M packet
# tokens plus fixed random handle codes, no widget content -- the "opaque codes only" design every other family's
# probe already follows (nets/probes.py's own docstring), which this migration now brings the pointer probe into
# line with. `target_slot` (query name `slot`, kept identical to the old dict key) is a fixed NW=80-way
# classification of the packet's target widget by SLOT INDEX (stable within an episode, docs/architecture.md
# ComputerWorld notes) rather than by content -- a probe that can no longer partly cheat off widget text/role, a
# strictly harder and more honest test of what `z` encodes than the pre-migration probe was.
POINTER_PROBE_PRESET = "probes:pointer-v1"


def pointer_probe_specs(lv_min: float | None = None):
    """`preset:probes:pointer-v1`, with the Gaussian `rel` query's `params.lv_min` overridden when given (former CLI
    `--lv-min`; D-085 bounded NLL, as the arm's semfix). Mirrors `policies.psi0.nets._with_lv_min`."""
    from dataclasses import replace
    from rrp.policies.relations.base import get_factor, resolve
    specs = resolve([f"preset:{POINTER_PROBE_PRESET}"])
    if lv_min is None:
        return specs
    out = []
    for s in specs:
        if get_factor(s.name).readout.loss == "gauss":
            p = dict(s.p); p["lv_min"] = lv_min
            s = replace(s, params=tuple(sorted(p.items())))
        out.append(s)
    return tuple(out)


def new_pointer_probe(dz=16, D=96, heads=4, metadata_only=False, seed=1234, lv_min: float | None = None):
    """The pointer packet probe: a `ReadoutProbe` (docs/relations.md 4) on `pointer_probe_specs(lv_min)`. Old
    `PointerProbe` checkpoints do not strictly load (different architecture); `load_pointer_probe_state` drops them
    and refits, the same fallback psi0 (R5) uses for the identical situation (D-144 addendum a, which names R6)."""
    from rrp.policies.nets.probes import ReadoutProbe
    return ReadoutProbe(dz, len(KNOT_TIMES), specs=pointer_probe_specs(lv_min), width=D, heads=heads,
                        max_assemblies=1, metadata_only=metadata_only, seed=seed)


def run_pointer_probe(P, z) -> dict:
    """P(z) narrowed to the former `PointerProbe` output shapes (M = 1 squeezed out of every `knot×asm` output):
    slot [B,K,NW] classification logits, rel [B,K,4] (Gaussian mu(2)+logvar(2)), phase [B,K,n_phases] logits."""
    import torch
    B = z.shape[0]
    zmask = torch.ones(B, 1, dtype=torch.bool, device=z.device)
    out = P(z, zmask)
    return {k: v[:, :, 0] for k, v in out.items()}


def _is_old_pointer_probe_state(sd: dict) -> bool:
    """True for a pre-D-144-R6 `PointerProbe` state dict (module names `bag` / `wf` / `role` / `rel` / `phase` /
    bare `kq`, none of which the new `ReadoutProbe` layout uses)."""
    return any(k.startswith(("bag.", "wf.", "role.", "rel.", "phase.")) or k == "kq" for k in sd)


def load_pointer_probe_state(sd: dict, *, dz=16, D=96, heads=4, metadata_only=False, seed=1234,
                             lv_min: float | None = None, device="cpu"):
    """`new_pointer_probe(...)`, loaded from `sd`: an old-architecture `PointerProbe` state is dropped (fresh init,
    refit -- see `new_pointer_probe`'s docstring); a post-D-144 `ReadoutProbe` state loads strictly."""
    P = new_pointer_probe(dz=dz, D=D, heads=heads, metadata_only=metadata_only, seed=seed, lv_min=lv_min).to(device)
    if _is_old_pointer_probe_state(sd):
        return P
    missing, unexpected = P.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"PointerProbe: key mismatch missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    return P
