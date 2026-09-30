"""Loaders for trained bundles (W4: moved unchanged out of the training modules, which re-export every name), so that
evaluation, deployment and the workbench load trained models without importing training code.

arm (from rrp.learning.latent_train): `load_representation` -> (LatentConfig, encoder E, system 0 R, probe P, result)
with weight-fingerprinted compatibility IDs.
legged (from rrp.learning.legged_latent_train): `load_rep`, `checkpoint_provenance`, `legged_flags` /
`LEGGED_FLAG_KEYS`, and `_dev` (the legged device helper; apply_cap failures are ignored there, unlike the arm one).
"""
from __future__ import annotations

from pathlib import Path

import torch

from rrp.policies.features.legged import H
from rrp.policies.nets.checkpoint import load_checkpoint
from rrp.policies.nets.probes import ReadoutProbe
from rrp.policies.nets.legged_latent import LeggedEncoder, LeggedRealizer, LeggedProbe
from rrp.policies.nets.semantic_latent import LatentConfig, TargetEncoder

# ------------------------------------------------------------------ arm/dual packet probe (D-144 R1: PacketProbe ->
# ReadoutProbe, preset `probes:arm-packet-v1`; docs/relations.md 10). A saved `probe` config dict is still the legacy
# PacketProbe kwargs (dz/knots passed separately; optional `goal_effect`; `n_operators` no longer needed, the
# registered ReadoutDef fixes the subtask head width): translated here into ReadoutProbe specs, not into a new format,
# so old and new checkpoints load through the same path.
_DESIRED_DELTA_PREFIX = "heads.desired_delta."


def _readout_probe_specs(probe_cfg: dict) -> tuple[list, dict]:
    kw = dict(probe_cfg)
    goal_effect = kw.pop("goal_effect", False)
    kw.pop("n_operators", None)
    specs = ["preset:probes:arm-packet-v1"] + (["probe.arm.goal_effect"] if goal_effect else [])
    return specs, kw


def _remap_probe_state_dict(sd: dict) -> dict:
    """Checkpoints from before the desired_delta -> observed_effect rename: ReadoutProbe (unlike PacketProbe) keeps
    no `desired_delta` alias head, so the state-dict key is remapped in the load path instead (R1 brief)."""
    return {(k.replace(_DESIRED_DELTA_PREFIX, "heads.observed_effect.", 1)
             if k.startswith(_DESIRED_DELTA_PREFIX) else k): v for k, v in sd.items()}


def load_representation(path: Path, dev):
    st = load_checkpoint(path, map_location=dev)
    cfg = LatentConfig(**st["config"]["latent"])
    from rrp.policies.system0 import make_realizer
    specs, probe_kw = _readout_probe_specs(st["config"].get("probe", {}))
    E, R, P = TargetEncoder(cfg).to(dev), make_realizer(cfg.dz, cfg.realizer_layers, st["config"].get("realizer_arch")).to(dev), \
        ReadoutProbe(cfg.dz, cfg.knots, specs=specs, **probe_kw).to(dev)
    E.load_state_dict(st["model"]["E"]); R.load_state_dict(st["model"]["R"])
    P.load_state_dict(_remap_probe_state_dict(st["model"]["P"]))
    R.anchor = bool(st["config"].get("realizer_anchor", False))    # ladder: anchored realizer input (col 28)
    R.drop_qd = bool(st["config"].get("realizer_drop_qd", False))  # ladder: realizer input without joint velocity (col 27)
    for m in (E, R, P):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)    # frozen parameters; gradients still flow THROUGH P to its input z
    from rrp.policies.system0 import bundle_versions
    lsv, rcv = bundle_versions(cfg.version(), st["model"]["E"], st["model"]["R"])
    R.bundle_versions = (lsv, rcv)
    res = dict(st["extra"]["result"], config_latent_space_version=st["extra"]["result"]["latent_space_version"],
               latent_space_version=lsv, realizer_compat_version=rcv)
    return cfg, E, R, P, res


# ------------------------------------------------------------------ legged
def _dev():
    from rrp.ops.workload import select_device
    return select_device(on_cap_error="ignore")


LEGGED_FLAG_KEYS = ("semantic_weight", "beta_kl", "qd_dropout", "probe_lv_min", "packet_semantic_weight",
                    "packet_tau_min", "gen_frac", "zero_qd")


def legged_flags(cfg: dict | None) -> dict:
    """Training flags that change what a legged checkpoint means. Legged system 0 has no previous-action input,
    so the arm B-1 flag does not apply (recorded as prev_action_input=False instead of a zero_prev_action value)."""
    cfg = cfg or {}
    fl = dict(prev_action_input=False)
    for src in (cfg, cfg.get("latent") or {}):
        for k in LEGGED_FLAG_KEYS:
            if k in src:
                fl[k] = src[k]
    return fl


def checkpoint_provenance(st: dict, path=None, verify: bool = True):
    """Provenance of a loaded legged checkpoint dict. Bare (pre-W3) checkpoints -> legacy=True,
    'unfingerprinted'. With verify, stored weight fingerprints must match the loaded tensors."""
    from rrp.core.provenance import Provenance, legacy_provenance, weights_digest, is_state_dict, UNFINGERPRINTED
    if isinstance(st.get("_provenance"), dict):
        prov = Provenance.from_json(st["_provenance"])
        if verify:
            for k, d in prov.weights.items():
                if k in st and is_state_dict(st[k]) and weights_digest(st[k]) != d:
                    raise ValueError(f"{path}: weights fingerprint mismatch for {k!r} (file modified after save?)")
        return prov
    src = f"learned:{Path(path).parent.name}/{Path(path).name}" if path else "learned:unknown"
    return legacy_provenance(src, flags=legged_flags(st.get("cfg")), notes=UNFINGERPRINTED)


def load_rep(path, dev):
    st = torch.load(str(path), map_location=dev, weights_only=False)
    prov = checkpoint_provenance(st, path)           # raises on a fingerprint mismatch; legacy files are marked
    if prov.legacy:
        print(f"[legged] {path}: legacy checkpoint ({prov.notes}); compatibility IDs are fingerprinted at load",
              flush=True)
    lc = st["cfg"]["latent"]
    E = LeggedEncoder(dz=lc["dz"], D=lc["width"], H=H).to(dev)
    R = LeggedRealizer(dz=lc["dz"], D=lc["width"]).to(dev)
    P = LeggedProbe(dz=lc["dz"]).to(dev)
    E.load_state_dict(st["E"]); R.load_state_dict(st["R"]); P.load_state_dict(st["P"])
    for m in (E, R, P):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return st["cfg"], E, R, P, st["result"]
