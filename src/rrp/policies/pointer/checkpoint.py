"""Pointer nets access and checkpoints: `nets()` (the torch classes of `rrp.policies.nets.pointer`, imported lazily so
`import rrp.policies.pointer` stays torch-free for the scripted route), the pre-D-144-R6 `Block` -> `RelBlock` key
map, module loading and `load_pointer_bundle`."""
from __future__ import annotations

from rrp.policies.pointer.probe import load_pointer_probe_state

_NETS: dict | None = None


def nets() -> dict:
    """{class name: class} of the pointer nets (UICtx, PointerEncoder, PointerRealizer, PointerFlow, PointerBC) plus
    the `cross_relblock` / `self_relblock` builders."""
    global _NETS
    if _NETS is None:
        from rrp.policies.nets import pointer as m
        _NETS = dict(UICtx=m.UICtx, PointerEncoder=m.PointerEncoder, PointerRealizer=m.PointerRealizer,
                     PointerFlow=m.PointerFlow, PointerBC=m.PointerBC, cross_relblock=m.cross_relblock,
                     self_relblock=m.self_relblock)
    return _NETS


# ------------------------------------------------------------------------------------- D-144 R6: RelBlock checkpoint map
# `_BLOCK_PATHS[cls]`: every (dotted path prefix, mode) of a `blocks`-style ModuleList the class holds, `mode` being
# which `RelBlock` stage the former bespoke `Block`'s one attention module becomes ("cross" -> `.x`, old `n1` stays
# `n1`; "self" -> `.s`, old `n1` becomes `n2`, the LayerNorm immediately before `.s`). Old `n2` (pre-MLP LayerNorm)
# always becomes `n3`; old `m` always stays `m`. See `cross_relblock` / `self_relblock` above for why the OTHER
# stage needs no entry (it is left at its zero-init-output construction, an exact no-op).
_BLOCK_PATHS = {
    "PointerEncoder": (("ctx.blocks.", "self"), ("blocks.", "cross")),
    "PointerRealizer": (("blocks.", "cross"),),
    "PointerFlow": (("ctx.blocks.", "self"), ("blocks.", "cross"), ("self_blocks.", "self")),
    "PointerBC": (("ctx.blocks.", "self"), ("blocks.", "cross"), ("self_blocks.", "self")),
}


def _remap_block_state(sd: dict, prefix: str, mode: str) -> dict:
    """One (prefix, mode) entry of `_BLOCK_PATHS`: the old `Block` keys under `prefix` -> their `RelBlock` layout
    (see `_BLOCK_PATHS`'s docstring). Raises on a key it does not recognize (never silently drops data)."""
    ln1_new = "n1" if mode == "cross" else "n2"
    att_new = "x" if mode == "cross" else "s"
    out = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        i, sub = k[len(prefix):].split(".", 1)
        if sub.startswith("n1."):
            out[f"{prefix}{i}.{ln1_new}.{sub[len('n1.'):]}"] = v
        elif sub.startswith("a."):
            out[f"{prefix}{i}.{att_new}.{sub[len('a.'):]}"] = v
        elif sub.startswith("n2."):
            out[f"{prefix}{i}.n3.{sub[len('n2.'):]}"] = v
        elif sub.startswith("m."):
            out[f"{prefix}{i}.{sub}"] = v
        else:
            raise ValueError(f"_remap_block_state: unmapped old block key {k!r}")
    return out


def load_pointer_module(cls_name: str, ctor, sd: dict, device="cpu"):
    """Construct `ctor()` and load `sd`, remapping every `_BLOCK_PATHS[cls_name]` path whose checkpoint predates
    D-144 R6 (detected per-path by the old block's `.a.` attention key, which the new layout never has; a
    post-D-144 checkpoint has no such key anywhere and passes through unremapped). The only keys allowed to stay
    missing after a remap are the OTHER (zero-init, never-loaded) stage's parameters."""
    m = ctor().to(device)
    sd = dict(sd)
    expect_missing: set[str] = set()
    for prefix, mode in _BLOCK_PATHS.get(cls_name, ()):
        old = any(k.startswith(prefix) and k[len(prefix):].split(".", 1)[1].startswith("a.") for k in sd)
        if not old:
            continue
        remapped = _remap_block_state(sd, prefix, mode)
        for k in [k for k in sd if k.startswith(prefix)]:
            del sd[k]
        sd.update(remapped)
        unused = ("n2.", "s.") if mode == "cross" else ("n1.", "x.")   # the OTHER stage: its LayerNorm too
        expect_missing |= {k for k in m.state_dict()
                          if k.startswith(prefix) and k[len(prefix):].split(".", 1)[1].startswith(unused)}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if k not in expect_missing] + list(unexpected)
    if bad:
        raise RuntimeError(f"{cls_name}: key mismatch {bad[:5]}")
    return m


def load_pointer_bundle(path: str, device="cpu", allow_factor_mismatch: bool = False) -> dict:
    """A pointer checkpoint (rrp.harness.train.pointer): {'kind', 'config', 'state' (module -> state_dict), 'versions'}.
    `E` / `R` / `S` / `BC` load through `load_pointer_module` (RelBlock checkpoint map, D-144 R6: strict for a
    post-R6 checkpoint, key-mapped for a pre-R6 one); `P` (the packet probe) through `load_pointer_probe_state`
    (different architecture pre/post R6: key-mapped load is a refit, not a strict load, for an old checkpoint).
    C1: a factor-bearing module (`E` / `S` / `BC`) is rebuilt with the specs saved in `config["arch"][k]["factors"]`
    (absent = the empty `POINTER_FACTORS_PRESET`) and the structure hash in `versions["factors"]` must equal theirs
    (`relations.require_factors`; a checkpoint without a hash is refused; `allow_factor_mismatch` = a deliberate
    ablation load)."""
    import torch
    from rrp.policies.pointer.spec import POINTER_FACTORS_PRESET
    from rrp.policies.relations.base import require_factors, resolve
    st = torch.load(path, map_location=device, weights_only=False)
    N, cfg = nets(), st["config"]
    for k in ("E", "S", "BC"):
        if k in st["state"]:
            require_factors(st.get("versions"), resolve(cfg["arch"].get(k, {}).get("factors"),
                                                        default=POINTER_FACTORS_PRESET), allow_factor_mismatch)
    mods = {}
    for k, sd in st["state"].items():
        if k == "P":
            mods[k] = load_pointer_probe_state(sd, device=device, **cfg["arch"].get("P", {})).eval()
            continue
        cls = {"E": "PointerEncoder", "R": "PointerRealizer", "S": "PointerFlow", "BC": "PointerBC"}[k]
        mods[k] = load_pointer_module(cls, lambda cls=cls, kw=cfg["arch"].get(k, {}): N[cls](**kw), sd,
                                      device=device).eval()
    return dict(st, modules=mods)
