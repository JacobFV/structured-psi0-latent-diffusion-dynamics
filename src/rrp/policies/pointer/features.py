"""Public featurization of the learned pointer policies (torch-free except `collate_public`).

The ONLY definition of what the learned pointer policies (system i, BC) may see: widget descriptors (slot-indexed;
label/role/box/depth/visible/focused/disabled/focusable/bound task entity), the instruction text, the pointer (qpos x, y)
and button sensor, the clock, and an efference copy of the policy's OWN executed events (button edges with the pointer
position, typed keys) - CW does not show typed text in the scene, so without it typing progress is unobservable. No
env.truth(), no teacher state, no CW scene internals. Position features are normalized by the env's screen half
extents (`screen_half(env.spec)`).
"""
from __future__ import annotations

import numpy as np

from rrp.policies.pointer.spec import (LC, LI, N_BOUND, NH, NW, PointerGeometry, ROLE_IDS, WF)


def _vocab():
    from rrp.envs.computerworld import KEY_NAMES, KEY_VOCAB, PRINTABLE
    return KEY_NAMES, KEY_VOCAB, PRINTABLE


def sym_of_char(c: str) -> int:
    """Shared symbol codes: 0 pad, 1..95 printable ASCII, 96 other character, 97.. named keys."""
    _, _, pr = _vocab()
    return pr.index(c) + 1 if c in pr else 96


def sym_of_key(k: int) -> int:
    names, vocab, _ = _vocab()
    return 97 + k if k < len(names) else sym_of_char(vocab[k])


N_SYM = 97 + 14
N_KEYCLS = 110                 # 0 = no key, 1 + KEY_VOCAB index


def codes(text: str, n: int) -> np.ndarray:
    out = np.zeros(n, np.int16)
    for i, c in enumerate((text or "")[:n]):
        out[i] = sym_of_char(c)
    return out


def bound_id(e) -> int:
    """Bucket (1 .. N_BOUND - 1; 0 = unbound) of a bound task entity: a stable hash of its id."""
    import zlib
    return 0 if e is None else 1 + zlib.crc32(e.id.encode()) % (N_BOUND - 1)


def check_bound_ids(descriptors) -> None:
    """Two DIFFERENT bound entities of one scene sharing a bucket would be one entity to every learned policy; raise
    instead of silently merging them."""
    seen: dict[int, str] = {}
    for d in descriptors:
        e = d.bound_entity
        if e is None or d.descriptor == "null" or d.bbox_xyxy is None:
            continue
        b = bound_id(e)
        if seen.setdefault(b, e.id) != e.id:
            raise ValueError(f"bound entity id collision: {seen[b]!r} and {e.id!r} share bucket {b} of "
                             f"{N_BOUND - 1}; widen N_BOUND (and retrain) or rename the entities")


def screen_half(spec) -> np.ndarray:
    """Half extents (m) of the screen: normalization of every position feature to [-1, 1]."""
    return PointerGeometry.from_spec(spec).half


def ui_widget_fields(table) -> dict:
    """R20's PUBLIC `ui_public_fields` / `ui_edges` (`rrp.envs.computerworld`, vocab `ui-rel-v1`), sliced/padded to
    NW slots -- `table` is the env's slot-indexed `scene_widgets` record list (`SlotRegistry.assign`, the exact same
    table `ComputerWorldEnv.observe()` passes to `descriptors()`), so slot `s` here is slot `s` of
    `obs.object_descriptors` / `widget_features` 1:1. `wzlayer` / `wparent` / `wfocusrank` and the `[NW, NW, 4]`
    `wuiedges` graph are what `UICtx`'s `preset:ui` factors read at the widget self-attention (`_relctx`)."""
    from rrp.envs.computerworld import UI_REL_VOCAB, ui_edges, ui_public_fields
    n = min(len(table), NW)
    t = list(table[:n])
    f, e = ui_public_fields(t), ui_edges(t)
    zlayer, parent, focusr = np.zeros(NW, np.float32), np.full(NW, -1, np.int64), np.full(NW, -1, np.int64)
    zlayer[:n], parent[:n], focusr[:n] = f["zlayer"], f["parent_id"], f["focus_rank"]
    edges = np.zeros((NW, NW, len(UI_REL_VOCAB)), bool)
    edges[:n, :n] = e
    return dict(wzlayer=zlayer, wparent=parent, wfocusrank=focusr, wuiedges=edges)


def widget_features(obs, half, table=None) -> dict:
    """Descriptor slots -> fixed arrays (slot index = row; slots >= NW are dropped). `wpos3d` / `wcamuvd` (D-144
    R20 follow-up): the same screen-geometry `widget_position` already puts in `d.position_estimate` (the 1 mm/px
    `ScreenFrame` mapping, `+` z-layer depth when `depth="stack"`) -- `pos3d` world-frame metres, `cam_uvd` the
    already-computed normalized screen (u, v) `+` the same depth, matching `geo.*`'s field kinds (`catalog.py`).
    `wgeo_ok` (C1): which slots' geometry is real (== `wmask` here; `Demos` clears it for packs collected before the
    geometry fields). `table` (the env's raw `scene_widgets` slot table, `env_widget_table(env)`) feeds R20's public
    UI fields / `ui-rel-v1` edges (`ui_widget_fields`) that `preset:ui` factors read. The key set is the same with or
    without it (an absent table is an EMPTY one: zero z-layer, parent / focus rank -1, no edges), so the ONE
    featurizer path of training collect (`harness.train.pointer.collect_episode`), `Demos.batch` and live rollout
    yields identical batch keys; every caller with an env passes `env_widget_table(env)`."""
    check_bound_ids(obs.object_descriptors[:NW])
    ch = np.zeros((NW, LC), np.int16)
    role = np.zeros(NW, np.int8)
    bound = np.zeros(NW, np.int8)
    wf = np.zeros((NW, WF), np.float32)
    m = np.zeros(NW, bool)
    pos3d = np.zeros((NW, 3), np.float32)
    camuvd = np.zeros((NW, 3), np.float32)
    for d in obs.object_descriptors[:NW]:
        if d.descriptor == "null" or d.bbox_xyxy is None:
            continue
        a = d.attributes or {}
        s = d.slot
        ch[s] = codes(a.get("label", ""), LC)
        role[s] = ROLE_IDS.get(d.descriptor, 9)
        bound[s] = bound_id(d.bound_entity)
        x, y, z = d.position_estimate
        x0, y0, x1, y1 = d.bbox_xyxy
        wf[s] = [x / half[0], y / half[1], (x1 - x0) * 1e-3 / half[0], (y1 - y0) * 1e-3 / half[1], z / 0.01,
                 float(d.visible), a.get("focused") == "true", a.get("disabled") == "true", a.get("focusable") == "true",
                 float(d.bound_entity is not None), 0.0]
        m[s] = True
        pos3d[s] = [x, y, z]
        camuvd[s] = [x / half[0], y / half[1], z]
    out = dict(wch=ch, wrole=role, wbound=bound, wf=wf, wmask=m, wpos3d=pos3d, wcamuvd=camuvd, wgeo_ok=m.copy())
    out.update(ui_widget_fields(table if table is not None else []))
    return out


class EventHistory:
    """Efference copy of the policy's own executed events: (kind 1 down / 2 up / 3 key, symbol, x, y, tick).
    `n_typed` (D-146 C2) counts the PRINTABLE keys it has executed since reset (the last NH events are all `array`
    keeps): with the instruction it says which character is next (`ntyped` in `public_features`)."""

    def __init__(self):
        self.ev: list[tuple[int, int, float, float, int]] = []
        self.button = False
        self.n_typed = 0

    def push(self, tick: int, groups: dict | None, half) -> None:
        if not groups:
            return
        x, y = groups.get("pointer", [np.nan, np.nan])
        if "button" in groups:
            b = groups["button"][0] >= 0.5
            if b != self.button:
                self.ev.append((1 if b else 2, 0, x / half[0], y / half[1], tick))
                self.button = b
        k = int(round(groups.get("key", [-1])[0]))
        if k >= 0:
            self.ev.append((3, sym_of_key(k), x / half[0], y / half[1], tick))
            self.n_typed += k >= len(_vocab()[0])

    def array(self, tick: int) -> np.ndarray:
        """[NH, 5] (kind, symbol, x, y, age in ticks) of the last NH events, oldest first; kind 0 = padding."""
        out = np.zeros((NH, 5), np.float32)
        ev = self.ev[-NH:]
        for i, (k, s, x, y, t) in enumerate(ev):
            out[i] = [k, s, np.nan_to_num(x), np.nan_to_num(y), tick - t]
        return out


def public_features(obs, half, hist: EventHistory, tick: int, *, table) -> dict:
    """One tick of public input (numpy, unbatched): the ONE featurizer of training collect, `Demos` and live rollout.
    `table` is required (`env_widget_table(env)`, taken right after `env.observe()`): see `widget_features`."""
    f = widget_features(obs, half, table)
    q = obs.measured_node_state.qpos
    btn = float(obs.declared_sensor_channels[0].values[0]) if obs.declared_sensor_channels else 0.0
    f.update(instr=codes(obs.instruction or "", LI), ptr=np.array([q[0] / half[0], q[1] / half[1]], np.float32),
             btn=np.float32(btn), tick=np.float32(tick), hist=hist.array(tick), ntyped=np.float32(hist.n_typed))
    return f


def collate_public(fs: list[dict], device) -> dict:
    import torch
    out = {}
    for k in fs[0]:
        a = np.stack([np.asarray(f[k]) for f in fs])
        t = torch.from_numpy(a)
        out[k] = (t.long() if a.dtype.kind in "iu" else t if a.dtype == bool else t.float()).to(device)
    return out
