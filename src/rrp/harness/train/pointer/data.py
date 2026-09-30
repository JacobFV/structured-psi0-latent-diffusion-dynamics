"""Demo packs on one device: the tick tables, 7-tick demo chunks and probe labels every pointer trainer samples."""
from __future__ import annotations

import json

import numpy as np
import torch

from rrp.envs.computerworld import UI_REL_VOCAB
from rrp.harness.train.pointer.split import TASKS
from rrp.policies.pointer.spec import NW, PointerGeometry

# per-table widget arrays a pack stores = the widget half of `rrp.policies.pointer.public_features`' output, one
# featurizer path shared with live rollout (C1). `wpos3d` / `wcamuvd` (D-144 R20 follow-up geometry) and the public UI
# fields `wzlayer` / `wparent` / `wfocusrank` / `wuiedges` (R20, `ui_widget_fields`) are what the net's `_relctx` reads
# besides the five descriptor arrays; packs collected before them lack them and `Demos` flags that (`wgeo` / `wui`)
TABLE_KEYS = ("wch", "wrole", "wbound", "wf", "wmask", "wpos3d", "wcamuvd", "wzlayer", "wparent", "wfocusrank",
              "wuiedges")
_EDGE_SHAPE = (NW, NW, len(UI_REL_VOCAB))


def stack_tables(tabs: list[dict]) -> dict:
    """Per-table feature dicts -> the pack's table arrays (on-disk dtypes). `wuiedges` is bit-packed per table
    ([n, ceil(NW * NW * R / 8)] uint8; `unpack_edges` inverts it)."""
    st = lambda k, dt: np.stack([t[k] for t in tabs]).astype(dt)
    return dict(wch=st("wch", np.int16), wrole=st("wrole", np.int8), wbound=st("wbound", np.int8),
                wf=st("wf", np.float16), wmask=st("wmask", bool), wpos3d=st("wpos3d", np.float32),
                wcamuvd=st("wcamuvd", np.float32), wzlayer=st("wzlayer", np.float32),
                wparent=st("wparent", np.int64), wfocusrank=st("wfocusrank", np.int64),
                wuiedges=np.stack([np.packbits(np.asarray(t["wuiedges"], bool).reshape(-1)) for t in tabs]))


def unpack_edges(packed: torch.Tensor) -> torch.Tensor:
    """[..., ceil(NW * NW * R / 8)] uint8 (`stack_tables`) -> [..., NW, NW, R] bool."""
    bits = (packed[..., None].to(torch.uint8) >> torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)) & 1
    n = NW * NW * len(UI_REL_VOCAB)
    return bits.reshape(*packed.shape[:-1], -1)[..., :n].reshape(*packed.shape[:-1], *_EDGE_SHAPE).bool()


def pointer_geometry(task: str = "cw/calc_sum") -> PointerGeometry:
    """Screen size, pixel pitch and control rate of the pointer env, read from its spec (the one source of truth)."""
    from rrp.envs.base import make_env
    env = make_env("computerworld", task=task, body="cw_pointer", seed=0)
    try:
        return PointerGeometry.from_spec(env.spec)
    finally:
        env.close()


class Demos:
    """All collected ticks on one device. sample t -> public features at t, the 7-tick demo chunk t..t+6 (masked at the
    episode end) and probe labels at the knots' late ticks (j = 0, 2, 4, 6)."""
    H = 7
    LATE = (0, 2, 4, 6)

    def __init__(self, files: list[str], device, geom: PointerGeometry, val_frac: float = 0.05, seed: int = 0):
        self.geom = geom
        self.half = torch.tensor(geom.half_m, device=device)
        parts = []
        for f in files:
            p = dict(np.load(f, allow_pickle=False))
            if "geom" in p and PointerGeometry.from_dict(json.loads(str(p.pop("geom")))) != geom:
                raise ValueError(f"{f}: collected under a different pointer geometry than the env spec's {geom}")
            parts.append(p)
        cat = {}
        tab_off, tick_off, ep_off = 0, 0, 0
        for p in parts:
            p["tab"] = p["tab"] + tab_off
            p["ep"] = p["ep"] + ep_off
            p["ep_start"] = p["ep_start"] + tick_off
            p["ep_end"] = p["ep_end"] + tick_off
            if "wpos3d" not in p:               # a pack from before D-144's geometry: zeros, flagged invalid below
                n_tab = len(p["wch"])
                p["wpos3d"] = np.zeros((n_tab, *p["wch"].shape[1:2], 3), np.float32)
                p["wcamuvd"] = np.zeros((n_tab, *p["wch"].shape[1:2], 3), np.float32)
                p["wgeo"] = np.zeros(n_tab, bool)
            else:
                p["wgeo"] = np.ones(len(p["wch"]), bool)
            if "wuiedges" not in p:             # a pack from before the public UI fields: defaults, flagged below
                n_tab, nw = len(p["wch"]), p["wch"].shape[1]
                p["wzlayer"] = np.zeros((n_tab, nw), np.float32)
                p["wparent"] = np.full((n_tab, nw), -1, np.int64)
                p["wfocusrank"] = np.full((n_tab, nw), -1, np.int64)
                p["wuiedges"] = np.zeros((n_tab, -(-NW * NW * len(UI_REL_VOCAB) // 8)), np.uint8)
                p["wui"] = np.zeros(n_tab, bool)
            else:
                p["wui"] = np.ones(len(p["wch"]), bool)
            if "ep_drag" not in p:              # a pack from before the teacher drag label: flagged below
                n_ep = len(p["ep_seed"])
                p["ep_drag"] = np.full((n_ep, 2), -1, np.int16)
                p["ep_drags"] = np.zeros(n_ep, bool)
                p["ep_drag_ok"] = np.zeros(n_ep, bool)
            else:
                p["ep_drag_ok"] = np.ones(len(p["ep_seed"]), bool)
            p["ep_task"] = np.full(len(p["ep_seed"]), TASKS.index(str(p["task"])), np.int8)
            tab_off += len(p["wch"])
            tick_off += len(p["tab"])
            ep_off += len(p["ep_seed"])
            for k, v in p.items():
                if k == "task":
                    continue
                cat.setdefault(k, []).append(v)
        d = {k: np.concatenate(v) for k, v in cat.items()}
        self.goals = list(d.pop("ep_goal"))
        self.ep_seed, self.ep_task = d["ep_seed"], d["ep_task"]
        N = len(d["tab"])
        # `ntyped` (D-146 C2): printable keys the policy had executed before tick t of its episode (`EventHistory.n_typed`)
        from rrp.envs.computerworld import KEY_NAMES
        pr = (d["cmd_key"].astype(np.int64) > len(KEY_NAMES)).astype(np.int64)      # class 1 + idx; idx >= names
        before = np.cumsum(pr) - pr
        ntyped = (before - before[d["ep_start"][d["ep"]]]).astype(np.float32)
        # chunk indices (t + j within the episode, else -1)
        ep_end = d["ep_end"][d["ep"]]
        j = np.arange(self.H)[None]
        idx = np.arange(N)[:, None] + j
        idx = np.where(idx < ep_end[:, None], idx, -1)
        rng = np.random.default_rng(seed)
        n_ep = len(d["ep_seed"])
        val_ep = np.zeros(n_ep, bool)
        val_ep[rng.permutation(n_ep)[:max(1, int(val_frac * n_ep))]] = True
        self.val_mask = val_ep[d["ep"]]
        T = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x), device=device, dtype=dt)
        self.t = dict(wch=T(d["wch"].astype(np.int64)), wrole=T(d["wrole"].astype(np.int64)),
                      wbound=T(d["wbound"].astype(np.int64)), wf=T(d["wf"].astype(np.float32)), wmask=T(d["wmask"]),
                      wpos3d=T(d["wpos3d"].astype(np.float32)), wcamuvd=T(d["wcamuvd"].astype(np.float32)),
                      wgeo=T(d["wgeo"]), wui=T(d["wui"]), wzlayer=T(d["wzlayer"].astype(np.float32)),
                      wparent=T(d["wparent"].astype(np.int64)), wfocusrank=T(d["wfocusrank"].astype(np.int64)),
                      wuiedges=T(d["wuiedges"].astype(np.uint8)), tab=T(d["tab"].astype(np.int64)), ep=T(d["ep"].astype(np.int64)), ptr=T(d["ptr"]), btn=T(d["btn"]),
                      hist=T(d["hist"]), cmd_xy=T(d["cmd_xy"]), cmd_btn=T(d["cmd_btn"]),
                      cmd_key=T(d["cmd_key"].astype(np.int64)), slot=T(d["slot"].astype(np.int64)),
                      txy=T(np.nan_to_num(d["txy"], nan=0.0)), txy_ok=T(~np.isnan(d["txy"][:, 0])),
                      phase=T(d["phase"].astype(np.int64)), instr=T(d["ep_instr"].astype(np.int64)),
                      ep_drag=T(d["ep_drag"].astype(np.int64)), ep_drags=T(d["ep_drags"]),
                      tick=T((np.arange(N) - d["ep_start"][d["ep"]]).astype(np.float32)), ntyped=T(ntyped), chunk=T(idx.astype(np.int64)))
        self.N, self.device = N, device
        self.has_drag = bool(d["ep_drag_ok"].all())    # every pack carries the teacher's drag target (`drag_labels`)
        self.has_ui = bool(self.t["wui"].all())      # every pack carries the public UI fields (`preset:ui` needs them)
        self.train_idx = T(np.nonzero(~self.val_mask)[0].astype(np.int64))
        tick_task = self.ep_task[d["ep"]][~self.val_mask]
        cnt = np.bincount(tick_task, minlength=len(TASKS)).astype(np.float64)
        self.train_w = T((1.0 / cnt[tick_task]).astype(np.float32))       # every task equally likely per sample
        self.val_idx = T(rng.permutation(np.nonzero(self.val_mask)[0]).astype(np.int64))   # every task in any prefix

    def sample(self, n: int):
        """Task-balanced training sample indices."""
        return self.train_idx[torch.multinomial(self.train_w, n, replacement=True)]

    def drag_labels(self, ix):
        """`ui.drag_to`'s gt label from the TEACHER's drag (stored per episode at collect: handle slot, drop-target slot;
        `relgen.ui.teacher_drag_target`) for sample indices ix -> (y [B, NW, NW, 1] float, valid [B, NW, NW] bool), the
        layout of a relgen `Label`. Every pair of present widgets is a candidate and only (handle, target) is true; an
        episode whose teacher never drags has no true pair, and one whose handle or target is not in the tick's table carries
        no label (nothing valid)."""
        if not self.has_drag:
            raise ValueError("a demo pack lacks the teacher's drag target (`ep_drag`): recollect (`rrp train pointer collect`)")
        t = self.t
        ep = t["ep"][ix]
        mask = t["wmask"][t["tab"][ix]]
        valid = mask[:, :, None] & mask[:, None, :]
        src, dst, drags = t["ep_drag"][ep, 0], t["ep_drag"][ep, 1], t["ep_drags"][ep]
        r = torch.arange(len(ix), device=mask.device)
        sc, dc = src.clamp(min=0), dst.clamp(min=0)
        present = drags & (src >= 0) & (dst >= 0) & mask[r, sc] & mask[r, dc]
        y = torch.zeros(len(ix), NW, NW, device=mask.device)
        y[r, sc, dc] = present.float()
        valid = valid & ~(drags & ~present)[:, None, None]
        return y[..., None], valid

    def batch(self, ix):
        """-> (public batch b, demo chunk a, probe labels lab) for sample indices ix (a 1-D long tensor)."""
        t = self.t
        tab = t["tab"][ix]
        b = dict(wch=t["wch"][tab], wrole=t["wrole"][tab], wbound=t["wbound"][tab], wf=t["wf"][tab], wmask=t["wmask"][tab],
                 wpos3d=t["wpos3d"][tab], wcamuvd=t["wcamuvd"][tab], wgeo_ok=t["wmask"][tab] & t["wgeo"][tab][:, None],
                 wzlayer=t["wzlayer"][tab], wparent=t["wparent"][tab], wfocusrank=t["wfocusrank"][tab],
                 wuiedges=unpack_edges(t["wuiedges"][tab]),
                 instr=t["instr"][t["ep"][ix]], ptr=t["ptr"][ix], btn=t["btn"][ix], tick=t["tick"][ix],
                 hist=t["hist"][ix], ntyped=t["ntyped"][ix])
        ci = t["chunk"][ix]                                        # [B,H]
        valid = ci >= 0
        cc = ci.clamp(min=0)
        # normalized screen units -> pointer steps (MAX_STEP_PX): the realizer's output unit
        dxy = (t["cmd_xy"][cc] - t["ptr"][cc]) * self.half / self.geom.step_m
        a = dict(dxy=dxy * valid[..., None], xy=t["cmd_xy"][cc] * valid[..., None], btn=t["cmd_btn"][cc] * valid,
                 key=t["cmd_key"][cc] * valid, valid=valid, ptr=t["ptr"][cc], pbtn=t["btn"][cc])
        late = ci[:, list(self.LATE)]
        lv = late >= 0
        lc = late.clamp(min=0)
        lab = dict(slot=torch.where(lv, t["slot"][lc], -1),
                   rel=(t["txy"][lc] - t["ptr"][ix][:, None]), rel_ok=lv & t["txy_ok"][lc],
                   phase=torch.where(lv, t["phase"][lc], -1))
        return b, a, lab
