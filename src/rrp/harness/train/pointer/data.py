"""Demo packs on one device: the tick tables, 7-tick demo chunks and probe labels every pointer trainer samples."""
from __future__ import annotations

import json

import numpy as np
import torch

from rrp.harness.train.pointer.split import TASKS
from rrp.policies.pointer.spec import PointerGeometry

# per-table widget arrays a pack stores; `wpos3d` / `wcamuvd` (D-144 R20 follow-up geometry) are the ones the net's
# `_relctx` reads besides the five descriptor arrays, and packs collected before them lack them (`Demos`)
TABLE_KEYS = ("wch", "wrole", "wbound", "wf", "wmask", "wpos3d", "wcamuvd")


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
                      wgeo=T(d["wgeo"]), tab=T(d["tab"].astype(np.int64)), ep=T(d["ep"].astype(np.int64)), ptr=T(d["ptr"]), btn=T(d["btn"]),
                      hist=T(d["hist"]), cmd_xy=T(d["cmd_xy"]), cmd_btn=T(d["cmd_btn"]),
                      cmd_key=T(d["cmd_key"].astype(np.int64)), slot=T(d["slot"].astype(np.int64)),
                      txy=T(np.nan_to_num(d["txy"], nan=0.0)), txy_ok=T(~np.isnan(d["txy"][:, 0])),
                      phase=T(d["phase"].astype(np.int64)), instr=T(d["ep_instr"].astype(np.int64)),
                      tick=T((np.arange(N) - d["ep_start"][d["ep"]]).astype(np.float32)), chunk=T(idx.astype(np.int64)))
        self.N, self.device = N, device
        self.train_idx = T(np.nonzero(~self.val_mask)[0].astype(np.int64))
        tick_task = self.ep_task[d["ep"]][~self.val_mask]
        cnt = np.bincount(tick_task, minlength=len(TASKS)).astype(np.float64)
        self.train_w = T((1.0 / cnt[tick_task]).astype(np.float32))       # every task equally likely per sample
        self.val_idx = T(rng.permutation(np.nonzero(self.val_mask)[0]).astype(np.int64))   # every task in any prefix

    def sample(self, n: int):
        """Task-balanced training sample indices."""
        return self.train_idx[torch.multinomial(self.train_w, n, replacement=True)]

    def batch(self, ix):
        """-> (public batch b, demo chunk a, probe labels lab) for sample indices ix (a 1-D long tensor)."""
        t = self.t
        tab = t["tab"][ix]
        b = dict(wch=t["wch"][tab], wrole=t["wrole"][tab], wbound=t["wbound"][tab], wf=t["wf"][tab], wmask=t["wmask"][tab],
                 wpos3d=t["wpos3d"][tab], wcamuvd=t["wcamuvd"][tab], wgeo_ok=t["wmask"][tab] & t["wgeo"][tab][:, None],
                 instr=t["instr"][t["ep"][ix]], ptr=t["ptr"][ix], btn=t["btn"][ix], tick=t["tick"][ix],
                 hist=t["hist"][ix])
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
