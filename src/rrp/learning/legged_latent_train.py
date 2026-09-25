"""Training for the legged latent packet (stage A representation E+R+P; stage B system-i flow; post-hoc probes).

Stage A:  z ~ E(public ctx_t, morphology, demonstrated native targets a[t:t+40])
          L_real = || R(z, phase=j*0.02, local state_{t+j}) - a*_{t+j} ||^2  on policy joints, j ~ U{0..27}
          L_sem  = probe_loss(P(z), labels_t)   (semantic_weight; 0 => latent_nosem, capacity matched)
          L      = L_real + w_sem L_sem + beta KL
Stage B:  LeggedFlow generates z from PUBLIC context only toward the frozen E mean (standardized target);
          packet-semantic loss through the frozen P on z_hat_clean for tau >= tau_min.
Episodes are split per body into train / held-out (every 20th episode held out); all evaluations here use the
held-out episodes (same bodies, same teacher distribution, unseen seeds).

usage: python -m rrp.learning.legged_latent_train {rep,flow,probe} --config C.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from rrp.control.legged_latent import MAX_N, MAX_M, KNOT_TIMES, TICK_DT
from rrp.model.legged_latent import (LeggedEncoder, LeggedRealizer, LeggedProbe, LeggedFlow, probe_loss,
                                     probe_metrics)

H = 40                    # demonstrated ticks seen by E (0.8 s)
MAX_J = 27                # realizer phases 0..0.54 s
KNOT_TICKS = [int(round(k / TICK_DT)) for k in KNOT_TIMES]


def _dev():
    if torch.cuda.is_available():
        try:
            from rrp.ops.gpu import apply_cap
            apply_cap()
        except Exception:
            pass
        return torch.device("cuda")
    return torch.device("cpu")


class LeggedData:
    def __init__(self, root: Path, bodies: list[str], dev, holdout_every: int = 20, max_eps: int | None = None):
        self.dev = dev
        cols = {k: [] for k in ("q", "qd", "a", "amask", "imu", "touch", "osc", "ctx", "ev", "pose", "contact",
                                "body", "ep_end", "fell", "wpa", "wpb", "held_out")}
        self.static = dict(node_static=[], node_asm=[], asm_static=[], node_mask=[], asm_mask=[], asm_is_leg=[],
                           body_asm=[])
        self.body_names, self.ep_meta = [], []
        off = 0
        for bi, body in enumerate(bodies):
            shards = sorted((root / body).glob("s*.npz"))
            if not shards:
                raise FileNotFoundError(root / body)
            first = True
            for sh in shards:
                d = np.load(sh)
                meta = json.loads(sh.with_suffix(".json").read_text())
                if first:
                    self._add_static(d, meta)
                    self.body_names.append(body)
                    first = False
                N, nf, npol = d["node_static"].shape[0], int(d["nf"]), int(d["n_policy"])
                ep = d["ep"]
                for e in np.unique(ep):
                    if max_eps and sum(1 for m in self.ep_meta if m["body"] == body) >= max_eps:
                        break
                    sl = np.nonzero(ep == e)[0]
                    em = meta["episodes"][int(e)]
                    T = len(sl)
                    gid = len(self.ep_meta)
                    ho = (em["seed"] % holdout_every) == (holdout_every - 1)
                    self.ep_meta.append(dict(body=body, seed=em["seed"], status=em["status"], held_out=bool(ho),
                                             start=off, T=T))
                    pad = lambda x, w: np.pad(x, ((0, 0), (0, w - x.shape[1])))
                    cols["q"].append(pad(d["q"][sl], MAX_N)); cols["qd"].append(pad(d["qd"][sl], MAX_N))
                    cols["a"].append(pad(d["a"][sl], MAX_N))
                    am = np.zeros((T, MAX_N), bool); am[:, :npol] = True
                    cols["amask"].append(am)
                    cols["imu"].append(d["imu"][sl]); cols["osc"].append(d["osc"][sl].astype(np.float32))
                    cols["touch"].append(pad(d["touch"][sl].astype(np.float32), MAX_M))
                    cols["contact"].append(pad(d["contact"][sl].astype(np.float32), MAX_M))
                    cols["ctx"].append(d["ctx"][sl]); cols["ev"].append(d["ev"][sl]); cols["pose"].append(d["pose"][sl])
                    cols["body"].append(np.full(T, bi)); cols["ep_end"].append(np.full(T, off + T - 1))
                    cols["fell"].append(np.full(T, em["status"] == "fell"))
                    cols["wpa"].append(np.tile(np.array(em["waypoints"]["a"], np.float32), (T, 1)))
                    cols["wpb"].append(np.tile(np.array(em["waypoints"]["b"], np.float32), (T, 1)))
                    cols["held_out"].append(np.full(T, ho))
                    off += T
        t = lambda x: torch.from_numpy(np.concatenate(x)).to(dev)
        self.A = {k: t(v) for k, v in cols.items()}
        self.A["tick"] = torch.arange(off, device=dev)
        self.S = {k: torch.from_numpy(np.stack(v)).to(dev) for k, v in self.static.items()}
        self.n = off
        ho = self.A["held_out"].cpu().numpy()
        # rows that have a full horizon are not required: horizon indices clamp to the episode end
        self.train_idx = np.nonzero(~ho)[0]
        self.test_idx = np.nonzero(ho)[0]

    def _add_static(self, d, meta):
        N, M = d["node_static"].shape[0], d["asm_static"].shape[0]
        ns = np.zeros((MAX_N, d["node_static"].shape[1]), np.float32); ns[:N] = d["node_static"]
        na = np.zeros(MAX_N, np.int64); na[:N] = d["node_asm"]
        asm = np.zeros((MAX_M, d["asm_static"].shape[1]), np.float32); asm[:M] = d["asm_static"]
        nm = np.zeros(MAX_N, bool); nm[:N] = True
        amk = np.zeros(MAX_M, bool); amk[:M] = True
        leg = np.zeros(MAX_M, bool); leg[:int(d["nf"])] = True
        for k, v in (("node_static", ns), ("node_asm", na), ("asm_static", asm), ("node_mask", nm), ("asm_mask", amk),
                     ("asm_is_leg", leg), ("body_asm", np.int64(int(d["nf"])))):
            self.static[k].append(v)

    # ------------------------------------------------------------------ batches
    def ctx_batch(self, i):
        """PUBLIC inputs at row i (system i and E context)."""
        A, S = self.A, self.S
        bi = A["body"][i]
        b = {k: v[bi] for k, v in S.items()}
        b.update(q=A["q"][i], qd=A["qd"][i], imu=A["imu"][i], ctx=A["ctx"][i], asm_touch=A["touch"][i], osc=A["osc"][i])
        return b

    def beh(self, i):
        idx = torch.minimum(i[:, None] + torch.arange(H, device=self.dev)[None], self.A["ep_end"][i][:, None])
        return self.A["a"][idx].transpose(1, 2)                             # [B,N,H]

    def labels(self, i):
        A = self.A
        end = A["ep_end"][i]
        kt = torch.minimum(i[:, None] + torch.tensor(KNOT_TICKS, device=self.dev)[None], end[:, None])   # [B,K]
        contact_k = A["contact"][kt]                                        # [B,K,M]
        pose = A["pose"][i]
        c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        ev = A["ev"][i]
        wp = torch.where((ev == 0)[:, None], A["wpa"][i], A["wpb"][i])
        dx, dy = wp[:, 0] - pose[:, 0], wp[:, 1] - pose[:, 1]
        goal = torch.stack([c * dx + s * dy, -s * dx + c * dy], -1) / 2.0
        p2 = A["pose"][torch.minimum(i + H, end)]
        ex, ey = p2[:, 0] - pose[:, 0], p2[:, 1] - pose[:, 1]
        dyaw = torch.remainder(p2[:, 2] - pose[:, 2] + math.pi, 2 * math.pi) - math.pi
        disp = torch.stack([(c * ex + s * ey) / 0.5, (-s * ex + c * ey) / 0.5, dyaw], -1)
        fall = A["fell"][i] & ((end - i) <= H)
        return dict(contact_k=contact_k, goal=goal, goal_valid=ev < 3, disp=disp, subtask=ev.clamp(max=3), fall=fall)

    def realizer_batch(self, i, j):
        tj = torch.minimum(i + j, self.A["ep_end"][i])
        jj = tj - i
        b = self.ctx_batch(tj)
        return b, jj.float() * TICK_DT, self.A["a"][tj], self.A["amask"][tj]

    def sample(self, B, rng: np.random.Generator, test=False):
        pool = self.test_idx if test else self.train_idx
        return torch.from_numpy(rng.choice(pool, B)).to(self.dev)


def rep_step(E, R, P, data, i, j, w_sem, beta, train=True):
    b = data.ctx_batch(i)
    mu, lv = E(b, data.beh(i))
    z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if train else mu
    br, ph, a1, am = data.realizer_batch(i, j)
    pred = R(z, br, ph)
    m = am.float()
    l_real = (((pred - a1) ** 2) * m).sum() / m.sum()
    mm = b["asm_mask"][:, None, :, None].float().expand_as(mu)
    kl = (0.5 * (mu ** 2 + lv.exp() - 1 - lv) * mm).sum() / mm.sum()
    lab = data.labels(i)
    out = P(z, b["asm_mask"], b["body_asm"])
    l_sem, logs = probe_loss(out, lab, b)
    loss = l_real + w_sem * l_sem + beta * kl
    logs.update(real=float(l_real.detach()), kl=float(kl.detach()), sem=float(l_sem.detach()))
    return loss, logs, (z, out, lab, b, pred, a1, m)


def _agg(dst, res):
    for k, (x, n) in res.items():
        s_, n_ = dst.get(k, (0, 0)); dst[k] = (s_ + x, n_ + n)


def _fin(d):
    return {k: (x / n if n else None) for k, (x, n) in d.items()}


@torch.no_grad()
def eval_rep(E, R, P, data, n_batches=40, seed=99):
    for m in (E, R, P):
        m.eval()
    rng = np.random.default_rng(seed)
    agg, sh, real, zero = {}, {}, [], []
    for _ in range(n_batches):
        i = data.sample(256, rng, test=True)
        j = torch.from_numpy(rng.integers(0, MAX_J + 1, 256)).to(data.dev)
        _, _, (z, out, lab, b, pred, a1, m) = rep_step(E, R, P, data, i, j, 1.0, 0.0, train=False)
        real.append(float((((pred - a1) ** 2) * m).sum() / m.sum()))
        zero.append(float(((a1 ** 2) * m).sum() / m.sum()))
        _agg(agg, probe_metrics(out, lab, b))
        perm = torch.randperm(len(i), device=data.dev)
        _agg(sh, probe_metrics(P(z[perm], b["asm_mask"], b["body_asm"]), lab, b))
    for m in (E, R, P):
        m.train()
    return dict(split="held_out_episodes", realize_mse=float(np.mean(real)), zero_action_mse=float(np.mean(zero)),
                probes=_fin(agg), probes_shuffled_z=_fin(sh))


def _save(path, **kw):
    torch.save(kw, str(path) + ".tmp")
    Path(str(path) + ".tmp").rename(path)


def train_rep(cfg, out: Path):
    dev = _dev()
    torch.manual_seed(cfg.get("seed", 0))
    rng = np.random.default_rng(cfg.get("seed", 0))
    data = LeggedData(Path(cfg["data"]), cfg["bodies"], dev)
    lc = cfg["latent"]
    E = LeggedEncoder(dz=lc["dz"], D=lc["width"], H=H).to(dev)
    R = LeggedRealizer(dz=lc["dz"], D=lc["width"]).to(dev)
    P = LeggedProbe(dz=lc["dz"]).to(dev)
    params = [p for m in (E, R, P) for p in m.parameters()]
    steps, lr = cfg["steps"], cfg.get("lr", 3e-4)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "train_log.jsonl", "w")
    t0 = time.time()
    B = cfg.get("batch_size", 256)
    for step in range(1, steps + 1):
        i = data.sample(B, rng)
        j = torch.from_numpy(rng.integers(0, MAX_J + 1, B)).to(dev)
        loss, logs, _ = rep_step(E, R, P, data, i, j, lc["semantic_weight"], lc.get("beta_kl", 1e-3))
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sch.step()
        if step % 200 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=float(loss.detach()), gn=float(gn), **logs)) + "\n")
            log.flush()
    res = dict(steps=steps, wall_s=time.time() - t0, bodies=cfg["bodies"], n_rows=data.n,
               n_train_rows=len(data.train_idx), n_heldout_rows=len(data.test_idx),
               latent_space_version=f"legged-ls-{cfg['name']}", eval=eval_rep(E, R, P, data))
    _save(out / "representation.pt", E=E.state_dict(), R=R.state_dict(), P=P.state_dict(), cfg=cfg, result=res)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def load_rep(path, dev):
    st = torch.load(str(path), map_location=dev, weights_only=False)
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


def fit_probe(cfg, out: Path):
    """Post-hoc MEASUREMENT probe on detached frozen z (identical for sem / nosem); metadata-only control."""
    dev = _dev()
    rcfg, E, R, _, rres = load_rep(Path(cfg["representation"]), dev)
    data = LeggedData(Path(rcfg["data"]), rcfg["bodies"], dev)
    res = {}
    for mode in ("z", "metadata_only"):
        P = LeggedProbe(dz=rcfg["latent"]["dz"], metadata_only=(mode == "metadata_only"), seed=5).to(dev)
        opt = torch.optim.AdamW(P.parameters(), lr=3e-4, weight_decay=1e-4)
        rng = np.random.default_rng(5)
        for step in range(cfg.get("steps", 6000)):
            i = data.sample(256, rng)
            with torch.no_grad():
                b = data.ctx_batch(i)
                mu, _ = E(b, data.beh(i))
            loss, _ = probe_loss(P(mu, b["asm_mask"], b["body_asm"]), data.labels(i), b)
            opt.zero_grad(); loss.backward(); opt.step()
        P.eval()
        agg, sh = {}, {}
        rng2 = np.random.default_rng(1005)
        with torch.no_grad():
            for _ in range(40):
                i = data.sample(256, rng2, test=True)
                b = data.ctx_batch(i)
                mu, _ = E(b, data.beh(i))
                lab = data.labels(i)
                _agg(agg, probe_metrics(P(mu, b["asm_mask"], b["body_asm"]), lab, b))
                _agg(sh, probe_metrics(P(mu[torch.randperm(len(i), device=dev)], b["asm_mask"], b["body_asm"]), lab, b))
        res[mode] = dict(heldout=_fin(agg), heldout_shuffled_z=_fin(sh))
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_posthoc.json").write_text(json.dumps(dict(representation=cfg["representation"], **res), indent=1))
    return res


def train_flow(cfg, out: Path):
    dev = _dev()
    torch.manual_seed(cfg.get("seed", 0))
    rng = np.random.default_rng(cfg.get("seed", 0))
    rcfg, E, R, P, rres = load_rep(Path(cfg["representation"]), dev)
    data = LeggedData(Path(rcfg["data"]), rcfg["bodies"], dev)
    dz = rcfg["latent"]["dz"]
    F_ = LeggedFlow(dz=dz, D=cfg.get("width", 256), layers=cfg.get("layers", 4)).to(dev)
    # target standardization over valid (knot, assembly) entries
    with torch.no_grad():
        zs = []
        for _ in range(20):
            i = data.sample(256, rng)
            b = data.ctx_batch(i)
            mu, _ = E(b, data.beh(i))
            zs.append(mu[b["asm_mask"][:, None, :].expand(-1, mu.shape[1], -1)])
        z = torch.cat(zs)
        F_.z_mean.copy_(z.mean(0)); F_.z_std.copy_(z.std(0).clamp(min=1e-3))
    steps, lr = cfg["steps"], cfg.get("lr", 3e-4)
    opt = torch.optim.AdamW(F_.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    w = cfg.get("packet_semantic_weight", 0.0)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "train_log.jsonl", "w")
    t0 = time.time()
    B = cfg.get("batch_size", 256)
    for step in range(1, steps + 1):
        i = data.sample(B, rng)
        b = data.ctx_batch(i)
        with torch.no_grad():
            zt, _ = E(b, data.beh(i))
        lab = data.labels(i)
        fn = (lambda zc: probe_loss(P(zc, b["asm_mask"], b["body_asm"]), lab, b)) if w > 0 else None
        loss, logs = F_.loss(b, zt, fn, w, cfg.get("packet_tau_min", 0.6))
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(F_.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 200 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=float(loss.detach()), gn=float(gn), **logs)) + "\n")
            log.flush()
    res = dict(steps=steps, wall_s=time.time() - t0, representation=cfg["representation"],
               latent_space_version=rres["latent_space_version"], eval=eval_flow(F_, E, R, P, data))
    _save(out / "policy.pt", flow=F_.state_dict(), cfg=cfg, result=res)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


@torch.no_grad()
def eval_flow(F_, E, R, P, data, n_batches=30, seed=7, nfe=8):
    """Held-out episodes: packet probes on oracle E targets vs FREE samples (public context only) vs shuffled;
    realization error of R driven by free samples vs by oracle targets (open-loop, recorded states)."""
    F_.eval()
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=data.dev).manual_seed(seed)
    aggs = {k: {} for k in ("oracle", "free", "free_shuffled")}
    real = {"oracle": [], "free": [], "zero": []}
    for _ in range(n_batches):
        i = data.sample(256, rng, test=True)
        b = data.ctx_batch(i)
        zt, _ = E(b, data.beh(i))
        zf = F_.sample(b, nfe=nfe, generator=g)
        lab = data.labels(i)
        for name, z in (("oracle", zt), ("free", zf), ("free_shuffled", zf[torch.randperm(len(i), device=data.dev)])):
            _agg(aggs[name], probe_metrics(P(z, b["asm_mask"], b["body_asm"]), lab, b))
        j = torch.from_numpy(rng.integers(0, MAX_J + 1, len(i))).to(data.dev)
        br, ph, a1, am = data.realizer_batch(i, j)
        m = am.float()
        for name, z in (("oracle", zt), ("free", zf)):
            real[name].append(float((((R(z, br, ph) - a1) ** 2) * m).sum() / m.sum()))
        real["zero"].append(float(((a1 ** 2) * m).sum() / m.sum()))
    F_.train()
    return dict(split="held_out_episodes", realize_mse={k: float(np.mean(v)) for k, v in real.items()},
                **{k: _fin(v) for k, v in aggs.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rep", "flow", "probe"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override (smoke runs)")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    if a.steps:
        cfg["steps"] = a.steps
    fn = dict(rep=train_rep, flow=train_flow, probe=fit_probe)[a.stage]
    res = fn(cfg, Path(a.out))
    print(json.dumps(res, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
