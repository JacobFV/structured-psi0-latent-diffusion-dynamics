"""t1 humanoid diagnosis: why does the SEMANTIC packet fall on the deployable route (R2) while the capacity-matched
NO-semantic packet walks (D-084), and why does nosem stall on the stateless oracle route R1 (D-079/D-080)?

All measurements are OFFLINE on recorded, MATCHED states (the same states are scored by both variants):
  bc          states visited by the learned BC positive control
  r2_<v>      states visited by the deployable route of variant v (flow -> original system 0)
  r1_<v>      states visited by the stateless oracle route E(BC chunk) -> system 0 (training seed 0 only)
Buffers come from `legged_dagger collect --diag` (scripts/t1_diag_collect.sh): per tick the stateless BC chunk
(label a = chunk[:, 0]); at packet ticks also the received packet and the privileged shadow-teacher chunk
(DIAGNOSTIC label only; never an input to a deployable route).

Per (variant model, state set), at packet ticks i and ticks j = 0..19 after them:
  err_ratio[z]   system-0 MSE vs the BC label / hold-still MSE, for z in
                 oracle  = E(ctx, BC chunk) mean          (R1 packet)
                 teacher = E(ctx, shadow-teacher chunk)   (privileged diagnostic)
                 gen     = a fresh flow sample             (R2 packet)
                 recv    = the packet the route actually received (own route only)
                 post1   = oracle + 1 posterior sigma noise (the noise system 0 was trained with)
                 gendir_a= oracle + a (gen - oracle), a in {0.5, 2}
                 rand    = oracle + random direction with |gen - oracle|
  gap            |gen - oracle| / |oracle|; Mahalanobis gap mean(((gen - oracle)/sigma)^2) in posterior-sigma units
  gain           |R(gen) - R(oracle)| / |gen - oracle| (action change per unit packet error), and for random dirs
  per_j          err_ratio by tick-within-packet (oracle / gen)
  motion         |R(z) - hold| at j=0 (how much the decoded action moves) and the same for the BC / teacher chunks
Latent geometry on teacher held-out rows: KL per dim, active dims (KL>0.01), participation ratio of mu, |mu|, sigma.
Balance channel: ridge probe (episode-split) from z to IMU (gyro, gravity) now and +10/+20 ticks, and base speed.
Falls (own R2 route): onset tick (tilt > 0.35 rad), tick-within-packet at onset, stance foot, tilt direction,
and the received-packet error in the 1 s before onset vs non-fall episodes.

usage: python -m rrp.learning.legged_t1_diag --ts v2 --out artifacts/runs/t1_diag/diag_v2.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rrp.control.legged_latent import MAX_N
from rrp.learning.legged_latent_train import LeggedData, load_rep, _dev
from rrp.model.legged_latent import LeggedFlow

import os
BUF = Path(os.environ.get("T1DIAG_BUF", "artifacts/runs/t1_diag/buf"))


def load_buf(root: Path, dev):
    data = LeggedData(root, ["t1"], dev)
    ex = {k: [] for k in ("pk", "zpk", "tch")}
    for sh in sorted((root / "t1").glob("s*.npz")):
        d = np.load(sh)
        for k in ex:
            ex[k].append(d[k])
    ex = {k: np.concatenate(v) for k, v in ex.items()}
    assert len(ex["pk"]) == data.n
    data.pk = torch.from_numpy(ex["pk"]).to(dev)
    data.zpk = torch.from_numpy(ex["zpk"].astype(np.float32)).to(dev)
    tch = ex["tch"].astype(np.float32)
    data.tch = torch.from_numpy(tch).to(dev)
    return data


def load_models(v, ts, dev):
    rep = f"artifacts/runs/legged_rep_{v}_t1_{ts}/representation.pt"
    rcfg, E, R, P, rres = load_rep(Path(rep), dev)
    st = torch.load(f"artifacts/runs/legged_flow_{v}_t1_{ts}/policy.pt", map_location=dev, weights_only=False)
    F_ = LeggedFlow(dz=rcfg["latent"]["dz"], D=st["cfg"].get("width", 256), layers=st["cfg"].get("layers", 4)).to(dev)
    F_.load_state_dict(st["flow"]); F_.eval()
    return dict(E=E, R=R, P=P, F=F_, rcfg=rcfg)


def packet_rows(data):
    pi = torch.nonzero(data.pk).squeeze(-1)
    win = getattr(data, "window", None)                 # (lo, hi) ticks since episode start, optional
    if win is not None:
        starts = torch.tensor([em["start"] for em in data.ep_meta], device=data.dev)
        t = pi - starts[torch.bucketize(pi, starts, right=True) - 1]
        pi = pi[(t >= win[0]) & (t < win[1])]
    return pi


@torch.no_grad()
def realize_err(R, data, i, z, j):
    """sum of squared error vs BC label, sum of hold-still error, count  (at state i+j with packet z made at i)."""
    br, ph, a1, am = data.realizer_batch(i, j)
    tj = torch.minimum(i + j, data.A["ep_end"][i])
    hold = data.hold_still(tj)
    m = am.float()
    p = R(z, br, ph)
    return ((p - a1) ** 2 * m).sum(-1), ((hold - a1) ** 2 * m).sum(-1), p, hold, m


@torch.no_grad()
def score_set(M, data, own_recv: bool, gen_seed=0, chunk=512):
    E, R, F_ = M["E"], M["R"], M["F"]
    pi = packet_rows(data)
    g = torch.Generator(device=data.dev).manual_seed(gen_seed)
    keys = ("oracle", "teacher", "gen", "post1", "gendir_0.5", "gendir_2", "rand") + (("recv",) if own_recv else ())
    num = {k: 0.0 for k in keys}; den = 0.0
    perj = {k: np.zeros((20, 2)) for k in ("oracle", "gen", "teacher") + (("recv",) if own_recv else ())}
    gap_n = gap_d = mah = mah_n = gain_g = gain_r = gcount = 0.0
    mah_te = te_gap_n = 0.0
    motion = {k: 0.0 for k in ("R_oracle", "R_teacher", "R_gen", "bc_chunk", "teacher_chunk")}; mot_n = 0.0
    for s in range(0, len(pi), chunk):
        i = pi[s:s + chunk]
        B = len(i)
        b = data.ctx_batch(i)
        mu, lv = E(b, data.beh(i))
        mt, _ = E(b, data.tch[i][:, :, :40])
        zg = F_.sample(b, nfe=8, generator=g)
        am = b["asm_mask"][:, None, :, None].float().expand_as(mu)
        sig = (0.5 * lv).exp()
        d = (zg - mu) * am
        dn = d.flatten(1).norm(dim=1)
        rnd = torch.randn(mu.shape, generator=g, device=mu.device) * am
        rnd = rnd / rnd.flatten(1).norm(dim=1)[:, None, None, None] * dn[:, None, None, None]
        zs = dict(oracle=mu, teacher=mt, gen=zg, post1=mu + sig * torch.randn(mu.shape, generator=g, device=mu.device) * am,
                  **{"gendir_0.5": mu + 0.5 * d, "gendir_2": mu + 2 * d}, rand=mu + rnd)
        if own_recv:
            zs["recv"] = data.zpk[i]
        gap_n += float((d ** 2).sum()); gap_d += float((mu ** 2 * am).sum())
        mah += float(((d / sig) ** 2 * am).sum()); mah_n += float(am.sum())
        mah_te += float((((mt - mu) / sig) ** 2 * am).sum()); te_gap_n += float(((mt - mu) ** 2 * am).sum())
        for jj in range(0, 20):
            j = torch.full((B,), jj, device=data.dev)
            outs = {}
            for k, z in zs.items():
                e, h, p, hold, m = realize_err(R, data, i, z, j)
                outs[k] = p
                num[k] += float(e.sum())
                if k in perj:
                    perj[k][jj, 0] += float(e.sum()); perj[k][jj, 1] += float(h.sum())
            den += float(h.sum())
            dg = ((outs["gen"] - outs["oracle"]) ** 2 * m).sum(-1).sqrt()
            dr = ((outs["rand"] - outs["oracle"]) ** 2 * m).sum(-1).sqrt()
            ok = dn > 1e-6
            gain_g += float((dg[ok] / dn[ok]).sum()); gain_r += float((dr[ok] / dn[ok]).sum()); gcount += float(ok.sum())
            if jj == 0:
                mm = m
                n_j = mm.sum(-1).clamp(min=1)
                for k, key in (("oracle", "R_oracle"), ("teacher", "R_teacher"), ("gen", "R_gen")):
                    motion[key] += float(((((outs[k] - hold) ** 2) * mm).sum(-1) / n_j).sqrt().sum())
                bcc = data.beh(i)[:, :, :].transpose(1, 2)            # [B,40,N]
                tcc = data.tch[i][:, :, :40].transpose(1, 2)
                h0 = hold[:, None]
                motion["bc_chunk"] += float(((((bcc - h0) ** 2) * mm[:, None]).sum(-1) / n_j[:, None]).sqrt().mean(1).sum())
                motion["teacher_chunk"] += float(((((tcc - h0) ** 2) * mm[:, None]).sum(-1) / n_j[:, None]).sqrt().mean(1).sum())
                mot_n += B
    return dict(n_packets=int(len(pi)),
                err_ratio={k: num[k] / den for k in keys},
                gap_rel=math.sqrt(gap_n / gap_d), gap_mahalanobis=mah / mah_n,
                teacher_vs_oracle_gap_rel=math.sqrt(te_gap_n / gap_d), teacher_vs_oracle_mahalanobis=mah_te / mah_n,
                gain_gen=gain_g / gcount, gain_rand=gain_r / gcount,
                per_j={k: (v[:, 0] / np.maximum(v[:, 1], 1e-9)).round(5).tolist() for k, v in perj.items()},
                motion_rms_at_j0={k: v / mot_n for k, v in motion.items()})


@torch.no_grad()
def latent_geometry(M, data, n=40, seed=1):
    E = M["E"]
    rng = np.random.default_rng(seed)
    mus, kls, sigs = [], [], []
    for _ in range(n):
        i = data.sample(256, rng, test=True)
        b = data.ctx_batch(i)
        mu, lv = E(b, data.beh(i))
        am = b["asm_mask"][0]
        mu, lv = mu[:, :, am], lv[:, :, am]                           # [B,K,Mv,dz]
        mus.append(mu.flatten(1)); kls.append((0.5 * (mu ** 2 + lv.exp() - 1 - lv)).flatten(1)); sigs.append((0.5 * lv).exp().flatten(1))
    mu, kl, sg = torch.cat(mus), torch.cat(kls), torch.cat(sigs)
    klm = kl.mean(0)
    c = torch.cov(mu.T)
    ev = torch.linalg.eigvalsh(c).clamp(min=0)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    # signal-to-noise per coordinate: std of mu across states / mean posterior sigma
    snr = (mu.std(0) / sg.mean(0))
    return dict(kl_per_entry_mean=float(klm.mean()), kl_total_per_packet=float(klm.sum()),
                active_entries_kl_gt_0p01=int((klm > 0.01).sum()), active_entries_kl_gt_0p1=int((klm > 0.1).sum()),
                n_entries=int(klm.numel()), participation_ratio=pr, mu_norm=float(mu.norm(dim=1).mean()),
                sigma_mean=float(sg.mean()), snr_entries_gt1=int((snr > 1).sum()), snr_median=float(snr.median()),
                top_eig_frac=float(ev[-1] / ev.sum()), eig_frac_top10=float(ev[-10:].sum() / ev.sum()))


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    sx = Xtr.std(0).clamp(min=1e-6)
    A = (Xtr - mx) / sx
    W = torch.linalg.solve(A.T @ A + lam * len(A) * 1e-3 * torch.eye(A.shape[1], device=A.device), A.T @ (Ytr - my))
    P = ((Xte - mx) / sx) @ W + my
    ss = ((Yte - P) ** 2).sum(0); st = ((Yte - Yte.mean(0)) ** 2).sum(0).clamp(min=1e-9)
    return (1 - ss / st)


@torch.no_grad()
def balance_probe(Ms, data):
    """Ridge probes (train on even episodes, test on odd) from z to IMU now / +10 / +20 ticks and base speed.
    Same packet rows for every model; z = oracle E(BC chunk) and a generated sample."""
    pi = packet_rows(data)
    end = data.A["ep_end"][pi]
    ep = torch.bucketize(pi, torch.unique(data.A["ep_end"]), right=False)
    tr, te = (ep % 2 == 0), (ep % 2 == 1)
    Y = {}
    for dt in (0, 10, 20):
        Y[f"imu_t+{dt}"] = data.A["imu"][torch.minimum(pi + dt, end)]
    p0, p1 = data.A["pose"][pi], data.A["pose"][torch.minimum(pi + 20, end)]
    c, s = torch.cos(p0[:, 2]), torch.sin(p0[:, 2])
    ex, ey = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
    Y["disp_0.4s"] = torch.stack([c * ex + s * ey, -s * ex + c * ey,
                                  torch.remainder(p1[:, 2] - p0[:, 2] + math.pi, 2 * math.pi) - math.pi], -1)
    names = dict(imu=["gyro_x", "gyro_y", "gyro_z", "grav_x", "grav_y", "grav_z"], disp=["fwd", "lat", "yaw"])
    out = {}
    g = torch.Generator(device=data.dev).manual_seed(0)
    for v, M in Ms.items():
        zs = {"oracle": [], "gen": []}
        for s_ in range(0, len(pi), 512):
            i = pi[s_:s_ + 512]
            b = data.ctx_batch(i)
            mu, _ = M["E"](b, data.beh(i))
            zs["oracle"].append(mu[:, :, :5].flatten(1)); zs["gen"].append(M["F"].sample(b, nfe=8, generator=g)[:, :, :5].flatten(1))
        for zk, zl in zs.items():
            X = torch.cat(zl)
            for yk, y in Y.items():
                r2 = ridge_r2(X[tr], y[tr], X[te], y[te]).cpu().numpy()
                nm = names["imu" if yk.startswith("imu") else "disp"]
                out[f"{v}/{zk}/{yk}"] = {n: round(float(x), 4) for n, x in zip(nm, r2)}
    return out


@torch.no_grad()
def fall_analysis(data, M=None, own_recv=True):
    """Per episode of the recorded route: fall onset and its context; received-packet error before onset."""
    A = data.A
    grav = A["imu"][:, 3:6]
    tilt = torch.arccos((-grav[:, 2]).clamp(-1, 1))
    res = dict(episodes=[], summary={})
    pk = data.pk.cpu().numpy()
    for em in data.ep_meta:
        s, T = em["start"], em["T"]
        tl = tilt[s:s + T].cpu().numpy()
        pkr = np.nonzero(pk[s:s + T])[0]
        e = dict(seed=em["seed"], status=em["status"], T=T)
        if em["status"] == "fell":
            above = np.nonzero(tl > 0.35)[0]
            on = int(above[0]) if len(above) else T - 1
            last_pk = pkr[pkr <= on]
            e.update(onset_tick=on, ticks_before_end=T - 1 - on,
                     tick_in_packet=int(on - last_pk[-1]) if len(last_pk) else None,
                     touch_at_onset=A["touch"][s + on][:2].cpu().numpy().astype(int).tolist(),
                     touch_20_before=A["touch"][s + max(on - 20, 0)][:2].cpu().numpy().astype(int).tolist(),
                     grav_at_onset=[round(float(x), 3) for x in grav[s + on].cpu().numpy()],
                     tilt_max=float(tl.max()), ev_at_onset=int(A["ev"][s + on]))
            gx, gy = float(grav[s + on, 0]), float(grav[s + on, 1])
            e["tilt_dir"] = ("pitch" if abs(gx) >= abs(gy) else "roll") + ("+" if (gx if abs(gx) >= abs(gy) else gy) > 0 else "-")
        res["episodes"].append(e)
    fe = [e for e in res["episodes"] if e["status"] == "fell"]
    if fe:
        tip = [e["tick_in_packet"] for e in fe if e["tick_in_packet"] is not None]
        res["summary"] = dict(n=len(res["episodes"]), fell=len(fe), onset_s_mean=float(np.mean([e["onset_tick"] for e in fe])) * 0.02,
                              tick_in_packet_hist=np.bincount(tip, minlength=20).tolist(),
                              tilt_dir={k: sum(e["tilt_dir"] == k for e in fe) for k in ("pitch+", "pitch-", "roll+", "roll-")},
                              touch_at_onset={str(k): sum(tuple(e["touch_at_onset"]) == k for e in fe) for k in ((1, 1), (1, 0), (0, 1), (0, 0))},
                              ev_at_onset=np.bincount([e["ev_at_onset"] for e in fe], minlength=4).tolist())
    else:
        res["summary"] = dict(n=len(res["episodes"]), fell=0)
    if M is not None and own_recv:
        # received-packet system-0 error, by window relative to fall onset (fell eps) vs all packets of non-fall eps
        pre, calm, early = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
        pi = packet_rows(data)
        onset = {em["start"]: e.get("onset_tick") for em, e in zip(data.ep_meta, res["episodes"])}
        starts = torch.tensor([em["start"] for em in data.ep_meta], device=data.dev)
        for s_ in range(0, len(pi), 512):
            i = pi[s_:s_ + 512]
            b = data.ctx_batch(i)
            er, hr = torch.zeros(len(i), device=data.dev), torch.zeros(len(i), device=data.dev)
            for jj in range(20):
                e, h, *_ = realize_err(M["R"], data, i, data.zpk[i], torch.full((len(i),), jj, device=data.dev))
                er += e; hr += h
            epi = torch.bucketize(i, starts, right=True) - 1
            for k in range(len(i)):
                st = int(starts[epi[k]]); on = onset[st]; t = int(i[k]) - st
                tgt = calm if on is None else (pre if on - 50 <= t <= on else (early if t < on - 50 else None))
                if tgt is not None:
                    tgt[0] += float(er[k]); tgt[1] += float(hr[k])
        res["recv_err_ratio"] = dict(pre_onset_1s=pre[0] / max(pre[1], 1e-9), fell_eps_earlier=early[0] / max(early[1], 1e-9),
                                     non_fall_eps=calm[0] / max(calm[1], 1e-9))
    return res


@torch.no_grad()
def temporal_metrics(M, data, own_recv, gen_seed=5, chunk=512):
    """Packet-boundary action jump and sample spread (ratios to the hold-still error at the same states).
    jump_recv: at packet tick i, R(previous received packet, state_i, phase 0.4 s) vs R(new packet, state_i, 0)
    jump_gen:  the same with fresh flow samples at ctx_{i-20} and ctx_i (matched states, either model)
    spread:    two independent flow samples at ctx_i, both realized at state_i, phase 0
    jump_oracle: the same boundary jump with oracle packets E(BC chunk) at i-20 and i."""
    E, R, F_ = M["E"], M["R"], M["F"]
    pi = packet_rows(data)
    starts = torch.tensor([em["start"] for em in data.ep_meta], device=data.dev)
    epi = torch.bucketize(pi, starts, right=True) - 1
    prev_ok = (pi - 20) >= starts[epi]
    pi = pi[prev_ok]
    g = torch.Generator(device=data.dev).manual_seed(gen_seed)
    acc = {k: 0.0 for k in ("jump_gen", "jump_oracle", "spread", "jump_recv", "hold", "within_gen")}
    for s_ in range(0, len(pi), chunk):
        i = pi[s_:s_ + chunk]; ip = i - 20
        b, bp = data.ctx_batch(i), data.ctx_batch(ip)
        z1, z2, zp = F_.sample(b, nfe=8, generator=g), F_.sample(b, nfe=8, generator=g), F_.sample(bp, nfe=8, generator=g)
        mo, _ = E(b, data.beh(i)); mop, _ = E(bp, data.beh(ip))
        zero = torch.zeros(len(i), device=data.dev)
        br, _, a1, am = data.realizer_batch(i, zero.long())
        m = am.float()
        ph0, ph20 = zero, zero + 0.4
        hold = data.hold_still(i)
        d = lambda x, y: float((((x - y) ** 2) * m).sum())
        acc["jump_gen"] += d(R(zp, br, ph20), R(z1, br, ph0))
        acc["jump_oracle"] += d(R(mop, br, ph20), R(mo, br, ph0))
        acc["spread"] += d(R(z1, br, ph0), R(z2, br, ph0))
        # within-packet step change for reference: R(z, s_i, 0) vs R(z, s_{i+1}, 0.02) is dominated by gait; skip
        if own_recv:
            acc["jump_recv"] += d(R(data.zpk[ip], br, ph20), R(data.zpk[i], br, ph0))
        acc["hold"] += d(hold, a1)
    h = acc.pop("hold")
    return {k: v / h for k, v in acc.items()} | dict(n=int(len(pi)))


def main_extra(ts, out):
    dev = _dev()
    torch.manual_seed(0)
    Ms = {v: load_models(v, ts, dev) for v in ("sem", "nosem")}
    sets = {"bc": BUF / "bc", "r2_sem": BUF / f"r2_sem_{ts}", "r2_nosem": BUF / f"r2_nosem_{ts}"}
    if ts == "v2":
        sets.update(r1_sem=BUF / "r1_sem_v2", r1_nosem=BUF / "r1_nosem_v2")
    res = dict(ts=ts, sets={})
    for sn, root in sets.items():
        data = load_buf(root, dev)
        res["sets"][sn] = {v: temporal_metrics(M, data, sn in (f"r2_{v}", f"r1_{v}")) for v, M in Ms.items()}
        print(sn, json.dumps(res["sets"][sn]), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


def liftoff(data):
    """Per episode: first tick with a foot off the ground (touch==0) and tilt at 1/2/3 s."""
    out = []
    for em in data.ep_meta:
        s_, T = em["start"], em["T"]
        tt = data.A["touch"][s_:s_ + T, :2].cpu().numpy()
        g = data.A["imu"][s_:s_ + T, 3:6].cpu().numpy()
        tilt = np.arccos(np.clip(-g[:, 2], -1, 1))
        off = np.nonzero(tt.min(1) < 0.5)[0]
        out.append(dict(seed=em["seed"], status=em["status"], first_liftoff_tick=int(off[0]) if len(off) else None,
                        tilt_at={f"{k}s": (round(float(tilt[50 * k]), 3) if 50 * k < T else None) for k in (1, 2, 3)}))
    return out


def main_early(ts, out, lo, hi):
    dev = _dev()
    torch.manual_seed(0)
    Ms = {v: load_models(v, ts, dev) for v in ("sem", "nosem")}
    sets = {"bc": BUF / "bc", "r2_sem": BUF / f"r2_sem_{ts}", "r2_nosem": BUF / f"r2_nosem_{ts}"}
    res = dict(ts=ts, window_ticks=[lo, hi], sets={})
    for sn, root in sets.items():
        data = load_buf(root, dev)
        res["sets"][sn] = dict(liftoff=liftoff(data))
        data.window = (lo, hi)
        for v, M in Ms.items():
            res["sets"][sn][v] = score_set(M, data, sn == f"r2_{v}")
            print(sn, v, json.dumps(res["sets"][sn][v]["err_ratio"]), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="v2")
    ap.add_argument("--teacher-geometry", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--early", type=int, nargs=2, default=None, help="packet ticks window [lo, hi) since episode start")
    ap.add_argument("--extra", action="store_true", help="temporal metrics only (boundary jump, sample spread)")
    a = ap.parse_args(argv)
    if a.extra:
        return main_extra(a.ts, a.out)
    if a.early:
        return main_early(a.ts, a.out, *a.early)
    dev = _dev()
    torch.manual_seed(0)
    Ms = {v: load_models(v, a.ts, dev) for v in ("sem", "nosem")}
    sets = {"bc": BUF / "bc", "r2_sem": BUF / f"r2_sem_{a.ts}", "r2_nosem": BUF / f"r2_nosem_{a.ts}"}
    if a.ts == "v2":
        sets.update(r1_sem=BUF / "r1_sem_v2", r1_nosem=BUF / "r1_nosem_v2")
    res = dict(ts=a.ts, label="OFFLINE diagnostic on recorded states; labels = stateless BC chunk (learned) and "
               "shadow-teacher chunk (privileged, diagnostic only)", sets={}, falls={}, balance={}, geometry={})
    for sn, root in sets.items():
        data = load_buf(root, dev)
        stat = {}
        for em in data.ep_meta:
            stat[em["status"]] = stat.get(em["status"], 0) + 1
        res["sets"][sn] = dict(episodes=stat)
        for v, M in Ms.items():
            own = sn in (f"r2_{v}", f"r1_{v}")
            res["sets"][sn][v] = score_set(M, data, own)
            print(sn, v, json.dumps(res["sets"][sn][v]["err_ratio"]), flush=True)
        res["balance"][sn] = balance_probe(Ms, data)
        if sn.startswith("r"):
            v = sn.split("_")[1]
            res["falls"][sn] = fall_analysis(data, Ms[v])
        del data
        torch.cuda.empty_cache()
    rcfg = Ms["sem"]["rcfg"]
    td = LeggedData(Path(rcfg["data"]), rcfg["bodies"], dev)
    for v, M in Ms.items():
        res["geometry"][v] = latent_geometry(M, td)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["geometry"], indent=1))



@torch.no_grad()
def readout_and_teacher_err(ts="v2", out=None):
    """(a) requested forward/yaw displacement read from the packet by the variant's probe (sem: joint probe; nosem:
    post-hoc measurement probe) for z = E(BC chunk), E(teacher chunk), flow sample, at the same packet states;
    realized displacement over 0.8 s for reference.  (b) system-0 error vs the shadow-TEACHER first action at j=0
    (the only tick where the shadow state equals the recorded state), ratio to hold-still."""
    from rrp.model.legged_latent import LeggedProbe
    dev = _dev()
    Ms = {v: load_models(v, ts, dev) for v in ("sem", "nosem")}
    pp = torch.load(f"artifacts/runs/legged_rep_nosem_t1_{ts}/probe_posthoc.pt", map_location=dev, weights_only=False) \
        if Path(f"artifacts/runs/legged_rep_nosem_t1_{ts}/probe_posthoc.pt").exists() else None
    if pp is not None:
        Pn = LeggedProbe(dz=32).to(dev); Pn.load_state_dict(pp["state"]); Pn.eval(); Ms["nosem"]["P"] = Pn
    sets = {"bc": BUF / "bc", "r2_sem": BUF / f"r2_sem_{ts}", "r2_nosem": BUF / f"r2_nosem_{ts}"}
    if ts == "v2":
        sets.update(r1_sem=BUF / "r1_sem_v2", r1_nosem=BUF / "r1_nosem_v2")
    res = dict(ts=ts, nosem_probe=("posthoc" if pp is not None else "untrained joint probe (INVALID)"), sets={})
    g = torch.Generator(device=dev).manual_seed(11)
    for sn, root in sets.items():
        data = load_buf(root, dev)
        pi = packet_rows(data)
        end = data.A["ep_end"][pi]
        p0, p1 = data.A["pose"][pi], data.A["pose"][torch.minimum(pi + 40, end)]
        c, s_ = torch.cos(p0[:, 2]), torch.sin(p0[:, 2])
        real_fwd = (c * (p1[:, 0] - p0[:, 0]) + s_ * (p1[:, 1] - p0[:, 1])) / 0.5
        r = dict(realized_fwd_disp_norm=float(real_fwd.mean()))
        for v, M in Ms.items():
            acc = {k: [] for k in ("oracle", "teacher", "gen")}
            te = {k: [0.0, 0.0] for k in ("oracle", "teacher", "gen")}
            for s0 in range(0, len(pi), 512):
                i = pi[s0:s0 + 512]
                b = data.ctx_batch(i)
                zs = dict(oracle=M["E"](b, data.beh(i))[0], teacher=M["E"](b, data.tch[i][:, :, :40])[0],
                          gen=M["F"].sample(b, nfe=8, generator=g))
                br, ph, a1, am = data.realizer_batch(i, torch.zeros(len(i), device=dev, dtype=torch.long))
                m = am.float()
                lab = data.tch[i][:, :, 0]
                hold = data.hold_still(i)
                for k, z in zs.items():
                    acc[k].append(M["P"](z, b["asm_mask"], b["body_asm"])["disp"][:, :3])
                    te[k][0] += float((((M["R"](z, br, ph) - lab) ** 2) * m).sum()); te[k][1] += float((((hold - lab) ** 2) * m).sum())
            r[v] = dict(readout_fwd={k: float(torch.cat(x)[:, 0].mean()) for k, x in acc.items()},
                        readout_yaw_abs={k: float(torch.cat(x)[:, 2].abs().mean()) for k, x in acc.items()},
                        err_vs_teacher_j0={k: x[0] / x[1] for k, x in te.items()})
        res["sets"][sn] = r
        print(sn, json.dumps(r), flush=True)
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "readout":
        readout_and_teacher_err(sys.argv[2], sys.argv[3])
    else:
        main()
