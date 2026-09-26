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
    return torch.nonzero(data.pk).squeeze(-1)


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="v2")
    ap.add_argument("--teacher-geometry", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
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


if __name__ == "__main__":
    main()
