"""Pointer trainers (`rep`, `flow`, `bc`, `probe`) over one generic loop, `fit`. Each command only supplies its model
parts, a per-batch `step_fn` and an `eval_fn`; the optimizer, schedule, gradient clipping and logging are shared.
Screen size, tick period and step unit come from `data.geom` (the env spec's `PointerGeometry`), never from a literal."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from rrp.harness.train.pointer.data import Demos, pointer_geometry
from rrp.harness.train.pointer.losses import _agg, _fin, action_loss, action_metrics, probe_loss, probe_metrics
from rrp.harness.train.pointer.split import TASKS, check_no_leak, load_split


# ------------------------------------------------------------------------------------------------ shared helpers
def setup(a):
    """(device, Demos) for a trainer's args; seeds torch / numpy; refuses packs collected under another geometry and
    any split leak."""
    geom = pointer_geometry()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Demos(sorted(a.data), dev, geom, seed=a.seed)
    check_no_leak(data, load_split(a.split))
    return dev, data


def amp(dev):
    """bf16 autocast on CUDA (losses are computed in fp32)."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda"))


def set_lr(opt, step, total, lr, warm=500):
    f = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
    for g in opt.param_groups:
        g["lr"] = lr * max(f, 0.02)


def save_checkpoint(path, *, kind, state: dict, config: dict, versions: dict, metrics: dict):
    from rrp.core.provenance import weights_digest
    from rrp.policies.pointer import POINTER_FACTORS_PRESET
    from rrp.policies.relations.base import compat_hash, resolve
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    versions = dict(versions, factors=compat_hash(resolve([f"preset:{POINTER_FACTORS_PRESET}"])))
    blob = dict(kind=kind, state={k: m.state_dict() for k, m in state.items()}, config=config, versions=versions,
                digests={k: weights_digest(m.state_dict()) for k, m in state.items()}, metrics=metrics,
                saved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    torch.save(blob, path)
    Path(path).with_suffix(".json").write_text(json.dumps({k: v for k, v in blob.items() if k != "state"}, indent=1,
                                                          default=str))


def fit(params, step_fn, data, *, steps: int, batch: int, lr: float, clip: float | None = 1.0, eval_fn=None,
        log_every: int = 1000) -> list[dict]:
    """The one training loop. Per step: cosine-warmup lr, a task-balanced sample `ix`, `step_fn(ix) -> (loss, logs)`
    (which owns its forward pass, autocast and any RNG draws), backward, optional grad-norm clip, AdamW step. With
    `eval_fn() -> dict` a validation row is logged every `log_every` steps and at the last; returns the log rows."""
    params = list(params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    log, t0 = [], time.time()
    for step in range(steps):
        set_lr(opt, step, steps, lr)
        ix = data.sample(batch)
        loss, logs = step_fn(ix)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        if eval_fn is not None and (step % log_every == 0 or step == steps - 1):
            log.append(dict(step=step, wall=round(time.time() - t0, 1), **logs,
                            **{f"val_{k}": v for k, v in eval_fn().items()}))
            print(json.dumps(log[-1]), flush=True)
    return log


def _tile_forward(R, z, dt: float, ptr, pbtn, dev):
    """Realize the packet `z` [B, ...] at each of the H chunk ticks (phase j * dt): -> dxy [B,H,2], btn [B,H], key [B,H,K]."""
    B, H = ptr.shape[:2]
    zz = z[:, None].expand(B, H, *z.shape[1:]).reshape(B * H, *z.shape[1:])
    ph = (torch.arange(H, device=dev, dtype=torch.float32) * dt)[None].expand(B, H).reshape(-1)
    dxy, bl, kl = R(zz, ph, ptr.reshape(-1, 2), pbtn.reshape(-1))
    return dxy.reshape(B, H, 2), bl.reshape(B, H), kl.reshape(B, H, -1)


def _rep_forward(E, R, b, a, dev, dt: float, sample=True):
    mu, lv = E(b, a)
    z = mu + (0.5 * lv).exp() * torch.randn_like(mu) if sample else mu
    return (mu, lv, z, *_tile_forward(R, z, dt, a["ptr"], a["pbtn"], dev))


def _with_targets(a):
    """Realizer targets: dxy in steps relative to the pointer measured at each tick (already in a['dxy'])."""
    a["dxy_target"] = a["dxy"]
    return a


# ------------------------------------------------------------------------------------------------ rep
def cmd_rep(a):
    from rrp.policies.pointer import new_pointer_probe, nets, run_pointer_probe
    from rrp.policies.system0 import bundle_versions
    dev, data = setup(a)
    dt = data.geom.dt
    N = nets()
    arch = dict(E=dict(dz=a.dz), R=dict(dz=a.dz), P=dict(dz=a.dz, lv_min=a.lv_min))
    E, R, P = N["PointerEncoder"](**arch["E"]).to(dev), N["PointerRealizer"](**arch["R"]).to(dev), \
        new_pointer_probe(**arch["P"]).to(dev)
    w_sem = a.w_sem if a.variant == "semfix" else 0.0

    def step_fn(ix):
        b, ch, lab = data.batch(ix)
        ch = _with_targets(ch)
        with amp(dev):
            mu, lv, z, dxy, bl, kl = _rep_forward(E, R, b, ch, dev, dt)
            po = run_pointer_probe(P, z) if w_sem > 0 else None
        mu, lv = mu.float(), lv.float()
        Lxy, Lb, Lk = action_loss(dxy, bl, kl, ch)
        kl_div = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).mean()
        loss = a.w_xy * Lxy + Lb + Lk + a.beta * kl_div
        logs = dict(xy=float(Lxy.detach()), btn=float(Lb.detach()), key=float(Lk.detach()), kl=float(kl_div.detach()))
        if w_sem > 0:
            pl, pl_logs = probe_loss(po, lab, P.specs)
            loss = loss + w_sem * pl
            logs.update({f"p_{k}": v for k, v in pl_logs.items()})
        return loss, logs

    log = fit(list(E.parameters()) + list(R.parameters()) + list(P.parameters()), step_fn, data, steps=a.steps,
              batch=a.batch, lr=a.lr, eval_fn=lambda: _eval_rep(E, R, P if w_sem > 0 else None, data, dev),
              log_every=a.log_every)
    lsv, rcv = bundle_versions(f"cw_pointer_latent.v1-{a.variant}-dz{a.dz}", E.state_dict(), R.state_dict())
    cfg = dict(variant=a.variant, arch=arch, w_sem=w_sem, lv_min=a.lv_min, beta=a.beta, w_xy=a.w_xy, steps=a.steps,
               batch=a.batch, lr=a.lr, seed=a.seed, data=sorted(a.data), tasks=list(TASKS), target="latent")
    save_checkpoint(a.out, kind="pointer_rep", state=dict(E=E, R=R, P=P), config=cfg,
                    versions=dict(latent_space_version=lsv, realizer_compat_version=rcv), metrics=log[-1])


@torch.no_grad()
def _eval_rep(E, R, P, data, dev, n=4096) -> dict:
    from rrp.policies.pointer import run_pointer_probe
    E.eval(); R.eval()
    acc = {}
    vi = data.val_idx[:n]
    for s in range(0, len(vi), 1024):
        b, ch, lab = data.batch(vi[s:s + 1024])
        ch = _with_targets(ch)
        mu, lv, z, dxy, bl, kl = _rep_forward(E, R, b, ch, dev, data.geom.dt, sample=False)
        _agg(acc, action_metrics(dxy, bl, kl, ch))
        if P is not None:
            P.eval()
            _agg(acc, probe_metrics(run_pointer_probe(P, mu), lab, P.specs, data.geom))
            P.train()
    E.train(); R.train()
    return _fin(acc)


def frozen_mu(E, data, dev, bs=2048):
    """Posterior means of the frozen encoder for every sample: [N, K, 1, dz]."""
    out = []
    E.eval()
    with torch.no_grad():
        for s in range(0, data.N, bs):
            ix = torch.arange(s, min(s + bs, data.N), device=dev)
            b, ch, _ = data.batch(ix)
            out.append(E(b, ch)[0])
    return torch.cat(out)


def eng_targets(data, dev, bs=4096):
    """Engineered packets (rrp.policies.pointer encoding) of every sample's demo chunk: [N, K, 1, 14]."""
    from rrp.policies.pointer import ENG_DIM, N_KEYCLS, SLOT_W, packet_ticks
    ticks = packet_ticks(data.geom.dt)
    out = torch.zeros(data.N, 4, 1, ENG_DIM, device=dev)
    for s in range(0, data.N, bs):
        ix = torch.arange(s, min(s + bs, data.N), device=dev)
        _, ch, _ = data.batch(ix)
        for j, (k, sl) in enumerate(ticks):
            o = sl * SLOT_W
            v = ch["valid"][:, j].float()
            xy = ch["xy"][:, j] * data.half
            out[ix, k, 0, o:o + SLOT_W] = torch.stack([xy[:, 0], xy[:, 1], torch.zeros_like(v),
                                                       ch["btn"][:, j] * 2 - 1, ch["key"][:, j].float() / (N_KEYCLS - 1),
                                                       torch.zeros_like(v), torch.ones_like(v)], -1) * v[:, None]
    return out


# ------------------------------------------------------------------------------------------------ flow
def cmd_flow(a):
    from rrp.policies.pointer import ENG_DIM, ENG_VERSION, load_pointer_bundle, nets, run_pointer_probe
    dev, data = setup(a)
    N = nets()
    if a.target == "eng":
        Z = eng_targets(data, dev)
        dz, versions, P, variant, R = ENG_DIM, dict(latent_space_version=ENG_VERSION,
                                                   realizer_compat_version=ENG_VERSION), None, "eng", None
    else:
        rb = load_pointer_bundle(a.representation, dev)
        Z = frozen_mu(rb["modules"]["E"], data, dev)
        dz, versions, variant = Z.shape[-1], dict(rb["versions"]), rb["config"]["variant"]
        P = rb["modules"]["P"] if rb["config"]["w_sem"] > 0 else None
        R = rb["modules"]["R"]
        if P is not None:
            for p in P.parameters():
                p.requires_grad_(False)
    arch = dict(S=dict(dz=dz))
    S = N["PointerFlow"](**arch["S"]).to(dev)
    tr = Z[data.train_idx]
    S.z_mean.copy_(tr.reshape(-1, dz).mean(0))
    S.z_std.copy_(tr.reshape(-1, dz).std(0).clamp(min=1e-3))
    w_sem = a.w_sem if P is not None else 0.0

    def step_fn(ix):
        b, _, lab = data.batch(ix)
        with amp(dev):
            return S.loss(b, Z[ix], probe_fn=(lambda zc: probe_loss(run_pointer_probe(P, zc), lab, P.specs))
                          if P is not None else None, w_sem=w_sem)

    log = fit(S.parameters(), step_fn, data, steps=a.steps, batch=a.batch, lr=a.lr,
              eval_fn=lambda: _eval_flow(S, Z, data, dev, R), log_every=a.log_every)
    cfg = dict(variant=variant, target=a.target, representation=a.representation, arch=arch, w_sem=w_sem,
               steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed, data=sorted(a.data), tasks=list(TASKS))
    save_checkpoint(a.out, kind="pointer_flow", state=dict(S=S), config=cfg, versions=versions, metrics=log[-1])


@torch.no_grad()
def _eval_flow(S, Z, data, dev, R=None, n=2048) -> dict:
    """Validation: generated-packet error vs the target packet (standardized units) and, with R, the realized
    first-tick action from the generated packet vs the demo."""
    S.eval()
    acc = {}
    vi = data.val_idx[:n]
    for s in range(0, len(vi), 1024):
        ix = vi[s:s + 1024]
        b, ch, _ = data.batch(ix)
        zh = S.sample(b, nfe=8)
        e = (((zh - Z[ix]) / S.z_std) ** 2).mean(dim=(1, 2, 3))
        acc["z_mse_std"] = (acc.get("z_mse_std", (0, 0))[0] + float(e.sum()), acc.get("z_mse_std", (0, 0))[1] + len(e))
        if R is not None:
            ch = _with_targets(ch)
            dxy, bl, kl = _tile_forward(R, zh, data.geom.dt, ch["ptr"], ch["pbtn"], dev)
            _agg(acc, {f"gen_{k}": v for k, v in action_metrics(dxy, bl, kl, ch).items()})
    S.train()
    return _fin(acc)


# ------------------------------------------------------------------------------------------------ bc
def cmd_bc(a):
    from rrp.policies.pointer import nets
    dev, data = setup(a)
    N = nets()
    arch = dict(BC=dict())
    BC = N["PointerBC"]().to(dev)

    def to_steps(x):                                      # absolute position (normalized) -> pointer-step units
        return x * data.half / data.geom.step_m

    def step_fn(ix):
        b, ch, _ = data.batch(ix)
        with amp(dev):
            xy, bl, kl = BC(b)
        ch["dxy_target"] = to_steps(ch["xy"])
        Lxy, Lb, Lk = action_loss(to_steps(xy.float()), bl, kl, ch)
        return a.w_xy * Lxy + Lb + Lk, dict(xy=float(Lxy.detach()), btn=float(Lb.detach()), key=float(Lk.detach()))

    def eval_fn():
        BC.eval()
        acc = {}
        with torch.no_grad():
            vi = data.val_idx[:4096]
            for s in range(0, len(vi), 1024):
                b, ch, _ = data.batch(vi[s:s + 1024])
                xy, bl, kl = BC(b)
                ch["dxy_target"] = to_steps(ch["xy"])
                _agg(acc, action_metrics(to_steps(xy), bl, kl, ch))
        BC.train()
        return _fin(acc)

    log = fit(BC.parameters(), step_fn, data, steps=a.steps, batch=a.batch, lr=a.lr, eval_fn=eval_fn,
              log_every=a.log_every)
    cfg = dict(variant="bc", arch=arch, steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed, data=sorted(a.data),
               tasks=list(TASKS), w_xy=a.w_xy)
    save_checkpoint(a.out, kind="pointer_bc", state=dict(BC=BC), config=cfg, versions=dict(bc="cw_pointer_bc.v1"),
                    metrics=log[-1])


# ------------------------------------------------------------------------------------------------ probe
def fit_probe(Z, data, dev, *, dz: int, metadata_only: bool = False, steps: int, batch: int, lr: float, seed: int):
    """A post-hoc packet probe trained on frozen packets Z [N, K, 1, dz] with the shared loop (no clip, no eval)."""
    from rrp.policies.pointer import new_pointer_probe, run_pointer_probe
    torch.manual_seed(seed)
    P = new_pointer_probe(dz=dz, metadata_only=metadata_only, lv_min=-8.0).to(dev)

    def step_fn(ix):
        _, _, lab = data.batch(ix)
        with amp(dev):
            po = run_pointer_probe(P, Z[ix])
        return probe_loss(po, lab, P.specs)[0], {}

    fit(P.parameters(), step_fn, data, steps=steps, batch=batch, lr=lr, clip=None)
    P.eval()
    return P


def cmd_probe(a):
    """Post-hoc probes on frozen packets (E posterior means, or packets generated by a flow): the same probe recipe
    for every representation, plus the metadata-only control (no z). Reports held-out-episode metrics."""
    from rrp.policies.pointer import load_pointer_bundle, run_pointer_probe
    dev, data = setup(a)
    rb = load_pointer_bundle(a.representation, dev)
    Z = frozen_mu(rb["modules"]["E"], data, dev)
    if a.flow:
        fb = load_pointer_bundle(a.flow, dev)
        S = fb["modules"]["S"]
        with torch.no_grad():
            for s in range(0, data.N, 2048):
                ix = torch.arange(s, min(s + 2048, data.N), device=dev)
                Z[ix] = S.sample(data.batch(ix)[0], nfe=8)
    res = {}
    for name, meta in (("probe", False), ("metadata_only", True)):
        P = fit_probe(Z, data, dev, dz=Z.shape[-1], metadata_only=meta, steps=a.steps, batch=a.batch, lr=a.lr,
                      seed=a.seed)
        acc = {}
        with torch.no_grad():
            for s in range(0, len(data.val_idx), 2048):
                ix = data.val_idx[s:s + 2048]
                _, _, lab = data.batch(ix)
                _agg(acc, probe_metrics(run_pointer_probe(P, Z[ix]), lab, P.specs, data.geom))
        r = _fin(acc)
        r["rel_err_px"] = r.pop("rel_err")
        res[name] = r
        print(name, json.dumps(r), flush=True)
    out = dict(representation=a.representation, flow=a.flow, variant=rb["config"]["variant"], steps=a.steps,
               val_samples=int(len(data.val_idx)), **res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
