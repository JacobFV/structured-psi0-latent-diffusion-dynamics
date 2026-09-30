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
from rrp.harness.train.pointer.relmix import drag_label_dict, needs_drag
from rrp.harness.train.pointer.split import TASKS, check_no_leak, load_split
from rrp.policies.relations.base import provenance


# ------------------------------------------------------------------------------------------------ shared helpers
def setup(a):
    """(device, Demos) for a trainer's args; seeds torch / numpy from `--seed` (init, sampling, noise) and the
    train / validation episode split from the SEPARATE `--split-seed` (so a seed sweep compares models on one held-out
    set); refuses packs collected under another geometry and any split leak."""
    geom = pointer_geometry()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Demos(sorted(a.data), dev, geom, seed=a.split_seed)
    check_no_leak(data, load_split(a.split))
    return dev, data


def ui_fields_needed(specs) -> bool:
    """Does any active spec read the public UI fields (`wzlayer` / `wparent` / `wfocusrank` / `wuiedges`)?"""
    from rrp.policies.relations.base import get_factor
    for s in specs:
        d = get_factor(s.name)
        if s.control != "off" and (d.field in ("zlayer", "parent_id") or d.field.startswith("edges:")
                                   or d.label == "drag_to"):
            return True
    return False


def factor_specs(a, data):
    """Resolved `FactorSpec`s of `--factors` for the pointer family (`resolve(family="pointer", training=True)`: a spec
    the net cannot run, or a label it cannot attach, raises). Refuses what the pointer trainers cannot honour instead of
    skipping it: `mix > 0` (no relgen scene mix here) and UI factors on packs collected before the public UI fields."""
    from rrp.policies.pointer import POINTER_FACTORS_PRESET
    from rrp.policies.relations.base import FactorError, resolve
    items = [json.loads(x) if x.lstrip().startswith("{") else x for x in (getattr(a, "factors", None) or [])]
    specs = resolve(items or None, default=POINTER_FACTORS_PRESET, family="pointer", training=True)
    if ui_fields_needed(specs) and not data.has_ui:
        raise FactorError("UI factors need packs collected with the public UI fields (wzlayer / wparent / wfocusrank / "
                          "wuiedges); recollect (`rrp train pointer collect`)")
    if needs_drag(specs) and not data.has_drag:
        raise FactorError("ui.drag_to is supervised by the teacher's drag target, which these packs lack (`ep_drag`); "
                          "recollect (`rrp train pointer collect`)")
    return specs


def make_stream(a, data, specs, dev):
    """None (plain `data.sample` batches) unless a factor has `mix > 0`: then the `relation_batches` stream
    (`relmix.RelStream`, needs `--curriculum`)."""
    if any(s.control != "off" and s.mix is not None and s.mix > 0 for s in specs):
        from rrp.harness.train.pointer.relmix import RelStream
        return RelStream(a, data, specs, dev)
    return None


def factor_arch(specs) -> dict:
    """The `arch[<module>]["factors"]` entry of a factor-bearing net: JSON-able specs (`FactorSpec.to_dict`), rebuilt by
    `relations.resolve` in `load_pointer_bundle`."""
    return dict(factors=[s.to_dict() for s in specs])


def inject_labels(rc, labels: dict) -> None:
    """Replace the `ctx` token set's pair labels by `labels` (`{name: [B, NW, NW, 1], name + ".valid": [B, NW, NW]}`,
    the teacher's / a relgen row's gt labels), zero-padded past the NW widget tokens to the set's T like the net's own."""
    ts = rc.sets["ctx"]
    pad = ts.mask.shape[1] - next(iter(labels.values())).shape[1]
    for k, v in labels.items():
        v = v[..., 0] if k[-6:] != ".valid" and v.dim() == 4 else v
        ts.labels[k] = torch.nn.functional.pad(v.to(ts.mask.device), (0, pad, 0, pad))


def factor_loss(ctx, specs, labels=None, observe=None):
    """Supervision of the estimates the last training forward of `ctx` (a `UICtx`) wrote (`relations.estimates_loss`:
    probe-source factors such as `ui.drag_to`) -> (loss, logs); (0, {}) when no spec has a readout. Run in fp32 outside
    autocast. Never leaves a probe-source head unsupervised (D-146: no silent skips). `labels` (`drag_label_dict`) are
    the batch's gt labels: they REPLACE what the net derived from its own inputs, and a `drag_to` head without them is an
    error (the net's focused-widget proxy is not a label of a drag). `observe(metrics)` gets `estimates_loss`' metrics."""
    from rrp.policies.relations.base import FactorError, estimates_loss, get_factor
    rc, ctx.last_rc = ctx.last_rc, None
    if rc is None or not any(s.control != "off" and get_factor(s.name).readout is not None for s in specs):
        return 0.0, {}
    if needs_drag(specs) and (labels is None or "drag_to" not in labels):
        raise FactorError("ui.drag_to needs the teacher's drag label (`drag_label_dict`); none was given for this batch")
    if labels:
        inject_labels(rc, labels)
    cast = lambda v: tuple(x.float() for x in v) if isinstance(v, tuple) else v.float()
    rc.estimates = {k: cast(v) for k, v in rc.estimates.items()}
    loss, logs, metrics = estimates_loss(rc, specs)
    if observe is not None:
        observe(metrics)
    return loss, {f"fx_{k}": v for k, v in logs.items()}


def supervise(ctx, specs, data, ix, stream=None):
    """The factor loss of one training step: the main rows `ix` (teacher labels from `data`) plus, with a `relation_batches`
    `stream`, the relgen rows of the step (a forward of `ctx` alone; they have no demo chunk) and the scheduler's
    competence observation (the estimates' accuracy on the main rows)."""
    loss, logs = factor_loss(ctx, specs, drag_label_dict(data, specs, ix), None if stream is None else stream.observe)
    if stream is not None and (rel := stream.relgen_batch()) is not None and ctx.record_rc:
        b, lab = rel
        ctx(b)
        Lr, rlogs = factor_loss(ctx, specs, lab or None)
        loss, logs = loss + Lr, dict(logs, **{f"rel_{k}": v for k, v in rlogs.items()})
    return loss, logs


def record_estimates(ctx, specs) -> None:
    """Have `ctx` keep each forward's `RelCtx` for `factor_loss` (only when a spec has a readout)."""
    from rrp.policies.relations.base import get_factor
    ctx.record_rc = any(s.control != "off" and get_factor(s.name).readout is not None for s in specs)


def amp(dev):
    """bf16 autocast on CUDA (losses are computed in fp32)."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda"))


def set_lr(opt, step, total, lr, warm=500):
    f = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
    for g in opt.param_groups:
        g["lr"] = lr * max(f, 0.02)


def save_checkpoint(path, *, kind, state: dict, config: dict, versions: dict, metrics: dict, factors=()):
    """`factors`: the resolved specs of the checkpoint's factor-bearing net (E / S / BC; `()` = none), stamped into
    `versions["factors"]` (`relations.stamp_versions`) and checked on load (`load_pointer_bundle`)."""
    from rrp.core.provenance import weights_digest
    from rrp.policies.relations.base import stamp_versions
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    versions = stamp_versions(versions, tuple(factors))
    blob = dict(kind=kind, state={k: m.state_dict() for k, m in state.items()}, config=config, versions=versions,
                digests={k: weights_digest(m.state_dict()) for k, m in state.items()}, metrics=metrics,
                saved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    torch.save(blob, path)
    Path(path).with_suffix(".json").write_text(json.dumps({k: v for k, v in blob.items() if k != "state"}, indent=1,
                                                          default=str))


def fit(params, step_fn, data, *, steps: int, batch: int, lr: float, clip: float | None = 1.0, eval_fn=None,
        log_every: int = 1000, stream=None) -> list[dict]:
    """The one training loop. Per step: cosine-warmup lr, a task-balanced sample `ix` (or the next `relation_batches`
    step of `stream`), `step_fn(ix) -> (loss, logs)` (which owns its forward pass, autocast and any RNG draws), backward, optional grad-norm clip,
    AdamW step. With `eval_fn() -> dict` a validation row is logged every `log_every` steps and at the last; returns the log rows."""
    params = list(params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    log, t0 = [], time.time()
    for step in range(steps):
        set_lr(opt, step, steps, lr)
        ix = data.sample(batch) if stream is None else stream.next()
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
    specs = factor_specs(a, data)
    arch = dict(E=dict(dz=a.dz, **factor_arch(specs)), R=dict(dz=a.dz), P=dict(dz=a.dz, lv_min=a.lv_min))
    E, R, P = N["PointerEncoder"](**arch["E"]).to(dev), N["PointerRealizer"](**arch["R"]).to(dev), \
        new_pointer_probe(**arch["P"]).to(dev)
    record_estimates(E.ctx, specs)
    stream = make_stream(a, data, specs, dev)
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
        Lfx, fx_logs = supervise(E.ctx, specs, data, ix, stream)
        loss = a.w_xy * Lxy + Lb + Lk + a.beta * kl_div + Lfx
        logs = dict(xy=float(Lxy.detach()), btn=float(Lb.detach()), key=float(Lk.detach()), kl=float(kl_div.detach()),
                    **fx_logs)
        if w_sem > 0:
            pl, pl_logs = probe_loss(po, lab, P.specs)
            loss = loss + w_sem * pl
            logs.update({f"p_{k}": v for k, v in pl_logs.items()})
        return loss, logs

    log = fit(list(E.parameters()) + list(R.parameters()) + list(P.parameters()), step_fn, data, steps=a.steps,
              batch=a.batch, lr=a.lr, eval_fn=lambda: _eval_rep(E, R, P if w_sem > 0 else None, data, dev),
              log_every=a.log_every, stream=stream)
    lsv, rcv = bundle_versions(f"cw_pointer_latent.v1-{a.variant}-dz{a.dz}", E.state_dict(), R.state_dict())
    cfg = dict(variant=a.variant, arch=arch, w_sem=w_sem, lv_min=a.lv_min, beta=a.beta, w_xy=a.w_xy, steps=a.steps,
               batch=a.batch, lr=a.lr, seed=a.seed, split_seed=a.split_seed, data=sorted(a.data), tasks=list(TASKS),
               target="latent", factors=provenance(specs))
    save_checkpoint(a.out, kind="pointer_rep", state=dict(E=E, R=R, P=P), config=cfg,
                    versions=dict(latent_space_version=lsv, realizer_compat_version=rcv), metrics=log[-1],
                    factors=specs)


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


def eng_targets(data, dev, bs=4096, version=None):
    """Engineered packets (rrp.policies.pointer encoding `version`, default v1) of every sample's demo chunk:
    [N, K, 1, layout.dim]. v2 (D-146 C2): the key field is the KEY_BITS-bit +-1 code of the key class."""
    from rrp.policies.pointer import ENG_VERSION, N_KEYCLS, eng_layout, packet_ticks
    from rrp.policies.nets.pointer_vocab import KEY_BITS
    lay = eng_layout(version or ENG_VERSION)
    ticks = packet_ticks(data.geom.dt)
    out = torch.zeros(data.N, 4, 1, lay.dim, device=dev)
    for s in range(0, data.N, bs):
        ix = torch.arange(s, min(s + bs, data.N), device=dev)
        _, ch, _ = data.batch(ix)
        for j, (k, sl) in enumerate(ticks):
            o = sl * lay.slot_w
            v = ch["valid"][:, j].float()
            xy = ch["xy"][:, j] * data.half
            z, one = torch.zeros_like(v), torch.ones_like(v)
            btn = ch["btn"][:, j] * 2 - 1
            if lay.key_bits:
                bits = ((ch["key"][:, j][:, None] >> torch.arange(KEY_BITS, device=dev)) & 1).float() * 2 - 1
                row = torch.cat([torch.stack([xy[:, 0], xy[:, 1], z, btn, z, one], -1), bits], -1)
            else:
                row = torch.stack([xy[:, 0], xy[:, 1], z, btn, ch["key"][:, j].float() / (N_KEYCLS - 1), z, one], -1)
            out[ix, k, 0, o:o + lay.slot_w] = row * v[:, None]
    return out


# ------------------------------------------------------------------------------------------------ flow
def cmd_flow(a):
    from rrp.policies.pointer import eng_layout, load_pointer_bundle, nets, run_pointer_probe
    dev, data = setup(a)
    N = nets()
    copy_key = getattr(a, "key_head", "free") == "copy"
    ev = eng_layout(getattr(a, "eng_version", None) or "cw_pointer_eng.v1").version
    if a.target != "eng" and (copy_key or ev != "cw_pointer_eng.v1"):
        raise ValueError("--key-head copy / --eng-version v2 are for --target eng (a latent target has no key field)")
    if copy_key and ev != "cw_pointer_eng.v2":
        raise ValueError("--key-head copy needs --eng-version cw_pointer_eng.v2 (the key is a bit code there)")
    if a.target == "eng":
        Z = eng_targets(data, dev, version=ev)
        dz, versions, P, variant, R = eng_layout(ev).dim, dict(latent_space_version=ev, realizer_compat_version=ev), \
            None, "eng", None
    else:
        rb = load_pointer_bundle(a.representation, dev)
        Z = frozen_mu(rb["modules"]["E"], data, dev)
        dz, versions, variant = Z.shape[-1], dict(rb["versions"]), rb["config"]["variant"]
        P = rb["modules"]["P"] if rb["config"]["w_sem"] > 0 else None
        R = rb["modules"]["R"]
        if P is not None:
            for p in P.parameters():
                p.requires_grad_(False)
    specs = factor_specs(a, data)
    arch = dict(S=dict(dz=dz, **factor_arch(specs), **(dict(copy_key=True) if copy_key else {})))
    S = N["PointerFlow"](**arch["S"]).to(dev)
    record_estimates(S.ctx, specs)
    stream = make_stream(a, data, specs, dev)
    tr = Z[data.train_idx]
    S.z_mean.copy_(tr.reshape(-1, dz).mean(0))
    S.z_std.copy_(tr.reshape(-1, dz).std(0).clamp(min=1e-3))
    w_sem = a.w_sem if P is not None else 0.0

    def step_fn(ix):
        b, _, lab = data.batch(ix)
        with amp(dev):
            loss, logs = S.loss(b, Z[ix], probe_fn=(lambda zc: probe_loss(run_pointer_probe(P, zc), lab, P.specs))
                                if P is not None else None, w_sem=w_sem)
        Lfx, fx_logs = supervise(S.ctx, specs, data, ix, stream)
        return loss + Lfx, dict(logs, **fx_logs)

    log = fit(S.parameters(), step_fn, data, steps=a.steps, batch=a.batch, lr=a.lr,
              eval_fn=lambda: _eval_flow(S, Z, data, dev, R), log_every=a.log_every, stream=stream)
    cfg = dict(variant=variant, target=a.target, representation=a.representation, arch=arch, w_sem=w_sem,
               steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed, split_seed=a.split_seed, data=sorted(a.data),
               tasks=list(TASKS), factors=provenance(specs))
    save_checkpoint(a.out, kind="pointer_flow", state=dict(S=S), config=cfg, versions=versions, metrics=log[-1],
                    factors=specs)


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
    specs = factor_specs(a, data)
    arch = dict(BC=dict(**factor_arch(specs), **(dict(copy_key=True) if getattr(a, "key_head", "free") == "copy"
                                                 else {})))
    BC = N["PointerBC"](**arch["BC"]).to(dev)
    record_estimates(BC.ctx, specs)
    stream = make_stream(a, data, specs, dev)

    def to_steps(x):                                      # absolute position (normalized) -> pointer-step units
        return x * data.half / data.geom.step_m

    def step_fn(ix):
        b, ch, _ = data.batch(ix)
        with amp(dev):
            xy, bl, kl = BC(b)
        ch["dxy_target"] = to_steps(ch["xy"])
        Lxy, Lb, Lk = action_loss(to_steps(xy.float()), bl, kl, ch)
        Lfx, fx_logs = supervise(BC.ctx, specs, data, ix, stream)
        return a.w_xy * Lxy + Lb + Lk + Lfx, dict(xy=float(Lxy.detach()), btn=float(Lb.detach()), key=float(Lk.detach()),
                                                  **fx_logs)

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
              log_every=a.log_every, stream=stream)
    cfg = dict(variant="bc", arch=arch, steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed, split_seed=a.split_seed,
               data=sorted(a.data), tasks=list(TASKS), w_xy=a.w_xy, factors=provenance(specs))
    save_checkpoint(a.out, kind="pointer_bc", state=dict(BC=BC), config=cfg, versions=dict(bc="cw_pointer_bc.v1"),
                    metrics=log[-1], factors=specs)


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
