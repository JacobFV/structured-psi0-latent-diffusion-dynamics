"""Ψ₀ matched fine-tuning and its offline evaluations (W10; psi1z `train` + `fit_probes` + `openloop_cached`, D-140).

    rrp train psi0 --arm stageA ...       packet E/R/P (structured arm only; no VLM)
    rrp train psi0 --arm direct ...       "Ψ₀ direct"
    rrp train psi0 --arm structured ...   "Ψ₀ + structure": same transformer init, optimizer, schedule, batch, steps
                                          as direct; flow over z + bounded semantic loss through the frozen probe;
                                          actions from the frozen system 0
    rrp train psi0 probes ...             fresh probe + metadata-only control on frozen z (diagnostics)
    rrp train psi0 heldout ...            held-out open-loop L1 per action group

Fairness: identical cached trunk features, data, held-out episodes, batch, steps, optimizer and schedule for direct vs
structured; stage A compute is reported separately. Monitoring: per-loss gradient norms on the shared transformer
(D-085), bounded probe NLL (lv_min -4). No teacher actions enter any input (actions are only targets / E's input for the
packet target; system 0 sees z, morphology and the current state only). `--resume` continues from last.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rrp.policies.psi0.data import CachedDataset, collate
from rrp.policies.psi0 import load_launch_config, psi_home
from rrp.policies.psi0 import nets as N

GROUPS = [("hand_joints", 0, 14), ("arm_joints", 14, 28), ("waist_rp", 28, 30), ("waist_yaw", 30, 31),
          ("height", 31, 32), ("vx", 32, 33), ("vy", 33, 34), ("turn_flag", 34, 35), ("target_yaw", 35, 36)]


def to_dev(b, dev):
    out = {}
    for k, v in b.items():
        if isinstance(v, dict):
            out[k] = to_dev(v, dev)
        elif torch.is_tensor(v):
            out[k] = v.to(dev, non_blocking=True)
        else:
            out[k] = v
    return out


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def split_episodes(n_eps, n_val, seed):
    rng = np.random.default_rng(seed)
    val = set(int(x) for x in rng.choice(n_eps, n_val, replace=False))
    return [e for e in range(n_eps) if e not in val], sorted(val)


def grad_norms(parts, params):
    out = {}
    for k, l in parts.items():
        if not l.requires_grad:
            continue
        g = torch.autograd.grad(l, params, retain_graph=True, allow_unused=True)
        out[f"gn_{k}"] = float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)))
    return out


def load_model_cfg(run_dir):
    return load_launch_config(Path(run_dir)).model


def train(argv=None):
    ap = argparse.ArgumentParser(prog="rrp train psi0")
    ap.add_argument("--arm", choices=["stageA", "direct", "structured"], required=True)
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--run-dir", required=True, help="released SIMPLE run (model config + normalization)")
    ap.add_argument("--action-header", default=str(psi_home() / "cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"))
    ap.add_argument("--stage-a", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--val-eps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-sem", type=float, default=0.1)
    ap.add_argument("--w-kl", type=float, default=1e-3)
    ap.add_argument("--lv-min", type=float, default=-4.0)
    ap.add_argument("--z-noise", type=float, default=0.0)
    ap.add_argument("--w-grasp", type=float, default=0.0, help="grasp-region affordance probe weight (roadmap #24; 0 = off, no head)")
    ap.add_argument("--rec-w-cmd", type=float, default=1.0, help="stage A: reconstruction weight of vx/vyaw dims")
    ap.add_argument("--gn-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(a.feat_dir) / "meta.json").read_text())
    tr_eps, va_eps = split_episodes(meta["episodes"], a.val_eps, 1234)          # same split for every arm
    t_load = time.time()
    lh = a.arm != "stageA"
    ds = CachedDataset(a.feat_dir, a.labels_dir, episodes=set(tr_eps), seed=a.seed, load_hidden=lh)
    dv = CachedDataset(a.feat_dir, a.labels_dir, episodes=set(va_eps), seed=a.seed + 1, load_hidden=lh)
    print(f"[train] {a.arm}: train {len(ds)} frames / {len(tr_eps)} eps, val {len(dv)} / {len(va_eps)} eps, "
          f"labels {len(ds.labels)} eps, load {time.time() - t_load:.0f}s", flush=True)
    g = torch.Generator().manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, collate_fn=collate,
                                     num_workers=a.workers, generator=g, persistent_workers=a.workers > 0)
    dlv = torch.utils.data.DataLoader(dv, batch_size=a.batch, shuffle=False, collate_fn=collate)

    if a.arm == "stageA":
        model = N.StageA(grasp=a.w_grasp > 0).to(dev)
        if a.rec_w_cmd != 1.0:
            w = torch.ones(N.DA); w[list(N.CMD_DIMS)] = a.rec_w_cmd; model.rec_dim_w = w
        shared = list(model.E.parameters())
        lr = a.lr
    else:
        mcfg = load_model_cfg(a.run_dir)
        if a.arm == "direct":
            model = N.DirectHead(mcfg)
        else:
            A = N.load_stage_a(a.stage_a)
            zs = torch.load(Path(a.stage_a).parent / "z_stats.pt")
            model = N.StructuredHead(mcfg, A, zs["mean"], zs["std"], w_sem=a.w_sem, w_grasp=a.w_grasp)
        info = N.load_pretrained_blocks(model.header, a.action_header)
        print(f"[train] pretrained blocks: {info}", flush=True)
        model = model.to(dev)
        shared = list(model.header.transformer_blocks[-1].parameters())
        lr = a.lr
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.95, 0.999), weight_decay=1e-6)
    n_train = sum(p.numel() for p in params)
    step, t_gpu = 0, 0.0
    ck = out / "last.pt"
    if a.resume and ck.exists():
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"], strict=False); opt.load_state_dict(st["opt"]); step = st["step"]
        t_gpu = st.get("gpu_seconds", 0.0)
        print(f"[train] resumed at step {step}", flush=True)
    logf = open(out / "train_log.jsonl", "a")
    it = iter(dl)
    model.train()
    t0 = time.time()
    while step < a.steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        b = to_dev(b, dev)
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, a.steps, a.warmup, lr)
        ts = time.time()
        with torch.autocast("cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=dev == "cuda"):
            if a.arm == "stageA":
                loss, logs, parts = model.loss(b, w_kl=a.w_kl, w_sem=a.w_sem if "labels" in b else 0.0,
                                               lv_min=a.lv_min, z_noise=a.z_noise, w_grasp=a.w_grasp)
            else:
                loss, logs, parts = model.loss(b)
        rec = {"step": step, "loss": float(loss.detach()), **logs}
        if a.gn_every and (step + 1) % a.gn_every == 0:          # logged below (log_every divides gn_every)
            rec.update(grad_norms(parts, [p for p in shared if p.requires_grad]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        rec["gn_total"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        t_gpu += time.time() - ts
        step += 1
        if step % a.log_every == 0 or step == 1 or any(k.startswith("gn_") and k != "gn_total" for k in rec):
            rec.update(lr=opt.param_groups[0]["lr"], elapsed=time.time() - t0)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, gpu_seconds=t_gpu, args=vars(a)), ck)
    # validation (held-out episodes): flow/rec loss + probe metrics
    model.eval()
    vals, pm = [], {}
    with torch.no_grad(), torch.autocast("cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=dev == "cuda"):
        for b in dlv:
            b = to_dev(b, dev)
            l, logs, _ = (model.loss(b, w_kl=a.w_kl, w_sem=a.w_sem if "labels" in b else 0.0, lv_min=a.lv_min)
                          if a.arm == "stageA" else model.loss(b))
            vals.append(logs)
            if a.arm == "stageA" and "labels" in b:
                N.add_cmd_labels(b)
                mu, _ = model.E(model.morph, b["state0"], b["actions"])
                for k, (s_, n_) in N.probe_metrics(model.P(mu), b["labels"]).items():
                    ps, pn = pm.get(k, (0.0, 0)); pm[k] = (ps + s_, pn + n_)
    summ = dict(arm=a.arm, steps=step, batch=a.batch, lr=lr, trainable_params=n_train, gpu_seconds=t_gpu,
                train_eps=tr_eps, val_eps=va_eps, feat_meta=meta,
                val={k: float(np.mean([v[k] for v in vals if k in v])) for k in (vals[0] if vals else {})},
                val_probe={k: (s_ / n_ if n_ else None) for k, (s_, n_) in pm.items()}, args=vars(a))
    if a.arm == "stageA":
        zs = []
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(ds, batch_size=256, collate_fn=collate):
                b = to_dev(b, dev)
                mu, _ = model.E(model.morph, b["state0"], b["actions"])
                zs.append(mu.float().cpu())
        z = torch.cat(zs)                                          # [N, K, M, DZ]
        torch.save(dict(mean=z.mean(0), std=z.std(0).clamp(min=1e-3)), out / "z_stats.pt")
        torch.save(dict(model=model.state_dict(), args=vars(a)), out / "stage_a.pt")
    else:
        torch.save(dict(model={k: v for k, v in model.state_dict().items()}, args=vars(a), step=step), out / "final.pt")
    from rrp.core.provenance import make_provenance, source_label, weights_digest
    ck_out = out / ("stage_a.pt" if a.arm == "stageA" else "final.pt")
    summ["provenance"] = make_provenance(source_label("learned", str(ck_out)), weights=dict(head=weights_digest(model.state_dict())),
                                         flags=dict(arm=a.arm, zero_prev_action=True), notes="Ψ₀ matched fine-tune").model_dump(mode="json")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print("[train] done", json.dumps({k: summ[k] for k in ("arm", "steps", "gpu_seconds", "val", "val_probe")}), flush=True)


def zs_of(A, b, head=None, nfe=10):
    if head is None:
        mu, _ = A.E(A.morph, b["state0"], b["actions"])
        return mu
    return head.sample_z(b, nfe=nfe)


def fit_probes(argv=None):
    ap = argparse.ArgumentParser(prog="rrp train psi0 probes")
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--stage-a", required=True)
    ap.add_argument("--train-summary", required=True, help="summary.json of the run whose train/val split to reuse")
    ap.add_argument("--structured", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = "cuda"
    summ = json.loads(Path(a.train_summary).read_text())
    A = N.load_stage_a(a.stage_a); A = A.to(dev).eval()
    head = None
    lh = a.structured is not None
    if lh:
        zs = torch.load(Path(a.stage_a).parent / "z_stats.pt")
        head = N.StructuredHead(load_model_cfg(a.run_dir), A, zs["mean"], zs["std"])
        N.load_tolerant(head, torch.load(a.structured, weights_only=False)["model"]); head = head.to(dev).eval()
    ds = CachedDataset(a.feat_dir, a.labels_dir, episodes=set(summ["train_eps"]), load_hidden=lh)
    dv = CachedDataset(a.feat_dir, a.labels_dir, episodes=set(summ["val_eps"]), load_hidden=lh)
    res = {}
    for name, meta_only in (("probe_on_z", False), ("metadata_only_control", True)):
        torch.manual_seed(0)
        P = N.PacketProbe(metadata_only=meta_only).to(dev)
        opt = torch.optim.AdamW(P.parameters(), lr=3e-4)
        dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, drop_last=True, collate_fn=collate)
        it, step = iter(dl), 0
        while step < a.steps:
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            b = to_dev(b, dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z = zs_of(A, b, head).float()
            l, _ = N.probe_loss(P(z), b["labels"], lv_min=-4.0)
            opt.zero_grad(); l.backward(); opt.step(); step += 1
        pm = {}
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(dv, batch_size=128, collate_fn=collate):
                b = to_dev(b, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = zs_of(A, b, head).float()
                for k, (s_, n_) in N.probe_metrics(P(z), b["labels"]).items():
                    ps, pn = pm.get(k, (0.0, 0)); pm[k] = (ps + s_, pn + n_)
        res[name] = {k: (s_ / n_ if n_ else None) for k, (s_, n_) in pm.items()}
        print(name, json.dumps(res[name]), flush=True)
    res["z_source"] = "system_i_generated" if lh else "E(demonstrated chunk)"
    Path(a.out).write_text(json.dumps(res, indent=1))


def heldout(argv=None):
    """Held-out open-loop L1 per action group on cached features: released Ψ₀ header, direct, structured, and the
    DIAGNOSTIC oracle route R(E(demonstrated chunk)) (uses the target actions; never a deployable number)."""
    ap = argparse.ArgumentParser(prog="rrp train psi0 heldout")
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--direct", default=None)
    ap.add_argument("--structured", default=None)
    ap.add_argument("--stage-a", default=None)
    ap.add_argument("--val-episodes", required=True, help="comma list (from the training summary)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lc = load_launch_config(Path(a.run_dir))
    maxmin = lc.data.transform.field
    mcfg = lc.model
    eps = {int(x) for x in a.val_episodes.split(",")}
    ds = CachedDataset(a.feat_dir, None, episodes=eps, max_j=1)
    idx = [i for i, it in enumerate(ds.items) if it["fr"] % a.stride == 0]
    models = {}
    rel = N.DirectHead(mcfg)
    from safetensors import safe_open
    sd = {}
    with safe_open(f"{a.run_dir}/checkpoints/ckpt_40000/model.safetensors", "pt") as f:
        for k in f.keys():
            if k.startswith("action_header."):
                sd[k[len("action_header."):]] = f.get_tensor(k)
    rel.header.load_state_dict(sd, strict=True)
    models["psi0_released_ckpt40000"] = rel
    if a.direct:
        m = N.DirectHead(mcfg); m.load_state_dict(torch.load(a.direct, weights_only=False)["model"]); models["direct"] = m
    if a.structured:
        A = N.load_stage_a(a.stage_a)
        zs = torch.load(Path(a.stage_a).parent / "z_stats.pt")
        m = N.StructuredHead(mcfg, A, zs["mean"], zs["std"]); N.load_tolerant(m, torch.load(a.structured, weights_only=False)["model"])
        models["structured"] = m

        class Oracle(torch.nn.Module):       # DIAGNOSTIC (oracle): R(E(demonstrated chunk)) — uses the target actions
            def __init__(self, A):
                super().__init__(); self.A = A

            def sample(self, b, nfe=10, generator=None):
                mu, _ = self.A.E(self.A.morph, b["state0"], b["actions"])
                return self.A.R(self.A.morph, mu, b["state0"], torch.zeros(mu.shape[0], device=mu.device))
        models["oracle_R_of_E(actions)"] = Oracle(m.A)
    res = dict(val_episodes=sorted(eps), frames=len(idx), stride=a.stride, nfe=a.nfe, models={})
    for name, m in models.items():
        m = m.to("cuda").eval()
        errs, errs24 = [], []
        g = torch.Generator(device="cuda").manual_seed(0)
        for b0 in range(0, len(idx), 32):
            b = to_dev(collate([ds[i] for i in idx[b0:b0 + 32]]), "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = m.sample(b, nfe=a.nfe, generator=g).float()
            p = np.asarray(maxmin.denormalize(pred.cpu().numpy()))
            gt = np.asarray(maxmin.denormalize(b["actions"].float().cpu().numpy()))
            msk = b["amask"].cpu().numpy() > 0
            e = np.where(msk, np.abs(p - gt), np.nan)
            errs.append(np.nanmean(e, 1)); errs24.append(np.nanmean(e[:, :24], 1))
        E, E24 = np.concatenate(errs), np.concatenate(errs24)
        res["models"][name] = dict(l1={n: float(np.nanmean(E[:, s:t])) for n, s, t in GROUPS},
                                   l1_exec24={n: float(np.nanmean(E24[:, s:t])) for n, s, t in GROUPS})
        m.cpu()
        print(name, json.dumps(res["models"][name]["l1"]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("probes", "heldout"):
        return (fit_probes if argv[0] == "probes" else heldout)(argv[1:])
    return train(argv)
