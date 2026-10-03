"""Ψ₀ matched fine-tuning and its offline evaluations (W10; psi1z `train` + `fit_probes` + `openloop_cached`, D-140).

    rrp train psi0 --arm stageA ...       packet E/R/P (structured arm only; no VLM)
    rrp train psi0 --arm direct ...       "Ψ₀ direct"
    rrp train psi0 --arm structured ...   "Ψ₀ + structure": same transformer init, optimizer, schedule, batch, steps
                                          as direct; flow over z + bounded semantic loss through the frozen probe;
                                          actions from the frozen system 0; refuses to start unless the stage A passed
                                          the packet-use gate (`--gate`, written by `gate`, architecture 14.5 c)
    rrp train psi0 probes ...             fresh probe + metadata-only control on frozen z (diagnostics)
    rrp train psi0 gate ...               the packet-use gate on stage A alone: err(R(z_mean)) - err(R(E(a))) vs the margin
                                          -> `packet_gate.json` {gate: {gap, margin, passed, stage_a_sha256_16, ...}}
    rrp train psi0 heldout ...            held-out open-loop L1 per action group of every arm (the later model comparison)

Fairness: identical cached trunk features, data, held-out episodes, batch, steps, optimizer and schedule for direct vs
structured; stage A compute is reported separately. Monitoring: per-loss gradient norms on the shared transformer
(D-085), bounded probe NLL (lv_min -4). No teacher actions enter any input (actions are only targets / E's input for the
packet target; system 0 sees z, morphology and the current state only). `--resume` continues from last.pt.
Structure fix (D-141, architecture 14.5): stage A fits the constant-input mask on the feature cache before training
(dims with std < 1e-4 are zeroed for E, R and the structured head), drops R's state while training and adds the
permuted-packet hinge; checkpoints carry the factor list (`config["stage_a"]`) and its structure hash.
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

from rrp.core import compute
from rrp.core.provenance import file_digest
from rrp.policies.psi0.data import CachedDataset, collate
from rrp.policies.psi0 import load_launch_config, psi_home
from rrp.policies.psi0 import nets as N
from rrp.policies.nets.checkpoint import save_checkpoint

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


def parse_factors(text):
    """`--factors`: a JSON list of factor items (names, `preset:<name>`, spec dicts) or None (the arm's default)."""
    if text is None:
        return None
    items = json.loads(text)
    if not isinstance(items, list):
        raise SystemExit(f"--factors must be a JSON list of factor items, got {text!r}")
    return items


@torch.no_grad()
def state_std(ds, batch=1024):
    """Per-dim std of R's state input (`state_j`) over the whole feature cache (every item, every tick j)."""
    n, s1, s2 = 0, torch.zeros(N.DA, dtype=torch.float64), torch.zeros(N.DA, dtype=torch.float64)
    for b in torch.utils.data.DataLoader(ds, batch_size=batch, collate_fn=collate):
        x = b["state_j"].double()
        n += x.shape[0]; s1 += x.sum(0); s2 += (x ** 2).sum(0)
    mean = s1 / max(n, 1)
    return ((s2 / max(n, 1) - mean ** 2).clamp(min=0)).sqrt().float()


@torch.no_grad()
def packet_use(A, ds, z_mean, stride=4, batch=32, device="cpu", margin=N.PACKET_MARGIN):
    """Gate of architecture 14.5 (c) on held-out items: mean err(R(z_mean)) - err(R(E(a))) in normalized action units.
    `passed` iff the gap >= margin, i.e. R needs the packet."""
    A = A.to(device).eval()
    idx = [i for i, it in enumerate(ds.items) if it["fr"] % stride == 0]
    zm = z_mean.to(device)
    e_mean, e_enc = [], []
    for b0 in range(0, len(idx), batch):
        b = to_dev(collate([ds[i] for i in idx[b0:b0 + batch]]), device)
        em, ee = A.packet_gap(b, zm)
        e_mean.append(em.cpu()); e_enc.append(ee.cpu())
    if not e_mean:
        raise SystemExit("packet gate: no held-out frames")
    em, ee = torch.cat(e_mean), torch.cat(e_enc)
    gap = float((em - ee).mean())
    return dict(frames=int(em.numel()), err_z_mean=float(em.mean()), err_encoded=float(ee.mean()), gap=gap,
                margin=margin, passed=gap >= margin)


def require_gate(gate_path, stage_a_path):
    """The structured head refuses to train unless the gate file (`gate` command) reports a pass for THIS stage-A checkpoint."""
    if gate_path is None or not Path(gate_path).exists():
        raise SystemExit(f"structured arm refused: no packet-use gate at {gate_path}; run `rrp train psi0 gate "
                         f"--stage-a {stage_a_path} ...` first (architecture 14.5 c)")
    g = json.loads(Path(gate_path).read_text()).get("gate")
    if not g or g.get("stage_a_sha256_16") != file_digest(stage_a_path):
        raise SystemExit(f"structured arm refused: {gate_path} has no gate for {stage_a_path}")
    if not g["passed"]:
        raise SystemExit(f"structured arm refused: R does not use the packet (err(R(z_mean)) - err(R(E(a))) = "
                         f"{g['gap']:.4f} < margin {g['margin']}); retrain stage A")
    return g


def train(argv=None):
    ap = argparse.ArgumentParser(prog="rrp train psi0")
    ap.add_argument("--arm", choices=["stageA", "direct", "structured"], required=True)
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--run-dir", required=True, help="released SIMPLE run (model config + normalization)")
    ap.add_argument("--action-header", default=str(psi_home() / "cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"))
    ap.add_argument("--stage-a", default=None)
    ap.add_argument("--gate", default=None, help="structured arm: the `gate` command's packet_gate.json (required)")
    ap.add_argument("--factors", default=None, help="JSON list of relation-factor items (stageA: E/R/probe list; "
                    "structured: the context tokens' list); default = the arm's default lists")
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
    ap.add_argument("--p-state-dim", type=float, default=0.3, help="stage A: per-dim state dropout on R (14.5 b)")
    ap.add_argument("--p-state-all", type=float, default=0.1, help="stage A: whole-state dropout on R (14.5 b)")
    ap.add_argument("--w-perm", type=float, default=1.0, help="stage A: permuted-packet hinge weight (0 = off)")
    ap.add_argument("--perm-margin", type=float, default=N.PACKET_MARGIN)
    ap.add_argument("--gn-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    factors = parse_factors(a.factors)
    if a.arm == "structured":
        if not (a.stage_a and a.gate):
            raise SystemExit("--arm structured needs --stage-a and --gate")
        gate = require_gate(a.gate, a.stage_a)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device
    cx = compute.setup("psi0.train", dev)      # LEGACY['psi0']: bf16 CUDA autocast, as before
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
        model = N.StageA(grasp=a.w_grasp > 0, factors=factors)
        masked = model.fit_state_mask(state_std(ds))
        print(f"[train] constant-input mask: {len(masked)} state dims zeroed {masked}", flush=True)
        model = model.to(dev)
        cx.compile(model.E, "E")
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
            model = N.StructuredHead(mcfg, A, zs["mean"], zs["std"], w_sem=a.w_sem, w_grasp=a.w_grasp, factors=factors)
        info = N.load_pretrained_blocks(model.header, a.action_header)
        print(f"[train] pretrained blocks: {info}", flush=True)
        model = model.to(dev)
        cx.compile(model.header, "header")
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
        with cx.autocast():
            if a.arm == "stageA":
                loss, logs, parts = model.loss(b, w_kl=a.w_kl, w_sem=a.w_sem if "labels" in b else 0.0,
                                               lv_min=a.lv_min, z_noise=a.z_noise, w_grasp=a.w_grasp,
                                               p_state_dim=a.p_state_dim, p_state_all=a.p_state_all, w_perm=a.w_perm,
                                               perm_margin=a.perm_margin)
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
    with torch.no_grad(), cx.autocast():
        for b in dlv:
            b = to_dev(b, dev)
            l, logs, _ = (model.loss(b, w_kl=a.w_kl, w_sem=a.w_sem if "labels" in b else 0.0, lv_min=a.lv_min)
                          if a.arm == "stageA" else model.loss(b))
            vals.append(logs)
            if a.arm == "stageA" and "labels" in b:
                N.add_cmd_labels(b)
                mu, _ = model.E(model.morph, b["state0"], b["actions"])
                for k, (s_, n_) in N.probe_metrics(N.run_probe(model.P, mu), b["labels"], model.P.specs).items():
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
        save_checkpoint(out / "stage_a.pt", model=model, step=step, versions=dict(psi0_nets=N.NETS_VERSION),
                        config=dict(vars(a), stage_a=model.cfg), source=f"learned:{out.name}/stage_a.pt")
    else:
        save_checkpoint(out / "final.pt", model=model, step=step, versions=dict(psi0_nets=N.NETS_VERSION),
                        config=dict(vars(a), packet_gate=gate if a.arm == "structured" else None),
                        source=f"learned:{out.name}/final.pt")
    from rrp.core.provenance import make_provenance, source_label, weights_digest
    ck_out = out / ("stage_a.pt" if a.arm == "stageA" else "final.pt")
    summ["provenance"] = make_provenance(source_label("learned", str(ck_out)), weights=dict(head=weights_digest(model.state_dict())),
                                         flags=dict(arm=a.arm, zero_prev_action=True), notes="Ψ₀ matched fine-tune").model_dump(mode="json")
    cx.write_stamp(out)
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
    cx = compute.setup("psi0.probes", dev)
    summ = json.loads(Path(a.train_summary).read_text())
    A = N.load_stage_a(a.stage_a); A = A.to(dev).eval()
    head = None
    lh = a.structured is not None
    if lh:
        zs = torch.load(Path(a.stage_a).parent / "z_stats.pt")
        head = N.StructuredHead(load_model_cfg(a.run_dir), A, zs["mean"], zs["std"])
        N.load_structured(head, a.structured); head = head.to(dev).eval()
    ds = CachedDataset(a.feat_dir, a.labels_dir, episodes=set(summ["train_eps"]), load_hidden=lh)
    dv = CachedDataset(a.feat_dir, a.labels_dir, episodes=set(summ["val_eps"]), load_hidden=lh)
    res = {}
    for name, meta_only in (("probe_on_z", False), ("metadata_only_control", True)):
        torch.manual_seed(0)
        P = N.new_probe(metadata_only=meta_only, specs=A.specs).to(dev)
        opt = torch.optim.AdamW(P.parameters(), lr=3e-4)
        dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, drop_last=True, collate_fn=collate)
        it, step = iter(dl), 0
        while step < a.steps:
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            b = to_dev(b, dev)
            with torch.no_grad(), cx.autocast():
                z = zs_of(A, b, head).float()
            l, _ = N.probe_loss(N.run_probe(P, z), b["labels"], P.specs, lv_min=-4.0)
            opt.zero_grad(); l.backward(); opt.step(); step += 1
        pm = {}
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(dv, batch_size=128, collate_fn=collate):
                b = to_dev(b, dev)
                with cx.autocast():
                    z = zs_of(A, b, head).float()
                for k, (s_, n_) in N.probe_metrics(N.run_probe(P, z), b["labels"], P.specs).items():
                    ps, pn = pm.get(k, (0.0, 0)); pm[k] = (ps + s_, pn + n_)
        res[name] = {k: (s_ / n_ if n_ else None) for k, (s_, n_) in pm.items()}
        print(name, json.dumps(res[name]), flush=True)
    res["z_source"] = "system_i_generated" if lh else "E(demonstrated chunk)"
    Path(a.out).write_text(json.dumps(res, indent=1))


def packet_gate(argv=None):
    """The packet-use gate of architecture 14.5 (c) on the held-out episodes of a stage A, and nothing else: mean
    err(R(z_mean)) - err(R(E(a))) vs the margin -> `--out` (`packet_gate.json`): {"gate": {gap, margin, passed, frames, err_*,
    stage_a, stage_a_sha256_16, val_episodes}}. `train --arm structured --gate` reads it; the `gate` pipeline stage fails
    the node (GateFailed) when gap < margin. The diagnostic reads the target actions (E(a)): never a deployable number."""
    ap = argparse.ArgumentParser(prog="rrp train psi0 gate")
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--stage-a", required=True)
    ap.add_argument("--val-episodes", required=True, help="comma list (from the training summary)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    eps = {int(x) for x in a.val_episodes.split(",")}
    ds = CachedDataset(a.feat_dir, None, episodes=eps, max_j=1, load_hidden=False)
    zmean = torch.load(Path(a.stage_a).parent / "z_stats.pt")["mean"]
    g = dict(packet_use(N.load_stage_a(a.stage_a), ds, zmean, a.stride, device=a.device),
             stage_a=str(a.stage_a), stage_a_sha256_16=file_digest(a.stage_a), val_episodes=sorted(eps))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(gate=g), indent=1))
    print("packet gate", json.dumps(g), flush=True)
    return g


def heldout(argv=None):
    """Held-out open-loop L1 per action group on cached features: released Ψ₀ header, direct, structured, and the
    DIAGNOSTIC oracle route R(E(demonstrated chunk)) (uses the target actions; never a deployable number). The
    pre-head packet gate is the separate `gate` command."""
    ap = argparse.ArgumentParser(prog="rrp train psi0 heldout")
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--direct", default=None)
    ap.add_argument("--structured", default=None)
    ap.add_argument("--stage-a", default=None)
    ap.add_argument("--val-episodes", required=True, help="comma list (from the training summary)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.structured and not a.stage_a:
        raise SystemExit("heldout --structured needs --stage-a (the head is built over its stage A)")
    cx = compute.setup("psi0.heldout", a.device)
    eps = {int(x) for x in a.val_episodes.split(",")}
    lc = load_launch_config(Path(a.run_dir))
    maxmin = lc.data.transform.field
    mcfg = lc.model
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
        m = N.StructuredHead(mcfg, A, zs["mean"], zs["std"]); N.load_structured(m, a.structured)
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
        m = m.to(a.device).eval()
        errs, errs24 = [], []
        g = torch.Generator(device=a.device).manual_seed(0)
        for b0 in range(0, len(idx), 32):
            b = to_dev(collate([ds[i] for i in idx[b0:b0 + 32]]), a.device)
            with torch.no_grad(), cx.autocast():
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


def _hand_pred(P, z):
    lg = N.run_probe(P, z)["active_hand"].float()
    while lg.dim() > 2:
        lg = lg.mean(1)
    return lg.argmax(-1)


def flip_rate(seqs) -> float | None:
    """Fraction of consecutive pairs (within each sequence) whose value changes; None without pairs."""
    n = f = 0
    for q in seqs:
        for x, y in zip(q[:-1], q[1:]):
            n += 1; f += int(x != y)
    return f / n if n else None


def hand_consistency(argv=None):
    """DIAGNOSTIC (D-147 T7, not a result): does system i commit to one hand across chunks? An active-hand probe is fit on
    E(demonstrated chunk) packets (train episodes, as `probes`), then read on held-out chunk-start frames (every `--stride`
    frames = one executed chunk) from (a) E(chunk) (uses the target actions: oracle side), and (b) `--samples` independent
    system-i packets per frame (the closed loop re-samples one per chunk). Reports per-frame sample agreement, accuracy vs the
    label, and consecutive-chunk flip rates of label / E-decode / one-sample-per-chunk decode. Demo states only: it cannot
    show off-manifold behaviour."""
    ap = argparse.ArgumentParser(prog="rrp train psi0 handcons")
    for k in ("--feat-dir", "--labels-dir", "--stage-a", "--train-summary", "--structured", "--run-dir", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--stride", type=int, default=24)
    a = ap.parse_args(argv)
    dev = "cuda"
    cx = compute.setup("psi0.probes", dev)
    summ = json.loads(Path(a.train_summary).read_text())
    A = N.load_stage_a(a.stage_a).to(dev).eval()
    zs = torch.load(Path(a.stage_a).parent / "z_stats.pt")
    head = N.StructuredHead(load_model_cfg(a.run_dir), A, zs["mean"], zs["std"])
    N.load_structured(head, a.structured); head = head.to(dev).eval()
    ds = CachedDataset(a.feat_dir, a.labels_dir, episodes=set(summ["train_eps"]), load_hidden=False)
    dv = CachedDataset(a.feat_dir, a.labels_dir, episodes=set(summ["val_eps"]), load_hidden=True)
    torch.manual_seed(0)
    P = N.new_probe(specs=A.specs).to(dev)
    opt = torch.optim.AdamW(P.parameters(), lr=3e-4)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, drop_last=True, collate_fn=collate)
    it = iter(dl)
    for _ in range(a.steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        b = to_dev(b, dev)
        with torch.no_grad(), cx.autocast():
            z = zs_of(A, b, None).float()
        l, _ = N.probe_loss(N.run_probe(P, z), b["labels"], P.specs, lv_min=-4.0)
        opt.zero_grad(); l.backward(); opt.step()
    P.eval()
    idx = sorted((i for i, x in enumerate(dv.items) if x["fr"] % a.stride == 0), key=lambda i: (dv.items[i]["ep"], dv.items[i]["fr"]))
    gens = [torch.Generator(device=dev).manual_seed(1000 + k) for k in range(a.samples)]
    rows = []
    with torch.no_grad():
        for b0 in range(0, len(idx), 16):
            ii = idx[b0:b0 + 16]
            b = to_dev(collate([dv[i] for i in ii]), dev)
            with cx.autocast():
                pe = _hand_pred(P, zs_of(A, b, None).float())
                ps = torch.stack([_hand_pred(P, head.sample_z(b, generator=g).float()) for g in gens], 1)
            lab = b["labels"]["active_hand"].view(-1)
            for j, i in enumerate(ii):
                rows.append(dict(ep=int(dv.items[i]["ep"]), fr=int(dv.items[i]["fr"]), label=int(lab[j]), e=int(pe[j]),
                                 s=[int(x) for x in ps[j]]))
    lab_rows = [r for r in rows if r["label"] >= 0]
    eps = sorted({r["ep"] for r in rows})
    seq = lambda key: [[key(r) for r in rows if r["ep"] == e and r["label"] >= 0] for e in eps]
    res = dict(
        diagnostic="hand_consistency (DIAGNOSTIC; demo states only; E-decode uses the target actions)",
        frames=len(rows), labelled=len(lab_rows), stride=a.stride, samples=a.samples, val_episodes=eps,
        acc_E=float(np.mean([r["e"] == r["label"] for r in lab_rows])) if lab_rows else None,
        acc_sample=float(np.mean([s == r["label"] for r in lab_rows for s in r["s"]])) if lab_rows else None,
        sample_agreement=float(np.mean([max(r["s"].count(0), r["s"].count(1)) / len(r["s"]) for r in rows])),
        frac_frames_split=float(np.mean([0 < r["s"].count(1) < len(r["s"]) for r in rows])),
        flip_label=flip_rate(seq(lambda r: r["label"])), flip_E=flip_rate(seq(lambda r: r["e"])),
        flip_sample_per_chunk=[flip_rate(seq(lambda r, k=k: r["s"][k])) for k in range(a.samples)],
        rows=rows)
    res["flip_sample_per_chunk_mean"] = float(np.mean([x for x in res["flip_sample_per_chunk"] if x is not None]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("handcons", json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)
    return {k: v for k, v in res.items() if k != "rows"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("probes", "gate", "heldout", "handcons"):
        return dict(probes=fit_probes, gate=packet_gate, heldout=heldout, handcons=hand_consistency)[argv[0]](argv[1:])
    return train(argv)
