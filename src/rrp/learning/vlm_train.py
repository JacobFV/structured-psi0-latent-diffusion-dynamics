"""VLM-backed policy training + closed-loop evaluation with online rendering.

Policy = FlowPolicy(image_tokens=Q, image_dim=D) + learned Resampler over CACHED frozen VLM
features (rrp.data.vlm_features). The matched no-image baseline trains on the identical
(episode, t) samples with image_tokens=0. Online evaluation renders the front camera per policy
call, runs the frozen VLM, and injects image_tokens (reuses rrp.evaluation.runner.evaluate via a
LearnedPolicy subclass). Image controls at evaluation: real | shuffled (features of another
session in the batch) | blank (zero VLM features).
Every chunk is source="learned"; the scripted teacher only produced the demonstrations.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rrp.learning.checkpoint import save_checkpoint, load_checkpoint
from rrp.learning.data import episode_samples, collate_samples
from rrp.model.backbone import Resampler
from rrp.model.flow import FlowPolicy, PolicyConfig
from rrp.ops.jobs import CheckpointSignal

FEAT_VERSION = "feat-v1"


class VLMFlowPolicy(nn.Module):
    """FlowPolicy + resampler. Exposes the FlowPolicy interface used by LearnedPolicy
    (cfg / prepare / sample) with image tokens injected from `self.pending` when set."""

    def __init__(self, pcfg: PolicyConfig, rcfg: dict | None):
        super().__init__()
        self.policy = FlowPolicy(pcfg)
        self.resampler = Resampler(**rcfg) if rcfg else None
        self.cfg = pcfg
        self.pending = None           # (tokens [B,L,T,W], n_visual) for the next prepare()

    def image_tokens(self, feats, n_visual):
        return self.resampler(feats, n_visual)

    def attach(self, batch, feats=None, n_visual=None):
        if self.resampler is None:
            return batch
        if feats is None and self.pending is not None:
            feats, n_visual = self.pending
        if feats is None:
            raise ValueError("image policy called without VLM features")
        batch.extra["image_tokens"] = self.image_tokens(feats.to(next(self.parameters()).device), n_visual)
        return batch

    def loss(self, batch, target, valid, labels, feats=None, n_visual=None, **kw):
        return self.policy.loss(self.attach(batch, feats, n_visual), target, valid, labels, **kw)

    def prepare(self, batch, key=("uncached",), rewire_gen=None):
        return self.policy.prepare(self.attach(batch), key, rewire_gen)

    def sample(self, *a, **kw):
        return self.policy.sample(*a, **kw)

    def velocity(self, *a, **kw):
        return self.policy.velocity(*a, **kw)


def load_split(cfg: dict):
    """Cached episodes -> (train samples, heldout samples) at keyframe stride; split by seed rank."""
    from rrp.data.collect import read_episode
    from rrp.data.vlm_features import FeatureStore
    store = FeatureStore(Path(cfg["feature_cache"]))
    ds = Path(cfg["dataset"]) / "episodes"
    by_robot = {}
    for eid in store.index["episodes"]:
        rk = eid.split("pick_place_")[1].rsplit("_s", 1)[0]
        by_robot.setdefault(rk, []).append(eid)
    train, held = [], []
    for rk, eids in sorted(by_robot.items()):
        if rk not in cfg["train_robots"]:
            continue
        eids = sorted(eids, key=lambda e: int(e.rsplit("_s", 1)[1]))
        n_tr = cfg.get("train_episodes_per_robot", 25)
        for j, eid in enumerate(eids):
            pub = read_episode(ds / f"{eid}.public.pkl.gz")
            prv = read_episode(ds / f"{eid}.private.pkl.gz")
            ss = [s for s in episode_samples(pub, prv, cfg["horizon"], stride=cfg.get("every", 4))
                  if (eid, s.meta["t"]) in store]
            (train if j < n_tr else held).extend(ss)
    return store, train, held


def batches(samples, bs, rng, store, shuffle=True, drop_last=True):
    idx = list(range(len(samples)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        ch = [samples[j] for j in idx[i:i + bs]]
        if drop_last and len(ch) < bs:
            break
        batch, a, v, lab, eff = collate_samples(ch)
        feats = store.get([(s.meta["episode"], s.meta["t"]) for s in ch])
        yield batch, a, v, lab, feats


def build_model(cfg, store):
    pcfg = dict(cfg["policy"])
    rcfg = None
    if cfg.get("use_images", True):
        L, T, W = store.tokens.shape[1:]
        rcfg = dict(in_width=W, n_taps=T, **cfg.get("resampler", {}))
        pcfg.update(image_tokens=rcfg.get("queries", 16), image_dim=rcfg.get("dim", 256))
    return VLMFlowPolicy(PolicyConfig(**pcfg), rcfg), pcfg, rcfg


@torch.no_grad()
def heldout_loss(model, held, store, dev, n_visual, image_mode="real", max_batches=20):
    model.eval()
    g = torch.Generator(device=dev).manual_seed(0)
    rng = random.Random(0)
    tot, n = 0.0, 0
    for i, (batch, a, v, lab, feats) in enumerate(batches(held, 128, rng, store, shuffle=True, drop_last=False)):
        if i >= max_batches:
            break
        batch, a, v = batch.to(dev), a.to(dev), v.to(dev)
        if model.resampler is not None:
            feats = feats.to(dev)
            if image_mode == "shuffled":
                feats = feats.roll(1, 0)
            elif image_mode == "blank":
                feats = torch.zeros_like(feats)
        l, logs = model.loss(batch, a, v, None, feats=feats, n_visual=n_visual, generator=g)
        tot += logs["flow"]
        n += 1
    model.train()
    return tot / max(n, 1)


def train(cfg: dict, out_dir: Path) -> dict:
    from rrp.learning.behavior import device_setup
    dev, ginfo = device_setup()
    sig = CheckpointSignal()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    store, tr, held = load_split(cfg)
    nv = store.n_visual
    model, pcfg, rcfg = build_model(cfg, store)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("wd", 1e-4))
    total = cfg["epochs"] * max(1, len(tr) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 3e-4), total_steps=max(total, 10),
                                                pct_start=0.05)
    log = open(out_dir / "train_log.jsonl", "a")
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    step, t0 = 0, time.time()
    for epoch in range(cfg["epochs"]):
        for batch, a, v, lab, feats in batches(tr, cfg["batch_size"], rng, store):
            batch, a, v = batch.to(dev), a.to(dev), v.to(dev)
            lab = {k: t.to(dev) for k, t in lab.items()}
            loss, logs = model.loss(batch, a, v, lab if pcfg.get("aux", True) else None, feats=feats.to(dev),
                                    n_visual=nv, aux_weight=cfg.get("aux_weight", 0.1), generator=gen)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < total - 1:
                sched.step()
            step += 1
            if step % 50 == 0:
                log.write(json.dumps(dict(step=step, epoch=epoch, t=time.time() - t0, loss=float(loss.detach()),
                                          grad_norm=float(gn), **logs)) + "\n")
                log.flush()
            if sig.requested:
                break
        if sig.requested:
            break
    res = dict(steps=step, wall_s=time.time() - t0, train_samples=len(tr), heldout_samples=len(held),
               n_params=sum(p.numel() for p in model.parameters()),
               n_params_resampler=sum(p.numel() for p in model.resampler.parameters()) if model.resampler else 0,
               gpu=ginfo, interrupted=sig.requested, n_visual=nv, feature_spec=store.index["spec_key"],
               backbone=store.index["provenance"])
    res["heldout_flow_mse"] = {m: heldout_loss(model, held, store, dev, nv, m)
                               for m in (["real", "shuffled", "blank"] if model.resampler else ["real"])}
    cfg_saved = dict(cfg, policy=pcfg, resampler_cfg=rcfg)
    meta = save_checkpoint(out_dir / "policy.pt", model=model, optimizer=None, step=step,
                           versions=dict(policy=pcfg.get("name"), featurizer=FEAT_VERSION,
                                         backbone=store.index["spec_key"] if rcfg else None),
                           config=cfg_saved, extra=dict(result=res))
    res["checkpoint"] = meta
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("steps", "wall_s", "train_samples", "heldout_flow_mse")}), flush=True)
    return res


# ------------------------------------------------------------------ online VLM-backed evaluation
def make_eval_policy(ckpt: Path, device, nfe=8, execute_prefix=8, image_mode="real", vlm=None, size=256,
                     cameras=("front",)):
    from rrp.policy.runner import LearnedPolicy
    from rrp.model.backbone import Renderer, task_text

    st = load_checkpoint(ckpt, map_location=device)
    cfg = st["config"]
    model = VLMFlowPolicy(PolicyConfig(**cfg["policy"]), cfg.get("resampler_cfg")).to(device)
    model.load_state_dict(st["model"])
    model.eval()

    class VLMLearnedPolicy(LearnedPolicy):
        def __init__(self):
            super().__init__(model, None, device, nfe=nfe, execute_prefix=execute_prefix,
                             name=cfg["policy"].get("name", "vlm_policy") + ("" if image_mode == "real" else f"+{image_mode}"))
            self.renderers = {}
            self.vlm_lat, self.render_lat = [], []

        def chunks(self, sessions):
            if model.resampler is not None:
                t0 = time.perf_counter()
                frames = []
                for s in sessions:
                    k = id(s)
                    if k not in self.renderers:
                        if len(self.renderers) > 64:
                            for r in self.renderers.values():
                                r[0].close()
                            self.renderers.clear()
                        r = Renderer(s.model, size)
                        self.renderers[k] = (r, r.cameras(list(cameras)))
                    r, cams = self.renderers[k]
                    frames.append(r.render(s.data, cams))
                t1 = time.perf_counter()
                enc = vlm.encode(frames, [task_text(s) for s in sessions])
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                feats = enc["tokens"]
                if image_mode == "shuffled":
                    feats = feats.roll(1, 0)
                elif image_mode == "blank":
                    feats = torch.zeros_like(feats)
                model.pending = (feats, enc["n_visual"])
                self.render_lat.append((t1 - t0, len(sessions)))
                self.vlm_lat.append((t2 - t1, len(sessions)))
            try:
                return super().chunks(sessions)
            finally:
                model.pending = None

    return VLMLearnedPolicy(), cfg


def evaluate_main(a):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rrp.evaluation.runner import evaluate, summarize
    from rrp.ops.gpu import apply_cap
    from rrp.model.backbone import BackboneSpec, VLMBackbone, PSI0, FALLBACK
    ginfo = apply_cap()
    dev = "cuda"
    st = load_checkpoint(Path(a.checkpoint), map_location="cpu")
    cfg = st["config"]
    vlm = None
    if cfg.get("resampler_cfg"):
        bprov = st["extra"]["result"]["backbone"]
        src = PSI0 if bprov["source"]["repo_id"] == PSI0["repo_id"] else FALLBACK
        vlm = VLMBackbone(BackboneSpec(source=dict(src), taps=tuple(bprov["taps"]), n_text=bprov["n_text"]), device=dev)
        if BackboneSpec(source=dict(src), taps=tuple(bprov["taps"]), n_text=bprov["n_text"]).key() != st["versions"]["backbone"]:
            raise ValueError("backbone spec mismatch between checkpoint and online VLM")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(checkpoint=a.checkpoint, gpu=ginfo, modes={})
    for mode in a.image_modes.split(","):
        if vlm is None and mode != "real":
            continue
        torch.cuda.reset_peak_memory_stats()
        pol, _ = make_eval_policy(Path(a.checkpoint), dev, nfe=a.nfe, execute_prefix=a.prefix, image_mode=mode, vlm=vlm)
        seeds = list(range(a.seed_start, a.seed_start + a.episodes))
        per = {}
        for robot in a.robots.split(","):
            res = evaluate(pol, robot, seeds, method=f"{pol.name}", checkpoint=a.checkpoint, max_steps=a.max_steps,
                           batch=a.batch, out_path=out.with_name(out.stem + f".{mode}.jsonl"))
            per[robot] = summarize(res)
            print(mode, robot, json.dumps(per[robot]), flush=True)
        lat = {}
        if vlm is not None:
            for nm, xs in (("vlm", pol.vlm_lat), ("render", pol.render_lat)):
                call = np.array([x[0] for x in xs]) * 1e3
                per_img = np.array([x[0] / x[1] for x in xs]) * 1e3
                lat[nm] = dict(calls=len(xs), p50_ms_per_call=float(np.percentile(call, 50)),
                               p95_ms_per_call=float(np.percentile(call, 95)),
                               p50_ms_per_frame=float(np.percentile(per_img, 50)),
                               p95_ms_per_frame=float(np.percentile(per_img, 95)),
                               mean_batch=float(np.mean([x[1] for x in xs])))
        pl = np.array(pol.stats.latencies) * 1e3
        lat["policy_call_total"] = dict(p50_ms=float(np.percentile(pl, 50)), p95_ms=float(np.percentile(pl, 95)))
        summary["modes"][mode] = dict(per_robot=per, latency=lat, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        for r, _ in getattr(pol, "renderers", {}).values():
            r.close()
    if vlm is not None:
        # batch=1 live latency (the deployment case), same frames as a real call
        from rrp.model.backbone import Renderer
        from rrp.morphology.catalog import workbench_robots
        from rrp.sim.native import Session
        from rrp.sim.scenario import BUILDERS
        from rrp.model.backbone import task_text
        s = Session(BUILDERS["pick_place"](workbench_robots()[a.robots.split(",")[0]](), 7, n_distractors=2), seed=7)
        r = Renderer(s.model, 256)
        fr = [r.render(s.data, r.cameras(["front"]))]
        xs = []
        for i in range(30):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            vlm.encode(fr, [task_text(s)])
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        xs = np.array(xs[5:])
        summary["vlm_batch1_ms"] = dict(p50=float(np.percentile(xs, 50)), p95=float(np.percentile(xs, 95)), n=len(xs))
        summary["vlm_weights_bytes"] = sum(p.numel() * p.element_size() for p in vlm.model.parameters())
        summary["backbone"] = vlm.provenance()
        r.close()
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, default=str)[:3000])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--robots", required=True)
    e.add_argument("--episodes", type=int, default=16)
    e.add_argument("--seed-start", type=int, default=3000000)
    e.add_argument("--max-steps", type=int, default=300)
    e.add_argument("--nfe", type=int, default=8)
    e.add_argument("--prefix", type=int, default=8)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--image-modes", default="real,shuffled,blank")
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        cfg = json.loads(Path(a.config).read_text())
        d = Path(cfg["out_dir"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        train(cfg, d)
    else:
        evaluate_main(a)


if __name__ == "__main__":
    main()
