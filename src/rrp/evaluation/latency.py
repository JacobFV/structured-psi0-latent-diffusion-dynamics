"""Synchronized latency/memory measurement (P23). Run only when no other GPU lease is active;
the measurement records the broker's active-lease list as its external-load context."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from rrp.model.batch import collate_inputs
from rrp.model.flow import FlowPolicy, PolicyConfig


def _pct(xs):
    a = np.array(xs) * 1000
    return dict(p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                mean=float(a.mean()), n=len(a))


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def replicate_nodes(pi, n_target: int):
    """Synthetic larger body for shape/latency sweeps: tile action nodes (kinematic chain of copies)."""
    p = copy.deepcopy(pi)
    n0 = pi.act_node_feats.shape[0]
    reps = int(np.ceil(n_target / n0))
    p.act_node_feats = np.tile(pi.act_node_feats, (reps, 1))[:n_target]
    m = pi.tokens["morph"]
    extra = np.tile(m[:n0], (reps, 1))[:n_target]
    p.tokens["morph"] = np.concatenate([extra, m[n0:]], 0)
    p.token_kind["morph"] = np.concatenate([np.zeros(n_target, int), pi.token_kind["morph"][n0:]])
    p.pointer_text["morph"] = np.zeros((len(p.tokens["morph"]), pi.pointer_text["morph"].shape[1]), np.float32)
    rel = [r for r in pi.relations if not (r[0] == -1 or (r[0] == 0 and r[1] < n0) or (r[2] == 0 and r[3] < n0))]
    shift = n_target - n0
    rel = [[q, (qi + shift if q == 0 else qi), k, (ki + shift if k == 0 else ki), r] for q, qi, k, ki, r in rel]
    for i in range(n_target):
        rel.append([-1, i, 0, i, 0])
        if i > 0:
            rel.append([-1, i, 0, i - 1, 16])
    p.relations = np.array(rel, np.int64).reshape(-1, 5)
    p.pointers = np.array([[a, (b + shift if a == 0 else b), c, (d + shift if c == 0 else d)] for a, b, c, d in
                           pi.pointers], np.int64).reshape(-1, 4)
    p.q0 = np.zeros(n_target, np.float32)
    return p


@torch.no_grad()
def bench_policy(model: FlowPolicy, pi, dev, *, nfe_list=(1, 2, 4, 8, 16), reps: int = 50, batch: int = 1,
                 cached: bool = True) -> dict:
    model.eval()
    b = collate_inputs([pi] * batch).to(dev)
    H = model.cfg.horizon
    out = {}
    for nfe in nfe_list:
        # warmup
        for _ in range(3):
            c = model.prepare(b)
            model.sample(c, H, nfe=nfe)
        _sync(dev)
        prep, samp, tot = [], [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            c = model.prepare(b)
            _sync(dev)
            t1 = time.perf_counter()
            if cached:
                model.sample(c, H, nfe=nfe)
            else:   # uncached: rebuild context (and its K/V) at every function evaluation
                z = torch.randn(b.B, H, b.node_feats.shape[1], model.cfg.latent_dim, device=dev)
                for k in range(nfe):
                    cc = model.prepare(b)
                    z = z + (1.0 / nfe) * model.velocity(z, torch.full((b.B,), k / nfe, device=dev), cc)
            _sync(dev)
            t2 = time.perf_counter()
            prep.append(t1 - t0)
            samp.append(t2 - t1)
            tot.append(t2 - t0)
        out[nfe] = dict(prepare=_pct(prep), sample=_pct(samp), total=_pct(tot))
    return out


def run_latency_suite(checkpoints: dict[str, str], out_path: Path, dev=None, node_counts=(7, 16, 32, 64, 128)):
    from rrp.policy.runner import LearnedPolicy
    from rrp.sim.fixtures import make_pick_place_session
    from rrp.data.collect import featurizer_for
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = make_pick_place_session(seed=5, n_distractors=2)
    f = featurizer_for(s)
    for _ in range(10):
        s.step(None)
    obs = s.observe()
    pi = f(obs)
    res = dict(device=str(dev), torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
               t=time.time(), models={})
    for name, ck in checkpoints.items():
        pol = LearnedPolicy.from_checkpoint(ck, device=dev)
        m = pol.model
        entry = dict(params=sum(p.numel() for p in m.parameters()), cfg=m.cfg.__dict__)
        entry["cached"] = bench_policy(m, pi, dev)
        entry["uncached"] = bench_policy(m, pi, dev, nfe_list=(1, 8), cached=False, reps=20)
        dense = FlowPolicy(PolicyConfig(**dict(m.cfg.__dict__, attention="dense"))).to(dev)
        dense.load_state_dict(m.state_dict())
        entry["dense_action_attention"] = bench_policy(dense, pi, dev, nfe_list=(8,), reps=20)
        entry["node_sweep_nfe8"] = {n: bench_policy(m, replicate_nodes(pi, n), dev, nfe_list=(8,), reps=20)[8]
                                    for n in node_counts}
        entry["batch64_nfe8"] = bench_policy(m, pi, dev, nfe_list=(8,), reps=10, batch=64)[8]
        # end-to-end observation -> native command (featurize + collate + prepare + sample + decode)
        e2e = []
        for _ in range(30):
            t0 = time.perf_counter()
            ch = pol.chunks([s])
            _sync(dev)
            e2e.append(time.perf_counter() - t0)
        entry["end_to_end_obs_to_chunk"] = _pct(e2e)
        if dev.type == "cuda":
            entry["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        res["models"][name] = entry
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1))
    return res


@torch.no_grad()
def latent_latency_suite(flow_ckpt: str, out_path: Path, dev=None, nfe_list=(1, 2, 4, 8), reps=40,
                         direct_ckpt: str | None = None) -> dict:
    """R38 test 12: synchronized timings of the actual corrected inference path.
    system i: observe+featurize+collate+prepare+sample+packet construction (per replan, NFE sweep)
    system 0: per-tick realization (featurize local state + realizer forward + denormalize) vs 50 ms deadline
    end-to-end: observation -> first native command after a replan."""
    from rrp.policy.latent_runner import LatentPolicy
    from rrp.learning.latent_train import load_representation
    from rrp.learning.checkpoint import load_checkpoint
    from rrp.control.latent_realizer import LatentSystem0
    from rrp.sim.fixtures import make_pick_place_session
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep = load_checkpoint(flow_ckpt, map_location="cpu")["config"]["representation"]
    _, _, R, _, _ = load_representation(Path(rep), dev)
    s = make_pick_place_session(seed=5, n_distractors=2)
    for _ in range(10):
        s.step(None)
    res = dict(device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, t=time.time(),
               system_i={}, note="physics paused during inference (offline loop): these are compute latencies, "
                                 "not a real-time claim")
    for nfe in nfe_list:
        pol = LatentPolicy.from_checkpoint(flow_ckpt, device=dev, nfe=nfe)
        for _ in range(3):
            pol.packets([s])
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            p = pol.packets([s])[0]
            _sync(dev)
            ts.append(time.perf_counter() - t0)
        res["system_i"][nfe] = _pct(ts)
    s0 = LatentSystem0(R, pol.featurizer(s), latent_space_version=pol.lsv, realizer_compat_version=pol.rcv, device=dev)
    s0.receive(p, now=float(s.data.time), graph_version=s.runtime.graph_version)
    tk = []
    for _ in range(reps * 2):
        t0 = time.perf_counter()
        s0.tick(s, s.controller_version())
        _sync(dev)
        tk.append(time.perf_counter() - t0)
    res["system0_tick"] = _pct(tk)
    res["system0_deadline_s"] = 0.05
    res["system0_deadline_misses"] = int(sum(t > 0.05 for t in tk))
    e2e = []
    for _ in range(reps):
        t0 = time.perf_counter()
        q = pol.packets([s])[0]
        s0.receive(q, now=float(s.data.time), graph_version=s.runtime.graph_version)
        s0.tick(s, s.controller_version())
        _sync(dev)
        e2e.append(time.perf_counter() - t0)
    res["end_to_end_obs_to_first_command"] = _pct(e2e)
    if direct_ckpt:
        from rrp.policy.runner import LearnedPolicy
        dp = LearnedPolicy.from_checkpoint(direct_ckpt, device=dev)
        for _ in range(3):
            dp.chunks([s])
        dd = []
        for _ in range(reps):
            t0 = time.perf_counter()
            dp.chunks([s])
            _sync(dev)
            dd.append(time.perf_counter() - t0)
        res["baseline_direct_action_obs_to_chunk"] = _pct(dd)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1))
    return res
