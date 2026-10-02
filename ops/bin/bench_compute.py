#!/usr/bin/env python3
"""Per-trainer step-time benchmark of the `compute` block (unit prec, research/tracks/compute.md).

Runs each REAL trainer on REAL data from the shared store for a handful of optimizer steps under each compute setting and reports
the median wall time of one optimizer step (cuda-synchronised; the first WARM steps, which hold compile / autotune cost, are
reported separately), peak CUDA memory and the stamp's compile status. A global optimizer post-step hook counts the steps and aborts
the trainer after N, so the trainers themselves are untouched and write nothing outside the scratch dir.

  RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/accel-prec ops/bin/peer_run.sh --gpu --gpu-mem 12G --cpu 4 --mem 24G \
      --label prec_bench --max-seconds 1200 -- PY ops/bin/bench_compute.py --out /dev/shm/rrp-brandonin/bench_prec.json
Settings are interleaved per trainer (default, fp32, bf16, bf16+compile) so that drift of a shared GPU hits all of them alike.
The configs of in-flight runs are READ (copied with out_dir redirected); nothing under the store is written.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from rrp.core import compute
from rrp.core.compute import Compute

STORE = Path("artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SETTINGS = {                                          # name -> Compute (None precision = the trainer's historical one)
    "default": Compute(),
    "fp32": Compute(precision="fp32", tf32=False),
    "tf32": Compute(precision="fp32", tf32=True),
    "bf16": Compute(precision="bf16"),
    "bf16+compile": Compute(precision="bf16", compile="reduce-overhead", cuda_graphs=True),
    "bf16+compile-nograph": Compute(precision="bf16", compile="max-autotune"),
}
LEGGED_FACTORS = [{"name": f"probe.legged.{k}", "weight": 1.0} for k in ("contact", "goal", "disp", "subtask", "fall")] + ["preset:legged"]
BIG = 10 ** 9


class _Stop(Exception):
    pass


class _Timer:
    def __init__(self, n: int):
        self.n, self.t = n, []

    def __call__(self, opt, args, kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t.append(time.perf_counter())
        if len(self.t) > self.n:
            raise _Stop


def _json(p):
    return json.loads(Path(p).read_text())


# ---- cases: name -> (out_dir -> thunk running the trainer). Steps are huge: the timer aborts the run.
def arm_rep(out):
    from rrp.harness.train.latent_train import train_representation
    c = _json(STORE / "runs/armdiv/arm8div-semfix/train_rep_s2/config.json")
    c["out_dir"] = str(out)
    return lambda: train_representation(c, out)


def arm_flow(out):
    from rrp.harness.train.latent_train import train_latent_flow
    c = _json(STORE / "runs/relations/relations-geo/train_flow_s2/config.json")
    c.update(out_dir=str(out), snapshot_every=BIG)
    return lambda: train_latent_flow(c, out)


def arm_bc(out):
    from rrp.harness.train.behavior import train_policy
    c = _json(STORE / "runs/armdiv/bcv7divkf-1701/train_bc-bc1701_s1701/config.json")
    c.update(out_dir=str(out), checkpoint_every_steps=BIG)
    return lambda: train_policy(c, out)


def _legged_rep_cfg(steps, bs):
    return dict(name="bench", data=str(STORE / "datasets/legged_latent_v2"), bodies=["g1"], steps=steps, batch_size=bs, ckpt_every=BIG,
                snap_every=BIG, latent=dict(dz=32, width=192, factors=LEGGED_FACTORS, beta_kl=1e-3))


def legged_rep(out):
    from rrp.harness.train.legged_latent_train import train_rep
    return lambda: train_rep(_legged_rep_cfg(BIG, 256), out)


def legged_flow(out, scratch):
    from rrp.harness.train.legged_latent_train import train_flow
    rep = scratch / "legged_rep_seed" / "representation.pt"
    if not rep.exists():                                 # a 30-step representation, trained once under the default setting
        from rrp.harness.train.legged_latent_train import train_rep
        compute.configure(Compute())
        train_rep(_legged_rep_cfg(30, 64), rep.parent)
    c = dict(representation=str(rep), steps=BIG, batch_size=256, width=192, layers=2, semantic_weight=0.5, ckpt_every=BIG, snap_every=BIG)
    return lambda: train_flow(c, out)


def legged_bc(out):
    from rrp.harness.train.legged_bc import train
    c = dict(name="bench_bc", data=str(STORE / "datasets/legged_latent_v2"), bodies=["g1"], steps=BIG, batch_size=256,
             ckpt_every=BIG, snap_every=BIG, model=dict(width=192, enc_layers=2, dec_layers=2, factors=["preset:legged"]))
    return lambda: train(c, out)


def _pointer_args(out, **kw):
    from rrp.harness.train.pointer.split import SPLIT_PATH_V2
    packs = sorted(str(p) for p in (STORE / "runs/pointer/pointer-v2-data/collect_s0").glob("*.npz"))
    return argparse.Namespace(**dict(dict(data=packs, steps=BIG, batch=128, lr=3e-4, seed=0, split_seed=0, device=DEV, log_every=BIG,
                                          split=SPLIT_PATH_V2, factors=None, out=str(out / "m.pt"), variant="semfix", dz=16, w_sem=1.0,
                                          lv_min=-4.0, beta=3e-3, w_xy=5.0, target="latent"), **kw))


def pointer(stage):
    def f(out, scratch):
        from rrp.harness.train.pointer import train as T
        rep = scratch / "pointer_rep_seed" / "m.pt"
        if stage == "flow" and not rep.exists():          # a 30-step representation, trained once under the default setting
            compute.configure(Compute())
            T.cmd_rep(_pointer_args(rep.parent, steps=30, batch=64))
        a = _pointer_args(out, representation=str(rep))
        return lambda: getattr(T, f"cmd_{stage}")(a)
    return f


CASES = dict(arm_rep=arm_rep, arm_flow=arm_flow, arm_bc=arm_bc, legged_rep=legged_rep, legged_flow=legged_flow, legged_bc=legged_bc,
             pointer_rep=pointer("rep"), pointer_flow=pointer("flow"), pointer_bc=pointer("bc"))


def run_case(name, setting, n, warm, scratch):
    import inspect
    import torch._dynamo
    torch._dynamo.reset()
    torch.backends.cuda.matmul.allow_tf32 = False          # every setting starts from torch's defaults, not the previous run's
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    compute.configure(SETTINGS[setting])
    out = scratch / f"{name}-{setting}"
    out.mkdir(parents=True, exist_ok=True)
    make = CASES[name]
    tm = _Timer(n)
    t0, status, err = time.perf_counter(), "ok", None
    h = None
    try:
        compute.configure(Compute())                         # building (seed models, data) is not timed or compiled
        fn = make(out, scratch) if len(inspect.signature(make).parameters) == 2 else make(out)
        compute.configure(SETTINGS[setting])
        h = register_optimizer_step_post_hook(tm)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        fn()
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001 - one broken case must not end the benchmark
        status, err = "error", f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        if h is not None:
            h.remove()
    dts = [b - a for a, b in zip(tm.t, tm.t[1:])]
    cx = compute.active()
    rec = dict(trainer=name, setting=setting, status=status, error=err, steps_timed=max(0, len(dts) - warm),
               warm_s=round(sum(dts[:warm]), 2) if len(dts) > warm else None,
               peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
               effective=cx.stamp()["effective"] if cx else None, compiled=cx.compiled if cx else None,
               total_s=round(time.perf_counter() - t0, 1))
    if len(dts) > warm:
        d = sorted(dts[warm:])
        rec.update(step_ms_median=round(1000 * statistics.median(d), 2), step_ms_p10=round(1000 * d[len(d) // 10], 2),
                   step_ms_p90=round(1000 * d[(9 * len(d)) // 10], 2))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=",".join(c for c in CASES if not c.startswith("legged")),
                    help="legged_* need a pack in the CURRENT legged format (the v1/v2 packs on the peer predate it: goal_columns fails)")
    ap.add_argument("--settings", default="default,fp32,bf16,bf16+compile")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--budget-s", type=float, default=1000, help="stop starting new runs after this many seconds")
    a = ap.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="bench_prec_", dir=os.environ.get("BENCH_SCRATCH", "/dev/shm/rrp-brandonin")))
    results, t_start = [], time.time()
    for name in a.cases.split(","):
        for s in a.settings.split(","):
            if time.time() - t_start > a.budget_s:
                results.append(dict(trainer=name, setting=s, status="skipped_budget"))
                continue
            r = run_case(name, s, a.steps, a.warm, scratch)
            results.append(r)
            print(json.dumps(r), flush=True)
            Path(a.out).write_text(json.dumps(dict(
                host=os.uname().nodename, torch=torch.__version__, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                steps=a.steps, warm=a.warm, results=results), indent=1))
    sys.stdout.flush()
    os._exit(0)          # prefetch worker threads of the trainers may still be parked


if __name__ == "__main__":
    main()
