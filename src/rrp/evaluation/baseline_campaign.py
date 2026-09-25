"""Resumable, idempotent campaign cells for the two BASELINE methods of the sealed latent_slice1 protocol.

One call = one (method, seed, target, budget) cell:
  source training (shared per method/seed; built once under a file lock) -> [SFT on `budget` target demo episodes]
  -> closed-loop evaluation on the sealed eval scenes -> cell JSON with Wilson CI and acquisition accounting.
target="source" evaluates source competence on the held-out source bodies (50 episodes, as in run_latent_cell).
Every step is skipped when its output exists. Controller source label: learned:<checkpoint> (native ActionChunk path).

Fairness with the latent methods (run_latent_cell):
  * source data: the same stride-1 pack artifacts/packed/latent_pp_v3dart_s1_H16 (13 source bodies, DART incl.
    failures), subsampled to t % 2 == 0 rows = the stride-2 chunk set named by policy-small-structured.json;
  * adaptation: the same target packs, nested episode choice, row sampler, batch, lr 1e-4 and update counts
    {5: 150, 20: 300, 100: 600} (rrp.learning.sft.sft_packed);
  * evaluation: the same scene builder, eval seeds, max_steps, replan period (execute_prefix = replan_ticks), nfe.
"""
from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path

BASELINE_METHODS = ("baseline_direct_action", "baseline_action_only_codec")
SOURCE_PACK = "artifacts/packed/latent_pp_v3dart_s1_H16"
SFT_STEPS = {5: 150, 20: 300, 100: 600}          # identical to run_latent_cell
SFT_LR = 1e-4


def _done(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def source_config(method: str, seed: int, cell: Path, smoke: bool = False) -> dict:
    base = json.loads(Path("configs/model/policy-small-structured.json").read_text())
    cfg = dict(base, seed=seed, out_dir=str(cell / "source"), name=f"{method}_seed{seed}",
               packed_dir=SOURCE_PACK, packed_stride=base.get("stride", 2), prefetch=True, exact_resume=True,
               checkpoint_every_steps=1000, zero_prev_action=True)   # D-045: deployment-consistent input (B-1)
    cfg["policy"] = dict(base["policy"], name=f"{method}_seed{seed}")
    if method == "baseline_action_only_codec":
        ccfg = json.loads(Path("configs/model/codec-small.json").read_text())
        cfg["codec_checkpoint"] = str(cell / "codec" / "codec.pt")
        cfg["policy"]["latent_dim"] = ccfg["codec"]["latent_dim"]
    if smoke:
        cfg["epochs"] = 1
        cfg["smoke_max_steps"] = 30
    return cfg


def codec_config(seed: int, cell: Path, smoke: bool = False) -> dict:
    ccfg = json.loads(Path("configs/model/codec-small.json").read_text())
    cfg = dict(ccfg, seed=seed, out_dir=str(cell / "codec"), name=f"codec_seed{seed}")
    if smoke:
        cfg["episodes_per_robot"] = 3
        cfg["epochs"] = 1
    return cfg


def ensure_source(method: str, seed: int, cell: Path, smoke: bool = False) -> Path:
    from rrp.learning.behavior import train_codec, train_policy
    src = cell / "source"
    with _lock(cell / ".source.lock"):
        if method == "baseline_action_only_codec" and not _done(cell / "codec" / "codec.pt"):
            cc = codec_config(seed, cell, smoke)
            (cell / "codec").mkdir(parents=True, exist_ok=True)
            (cell / "codec" / "config.json").write_text(json.dumps(cc, indent=1))
            train_codec(cc, cell / "codec")
        if not _done(src / "policy.pt"):
            cfg = source_config(method, seed, cell, smoke)
            src.mkdir(parents=True, exist_ok=True)
            (src / "config.json").write_text(json.dumps(cfg, indent=1))
            res = train_policy(cfg, src)
            if res.get("interrupted"):
                raise SystemExit("source training interrupted (checkpointed; rerun resumes)")
    return src / "policy.pt"


def _source_accounting(cell: Path) -> dict:
    out = {}
    for name, f in (("source_policy", cell / "source" / "result.json"), ("codec", cell / "codec" / "result.json")):
        if f.exists():
            r = json.loads(f.read_text())
            cfg = json.loads((f.parent / "config.json").read_text())
            out[name] = dict(optimizer_updates=r.get("steps"), batch_size=cfg.get("batch_size"),
                             chunk_presentations=(r.get("steps") or 0) * cfg.get("batch_size", 0),
                             train_chunks=r.get("train_chunks"), wall_s=r.get("wall_s"),
                             data=cfg.get("packed_dir") or cfg.get("dataset"),
                             stride=cfg.get("packed_stride") or cfg.get("stride"))
    return out


def evaluate_checkpoint(ck: Path, robots: list[str], out: Path, *, protocol: dict, method: str, seed: int, n: int,
                        device: str, batch: int = 25) -> dict:
    from rrp.policy.runner import LearnedPolicy
    from rrp.evaluation.runner import evaluate, summarize
    sm = out.with_suffix(".summary.json")
    if _done(sm):
        return json.loads(sm.read_text())
    if out.exists():
        out.unlink()                  # partial rows from an interrupted run: redo this eval completely
    ev = protocol["eval"]
    pol = LearnedPolicy.from_checkpoint(ck, device=device, nfe=ev["nfe"], execute_prefix=ev["replan_ticks"], seed=seed)
    s = {}
    t0 = time.time()
    for r in robots:
        res = evaluate(pol, r, list(range(ev["seed_start"], ev["seed_start"] + n)), method=method,
                       checkpoint=str(ck), max_steps=ev["max_steps"], batch=batch, out_path=out)
        s[r] = summarize(res)
        att = [x for x in res if x.outcome != "infeasible"]
        s[r].update(policy_calls=sum(x.policy_calls for x in att), control_steps=sum(x.steps for x in att))
    s["_meta"] = dict(controller_source=f"learned:{ck}", device=device, wall_s=time.time() - t0,
                      seed_start=ev["seed_start"], episodes=n, nfe=ev["nfe"], execute_prefix=ev["replan_ticks"],
                      max_steps=ev["max_steps"])
    sm.write_text(json.dumps(s, indent=1))
    return s


def run_baseline_cell(protocol: dict, method: str, seed: int, target: str, budget: int, *,
                      root: Path = Path("artifacts/runs/latent_slice1"), target_packed: str = "artifacts/packed/latent_targets",
                      eval_device: str | None = None, train_only: bool = False, smoke: bool = False) -> dict:
    import torch
    if method not in BASELINE_METHODS:
        raise ValueError(f"not a baseline method: {method}")
    if seed not in protocol["seeds"] and not smoke:
        raise ValueError(f"seed {seed} not in the sealed protocol")
    if target != "source" and (target not in protocol["targets"] or budget not in protocol["sft_budgets"]):
        raise ValueError(f"cell {target}/{budget} not in the sealed protocol")
    cell = root / method / f"seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    dev = eval_device or ("cuda" if torch.cuda.is_available() else "cpu")
    src_ck = ensure_source(method, seed, cell, smoke)
    ck, sft_res = src_ck, None
    if target != "source" and budget > 0:
        from rrp.learning.sft import sft_packed
        d = cell / "sft" / f"{target}_b{budget}"
        with _lock(cell / f".sft_{target}_b{budget}.lock"):
            if not _done(d / "policy.pt"):
                sft_packed(src_ck, Path(target_packed) / target, budget, seed=seed, out_dir=d,
                           steps=(10 if smoke else SFT_STEPS[budget]), lr=SFT_LR)
        ck = d / "policy.pt"
        sft_res = json.loads((d / "result.json").read_text())
    if train_only:
        return dict(method=method, seed=seed, target=target, budget=budget, checkpoint=str(ck), trained=True)
    tag = "source" if target == "source" else f"{target}_b{budget}"
    robots = protocol["source_heldout_eval_robots"] if target == "source" else [target]
    n = 3 if smoke else (50 if target == "source" else protocol["eval"]["episodes"])
    summ = evaluate_checkpoint(ck, robots, cell / "eval" / f"{tag}.jsonl", protocol=protocol, method=method,
                               seed=seed, n=n, device=dev)
    per = {r: summ[r] for r in robots}
    k = sum(v["successes"] for v in per.values())
    att = sum(v["attempted"] for v in per.values())
    from rrp.evaluation.statistics import wilson
    res = dict(protocol=protocol["id"], protocol_status=protocol["status"], method=method, seed=seed, target=target,
               budget=budget, smoke=smoke, controller_source=f"learned:{ck}", action_path="native_ActionChunk",
               kind=("source_competence" if target == "source" else
                     "zero_shot_transfer_of_source_controller" if budget == 0 else "target_sft_adaptation"),
               successes=k, attempted=att, success_rate=(k / att if att else None), wilson95=list(wilson(k, att)),
               per_robot=per, eval_meta=summ.get("_meta"),
               acquisition=dict(target_demo_episodes=(sft_res or {}).get("demo_episodes", 0),
                                target_demo_transitions=(sft_res or {}).get("demo_control_transitions", 0),
                                target_optimizer_updates=(sft_res or {}).get("optimizer_updates", 0),
                                target_chunk_presentations=(sft_res or {}).get("chunk_presentations", 0),
                                source_pretraining=_source_accounting(cell)),
               sft=sft_res, eval_rows=str(cell / "eval" / f"{tag}.jsonl"))
    (cell / "cells").mkdir(exist_ok=True)
    (cell / "cells" / f"{tag}.json").write_text(json.dumps(res, indent=1))
    return res
