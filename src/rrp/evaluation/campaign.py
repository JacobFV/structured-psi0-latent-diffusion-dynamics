"""Resumable primary campaign (P20/P21). One call = one (method, seed) cell chain:
source training -> source-competence eval -> zero-shot target eval -> SFT(5/20/100) -> eval.
Each step is skipped if its output exists (no duplicate runs after a restart)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from rrp.evaluation.registry import ExperimentRegistry


def _done(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def run_cell(protocol: dict, method: str, seed: int, *, root: Path = Path("artifacts/runs/primary")) -> dict:
    from rrp.learning.behavior import train_policy
    from rrp.learning.sft import sft
    from rrp.policy.runner import LearnedPolicy
    from rrp.evaluation.runner import evaluate, summarize
    reg = ExperimentRegistry("research/registry.jsonl")
    mcfg = json.loads(Path(protocol["methods"][method]).read_text())
    cell = root / method / f"seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    run_id = f"{protocol['id']}/{method}/seed{seed}"
    reg.register(run_id, dict(protocol=protocol, method_config=mcfg, seed=seed), sealed=True,
                 stage=protocol.get("stage", "primary_confirmation"), question=protocol["question"])
    reg.update(run_id, "running")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = cell / "source"
    if not _done(src / "policy.pt"):
        cfg = dict(mcfg, seed=seed, out_dir=str(src))
        src.mkdir(parents=True, exist_ok=True)
        (src / "config.json").write_text(json.dumps(cfg, indent=1))
        train_policy(cfg, src)
    summ = {}
    ev = protocol["eval"]
    def eval_ckpt(ck: Path, robots: list[str], tag: str, n: int, seed0: int):
        out = cell / "eval" / f"{tag}.jsonl"
        sm = out.with_suffix(".summary.json")
        if _done(sm):
            return json.loads(sm.read_text())
        if out.exists():
            out.unlink()      # partial file from an interrupted run: redo this cell completely
        pol = LearnedPolicy.from_checkpoint(ck, device=dev, nfe=ev["nfe"], execute_prefix=ev["prefix"], seed=seed)
        s = {}
        for r in robots:
            res = evaluate(pol, r, list(range(seed0, seed0 + n)), method=method, checkpoint=str(ck),
                           max_steps=ev["max_steps"], batch=ev["batch"], out_path=out)
            s[r] = summarize(res)
        sm.write_text(json.dumps(s, indent=1))
        return s
    summ["source_competence"] = eval_ckpt(src / "policy.pt", protocol["source_eval_robots"], "source_competence",
                                          ev["source_episodes"], ev["seed_start"])
    for target in protocol["targets"]:
        summ[f"{target}/0"] = eval_ckpt(src / "policy.pt", [target], f"{target}_b0", ev["episodes"], ev["seed_start"])
        for b in protocol["budgets"]:
            d = cell / "sft" / f"{target}_b{b}"
            if not _done(d / "policy.pt"):
                sft(src / "policy.pt", Path(mcfg["dataset"]), target, b, seed=seed, out_dir=d,
                    steps=protocol["sft"]["steps_for_budget"][str(b)], lr=protocol["sft"]["lr"],
                    modules=protocol["sft"]["modules"])
            summ[f"{target}/{b}"] = eval_ckpt(d / "policy.pt", [target], f"{target}_b{b}", ev["episodes"],
                                              ev["seed_start"])
            if protocol.get("retention") and b == max(protocol["budgets"]):
                summ[f"{target}/{b}/retention"] = eval_ckpt(d / "policy.pt", protocol["source_eval_robots"][:1],
                                                            f"{target}_b{b}_retention", ev["retention_episodes"],
                                                            ev["seed_start"])
    (cell / "summary.json").write_text(json.dumps(summ, indent=1))
    reg.update(run_id, "completed", summary_path=str(cell / "summary.json"))
    return summ
