"""Resumable campaign cell for the sealed latent_slice1 protocol (latent methods). Each step is skipped when its
outputs exist. The representation (latent meaning + system 0) is FROZEN per variant; seeds vary system i."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from rrp.evaluation.registry import ExperimentRegistry


def _done(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def run_latent_cell(protocol: dict, method: str, seed: int, *, base_flow_config: str, target_packed: str,
                    root: Path = Path("artifacts/runs/latent_slice1")) -> dict:
    from rrp.learning.latent_train import train_latent_flow, sft_latent_flow, load_representation
    from rrp.policy.latent_runner import LatentPolicy
    from rrp.evaluation.latent_eval import evaluate_latent
    from rrp.evaluation.statistics import wilson
    reg = ExperimentRegistry("research/registry.jsonl")
    cell = root / method / f"seed{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    run_id = f"{protocol['id']}/{method}/seed{seed}"
    base = json.loads(Path(base_flow_config).read_text())
    reg.register(run_id, dict(protocol_hash=protocol.get("id"), method=method, seed=seed, base_flow_config=base),
                 sealed=True, stage="primary_confirmation", question=protocol["question"])
    reg.update(run_id, "running")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = cell / "source"
    if not _done(src / "policy.pt"):
        cfg = dict(base, seed=seed, out_dir=str(src))
        src.mkdir(parents=True, exist_ok=True)
        (src / "config.json").write_text(json.dumps(cfg, indent=1))
        train_latent_flow(cfg, src)
    ev = protocol["eval"]
    _, _, R, P, _ = load_representation(Path(base["representation"]), dev)

    def eval_ckpt(ck: Path, robots, tag, n):
        out = cell / "eval" / f"{tag}.jsonl"
        sm = out.with_suffix(".summary.json")
        if _done(sm):
            return json.loads(sm.read_text())
        if out.exists():
            out.unlink()
        pol = LatentPolicy.from_checkpoint(ck, device=dev, nfe=ev["nfe"], seed=seed)
        s = {}
        for r in robots:
            res = evaluate_latent(pol, R, P, r, list(range(ev["seed_start"], ev["seed_start"] + n)), method=method,
                                  replan_ticks=ev["replan_ticks"], max_steps=ev["max_steps"], batch=25, out_path=out,
                                  device=dev)
            att = [x for x in res if x.outcome != "infeasible"]
            k = sum(x.privileged_success for x in att)
            s[r] = dict(attempted=len(att), successes=k, wilson95=wilson(k, len(att)),
                        system_i_calls=sum(x.system_i_calls for x in att), system0_ticks=sum(x.system0_ticks for x in att))
        sm.write_text(json.dumps(s, indent=1))
        return s

    summ = {"source_competence": eval_ckpt(src / "policy.pt", protocol["source_heldout_eval_robots"], "source", 50)}
    budgets = [b for b in protocol["sft_budgets"] if b > 0]
    for tgt in protocol["targets"]:
        summ[f"{tgt}/0"] = eval_ckpt(src / "policy.pt", [tgt], f"{tgt}_b0", ev["episodes"])
        for b in budgets:
            d = cell / "sft" / f"{tgt}_b{b}"
            if not _done(d / "policy.pt"):
                sft_latent_flow(src / "policy.pt", Path(target_packed) / tgt, b, seed=seed, out_dir=d,
                                steps={5: 150, 20: 300, 100: 600}[b])
            summ[f"{tgt}/{b}"] = eval_ckpt(d / "policy.pt", [tgt], f"{tgt}_b{b}", ev["episodes"])
    (cell / "summary.json").write_text(json.dumps(summ, indent=1))
    reg.update(run_id, "completed", summary_path=str(cell / "summary.json"))
    return summ
