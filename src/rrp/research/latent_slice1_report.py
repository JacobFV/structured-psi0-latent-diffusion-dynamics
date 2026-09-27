"""Aggregate the sealed latent_slice1 four-way comparison from raw campaign outputs.

Reads, for every method directory under the campaign root (baselines: run_baseline_cell; latent methods:
run_latent_cell -- both write eval/<tag>.summary.json with per-robot attempted/successes, and sft/<tag>/result.json),
so the latent methods drop in without changes. Per method x target x budget: pooled success over seeds with a Wilson
95% CI, per-seed rates, and acquisition accounting (target demo episodes / control transitions / optimizer updates;
source pre-training updates reported separately). Budget 0 = zero-shot transfer of the source-trained controller;
budgets > 0 = supervised target adaptation of that controller.
"""
from __future__ import annotations

import json
from pathlib import Path

from rrp.evaluation.adaptation import first_sustained_crossing
from rrp.evaluation.statistics import wilson

METHOD_ORDER = ["baseline_direct_action", "baseline_action_only_codec", "latent_nosem", "latent_sem"]


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def collect(protocol: dict, root: Path) -> dict:
    rows = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        method = mdir.name
        for sdir in sorted(mdir.glob("seed*")):
            seed = int(sdir.name[4:])
            if seed not in protocol["seeds"]:
                continue
            src_res = _load(sdir / "source" / "result.json") or {}
            src_cfg = _load(sdir / "source" / "config.json") or {}
            src_updates = src_res.get("steps")
            src_bs = src_cfg.get("batch_size", 128)
            for tgt in ["source"] + protocol["targets"]:
                for b in ([0] if tgt == "source" else protocol["sft_budgets"]):
                    tag = "source" if tgt == "source" else f"{tgt}_b{b}"
                    sm = _load(sdir / "eval" / f"{tag}.summary.json")
                    if not sm:
                        continue
                    per = {r: v for r, v in sm.items() if not r.startswith("_")}
                    k = sum(v["successes"] for v in per.values())
                    n = sum(v["attempted"] for v in per.values())
                    sft = _load(sdir / "sft" / tag / "result.json") if b > 0 else None
                    rows[(method, tgt, b, seed)] = dict(
                        k=k, n=n, demo_episodes=(sft or {}).get("demo_episodes", 0),
                        demo_transitions=(sft or {}).get("demo_control_transitions", 0),
                        sft_updates=(sft or {}).get("optimizer_updates", 0),
                        source_updates=src_updates, source_chunk_presentations=(src_updates or 0) * src_bs)
    return rows


def aggregate(protocol: dict, root: Path) -> dict:
    rows = collect(protocol, root)
    methods = [m for m in METHOD_ORDER if any(r[0] == m for r in rows)] + \
        sorted({r[0] for r in rows} - set(METHOD_ORDER))
    table = []
    for m in methods:
        for tgt in ["source"] + protocol["targets"]:
            for b in ([0] if tgt == "source" else protocol["sft_budgets"]):
                per = {s: rows[(m, tgt, b, s)] for s in protocol["seeds"] if (m, tgt, b, s) in rows}
                if not per:
                    continue
                k = sum(v["k"] for v in per.values())
                n = sum(v["n"] for v in per.values())
                one = next(iter(per.values()))
                table.append(dict(method=m, target=tgt, budget=b, seeds_done=sorted(per), successes=k, attempted=n,
                                  rate=(k / n if n else None), wilson95=list(wilson(k, n)),
                                  per_seed={s: (v["k"], v["n"]) for s, v in per.items()},
                                  demo_episodes=one["demo_episodes"],
                                  demo_transitions_per_seed={s: v["demo_transitions"] for s, v in per.items()},
                                  sft_updates=one["sft_updates"],
                                  source_updates_per_seed={s: v["source_updates"] for s, v in per.items()}))
    crossings = []
    for m in methods:
        for tgt in protocol["targets"]:
            for s in protocol["seeds"]:
                bs = protocol["sft_budgets"]
                if all((m, tgt, b, s) in rows for b in bs):
                    rates = [rows[(m, tgt, b, s)]["k"] / max(rows[(m, tgt, b, s)]["n"], 1) for b in bs]
                    c = first_sustained_crossing(bs, rates, protocol["threshold"],
                                                 protocol.get("sustained_requires_next_checkpoint", True))
                    crossings.append(dict(method=m, target=tgt, seed=s, budget=c.budget, censored=c.censored,
                                          provisional=c.provisional))
    return dict(protocol=protocol["id"], status=protocol["status"], root=str(root), table=table, crossings=crossings)


def _ci(r):
    return f"[{r['wilson95'][0]:.2f}, {r['wilson95'][1]:.2f}]"


def to_markdown(agg: dict, protocol: dict) -> str:
    L = [f"Protocol `{agg['protocol']}` ({agg['status']}); raw outputs under `{agg['root']}`.",
         "Rate = successes / attempted (infeasible scenes excluded), pooled over seeds; Wilson 95% CI. "
         "Eval scenes seed_start 2,000,000; source competence = held-out source bodies, 50 episodes per seed.", "",
         "| method | target | budget | kind | successes/attempted | rate | Wilson 95% | per seed (k/n) | target demo eps | "
         "target transitions (per seed) | SFT updates | source updates (per seed) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg["table"]:
        kind = "source competence" if r["target"] == "source" else ("zero-shot transfer" if r["budget"] == 0 else "SFT")
        ps = " ".join(f"{s}:{k}/{n}" for s, (k, n) in r["per_seed"].items())
        tr = " ".join(str(v) for v in r["demo_transitions_per_seed"].values())
        su = " ".join(str(v) for v in r["source_updates_per_seed"].values())
        L.append(f"| {r['method']} | {r['target']} | {r['budget']} | {kind} | {r['successes']}/{r['attempted']} | "
                 f"{(r['rate'] or 0):.3f} | {_ci(r)} | {ps} | {r['demo_episodes']} | {tr} | {r['sft_updates']} | {su} |")
    if agg["crossings"]:
        L += ["", f"First sustained crossing of {protocol['threshold']} (requires next checkpoint):", "",
              "| method | target | seed | budget | censored | provisional |", "|---|---|---|---|---|---|"]
        for c in agg["crossings"]:
            L.append(f"| {c['method']} | {c['target']} | {c['seed']} | {c['budget']} | {c['censored']} | {c['provisional']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="configs/eval/latent_slice1.json")
    ap.add_argument("--root", default="artifacts/runs/latent_slice1")
    ap.add_argument("--out-json", default="artifacts/runs/latent_slice1/aggregate.json")
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args(argv)
    protocol = json.loads(Path(a.protocol).read_text())
    agg = aggregate(protocol, Path(a.root))
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(agg, indent=1))
    md = to_markdown(agg, protocol)
    if a.out_md:
        Path(a.out_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
