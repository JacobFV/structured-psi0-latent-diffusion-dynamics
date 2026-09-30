"""`rrp run-dag <recipe.yaml>`: plan / run / resume a pipeline DAG through the broker (rrp.orchestration.dag)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def cmd_run_dag(a):
    from rrp.core.provenance import repo_root
    from rrp.harness.dag import (DagError, Executor, Ledger, OpsRunner, default_ledger_path, format_plan,
                                       load_dag, plan_dag, resolve_recipe)
    from rrp.harness.pipelines.base import _load_families, stage_versions
    _load_families()                       # stages a recipe names (relations_data, extension stages) register on import
    root = Path(a.root).resolve() if a.root else repo_root()
    try:
        recipe = resolve_recipe(a.dag, root)
    except DagError as e:
        raise SystemExit(str(e))
    plan = plan_dag(load_dag(recipe), source=str(recipe))
    points = [dict(kv.split("=", 1) for kv in p.split(",")) for p in (a.point or [])]
    if a.only or points:
        plan = plan.select(a.only, points)
    runner = OpsRunner(root, peer_repo=a.peer_repo)
    ledger_path = Path(a.ledger) if a.ledger else default_ledger_path(root, plan)
    ledger = Ledger(ledger_path)
    cross = [f"{nid} <- {d}" for nid, n in plan.nodes.items() for d in n.deps
             if plan.nodes[d].placement != n.placement]
    if a.show_config:
        n = plan.nodes[a.show_config]
        from rrp.core.runconfig import RunIndex
        print(json.dumps(dict(runconfig=n.rc.model_dump(mode="json"),
                              native=n.rc.to_native(RunIndex.load(root=root))), indent=1))
        return 0
    if a.dry_run:
        print(format_plan(plan, ledger, runner))
        print(f"ledger: {ledger_path}")
        if cross:
            print("WARNING: cross-placement edges (artifacts are NOT transferred automatically; pre-sync them): "
                  + "; ".join(cross[:10]))
        return 0
    if cross and not a.allow_cross_placement:
        raise SystemExit("refusing: cross-placement dependencies " + "; ".join(cross[:5]) +
                         " (artifacts are not transferred automatically; use one placement or --allow-cross-placement)")
    ledger.lock()
    for nid in a.reset or []:
        if nid not in plan.nodes:
            raise SystemExit(f"--reset: unknown node {nid}")
        e = ledger.node(nid)
        if e.get("state") == "running":
            raise SystemExit(f"--reset {nid}: node is running (lease {e['attempts'][-1].get('lease_id')}); "
                             "run-dag never stops leases")
        ledger.data["nodes"][nid] = dict(state="planned", attempts=[], reset_from=e)
    if a.retry_failed:
        for nid in plan.order:
            e = ledger.node(nid)
            if e["state"] in ("failed", "blocked"):
                e.update(state="planned", attempts=[], previous_attempts=e.get("attempts", []))
    ledger.save()
    ex = Executor(plan, ledger, runner, max_parallel=a.max_parallel or int(plan.defaults.get("max_parallel", 4)),
                  poll_s=a.poll, admission_timeout_s=float(plan.defaults.get("admission_timeout_s", 10800)),
                  max_parallel_gpu=_opt(a.max_parallel_gpu, plan.defaults.get("max_parallel_gpu"), int),
                  max_cpu=_opt(a.max_cpu, plan.defaults.get("max_cpu"), float),
                  max_mem_gib=_opt(a.max_mem_gib, plan.defaults.get("max_mem_gib"), float),
                  budget_dir=ledger_path.parent.parent if plan.defaults.get("shared_budget") else None,
                  adopt_stale=a.adopt_stale, pins=stage_versions)
    try:
        summ = ex.run()
    except DagError as e:
        raise SystemExit(str(e))
    if summ.get("stale"):
        print(f"{summ['stale']} node(s) STALE (completed under other code): rerun with --adopt-stale to accept them, "
              "or --reset <node> to recompute", file=sys.stderr)
    return 0 if not any(summ.get(k, 0) for k in ("failed", "blocked", "stale")) else 1


def _opt(cli, dflt, typ):
    v = cli if cli is not None else dflt
    return None if v is None else typ(v)


def register(sub):
    p = sub.add_parser("run-dag", help="run a recipe (recipes/<track>/*.yaml) through the broker; JSON ledger, resumable")
    p.add_argument("dag", help="recipe file, or a name under recipes/ (armdiv/arm_lineage_v7div)")
    p.add_argument("--dry-run", action="store_true", help="print the plan (nodes, resources, placement, outputs, commands)")
    p.add_argument("--only", help="regex over node ids (dependencies are included)")
    p.add_argument("--point", action="append", help="matrix point filter, e.g. variant=semfix,seed=2 (repeatable)")
    p.add_argument("--ledger", help="ledger path (default artifacts/runs/<track>/_dags/<name>/ledger.json)")
    p.add_argument("--reset", action="append", help="forget a node's ledger entry (repeatable)")
    p.add_argument("--retry-failed", action="store_true", help="re-plan failed/blocked nodes (a manual decision, D-061)")
    p.add_argument("--adopt-stale", action="store_true",
                   help="accept completed nodes whose outputs came from other code (same config and version pins); recorded in the ledger")
    p.add_argument("--max-parallel", type=int)
    p.add_argument("--max-parallel-gpu", type=int, help="cap on running GPU nodes (default defaults.max_parallel_gpu)")
    p.add_argument("--max-cpu", type=float, help="cap on summed declared CPU of running nodes (default defaults.max_cpu)")
    p.add_argument("--max-mem-gib", type=float, help="cap on summed declared memory (GiB) of running nodes (default defaults.max_mem_gib)")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--peer-repo", help="peer code dir (default $RRP_PEER_REPO; must be /dev/shm/rrp-brandonin/wt/<track>)")
    p.add_argument("--allow-cross-placement", action="store_true")
    p.add_argument("--show-config", help="print one node's RunConfig and native config")
    p.add_argument("--root", help="repo root (default: this checkout)")
    p.set_defaults(fn=cmd_run_dag)
