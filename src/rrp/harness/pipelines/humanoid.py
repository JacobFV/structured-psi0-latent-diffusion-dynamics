"""Humanoid stages (family `humanoid`; docs/architecture.md 14.4, research/tracks/humanoid.md section 4). One family, one task per
recipe instance, contact model in flags.contact_version:

| stage | what it does |
|---|---|
| collect | per body and seed one shard of the scripted teacher (task registry) driving a REGISTERED actor (`rl_expert`, spec `<body>:<version>`), `rrp data legged-latent-collect --task T --tracker-id S` |
| pack | nested target-demo packs: the FIRST N demos (by seed) of a body linked under `n<N>/<body>/`, and `pack.json` = the acquisition record of every (task, body, N): demos, teacher ticks, env samples, matched updates |
| train_rep, train_flow, train_bc | the legged trainers, behind the humanoid guards: sealed dataset, contact version, waypoint-free shards refused with a reason, and (options.adapt) the matched-update check + `acquisition.json` |
| eval_transfer | `rrp eval humanoid-transfer --scope dev`: bodies x methods x budgets x training seeds, Level 1 apart from Level 2, one acquisition accounting, tables |
| sealed_eval | the same on the SEALED bodies and evaluation scenes; needs options.sealed: true; every cell runs once (SealedSplit log `artifacts/runs/humanoid/sealed_log.jsonl`) |

Source labels are per method in the transfer config (learned / bc / privileged_teacher / scripted_teacher); a method whose declared label
differs from its policy's own is an error. Demo budgets count teacher ticks, not just episodes (`teacher_ticks`), and every method of one
(level, body, budget) group must carry the same acquisition record.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from rrp.core.runconfig import META, register_family
from rrp.core.sealed import SealedSplit
from rrp.harness.pipelines.base import StageContext, StageError, register
from rrp.harness.pipelines.legged import _json_safe, _seed_list, check_contact_version, physics_env

FAMILY = "humanoid"
register_family(FAMILY, flag_spec={"train_rep": {"qd_dropout": "latent.qd_dropout", "contact_version": META}},
                default_stage_flags={"contact_version": META},
                doc="humanoid transfer stages (collect, pack, train_*, eval_transfer, sealed_eval); contact model in flags.contact_version")

SHARD_ARRAYS = (".npz", ".json", ".manifest.json")


# ---------------------------------------------------------------------------------------------------- data helpers
def shard_index(data: str | Path, body: str) -> list[dict]:
    """The single-seed shards of `<data>/<body>/` sorted by seed: [{seed, stem, ticks, status}]. A shard holding more than one
    episode is refused (a demo budget counts whole shards, so a shard is one demo)."""
    out = []
    for js in sorted(Path(data, body).glob("s*.json")):
        if js.name.endswith(".manifest.json"):
            continue
        eps = json.loads(js.read_text())["episodes"]
        if len(eps) != 1:
            raise StageError(f"{js}: {len(eps)} episodes; a humanoid demo shard holds exactly one (collect with one seed per shard)")
        e = eps[0]
        out.append(dict(seed=int(e["seed"]), stem=js.stem, ticks=int(e["ticks"]), status=e.get("status")))
    return sorted(out, key=lambda r: r["seed"])


def make_pack(data: str | Path, out: str | Path, *, task: str, bodies, budgets, updates: dict) -> dict:
    """Nested packs `<out>/n<N>/<body>/`: the first N shards of the body (by seed), linked, never copied. Returns the pack record
    (also written by the stage): records[`<task>|<body>|n<N>`] = demos / teacher_ticks / env_samples 0 / updates / seeds."""
    out = Path(out)
    records = {}
    for body in bodies:
        idx = shard_index(data, body)
        for n in sorted(int(b) for b in budgets):
            if len(idx) < n:
                raise StageError(f"pack {body}: budget {n} demos but only {len(idx)} collected")
            if str(n) not in {str(k) for k in updates}:
                raise StageError(f"pack: no matched update count for budget {n} (options.updates)")
            d = out / f"n{n}" / body
            d.mkdir(parents=True, exist_ok=True)
            for r in idx[:n]:
                for ext in SHARD_ARRAYS:
                    src, dst = Path(data, body, r["stem"] + ext).resolve(), d / (r["stem"] + ext)
                    if not src.exists():
                        continue
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    os.symlink(src, dst)
            records[f"{task}|{body}|n{n}"] = dict(task=task, body=body, budget=n, demos=n, teacher_ticks=sum(r["ticks"] for r in idx[:n]),
                                                  env_samples=0, updates=int({str(k): v for k, v in updates.items()}[str(n)]),
                                                  seeds=[r["seed"] for r in idx[:n]], dir=str(d))
    return dict(task=task, records=records)


def acquisition_record(pack: dict, task: str, body: str, budget: int, steps: int) -> dict:
    """The run-side acquisition of an adapting trainer: the pack record, after checking that the trainer's update count is the
    matched one. Raises StageError otherwise."""
    rec = (pack.get("records") or {}).get(f"{task}|{body}|n{budget}")
    if rec is None:
        raise StageError(f"pack has no record for {task}|{body}|n{budget}")
    if int(steps) != int(rec["updates"]):
        raise StageError(f"{task}|{body}|n{budget}: {steps} updates, the matched count is {rec['updates']} (demo budgets are compared at equal updates)")
    return {k: rec[k] for k in ("task", "body", "budget", "demos", "teacher_ticks", "env_samples", "updates", "seeds")}


def check_waypoint_free(data: str | Path, bodies) -> None:
    """The legged trainers' `LeggedData` reads `episodes[i]['waypoints']` of every shard; humanoid tasks other than the waypoint task record
    none. Refuse with the reason instead of a KeyError deep in the loader (the loader is the legged owner's)."""
    for b in bodies:
        for js in sorted(Path(data, b).glob("s*.json")):
            if js.name.endswith(".manifest.json"):
                continue
            eps = json.loads(js.read_text())["episodes"]
            if any(not e.get("waypoints") for e in eps):
                raise StageError(f"{js}: episodes carry no `waypoints`; LeggedData (rrp.harness.train.legged_latent_train) requires them, so the "
                                 "task's shards cannot be trained on until the loader is task-agnostic (research/tracks/humanoid.md, H6)")


# ---------------------------------------------------------------------------------------------------- stages
@register(FAMILY, "collect", source="scripted_teacher")
def collect(ctx: StageContext) -> dict:
    """Teacher demos, one shard per (body, seed). options: task, bodies, trackers ({body: '<body>:<version>'}), seeds (range spec),
    sigmas, workers, max_steps. Sealed bodies only on target-adaptation seeds (SealedSplit)."""
    o = ctx.opts
    task, bodies, trackers = o["task"], list(o["bodies"]), dict(o.get("trackers") or {})
    seeds = _seed_list(o["seeds"])
    split = SealedSplit.load()
    split.assert_train_allowed(bodies, seeds, what="collect")
    out = o.get("out_dir", ctx.rc.out)
    jobs = []
    for b in bodies:
        (ctx.root / out / b).mkdir(parents=True, exist_ok=True)
        for s in seeds:
            if (ctx.root / out / b / f"s{s}-{s}.npz").exists():
                continue
            argv = ["-m", "rrp.cli", "data", "legged-latent-collect", "--body", b, "--seeds", f"{s}-{s}", "--out", out, "--task", task,
                    "--sigmas", str(o.get("sigmas", "0,0.1,0.2,0.3"))]
            if trackers.get(b):
                argv += ["--tracker-id", trackers[b]]
            if o.get("max_steps"):
                argv += ["--max-steps", str(o["max_steps"])]
            jobs.append((argv, physics_env(ctx, OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=""), ctx.root / out / b / f"log_s{s}.txt"))
    ctx.run_parallel(jobs, int(o.get("workers", 1)))
    from rrp.harness.data.manifest import read_manifest
    want, per = ctx.rc.flags.contact_version, {}
    for b in bodies:
        idx = shard_index(ctx.root / out, b)
        got = {r["seed"] for r in idx}
        if got != set(seeds):
            raise StageError(f"{b}: shards for seeds {sorted(set(seeds) - got)[:5]} missing")
        cvs, tasks = set(), set()
        for r in idx:
            sh = ctx.root / out / b / (r["stem"] + ".json")
            tasks.add(json.loads(sh.read_text()).get("task"))
            prov = read_manifest(sh.with_suffix(".manifest.json")).get("provenance")
            cvs.add(getattr(getattr(prov, "physics", None), "contact_version", None))
        if tasks != {task}:
            raise StageError(f"{b}: shards record task(s) {sorted(map(str, tasks))}, expected {task!r}")
        if cvs != {want}:
            raise StageError(f"{b}: shards record contact version(s) {sorted(map(str, cvs))} != {want!r}")
        per[b] = dict(episodes=len(idx), ticks=sum(r["ticks"] for r in idx), success=sum(r["status"] == "success" for r in idx),
                      sealed=split.is_sealed_body(b))
    return dict(outputs={"data": out}, metrics=dict(task=task, seeds=o["seeds"], contact_version=want, bodies=per),
                source_detail=f"scripted_teacher:{task} -> " + "|".join(f"{b}:{t}" for b, t in sorted(trackers.items())))


@register(FAMILY, "pack", source="scripted_teacher")
def pack(ctx: StageContext) -> dict:
    """Nested target-demo packs and their acquisition record. options: task, bodies, budgets, updates ({budget: matched updates});
    input `data` (a collect output)."""
    o = ctx.opts
    data = ctx.root / ctx.inp("data")
    rec = make_pack(data, ctx.out, task=o["task"], bodies=o["bodies"], budgets=o["budgets"], updates=o["updates"])
    split = SealedSplit.load()
    for n in sorted(int(b) for b in o["budgets"]):
        split.assert_dataset_allowed(ctx.out / f"n{n}", o["bodies"])
    (ctx.out / "pack.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    return dict(outputs={"pack": str(Path(ctx.rc.out) / "pack.json"), "data": ctx.rc.out},
                metrics=dict(task=o["task"], records={k: {x: v[x] for x in ("demos", "teacher_ticks", "updates")} for k, v in rec["records"].items()}),
                source_detail=str(ctx.inp("data")))


def _guard(ctx: StageContext, data, bodies, steps) -> None:
    from rrp.harness.pipelines.legged import sealed_data_guard
    sealed_data_guard(data, bodies)
    check_contact_version(ctx, str(data))
    check_waypoint_free(data, bodies)
    a = ctx.opts.get("adapt")
    if a:
        pk = json.loads((ctx.root / ctx.inp("pack")).read_text())
        acq = acquisition_record(pk, a["task"], a["body"], int(a["budget"]), int(steps))
        ctx.out.mkdir(parents=True, exist_ok=True)
        (ctx.out / "acquisition.json").write_text(json.dumps(dict(acq, stage=ctx.rc.stage, pack_sha256=hashlib.sha256(
            (ctx.root / ctx.inp("pack")).read_bytes()).hexdigest()), indent=1, sort_keys=True))


@register(FAMILY, "train_rep", source="learned")
def train_rep(ctx: StageContext) -> dict:
    """Legged Stage A on humanoid shards (options.adapt: matched-update accounting)."""
    from rrp.harness.train.legged_latent_train import train_rep as fn
    cfg = ctx.native
    _guard(ctx, cfg["data"], cfg["bodies"], cfg["steps"])
    res = fn(cfg, ctx.out)
    rep = str(Path(ctx.rc.out) / "representation.pt")
    return dict(outputs={"representation": rep}, metrics=_json_safe(res), source_detail=rep)


@register(FAMILY, "train_flow", source="learned")
def train_flow(ctx: StageContext) -> dict:
    """Legged system i on a frozen humanoid representation."""
    from rrp.harness.train.legged_latent_train import load_legged_rep, train_flow as fn
    cfg = ctx.native
    import torch
    rcfg = load_legged_rep(ctx.root / cfg["representation"], torch.device("cpu"))[0]
    _guard(ctx, rcfg["data"], rcfg["bodies"], cfg["steps"])
    res = fn(cfg, ctx.out)
    pol = str(Path(ctx.rc.out) / "policy.pt")
    return dict(outputs={"policy": pol}, metrics=_json_safe(res), source_detail=pol)


@register(FAMILY, "train_bc", source="bc")
def train_bc(ctx: StageContext) -> dict:
    """Whole-policy BC positive control on humanoid shards (source bc)."""
    from rrp.harness.train.legged_bc import train as fn
    cfg = ctx.native
    _guard(ctx, cfg["data"], cfg["bodies"], cfg["steps"])
    res = fn(cfg, ctx.out)
    pol = str(Path(ctx.rc.out) / "policy.pt")
    return dict(outputs={"policy": pol}, metrics=_json_safe(res), source_detail=pol)


def _transfer(ctx: StageContext, scope: str) -> dict:
    o = ctx.opts
    cfg = dict(o["transfer"])
    ctx.out.mkdir(parents=True, exist_ok=True)
    cpath = ctx.out / "transfer_config.json"
    cpath.write_text(json.dumps(cfg, indent=1, sort_keys=True))
    argv = ["-m", "rrp.cli", "eval", "humanoid-transfer", "--config", str(cpath), "--root", str(ctx.root), "--out", str(ctx.out),
            "--scope", scope, "--run"]
    if o.get("variant"):
        argv += ["--variant", str(o["variant"])]           # a recipe point evaluates only its own methods / training seed
    if o.get("train_seed") is not None:
        argv += ["--train-seed", str(o["train_seed"])]
    if scope == "sealed":
        if not o.get("sealed"):
            raise StageError("sealed_eval needs options.sealed: true (sealed cells run once and are logged)")
        argv.append("--sealed")
    ctx.run(argv, env=physics_env(ctx, OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=""))
    t = json.loads((ctx.out / "tables.json").read_text())
    if t["accounting_violations"]:
        raise StageError("acquisition accounting violations: " + "; ".join(t["accounting_violations"][:3]))
    rel = Path(ctx.rc.out)
    return dict(outputs={"tables": str(rel / "tables.json"), "results": str(rel / "results.jsonl"), "report": str(rel / "tables.md")},
                metrics=dict(scope=scope, coverage=t["coverage"], not_done=t["not_done"]),
                source_detail="transfer:" + str(cfg.get("name")))


@register(FAMILY, "eval_transfer", source="learned", flags=["contact_version"])
def eval_transfer(ctx: StageContext) -> dict:
    """Level 1 / Level 2 transfer matrix on the non-sealed bodies, development scenes (options.transfer = the config; options.variant /
    train_seed restrict a recipe point to its own cells)."""
    return _transfer(ctx, "dev")


@register(FAMILY, "sealed_eval", source="learned", flags=["contact_version"])
def sealed_eval(ctx: StageContext) -> dict:
    """The same matrix on the sealed bodies, evaluation scenes, each cell once (options.sealed: true is required)."""
    return _transfer(ctx, "sealed")
