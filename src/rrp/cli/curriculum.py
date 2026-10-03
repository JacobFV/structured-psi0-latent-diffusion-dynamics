"""Fanout unit R11 (docs/relations.md 5.5, section 10): the two curriculum CLI surfaces named in its row.

`rrp steer <run> <op...>` appends a validated-at-decision-time steer to `<run>/steer.jsonl` (picked up at the run's
next decision interval; no restart, no code -- this command only writes the line, `Scheduler.steer` / `.validate`
do the real work when the training loop's `relation_batches` next decides); `rrp steer <run>` alone lists the run's
steer lines and what became of each (pending / applied at step / REJECTED). `rrp suite relations-curriculum` (registered into
`rrp.cli.tools.TOOLS`) reads a run's `schedule.jsonl` (written by `Scheduler.export`, one `ScheduleState` per
decision) and prints competence by composition depth, the schedule history with its reasons, and (given eval
records) interfering pairs (`rrp.harness.data.relgen.curriculum.interfering_pairs`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ------------------------------------------------------------------ rrp steer
def _steer_status(run: Path) -> list[str]:
    """Each line of `<run>/steer.jsonl` with what the run did with it: pending (not yet read at a decision), applied at
    step N, or REJECTED (reason), from the last records of `<run>/schedule.jsonl`."""
    lines = [l for l in (run / "steer.jsonl").read_text().splitlines() if l.strip()] if (run / "steer.jsonl").exists() else []
    recs = _read_jsonl(run / "schedule.jsonl")
    read = recs[-1].get("steer_line", 0) if recs else 0
    done = {}
    k = 0
    for r in recs:
        for op in r.get("steers", []):
            done[k] = op
            k += 1
    out = []
    for i, line in enumerate(lines):
        op = done.get(i)
        state = "pending" if i >= read else ("rejected: " + op["reason"] if op and "REJECTED" in op.get("reason", "")
                                             else f"applied at step {op['at']}" if op else "read")
        out.append(f"{i + 1:>3}  [{state}]  {line}")
    return out


def cmd_steer(a) -> int:
    from rrp.harness.data.relgen.curriculum import parse_steer
    run = Path(a.run)
    op_text = " ".join(a.op)
    if not op_text:                                     # no op: what this run's steer log holds and what became of it
        rows = _steer_status(run)
        print("\n".join(rows) if rows else f"no steers recorded at {run / 'steer.jsonl'}")
        return 0
    # `op` is argparse.REMAINDER, so it swallows every token after `run` verbatim, including anything that looks
    # like a flag -- author/reason are only settable through the op's own JSON form (docs/relations.md 5.5:
    # {"op": "boost", ..., "author": "...", "reason": "..."}), never as separate `rrp steer` flags.
    op = parse_steer(op_text)
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "steer.jsonl", "a") as fh:
        fh.write(op.to_json() + "\n")
    print(op.to_json())
    return 0


def register(sub) -> None:
    p = sub.add_parser("steer", help="append a curriculum steer op to <run>/steer.jsonl (docs/relations.md 5.5)")
    p.add_argument("run", help="run directory (its steer.jsonl is appended to)")
    p.add_argument("op", nargs=argparse.REMAINDER,
                   help='omit to list the run\'s steers and their status; else the op: CLI grammar (docs/relations.md 5.5), e.g. boost ix.support x2 for 5000, or one '
                        'JSON object (quoted) to also set author/reason, e.g. \'{"op": "pin", "factor": "geo.above", '
                        '"value": 2, "author": "lead", "reason": "lags"}\'')
    p.set_defaults(fn=cmd_steer)


# ------------------------------------------------------------------ rrp suite relations-curriculum
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def suite_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rrp suite relations-curriculum",
                                description="curriculum schedule / competence-by-depth / interfering-pairs report")
    p.add_argument("run", help="run directory containing schedule.jsonl")
    p.add_argument("--records", help="JSONL of eval records ({'active': [...], 'metric': {factor: value}}) for "
                                     "interference; default <run>/eval_records.jsonl")
    p.add_argument("--metric", default="competence")
    p.add_argument("--threshold", type=float, default=0.1)
    a = p.parse_args(argv)
    run = Path(a.run)
    hist = _read_jsonl(run / "schedule.jsonl")
    if not hist:
        print(f"no decisions recorded yet at {run / 'schedule.jsonl'}")
        return 0
    print(f"schedule history ({len(hist)} decisions), {run / 'schedule.jsonl'}:")
    for h in hist:
        print(f"  step {h.get('step', '?'):>8}  full_world={h.get('full_world', 0.0):.3f}  level={h.get('level', {})}")
        for r in h.get("reasons", []) or []:
            print(f"    - {r}")
    print("\ncompetence by factor x composition depth:")
    by_level: dict[tuple[str, int], list[float]] = {}
    for h in hist:
        for f, lvl in (h.get("level") or {}).items():
            c = (h.get("signals") or {}).get(f, {}).get("competence")
            if c is not None:
                by_level.setdefault((f, lvl), []).append(c)
    if not by_level:
        print("  (no signals recorded yet -- observe() has not been called, or nothing has been observed)")
    for (f, lvl), vals in sorted(by_level.items()):
        print(f"  {f} @ k={lvl}: competence={sum(vals) / len(vals):.3f} (n={len(vals)})")
    records_path = Path(a.records) if a.records else run / "eval_records.jsonl"
    records = _read_jsonl(records_path)
    print(f"\ninterfering pairs (metric={a.metric}, threshold={a.threshold}):")
    if not records:
        print(f"  no eval records at {records_path} (interference not computed)")
        return 0
    from rrp.harness.data.relgen.curriculum import interfering_pairs
    for r in records:
        r["active"] = frozenset(r.get("active") or ())
    factors = sorted({f for r in records for f in r["active"]})
    pairs = interfering_pairs(records, factors, metric=a.metric, threshold=a.threshold)
    if not pairs:
        print("  none above threshold")
    for f, g, v in pairs:
        print(f"  {f} vs {g}: {v:+.3f}")
    return 0


# ------------------------------------------------------------------ rrp suite relations-compare (T9 tables)
def _depth_table(runs: dict[tuple[str, int], Path]) -> list[dict]:
    """Per factor set x factor x composition depth k: mean EMA competence over the scheduler decisions at that depth (all
    seeds pooled), the number of decisions and the first step the depth was reached, from each run's `schedule.jsonl`."""
    cells: dict[tuple[str, str, int], dict] = {}
    for (fset, seed), run in sorted(runs.items()):
        for h in _read_jsonl(run / "schedule.jsonl"):
            for f, k in (h.get("level") or {}).items():
                c = cells.setdefault((fset, f, int(k)), {"vals": [], "n": 0, "first": {}})
                c["n"] += 1
                c["first"].setdefault(seed, h.get("step"))
                comp = ((h.get("signals") or {}).get(f) or {}).get("competence")
                if comp is not None:
                    c["vals"].append(comp)
    return [{"set": s, "factor": f, "depth": k, "competence": (sum(c["vals"]) / len(c["vals"]) if c["vals"] else None),
             "decisions": c["n"], "first_step_by_seed": c["first"]} for (s, f, k), c in sorted(cells.items())]


def _final_metrics(runs: dict[tuple[str, int], Path]) -> list[dict]:
    """Per run, the mean over the last 20 train-log records (2000 steps) of the flow loss and the relgen terms (a factor set's
    cost to the action loss); `step` is the last record's step."""
    out = []
    for (fset, seed), run in sorted(runs.items()):
        log = _read_jsonl(run / "train_log.jsonl")[-20:]
        row = {"set": fset, "seed": seed, "step": log[-1].get("step") if log else None}
        for k in ("flow", "relgen", "relgen_raw", "n_relgen"):
            v = [r[k] for r in log if k in r]
            row[k] = sum(v) / len(v) if v else None
        out.append(row)
    return out


_STAGES = ("approach", "grasp", "lift", "transport", "place")


_V3_EVALS = (("dev", "eval_r2-dev_s{seed}"), ("heldout", "heldout-dev_s{seed}"))


def _eval_rows(root: Path, lineage: str, seed: int, evals=_V3_EVALS) -> dict[tuple, dict]:
    """(stage, body, env seed) -> the eval row of one run's deployable-route evaluation, `evals` = (stage, dir format) pairs
    (default: `eval_r2-dev_s<seed>` dev bodies and `heldout-dev_s<seed>` held-out bodies), one `<body>/generated_*.jsonl` each
    (arm._ladder_eval). Rows carry the privileged evaluator's success, the stage it failed at and the closest tcp-cube distance
    (graded outcomes for when success is at a floor)."""
    out = {}
    for stage, fmt in evals:
        for f in sorted((root / lineage / fmt.format(seed=seed)).glob("*/generated_*.jsonl")):
            for r in _read_jsonl(f):
                if "privileged_success" in r and "seed" in r:
                    out[(stage, f.parent.name, int(r["seed"]))] = r
    return out


def _stage_rank(r: dict) -> int:
    """Furthest stage reached: index of the failed stage (approach 0 ... place 4), 5 for a success."""
    return len(_STAGES) if r.get("privileged_success") else _STAGES.index(r["failed_stage"]) if r.get("failed_stage") in _STAGES else 0


def _interference_table(root: Path, sets: list[str], seeds: list[int], base: str) -> list[dict]:
    """Each factor set against `base` at equal steps / data / training seed, per body and pooled: success rates, the paired
    difference (same training seed, same body, same env seed) with a bootstrap 95% CI and the exact McNemar p, plus the graded
    paired differences (set - base) of the furthest stage reached and the closest tcp-cube distance (m, lower is better)."""
    from rrp.harness.eval.statistics import mcnemar_exact, paired_bootstrap_ci, wilson
    ev = {(s, seed): _eval_rows(root, f"relations-{s}", seed) for s in [base, *sets] for seed in seeds}
    rows = []
    for s in sets:
        for body in [None] + sorted({k[:2] for r in ev.values() for k in r}):
            keys = [(seed, k) for seed in seeds for k in ev[(base, seed)] if k in ev[(s, seed)]
                    and (body is None or k[:2] == body)]
            if not keys:
                continue
            ra = [ev[(s, seed)][k] for seed, k in keys]
            rb = [ev[(base, seed)][k] for seed, k in keys]
            a = [float(r["privileged_success"]) for r in ra]
            b = [float(r["privileged_success"]) for r in rb]
            n10 = sum(1 for x, y in zip(a, b) if x and not y)
            n01 = sum(1 for x, y in zip(a, b) if y and not x)
            lo, hi = wilson(int(sum(a)), len(a))
            rows.append({"set": s, "stage": "all" if body is None else body[0], "body": "pooled" if body is None else body[1],
                         "n_pairs": len(keys), "rate_set": sum(a) / len(a), "rate_set_wilson95": [lo, hi],
                         "rate_base": sum(b) / len(b), "delta": paired_bootstrap_ci(a, b), "set_only": n10, "base_only": n01,
                         "mcnemar_p": mcnemar_exact(n10, n01),
                         "stage_rank_delta": paired_bootstrap_ci([_stage_rank(r) for r in ra], [_stage_rank(r) for r in rb]),
                         "min_tcp_cube_m_delta": paired_bootstrap_ci([r["min_tcp_cube_m"] for r in ra],
                                                                     [r["min_tcp_cube_m"] for r in rb])})
    return rows


def _md(title: str, rows: list[dict], cols: list[str]) -> str:
    fmt = lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))        # noqa: E731
    return "\n".join([f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
                      *("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows), ""])


def compare_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rrp suite relations-compare",
                                description="T9 tables: competence by composition depth (schedule.jsonl), the factor-set-vs-base "
                                            "interference table (paired dev + held-out eval rows) and the final training metrics")
    p.add_argument("--root", default="artifacts/runs/relations")
    p.add_argument("--sets", default="geo,ix,task")
    p.add_argument("--base", default="base")
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--out", help="directory for tables.json / tables.md (default <root>/tables)")
    p.add_argument("--v8div", action="store_true", help="T9 on the v8div lineage: sets relations8-<set> vs the control lineage")
    p.add_argument("--control", default="artifacts/runs/armdiv/arm8div-semfix", help="--v8div control lineage directory")
    a = p.parse_args(argv)
    root, sets, seeds = Path(a.root), a.sets.split(","), [int(s) for s in a.seeds.split(",")]
    if a.v8div:
        compare_v8div(root, Path(a.control), sets, seeds, Path(a.out) if a.out else root / "tables")
        return 0
    runs = {(s, seed): root / f"relations-{s}" / f"train_flow_s{seed}" for s in [a.base, *sets] for seed in seeds
            if (root / f"relations-{s}" / f"train_flow_s{seed}").is_dir()}
    depth = _depth_table(runs)
    train = _final_metrics(runs)
    interf = _interference_table(root, sets, seeds, a.base)
    out = Path(a.out) if a.out else root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.json").write_text(json.dumps({"competence_by_depth": depth, "training_final": train,
                                                 "vs_base": interf}, indent=1, default=str))
    flat = [{**r, "delta_mean": r["delta"]["mean"], "delta_lo": r["delta"]["lo"], "delta_hi": r["delta"]["hi"],
             "rank_d": r["stage_rank_delta"]["mean"], "rank_lo": r["stage_rank_delta"]["lo"], "rank_hi": r["stage_rank_delta"]["hi"],
             "tcp_d": r["min_tcp_cube_m_delta"]["mean"], "tcp_lo": r["min_tcp_cube_m_delta"]["lo"],
             "tcp_hi": r["min_tcp_cube_m_delta"]["hi"]} for r in interf]
    md = "\n".join([_md("competence by factor x composition depth (scheduler EMA competence)", depth,
                        ["set", "factor", "depth", "competence", "decisions"]),
                    _md("factor set vs base (paired: training seed x body x env seed)", flat,
                        ["set", "stage", "body", "n_pairs", "rate_set", "rate_base", "delta_mean", "delta_lo", "delta_hi",
                         "set_only", "base_only", "mcnemar_p", "rank_d", "rank_lo", "rank_hi", "tcp_d", "tcp_lo", "tcp_hi"]),
                    _md("final training record", train, ["set", "seed", "step", "flow", "relgen", "relgen_raw", "n_relgen"])])
    (out / "tables.md").write_text(md)
    print(md)
    return 0


# ------------------------------------------------------------------ T9 on the v8div lineage (research/tracks/relations.md pre-registration)
V8_EVALS = (("final", "eval_r2-final_s{seed}"), ("heldout", "heldout-final_s{seed}"), ("newarms", "eval_r2-newarms_s{seed}"))
V8_GROUPS = {"primary": ("final", "heldout"), "newarms": ("newarms",)}
V8_FLOWS = ("train_flow_s{seed}", "flow_ft-ft_s{seed}")
COMPETENCE_MIN = 0.5
Z_BONF3 = 2.393980          # two-sided 1 - 0.05/3: the headline claim over the three single sets


def _contrast_rows(ev: dict, s: str, ctl: str, seeds: list[int]) -> list[dict]:
    """Set `s` vs the control per group (primary = the G3 source bodies, newarms) x body and pooled: Wilson rates, the
    unpaired Newcombe 95% interval of the difference (the pre-registered contrast), the paired McNemar test (same training
    seed, body and env seed) and the graded paired differences (stage reached, closest tcp-cube distance)."""
    from rrp.harness.eval.statistics import mcnemar_exact, newcombe_diff, paired_bootstrap_ci, wilson
    rows = []
    for g, stages in V8_GROUPS.items():
        bodies = sorted({k[1] for seed in seeds for k in ev[(ctl, seed)] if k[0] in stages})
        for body in [None, *bodies]:
            keys = [(seed, k) for seed in seeds for k in ev[(ctl, seed)] if k[0] in stages and (body is None or k[1] == body)
                    and k in ev[(s, seed)]]
            if not keys:
                continue
            ra, rb = [ev[(s, seed)][k] for seed, k in keys], [ev[(ctl, seed)][k] for seed, k in keys]
            a, b = [int(bool(r["privileged_success"])) for r in ra], [int(bool(r["privileged_success"])) for r in rb]
            n = len(keys)
            lo, hi = newcombe_diff(sum(a), n, sum(b), n)
            blo, bhi = newcombe_diff(sum(a), n, sum(b), n, z=Z_BONF3)
            n10 = sum(1 for x, y in zip(a, b) if x and not y)
            n01 = sum(1 for x, y in zip(a, b) if y and not x)
            rows.append({"set": s, "group": g, "body": body or "pooled", "n": n, "k_set": sum(a), "k_ctl": sum(b),
                         "rate_set": sum(a) / n, "rate_set_wilson95": list(wilson(sum(a), n)), "rate_ctl": sum(b) / n,
                         "rate_ctl_wilson95": list(wilson(sum(b), n)), "delta": (sum(a) - sum(b)) / n, "newcombe95": [lo, hi],
                         "newcombe_bonf3": [blo, bhi],
                         "set_only": n10, "ctl_only": n01, "mcnemar_p": mcnemar_exact(n10, n01),
                         "stage_rank_delta": paired_bootstrap_ci([_stage_rank(r) for r in ra], [_stage_rank(r) for r in rb]),
                         "min_tcp_cube_m_delta": paired_bootstrap_ci([r["min_tcp_cube_m"] for r in ra],
                                                                     [r["min_tcp_cube_m"] for r in rb])})
    return rows


def _final_competence(run: Path) -> dict[str, float]:
    """Per scheduled factor (the keys of the last decision's `share`), the competence signal at the run's last scheduler
    decision; None when that decision carries no signal for it (never observed: counts as not learned). {"<run>": None}
    when the run has no schedule at all."""
    last = (_read_jsonl(run / "schedule.jsonl")[-1:] or [None])[0]
    if last is None:
        return {"<no schedule>": None}
    sig = last.get("signals") or {}
    return {f: (sig.get(f) or {}).get("competence") for f in (last.get("share") or sig)}


def _verdict(primary: dict | None, comp: dict) -> str:
    """The pre-registered reading: validity (every scheduled factor competent >= COMPETENCE_MIN at the end of Fft in every
    seed), then helps / hurts / no detectable effect from the primary pooled Newcombe interval."""
    if primary is None:
        return "incomplete"
    lo, hi = primary["newcombe95"]
    eff = "helps" if lo is not None and lo > 0 else "hurts" if hi is not None and hi < 0 else "no detectable effect"
    blo, bhi = primary["newcombe_bonf3"]
    if eff != "no detectable effect":
        eff += " (holds at 1-0.05/3)" if (blo > 0 if eff == "helps" else bhi < 0) else " (not at 1-0.05/3)"
    bad = sorted(f"{f}@s{seed}" for seed, fs in comp.items() for f, c in fs.items() if c is None or c < COMPETENCE_MIN)
    return eff if not bad else f"{eff} (INVALID as a test of factor use: not learned {', '.join(bad)})"


def compare_v8div(root: Path, control: Path, sets: list[str], seeds: list[int], out: Path) -> dict:
    ctl = "__control__"
    ev = {(ctl, seed): _eval_rows(control.parent, control.name, seed, V8_EVALS) for seed in seeds}
    ev.update({(s, seed): _eval_rows(root, f"relations8-{s}", seed, V8_EVALS) for s in sets for seed in seeds})
    flows = {(f"{s}:{fmt.split('_s')[0]}", seed): root / f"relations8-{s}" / fmt.format(seed=seed)
             for s in sets for seed in seeds for fmt in V8_FLOWS if (root / f"relations8-{s}" / fmt.format(seed=seed)).is_dir()}
    depth, train = _depth_table(flows), _final_metrics(flows)
    contrast = [r for s in sets for r in _contrast_rows(ev, s, ctl, seeds)]
    ctl_rows = [{"group": g, "k": sum(int(bool(r["privileged_success"])) for seed in seeds for k, r in ev[(ctl, seed)].items()
                                       if k[0] in st), "n": sum(1 for seed in seeds for k in ev[(ctl, seed)] if k[0] in st)}
                for g, st in V8_GROUPS.items()]
    verdicts = {}
    for s in sets:
        comp = {seed: _final_competence(root / f"relations8-{s}" / V8_FLOWS[1].format(seed=seed)) for seed in seeds}
        prim = next((r for r in contrast if r["set"] == s and r["group"] == "primary" and r["body"] == "pooled"), None)
        verdicts[s] = {"verdict": _verdict(prim, comp), "final_competence": comp}
    res = {"control": str(control), "control_rates": ctl_rows, "contrast": contrast, "competence_by_depth": depth,
           "training_final": train, "verdicts": verdicts}
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables_v8div.json").write_text(json.dumps(res, indent=1, default=str))
    flat = [{**r, "w_lo": r["rate_set_wilson95"][0], "w_hi": r["rate_set_wilson95"][1], "nc_lo": r["newcombe95"][0],
             "nc_hi": r["newcombe95"][1], "nb_lo": r["newcombe_bonf3"][0], "nb_hi": r["newcombe_bonf3"][1], "rank_d": r["stage_rank_delta"]["mean"], "tcp_d": r["min_tcp_cube_m_delta"]["mean"]}
            for r in contrast]
    md = "\n".join([_md("control (arm8div-semfix) success", ctl_rows, ["group", "k", "n"]),
                     _md("factor set - control (Newcombe 95%; paired McNemar; graded paired deltas)", flat,
                         ["set", "group", "body", "n", "k_set", "k_ctl", "rate_set", "w_lo", "w_hi", "rate_ctl", "delta",
                          "nc_lo", "nc_hi", "nb_lo", "nb_hi", "set_only", "ctl_only", "mcnemar_p", "rank_d", "tcp_d"]),
                     _md("competence by factor x composition depth (scheduler EMA; F0 = train_flow, Fft = flow_ft-ft)", depth,
                         ["set", "factor", "depth", "competence", "decisions"]),
                     _md("final training record (last 2000 steps)", train, ["set", "seed", "step", "flow", "relgen", "n_relgen"]),
                     _md("pre-registered verdicts", [{"set": s, **v} for s, v in verdicts.items()], ["set", "verdict"])])
    (out / "tables_v8div.md").write_text(md)
    print(md)
    return res
