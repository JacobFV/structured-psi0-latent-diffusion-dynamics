"""Fanout unit R11 (docs/relations.md 5.5, section 10): the two curriculum CLI surfaces named in its row.

`rrp steer <run> <op...>` appends a validated-at-decision-time steer to `<run>/steer.jsonl` (picked up at the run's
next decision interval; no restart, no code -- this command only writes the line, `Scheduler.steer` / `.validate`
do the real work when the training loop next calls `decide()`). `rrp suite relations-curriculum` (registered into
`rrp.cli.tools.TOOLS`) reads a run's `schedule.jsonl` (written by `Scheduler.export`, one `ScheduleState` per
decision) and prints competence by composition depth, the schedule history with its reasons, and (given eval
records) interfering pairs (`rrp.harness.data.relgen.curriculum.interfering_pairs`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ------------------------------------------------------------------ rrp steer
def cmd_steer(a) -> int:
    from rrp.harness.data.relgen.curriculum import parse_steer
    op_text = " ".join(a.op)
    if not op_text:
        raise SystemExit("usage: rrp steer <run> <op...> (e.g. rrp steer artifacts/runs/foo boost ix.support x2 for 5000)")
    # `op` is argparse.REMAINDER, so it swallows every token after `run` verbatim, including anything that looks
    # like a flag -- author/reason are only settable through the op's own JSON form (docs/relations.md 5.5:
    # {"op": "boost", ..., "author": "...", "reason": "..."}), never as separate `rrp steer` flags.
    op = parse_steer(op_text)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "steer.jsonl", "a") as fh:
        fh.write(op.to_json() + "\n")
    print(op.to_json())
    return 0


def register(sub) -> None:
    p = sub.add_parser("steer", help="append a curriculum steer op to <run>/steer.jsonl (docs/relations.md 5.5)")
    p.add_argument("run", help="run directory (its steer.jsonl is appended to)")
    p.add_argument("op", nargs=argparse.REMAINDER,
                   help='the op: CLI grammar (docs/relations.md 5.5), e.g. boost ix.support x2 for 5000, or one '
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
