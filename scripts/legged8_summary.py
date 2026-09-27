"""W8 combined contact_v2 summary across bodies from the per-body tables (scripts/legged8_compare.py JSONs) and the
D-112 dataset-gate JSONs. Per body: tracker, dataset gate, context-halt effect (per seed, pooled with CI), fixsem-nosem
pooled difference, per-seed ordering and exact permutation p, irrelevant control, goal steering, R2 success totals and
the teacher/BC references. Caveats are carried verbatim.
usage: python scripts/legged8_summary.py OUT_PREFIX DIR KEY=LABEL ...   (KEY = legged8_compare_<KEY>.json, gate_<KEY>.json in DIR)"""
import json
import sys
from pathlib import Path

out, D = Path(sys.argv[1]), Path(sys.argv[2])
bodies = [a.split("=", 1) for a in sys.argv[3:]]
HALT, CTRL, STEER = "ctx halt Δforward m", "ctx mirror INACTIVE Δforward m", "ctx ACTIVE−INACTIVE toward m"


def f(x):
    return "–" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"


def seeds(v, lab):
    return " / ".join(f"{v[s]['effects'][lab][0]:+.2f}" for s in sorted(k for k in v if k != "pooled"))


def succ(v, k):
    rows = [v[s]["r2"][k] for s in v if s != "pooled" and v[s]["r2"].get(k)]
    return f"{sum(r['success'] for r in rows)}/{sum(r['n'] for r in rows)} ({sum(r['fell'] for r in rows)} fell)"


res, md = {}, ["# W8: legged semantic supervision on contact_v2 (anymal_c, go2, t1)", "",
               "Protocol D-090 (R2 deployable route; edits on snap_s4000, dev seeds 10000-10019, window t=2-5 s; 3 training seeds per "
               "variant; exact one-sided permutation over the 3 vs 3 training-seed means). Sources: learned:<flow ckpt>; "
               "commands scripted_teacher; gait learned_tracker (per body below).", "",
               "| body | tracker | D-112 dataset gate | ctx halt semfix s0/s1/s2 → pooled | ctx halt nosem s0/s1/s2 → pooled | "
               "pooled semfix−nosem | every seed ordered (p one-sided) | inactive control semfix / nosem | goal steering semfix / nosem "
               "(ordered, p) | R2 success final: semfix / nosem | teacher / BC |", "|" + "---|" * 11]
for key, label in bodies:
    c = json.loads((D / f"legged8_compare_{key}.json").read_text())
    g = json.loads((D / f"gate_{key}.json").read_text())["gate"] if (D / f"gate_{key}.json").exists() else None
    V, T = c["variants"], c["tests_fixsem_vs_nosem"]
    th, ts = T[HALT], T[STEER]
    gate = "–" if g is None else (f"{'PASS' if g['passed'] else 'FAIL'} (slip<0.15 {100 * g['slip_lt_0p15_frac']:.1f}%, "
                                  f"falls@0 {g['noise0_falls']})")
    ref = c["references"]
    row = dict(body=key, label=label, caveat=c.get("caveat"), gate=g, halt=th, steer=ts,
               halt_pooled={v: V[v]["pooled"]["effects"][HALT] for v in ("semfix", "nosem")},
               control={v: V[v]["pooled"]["effects"][CTRL] for v in ("semfix", "nosem")},
               r2_final={v: succ(V[v], "r2_final") for v in ("semfix", "nosem")},
               r2_snap={v: succ(V[v], "r2_snap_s4000") for v in ("semfix", "nosem")},
               references={k: (None if d is None else f"{d['success']}/{d['n']}") for k, d in ref.items()})
    res[key] = row
    p, q = th["permutation"], ts["permutation"]
    md.append(f"| {key} | {label} | {gate} | {seeds(V['semfix'], HALT)} → {f(V['semfix']['pooled']['effects'][HALT])} | "
              f"{seeds(V['nosem'], HALT)} → {f(V['nosem']['pooled']['effects'][HALT])} | {f(th['pooled_diff_fixsem_minus_nosem'])} | "
              f"{p['every_seed_ordered']} ({p['p_one_sided']}) | {V['semfix']['pooled']['effects'][CTRL][0]:+.3f} / "
              f"{V['nosem']['pooled']['effects'][CTRL][0]:+.3f} | {V['semfix']['pooled']['effects'][STEER][0]:+.2f} / "
              f"{V['nosem']['pooled']['effects'][STEER][0]:+.2f} ({q['every_seed_ordered']}, {q['p_one_sided']} {q['direction']}) | "
              f"{row['r2_final']['semfix']} / {row['r2_final']['nosem']} | {row['references'].get('teacher')} / {row['references'].get('bc')} |")
cav = [(k, r["caveat"]) for k, r in res.items() if r["caveat"]]
if cav:
    md += ["", "Caveats:"] + [f"- **{k}: {c}**" for k, c in cav]
Path(f"{out}.json").write_text(json.dumps(res, indent=1, default=str))
Path(f"{out}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
