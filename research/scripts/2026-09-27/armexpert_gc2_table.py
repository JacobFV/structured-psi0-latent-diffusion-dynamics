"""D-110 (4): existing arm routes, grasp_v1 (recorded) vs grasp_v2 (re-evaluated, same command + RRP_GRASP_CONTACT=v2).
argv: recorded_v1_root (ladder_v1 copy) gc2eval_root (armexpert_gc2eval) out.json"""
import json
import sys
from pathlib import Path

REC, NEW, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ROUTES = [("frozen sem, seed 1 (D-078)", ""), ("semfix, seed 1", "sf"), ("nosem, seed 1", "ns"),
          ("frozen sem, seed 2 (D-095)", "se2"), ("semfix, seed 2", "sf2"), ("nosem, seed 2", "ns2")]
SETS = [("panda_pg2", (3000000, 3000100, 3000200)), ("parm6_tf3", (3000000, 3000100, 3000200)),
        ("parm5s_tf3", (3000000,)), ("parm5l_pg2", (3000000,))]


def load(p):
    try:
        d = json.loads(Path(p).read_text())
        return d["success"], d["n"], d.get("failed_stage", {})
    except (FileNotFoundError, KeyError, ValueError):
        return None


def rec_gen(tg, r, s):
    tag = f"flow{tg}gdag2h_rz{tg}gendag3_noqd"
    p = REC / r / f"generated_zero_{tag}_s{s}.summary.json"
    if not p.exists() and s == 3000000:               # frozen-sem held-out rows were written without the seed suffix
        p = REC / r / f"generated_zero_{tag}.summary.json"
    return load(p)


def new_gen(gc, tg, r, s):
    tag = f"flow{tg}gdag2h_rz{tg}gendag3_noqd"
    return load(NEW / f"grasp_{gc}" / r / f"generated_zero_{tag}_s{s}.summary.json")


def new_bc(gc, r, s):
    return load(NEW / f"grasp_{gc}" / r / f"learned_bc_direct1701_final_s{s}.summary.json")


def agg(cells):
    if any(c is None for c in cells):
        return None
    return sum(c[0] for c in cells), sum(c[1] for c in cells)


res, lines = {}, []
hdr = "| route | " + " | ".join(f"{r} ({'90' if len(ss) == 3 else '30'}) v1 -> v2" for r, ss in SETS) + " | total v1 -> v2 |"
lines += [hdr, "|" + "---|" * (len(SETS) + 2)]
rows = [(name, lambda r, s, tg=tg: rec_gen(tg, r, s), lambda r, s, tg=tg: new_gen("v2", tg, r, s)) for name, tg in ROUTES]
rows.append(("BC direct1701 final (sprint_bc)", lambda r, s: new_bc("v1", r, s), lambda r, s: new_bc("v2", r, s)))
for name, f1, f2 in rows:
    e, t1, t2 = {}, [0, 0], [0, 0]
    cells = []
    for r, ss in SETS:
        a, b = agg([f1(r, s) for s in ss]), agg([f2(r, s) for s in ss])
        e[r] = dict(grasp_v1=a, grasp_v2=b, per_set={s: dict(grasp_v1=f1(r, s), grasp_v2=f2(r, s)) for s in ss})
        fmt = lambda x: f"{x[0]}/{x[1]}" if x else "-"
        cells.append(f"{fmt(a)} -> {fmt(b)}")
        if a and b:
            t1 = [t1[0] + a[0], t1[1] + a[1]]
            t2 = [t2[0] + b[0], t2[1] + b[1]]
    res[name] = e
    lines.append(f"| {name} | " + " | ".join(cells) + f" | {t1[0]}/{t1[1]} -> {t2[0]}/{t2[1]} |")
OUT.write_text(json.dumps(res, indent=1, default=str))
print("\n".join(lines))
