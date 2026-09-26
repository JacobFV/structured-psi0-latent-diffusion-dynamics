"""Sprint BC learning-curve table from artifacts/runs/baselines_bc_ladder/<robot>/learned_<tag>.summary.json."""
import json
import re
from pathlib import Path

R = Path("artifacts/runs/baselines_bc_ladder")
rows = {}
for f in sorted(R.glob("*/learned_*.summary.json")):
    tag = f.name[len("learned_"):-len(".summary.json")]
    rows.setdefault(tag, {})[f.parent.name] = json.loads(f.read_text())
key = lambda t: (t.split("_u")[0], float("inf") if t.endswith("final") else int(re.sub(r"\D", "", t.split("_u")[1])))
print("| checkpoint | panda_pg2 | parm6_tf3 | failed stages panda / parm6 |\n|---|---|---|---|")
for t in sorted(rows, key=key):
    c = []
    for r in ("panda_pg2", "parm6_tf3"):
        s = rows[t].get(r)
        c.append(f"{s['success']}/{s['n']} [{s['wilson95'][0]:.2f}, {s['wilson95'][1]:.2f}]" if s else "-")
    fs = [", ".join(f"{k} {v}" for k, v in rows[t][r]["failed_stage"].items() if k != "success") if r in rows[t] else "-"
          for r in ("panda_pg2", "parm6_tf3")]
    print(f"| learned:{t} | {c[0]} | {c[1]} | {fs[0] or 'none'} / {fs[1] or 'none'} |")
