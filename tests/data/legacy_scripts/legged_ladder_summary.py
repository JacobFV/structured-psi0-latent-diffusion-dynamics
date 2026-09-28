"""Summarize legged ladder rows: success / stages / Wilson 95% per file; writes <file>.summary.json next to each."""
import json, math, sys
from pathlib import Path

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))

for f in sys.argv[1:]:
    if f.endswith("summary.json"):
        continue
    rows = [json.loads(l) for l in open(f)]
    if not rows:
        continue
    st = {}
    for r in rows:
        st[r["failure_stage"]] = st.get(r["failure_stage"], 0) + 1
    k = sum(r["success"] for r in rows)
    s = dict(file=f, source=rows[0]["source"], edit=rows[0]["edit"], n=len(rows), success=k, wilson95=wilson(k, len(rows)),
             stages=st, seeds=sorted(r["seed"] for r in rows)[:1] + sorted(r["seed"] for r in rows)[-1:],
             events_a=sum(r["events"].get("walk_to_a") == "succeeded" for r in rows),
             events_b=sum(r["events"].get("walk_to_b") == "succeeded" for r in rows))
    Path(f).with_suffix(".summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s))
