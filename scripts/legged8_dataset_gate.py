"""D-112 legged dataset gate over collected shards (rrp.pipelines.legged.dataset_gate), with an optional bit-exactness
check of a REPLAY against the original shards: datasets collected before the collector recorded motion metrics are
re-collected (same seeds; the collection is deterministic given the seed) with the motion recorder attached, every
array is compared to the stored shard, and only then are the replay's slip metrics attributed to the stored dataset.
usage: python scripts/legged8_dataset_gate.py OUT.json DATA_DIR BODY [REFERENCE_DATA_DIR]"""
import json
import sys
from pathlib import Path

import numpy as np

from rrp.pipelines.legged import dataset_gate

out, d, body = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ref = Path(sys.argv[4]) if len(sys.argv) > 4 else None
shards = sorted(p for p in (d / body).glob("s*-*.json") if not p.name.endswith(".manifest.json"))
eps = [e for sh in shards for e in json.loads(sh.read_text())["episodes"]]
res = dict(data=str(d), body=body, shards=[s.name for s in shards], gate=dataset_gate(eps))
by_sigma = {}
for e in eps:
    k = str(e["sigma"])
    sr = (e.get("motion") or {}).get("slip_ratio")
    b = by_sigma.setdefault(k, dict(n=0, slip_lt_0p15=0, fell=0, slip=[]))
    b["n"] += 1; b["fell"] += e["status"] == "fell"
    if sr is not None:
        b["slip_lt_0p15"] += sr < 0.15; b["slip"].append(sr)
for b in by_sigma.values():
    s = b.pop("slip")
    b["slip_median"] = round(float(np.median(s)), 4) if s else None
res["by_sigma"] = by_sigma
res["tracker_sha256"] = sorted({str(e.get("tracker_sha256")) for e in eps})
res["actuator_limits"] = sorted({str((e.get("physics") or {}).get("actuator_limits")) for e in eps})
if ref is not None:
    cmp = []
    for sh in shards:
        a = np.load(sh.with_suffix(".npz")); r = np.load((ref / body / sh.name).with_suffix(".npz"))
        keys = sorted(set(a.files) | set(r.files))
        diff = [k for k in keys if k not in a.files or k not in r.files or a[k].shape != r[k].shape
                or not np.array_equal(a[k], r[k])]
        cmp.append(dict(shard=sh.name, identical=not diff, differing_arrays=diff))
    res["replay_vs_reference"] = dict(reference=str(ref), all_identical=all(c["identical"] for c in cmp), shards=cmp)
    if not res["replay_vs_reference"]["all_identical"]:
        res["gate"]["passed"] = None
        res["gate"]["note"] = "replay differs from the stored dataset: slip metrics NOT attributable"
out.write_text(json.dumps(res, indent=1))
print(json.dumps(dict(gate=res["gate"], by_sigma=by_sigma, replay=res.get("replay_vs_reference", {}).get("all_identical")), indent=1))
