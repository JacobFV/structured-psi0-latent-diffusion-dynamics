"""Replay the W8 anymal_c collection (same seeds, sigmas and teacher draws) with the read-only motion recorder, check
that every episode reproduces (status, steps, ticks), and gate the dataset (report only).
PEER ONLY (D-115): launch with scripts/peer_run.sh ... -- PY scripts/robust_backfill_w8data.py 2"""
import gc, glob, json, sys
from multiprocessing import get_context
D = "artifacts/runs/robust/gates/w8_anymal_c_data"

def job(e):
    from rrp.data.legged_latent_collect import collect_episode
    arr, meta, _ = collect_episode("anymal_c", e["seed"], e["sigma"])
    out = dict(seed=e["seed"], sigma=e["sigma"], body="anymal_c", status=meta["status"], steps=meta["steps"],
               ticks=meta["ticks"], tracker_version=meta["tracker_version"], tracker_sha256=meta["tracker_sha256"],
               motion=meta["motion"], reproduced=(meta["status"], meta["steps"], meta["ticks"]) == (e["status"], e["steps"], e["ticks"]),
               orig=dict(status=e["status"], steps=e["steps"], ticks=e["ticks"]))
    del arr; gc.collect()
    return out

if __name__ == "__main__":
    eps = [e for f in sorted(glob.glob(f"{D}/orig/s*.json")) for e in json.load(open(f))["episodes"]]
    done = {}
    try:
        for l in open(f"{D}/replay.jsonl"):
            r = json.loads(l); done[r["seed"]] = r
    except FileNotFoundError:
        pass
    todo = [e for e in eps if e["seed"] not in done]
    with open(f"{D}/replay.jsonl", "a") as f, get_context("spawn").Pool(int(sys.argv[1]), maxtasksperchild=50) as pool:
        for r in pool.imap_unordered(job, todo):
            f.write(json.dumps(r) + "\n"); f.flush()
    rows = [json.loads(l) for l in open(f"{D}/replay.jsonl")]
    from rrp.evaluation.gates import check_legged_dataset
    rep = check_legged_dataset(rows, dict(name="legged8-anymal_c-v2data/collect_s0 (W8 wave 1), replayed with motion recording"))
    rep["replay"] = dict(n=len(rows), reproduced=sum(r["reproduced"] for r in rows),
                         tracker_sha256=sorted({r["tracker_sha256"] for r in rows}))
    json.dump(rep, open(f"{D}/gate_report.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("verdict", "failed", "replay")}), [(c["name"], c["status"], c["value"], c["note"]) for c in rep["criteria"]])
