"""W5 parity: compare pipeline re-runs with the recorded legacy outputs.
usage: w5_parity_compare.py OUT.json rows NEW.jsonl OLD.jsonl [rows NEW OLD ...] [train NEW_DIR OLD_DIR ...]
rows : per-episode ladder rows, matched by seed; every key compared (floats exactly), differing keys listed.
train: train_log.jsonl (every logged step, every key except wall time 't') and result.json leaves; with torch available
       also the weights digest of representation.pt (contracts.provenance.weights_digest)."""
import json
import sys
from pathlib import Path

VOLATILE = {"t", "wall_s", "generated_at", "wall", "elapsed_s", "time_s"}


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{p}.{k}" if p else k)
    elif isinstance(x, list) and all(not isinstance(v, (dict, list)) for v in x):
        yield p, tuple(x)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, x


def diff(a, b, skip=VOLATILE):
    la, lb = dict(leaves(a)), dict(leaves(b))
    out = {}
    for k in sorted(set(la) | set(lb)):
        if k.split(".")[-1] in skip:
            continue
        if la.get(k) != lb.get(k):
            out[k] = (la.get(k), lb.get(k))
    return out


def rows(new, old):
    rn = {r["seed"]: r for r in map(json.loads, Path(new).read_text().splitlines()) if r}
    ro = {r["seed"]: r for r in map(json.loads, Path(old).read_text().splitlines()) if r}
    per = {s: diff(rn[s], ro[s]) for s in sorted(set(rn) & set(ro))}
    keys = sorted({k for d in per.values() for k in d})
    return dict(new=new, old=old, n_new=len(rn), n_old=len(ro), seeds_equal=sorted(rn) == sorted(ro),
                success_new=sum(r["privileged_success"] for r in rn.values()),
                success_old=sum(r["privileged_success"] for r in ro.values()),
                outcome_equal=all(rn[s]["outcome"] == ro[s]["outcome"] for s in per),
                identical_rows=sum(1 for d in per.values() if not d), differing_keys=keys,
                example={k: next(d[k] for d in per.values() if k in d) for k in keys[:8]})


def train(new, old):
    new, old = Path(new), Path(old)
    ln = [json.loads(l) for l in (new / "train_log.jsonl").read_text().splitlines() if l.strip()]
    lo = [json.loads(l) for l in (old / "train_log.jsonl").read_text().splitlines() if l.strip()]
    lo = [r for r in lo if r.get("step") in {x.get("step") for x in ln}] if len(lo) > len(ln) else lo
    step_diffs = [(a.get("step"), diff(a, b)) for a, b in zip(ln, lo) if diff(a, b)]
    maxrel = 0.0
    for a, b in zip(ln, lo):
        for k in a:
            if k in VOLATILE or not isinstance(a[k], float) or not isinstance(b.get(k), float):
                continue
            maxrel = max(maxrel, abs(a[k] - b[k]) / max(abs(b[k]), 1e-12))
    rn, ro = json.loads((new / "result.json").read_text()), json.loads((old / "result.json").read_text())
    out = dict(new=str(new), old=str(old), log_rows=(len(ln), len(lo)), identical_log_rows=len(ln) - len(step_diffs),
               first_diff_step=step_diffs[0][0] if step_diffs else None, max_rel_diff_logged=maxrel,
               result_diffs=diff(rn, ro))
    try:
        import torch
        from rrp.contracts.provenance import weights_digest
        dg = []
        for d in (new, old):
            st = torch.load(d / "representation.pt", map_location="cpu", weights_only=False)
            m = st.get("model", st)
            dg.append({k: weights_digest(v) for k, v in m.items() if isinstance(v, dict)} if isinstance(m, dict) else None)
        out["weights_digest_new_old"] = dg
        out["weights_identical"] = dg[0] == dg[1]
    except Exception as e:  # noqa: BLE001
        out["weights_digest_error"] = repr(e)
    return out


if __name__ == "__main__":
    res, a = [], sys.argv[2:]
    while a:
        kind, n, o, a = a[0], a[1], a[2], a[3:]
        res.append(dict(kind=kind, **(rows(n, o) if kind == "rows" else train(n, o))))
    Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])
