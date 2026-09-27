"""D-116 (b) check, read-only (never touches the broker or leases):
1. replay the last N logged peer watchdog samples (ops/watchdog/peer.jsonl) through evaluate() with the deployed config
   (subtract_shmem=False, project = memory.current + GPU as logged) -> reproduces the logged levels;
2. record N live samples (2 s apart) with collect_sample() of this code, which measures both project memory definitions,
   and evaluate each with the OLD (memory.current + GPU) and NEW (anon + shmem + kernel + GPU) project memory.
usage (peer): scripts/peer_run.sh --cpu 1 --mem 1G --label w6_wdreplay -- PY scripts/watchdog_replay_d116.py OUT.json [N]"""
import collections, json, sys, time
from pathlib import Path
from rrp.orchestration.runtime import load_config, ops_root, repo_root
from rrp.orchestration.watchdog import WatchdogConfig, WatchdogState, collect_sample, evaluate

out, N = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 200
c = load_config()["peer"]
wc = WatchdogConfig(memory_reserve_bytes=c["memory_reserve_bytes"], disk_reserve_bytes=c["disk_reserve_bytes"],
                    startup_memory_bytes=c["enforced"]["memory_bytes"], startup_cpu_cores=c["enforced"]["cpu_cores"],
                    disk_path=str(repo_root()), fraction=1.0, psi_full_avg10_shed=101.0, cpu_fraction=1.0,
                    subtract_shmem=False, sample_interval_s=2.0)
log = [json.loads(l) for l in (ops_root() / "ops/watchdog/peer.jsonl").read_text().splitlines()[-N:]]
st = WatchdogState()
replay = [evaluate(r["sample"], wc, st).level for r in log]
res = dict(config=dict(startup=wc.startup_memory_bytes, reserve=wc.memory_reserve_bytes),
           logged=dict(n=len(log), t0=log[0]["t"], t1=log[-1]["t"], levels=dict(collections.Counter(r["level"] for r in log)),
                       replay_levels=dict(collections.Counter(replay)),
                       agree=sum(a == r["level"] for a, r in zip(replay, log)),
                       logged_reasons=dict(collections.Counter(x.split(":")[0].split("_exceeds")[0] for r in log for x in r["reasons"]))))
so, sn, sw = WatchdogState(), WatchdogState(), WatchdogState()
rows = []
for i in range(N):
    s = collect_sample(wc, sw, gpu=True)
    old = dict(s, project_memory=(s["project_memory_current"] + (s.get("project_gpu_bytes") or 0))
               if s.get("project_memory_current") is not None else None)
    vo, vn = evaluate(old, wc, so), evaluate(s, wc, sn)
    rows.append(dict(t=time.time(), avail=s["memory_available"], psi=s["psi_full_avg10"], shmem=s.get("shmem"),
                     p_old=old["project_memory"], p_new=s["project_memory"], src=s["project_memory_source"],
                     gpu=s.get("project_gpu_bytes"), old=vo.level, new=vn.level, live_old=vo.live_memory_bytes,
                     live_new=vn.live_memory_bytes, reasons_old=vo.reasons, reasons_new=vn.reasons))
    time.sleep(2.0)
G = 2 ** 30
res["live"] = dict(n=len(rows), levels_old=dict(collections.Counter(r["old"] for r in rows)),
                   levels_new=dict(collections.Counter(r["new"] for r in rows)),
                   sources=dict(collections.Counter(r["src"] for r in rows)),
                   p_old_gib=[round(min(r["p_old"] for r in rows) / G, 1), round(max(r["p_old"] for r in rows) / G, 1)],
                   p_new_gib=[round(min(r["p_new"] for r in rows) / G, 1), round(max(r["p_new"] for r in rows) / G, 1)],
                   live_old_gib=[round(min(r["live_old"] for r in rows) / G, 1), round(max(r["live_old"] for r in rows) / G, 1)],
                   live_new_gib=[round(min(r["live_new"] for r in rows) / G, 1), round(max(r["live_new"] for r in rows) / G, 1)],
                   psi_max=max(r["psi"] or 0 for r in rows), avail_min_gib=round(min(r["avail"] for r in rows) / G, 1),
                   disagreements=[r for r in rows if r["old"] != r["new"]][:10])
res["rows"] = rows
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
