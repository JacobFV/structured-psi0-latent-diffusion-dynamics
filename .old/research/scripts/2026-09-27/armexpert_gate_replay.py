"""Post-hoc W6 dataset gate for an arm collection made before the gated collector (D-114): replay every feasible
episode with the SAME generator inputs (robot, seed, distractors, DART noise/seed, teacher version, grasp contact) through
the current collector, which records `motion` (rrp.envs.motion_quality.ArmMotionRecorder), and check that the replay
reproduces the stored episode EXACTLY (all native commands equal). Only exact replays are used for the gate.
Nothing is written into the dataset; the output is <out>.jsonl (one row per episode, resumable) + gate_report.json.
Usage: armexpert_gate_replay.py <dataset_dir> <out_prefix> [workers]"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _job(args):
    ds, metas = args
    from rrp.bodies.catalog import workbench_robots
    from rrp.data.collect import collect_teacher_episode
    from rrp.envs.native import Session
    from rrp.envs.scenario import BUILDERS
    cfg = json.loads((Path(ds) / "data_config.json").read_text()) if (Path(ds) / "data_config.json").exists() else {}
    if cfg.get("grasp_contact"):
        os.environ["RRP_GRASP_CONTACT"] = cfg["grasp_contact"]
    rows, cache = [], {}
    for m in metas:
        rk = m["robot_key"]
        if rk not in cache:
            cache.clear()
            cache[rk] = workbench_robots()[rk]()
        sd = m["seed"]
        sess = Session(BUILDERS["pick_place"](cache[rk], sd, n_distractors=m["n_distractors"]), seed=sd)
        rec = collect_teacher_episode(sess, episode_id=m["episode_id"], exec_noise=m.get("exec_noise", 0.0), noise_seed=sd,
                                      teacher_version=cfg.get("teacher_version"))
        stored = pickle.loads(gzip.decompress((Path(ds) / "episodes" / f"{m['episode_id']}.public.pkl.gz").read_bytes()))
        same = stored["actions"] == rec.public["actions"] and stored["meta"]["status"] == rec.public["meta"]["status"]
        rows.append(dict(episode_id=m["episode_id"], robot_key=rk, seed=sd, exec_noise=m.get("exec_noise", 0.0),
                         status=m["status"], replay_status=rec.public["meta"]["status"], exact_replay=bool(same),
                         physics=rec.public["meta"].get("physics"), motion=rec.public["meta"].get("motion")))
    return rows


def main():
    ds, out = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    man = json.loads((ds / "manifest.json").read_text())
    eps = [e for e in man["episodes"] if e.get("status") in ("success", "failure")]
    rows_path = out.with_suffix(".jsonl")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if rows_path.exists():
        for l in rows_path.read_text().splitlines():
            try:
                done.add(json.loads(l)["episode_id"])
            except ValueError:
                pass
    todo = [e for e in eps if e["episode_id"] not in done][: int(os.environ.get("LIMIT", "1000000"))]
    todo.sort(key=lambda e: (e["robot_key"], e["seed"], e.get("exec_noise", 0.0)))
    chunks = [(str(ds), todo[i:i + 20]) for i in range(0, len(todo), 20)]
    print(f"[gate_replay] {len(done)} done, {len(todo)} to replay in {len(chunks)} chunks", flush=True)
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex, open(rows_path, "a") as f:
        for i, rs in enumerate(ex.map(_job, chunks)):
            for r in rs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"[gate_replay] {i + 1}/{len(chunks)}", flush=True)
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    from rrp.evaluation.gates import check_dataset
    exact = [r for r in rows if r["exact_replay"]]
    rep = check_dataset(man, exact)
    rep["replay"] = dict(episodes=len(rows), exact=len(exact), not_exact=[r["episode_id"] for r in rows if not r["exact_replay"]][:50],
                         source_manifest=str(ds / "manifest.json"), manifest_hash=man.get("manifest_hash"),
                         method="replayed with the same generator inputs through the gated collector; actions compared")
    (out.parent / "gate_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("verdict", "failed")}), rep["replay"]["exact"], "/", rep["replay"]["episodes"], flush=True)


if __name__ == "__main__":
    main()
