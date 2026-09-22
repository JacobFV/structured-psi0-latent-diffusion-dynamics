"""Parallel teacher data generation (one broker lease; workers share its CPU quota)."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rrp.data.collect import collect_teacher_episode, write_episode
from rrp.data.manifest import write_manifest

_ROBOT_CACHE = {}


def _job(args):
    robot_key, task, seed, n_distr, out_dir, split, noise = (list(args) + [0.0])[:7]
    eid = f"{task}_{robot_key}_s{seed}" + (f"_dart{int(noise * 1000)}" if noise else "")
    done = Path(out_dir) / "episodes" / f"{eid}.public.pkl.gz"
    if done.exists() and (Path(out_dir) / "episodes" / f"{eid}.private.pkl.gz").exists():
        from rrp.data.collect import read_episode
        import hashlib
        try:   # resume: reuse the completed episode (verified readable)
            meta = read_episode(done)["meta"]
            meta = dict(meta, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                                     for p in (done, done.with_name(f"{eid}.private.pkl.gz"))}, resumed=True)
            return meta
        except Exception:  # noqa: BLE001 - corrupt partial file: regenerate
            pass
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    if robot_key not in _ROBOT_CACHE:
        _ROBOT_CACHE.clear()            # bounded memory: keep one robot per worker
        _ROBOT_CACHE[robot_key] = workbench_robots()[robot_key]()
    robot = _ROBOT_CACHE[robot_key]
    kw = {"n_distractors": n_distr} if task == "pick_place" else {}
    sess = Session(BUILDERS[task](robot, seed, **kw), seed=seed)
    rec = collect_teacher_episode(sess, episode_id=eid, split_lineage=dict(split=split, robot_key=robot_key),
                                  exec_noise=noise, noise_seed=seed)
    rec.public["meta"]["robot_key"] = robot_key
    return write_episode(rec, Path(out_dir) / "episodes")


def generate(config: dict) -> dict:
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for item in config["items"]:
        for k in range(item["episodes"]):
            seed = item["seed_start"] + k
            for noise in [0.0] + list(item.get("dart_noise", config.get("dart_noise", []))):
                jobs.append((item["robot"], item.get("task", "pick_place"), seed,
                             k % (config.get("max_distractors", 2) + 1), str(out), item["split"], noise))
    t0 = time.time()
    metas = []
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=config.get("workers", os.cpu_count()),
                             mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_job, j) for j in jobs]   # jobs are grouped by robot (config order)
        for i, f in enumerate(as_completed(futs)):
            try:
                metas.append(f.result())
            except Exception as e:  # noqa: BLE001 - failures are data too
                metas.append(dict(status="generation_error", error=repr(e)[:300]))
            if i % 200 == 0:
                print(f"[generate] {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    man = write_manifest(out, config["name"], sorted(metas, key=lambda m: m.get("episode_id", "")),
                         extra=dict(config=config, wall_s=time.time() - t0))
    print(json.dumps({"name": config["name"], "status_counts": man["status_counts"], "wall_s": time.time() - t0}))
    return man
