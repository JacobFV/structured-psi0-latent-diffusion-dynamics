"""Parallel teacher data generation (one broker lease; workers share its CPU quota)."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rrp.harness.data.collect import collect_teacher_episode, write_episode
from rrp.harness.data.manifest import write_manifest, dataset_provenance
from rrp.core.provenance import FEATURIZER_VERSION

_ROBOT_CACHE = {}


def _job(args):
    args = list(args)
    args += [0.0, None, None][len(args) - 6:]     # defaults: noise 0.0, patient None (unpaired), teacher v1 default
    robot_key, task, seed, n_distr, out_dir, split, noise, patient, teacher_version = args[:9]
    gc = args[9] if len(args) > 9 else None
    if gc:                                   # grasp contact version for this worker's scenes (rrp.bodies.grasp_contact)
        os.environ["RRP_GRASP_CONTACT"] = gc
    ds = args[10] if len(args) > 10 else None
    if ds:                                   # D-118 contact-safe DART variant ("proximity")
        os.environ["RRP_DART_SAFETY"] = ds
    extra = (args[11] if len(args) > 11 else None) or {}    # D-126 options (absent = historical jobs)
    ekw = {}
    if extra.get("dart_descent_sigma"):
        ekw["dart_descent_sigma"] = float(extra["dart_descent_sigma"])
    if extra.get("teacher_kw"):
        ekw["teacher_kw"] = dict(extra["teacher_kw"])
    eid = f"{task}_{robot_key}_s{seed}" + (f"_p{patient}" if patient is not None else "") + \
        (f"_dart{int(noise * 1000)}" if noise else "")
    done = Path(out_dir) / "episodes" / f"{eid}.public.pkl.gz"
    if done.exists() and (Path(out_dir) / "episodes" / f"{eid}.private.pkl.gz").exists():
        from rrp.harness.data.collect import read_episode
        import hashlib
        try:   # resume: reuse the completed episode (verified readable)
            meta = read_episode(done)["meta"]
            meta = dict(meta, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                                     for p in (done, done.with_name(f"{eid}.private.pkl.gz"))}, resumed=True)
            return meta
        except Exception:  # noqa: BLE001 - corrupt partial file: regenerate
            pass
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.envs.mujoco.session import Session
    if robot_key not in _ROBOT_CACHE:
        _ROBOT_CACHE.clear()            # bounded memory: keep one robot per worker
        _ROBOT_CACHE[robot_key] = workbench_robots()[robot_key]()
    robot = _ROBOT_CACHE[robot_key]
    kw = {"n_distractors": n_distr} if task == "pick_place" else {}
    if extra.get("object_variation"):        # D-126 #35: per-episode task-object spec (own RNG stream per seed)
        if task != "pick_place":
            raise ValueError("object_variation is implemented for pick_place only")
        from rrp.envs.mujoco.scenario import sample_object_spec
        kw["object_spec"] = sample_object_spec(extra["object_variation"], seed)
    if task == "pick_place_paired":      # n_distr = number of objects - 1; patient = assigned physical cube
        kw = {"n_objects": n_distr + 1, "patient": patient}
    sess = Session(BUILDERS[task](robot, seed, **kw), seed=seed)
    rec = collect_teacher_episode(sess, episode_id=eid, split_lineage=dict(split=split, robot_key=robot_key),
                                  exec_noise=noise, noise_seed=seed, teacher_version=teacher_version,
                                  dart_safety=os.environ.get("RRP_DART_SAFETY") or None, **ekw)
    rec.public["meta"]["robot_key"] = robot_key
    if task == "pick_place_paired":
        rec.public["meta"]["pair"] = dict(sess.scenario.meta, pair_id=f"{robot_key}_s{seed}" +
                                          (f"_dart{int(noise * 1000)}" if noise else ""))
    return write_episode(rec, Path(out_dir) / "episodes")


def _extra_options(config: dict) -> dict | None:
    """D-126 collection options (all default OFF; None when none is set, so historical job tuples are unchanged):
    dart_descent_sigma (#5; needs dart_safety "phase"), ik_limit_margin (#8; needs teacher_version "v2lim"),
    object_variation (#35; rrp.envs.scenario.sample_object_spec ranges, pick_place only)."""
    extra = {}
    if config.get("dart_descent_sigma"):
        if config.get("dart_safety") != "phase":
            raise ValueError("dart_descent_sigma needs dart_safety 'phase'")
        extra["dart_descent_sigma"] = float(config["dart_descent_sigma"])
    if config.get("ik_limit_margin"):
        extra["teacher_kw"] = dict(ik_limit_margin=float(config["ik_limit_margin"]))
    if config.get("object_variation"):
        extra["object_variation"] = dict(config["object_variation"])
    return extra or None


def generate(config: dict) -> dict:
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    tv = config.get("teacher_version")          # None = the v1 default teacher (historical datasets)
    gcv = config.get("grasp_contact")           # None = $RRP_GRASP_CONTACT or grasp_v1 (historical datasets)
    dsv = config.get("dart_safety")             # None = unguarded DART (historical datasets); "proximity" (D-118)
    extra = _extra_options(config)              # D-126: None for every historical config (job tuples unchanged)
    if extra and extra.get("teacher_kw") and tv not in ("v2lim",):
        raise ValueError("ik_limit_margin needs teacher_version 'v2lim' (a teacher version bump, D-126 #8)")
    for item in config["items"]:
        for k in range(item["episodes"]):
            seed = item["seed_start"] + k
            task = item.get("task", config.get("task", "pick_place"))
            for noise in [0.0] + list(item.get("dart_noise", config.get("dart_noise", []))):
                if task == "pick_place_paired":   # every assignment of the same physical scene (balanced pairs)
                    lo, hi = config.get("paired_objects", [2, 3])
                    n_obj = lo + k % (hi - lo + 1)
                    for p in range(n_obj):
                        jobs.append((item["robot"], task, seed, n_obj - 1, str(out), item["split"], noise, p, tv, gcv, dsv)
                                    + ((extra,) if extra else ()))
                    continue
                jobs.append((item["robot"], task, seed,
                             k % (config.get("max_distractors", 2) + 1), str(out), item["split"], noise, None, tv, gcv, dsv)
                            + ((extra,) if extra else ()))
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
    metas = sorted(metas, key=lambda m: m.get("episode_id", ""))
    src = "scripted_teacher"
    flags = dict(privileged_teacher=True, dart_noise=config.get("dart_noise", []))
    if tv:
        from rrp.policies.teachers.arm_smooth import teacher_source, teacher_version_id
        src = teacher_source(tv)
        flags["teacher_version"] = teacher_version_id(tv)
    gvs = sorted({(m.get("physics") or {}).get("grasp_contact_version") or "grasp_v1" for m in metas if m.get("physics")})
    flags["grasp_contact_version"] = gvs[0] if len(gvs) == 1 else gvs
    if dsv:
        flags["dart_safety"] = dict(mode=dsv, margin_m=0.005)
    if extra and extra.get("dart_descent_sigma"):
        flags["dart_safety"]["descent_sigma"] = float(extra["dart_descent_sigma"])
    if extra and extra.get("teacher_kw"):
        flags["teacher_kw"] = dict(extra["teacher_kw"])
    if extra and extra.get("object_variation"):
        from rrp.envs.mujoco.scenario import OBJECT_SPEC_VERSION
        flags["object_variation"] = dict(extra["object_variation"], version=OBJECT_SPEC_VERSION)
    prov = dataset_provenance(metas, source=src, featurizer_version=FEATURIZER_VERSION, flags=flags)
    man = write_manifest(out, config["name"], metas, extra=dict(config=config, wall_s=time.time() - t0,
                                                                source=prov.source, privileged_teacher=True,
                                                                featurizer=FEATURIZER_VERSION), provenance=prov)
    print(json.dumps({"name": config["name"], "status_counts": man["status_counts"], "wall_s": time.time() - t0}))
    return man
