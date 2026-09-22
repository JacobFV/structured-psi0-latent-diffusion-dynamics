"""Rendered-image VLM feature cache for dataset episodes (peer only: MUJOCO_GL=egl, GPU VLM).

Dataset episodes store public inputs but not pixels. Episodes are regenerated deterministically
(robot key + seed + n_distractors -> same scenario, same scripted-teacher trajectory, exactly as
rrp.data.collect.collect_teacher_episode) and rendered at keyframes t % every == 0. Determinism
is CHECKED against the stored public q0 at every keyframe (max abs deviation recorded; episodes
that diverge are rejected, never silently used).

Cache entries are keyed per frame by (backbone spec key incl. revision/taps, processor hash,
image hash, task-text hash). Stored: fp16 resampler-input tokens [K, L, n_taps, W]
(all visual tokens + last n_text tokens), i.e. bounded per frame.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_ROBOTS = {}


def replay_render(robot_key: str, seed: int, n_distractors: int, task: str = "pick_place", every: int = 4,
                  cameras=("front",), size: int = 256, max_steps: int = 600, ref_q0=None) -> dict:
    """Re-simulate a teacher episode and render keyframes. Mirrors collect_teacher_episode."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rrp.control.teachers import PickPlaceTeacher
    from rrp.data.collect import featurizer_for, privileged_labels
    from rrp.model.backbone import Renderer, task_text
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.native import Session
    from rrp.sim.scenario import BUILDERS
    if robot_key not in _ROBOTS:
        _ROBOTS.clear()
        _ROBOTS[robot_key] = workbench_robots()[robot_key]()
    robot = _ROBOTS[robot_key]
    kw = {"n_distractors": n_distractors} if task == "pick_place" else {}
    sess = Session(BUILDERS[task](robot, seed, **kw), seed=seed)
    feat = featurizer_for(sess)
    teacher = PickPlaceTeacher(sess)
    f = teacher.feasibility()
    rend = Renderer(sess.model, size)
    cams = rend.cameras(list(cameras))
    frames, ts, dev = [], [], 0.0
    obs = sess.observe()
    prev = None
    if f["feasible"]:
        for k in range(max_steps):
            pi = feat(obs, prev)
            if k % every == 0:
                frames.append(rend.render(sess.data, cams))
                ts.append(k)
                if ref_q0 is not None and k < len(ref_q0):
                    dev = max(dev, float(np.max(np.abs(np.asarray(pi.q0) - np.asarray(ref_q0[k])))))
            cmd = teacher.act()
            a = feat.aspace.normalize([cmd.groups], pi.q0)[0]
            privileged_labels(sess, feat)          # keep the exact call sequence of collection
            res = sess.step(cmd)
            obs = res.observation
            prev = a
            if teacher.done:
                break
    rend.close()
    return dict(t=ts, frames=frames, cameras=cams, text=task_text(sess), q0_max_dev=dev, steps=k + 1)


def _job(args):
    eid, robot_key, seed, nd, every, cams, size, ref_q0 = args
    try:
        out = replay_render(robot_key, seed, nd, every=every, cameras=cams, size=size, ref_q0=ref_q0)
        out["eid"] = eid
        return out
    except Exception as e:  # noqa: BLE001
        return dict(eid=eid, error=repr(e)[:300])


def select_episodes(ds_dir: Path, robots: list[str], per_robot: int, offset: int = 0) -> list[dict]:
    """Deterministic selection: success episodes sorted by seed, [offset, offset+per_robot) per robot.
    Works while the manifest is still being written (falls back to episode files)."""
    from rrp.data.collect import read_episode
    out = []
    ep = ds_dir / "episodes"
    for rk in robots:
        files = sorted(ep.glob(f"pick_place_{rk}_s*.public.pkl.gz"), key=lambda p: int(p.name.split("_s")[-1].split(".")[0]))
        got = 0
        for p in files:
            m = read_episode(p)["meta"]
            if m["status"] != "success":
                continue
            if got >= offset:
                out.append(dict(eid=m["episode_id"], robot=rk, seed=m["seed"], n_distractors=m["n_distractors"],
                                path=str(p)))
            got += 1
            if got >= offset + per_robot:
                break
    return out


def build_cache(cfg: dict) -> dict:
    """cfg: dataset, robots, per_robot, offset, every, cameras, size, out_dir, workers, batch, taps."""
    import torch
    from rrp.data.collect import read_episode
    from rrp.model.backbone import BackboneSpec, VLMBackbone, PSI0, FALLBACK, image_hash, text_hash
    from rrp.ops.gpu import apply_cap
    ginfo = apply_cap()
    src = FALLBACK if cfg.get("fallback") else PSI0
    spec = BackboneSpec(source=dict(src), taps=tuple(cfg.get("taps", (16, 28))), n_text=cfg.get("n_text", 8))
    t0 = time.time()
    vlm = VLMBackbone(spec, device="cuda")
    load_s = time.time() - t0
    prov = vlm.provenance()
    out = Path(cfg["out_dir"]) / spec.key()
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    eps = select_episodes(Path(cfg["dataset"]), cfg["robots"], cfg["per_robot"], cfg.get("offset", 0))
    todo = [e for e in eps if not (out / f"{e['eid']}.pt").exists()]
    jobs = []
    for e in todo:
        ref = read_episode(e["path"])["q0"]
        jobs.append((e["eid"], e["robot"], e["seed"], e["n_distractors"], cfg.get("every", 4),
                     tuple(cfg.get("cameras", ["front"])), cfg.get("size", 256), ref))
    stats = dict(frames=0, vlm_s=0.0, rejected=[], errors=[], max_q0_dev=0.0)
    B = cfg.get("batch", 32)
    tol = cfg.get("q0_tol", 1e-6)
    seen_robot = set()
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=cfg.get("workers", 6), max_tasks_per_child=50,
                             mp_context=mp.get_context("spawn")) as ex:   # no fork after CUDA init
        for r in ex.map(_job, jobs):
            if "error" in r:
                stats["errors"].append(r)
                continue
            stats["max_q0_dev"] = max(stats["max_q0_dev"], r["q0_max_dev"])
            if r["q0_max_dev"] > tol:
                stats["rejected"].append(dict(eid=r["eid"], q0_max_dev=r["q0_max_dev"]))
                continue
            toks = []
            t1 = time.time()
            for i in range(0, len(r["frames"]), B):
                fr = r["frames"][i:i + B]
                enc = vlm.encode(fr, [r["text"]] * len(fr))
                toks.append(enc["tokens"])
            torch.cuda.synchronize()
            stats["vlm_s"] += time.time() - t1
            tokens = torch.cat(toks, 0)
            keys = [dict(spec=spec.key(), proc=vlm.proc_hash, img=[image_hash(im) for im in fr_],
                         text=text_hash(r["text"])) for fr_ in r["frames"]]
            torch.save(dict(eid=r["eid"], t=r["t"], tokens=tokens, n_visual=enc["n_visual"], keys=keys,
                            text=r["text"], cameras=r["cameras"], q0_max_dev=r["q0_max_dev"]),
                       out / f"{r['eid']}.pt")
            stats["frames"] += len(r["t"])
            rk = r["eid"].split("pick_place_")[1].rsplit("_s", 1)[0]
            if rk not in seen_robot:        # one real rendered frame per robot kept as evidence
                seen_robot.add(rk)
                from PIL import Image
                Image.fromarray(r["frames"][len(r["frames"]) // 2][0]).save(out / "samples" / f"{r['eid']}.png")
            if stats["frames"] and len(seen_robot) and stats["frames"] % 500 < len(r["t"]):
                print(f"[vlm_cache] {stats['frames']} frames {time.time() - t0:.0f}s", flush=True)
    index = dict(provenance=prov, spec_key=spec.key(), config=cfg, load_s=load_s, gpu=ginfo,
                 episodes=sorted(p.stem for p in out.glob("*.pt")), n_visual=None, stats=stats,
                 peak_gpu_bytes=torch.cuda.max_memory_allocated(), wall_s=time.time() - t0)
    sample = next(out.glob("*.pt"), None)
    if sample is not None:
        s = torch.load(sample)
        index["n_visual"] = s["n_visual"]
        index["token_shape_per_frame"] = list(s["tokens"].shape[1:])
    (out / "index.json").write_text(json.dumps(index, indent=1, default=str))
    print(json.dumps({k: index[k] for k in ("spec_key", "n_visual", "stats", "wall_s")}, default=str)[:2000])
    return index


class FeatureStore:
    """Loads a cache dir into memory; lookup (eid, t) -> tokens [L, T, W] fp16."""

    def __init__(self, cache_dir: Path, eids: set | None = None, expect_spec: str | None = None):
        import torch
        cache_dir = Path(cache_dir)
        self.index = json.loads((cache_dir / "index.json").read_text())
        if expect_spec and self.index["spec_key"] != expect_spec:
            raise ValueError(f"feature cache spec {self.index['spec_key']} != expected {expect_spec}")
        self.rows = {}
        chunks = []
        n = 0
        for p in sorted(cache_dir.glob("*.pt")):
            if eids is not None and p.stem not in eids:
                continue
            d = torch.load(p)
            for i, t in enumerate(d["t"]):
                self.rows[(d["eid"], t)] = n + i
            n += len(d["t"])
            chunks.append(d["tokens"])
            self.n_visual = d["n_visual"]
        self.tokens = torch.cat(chunks, 0) if chunks else None

    def __contains__(self, k):
        return k in self.rows

    def get(self, keys: list[tuple]):
        import torch
        idx = torch.tensor([self.rows[k] for k in keys])
        return self.tokens[idx]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    build_cache(json.loads(Path(a.config).read_text()))


if __name__ == "__main__":
    main()
