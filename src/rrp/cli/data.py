"""ML/data CLI commands."""
from __future__ import annotations

import json
from pathlib import Path


def cmd_data_generate(a):
    from rrp.harness.data.generate import generate
    cfg = json.loads(open(a.config).read())
    if a.workers:
        cfg["workers"] = a.workers
    generate(cfg)


def cmd_data_pack(a):
    from rrp.harness.data.packed import pack_dataset
    cfg = json.loads(open(a.config).read())
    meta = pack_dataset(Path(cfg["dataset"]), Path(a.out), set(cfg["train_robots"]), cfg["horizon"],
                        stride=cfg.get("stride", 1), include_dart_failures=cfg.get("include_dart_failures", False),
                        limit_per_robot=cfg.get("episodes_per_robot"))
    print(json.dumps(meta))


def record_psi0_labels(task: str, out, episodes, *, data_root=None, batch: int = 1, env_kw: dict | None = None):
    """Probe labels of SIMPLE training episodes: `psi0_replay` (recorded rows) x `simple` (MuJoCo only, no rendering,
    `split="train"`) through `harness.rollout` with the `LabelRecorder` hook -> `<out>/episode_<e:06d>.npz` plus one
    JSONL row per episode (`<out>/labels.jsonl`). `env_kw` adds env kwargs (e.g. a worker command in tests). The labels come from env.truth(): source `privileged:sim_replay`,
    labels only, never a policy input. Returns the episodes."""
    from rrp.harness.eval.evaluate import evaluate
    from rrp.policies.psi0 import make_replay
    from rrp.policies.psi0.data import LABEL_SOURCE, LabelRecorder
    name = task.split("/", 1)[-1]
    out = Path(out)
    rec = LabelRecorder(out, mp=name.endswith("MP-v0"))
    return evaluate(make_replay(task=name, data_root=data_root), "simple", f"simple/{name}", "g1_simple", list(episodes),
                    batch=batch, hooks=[rec], out=out / "labels.jsonl", row_extra=dict(label_source=LABEL_SOURCE),
                    env_kw=dict(split="train", render=False, sim_mode="mujoco", **(env_kw or {})))


def _episodes(spec: str) -> list[int]:
    """'0:20' (half-open range) or '0,3,7'."""
    if ":" in spec:
        lo, hi = (int(x) for x in spec.split(":"))
        return list(range(lo, hi))
    return [int(x) for x in spec.split(",")]


def cmd_data_psi0_labels(a):
    eps = record_psi0_labels(a.task, a.out, _episodes(a.episodes), data_root=a.data_root, batch=a.batch)
    print(json.dumps(dict(task=a.task, episodes=len(eps), out=str(a.out))))


def register(sub):
    d = sub.add_parser("data", help="dataset generation").add_subparsers(dest="data_cmd", required=True)
    g = d.add_parser("generate")
    g.add_argument("--config", required=True)
    g.add_argument("--workers", type=int)
    g.set_defaults(fn=cmd_data_generate)
    pk = d.add_parser("pack", help="pack a training config's data into memory-mapped arrays")
    pk.add_argument("--config", required=True)
    pk.add_argument("--out", required=True)
    pk.set_defaults(fn=cmd_data_pack)
    pl = d.add_parser("psi0-labels", help="probe labels of SIMPLE training episodes (psi0_replay + LabelRecorder, MuJoCo only)")
    pl.add_argument("--task", required=True, help="SIMPLE task, e.g. G1WholebodyTabletopGraspMP-v0")
    pl.add_argument("--out", required=True)
    pl.add_argument("--episodes", required=True, help="training episodes: A:B (half-open) or a comma list")
    pl.add_argument("--data-root", default=None, help="LeRobot data root (default: <psi_home>/data/simple)")
    pl.add_argument("--batch", type=int, default=1)
    pl.set_defaults(fn=cmd_data_psi0_labels)
