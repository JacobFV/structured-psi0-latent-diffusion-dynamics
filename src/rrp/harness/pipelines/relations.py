"""relations_data: the one relation-factor data stage (D-144, docs/relations.md 5.3, docs/architecture.md 14.2).

For a run's resolved factor list (`rrp.policies.relations.base.resolve`), this stage collects or relabels episode
snapshots through `rrp.envs.base.StateView`, computes each active factor's label (its `FactorDef.label`, a
`rrp.harness.data.relgen.LABELS` entry) and applies its generators (`FactorDef.gen`, names of
`rrp.harness.data.relgen.TRANSFORMS`; a name in `PARTS` is a scene part, recorded but not run here), then writes
labelled shards under `<out_root>/<factor>/<version>/` with a manifest of per-sample provenance (active set, label
versions, transforms, env, task, seed). No per-factor scripts: the same `relations_data` call runs any factor that
has a `label`.

Registered as stage `relations_data` of the families `relations`, `arm`, `legged`, `pointer` and `psi0`, so any of
their recipes takes it as a node. Options: `env`, `task`, `body`, `seeds`, `policy` (default `teacher:<task>`, a
scripted teacher: the labels are privileged truth and the manifest says so), `snapshot_every`, `max_snapshots`,
`batch`, `env_kw`. The shards land under `<out>/relgen/`; the stage manifest carries each shard's hash and the
catalog version. Collection is `SnapshotCollector`, a `harness.rollout` hook (episodes run through
`harness.eval.evaluate`); relabelling recorded snapshots calls `relations_data` with `EpisodeSnapshot`s and needs no
physics.

Strictness: a factor without a `.label` is not a relgen producer (a `bias`/`mask`/`readout` entry over public
fields, or one still `planned`), and a label whose `needs` the view's `caps` do not cover is masked for that
snapshot; both are reported by `producer_status`, never an error. A `label` (or `gen`) naming nothing registered is
an error (`RelgenError`), except for `probe.*` factors, whose labels are family-level readouts owned by the probe
code, not relgen `LABELS`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from rrp.harness.data.mix import RelgenError, write_shard
from rrp.harness.data.relgen import (LABELS, PARTS, TRANSFORMS, Label, Sample, TokenIndex, label_record, label_runs_in,
                                     load_families)
from rrp.harness.pipelines.base import StageContext, StageError, apply_run_context, register_stage
from rrp.policies.relations.base import FactorSpec, get_factor, resolve

STAGE = "relations_data"
STAGE_FAMILIES = ("relations", "arm", "legged", "pointer", "psi0")


@dataclass(frozen=True)
class EpisodeSnapshot:
    """One episode moment to label: env id, task, seed/step (provenance) and the privileged view + token index a
    label needs. `view` duck-types `rrp.envs.base.StateView` (only `.caps` and the accessors the run's labels
    actually call are required, so a fixture StateView is enough for the unit suite)."""
    env: str
    task: str
    seed: int
    view: Any
    index: TokenIndex
    step: int = 0


def _base_sample(ep: EpisodeSnapshot, active: tuple[str, ...]) -> Sample:
    return {"inputs": {"env": ep.env, "task": ep.task, "seed": ep.seed, "step": ep.step, "tokens": ep.index.sets},
            "labels": {},
            "provenance": {"active": list(active), "parts": [], "transforms": [], "labels": [],
                           "env": ep.env, "task": ep.task, "seed": ep.seed, "step": ep.step}}


def _check_names(fdef) -> None:
    """A label or transform naming nothing registered is an error (never a silent skip); `probe.*` factors carry
    family-level readout labels that live outside `LABELS`, and a `gen` in `PARTS` is a scene part (R11 composes it)."""
    load_families()
    if fdef.label and fdef.label not in LABELS and not fdef.name.startswith("probe."):
        raise RelgenError(f"{fdef.name}: label {fdef.label!r} is not in relgen LABELS ({sorted(LABELS)}); "
                          "register it, or fix the factor entry")
    for g in fdef.gen:
        if g not in TRANSFORMS and g not in PARTS:
            raise RelgenError(f"{fdef.name}: gen {g!r} is neither a TRANSFORM ({sorted(TRANSFORMS)}) nor a scene "
                              f"PART ({sorted(PARTS)})")


def producer_status(specs: Sequence[FactorSpec], caps=None) -> dict[str, str | None]:
    """factor -> None when it produces relgen rows, else why not ("no label", "label needs <caps>", "probe label")."""
    out: dict[str, str | None] = {}
    for s in specs:
        d = get_factor(s.name)
        _check_names(d)
        if not d.label:
            out[s.name] = "no label"
        elif d.label not in LABELS:
            out[s.name] = "probe label (family-level readout)"
        elif caps is not None and not label_runs_in(d.label, caps):
            out[s.name] = f"label needs {sorted(LABELS[d.label].needs)}"
        else:
            out[s.name] = None
    return out


def label_episode(ep: EpisodeSnapshot, factor_specs: Sequence[FactorSpec], *,
                  rng: np.random.Generator) -> dict[str, list[Sample]]:
    """Compute + transform the samples of one episode snapshot, one row list per factor that produced any (a factor
    without a `.label`, or whose label's `needs` the view's `caps` does not cover, contributes nothing -- masked,
    not an error; a label / transform that does not exist raises `RelgenError`). A factor's `gen` transforms run in
    order and may multiply one sample into several (e.g. reveal / cf_swap); every resulting row keeps the factor's
    full `active` set and env/task/seed provenance; a `gen` that is a scene part is recorded in `parts`."""
    active = tuple(s.name for s in factor_specs)
    out: dict[str, list[Sample]] = {}
    for fspec in factor_specs:
        fdef = get_factor(fspec.name)
        _check_names(fdef)
        if not fdef.label or fdef.label not in LABELS or not label_runs_in(fdef.label, ep.view.caps):
            continue
        ldef = LABELS[fdef.label]
        label = ldef.fn(ep.view, ep.index)
        sample = _base_sample(ep, active)
        sample["labels"][fdef.label] = label
        sample["provenance"]["labels"] = [label_record(label, fdef.label)]
        sample["provenance"]["parts"] = [g for g in fdef.gen if g in PARTS]
        rows = [sample]
        for gname in fdef.gen:
            tdef = TRANSFORMS.get(gname)
            if tdef is None:
                continue
            nxt: list[Sample] = []
            for row in rows:
                variants = tdef.fn(row, rng, dict(fspec.p)) or [row]
                for v in variants:
                    v["provenance"] = dict(v.get("provenance", row["provenance"]))
                    v["provenance"]["transforms"] = list(v["provenance"].get("transforms", [])) + [gname]
                nxt.extend(variants)
            rows = nxt
        out[fspec.name] = rows
    return out


def _write_all(by_factor: dict[str, list[Sample]], out_root: Path, seed: int) -> dict:
    """One shard per producing factor; the shard id is `seed<seed>-<hash of the rows' provenance>`, so writing the same
    collection again replaces its rows instead of doubling them (resume / rerun)."""
    manifests = {}
    for name, rows in by_factor.items():
        fdef = get_factor(name)
        blob = json.dumps([r["provenance"] for r in rows], sort_keys=True, default=str)
        manifests[name] = write_shard(name, fdef.version, rows, Path(out_root),
                                      shard_id=f"seed{seed}-{hashlib.sha256(blob.encode()).hexdigest()[:8]}")
    return {"factors": manifests}


def relations_data(factors: Sequence[str | dict | FactorSpec] | None, episodes: Iterable[EpisodeSnapshot],
                   out_root: Path | str, *, seed: int = 0, default_preset: str | None = None) -> dict:
    """Relabel recorded snapshots: resolve `factors`, label + transform every episode snapshot, write one shard +
    manifest per producing factor under `<out_root>/<factor>/<version>/`. Returns `{"factors": {name: manifest}}`; a
    run whose factor list produces nothing (no episodes, or no factor with a `.label` runs on any view) writes
    nothing and returns `{"factors": {}}` -- never an error, since which factors are producers depends on the
    catalog sections other units fill in."""
    specs = resolve(factors, default=default_preset)
    rng = np.random.default_rng(seed)
    by_factor: dict[str, list[Sample]] = {}
    for ep in episodes:
        for name, rows in label_episode(ep, specs, rng=rng).items():
            by_factor.setdefault(name, []).extend(rows)
    return _write_all(by_factor, Path(out_root), seed)


class SnapshotCollector:
    """A `harness.rollout` hook: labels the run's factors from `env.state_view()` at reset and every `every` ticks (at
    most `max_per_episode` per episode). Labelling happens at capture time, on the live view, so a view that reads
    the simulator lazily is never labelled from a later state. `seeds` is the seed list the rollout was given
    (episodes reset in that order); rows accumulate in `by_factor`."""

    def __init__(self, env_id: str, task: str, specs: Sequence[FactorSpec], seeds: Sequence[int], *,
                 rng: np.random.Generator, every: int = 10, max_per_episode: int = 8):
        if every < 1 or max_per_episode < 1:
            raise ValueError("every and max_per_episode must be >= 1")
        self.env_id, self.task, self.specs, self.seeds, self.rng = env_id, task, tuple(specs), list(seeds), rng
        self.every, self.max_per_episode = every, max_per_episode
        self.by_factor: dict[str, list[Sample]] = {}
        self._n_reset = 0
        self._ep: dict[int, dict] = {}

    def _capture(self, i: int, env, tick: int) -> None:
        st = self._ep[i]
        view = env.state_view()
        idx = TokenIndex(sets={"ctx": [e.id for e in view.entities()]})
        ep = EpisodeSnapshot(env=self.env_id, task=self.task, seed=st["seed"], view=view, index=idx, step=tick)
        for name, rows in label_episode(ep, self.specs, rng=self.rng).items():
            self.by_factor.setdefault(name, []).extend(rows)
        st["n"] += 1

    def on_reset(self, i, env, obs):
        self._ep[i] = {"seed": int(self.seeds[self._n_reset]), "n": 0, "tick": 0}
        self._n_reset += 1
        self._capture(i, env, 0)

    def on_step(self, i, env, act, step):
        st = self._ep[i]
        st["tick"] += 1
        if st["tick"] % self.every == 0 and st["n"] < self.max_per_episode:
            self._capture(i, env, st["tick"])


# ------------------------------------------------------------------------------------------------ the stage
def _opt_seeds(o: dict) -> list[int]:
    if o.get("seeds") is not None:
        return [int(x) for x in o["seeds"]]
    if o.get("n_episodes") is None:
        raise StageError(f"{STAGE}: options.seeds (a list) or options.n_episodes (+ seed_start) is required")
    return list(range(int(o.get("seed_start", 0)), int(o.get("seed_start", 0)) + int(o["n_episodes"])))


def collect_by_factor(o: dict, specs: Sequence[FactorSpec], *, seed: int) -> tuple[dict[str, list[Sample]], list[str]]:
    """Run `options.policy` (default the task's scripted teacher) on `options.env` / `task` / `body` over the seeds with
    a `SnapshotCollector` hook. Returns the labelled rows per factor and the seeds' episode outcomes."""
    from rrp.harness.eval.evaluate import evaluate
    missing = [k for k in ("env", "task", "body") if not o.get(k)]
    if missing:
        raise StageError(f"{STAGE}: options.{', options.'.join(missing)} required")
    seeds = _opt_seeds(o)
    col = SnapshotCollector(o["env"], o["task"], specs, seeds, rng=np.random.default_rng(seed),
                            every=int(o.get("snapshot_every", 10)), max_per_episode=int(o.get("max_snapshots", 8)))
    eps = evaluate(o.get("policy") or f"teacher:{o['task']}", o["env"], o["task"], o["body"], seeds,
                   batch=int(o.get("batch", 4)), hooks=[col], env_kw=o.get("env_kw"))
    return col.by_factor, [e.outcome for e in eps]


def relations_data_stage(ctx: StageContext) -> dict:
    """relations_data: label the run's factors on teacher-driven snapshots and write hashed shards (`<out>/relgen`)."""
    from rrp.core.provenance import file_digest
    with apply_run_context(ctx.rc) as rctx:
        specs = rctx.specs
    if not specs:
        raise StageError(f"{STAGE}: params.factors resolves to no factors")
    seed = ctx.rc.seed
    by_factor, outcomes = collect_by_factor(ctx.opts, specs, seed=seed)
    root = ctx.out / "relgen"
    res = _write_all(by_factor, root, seed)
    shards = {}
    for name, man in res["factors"].items():
        d = root / name / get_factor(name).version
        shards[f"{name}/{get_factor(name).version}"] = dict(
            manifest_hash=man["manifest_hash"], n_rows=man["n_episodes"],
            npz={f.name: file_digest(f) for f in sorted(d.glob("*.npz"))})
    status = producer_status(specs)
    if not shards:
        raise StageError(f"{STAGE}: no factor produced rows ({ {k: v for k, v in status.items() if v} })")
    return dict(outputs={"shards": str(ctx.rc.out + "/relgen")},
                metrics=dict(shards=shards, n_rows=sum(v["n_rows"] for v in shards.values()),
                             not_producers={k: v for k, v in status.items() if v}, outcomes=outcomes,
                             seeds=_opt_seeds(ctx.opts), driver=ctx.opts.get("policy") or f"teacher:{ctx.opts['task']}"),
                source="privileged_teacher")


def _register() -> None:
    for fam in STAGE_FAMILIES:
        register_stage(fam, STAGE, relations_data_stage, source="privileged_teacher")


_register()
