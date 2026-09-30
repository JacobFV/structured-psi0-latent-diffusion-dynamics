"""relations_data: the one relation-factor data stage (D-144, docs/relations.md 5.3; fanout unit R10).

For a run's resolved factor list (`rrp.policies.relations.base.resolve`), this stage collects or relabels episode
snapshots through `rrp.envs.base.StateView`, computes each active factor's label (its `FactorDef.label`, a
`rrp.harness.data.relgen.LABELS` entry) and applies its generators (`FactorDef.gen`, names of
`rrp.harness.data.relgen.TRANSFORMS`), then writes labelled shards under `artifacts/relgen/<factor>/<version>/` with
a manifest of per-sample provenance (active set, label versions, transforms, env, task, seed). No per-factor scripts:
the same `relations_data` call runs any factor that has a `label`.

This stage does not talk to a simulator itself -- collection is each env's job (a caller hands it `EpisodeSnapshot`s,
either freshly collected or read back from recorded snapshots for relabelling, per 5.3: "relabelling reads recorded
snapshots so no new physics is needed where snapshots exist"). A factor without a `.label` (a `bias`/`mask`/`readout`
entry over public fields, or one still `status="planned"`) is simply not a relgen producer; it is skipped, not an
error, so this stage runs correctly against catalog sections other units have not filled in yet.

`compose`/scenes (`ScenePart.build`, `.vary`) are unit R11's job (progressive composition, docs/relations.md 5.5);
this stage applies only `TRANSFORMS`, factor-agnostic sample transforms that need no new scene.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from rrp.harness.data.manifest import write_manifest
from rrp.harness.data.relgen import LABELS, TRANSFORMS, Label, Sample, TokenIndex, label_record, label_runs_in
from rrp.policies.relations.base import FactorSpec, get_factor, resolve

SHARD_ROOT = "artifacts/relgen"
MANIFEST_SCHEMA = "relgen-shard-1"


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


def label_episode(ep: EpisodeSnapshot, factor_specs: Sequence[FactorSpec], *,
                  rng: np.random.Generator) -> dict[str, list[Sample]]:
    """Compute + transform the samples of one episode snapshot, one row list per factor that produced any (a factor
    without a `.label`, or whose label's `needs` the view's `caps` does not cover, contributes nothing -- masked,
    not an error). A factor's `gen` transforms run in order and may multiply one sample into several (e.g. reveal /
    cf_swap); every resulting row keeps the factor's full `active` set and env/task/seed provenance."""
    active = tuple(s.name for s in factor_specs)
    out: dict[str, list[Sample]] = {}
    for fspec in factor_specs:
        fdef = get_factor(fspec.name)
        if not fdef.label or fdef.label not in LABELS or not label_runs_in(fdef.label, ep.view.caps):
            continue
        ldef = LABELS[fdef.label]
        label = ldef.fn(ep.view, ep.index)
        sample = _base_sample(ep, active)
        sample["labels"][fdef.label] = label
        sample["provenance"]["labels"] = [label_record(label, fdef.label)]
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


def relations_data(factors: Sequence[str | dict | FactorSpec] | None, episodes: Iterable[EpisodeSnapshot],
                    out_root: Path | str = SHARD_ROOT, *, seed: int = 0, default_preset: str | None = None) -> dict:
    """The stage: resolve `factors`, label + transform every episode snapshot, write one shard + manifest per
    producing factor under `<out_root>/<factor>/<version>/`. Returns `{"factors": {name: manifest}}`; a run whose
    factor list produces nothing (no episodes, or no factor with a `.label` runs on any view) writes nothing and
    returns `{"factors": {}}` -- never an error, since which factors are producers depends on the catalog sections
    other units fill in."""
    specs = resolve(factors, default=default_preset)
    rng = np.random.default_rng(seed)
    by_factor: dict[str, list[Sample]] = {}
    for ep in episodes:
        for name, rows in label_episode(ep, specs, rng=rng).items():
            by_factor.setdefault(name, []).extend(rows)
    out_root = Path(out_root)
    manifests = {}
    for name, rows in by_factor.items():
        fdef = get_factor(name)
        manifests[name] = write_shard(name, fdef.version, rows, out_root, shard_id=f"seed{seed}")
    return {"factors": manifests}


def write_shard(factor: str, version: str, samples: Sequence[Sample], out_root: Path, *, shard_id: str) -> dict:
    """Write one shard (`<factor>.value`/`.valid` arrays per label, per sample, in one `.npz`) plus its manifest
    entry (`rrp.harness.data.manifest.write_manifest`, the one manifest writer): one row per sample with its full
    provenance record (active set, label versions, transforms, env, task, seed) and the array keys that hold it."""
    d = Path(out_root) / factor / version
    d.mkdir(parents=True, exist_ok=True)
    npz_path = d / f"{shard_id}.npz"
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for i, s in enumerate(samples):
        keys = []
        for lname, lab in s["labels"].items():
            vk, ok = f"{i}.{lname}.value", f"{i}.{lname}.valid"
            arrays[vk] = np.asarray(lab.value)
            arrays[ok] = np.asarray(lab.valid)
            keys.append(lname)
        rows.append({"shard": npz_path.name, "row": i, "status": "ok", "label_keys": keys, **s["provenance"]})
    np.savez_compressed(npz_path, **arrays)
    existing = read_shard_manifest(d) or {}
    all_rows = (existing.get("episodes") or []) + rows
    return write_manifest(d, factor, all_rows, {"schema": MANIFEST_SCHEMA, "factor": factor, "version": version},
                          filename="manifest.json")


def read_shard_manifest(shard_dir: Path) -> dict | None:
    from rrp.harness.data.manifest import read_manifest
    p = Path(shard_dir) / "manifest.json"
    return read_manifest(p) if p.exists() else None


def load_shard_rows(factor: str, version: str, out_root: Path | str = SHARD_ROOT) -> list[Sample]:
    """Read one factor/version's shard rows back into `Sample`s (labels rehydrated as `Label`s) -- what
    `harness.data.mix.mixed_batches` pools from (unit R10)."""
    d = Path(out_root) / factor / version
    man = read_shard_manifest(d)
    if not man:
        return []
    by_shard: dict[str, np.lib.npyio.NpzFile] = {}
    out = []
    for row in man["episodes"]:
        shard = row["shard"]
        if shard not in by_shard:
            by_shard[shard] = np.load(d / shard)
        z = by_shard[shard]
        i = row["row"]
        labels = {}
        for lname in row.get("label_keys", []):
            labels[lname] = Label(value=z[f"{i}.{lname}.value"], valid=z[f"{i}.{lname}.valid"],
                                  prov=next((r["prov"] for r in row.get("labels", []) if r["label"] == lname), "gt"),
                                  version=next((r["version"] for r in row.get("labels", []) if r["label"] == lname), ""))
        prov = {k: v for k, v in row.items() if k not in ("shard", "row", "status", "label_keys")}
        out.append({"inputs": {}, "labels": labels, "provenance": prov})
    return out
