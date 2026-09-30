"""Relation data generation (D-144, docs/relations.md section 5): the data side of the relation factors, as
declarative as the factors themselves.

  LABELS      label functions written ONCE against `rrp.envs.base.StateView` (privileged sim state) or public task
              structure; they run in every env whose `StateView.caps` cover their `needs`
  PARTS       composable scene parts: each declares the dynamics / factors it activates, requires and conflicts with,
              and optionally `vary` (decoupling pairs: copies differing only in one factor's value)
  TRANSFORMS  generic, factor-agnostic sample transforms (reveal, surprise, cf_swap, noise, occlude, subsample)
`compose(parts)` is the one composition operator; the `Scheduler` (relgen.curriculum) picks active sets; the
`relations_data` stage (harness.pipelines.relations) writes shards; `harness.data.mix` interleaves them with task data.
Label families live in relgen/{geometry,contact,support,task,body,ui}.py (one module per fanout unit).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

LabelProvKind = str      # "gt" | "task" | "estimator:<name>" | "synthetic:<generator>"


@dataclass
class TokenIndex:
    """Which tokens a label is computed for: per token set, the token slots and their (privileged) entity ids."""
    sets: dict[str, list[str | None]]            # set name -> entity id per token (None = null / not an entity)


@dataclass
class Label:
    value: np.ndarray                            # [T, d] (arity 1) or [T, T, d] (arity 2)
    valid: np.ndarray                            # [T] or [T, T] bool
    prov: str
    version: str


@dataclass(frozen=True)
class LabelDef:
    name: str
    version: str
    arity: int
    needs: frozenset
    fn: Callable                                 # (StateView, TokenIndex) -> Label
    prov: str = "gt"


@dataclass
class SceneDraft:
    """A scene under construction: env id, scene-builder kwargs, entities, task events and the ACTIVE SET (the
    dynamics / factors present). Parts mutate it; `compose` resolves layout and task-graph merge."""
    env: str
    kwargs: dict = field(default_factory=dict)
    entities: list = field(default_factory=list)
    events: list = field(default_factory=list)
    active: frozenset = frozenset()
    parts: tuple = ()
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScenePart:
    name: str
    version: str
    activates: frozenset
    build: Callable                              # (SceneDraft, np.random.Generator) -> None
    requires: frozenset = frozenset()
    conflicts: frozenset = frozenset()
    envs: tuple = ()
    vary: Callable | None = None                 # (SceneDraft, rng, factor) -> list[SceneDraft]


@dataclass(frozen=True)
class TransformDef:
    name: str
    version: str
    fn: Callable                                 # (sample: dict, rng, params: dict) -> list[dict]


LABELS: dict[str, LabelDef] = {}
PARTS: dict[str, ScenePart] = {}
TRANSFORMS: dict[str, TransformDef] = {}


def register_label(d: LabelDef) -> LabelDef:
    LABELS[d.name] = d
    return d


def register_part(p: ScenePart) -> ScenePart:
    PARTS[p.name] = p
    return p


def register_transform(t: TransformDef) -> TransformDef:
    TRANSFORMS[t.name] = t
    return t


def _lookup(table: dict, kind: str, name: str):
    try:
        return table[name]
    except KeyError:
        raise ValueError(f"unknown relgen {kind} {name!r}; registered: {sorted(table)} "
                         f"(call load_families() first if the registry is empty)") from None


def label_def(name: str) -> LabelDef:
    return _lookup(LABELS, "label", name)


def part_def(name: str) -> ScenePart:
    return _lookup(PARTS, "part", name)


def transform_def(name: str) -> TransformDef:
    return _lookup(TRANSFORMS, "transform", name)


def label_runs_in(label: str, caps) -> bool:
    return label_def(label).needs <= frozenset(caps)


_FAMILY_MODULES = ("geometry", "contact", "support", "task", "body", "ui", "transforms")


def load_families() -> None:
    """Import every label / part / transform module (the registries are filled by import side effects) and hand the
    registries to the policy side (`relations.base.resolve(env_caps=..., training=True)` checks factors against
    them without importing harness). Raises if a registry stays empty."""
    import importlib
    from rrp.policies.relations.base import register_data
    for m in _FAMILY_MODULES:
        importlib.import_module(f"{__name__}.{m}")
    for kind, reg in (("labels", LABELS), ("parts", PARTS), ("transforms", TRANSFORMS)):
        if not reg:
            raise RuntimeError(f"relgen {kind} registry is empty after loading {_FAMILY_MODULES}")
    register_data(LABELS, PARTS, TRANSFORMS)


class ComposeError(ValueError):
    """`compose()` could not realize the requested active set: an unsatisfiable `requires`, no registered part
    activates something the set needs, or two chosen parts conflict."""


_MAX_EXACT_COVER = 12   # PARTS below this size: exact minimal cover; at/above it: deterministic greedy cover


def _min_cover(target: frozenset, universe: dict[str, "ScenePart"]) -> tuple[str, ...] | None:
    """Smallest-cardinality subset of `universe` (name -> ScenePart) whose union of `.activates` covers `target`;
    None if no subset covers it. Deterministic tie-break: lexicographic on sorted part names, smallest size first.
    Exact (exhaustive) search below `_MAX_EXACT_COVER` candidates; a size-ranked greedy cover above it (still
    deterministic, not guaranteed minimal at that scale -- the registry stays small in practice, docs/relations.md 5.2)."""
    if not target:
        return ()
    names = sorted(universe)
    if len(names) <= _MAX_EXACT_COVER:
        for r in range(1, len(names) + 1):
            for combo in itertools.combinations(names, r):
                covered = frozenset().union(*(universe[n].activates for n in combo))
                if target <= covered:
                    return combo
        return None
    remaining, chosen, pool = set(target), [], set(names)
    while remaining:
        gains = {n: len(universe[n].activates & remaining) for n in pool}
        best = max(pool, key=lambda n: (gains[n], n), default=None)
        if best is None or gains[best] == 0:
            return None
        chosen.append(best)
        remaining -= universe[best].activates
        pool.discard(best)
    return tuple(sorted(chosen))


def _resolve_layout(entities: list, min_gap: float = 0.15) -> None:
    """Generic, factor-agnostic non-overlapping placement pass (docs/relations.md 5.2): entities a part added with
    an explicit `pos` (x, y, z) and a `radius` / scalar-or-first-axis `extent` are nudged apart along x, in
    increasing-x order, until pairwise clearance >= `min_gap`; entities without geometry (a part that places itself,
    e.g. relative to another entity) are left untouched. Mutates `entities` in place."""
    def _r(e: dict) -> float:
        if "radius" in e:
            return float(e["radius"])
        ext = e.get("extent")
        if isinstance(ext, (list, tuple, np.ndarray)) and len(ext):
            return float(ext[0])
        if isinstance(ext, (int, float)):
            return float(ext)
        return 0.1
    placed = sorted((e for e in entities if isinstance(e, dict) and "pos" in e), key=lambda e: e["pos"][0])
    for i in range(1, len(placed)):
        a, b = placed[i - 1], placed[i]
        need = _r(a) + _r(b) + min_gap
        gap = b["pos"][0] - a["pos"][0]
        if gap < need:
            b["pos"] = (a["pos"][0] + need,) + tuple(b["pos"][1:])


def _merge_events(events: list) -> list:
    """Union-merge task events from every composed part, order-preserving, dropping exact duplicates (a shared
    prerequisite two parts both declare, e.g.)."""
    out, seen = [], set()
    for e in events:
        key = json.dumps(e, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def compose(parts, env: str, rng) -> SceneDraft:
    """The one composition operator (unit R11, docs/relations.md 5.2 + 5.5): `parts` is the desired ACTIVE SET (an
    iterable of dynamics / factor names, e.g. one of `Scheduler.sample`'s active-set entries) -- not a hand-picked
    list of `ScenePart`s. `compose` finds the smallest cover of registered `PARTS` (restricted to this `env`) whose
    `.activates` realize the target, closes their `.requires` transitively by covering whatever is still missing
    (never silently dropping a requirement), rejects any part whose `.conflicts` collides with the final active set,
    builds each chosen part in deterministic (sorted-name) order, resolves a simple non-overlapping layout of the
    entities they added and merges their task events. No hand-written per-combination code: a newly registered part
    or factor is covered automatically the next time its name appears in a target set.

    Raises `ComposeError` if no registered part (for this env) can supply something the target or its closure needs,
    or if the chosen parts conflict. An empty target composes an empty (no-op) `SceneDraft`."""
    target = frozenset(parts)
    universe = {n: p for n, p in PARTS.items() if not p.envs or env in p.envs}
    if not target:
        return SceneDraft(env=env, active=frozenset())

    def _active(names) -> frozenset:
        return frozenset().union(*(universe[n].activates for n in names)) if names else frozenset()

    chosen = list(_min_cover(target, universe) or ())
    if not chosen:
        raise ComposeError(f"no registered part for env {env!r} activates {sorted(target)}")
    active = _active(chosen)
    seen = set(chosen)
    while True:
        needed = frozenset().union(*(universe[n].requires for n in chosen))
        missing = needed - active
        if not missing:
            break
        more = _min_cover(missing, {n: p for n, p in universe.items() if n not in seen})
        if not more:
            raise ComposeError(f"cannot satisfy requires {sorted(missing)} for env {env!r} "
                               f"(active set {sorted(target)}, chosen parts {sorted(chosen)})")
        chosen += list(more)
        seen |= set(more)
        active = _active(chosen)
    for n in chosen:
        p = universe[n]
        bad = p.conflicts & (active - p.activates)
        if bad:
            raise ComposeError(f"part {n!r} conflicts with active {sorted(bad)} (active set {sorted(target)})")
    chosen = sorted(chosen)
    draft = SceneDraft(env=env, active=active, parts=tuple(chosen))
    for n in chosen:
        universe[n].build(draft, rng)
    _resolve_layout(draft.entities)
    draft.events = _merge_events(draft.events)
    return draft


def label_record(label: Label, name: str) -> dict:
    """Per-sample provenance entry of one label."""
    return {"label": name, "prov": label.prov, "version": label.version}


Sample = dict[str, Any]   # {"inputs": ..., "labels": {name: Label}, "provenance": {"active": [...], "parts": [...],
                          #  "transforms": [...], "env": ..., "task": ..., "seed": ..., "step": ...}}


# ------------------------------------------------------------------ `rrp factors coverage`
def _env_caps() -> dict[str, frozenset]:
    """StateView caps per registered env (each env's own declaration; the non-MuJoCo envs set theirs in a heavy
    constructor, so they are literals here, kept equal by tests/unit/test_relations_runtime.py's env-caps check)."""
    from rrp.envs.mujoco.session import MujocoStateView
    tracker = frozenset({"poses", "contacts"})                      # envs.warp.tracker_env / envs.simple
    return {"mujoco/arm": MujocoStateView.CAPS, "mujoco/dual": MujocoStateView.CAPS,
            "mujoco/legged": MujocoStateView.CAPS, "warp/legged": tracker, "simple": tracker,
            "computerworld": frozenset({"poses", "ui_tree"})}


def coverage(out: Any = None) -> dict:
    """Factor x net family x env: does the factor list resolve for the family, does its label run in the env
    (StateView caps), is its scene part / transform available there, and does the training resolve (probe source,
    mix 0.5) pass. Labels absent from `LABELS` (the legacy `probe.*` packet labels) are family-level: no relgen label
    to run. Written to `out` (a path) when given."""
    from rrp.policies.relations import catalog  # noqa: F401  (fills FACTORS / FAMILIES)
    from rrp.policies.relations.base import FACTORS, FAMILIES, FactorError, resolve
    load_families()
    envs = _env_caps()
    emitters = [n for n, d in FACTORS.items() if d.op == "bilinear" and "emits" in d.p]
    rows = {}
    for name, d in sorted(FACTORS.items()):
        if d.status != "implemented":
            continue
        row = {"label": d.label or None, "gen": list(d.gen), "form": d.form, "families": {}, "resolved_as": {},
               "envs": {}, "training": {}}
        for env, caps in envs.items():
            row["envs"][env] = {
                "label_runnable": None if d.label not in LABELS else LABELS[d.label].needs <= caps,
                "parts": {g: (g in TRANSFORMS) or (g in PARTS and (not PARTS[g].envs or env in PARTS[g].envs))
                          for g in d.gen}}
        for fam in FAMILIES:
            # first list that resolves: the factor alone, then with the probe source, then with the emitters it reads
            attempts = [[name], [{"name": name, "source": "probe"}] if "probe" in d.sources else None,
                        emitters + [{"name": name, "source": "probe"} if "probe" in d.sources else name]]
            err, ok = None, None
            for att in attempts:
                if att is None:
                    continue
                try:
                    resolve(att, family=fam)
                    ok = att
                    break
                except FactorError as e:
                    err = err or str(e)
            row["families"][fam] = True if ok else f"refused: {err}"
            if not ok:
                continue
            row["resolved_as"][fam] = ok                     # the factor list that runs this factor in the family
            row["training"][fam] = {}
            for env, caps in envs.items():
                item = {**(ok[-1] if isinstance(ok[-1], dict) else {"name": name}), "mix": 0.5 if d.gen else None}
                try:
                    resolve(ok[:-1] + [item], family=fam, env_caps=caps, env=env, training=True)
                    row["training"][fam][env] = True
                except FactorError as e:
                    row["training"][fam][env] = f"refused: {e}"
        rows[name] = row
    doc = {"version": 1, "envs": {e: sorted(c) for e, c in envs.items()}, "families": sorted(FAMILIES), "factors": rows}
    if out is not None:
        from pathlib import Path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


def coverage_main(argv: list[str]) -> int:
    import argparse
    from rrp.core.paths import rrp_home
    ap = argparse.ArgumentParser(prog="rrp factors")
    ap.add_argument("what", choices=["coverage"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or str(rrp_home() / "artifacts" / "runs" / "relations" / "coverage" / "coverage.json")
    doc = coverage(out)
    n = len(doc["factors"])
    ok = sum(1 for r in doc["factors"].values() if any(v is True for v in r["families"].values()))
    print(f"coverage: {n} factors, {ok} resolve in at least one family -> {out}")
    return 0
