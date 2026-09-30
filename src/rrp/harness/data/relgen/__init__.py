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


def label_runs_in(label: str, caps) -> bool:
    return LABELS[label].needs <= frozenset(caps)


def compose(parts, env: str, rng) -> SceneDraft:
    """The one composition operator (unit R11): union of `activates`, `requires` closed over PARTS, `conflicts`
    rejected, env compatibility checked, layout resolved, task events merged."""
    raise NotImplementedError("relgen.compose is implemented by fanout unit R11 (docs/relations.md section 10)")


def label_record(label: Label, name: str) -> dict:
    """Per-sample provenance entry of one label."""
    return {"label": name, "prov": label.prov, "version": label.version}


Sample = dict[str, Any]   # {"inputs": ..., "labels": {name: Label}, "provenance": {"active": [...], "parts": [...],
                          #  "transforms": [...], "env": ..., "task": ..., "seed": ..., "step": ...}}
