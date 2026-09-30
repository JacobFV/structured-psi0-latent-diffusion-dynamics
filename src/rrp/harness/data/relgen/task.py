"""R18 (D-144, docs/relations.md section 10 row R18; section 6 first wave row `task.next_contact`; catalog:
research/relations_catalog.md F "task, procedure and causality" / J "epistemic"): the PRIVILEGED label `next_contact`
and the `reveal` / `surprise` (R9) glue that turns it into a progressive-supervision target over a candidate list.

The candidate INTERACTION EDGES themselves -- the soft PUBLIC `EdgeSet` a net's collate path emits (manipulator ->
graspable, object -> support / destination) -- are `rrp.policies.nets.batch.candidate_interaction_edges` (this
unit's other owned file); a `bilinear` factor (`task.next_contact`, `catalog.py`) reads token HIDDENS, never an
`EdgeSet` directly (docs 3.2), so this module never imports `nets.batch` (labels stay on the privileged
`StateView` side of the deploy boundary, docs section 7; `policies/` never imports `relgen`).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from rrp.envs.base import StateView
from rrp.harness.data.relgen import Label, LabelDef, TokenIndex, register_label

TASK_LABELS_VERSION = "1"


def manipulator_ids(view: StateView) -> set:
    """Public manipulator entity ids: every bound manipulator assembly is `EntityState.kind == "assembly"`
    (`envs.mujoco.session.Session._manip_entities` / `DualSession`, docs/relations.md 5.1)."""
    return {e.id for e in view.entities() if e.kind == "assembly"}


def touching_entities(view: StateView) -> set:
    """Public-manipulator contact partners right now (`StateView.contacts()`, cap "contacts"): every entity id
    touching -- but not itself -- a manipulator entity. Privileged (simulator contact truth); labels only."""
    manip = manipulator_ids(view)
    out = set()
    for c in view.contacts():
        if c.a in manip and c.b not in manip:
            out.add(c.b)
        elif c.b in manip and c.a not in manip:
            out.add(c.a)
    return out


def _next_contact_fn(view: StateView, index: TokenIndex) -> Label:
    """Ground truth of `task.next_contact` (section 6 first wave; needs cap "contacts"): per token of the `ctx`
    token set, 1.0 iff that token's public entity id is CURRENTLY touching a manipulator, else 0.0. A token with no
    entity id (`index.sets["ctx"][t] is None`: a pad / non-entity slot) is invalid, never a candidate target."""
    ids = index.sets.get("ctx", [])
    touching = touching_entities(view)
    value = np.array([[1.0 if (i is not None and i in touching) else 0.0] for i in ids], dtype=np.float64)
    valid = np.array([i is not None for i in ids], dtype=bool)
    return Label(value=value, valid=valid, prov="gt", version=TASK_LABELS_VERSION)


register_label(LabelDef(name="next_contact", version=TASK_LABELS_VERSION, arity=1, needs=frozenset({"contacts"}),
                        fn=_next_contact_fn, prov="gt"))


# ------------------------------------------------------------------ reveal / surprise glue (docs 5.4; R9 transforms reused unmodified)
def next_contact_sample(candidates: Sequence, view: StateView, *, prior: dict | None = None,
                        evidence: Sequence[dict] | None = None) -> dict:
    """A `relgen.Sample`-shaped dict (`{"inputs": {"candidates", "prior", "evidence"}, "labels": {}, "provenance":
    {}}`) ready for `TRANSFORMS["reveal"]` / `TRANSFORMS["surprise"]` (R9, unmodified): `candidates` names the
    public entity ids one R18 candidate-edge row offers (e.g. the "graspable" row of
    `nets.batch.candidate_interaction_edges` for one manipulator sensor token). When the caller does not supply its
    own `evidence`, the privileged contact truth (`touching_entities`) supplies a single `t=0` exclusion event
    ruling out every OTHER candidate -- but only once something is actually touching: "nothing touching yet" is
    the absence of evidence (uniform prior), never evidence that excludes every candidate (which `reveal` /
    `surprise` would instead read as a contradiction and fall back to "no information", docs 5.4's Bayes posterior
    of section `_posterior` -- a different, wrong signal from genuine not-yet-observed). A caller building a
    genuine multi-step episode instead passes `evidence` (one exclusion event per elapsed step, growing as
    candidates are ruled out over time) for real progressive reveal; `surprise` additionally needs
    `params={"rate": ..., "after": ...}` when applying `TRANSFORMS["surprise"].fn`."""
    touching = touching_entities(view)
    if evidence is None:
        evidence = []
        if touching:
            excluded = [c for c in candidates if c not in touching]
            if excluded:
                evidence = [{"t": 0, "excludes": excluded}]
    return {"inputs": {"candidates": list(candidates), "prior": prior, "evidence": list(evidence)},
            "labels": {}, "provenance": {"label": "next_contact", "prov": "gt", "version": TASK_LABELS_VERSION}}
