"""System II harness (roadmap #31, D-126): language instruction -> system II -> public task-context TARGETS -> evaluation.

Library home of the system-II evaluation (promoted from `rrp.research.system2_eval`, which keeps working unchanged).
Harness only: nothing here trains a model; default-off everywhere (`make_system2("off") is None`, a no-op target leaves
the context byte-identical).

Pipeline
  instruction text (+ the declared public `system2` camera image at the initial state, when the system reads images)
    -> `System2.ground(instruction, image)` -> `System2Target`
    -> targets on the public task context that system i conditions on:
         order        : which marker is bound to the first / second `walk_to` event. Enters system i through the PUBLIC
                        task graph (entity descriptors -> detector slots -> body-frame waypoint estimates ctx[8:16]),
                        i.e. `scenario_for(body, seed, truth, target)`; exactly what `rrp.research.system2_eval` did.
         halt         : task view set to `halt` (ctx[16:20] one-hot), same math as the legged eval's `halt` edit.
         mirror_goal  : lateral waypoint estimates mirrored (ctx[9], ctx[13]), same math as the `mirror_goal` edit.
    -> evaluation: `ground_eval` (grounding accuracy of the order vs the instruction truth, blank/shuffled image
       controls) and `closed_loop_eval` (the existing legged latent route with the system-II-produced binding instead
       of the declared task view; privileged evaluator judges against the instruction truth).

Source labels (on every target and row; unmistakable):
  oracle_diagnostic:instruction_truth   OracleSystem2 — derived from the instruction TRUTH (privileged). DIAGNOSTIC only.
  default:declared_task_binding         DefaultSystem2 — the task file's fixed binding (orange first); instruction ignored.
  mock:<name>                           MockSystem2 — tests / plumbing only.
  learned:<vlm ckpt>#<readout>          VLMSystem2 — frozen psi0 System-II VLM (+ optional trained probe readout).

Public vs privileged: only OracleSystem2 receives `privileged_truth`; the harness passes it to no other system
(`System2.needs_truth`). The VLM sees the instruction text and the declared public camera image only.

usage: python -m rrp.harness.eval.system2 ground --system vlm:<local_dir> --out artifacts/runs/X [--small]
       python -m rrp.harness.eval.system2 closed_loop --flow F/policy.pt --systems oracle,default,vlm:<dir> --out artifacts/runs/X
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from rrp.policies.nets.system2_vlm import COLORS

SYSTEM2_HARNESS_VERSION = "s2h-1"

DECLARED_ORDER = ("orange", "cyan")      # the task file's fixed binding: waypoint_a = orange marker (system i before #31)

# legged public context layout (rrp.features.legged.public_context, GLOBAL_DIM = 22)
CTX_DIM = 22
CTX_WP_A = slice(8, 12)                  # (bx/2, by/2, min(dist,5)/2, valid) of the entity bound to the 1st walk_to
CTX_WP_B = slice(12, 16)                 # same for the 2nd walk_to
CTX_LATERAL = (9, 13)                    # body-frame lateral estimates (a, b) -> mirror_goal
CTX_EVENT = slice(16, 20)                # one-hot over EVENTS + done: walk_to_a, walk_to_b, halt, done
EVENT_HALT = 2

# ---------------------------------------------------------------- instructions and the declared public scene view
TEMPLATES = {
    "color": [
        "Walk to the {a} marker, then walk to the {b} marker and stop there.",
        "Go to the {a} post first. After that, go to the {b} post and halt.",
        "First visit the {a} waypoint, then the {b} waypoint, then stand still.",
        "Head over to the {a} marker. Next, continue to the {b} marker and stop.",
        "Your route: {a} marker first, {b} marker second. Stop at the end.",
        "Before going to the {b} marker, walk to the {a} marker. Finish at the {b} one.",
    ],
    "distance": [
        "Walk first to the marker that is {rel} to you, then to the other marker, and stop.",
        "Go to the {rel} of the two posts first, then the remaining one, then halt.",
        "Visit the {rel} waypoint first and the other waypoint second.",
        "Start with whichever marker is {rel} from you; end at the other one.",
    ],
}
HELD_OUT_TEMPLATES = {"color": {4, 5}, "distance": {3}}


def make_instruction(seed: int, family: str, waypoints: dict, template: int | None = None):
    """Returns (text, truth_order (first, second) colors, template index). waypoint_a is orange, b is cyan
    physically (scene builder); the robot starts at the origin."""
    rng = random.Random(seed * 7 + (0 if family == "color" else 1))
    ts = TEMPLATES[family]
    ti = rng.randrange(len(ts)) if template is None else template
    if family == "color":
        first = rng.choice(COLORS)
        second = COLORS[1 - COLORS.index(first)]
        text = ts[ti].format(a=first, b=second)
    else:
        da = math.hypot(*waypoints["a"]); db = math.hypot(*waypoints["b"])
        rel = rng.choice(("nearest", "farthest"))
        text = ts[ti].format(rel=rel)
        near = "orange" if da < db else "cyan"
        first = near if rel == "nearest" else COLORS[1 - COLORS.index(near)]
        second = COLORS[1 - COLORS.index(first)]
    return text, (first, second), ti


def public_task(order):
    """Task graph with the system-II binding: entity descriptors in the instructed/predicted order."""
    from rrp.envs.mujoco.scenario import load_task
    t = copy.deepcopy(load_task("waypoint_contact"))
    for d in t["entity_declarations"]:
        if d["id"] == "waypoint_a":
            d["descriptor"] = f"{order[0]} waypoint marker"
        if d["id"] == "waypoint_b":
            d["descriptor"] = f"{order[1]} waypoint marker"
    return t


def scenario_with_truth(body, seed, truth_order, public_order):
    """Scene is unchanged (orange at a, cyan at b). PUBLIC task graph = public_order binding; PRIVILEGED
    ObjectDecl.task_entity mapping = the instruction truth (used only by the privileged evaluator)."""
    from rrp.envs.mujoco.legged import build_waypoint_contact
    from rrp.envs.mujoco.scenario import ObjectDecl
    sc = build_waypoint_contact(body, seed, task=public_task(public_order))
    phys = {"orange": "waypoint_a", "cyan": "waypoint_b"}
    ent = {truth_order[0]: "waypoint_a", truth_order[1]: "waypoint_b"}
    sc.objects = [ObjectDecl(phys[c], f"{c} waypoint marker", "feature", radius=0.12, task_entity=ent[c])
                  for c in COLORS]
    return sc


SYSTEM2_CAMERA = dict(lookat=(1.8, 0.0, 0.0), distance=8.5, azimuth=0.0, elevation=-72.0)   # declared, static


def render_public(sc, size=384, camera="system2"):
    """Declared public scene camera at the initial state (the system-II input image). `system2` is a fixed,
    declared virtual camera behind/above the start pose looking over the workspace (static world pose)."""
    import mujoco
    d = mujoco.MjData(sc.model)
    mujoco.mj_forward(sc.model, d)
    # place the body at its default standing pose for the render
    from rrp.envs.mujoco.legged_core import LeggedBinding
    b = LeggedBinding(sc.model, sc.robots[0].meta, sc.robots[0].prefix)
    b.set_default(d)
    mujoco.mj_forward(sc.model, d)
    r = mujoco.Renderer(sc.model, size, size)
    if camera == "system2":
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = SYSTEM2_CAMERA["lookat"]
        cam.distance, cam.azimuth, cam.elevation = (SYSTEM2_CAMERA[k] for k in ("distance", "azimuth", "elevation"))
        r.update_scene(d, camera=cam)
    else:
        r.update_scene(d, camera=camera)
    img = r.render().copy()
    r.close()
    return img


# ---------------------------------------------------------------- targets
@dataclass(frozen=True)
class System2Target:
    """System-II output: targets on the public task context of system i. None / False fields = no change."""
    source: str                                   # see module doc; never empty
    order: tuple[str, str] | None = None          # (first, second) marker colors bound to walk_to_a / walk_to_b
    halt: bool = False                            # task view -> halt
    mirror_goal: bool = False                     # mirror lateral waypoint estimates (goal side)
    diagnostic: bool = False                      # True for truth-derived (oracle) targets
    instruction: str | None = None
    confidence: float | None = None               # e.g. VLM logit(orange) - logit(cyan)
    info: dict = field(default_factory=dict)      # readout details (generation, top1, ...)
    harness_version: str = SYSTEM2_HARNESS_VERSION

    def __post_init__(self):
        if not self.source:
            raise ValueError("System2Target.source must be a non-empty label")
        if self.order is not None:
            o = tuple(self.order)
            if sorted(o) != sorted(COLORS):
                raise ValueError(f"order must be a permutation of {COLORS}, got {o}")
            object.__setattr__(self, "order", o)
        if self.source.startswith("oracle") and not self.diagnostic:
            raise ValueError("oracle-sourced targets must be marked diagnostic=True")

    def binding(self, declared=DECLARED_ORDER) -> tuple[str, str]:
        return self.order if self.order is not None else tuple(declared)

    def context_edits(self) -> bool:
        """True when the target changes ctx entries directly (halt / mirror_goal); order enters via the task graph."""
        return bool(self.halt or self.mirror_goal)

    def to_row(self) -> dict:
        d = asdict(self)
        d["order"] = list(self.order) if self.order is not None else None
        return d


def apply_target(ctx: np.ndarray, target: System2Target | None, *, declared_order=DECLARED_ORDER,
                 binding_in_ctx: bool = False) -> np.ndarray:
    """Pure: legged public context vector(s) [..., 22] -> edited copy. `target=None` (system II off) or a target with
    no edits returns `ctx` itself, unchanged.

    halt        : ctx[..., 16:20] = one_hot(halt)                      (== LatentLeggedController._ctx mode "halt")
    mirror_goal : ctx[..., 9], ctx[..., 13] negated                    (== mode "mirror_goal")
    order       : ONLY with binding_in_ctx=True (the session was built with `declared_order` and the binding is applied
                  at the context level): swaps the waypoint slots ctx[8:12] <-> ctx[12:16] when order != declared.
                  Caveat: the public runtime's event progression still follows the session's task graph, so the
                  faithful route is the task-graph binding (`scenario_for`); binding_in_ctx is a context-only probe.
    """
    if target is None:
        return ctx
    swap = binding_in_ctx and target.order is not None and tuple(target.order) != tuple(declared_order)
    if not (swap or target.halt or target.mirror_goal):
        return ctx
    c = np.array(ctx, copy=True)
    if c.shape[-1] != CTX_DIM:
        raise ValueError(f"legged public context has {CTX_DIM} entries, got {c.shape}")
    if swap:
        a = c[..., CTX_WP_A].copy()
        c[..., CTX_WP_A] = c[..., CTX_WP_B]
        c[..., CTX_WP_B] = a
    if target.mirror_goal:
        for i in CTX_LATERAL:
            c[..., i] = -c[..., i]
    if target.halt:
        c[..., CTX_EVENT] = np.eye(4, dtype=c.dtype)[EVENT_HALT]
    return c


class System2ContextProvider:
    """Opt-in hook for the legged latent controller: `provider(ctx) -> ctx` applies a target's ctx-level edits.

    Default-off: `System2ContextProvider(None)` is the identity. Accepts a numpy vector or a torch tensor [1, 22]
    (returns the same type / device / dtype). `record()` is what a row should store under "system2"."""

    def __init__(self, target: System2Target | None, binding_in_ctx: bool = False):
        self.target = target
        self.binding_in_ctx = binding_in_ctx

    @property
    def active(self) -> bool:
        return self.target is not None

    def __call__(self, ctx, session=None):
        if self.target is None:
            return ctx
        if hasattr(ctx, "detach"):                      # torch tensor
            import torch
            arr = ctx.detach().cpu().numpy()
            new = apply_target(arr, self.target, binding_in_ctx=self.binding_in_ctx)
            return ctx if new is arr else torch.from_numpy(new).to(device=ctx.device, dtype=ctx.dtype)
        return apply_target(ctx, self.target, binding_in_ctx=self.binding_in_ctx)

    def record(self) -> dict | None:
        return None if self.target is None else dict(self.target.to_row(), binding_in_ctx=self.binding_in_ctx)


# ---------------------------------------------------------------- systems
@runtime_checkable
class System2(Protocol):
    source: str
    needs_image: bool
    needs_truth: bool

    def ground(self, instruction: str, image: np.ndarray | None = None, *,
               privileged_truth: tuple[str, str] | None = None) -> System2Target: ...


class OracleSystem2:
    """DIAGNOSTIC: target order = the instruction TRUTH (privileged). Upper reference, never a deployable system II."""
    source = "oracle_diagnostic:instruction_truth"
    needs_image = False
    needs_truth = True

    def ground(self, instruction, image=None, *, privileged_truth=None):
        if privileged_truth is None:
            raise ValueError("OracleSystem2 needs privileged_truth (it is a diagnostic upper reference)")
        return System2Target(source=self.source, order=tuple(privileged_truth), diagnostic=True,
                             instruction=instruction, info=dict(privileged=True))


class DefaultSystem2:
    """The declared task view: fixed binding, instruction ignored (what system i had before system II)."""
    needs_image = False
    needs_truth = False

    def __init__(self, order=DECLARED_ORDER):
        self.order = tuple(order)
        self.source = "default:declared_task_binding" + ("" if self.order == DECLARED_ORDER else f"({','.join(order)})")

    def ground(self, instruction, image=None, *, privileged_truth=None):
        return System2Target(source=self.source, order=self.order, instruction=instruction)


class MockSystem2:
    """Tests / plumbing only (label mock:<name>). `fn(instruction, image) -> dict(order=..., halt=..., ...)` or a
    fixed `order`."""
    needs_truth = False

    def __init__(self, name="mock", order=None, fn: Callable[[str, Any], dict] | None = None, needs_image=False,
                 **fixed):
        self.source = f"mock:{name}"
        self.order, self.fn, self.fixed, self.needs_image = order, fn, fixed, needs_image
        self.calls: list[tuple[str, bool]] = []

    def ground(self, instruction, image=None, *, privileged_truth=None):
        self.calls.append((instruction, image is not None))
        kw = dict(self.fixed)
        if self.order is not None:
            kw["order"] = tuple(self.order)
        if self.fn is not None:
            kw.update(self.fn(instruction, image))
        return System2Target(source=self.source, instruction=instruction, **kw)


class VLMSystem2:
    """Frozen psi0 System-II VLM adapter (`rrp.models.system2_vlm.System2`, built lazily; or inject `model` with
    `.run(images, texts) -> dict(score, feat, gen, top1)`). Readouts: `zero_shot` (score > 0 -> orange first) or
    `probe` (`probe(feat [B, F]) -> bool [B]`, a trained head on frozen features, labelled +probe:<name>)."""
    needs_image = True
    needs_truth = False

    def __init__(self, ckpt: str | None = None, model=None, device=None, readout="zero_shot", probe=None,
                 probe_label=None):
        if readout not in ("zero_shot", "probe"):
            raise ValueError(readout)
        if readout == "probe" and probe is None:
            raise ValueError("readout=probe needs a probe callable")
        self.ckpt, self._model, self.device = ckpt, model, device
        self.readout, self.probe = readout, probe
        name = ckpt or ("injected" if model is not None else "psi0_system2")
        self.source = f"learned:{name}#{readout}" + (f"+probe:{probe_label}" if probe_label else "")

    @property
    def model(self):
        if self._model is None:
            from rrp.policies.nets.system2_vlm import System2 as _VLM
            self._model = _VLM(device=self.device, local_dir=self.ckpt)
        return self._model

    def provenance(self):
        p = getattr(self.model, "provenance", None)
        return p() if callable(p) else None

    def ground_batch(self, instructions, images):
        if any(im is None for im in images):
            raise ValueError("VLMSystem2 reads the declared public camera image; got None")
        o = self.model.run(list(images), list(instructions))
        if self.readout == "zero_shot":
            first = [float(s) > 0 for s in o["score"]]
        else:
            first = [bool(v) for v in np.asarray(self.probe(np.asarray(o["feat"])))]
        out = []
        for j, text in enumerate(instructions):
            order = ("orange", "cyan") if first[j] else ("cyan", "orange")
            info = {k: (o[k][j] if k in o else None) for k in ("gen", "top1")}
            out.append(System2Target(source=self.source, order=order, instruction=text,
                                     confidence=float(o["score"][j]), info=info))
        return out

    def ground(self, instruction, image=None, *, privileged_truth=None):
        return self.ground_batch([instruction], [image])[0]


def make_system2(spec: str | None):
    """CLI spec -> system (or None). off | oracle | default | mock[:name] | vlm | vlm:<local weights dir>."""
    if spec is None or spec in ("", "off", "none"):
        return None
    kind, _, arg = spec.partition(":")
    if kind == "oracle":
        return OracleSystem2()
    if kind == "default":
        return DefaultSystem2()
    if kind == "mock":
        return MockSystem2(arg or "mock", order=DECLARED_ORDER)
    if kind == "vlm":
        return VLMSystem2(ckpt=arg or None)
    raise ValueError(f"unknown --system2 spec {spec!r} (off|oracle|default|mock[:name]|vlm[:<ckpt>])")


def _ground(system, instruction, image, truth):
    return system.ground(instruction, image, privileged_truth=truth if getattr(system, "needs_truth", False) else None)


# ---------------------------------------------------------------- episode wiring (closed loop)
def scenario_for(body, seed, truth_order, target: System2Target | None):
    """Scenario whose PUBLIC task graph carries the target's binding (declared binding when target is None / has no
    order) and whose PRIVILEGED evaluator mapping is the instruction truth."""
    order = target.binding() if target is not None else DECLARED_ORDER
    return scenario_with_truth(body, seed, tuple(truth_order), order)


def system2_episode(system, body, seed, family="color", template=None, image=None):
    """One closed-loop episode setup: instruction (seeded) -> system II -> (scenario, provider, record).
    `system=None` (off) -> (None, identity provider, None): the caller keeps its default scenario unchanged."""
    if system is None:
        return None, System2ContextProvider(None), None
    from rrp.envs.mujoco.legged import build_waypoint_contact
    sc0 = build_waypoint_contact(body, seed)
    text, truth, ti = make_instruction(seed, family, sc0.meta["waypoints"], template)
    if image is None and getattr(system, "needs_image", False):
        image = render_public(sc0)
    tgt = _ground(system, text, image, truth)
    rec = dict(harness_version=SYSTEM2_HARNESS_VERSION, instruction=text, family=family, template=ti,
               truth=list(truth), binding=list(tgt.binding()), binding_correct=list(tgt.binding()) == list(truth),
               target=tgt.to_row())
    return scenario_for(body, seed, truth, tgt), System2ContextProvider(tgt), rec


def _attach_provider(ctl, provider):
    """Uses the controller's `context_fn` hook when it has one; otherwise wraps its `_ctx` (instance-local)."""
    if not provider.active or not provider.target.context_edits():
        return
    if hasattr(ctl, "context_fn"):
        ctl.context_fn = provider
        return
    inner = ctl._ctx
    ctl._ctx = lambda ad, mode=None: provider(inner(ad, mode))


# ---------------------------------------------------------------- evaluation
def grounding_items(body="go2", splits=None, families=None, render=True):
    """Scene seeds x instruction families -> items (split, seed, family, template, held_out_template, text, truth,
    img). Rendering needs MuJoCo; `render=False` for text-only systems."""
    from rrp.envs.mujoco.legged import build_waypoint_contact
    splits = splits or {"train": range(0, 160), "test": range(5000, 5080)}
    families = families or list(TEMPLATES)
    items = []
    for split, seeds in splits.items():
        for sd in seeds:
            sc = build_waypoint_contact(body, sd)
            img = render_public(sc) if render else None
            for fam in families:
                text, truth, ti = make_instruction(sd, fam, sc.meta["waypoints"])
                items.append(dict(split=split, seed=sd, family=fam, template=ti,
                                  held_out_template=ti in HELD_OUT_TEMPLATES[fam], text=text, truth=list(truth), img=img))
    return items


def _ground_many(system, texts, images, truths):
    if hasattr(system, "ground_batch") and not getattr(system, "needs_truth", False):
        return system.ground_batch(texts, images)
    return [_ground(system, t, im, tr) for t, im, tr in zip(texts, images, truths)]


def ground_eval(system, items, controls=("blank", "shuffled"), batch=8, out: Path | None = None, log=print):
    """Grounding accuracy of the target ORDER vs the instruction truth (same split / template semantics as
    `rrp.research.system2_eval.ground`). Controls (image systems only): blank image, shuffled image (another scene).
    Returns dict(rows, summary). Writes ground_rows.jsonl / ground_summary.json when `out` is given."""
    uses_img = getattr(system, "needs_image", False) and items and items[0].get("img") is not None
    controls = tuple(controls) if uses_img else ()
    blank = np.full_like(items[0]["img"], 128) if uses_img else None
    rows, t0 = [], time.time()
    for k in range(0, len(items), batch):
        chunk = items[k:k + batch]
        texts = [c["text"] for c in chunk]
        truths = [tuple(c["truth"]) for c in chunk]
        imgs = {"real": [c.get("img") for c in chunk]}
        if "blank" in controls:
            imgs["blank"] = [blank for _ in chunk]
        if "shuffled" in controls:
            imgs["shuffled"] = [items[(k + j + 37) % len(items)]["img"] for j in range(len(chunk))]
        preds = {name: _ground_many(system, texts, ims, truths) for name, ims in imgs.items()}
        for j, c in enumerate(chunk):
            r = {kk: v for kk, v in c.items() if kk != "img"}
            tg = preds["real"][j]
            r.update(harness_version=SYSTEM2_HARNESS_VERSION, system2_source=tg.source, diagnostic=tg.diagnostic,
                     predicted=list(tg.binding()), correct=list(tg.binding()) == list(c["truth"]),
                     confidence=tg.confidence, info=tg.info)
            for name in controls:
                r[f"correct_{name}"] = list(preds[name][j].binding()) == list(c["truth"])
            rows.append(r)
        if log:
            log(f"{k + len(chunk)}/{len(items)} {time.time() - t0:.0f}s")
    summ = summarize_grounding(rows, controls)
    summ.update(harness_version=SYSTEM2_HARNESS_VERSION, system2_source=getattr(system, "source", None),
                n=len(rows), wall_s=time.time() - t0)
    if out is not None:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        with open(out / "ground_rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        (out / "ground_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return dict(rows=rows, summary=summ)


def summarize_grounding(rows, controls=()):
    acc = lambda key, rs: (float(np.mean([r[key] for r in rs])) if rs else None)
    res = {}
    for fam in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == fam]
        te = [r for r in rs if r.get("split") == "test"] or rs
        res[fam] = dict(n=len(rs), n_test=len(te),
                        base_rate_orange_first=float(np.mean([r["truth"][0] == "orange" for r in te])),
                        acc_all=acc("correct", rs), acc_test=acc("correct", te),
                        acc_test_heldout_templates=acc("correct", [r for r in te if r.get("held_out_template")]),
                        acc_test_train_templates=acc("correct", [r for r in te if not r.get("held_out_template")]),
                        **{f"acc_test_{c}_image": acc(f"correct_{c}", te) for c in controls})
    return dict(per_family=res)


def closed_loop_eval(flow: Path, systems: dict, out: Path, seeds=range(10000, 10020), body="go2", family="color",
                     max_s=60.0, dev=None, log=print):
    """Legged latent route (system i flow + system 0) with the public task binding produced by each system II.
    Conditions differ ONLY in the public binding (+ any ctx-level target edits); the privileged evaluator judges
    against the instruction truth. `systems`: condition name -> System2 (e.g. oracle / default / system2_zero_shot).
    Same semantics as `rrp.research.system2_eval.closed_loop` with {oracle, default, system2_*} conditions."""
    import torch
    from rrp.harness.eval.legged_latent_eval import LatentLeggedController, run_episode
    from rrp.envs.mujoco.legged import build_waypoint_contact
    dev = dev or torch.device("cpu")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "closed_loop.jsonl", "w") as f:
        for sd in seeds:
            sc0 = build_waypoint_contact(body, sd)
            text, truth, ti = make_instruction(sd, family, sc0.meta["waypoints"])
            img = render_public(sc0) if any(getattr(s, "needs_image", False) for s in systems.values()) else None
            for cond, system in systems.items():
                tgt = _ground(system, text, img, truth)
                sc = scenario_for(body, sd, truth, tgt)
                ctl = LatentLeggedController(flow, dev, seed=sd)
                _attach_provider(ctl, System2ContextProvider(tgt))
                row, _ = run_episode(ctl, body, sd, max_s, scenario=sc)
                r = dict(harness_version=SYSTEM2_HARNESS_VERSION, seed=sd, body=body, condition=cond, family=family,
                         template=ti, instruction=text, truth=list(truth), binding=list(tgt.binding()),
                         binding_correct=list(tgt.binding()) == list(truth), success=row["success"], fell=row["fell"],
                         sim_time=row["sim_time"], source=row["source"], system2_source=tgt.source,
                         diagnostic=tgt.diagnostic, target=tgt.to_row(), events=row["events"],
                         failure_stage=row.get("failure_stage"))
                rows.append(r)
                f.write(json.dumps(r, default=str) + "\n"); f.flush()
                if log:
                    log(json.dumps({k: r[k] for k in ("seed", "condition", "binding_correct", "success", "fell")}))
    summ = summarize_closed_loop(rows)
    (out / "closed_loop_summary.json").write_text(json.dumps(dict(
        harness_version=SYSTEM2_HARNESS_VERSION, flow=str(flow), body=body, family=family,
        system2_sources={c: getattr(s, "source", None) for c, s in systems.items()}, per_condition=summ), indent=1))
    return dict(rows=rows, summary=summ)


def summarize_closed_loop(rows):
    summ = {}
    for cond in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in rows if r["condition"] == cond]
        summ[cond] = dict(n=len(rs), system2_source=rs[0]["system2_source"], diagnostic=rs[0]["diagnostic"],
                          binding_correct=sum(r["binding_correct"] for r in rs),
                          success=sum(r["success"] for r in rs), fell=sum(r["fell"] for r in rs))
    return summ


def _range(s):
    lo, _, hi = s.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["ground", "closed_loop"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", default="vlm", help="ground: off|oracle|default|mock|vlm[:<ckpt>]")
    ap.add_argument("--systems", default="oracle,default,vlm",
                    help="closed_loop: comma list of specs; condition name = spec (oracle is DIAGNOSTIC)")
    ap.add_argument("--flow", default=None)
    ap.add_argument("--seeds", default="10000-10019")
    ap.add_argument("--small", action="store_true", help="ground smoke: 8 train + 8 test seeds")
    ap.add_argument("--body", default="go2")
    ap.add_argument("--family", default="color")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.stage == "ground":
        system = make_system2(a.system)
        if system is None:
            raise SystemExit("--system off: nothing to ground")
        splits = ({"train": range(0, 8), "test": range(5000, 5008)} if a.small else None)
        items = grounding_items(a.body, splits, render=getattr(system, "needs_image", False))
        res = ground_eval(system, items, out=out)
        print(json.dumps(res["summary"], indent=1, default=str))
    else:
        if not a.flow:
            raise SystemExit("closed_loop needs --flow")
        systems = {s: make_system2(s) for s in a.systems.split(",") if make_system2(s) is not None}
        res = closed_loop_eval(Path(a.flow), systems, out, _range(a.seeds), body=a.body, family=a.family)
        print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
