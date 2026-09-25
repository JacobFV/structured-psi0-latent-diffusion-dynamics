"""System II (psi0 Qwen3-VL System-II backbone) as a PUBLIC task-binding provider for system i (legged track).

Role in the corrected architecture: system II reads a rendered image from a declared public scene camera plus the
operator's instruction text, and outputs TASK STRUCTURE: which detected marker is bound to the first `walk_to`
event and which to the second (the task graph's entity descriptors). That binding enters system i only through the
public task view (`LeggedSession._bind_entities` -> detector slots -> body-frame waypoint estimates + active event).
System II never sees simulator state, privileged labels or future outcomes, and its output is not a packet.

Two readouts of the frozen VLM, reported separately:
  zero_shot  : constrained answer scoring — logit of the first token of " orange" vs " cyan" after the question.
  probe      : a logistic readout trained on frozen pooled hidden states (train templates / seeds), evaluated on
               held-out templates and seeds. It is a trained head on system-II features, labelled as such.
Controls: blank image (image content removed), shuffled image (another scene).

Instruction families:
  color    : order stated by color ("walk to the cyan marker, then the orange one") — text alone suffices.
  distance : order stated by spatial relation ("first the marker nearer to you") — needs the image.
"""
from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch

COLORS = ("orange", "cyan")

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
QUESTION = " Question: which marker should the robot walk to first? Answer with one word: orange or cyan."


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
    from rrp.sim.scenario import load_task
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
    from rrp.sim.legged import build_waypoint_contact
    from rrp.sim.scenario import ObjectDecl
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
    from rrp.control.legged_core import LeggedBinding
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


class System2:
    """Frozen psi0 System-II VLM wrapper: answer scoring + pooled features."""

    def __init__(self, device="cuda", local_dir=None):
        from rrp.model.backbone import VLMBackbone, BackboneSpec
        self.bb = VLMBackbone(BackboneSpec(), device=device, local_dir=local_dir)
        tok = self.bb.processor.tokenizer
        self.cand = {c: tok.encode(" " + c, add_special_tokens=False)[0] for c in COLORS}
        self.cand_nospace = {c: tok.encode(c, add_special_tokens=False)[0] for c in COLORS}

    def provenance(self):
        return self.bb.provenance()

    @torch.no_grad()
    def run(self, images, texts):
        """images: list of HxWx3 uint8 (one per sample); returns dict(score [B] = logit(orange) - logit(cyan) at the
        answer position, feat [B, F] pooled hidden states, generated text of 4 greedy tokens)."""
        from PIL import Image
        bb = self.bb
        prompts = [bb.processor.apply_chat_template(bb._messages(1, t + QUESTION), tokenize=False,
                                                    add_generation_prompt=True) for t in texts]
        bb.processor.tokenizer.padding_side = "left"
        inputs = bb.processor(text=prompts, images=[Image.fromarray(im) for im in images], padding=True,
                              return_tensors="pt").to(bb.device)
        out = bb.model(**inputs, output_hidden_states=True, return_dict=True)
        lg = out.logits[:, -1].float()
        sc = torch.maximum(lg[:, self.cand["orange"]], lg[:, self.cand_nospace["orange"]]) - \
            torch.maximum(lg[:, self.cand["cyan"]], lg[:, self.cand_nospace["cyan"]])
        ids = inputs["input_ids"]
        is_img = ids == bb.image_token_id
        feats = []
        for t in (16, 28):
            h = out.hidden_states[t].float()
            txt = h[:, -8:].mean(1)
            vis = (h * is_img[..., None]).sum(1) / is_img.sum(1, keepdim=True).clamp(min=1)
            feats += [txt, vis]
        gen = bb.model.generate(**inputs, max_new_tokens=4, do_sample=False)
        texts_out = bb.processor.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
        return dict(score=sc.cpu().numpy(), feat=torch.cat(feats, -1).cpu().numpy(), gen=texts_out,
                    top1=[bb.processor.tokenizer.decode([int(i)]) for i in lg.argmax(-1).tolist()])
