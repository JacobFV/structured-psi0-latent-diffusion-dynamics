"""Causal packet edits (`rrp train pointer edit`): probe-guided retargeting of a received packet, realized by system 0
in closed loop. The probe is a diagnostic; the causal claim is the rollout (does the pointer end at the new target)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from rrp.harness.hooks import budget_task
from rrp.harness.rollout import rollout
from rrp.harness.train.pointer.data import Demos
from rrp.harness.train.pointer.split import load_split, make_split_env
from rrp.harness.train.pointer.train import fit_probe, frozen_mu, setup
from rrp.policies.pointer import NW


class _EditedPacketPolicy:
    """System 0 realizing ONE received (edited) packet, no replanning, as a `Policy` on `harness.rollout` (`max_steps` =
    the packet's ticks). It is also the episode's hook: `on_end` reads the pointer before rollout closes the env."""

    def __init__(self, s0, env):
        from rrp.policies.base import PolicyInfo, Requirements
        self.s0, self.env, self.end_px = s0, env, None
        self.info = PolicyInfo("pointer_edit", "learned", "edited_packet",
                               Requirements(frozenset({"cartesian_position", "button", "discrete"}),
                                            observations=frozenset()))

    def reset(self, spec, task, seeds, *, envs=None):
        pass

    def act(self, obs):
        from rrp.policies.base import Act
        return {i: Act(self.s0.tick(self.env)) for i in obs}

    def on_end(self, i, env, ep):
        self.end_px = (env.pointer.u, env.pointer.v)
        return {}


def cmd_edit(a):
    """Probe-guided retargeting (causal use of the packet). At the first packet of an episode (pointer at its start,
    target = the teacher's first widget), z is edited by gradient steps on a post-hoc probe's target-slot logit toward
    another visible widget, with an L2 anchor to the original z. System 0 then realizes ONLY the edited packet (no
    replanning) for 7 ticks; we measure whether the pointer ends inside the new target's box vs the original's. The
    control applies a random edit of the same norm. The probe is trained on frozen E means (cmd_probe recipe)."""
    from rrp.envs.computerworld import scene_widgets
    from rrp.policies.pointer import (LearnedSystem0, PointerSystemI, load_pointer_bundle, pointer_packet,
                                      run_pointer_probe)
    dev, data = setup(a)
    rb, fb = load_pointer_bundle(a.representation, dev), load_pointer_bundle(a.flow, dev)
    Z = frozen_mu(rb["modules"]["E"], data, dev)
    P = fit_probe(Z, data, dev, dz=Z.shape[-1], steps=a.probe_steps, batch=512, lr=1e-3, seed=a.seed)
    for p in P.parameters():
        p.requires_grad_(False)
    lsv, rcv = rb["versions"]["latent_space_version"], rb["versions"]["realizer_compat_version"]
    R = rb["modules"]["R"]
    si = PointerSystemI(fb["modules"]["S"], lsv=lsv, rcv=rcv, device=dev, seed=a.seed)
    rows = []
    rng = random.Random(a.seed)
    split = load_split(a.split)
    for task in ("cw/calc_sum", "cw/fill_form"):
        for seed in split["seeds"][task]["dev"][:a.episodes]:
            for mode in ("probe", "random", "none"):
                env = make_split_env(split, task, seed)
                si.reset([env])
                obs = env.observe()
                p0 = si.packets([env])[0]
                z0 = torch.from_numpy(np.asarray(p0.z, np.float32))[None].to(dev)
                with torch.no_grad():
                    orig = int(run_pointer_probe(P, z0)["slot"][0, -1].argmax())
                ws = [w for w in scene_widgets(env.scene()) if w["visible"] and w["box"] and w["role"] in
                      ("button", "textbox")]
                slots = {env.slots.slots[w["key"]]: w for w in ws if env.slots.slots.get(w["key"], NW) < NW}
                cand = [s for s in slots if s != orig and s in slots]
                if orig not in slots or not cand:
                    env.close()
                    continue
                new = random.Random(seed * 7 + 1).choice(cand)
                z = z0.clone()
                if mode != "none":
                    zv = z0.clone().requires_grad_(True)
                    opt_edit = torch.optim.Adam([zv], lr=a.edit_lr)
                    for _ in range(a.edit_steps):
                        s_ = run_pointer_probe(P, zv)["slot"][0]                       # [K, NW]
                        L = torch.nn.functional.cross_entropy(s_, torch.full((s_.shape[0],), new, device=dev)) \
                            + a.anchor * ((zv - z0) ** 2).mean()
                        opt_edit.zero_grad()
                        L.backward()
                        opt_edit.step()
                    delta = (zv.detach() - z0)
                    if mode == "random":
                        r = torch.randn_like(delta)
                        delta = r / r.norm() * delta.norm()
                    z = z0 + delta
                with torch.no_grad():
                    pr = run_pointer_probe(P, z)["slot"][0, -1].argmax().item()
                pk = pointer_packet(env, obs, z[0].cpu().numpy(), lsv=lsv, rcv=rcv, source="learned",
                                    name=f"edit:{mode}")
                s0 = LearnedSystem0(R, env, latent_space_version=lsv, realizer_compat_version=rcv, device=dev)
                s0.receive(pk, now=float(env.time), graph_version=0)
                ctr = lambda w: np.array([(w["box"][0] + w["box"][2]) / 2, (w["box"][1] + w["box"][3]) / 2])
                p0 = np.array([env.pointer.u, env.pointer.v], float)
                pol = _EditedPacketPolicy(s0, env)
                ep, = rollout(lambda sd: env, pol, budget_task("pointer_edit", env.spec.env_id), [seed], batch=1,
                              max_steps=Demos.H, hooks=[pol])
                if ep.outcome == "crash":
                    raise RuntimeError(f"pointer edit {task} seed {seed}: {ep.failure_reason}: {ep.metrics.get('note', '')}")
                u, v = pol.end_px
                p1 = np.array([u, v], float)
                inside = lambda w: w["box"][0] <= u < w["box"][2] and w["box"][1] <= v < w["box"][3]
                rows.append(dict(task=task, seed=seed, mode=mode, orig_slot=orig, new_slot=new, probe_after=pr,
                                 ptr_px=[u, v], at_new=inside(slots[new]), at_orig=inside(slots[orig]),
                                 d_new=[float(np.linalg.norm(p0 - ctr(slots[new]))), float(np.linalg.norm(p1 - ctr(slots[new])))],
                                 d_orig=[float(np.linalg.norm(p0 - ctr(slots[orig]))),
                                         float(np.linalg.norm(p1 - ctr(slots[orig])))],
                                 z_delta=float((z - z0).norm())))
    summ = {}
    for mode in ("probe", "random", "none"):
        r = [x for x in rows if x["mode"] == mode]
        summ[mode] = dict(n=len(r), at_new=sum(x["at_new"] for x in r), at_orig=sum(x["at_orig"] for x in r),
                          probe_reads_new=sum(x["probe_after"] == x["new_slot"] for x in r),
                          closer_to_new_than_orig=sum(x["d_new"][1] < x["d_orig"][1] for x in r),
                          mean_px_toward_new=float(np.mean([x["d_new"][0] - x["d_new"][1] for x in r])) if r else None,
                          mean_px_toward_orig=float(np.mean([x["d_orig"][0] - x["d_orig"][1] for x in r])) if r else None)
    out = dict(representation=a.representation, flow=a.flow, variant=rb["config"]["variant"], summary=summ, rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summ))
