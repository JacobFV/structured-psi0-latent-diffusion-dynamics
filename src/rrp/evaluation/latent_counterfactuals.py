"""Section-6 tests on the trained representation:
 (a) counterexample: IDENTICAL demonstrated joint trajectory + morphology, DIFFERENT legitimate task/object context
     -> the contextual encoder must produce different z, and packet probes must answer the context-dependent
     queries differently (an action-only codec cannot, by construction);
 (b) compatible embodiment swap: same scene seed and goal, different gripper module (different native trajectories)
     -> packet-probe goal semantics (focus handle, subtask, desired delta of the focused object) should agree."""
from __future__ import annotations

import copy
import numpy as np
import torch

from rrp.learning.data import load_episodes, episode_samples
from rrp.model.batch import collate_inputs
from rrp.model.semantic_latent import assembly_tokens


def _encode(E, pis, a, dev):
    b = collate_inputs(pis).to(dev)
    N = b.node_feats.shape[1]
    aa = torch.zeros(len(pis), a.shape[1], N, device=dev)
    aa[:, :, :a.shape[2]] = torch.as_tensor(a, device=dev)
    v = torch.zeros_like(aa, dtype=torch.bool)
    v[:, :, :a.shape[2]] = True
    af, am, ai = assembly_tokens(b)
    mu, _ = E(b, aa, v, af, am, ai)
    return mu, am, b


@torch.no_grad()
def counterexample(E, P, ds_dir, robot="panda_pg2", n=20, dev="cpu", seed=0, per_episode=1) -> dict:
    """Rebind the task PATIENT to an unbound object slot (a legitimate supplied-context change) while keeping the
    demonstrated actions, body state and all scene tokens identical.

    v2 of this test (track `binding`, 2026-09-25). The v1 edit (`counterexample_v1`) had two flaws: (1) it retargeted
    only relations whose KEY was slot 0, leaving slot 0's own patient_of / pred_arg edges (query side) in place, so
    the edited graph bound the patient to BOTH slots; (2) its only readout was the focus argmax of a probe that had
    never seen a focused slot >= 2 (in the training data the patient is always canonical slot 0), so it could not
    flip even for a binding-aware z. Here the edit is the symmetric slot-edge swap of model/binding_aug.rebind, the
    expected label is recomputed from the edited relations with the public focus rule, and the headline metric is
    `focus_follows`: probe focus is ON for the newly bound slot and OFF for the old one."""
    from rrp.model.binding_aug import focus_from_batch, rebind, slot_has_edges
    rng = np.random.default_rng(seed)
    eps = load_episodes(ds_dir, robots={robot}, limit_per_robot=n)
    rows = []
    for pub, prv in eps:
        smp = episode_samples(pub, prv, 16, stride=10)
        for i in rng.choice(len(smp), size=min(per_episode, len(smp)), replace=False):
            s = smp[i]
            pi = s.pi
            n_slots = len(pi.tokens["scene"])
            if n_slots < 3:
                continue
            b = collate_inputs([pi]).to(dev)
            o, T = b.bank_offset["task"], b.bank_tokens["task"].shape[1]
            os_ = b.bank_offset["scene"]
            patient = b.ctx_rel[0, o:o + T, os_:os_ + n_slots, 4].any(0)           # event -patient_of-> slot
            bound = slot_has_edges(b)[0, :n_slots]
            if not patient.any() or bound.all():
                continue
            src = int(torch.nonzero(patient)[0])
            dst = int(torch.nonzero(~bound)[0])
            bc = rebind(b, torch.tensor([src], device=dev), torch.tensor([dst], device=dev))
            f0, f1 = focus_from_batch(b)[0], focus_from_batch(bc)[0]
            mu, am, _ = _encode_batch(E, [b, bc], np.stack([s.a, s.a]), dev)
            out = P(mu, am, n_slots)
            foc = torch.sigmoid(out["focused_on"][..., 0])
            pf, pc = foc[0] > 0.5, foc[1] > 0.5
            rows.append(dict(src=src, dst=dst, z_dist=float((mu[0] - mu[1]).norm()), z_norm=float(mu[0].norm()),
                             label_orig=f0.int().tolist(), label_cf=f1.int().tolist(),
                             focus_orig=foc[0].tolist(), focus_cf=foc[1].tolist(),
                             argmax_changed=bool(foc[0].argmax() != foc[1].argmax()),
                             focus_follows=bool(pc[dst] == f1[dst] and pc[src] == f1[src]),
                             exact_orig=bool(torch.equal(pf, f0)), exact_cf=bool(torch.equal(pc, f1))))
    m = lambda k: float(np.mean([r[k] for r in rows])) if rows else None
    chg = [r for r in rows if r["label_orig"] != r["label_cf"]]     # rebinding changes the focus label (active event)
    return dict(test="v2_symmetric_rebind", n=len(rows), n_label_changed=len(chg),
                focus_follows_changed=float(np.mean([r["focus_follows"] for r in chg])) if chg else None,
                mean_z_dist=m("z_dist"),
                mean_rel_z_dist=float(np.mean([r["z_dist"] / max(r["z_norm"], 1e-6) for r in rows])) if rows else None,
                focus_follows=m("focus_follows"), focus_exact_cf=m("exact_cf"), focus_exact_orig=m("exact_orig"),
                focus_argmax_changed=m("argmax_changed"), rows=rows[:5])


def _encode_batch(E, batches, a, dev):
    from rrp.model.binding_aug import cat_batch
    b = batches[0]
    for x in batches[1:]:
        b = cat_batch(b, x)
    N = b.node_feats.shape[1]
    aa = torch.zeros(len(a), a.shape[1], N, device=dev)
    aa[:, :, :a.shape[2]] = torch.as_tensor(a, device=dev)
    v = torch.zeros_like(aa, dtype=torch.bool)
    v[:, :, :a.shape[2]] = True
    af, am, ai = assembly_tokens(b)
    mu, _ = E(b, aa, v, af, am, ai)
    return mu, am, b


@torch.no_grad()
def counterexample_v1(E, P, ds_dir, robot="panda_pg2", n=20, dev="cpu", seed=0) -> dict:
    """ORIGINAL (flawed, kept for reproducing D-032): one-sided edit + argmax readout. See `counterexample`."""
    rng = np.random.default_rng(seed)
    eps = load_episodes(ds_dir, robots={robot}, limit_per_robot=n)
    rows = []
    for pub, prv in eps:
        smp = episode_samples(pub, prv, 16, stride=10)
        s = smp[rng.integers(len(smp))]
        pi = s.pi
        n_slots = len(pi.tokens["scene"])
        if n_slots < 3:
            continue
        pj = copy.deepcopy(pi)
        other = 2
        R = pj.relations.copy()
        R[(R[:, 2] == 1) & (R[:, 3] == 0)] = R[(R[:, 2] == 1) & (R[:, 3] == 0)] * [1, 1, 1, 0, 1] + [0, 0, 0, other, 0]
        pj.relations = R
        P_ = pj.pointers.copy()
        P_[(P_[:, 2] == 1) & (P_[:, 3] == 0), 3] = other
        pj.pointers = P_
        mu, am, b = _encode(E, [pi, pj], np.stack([s.a, s.a]), dev)
        out = P(mu, am, n_slots)
        foc = torch.sigmoid(out["focused_on"][..., 0])
        rows.append(dict(z_dist=float((mu[0] - mu[1]).norm()), z_norm=float(mu[0].norm()),
                         focus_orig=foc[0].tolist(), focus_cf=foc[1].tolist(),
                         argmax_changed=bool(foc[0].argmax() != foc[1].argmax())))
    return dict(test="v1_one_sided", n=len(rows),
                mean_z_dist=float(np.mean([r["z_dist"] for r in rows])) if rows else None,
                mean_rel_z_dist=float(np.mean([r["z_dist"] / max(r["z_norm"], 1e-6) for r in rows])) if rows else None,
                focus_argmax_changed=float(np.mean([r["argmax_changed"] for r in rows])) if rows else None, rows=rows[:5])


@torch.no_grad()
def embodiment_swap(E, P, ds_dir, arm="parm6", seeds=range(0, 20), dev="cpu") -> dict:
    """Same arm, same scene seed, pg2 vs tf3 gripper; compare packet-probe goal semantics at the same teacher phase."""
    agree, n = dict(focus=0, subtask=0), 0
    ddist = []
    for sd in seeds:
        e1 = load_episodes(ds_dir, robots={f"{arm}_pg2"}, seeds=(sd, sd + 1))
        e2 = load_episodes(ds_dir, robots={f"{arm}_tf3"}, seeds=(sd, sd + 1))
        e1 = [e for e in e1 if e[0]["meta"].get("exec_noise", 0) == 0]
        e2 = [e for e in e2 if e[0]["meta"].get("exec_noise", 0) == 0]
        if not e1 or not e2:
            continue
        (p1, q1), (p2, q2) = e1[0], e2[0]
        for ph in ("pregrasp", "lift", "transport"):
            i1 = [i for i, x in enumerate(q1["phases"]) if x == ph]
            i2 = [i for i, x in enumerate(q2["phases"]) if x == ph]
            if not i1 or not i2:
                continue
            s1 = episode_samples(p1, q1, 16, stride=1)[i1[len(i1) // 2]]
            s2 = episode_samples(p2, q2, 16, stride=1)[i2[len(i2) // 2]]
            outs = []
            for s in (s1, s2):
                mu, am, b = _encode(E, [s.pi], s.a[None], dev)
                outs.append(P(mu, am, len(s.pi.tokens["scene"])))
            f1, f2 = [o["focused_on"][0, :, 0].argmax().item() for o in outs]
            t1, t2 = [o["subtask"][0, 0].argmax().item() for o in outs]
            d1, d2 = [o["desired_delta"][0, f1, :3] / 10 for o in outs]
            agree["focus"] += int(f1 == f2)
            agree["subtask"] += int(t1 == t2)
            ddist.append(float((d1 - d2).norm()))
            n += 1
    return dict(pairs=n, focus_agreement=agree["focus"] / n if n else None,
                subtask_agreement=agree["subtask"] / n if n else None,
                desired_delta_disagreement_m=float(np.mean(ddist)) if ddist else None)
