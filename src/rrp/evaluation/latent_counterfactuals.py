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
def counterexample(E, P, ds_dir, robot="panda_pg2", n=20, dev="cpu", seed=0) -> dict:
    """Swap which object slot is BOUND as the task patient (a legitimate supplied-context change) while keeping the
    demonstrated actions and body state identical."""
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
        # counterfactual context: the task entity that was bound to slot 0 is now bound to another object slot
        # (role pointers retargeted), trajectory unchanged
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
    return dict(n=len(rows), mean_z_dist=float(np.mean([r["z_dist"] for r in rows])) if rows else None,
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
