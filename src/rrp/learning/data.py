"""Chunk dataset over teacher episodes: (PolicyInput_t, a[t:t+H] normalized, masks, labels).

Actions are normalized with PUBLIC spec quantities only (see features.ActionSpace): deltas
from the measured joint position at chunk start for arm joints, [-1,1] closure for grippers.
Privileged labels come from *.private files and are used only as training targets.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rrp.data.collect import read_episode
from rrp.model.batch import collate_inputs


@dataclass
class Sample:
    pi: object
    a: np.ndarray        # [H, N]
    valid: np.ndarray    # [H, N]
    labels: dict
    effect: np.ndarray   # [H, 4]
    meta: dict


def load_episodes(ds_dir: Path, robots: set[str] | None = None, statuses=("success",), limit_per_robot=None,
                  seeds: tuple | None = None, episode_ids: set | None = None) -> list[tuple[dict, dict]]:
    man = json.loads((ds_dir / "manifest.json").read_text())
    out, per = [], {}
    for m in man["episodes"]:
        if m.get("status") not in statuses:
            continue
        rk = m.get("robot_key") or m.get("split_lineage", {}).get("robot_key")
        if robots is not None and rk not in robots:
            continue
        if seeds is not None and not (seeds[0] <= m["seed"] < seeds[1]):
            continue
        if episode_ids is not None and m["episode_id"] not in episode_ids:
            continue
        if limit_per_robot and per.get(rk, 0) >= limit_per_robot:
            continue
        per[rk] = per.get(rk, 0) + 1
        eid = m["episode_id"]
        pub = read_episode(ds_dir / "episodes" / f"{eid}.public.pkl.gz")
        prv = read_episode(ds_dir / "episodes" / f"{eid}.private.pkl.gz")
        out.append((pub, prv))
    return out


def episode_samples(pub: dict, prv: dict, H: int, stride: int = 1) -> list[Sample]:
    from rrp.data.features import ActionSpace
    asd = pub["action_space"]
    aspace = ActionSpace(**asd)
    T = len(pub["inputs"])
    out = []
    manips = prv["manipulators"]
    for t in range(0, T, stride):
        pi = pub["inputs"][t]
        q0 = pub["q0"][t]
        seq = pub["actions"][t:t + H]
        n_valid = len(seq)
        seq = seq + [seq[-1]] * (H - n_valid)
        a = aspace.normalize(seq, q0)
        valid = np.zeros_like(a, bool)
        valid[:n_valid] = True
        lab = prv["labels"][t]
        lab_end = prv["labels"][min(t + H - 1, T - 1)]
        S = lab["slot_pos"].shape[0]
        # focus label from PUBLIC runtime status: slots bound as patient/destination of active events
        focus = np.zeros(S, bool)
        status = pub["statuses"][t]
        ti = pi.meta
        tcp_now = prv["labels"][t]["tcp_pos"][0]
        eff = np.zeros((H, 4), np.float32)
        for h in range(H):
            lt = prv["labels"][min(t + h, T - 1)]
            eff[h, :3] = (lt["tcp_pos"][0] - tcp_now) * 10
        grip_cols = [n for n, g in enumerate(aspace.is_gripper) if g]
        if grip_cols:
            eff[:, 3] = a[:, grip_cols[0]]
        labels = dict(held=lab["slot_held"][:, 0], contact=lab["slot_contact"][:, 0], visible=lab["slot_visible"],
                      rel_tcp=lab["slot_rel_tcp"][:, 0], future_disp=(lab_end["slot_pos"] - lab["slot_pos"]),
                      gaze=lab["slot_gaze_angle"], focus=focus, slot_valid=np.ones(S, bool),
                      status=status)
        out.append(Sample(pi, a.astype(np.float32), valid, labels, eff, dict(episode=pub["meta"]["episode_id"], t=t,
                                                                              robot=pub["meta"].get("robot_key"))))
    return out


def focus_from_inputs(pi, S):
    """Focus = slots pointed to by patient/destination role tokens of ACTIVE events (public)."""
    return np.zeros(S, bool)


class ChunkDataset:
    def __init__(self, episodes: list[tuple[dict, dict]], H: int, stride: int = 1):
        self.samples: list[Sample] = []
        for pub, prv in episodes:
            self.samples.extend(episode_samples(pub, prv, H, stride))
        self.H = H

    def __len__(self):
        return len(self.samples)

    def batches(self, batch_size: int, rng: random.Random, shuffle=True, drop_last=True):
        idx = list(range(len(self.samples)))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx) - (batch_size if drop_last else 0) + (0 if drop_last else batch_size - 1),
                       batch_size):
            chunk = [self.samples[j] for j in idx[i:i + batch_size]]
            if len(chunk) < (batch_size if drop_last else 1):
                break
            yield collate_samples(chunk)


def collate_samples(chunk: list[Sample]):
    batch = collate_inputs([s.pi for s in chunk])
    B = len(chunk)
    N = batch.node_feats.shape[1]
    H = chunk[0].a.shape[0]
    S = batch.bank_tokens["scene"].shape[1]
    a = np.zeros((B, H, N), np.float32)
    v = np.zeros((B, H, N), bool)
    eff = np.stack([s.effect for s in chunk]).astype(np.float32)
    lab = {k: np.zeros((B, S) + shp, dt) for k, shp, dt in
           (("held", (), bool), ("contact", (), bool), ("visible", (), bool), ("focus", (), bool),
            ("slot_valid", (), bool), ("rel_tcp", (3,), np.float32), ("future_disp", (3,), np.float32),
            ("gaze", (), np.float32))}
    for i, s in enumerate(chunk):
        n = s.a.shape[1]
        a[i, :, :n] = s.a
        v[i, :, :n] = s.valid
        k = len(s.labels["held"])
        for key in lab:
            lab[key][i, :k] = s.labels[key]
        # focus from public task tokens: role tokens (kind 1) of active events pointing to slots
        pi = s.pi
        so = 0
        act_events = set()
        tt, tk = pi.tokens["task"], pi.token_kind["task"]
        from rrp.data.features import HASH_DIM
        ev_idx = [j for j in range(len(tk)) if tk[j] == 0]
        for j in ev_idx:
            if tt[j, HASH_DIM + 2] > 0.5:       # status one-hot index 2 == active
                act_events.add(j)
        for (qb, qi, kb, ki, r) in pi.relations:
            if qb == 2 and kb == 1 and r in (4, 5, 6) and False:
                pass
        for (qb, qi, kb, ki, r) in pi.relations:
            # event (task bank) -> slot (scene bank) with patient/target/destination relation
            if qb == 2 and qi in act_events and kb == 1 and r in (4, 5, 6):
                lab["focus"][i, ki] = True
    tens = {k: torch.from_numpy(v_) for k, v_ in lab.items()}
    return batch, torch.from_numpy(a)[..., None], torch.from_numpy(v), tens, torch.from_numpy(eff)
