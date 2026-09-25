"""Packed, memory-mapped chunk datasets.

Unpickling every episode into Python objects cost 11-20 GB per training process (duplicated per
process). `pack_dataset` converts a dataset once into fixed-shape float16/bool/int arrays in .npy
files; `PackedChunkDataset` memory-maps them so concurrent jobs share one page-cache copy and each
process needs only its batch working set. Batches are identical in content to the pickle path
(same featurized tokens, relations, pointers, actions, masks and labels), just padded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from rrp.data.features import BANKS, HASH_DIM, N_REL
from rrp.learning.data import load_episodes, episode_samples
from rrp.model.batch import Batch, BANK_DIMS, NODE_DIM

MAX_T = {"morph": 16, "scene": 8, "task": 24, "interact": 12}
OPERATORS = ["none", "grasp", "place", "reach", "maintain_support", "estimate_frame", "align_axis", "insert",
             "give", "receive", "walk_to", "halt", "maintain_hold", "release"]   # appended only (indices are stable)


def local_sensors(pi) -> np.ndarray:
    """Declared local sensors for system 0: touch summary (log max, count>0.2N, log mean) + gripper width."""
    from rrp.data.features import text_hash
    th = text_hash("touch")
    out = np.zeros(4, np.float32)
    it, ik = pi.tokens["interact"], pi.token_kind["interact"]
    for j in range(len(ik)):
        if ik[j] != 1:
            continue
        if np.allclose(it[j, 16:], th):
            out[:3] = it[j, 13:16]
        else:
            out[3] = it[j, 13]
    return out


def active_operator(pi) -> int:
    """Public subtask label: operator of the first ACTIVE event (runtime status is public)."""
    from rrp.data.features import text_hash
    tt, tk = pi.tokens["task"], pi.token_kind["task"]
    for j in range(len(tk)):
        if tk[j] == 0 and tt[j, HASH_DIM + 2] > 0.5:
            for k, op in enumerate(OPERATORS):
                if np.allclose(tt[j, :HASH_DIM], text_hash(op), atol=1e-5):
                    return k
    return 0
MAX_N, MAX_S, MAX_R, MAX_P = 12, 8, 160, 32


def _focus(pi, S):
    f = np.zeros(S, bool)
    tt, tk = pi.tokens["task"], pi.token_kind["task"]
    act = {j for j in range(len(tk)) if tk[j] == 0 and tt[j, HASH_DIM + 2] > 0.5}
    for (qb, qi, kb, ki, r) in pi.relations:
        if qb == 2 and qi in act and kb == 1 and r in (4, 5, 6) and ki < S:
            f[ki] = True
    return f


def pack_dataset(ds_dir: Path, out_dir: Path, robots: set[str] | None, H: int, stride: int = 1,
                 include_dart_failures: bool = False, statuses=("success",), limit_per_robot=None,
                 seeds=None, chunk_episodes: int = 200, limits: dict | None = None, multi_m: int = 0) -> dict:
    """limits: override MAX_T/N/R (dual-arm inputs are larger). multi_m > 0 additionally stores multi-assembly
    arrays (rrp.learning.dual_latent): per-slot privileged labels held_m/contact_m/rel_tcp_m in packet order, public
    per-slot subtask_m and local sensors local_m, and node_asm (packet slot of each action node)."""
    MAX_T_, MAX_N_, MAX_R_, MAX_P_ = dict(MAX_T), MAX_N, MAX_R, MAX_P
    if limits:
        MAX_T_.update(limits.get("T", {}))
        MAX_N_, MAX_R_, MAX_P_ = limits.get("N", MAX_N_), limits.get("R", MAX_R), limits.get("P", MAX_P)
    trunc = dict(rel=0, ptr=0, **{b: 0 for b in BANKS})
    out_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads((ds_dir / "manifest.json").read_text())
    per, ids = {}, []
    for m in man["episodes"]:          # global selection (limit applies across the whole dataset)
        rk = m.get("robot_key") or m.get("split_lineage", {}).get("robot_key")
        ok = m.get("status") in statuses or (include_dart_failures and m.get("exec_noise", 0) > 0
                                             and m.get("status") == "failure")
        if not ok or (robots is not None and rk not in robots):
            continue
        if seeds is not None and not (seeds[0] <= m["seed"] < seeds[1]):
            continue
        if limit_per_robot and per.get(rk, 0) >= limit_per_robot:
            continue
        per[rk] = per.get(rk, 0) + 1
        ids.append(m["episode_id"])
    # streaming: load episodes in groups to bound memory while packing
    arrays = {}
    n = 0
    specs = dict(
        **{f"tok_{b}": ((MAX_T_[b], BANK_DIMS[b]), np.float16) for b in BANKS},
        **{f"kind_{b}": ((MAX_T_[b],), np.int8) for b in BANKS},
        **{f"text_{b}": ((MAX_T_[b], HASH_DIM), np.float16) for b in BANKS},
        **{f"len_{b}": ((), np.int16) for b in BANKS},
        node=((MAX_N_, NODE_DIM), np.float16), n_nodes=((), np.int16),
        rel=((MAX_R_, 5), np.int16), n_rel=((), np.int16), ptr=((MAX_P_, 4), np.int16), n_ptr=((), np.int16),
        a=((H, MAX_N_), np.float16), valid=((H, MAX_N_), np.bool_), eff=((H, 4), np.float16),
        held=((MAX_S,), np.bool_), contact=((MAX_S,), np.bool_), visible=((MAX_S,), np.bool_),
        focus=((MAX_S,), np.bool_), slot_valid=((MAX_S,), np.bool_), rel_tcp=((MAX_S, 3), np.float16),
        future_disp=((MAX_S, 3), np.float16), gaze=((MAX_S,), np.float16),
        ep_idx=((), np.int32), t=((), np.int32), ep_len=((), np.int32), local=((4,), np.float16),
        subtask=((), np.int8), robot_id=((), np.int16))
    if multi_m:
        M_ = multi_m
        specs.update(held_m=((MAX_S, M_), np.bool_), contact_m=((MAX_S, M_), np.bool_), rel_tcp_m=((MAX_S, M_, 3), np.float16),
                     subtask_m=((M_,), np.int8), local_m=((M_, 4), np.float16), node_asm=((MAX_N_,), np.int8),
                     slot_col=((M_,), np.int8))
    buf = {k: [] for k in specs}
    robots_seen = []
    ep_counter = 0
    robot_ids: dict[str, int] = {}
    for g in range(0, len(ids), chunk_episodes):
        eps = load_episodes(ds_dir, robots=robots, statuses=statuses, seeds=seeds,
                            episode_ids=set(ids[g:g + chunk_episodes]), include_dart_failures=include_dart_failures)
        for pub, prv in eps:
            ep_counter += 1
            ep_len = len(pub["inputs"])
            rk = pub["meta"].get("robot_key") or ""
            rid = robot_ids.setdefault(rk, len(robot_ids))
            if multi_m:
                from rrp.learning import dual_latent as DL
                cols = DL.label_columns_for_slots(pub["meta"], list(prv["manipulators"]), multi_m)
            for smp in episode_samples(pub, prv, H, stride):
                pi = smp.pi
                row = {}
                for b in BANKS:
                    T = min(len(pi.tokens[b]), MAX_T_[b])
                    x = np.zeros((MAX_T_[b], BANK_DIMS[b]), np.float16)
                    x[:T, :pi.tokens[b].shape[1]] = pi.tokens[b][:T]
                    row[f"tok_{b}"] = x
                    k = np.zeros(MAX_T_[b], np.int8)
                    k[:T] = pi.token_kind[b][:T]
                    row[f"kind_{b}"] = k
                    t = np.zeros((MAX_T_[b], HASH_DIM), np.float16)
                    pt = pi.pointer_text[b][:T]
                    t[:len(pt)] = pt
                    row[f"text_{b}"] = t
                    row[f"len_{b}"] = np.int16(T)
                N = min(pi.act_node_feats.shape[0], MAX_N_)
                nd = np.zeros((MAX_N_, NODE_DIM), np.float16)
                nd[:N] = pi.act_node_feats[:N]
                row["node"], row["n_nodes"] = nd, np.int16(N)
                R = np.asarray(pi.relations)[:MAX_R_]
                rr = np.zeros((MAX_R_, 5), np.int16)
                rr[:len(R)] = R
                row["rel"], row["n_rel"] = rr, np.int16(len(R))
                P = np.asarray(pi.pointers)[:MAX_P_]
                pp = np.zeros((MAX_P_, 4), np.int16)
                trunc["ptr"] += len(pi.pointers) > MAX_P_
                trunc["rel"] += len(pi.relations) > MAX_R_
                for b in BANKS:
                    trunc[b] += len(pi.tokens[b]) > MAX_T_[b]
                pp[:len(P)] = P
                row["ptr"], row["n_ptr"] = pp, np.int16(len(P))
                a = np.zeros((H, MAX_N_), np.float16)
                v = np.zeros((H, MAX_N_), bool)
                a[:, :N] = smp.a[:, :N]
                v[:, :N] = smp.valid[:, :N]
                row["a"], row["valid"], row["eff"] = a, v, smp.effect.astype(np.float16)
                S = min(len(smp.labels["held"]), MAX_S)
                for key in ("held", "contact", "visible", "slot_valid"):
                    z = np.zeros(MAX_S, bool)
                    z[:S] = smp.labels[key][:S]
                    row[key] = z
                fz = np.zeros(MAX_S, bool)
                fz[:S] = _focus(pi, S)
                row["focus"] = fz
                for key, w in (("rel_tcp", 3), ("future_disp", 3)):
                    z = np.zeros((MAX_S, w), np.float16)
                    z[:S] = smp.labels[key][:S]
                    row[key] = z
                gz = np.zeros(MAX_S, np.float16)
                gz[:S] = smp.labels["gaze"][:S]
                row["gaze"] = gz
                row["ep_idx"], row["t"], row["ep_len"] = np.int32(ep_counter), np.int32(smp.meta["t"]), np.int32(ep_len)
                row["local"] = local_sensors(pi).astype(np.float16)
                row["subtask"] = np.int8(active_operator(pi))
                row["robot_id"] = np.int16(rid)
                if multi_m:
                    row.update(DL.multi_labels(prv["labels"][smp.meta["t"]], cols, S, MAX_S, multi_m))
                    row["subtask_m"] = DL.assembly_operators(pi, OPERATORS, multi_m).astype(np.int8)
                    row["local_m"] = DL.local_sensors_multi(pi, multi_m).astype(np.float16)
                    na = np.zeros(MAX_N_, np.int8)
                    na[:N] = DL.node_assembly_index(pi, multi_m)[:N]
                    row["node_asm"] = na
                    row["slot_col"] = np.array(cols, np.int8)
                for k2 in specs:
                    buf[k2].append(row[k2])
                robots_seen.append(smp.meta.get("robot"))
                n += 1
        del eps
    for k2, (shape, dt) in specs.items():
        arr = np.stack(buf[k2]).astype(dt) if buf[k2] else np.zeros((0,) + shape, dt)
        np.save(out_dir / f"{k2}.npy", arr)
        buf[k2] = None
    meta = dict(n=n, H=H, stride=stride, source=str(ds_dir), robot_ids=robot_ids, operators=OPERATORS, robots=sorted(set(r for r in robots_seen if r)),
                max=dict(T=MAX_T_, N=MAX_N_, S=MAX_S, R=MAX_R_, P=MAX_P_), truncated_rows=trunc, include_dart_failures=include_dart_failures,
                multi_m=multi_m, statuses=list(statuses))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


class PackedChunkDataset:
    def __init__(self, packed_dir: Path, stride: int = 1):
        """stride > 1 keeps rows with t % stride == 0 of a stride-1 pack: the same chunk set that
        pack_dataset(..., stride=stride) would produce (episode_samples uses range(0, T, stride))."""
        self.dir = Path(packed_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.arr = {p.stem: np.load(p, mmap_mode="r") for p in self.dir.glob("*.npy")}
        self.H = self.meta["H"]
        self.rows = None
        if stride > 1:
            if self.meta["stride"] != 1:
                raise ValueError("row subsampling needs a stride-1 pack")
            self.rows = np.nonzero(np.asarray(self.arr["t"]) % stride == 0)[0]

    def __len__(self):
        return self.meta["n"] if self.rows is None else len(self.rows)

    def batches(self, batch_size, rng, shuffle=True, drop_last=True, start_batch=0):
        idx = np.arange(len(self))
        if shuffle:
            idx = np.array(rng.sample(range(len(self)), len(self)))
        if self.rows is not None:
            idx = self.rows[idx]
        stop = len(idx) - (len(idx) % batch_size if drop_last else 0)
        for i in range(start_batch * batch_size, stop, batch_size):
            sel = np.sort(idx[i:i + batch_size])
            if len(sel) == 0:
                break
            yield self.collate(sel)

    def collate(self, sel):
        A = {k: np.asarray(v[sel]) for k, v in self.arr.items()}
        B = len(sel)
        Tn = {b: max(int(A[f"len_{b}"].max()), 1) for b in BANKS}
        offs, o = {}, 0
        for b in BANKS:
            offs[b] = o
            o += Tn[b]
        C = o
        N = max(int(A["n_nodes"].max()), 1)
        toks = {b: torch.from_numpy(A[f"tok_{b}"][:, :Tn[b]].astype(np.float32)) for b in BANKS}
        masks = {b: torch.from_numpy(np.arange(Tn[b])[None, :] < A[f"len_{b}"][:, None]) for b in BANKS}
        kinds = {b: torch.from_numpy(A[f"kind_{b}"][:, :Tn[b]].astype(np.int64)) for b in BANKS}
        texts = {b: torch.from_numpy(A[f"text_{b}"][:, :Tn[b]].astype(np.float32)) for b in BANKS}
        offs_arr = np.array([offs[b] for b in BANKS])
        ctx_rel = np.zeros((B, C, C, N_REL), bool)
        act_rel = np.zeros((B, N, C, N_REL), bool)
        node_rel = np.zeros((B, N, N, N_REL), bool)
        ptr = -np.ones((B, A["ptr"].shape[1], 2), np.int64)
        for i in range(B):
            R = A["rel"][i, :A["n_rel"][i]].astype(np.int64)
            n = int(A["n_nodes"][i])
            if len(R):
                ok = (R[:, 3] < np.array([Tn[b] for b in BANKS])[R[:, 2]]) & \
                     ((R[:, 0] == -1) | (R[:, 1] < np.array([Tn[b] for b in BANKS])[np.clip(R[:, 0], 0, 3)]))
                R = R[ok]
                kc = offs_arr[R[:, 2]] + R[:, 3]
                a_ = R[:, 0] == -1
                act_rel[i, R[a_, 1], kc[a_], R[a_, 4]] = True
                c_ = ~a_
                ctx_rel[i, offs_arr[R[c_, 0]] + R[c_, 1], kc[c_], R[c_, 4]] = True
                kp = (R[:, 0] == 0) & (R[:, 2] == 0) & (R[:, 4] == 16) & (R[:, 1] < n) & (R[:, 3] < n)
                node_rel[i, R[kp, 1], R[kp, 3], 16] = True
                node_rel[i, R[kp, 3], R[kp, 1], 16] = True
            P = A["ptr"][i, :A["n_ptr"][i]].astype(np.int64)
            if len(P):                 # drop pointers into tokens cut by bank truncation (else index past the bank)
                ln = np.array([A[f"len_{b}"][i] for b in BANKS])
                P = P[(P[:, 1] < ln[P[:, 0]]) & (P[:, 3] < ln[P[:, 2]])]
            if len(P):
                ptr[i, :len(P), 0] = offs_arr[P[:, 0]] + P[:, 1]
                ptr[i, :len(P), 1] = offs_arr[P[:, 2]] + P[:, 3]
        nm = np.arange(N)[None, :] < A["n_nodes"][:, None]
        batch = Batch(toks, masks, kinds, texts, offs, torch.cat([masks[b] for b in BANKS], 1),
                      torch.from_numpy(A["node"][:, :N].astype(np.float32)), torch.from_numpy(nm),
                      torch.from_numpy(ctx_rel), torch.from_numpy(act_rel), torch.from_numpy(node_rel),
                      torch.from_numpy(ptr), {})
        S = Tn["scene"]
        lab = {k: torch.from_numpy(A[k][:, :S].astype(np.float32 if A[k].dtype == np.float16 else A[k].dtype))
               for k in ("held", "contact", "visible", "focus", "slot_valid", "rel_tcp", "future_disp", "gaze")}
        a = torch.from_numpy(A["a"][:, :, :N].astype(np.float32))[..., None]
        v = torch.from_numpy(A["valid"][:, :, :N])
        eff = torch.from_numpy(A["eff"].astype(np.float32))
        return batch, a, v, lab, eff
