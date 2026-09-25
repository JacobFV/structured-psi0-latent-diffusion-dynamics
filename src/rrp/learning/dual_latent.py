"""Multi-assembly (dual-arm) support for the controller-facing latent packet.

The packet axis M lists the controllable grasping assemblies in a FIXED, declared order: the order of the
gripper/hand assembly tokens in the morph bank. For the dual scenarios this is robot order (r0, r1), and the
scenarios mount the `left` role on r0 and `right` on r1, so packet slot m == role order (left, right). For one body
carrying both grippers (ALOHA) it is that body's declared assembly order. The order and multiplicity are part of the
representation: the realizer routes each action node ONLY to its own assembly's packet slot, and probes address
slots by fixed opaque codes. A missing assembly is an explicit null slot (mask False).

All quantities here are PUBLIC (derived from the featurized PolicyInput: morph tokens, relations, runtime
status tokens, touch/width tokens). Privileged per-manipulator labels are permuted into packet order via the
public role bindings recorded in the episode meta; they are supervision/diagnostic targets only.
"""
from __future__ import annotations

import numpy as np

from rrp.data.features import HASH_DIM, text_hash

REL_NODE_IN_ASM, REL_NODE_ACTOR_OF, REL_KIN_PARENT = 1, 13, 16
BANK_MORPH, BANK_TASK, BANK_INTERACT = 0, 2, 3
ASM_GRIPPER_ONEHOT = (1, 2)


def gripper_token_positions(pi, max_m: int = 2) -> list[int]:
    """Morph-bank positions of grasping-assembly tokens, in packet order (same rule as assembly_tokens)."""
    tok, kind = pi.tokens["morph"], pi.token_kind["morph"]
    pos = [j for j in range(len(kind)) if kind[j] == 2 and tok[j, list(ASM_GRIPPER_ONEHOT)].sum() > 0.5]
    return pos[:max_m]


def node_assembly_index(pi, max_m: int = 2) -> np.ndarray:
    """Packet slot owning each action node. A node in a grasping assembly -> that slot; otherwise the slot of the
    first grasping-assembly node among its kinematic descendants (an arm drives its own gripper); fallback: robot
    segment (meta nodes_per_robot), else 0."""
    N = pi.act_node_feats.shape[0]
    gpos = gripper_token_positions(pi, max_m)
    slot_of_tok = {p: m for m, p in enumerate(gpos)}
    R = np.asarray(pi.relations).reshape(-1, 5)
    own = np.full(N, -1)
    parent = np.full(N, -1)
    for qb, qi, kb, ki, r in R.tolist():
        if qb == -1 and kb == BANK_MORPH and qi < N:
            if r == REL_NODE_IN_ASM and ki in slot_of_tok:
                own[qi] = slot_of_tok[ki]
            elif r == REL_KIN_PARENT and ki < N:
                parent[qi] = ki
    out = own.copy()
    for i in range(N):
        if out[i] >= 0:
            continue
        # descendants of i that are in a grasping assembly
        for j in range(N):
            if own[j] < 0:
                continue
            k, hops = j, 0
            while k >= 0 and hops <= N:
                if k == i:
                    out[i] = own[j]
                    break
                k, hops = parent[k], hops + 1
            if out[i] >= 0:
                break
    if (out < 0).any():
        npr = pi.meta.get("nodes_per_robot")
        if npr and len(npr) == len(gpos):
            seg = np.repeat(np.arange(len(npr)), npr)[:N]
            out = np.where(out < 0, seg, out)
    return np.where(out < 0, 0, out).astype(np.int64)


def local_sensors_multi(pi, max_m: int = 2) -> np.ndarray:
    """Declared local sensors per packet slot [M,4]: touch summary (log max, count>0.2N, log mean) + grip width
    of the assembly each sensor token is linked to (node_in_assembly relation interact -> morph)."""
    gpos = gripper_token_positions(pi, max_m)
    slot_of_tok = {p: m for m, p in enumerate(gpos)}
    out = np.zeros((max_m, 4), np.float32)
    th = text_hash("touch")
    it, ik = pi.tokens["interact"], pi.token_kind["interact"]
    link = {}
    for qb, qi, kb, ki, r in np.asarray(pi.relations).reshape(-1, 5).tolist():
        if qb == BANK_INTERACT and kb == BANK_MORPH and r == REL_NODE_IN_ASM and ki in slot_of_tok:
            link[qi] = slot_of_tok[ki]
    for j in range(len(ik)):
        if ik[j] != 1 or j not in link:
            continue
        m = link[j]
        if np.allclose(it[j, 16:], th):
            out[m, :3] = it[j, 13:16]
        else:
            out[m, 3] = it[j, 13]
    return out


def assembly_operators(pi, operators: list[str], max_m: int = 2) -> np.ndarray:
    """Public per-slot subtask label: operator of the first ACTIVE event whose actor nodes belong to that slot
    (node_actor_of relations + runtime status one-hot in the task bank). 0 ('none') if no active event."""
    node_slot = node_assembly_index(pi, max_m)
    tt, tk = pi.tokens["task"], pi.token_kind["task"]
    active = {j for j in range(len(tk)) if tk[j] == 0 and tt[j, HASH_DIM + 2] > 0.5}
    hashes = [text_hash(op) for op in operators]
    ev_slots: dict[int, set] = {}
    for qb, qi, kb, ki, r in np.asarray(pi.relations).reshape(-1, 5).tolist():
        if qb == -1 and kb == BANK_TASK and r == REL_NODE_ACTOR_OF and ki in active and qi < len(node_slot):
            ev_slots.setdefault(ki, set()).add(int(node_slot[qi]))
    out = np.zeros(max_m, np.int64)
    for m in range(max_m):
        for ev in sorted(ev_slots):
            if m in ev_slots[ev]:
                for k, h in enumerate(hashes):
                    if np.allclose(tt[ev, :HASH_DIM], h, atol=1e-5):
                        out[m] = k
                        break
                break
    return out


def label_columns_for_slots(meta: dict, manipulators: list[str], max_m: int = 2, spec_grip_order=None) -> list[int]:
    """Column of the privileged per-manipulator label arrays (order `manipulators`) that belongs to each packet
    slot. Two-body pairs: slot m = robot m. One body with several grippers: `spec_grip_order` (list of assembly ids
    in declared order) is required. Missing -> -1 (null slot)."""
    roles = meta.get("roles", {})
    robots = sorted({v["robot"] for v in roles.values()})
    cols = [-1] * max_m
    if len(robots) == len(roles):                      # one grasping assembly per robot
        for ent, v in roles.items():
            if v["robot"] < max_m and ent in manipulators:
                cols[v["robot"]] = manipulators.index(ent)
        return cols
    if spec_grip_order is None:
        raise ValueError("one body with several grippers: need the declared gripper assembly order")
    for ent, v in roles.items():
        if v["assembly"] in spec_grip_order and ent in manipulators:
            m = spec_grip_order.index(v["assembly"])
            if m < max_m:
                cols[m] = manipulators.index(ent)
    return cols


def multi_labels(lab: dict, cols: list[int], S: int, max_s: int, max_m: int = 2) -> dict:
    """Privileged per-manipulator labels permuted into packet slot order (supervision / diagnostics only)."""
    held = np.zeros((max_s, max_m), bool)
    contact = np.zeros((max_s, max_m), bool)
    rel = np.zeros((max_s, max_m, 3), np.float32)
    for m, c in enumerate(cols):
        if c < 0:
            continue
        held[:S, m] = lab["slot_held"][:S, c]
        contact[:S, m] = lab["slot_contact"][:S, c]
        rel[:S, m] = lab["slot_rel_tcp"][:S, c]
    return dict(held_m=held, contact_m=contact, rel_tcp_m=rel.astype(np.float16))


def concat_packed(dirs: list, out_dir) -> dict:
    """Concatenate packed dirs with identical array shapes (episode ids offset; robot ids remapped)."""
    import json
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = [json.loads((Path(d) / "meta.json").read_text()) for d in dirs]
    names = sorted(p.stem for p in Path(dirs[0]).glob("*.npy"))
    for d in dirs[1:]:
        if sorted(p.stem for p in Path(d).glob("*.npy")) != names:
            raise ValueError(f"array set differs in {d}")
    robot_ids: dict[str, int] = {}
    remaps = []
    for m in metas:
        remaps.append({v: robot_ids.setdefault(k, len(robot_ids)) for k, v in m["robot_ids"].items()})
    n = sum(m["n"] for m in metas)
    for name in names:
        arrs = [np.load(Path(d) / f"{name}.npy", mmap_mode="r") for d in dirs]
        out = np.lib.format.open_memmap(out_dir / f"{name}.npy", mode="w+", dtype=arrs[0].dtype,
                                        shape=(n,) + arrs[0].shape[1:])
        o, ep_off = 0, 0
        for a, m, rm in zip(arrs, metas, remaps):
            k = len(a)
            if name == "ep_idx":
                out[o:o + k] = np.asarray(a) + ep_off
                ep_off += int(np.asarray(a).max()) + 1 if k else 0
            elif name == "robot_id":
                out[o:o + k] = np.vectorize(lambda x: rm[int(x)])(np.asarray(a)) if k else a
            else:
                out[o:o + k] = a
            o += k
        out.flush()
        del out
    meta = dict(metas[0], n=n, robot_ids=robot_ids, source=[m["source"] for m in metas],
                parts=[dict(dir=str(d), n=m["n"], truncated_rows=m.get("truncated_rows")) for d, m in zip(dirs, metas)],
                robots=sorted({r for m in metas for r in m.get("robots", [])}))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
