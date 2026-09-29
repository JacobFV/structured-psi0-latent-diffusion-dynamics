"""Ψ₀ training data (W10; psi1z `features` + `data` + `compat_psi`, and the label logging of `replay_labels`, D-140).

1. Frozen-VLM feature cache (`cache_features`, psi venv, GPU): Ψ₀'s SIMPLE recipe freezes the VLM, so its last-layer
   hidden states are a deterministic function of (image, instruction); caching them makes the matched fine-tunes cheap
   and gives both arms identical inputs. Image augmentation is OFF for both arms (upstream trains with it; reported).
   Per frame: hidden [N, 2048] bf16, mask, ids, normalized state [36], normalized chunk [30, 36], action mask, raw
   actions, episode/frame, and the target-object name token positions (entity pooling; deploy-time information).
   Shards are flattened into a page-cache-backed memmap (`hidden.npy` + `small.pt`).
2. Probe labels (`LabelRecorder` hook on the `simple` env driven by `psi0_replay`, MuJoCo-only, no rendering): per step
   target pose, pelvis pose, palm positions, hand-target contacts and contact points from `env.truth()`. Source
   `privileged:sim_replay`: LABELS ONLY, they never enter a policy input.
3. `CachedDataset` / `collate`: frame t of episode e -> VLM features, state0, actions, amask, realization tick j and
   state_j, and the labels at the packet knots (see `EpisodeLabels.labels`).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from rrp.policies.psi0 import find_subseq, load_launch_config, object_name, psi_home, psi_runtime
from rrp.policies.psi0.nets import KNOT_STEPS, TP

LABEL_SOURCE = "privileged:sim_replay"
MP_WARMUP = 60        # the MP agent stands 60 steps before the first recorded row: label step t <-> recorded row t


# ---------------------------------------------------------------- lerobot video decode (psi venv: torchvision 0.29)
# torch 2.14 / torchvision 0.29 (needed for sm_121) removed `torchvision.io.VideoReader`, which lerobot's "pyav" path
# uses: replace lerobot's decode_video_frames with an exact-timestamp PyAV decoder (same contract: float32 [T, C, H, W]
# in [0, 1], nearest frame within tolerance); decoded videos are cached per path (LRU of 2).
_VCACHE: "OrderedDict[str, tuple[np.ndarray, np.ndarray]]" = OrderedDict()


def _decode_all(path: str):
    if path in _VCACHE:
        _VCACHE.move_to_end(path)
        return _VCACHE[path]
    import av
    frames, ts = [], []
    with av.open(path) as c:
        s = c.streams.video[0]
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24"))
            ts.append(float(f.pts * s.time_base))
    out = (np.stack(frames), np.asarray(ts))
    _VCACHE[path] = out
    while len(_VCACHE) > 2:
        _VCACHE.popitem(last=False)
    return out


def decode_video_frames(video_path, timestamps, tolerance_s, backend=None):
    frames, ts = _decode_all(str(video_path))
    q = np.asarray(timestamps, dtype=np.float64)
    d = np.abs(q[:, None] - ts[None])
    i = d.argmin(1)
    if not (d[np.arange(len(q)), i] < tolerance_s).all():
        raise AssertionError(f"timestamps {q} outside tolerance {tolerance_s} in {video_path}")
    return torch.from_numpy(frames[i]).permute(0, 3, 1, 2).float() / 255.0

def patch_lerobot_video():
    import lerobot.datasets.lerobot_dataset as ld
    import lerobot.datasets.video_utils as vu
    vu.decode_video_frames = decode_video_frames
    if hasattr(ld, "decode_video_frames"):
        ld.decode_video_frames = decode_video_frames


# ---------------------------------------------------------------- feature cache
def cache_features(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True, help="Psi0 VLM dir (HF format) or a fine-tuned run (uses its vlm_model.*)")
    ap.add_argument("--run-dir", required=True, help="run dir whose argv/run_config define the data transform")
    ap.add_argument("--data-root", default=str(psi_home() / "data/simple"))
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=2048)
    a = ap.parse_args(argv)
    patch_lerobot_video()
    psi_runtime()
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    lc = load_launch_config(Path(a.run_dir), a.data_root, a.repo_id)
    proc = AutoProcessor.from_pretrained(a.vlm)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(a.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    ds = lc.data(split="train", transform_kwargs=dict(vlm_processor=proc, no_aug=True))
    base = ds.raw_dataset.base_dataset
    n_eps = ds.raw_dataset.meta.total_episodes
    idx = []
    for e in range(n_eps):
        lo, hi = int(base.episode_data_index["from"][e]), int(base.episode_data_index["to"][e])
        idx += [(e, i - lo, i) for i in range(lo, hi, a.stride)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = proc.tokenizer
    meta = dict(repo_id=a.repo_id, vlm=a.vlm, run_dir=a.run_dir, frames=len(idx), stride=a.stride, episodes=n_eps,
                augmentation=False, created=time.strftime("%Y-%m-%dT%H:%M:%S"), shards=[])
    t0 = time.time()
    for s0 in range(0, len(idx), a.shard):
        chunk = idx[s0:s0 + a.shard]
        rec = dict(hidden=[], ids=[], state=[], actions=[], amask=[], raw_actions=[], ep=[], fr=[], ent=[], instr=[])
        for b0 in range(0, len(chunk), a.batch):
            items = [ds[g] for (_, _, g) in chunk[b0:b0 + a.batch]]
            msgs, texts = [], []
            for it in items:
                imgs = it.get("raw_images", it.get("observations"))
                content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": it["instruction"]}]
                m = [{"role": "user", "content": content}]
                msgs.append(m)
                texts.append(proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
            image_inputs, video_inputs = process_vision_info([m for m in msgs], image_patch_size=16)
            inp = proc(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = vlm(**inp, output_hidden_states=True, return_dict=True).hidden_states[-1]
            for j, it in enumerate(items):
                L = int(inp["attention_mask"][j].sum())
                ids = inp["input_ids"][j][inp["attention_mask"][j].bool()].tolist()
                hid = h[j][inp["attention_mask"][j].bool()].to(torch.bfloat16).cpu()
                assert hid.shape[0] == L
                name = object_name(it["instruction"]) or ""
                ent = []
                for variant in (" " + name, name):
                    ent = find_subseq(ids, tok.encode(variant, add_special_tokens=False)) if name else []
                    if ent:
                        break
                rec["hidden"].append(hid); rec["ids"].append(torch.tensor(ids, dtype=torch.int32))
                rec["state"].append(torch.as_tensor(np.asarray(it["states"], dtype=np.float32)).reshape(-1))
                rec["actions"].append(torch.as_tensor(np.asarray(it["actions"], dtype=np.float32)))
                am = np.asarray(it.get("actions_mask", np.ones((30, 36))), dtype=bool)
                rec["amask"].append(torch.as_tensor(np.broadcast_to(am if am.ndim == 2 else am[:, None], (30, 36)).copy()))
                rec["raw_actions"].append(torch.as_tensor(np.asarray(it["raw_actions"], dtype=np.float32)))
                rec["ent"].append(torch.tensor(ent, dtype=torch.int32))
                rec["instr"].append(it["instruction"])
            rec["ep"] += [c[0] for c in chunk[b0:b0 + a.batch]]; rec["fr"] += [c[1] for c in chunk[b0:b0 + a.batch]]
        p = out / f"shard_{s0 // a.shard:04d}.pt"
        torch.save(rec, p)
        meta["shards"].append(p.name)
        print(f"[features] {s0 + len(chunk)}/{len(idx)} frames, {time.time() - t0:.0f}s", flush=True)
    meta["seconds"] = time.time() - t0
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    build_memmap(out)
    for p in out.glob("shard_*.pt"):
        p.unlink()



# ---------------------------------------------------------------- dataset + labels
def _yaw(q):                       # MuJoCo quat wxyz
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_rot(q):                  # MuJoCo wxyz -> rotation matrix
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def grasp_point_object_frame(cpos_world, obj_pos, obj_quat):
    """Contact point expressed in the target's object frame (translation + rotation removed)."""
    return _quat_rot(obj_quat).T @ (np.asarray(cpos_world) - np.asarray(obj_pos))


def face_class(p_obj):
    """6-way contact face: dominant axis of the object-frame contact point and its sign (+x, -x, +y, -y, +z, -z)."""
    ax = int(np.argmax(np.abs(p_obj)))
    return 2 * ax + (0 if p_obj[ax] >= 0 else 1)


def _to_frame(p, origin, yaw):
    d = p - origin
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1], d[..., 2]], -1)


class EpisodeLabels:
    def __init__(self, npz: Path):
        d = np.load(npz)
        self.target = d["obj__target"][:, :3]
        self.pelvis = d["pelvis"]
        self.palm = np.stack([d["palm__left"], d["palm__right"]], 1)           # [T, 2, 3]
        self.contact = np.stack([d["contact__left"], d["contact__right"]], 1)  # [T, 2]
        self.T = len(self.target)
        # per-packet binding label: the hand in contact with the target at the packet end, else the hand whose next
        # contact starts first after t (-1: none). Time-varying for bimanual tasks (handover).
        self.z0 = float(self.target[0, 2])
        # grasp-region affordance labels (roadmap #24): first contact of each hand with the target, object frame
        self.grasp = None
        if "cpos__left" in d.files:
            tq = d["obj__target"][:, 3:7]
            self.grasp = []
            for h, key in enumerate(("cpos__left", "cpos__right")):
                cp = d[key]; ok = ~np.isnan(cp).any(1)
                if ok.any():
                    i = int(np.argmax(ok))
                    po = grasp_point_object_frame(cp[i], self.target[i], tq[i])
                    self.grasp.append((po.astype(np.float32), face_class(po)))
                else:
                    self.grasp.append(None)

    def _active(self, t):
        i1 = self.at(t + TP - 1)
        c1 = self.contact[i1]
        if c1.any():
            return int(np.argmax(c1)) if c1.sum() == 1 else 1 - int(self.contact[self.at(t)][1] if self.contact[self.at(t)].any() else 0)
        fut = self.contact[self.at(t):]
        if not fut.any():
            return -1
        first = [int(np.argmax(fut[:, h])) if fut[:, h].any() else 10 ** 9 for h in range(2)]
        return int(np.argmin(first))

    def at(self, t):
        # replay step s (1-based) is the state after executing recorded row s-1; frame t's observation ~ step t
        return min(max(t, 0), self.T - 1)

    def labels(self, t):
        i0 = self.at(t)
        org, yaw = self.pelvis[i0, :3], float(_yaw(self.pelvis[i0, 3:7]))
        ks = [self.at(t + k) for k in KNOT_STEPS]
        tgt = self.target[ks]
        hd = np.linalg.norm(self.palm[ks] - tgt[:, None], axis=-1)
        i1 = self.at(t + TP - 1)
        dp = _to_frame(self.pelvis[i1, :3], org, yaw)
        dyaw = (float(_yaw(self.pelvis[i1, 3:7])) - yaw + math.pi) % (2 * math.pi) - math.pi
        return dict(hand_dist=hd.astype(np.float32), contact=self.contact[ks].astype(np.int64),
                    lift=(tgt[:, 2] >= self.z0 + 0.03).astype(np.int64),
                    target_pos=_to_frame(tgt, org, yaw).astype(np.float32),
                    active_hand=np.int64(self._active(t)),
                    base_disp=np.array([dp[0], dp[1], dyaw], dtype=np.float32),
                    **(self._grasp_labels() if self.grasp is not None else {}))

    def _grasp_labels(self):
        pt = np.zeros((2, 3), np.float32); face = np.zeros(2, np.int64); valid = np.zeros(2, bool)
        for h, g in enumerate(self.grasp):
            if g is not None:
                pt[h], face[h], valid[h] = g[0], g[1], True
        return dict(grasp_pt=pt, grasp_face=face, grasp_valid=valid)


def build_memmap(feat_dir):
    """Flatten the shards into hidden.u16 (memmap, [N, L, 2048] bf16 bits; all prompts have the same length for a
    single-instruction task, else right-padded with lengths stored) + small.pt. Page-cache backed: keeps resident
    memory low (the host watchdog sheds jobs above ~10 GB resident, research/tracks/psi0.md)."""
    feat_dir = Path(feat_dir)
    meta = json.loads((feat_dir / "meta.json").read_text())
    if (feat_dir / "small.pt").exists():
        return
    small = dict(state=[], actions=[], amask=[], ent=[], ep=[], fr=[], length=[])
    Lmax, N = 0, 0
    for sh in meta["shards"]:
        r = torch.load(feat_dir / sh, weights_only=False)
        Lmax = max(Lmax, max(h.shape[0] for h in r["hidden"])); N += len(r["ep"])
    mm = np.lib.format.open_memmap(feat_dir / "hidden.npy", mode="w+", dtype=np.uint16, shape=(N, Lmax, 2048))
    n = 0
    for sh in meta["shards"]:
        r = torch.load(feat_dir / sh, weights_only=False)
        for i in range(len(r["ep"])):
            h = r["hidden"][i]
            mm[n, :h.shape[0]] = h.view(torch.int16).numpy().view(np.uint16)
            small["length"].append(h.shape[0])
            for k in ("state", "actions", "amask", "ent", "ep", "fr"):
                small[k].append(r[k][i])
            n += 1
    mm.flush(); del mm
    torch.save(small, feat_dir / "small.pt")


class CachedDataset(torch.utils.data.Dataset):
    """Cached frozen-VLM features (memmapped bf16) + actions/states + replay labels."""

    def __init__(self, feat_dir, label_dir=None, episodes=None, max_j=8, seed=0, load_hidden=True):
        feat_dir = Path(feat_dir)
        build_memmap(feat_dir)
        small = torch.load(feat_dir / "small.pt", weights_only=False)
        self.hidden = np.load(feat_dir / "hidden.npy", mmap_mode="r") if load_hidden else None
        self.items = []
        for i in range(len(small["ep"])):
            if episodes is not None and small["ep"][i] not in episodes:
                continue
            self.items.append({"row": i, **{k: small[k][i] for k in ("state", "actions", "amask", "ent", "ep", "fr", "length")}})
        self.index = {(it["ep"], it["fr"]): n for n, it in enumerate(self.items)}
        self.labels = {}
        if label_dir is not None:
            for e in sorted({it["ep"] for it in self.items}):
                p = Path(label_dir) / f"episode_{e:06d}.npz"
                if p.exists():
                    self.labels[e] = EpisodeLabels(p)
        self.max_j = max_j
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        it = self.items[n]
        j = int(self.rng.integers(0, self.max_j))
        m = self.index.get((it["ep"], it["fr"] + j))
        if m is None:
            j, m = 0, n
        if self.hidden is not None:
            h = torch.from_numpy(np.array(self.hidden[it["row"], :it["length"]])).view(torch.bfloat16)
        else:
            h = torch.zeros(1, 2048, dtype=torch.bfloat16)
        out = dict(hidden=h, ent=it["ent"], state0=it["state"][:36], actions=it["actions"],
                   amask=it["amask"].float(), j=torch.tensor(j), state_j=self.items[m]["state"][:36],
                   ep=it["ep"], fr=it["fr"], has_labels=it["ep"] in self.labels)
        if it["ep"] in self.labels:
            out["labels"] = {k: torch.as_tensor(v) for k, v in self.labels[it["ep"]].labels(it["fr"]).items()}
        return out


def collate(batch):
    B = len(batch)
    N = max(b["hidden"].shape[0] for b in batch)
    D = batch[0]["hidden"].shape[1]
    hid = torch.zeros(B, N, D, dtype=torch.bfloat16)
    mask = torch.zeros(B, N, dtype=torch.bool)
    ent = torch.zeros(B, N, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b["hidden"].shape[0]
        hid[i, :L] = b["hidden"]; mask[i, :L] = True
        if len(b["ent"]) and L > 1:
            ent[i, b["ent"].long()] = True
    out = dict(hidden=hid, mask=mask, ent=ent,
               **{k: torch.stack([b[k] for b in batch]) for k in ("state0", "actions", "amask", "j", "state_j")},
               ep=torch.tensor([b["ep"] for b in batch]), fr=torch.tensor([b["fr"] for b in batch]))
    if all(b["has_labels"] for b in batch):
        out["labels"] = {k: torch.stack([b["labels"][k] for b in batch]) for k in batch[0]["labels"]}
    return out


# ---------------------------------------------------------------- replay labels (hook)
class LabelRecorder:
    """Harness hook: records `env.truth()` every step of a `psi0_replay` episode on `simple` and writes
    `<out>/episode_<e:06d>.npz` in the format `EpisodeLabels` reads (qpos included for replay-fidelity checks)."""

    def __init__(self, out, mp: bool):
        self.out, self.mp = Path(out), mp
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = {}

    def on_reset(self, i, env, obs):
        self.log[i] = dict(pelvis=[], palm_l=[], palm_r=[], contact_l=[], contact_r=[], cpos_l=[], cpos_r=[],
                           target=[], reward=[], qpos=[])

    def on_act(self, i, obs, act):
        return act

    def on_step(self, i, env, act, step):
        t = env.truth(); L = self.log[i]
        L["pelvis"].append(t["pelvis"]); L["target"].append(t["objects"]["target"]); L["reward"].append(t["reward"])
        L["palm_l"].append(t["palm"]["left"]); L["palm_r"].append(t["palm"]["right"])
        L["contact_l"].append(t["contact"]["left:target"]); L["contact_r"].append(t["contact"]["right:target"])
        L["cpos_l"].append(t["contact_point"]["left"]); L["cpos_r"].append(t["contact_point"]["right"])
        L["qpos"].append(np.asarray(step.observation.measured_node_state.qpos, np.float32))

    def on_end(self, i, env, ep):
        L = {k: np.asarray(v) for k, v in self.log.pop(i).items()}
        if self.mp:
            L = {k: v[MP_WARMUP:] for k, v in L.items()}
        np.savez_compressed(self.out / f"episode_{ep.seed:06d}.npz", pelvis=L["pelvis"], reward=L["reward"], qpos=L["qpos"],
                            target_key="target", obj__target=L["target"], palm__left=L["palm_l"], palm__right=L["palm_r"],
                            contact__left=L["contact_l"], contact__right=L["contact_r"], cpos__left=L["cpos_l"],
                            cpos__right=L["cpos_r"])
        return dict(label_source=LABEL_SOURCE, label_steps=len(L["reward"]), max_reward=float(L["reward"].max(initial=0)))
