"""Training for the legged latent packet (stage A representation E+R+P; stage B system-i flow; post-hoc probes).

Stage A:  z ~ E(public ctx_t, morphology, demonstrated native targets a[t:t+40])
          L_real = || R(z, phase=j*0.02, local state_{t+j}) - a*_{t+j} ||^2  on policy joints, j ~ U{0..27}
          L_sem  = readout_loss(P(z), labels_t, specs) + estimates_loss(rc, specs)   (per-factor `weight`; 0 =>
                   latent_nosem, capacity matched)
          L      = L_real + L_sem + beta KL
Stage B:  LeggedFlow generates z from PUBLIC context only toward the frozen E mean (standardized target);
          packet-semantic loss through the frozen P on z_hat_clean for tau >= tau_min.
Episodes are split per body into train / held-out (every 20th episode held out); all evaluations here use the
held-out episodes (same bodies, same teacher distribution, unseen seeds).

Relation factors (D-144 / D-146, docs/relations.md 11): `latent.factors` lists the `probe.legged.*` queries AND the
relational factors (`preset:legged` = the nine `legged-rel-v1` edges + `leg.foothold` + `leg.com_support`; absent =
`legged-none`, the pre-relations nets). The resolved list builds E / F (`nets.legged_latent`), is stamped into the
checkpoint (`versions["factors"]` + `factors`), and is checked at load (`load_legged_rep`); a checkpoint with no stamp is
a `legged-none` one and loads only into `legged-none`.

usage: python -m rrp.cli train legged-latent {rep,flow,probe} --config C.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rrp.policies.features.legged import MAX_N, MAX_M, KNOT_TIMES, TICK_DT, H, MAX_J, KNOT_TICKS
from rrp.core.sealed import SealedSplit
from rrp.policies.bundles import _dev, checkpoint_provenance, legged_flags
from rrp.policies.nets.legged_latent import (LeggedFlow, build_legged_rep, legged_probe, legged_probe_read,
                                             legged_specs, probe_metrics, probe_terms, relational_specs,
                                             remap_legged_probe_state)
from rrp.policies.nets.probes import readout_loss, readout_metrics
from rrp.policies.nets.semantic_latent import legacy_latent_factors, packet_semantic_weight
from rrp.policies.relations.base import (FactorError, estimates_loss, require_factors, resolve, stamp_versions,
                                         to_json)

# D-144 sweep-flags (docs/relations.md 10, closing rel-r2c's open question 1 for legged now that R4 has merged) and
# its 2026-09-30 follow-up (retiring the retired legged_v2_* DAGs + the legacy legged / t1_diag rep configs'
# OWN flat `semantic_weight` / `probe_lv_min`, not just this file's internal reads): every legged dag and rep
# config under those paths now renders `params.latent.factors` directly (a `probe.legged.*` FactorSpec per query,
# the same shape `nets/semantic_latent.py`'s `LATENT_LEGACY_KEYS`-driven table uses for arm's `LatentConfig`) --
# `core/runconfig.py`'s `FLAG_SPEC[("legged", "train_rep")]` no longer maps `probe_lv_min` either. `_legged_probe_
# factors` below therefore no longer reads the flat keys itself; the ONE remaining legacy-key fallback (for
# not-yet-migrated on-disk configs / already-trained checkpoints whose saved `latent` blob predates this change,
# e.g. the archived `legged_latent/rep_{sem,nosem}_v1` configs, deliberately left flat -- see that codemod's own note) is
# `nets/semantic_latent.py::legacy_latent_factors`, the SAME `LATENT_LEGACY_KEYS` table arm's `LatentConfig.__init__`
# reads, parameterized here by legged's own prefix/query set.
_LEGGED_PROBE_QUERIES = ("contact", "goal", "disp", "subtask", "fall")


def _legged_probe_factors(lc: dict) -> tuple:
    if "factors" in lc:
        return tuple(lc["factors"])
    return legacy_latent_factors(lc, prefix="probe.legged", queries=_LEGGED_PROBE_QUERIES, omit_default_lv_min=True)


def _legged_specs(lc: dict) -> tuple:
    """The resolved factor list of a latent config: its probe queries and its relational factors (`legged-none`
    when it lists none)."""
    return resolve(_legged_probe_factors(lc), family="legged")


def _unit_probe_specs(specs) -> tuple:
    """`specs` with every readout at unit weight (the flow stage weights the whole probe term once, by the flow's own
    `packet_semantic_weight`; each query's `params.lv_min` is kept)."""
    from rrp.policies.relations.base import get_factor
    return tuple(replace(s, weight=None) if get_factor(s.name).form == "readout" else s for s in specs)


def _assert_sealed(cfg: dict) -> None:
    """Sealed-split guard (H5): no stage of this trainer reads a sealed body's or a held-out seed's data."""
    SealedSplit.load().assert_dataset_allowed(cfg["data"], cfg["bodies"])


def load_legged_rep(path, dev):
    """(cfg, E, R, P, result, specs) of a representation checkpoint, frozen and eval. The nets are built from the
    checkpoint's own resolved factor list; the saved `versions["factors"]` stamp must equal that list's structure hash
    (`require_factors`). A checkpoint written before the stamp existed is a `legged-none` one and is refused when its
    config asks for relational factors."""
    st = torch.load(str(path), map_location=dev, weights_only=False)
    prov = checkpoint_provenance(st, path)           # raises on a fingerprint mismatch; legacy files are marked
    if prov.legacy:
        print(f"[legged] {path}: legacy checkpoint ({prov.notes}); compatibility IDs are fingerprinted at load", flush=True)
    lc = st["cfg"]["latent"]
    specs = _legged_specs(lc)
    if "factors" in prov.versions:
        require_factors(prov.versions, specs)
    elif relational_specs(specs):
        raise FactorError(f"{path}: no factor stamp (a legged-none checkpoint) but its config lists relational factors "
                          f"{[s.name for s in relational_specs(specs)]}")
    E, R, P = build_legged_rep(lc, specs)
    E, R, P = E.to(dev), R.to(dev), P.to(dev)
    E.load_state_dict(st["E"]); R.load_state_dict(st["R"]); P.load_state_dict(remap_legged_probe_state(st["P"]))
    for m in (E, R, P):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return st["cfg"], E, R, P, st["result"], specs


# Optional per-tick shard arrays (written by the terrain collect, D-146 / H4): the PUBLIC terrain scan (`terrain`
# [T,77] elevation, `terrain_valid` [T,77]) and the training-only relation labels (`foothold_cell` [T,M]: -2 absent,
# -1 planted, >= 0 the scan cell of the next landing; `com_support` [T] margin, `com_support_valid` [T]). All-or-none
# across the shards of a dataset; a dataset without them trains with those labels masked and no terrain tokens.
_TERRAIN_KEYS = ("terrain", "terrain_valid")
_RELATION_LABEL_KEYS = ("foothold_cell", "com_support", "com_support_valid")


class LeggedData:
    def __init__(self, root: Path, bodies: list[str], dev, holdout_every: int = 20, max_eps: int | None = None):
        self.dev = dev
        cols = {k: [] for k in ("q", "qd", "a", "amask", "imu", "touch", "osc", "ctx", "ev", "pose", "contact",
                                "body", "ep_end", "fell", "wpa", "wpb", "held_out")}
        self.static = dict(node_static=[], node_asm=[], asm_static=[], node_mask=[], asm_mask=[], asm_is_leg=[],
                           body_asm=[])
        self.body_names, self.ep_meta = [], []
        off = 0
        for bi, body in enumerate(bodies):
            shards = sorted((root / body).glob("s*.npz"))
            if not shards:
                raise FileNotFoundError(root / body)
            first = True
            for sh in shards:
                d = dict(np.load(sh))           # decompress each array once
                meta = json.loads(sh.with_suffix(".json").read_text())
                if first:
                    self._add_static(d, meta)
                    self.body_names.append(body)
                    first = False
                N, nf, npol = d["node_static"].shape[0], int(d["nf"]), int(d["n_policy"])
                ep = d["ep"]
                for e in np.unique(ep):
                    if max_eps and sum(1 for m in self.ep_meta if m["body"] == body) >= max_eps:
                        break
                    sl = np.nonzero(ep == e)[0]
                    em = meta["episodes"][int(e)]
                    T = len(sl)
                    gid = len(self.ep_meta)
                    ho = (em["seed"] % holdout_every) == (holdout_every - 1)
                    self.ep_meta.append(dict(body=body, seed=em["seed"], status=em["status"], held_out=bool(ho),
                                             start=off, T=T))
                    pad = lambda x, w: np.pad(x, ((0, 0), (0, w - x.shape[1])))
                    cols["q"].append(pad(d["q"][sl], MAX_N)); cols["qd"].append(pad(d["qd"][sl], MAX_N))
                    cols["a"].append(pad(d["a"][sl], MAX_N))
                    am = np.zeros((T, MAX_N), bool); am[:, :npol] = True
                    cols["amask"].append(am)
                    cols["imu"].append(d["imu"][sl]); cols["osc"].append(d["osc"][sl].astype(np.float32))
                    cols["touch"].append(pad(d["touch"][sl].astype(np.float32), MAX_M))
                    cols["contact"].append(pad(d["contact"][sl].astype(np.float32), MAX_M))
                    cols["ctx"].append(d["ctx"][sl]); cols["ev"].append(d["ev"][sl]); cols["pose"].append(d["pose"][sl])
                    cols["body"].append(np.full(T, bi)); cols["ep_end"].append(np.full(T, off + T - 1))
                    cols["fell"].append(np.full(T, em["status"] == "fell"))
                    cols["wpa"].append(np.tile(np.array(em["waypoints"]["a"], np.float32), (T, 1)))
                    cols["wpb"].append(np.tile(np.array(em["waypoints"]["b"], np.float32), (T, 1)))
                    cols["held_out"].append(np.full(T, ho))
                    for k in _TERRAIN_KEYS + _RELATION_LABEL_KEYS:
                        if k in d:
                            x = d[k][sl]
                            if k == "foothold_cell":
                                x = np.pad(x.astype(np.int64), ((0, 0), (0, MAX_M - x.shape[1])), constant_values=-2)
                            elif k == "com_support":
                                x = x.astype(np.float32).reshape(T, 1)
                            cols.setdefault(k, []).append(x)
                    if "beh" in d:                   # DAgger buffer: stored stateless-expert chunk per tick
                        bh = d["beh"][sl].astype(np.float32)
                        cols.setdefault("beh", []).append(np.pad(bh, ((0, 0), (0, MAX_N - bh.shape[1]), (0, 0))))
                    off += T
        for k in _TERRAIN_KEYS + _RELATION_LABEL_KEYS:
            if k in cols and sum(len(x) for x in cols[k]) != sum(m["T"] for m in self.ep_meta):
                raise ValueError(f"{root}: optional shard array {k!r} is present in only some shards/episodes")
        t = lambda x: torch.from_numpy(np.concatenate(x)).to(dev)
        self.A = {k: t(v) for k, v in cols.items() if v}
        self.A["tick"] = torch.arange(off, device=dev)
        self.S = {k: torch.from_numpy(np.stack(v)).to(dev) for k, v in self.static.items()}
        self.n = off
        ho = self.A["held_out"].cpu().numpy()
        # rows that have a full horizon are not required: horizon indices clamp to the episode end
        self.train_idx = np.nonzero(~ho)[0]
        self.test_idx = np.nonzero(ho)[0]

    def _add_static(self, d, meta):
        N, M = d["node_static"].shape[0], d["asm_static"].shape[0]
        ns = np.zeros((MAX_N, d["node_static"].shape[1]), np.float32); ns[:N] = d["node_static"]
        na = np.zeros(MAX_N, np.int64); na[:N] = d["node_asm"]
        asm = np.zeros((MAX_M, d["asm_static"].shape[1]), np.float32); asm[:M] = d["asm_static"]
        nm = np.zeros(MAX_N, bool); nm[:N] = True
        amk = np.zeros(MAX_M, bool); amk[:M] = True
        leg = np.zeros(MAX_M, bool); leg[:int(d["nf"])] = True
        for k, v in (("node_static", ns), ("node_asm", na), ("asm_static", asm), ("node_mask", nm), ("asm_mask", amk),
                     ("asm_is_leg", leg), ("body_asm", np.int64(int(d["nf"])))):
            self.static[k].append(v)

    # ------------------------------------------------------------------ batches
    def ctx_batch(self, i):
        """PUBLIC inputs at row i (system i and E context)."""
        A, S = self.A, self.S
        bi = A["body"][i]
        b = {k: v[bi] for k, v in S.items()}
        b.update(q=A["q"][i], qd=A["qd"][i], imu=A["imu"][i], ctx=A["ctx"][i], asm_touch=A["touch"][i], osc=A["osc"][i])
        b.update({k: A[k][i] for k in _TERRAIN_KEYS if k in A})
        return b

    def train_batch(self, i):
        """`ctx_batch` plus the training-only `foothold_cell` label key (the relation graph turns it into the
        `foothold_next` label; it never enters a deploy forward, `legged_graph(deploy=True)` drops it)."""
        b = self.ctx_batch(i)
        if "foothold_cell" in self.A:
            b["foothold_cell"] = self.A["foothold_cell"][i]
        return b

    def beh(self, i):
        if "beh" in self.A:
            return self.A["beh"][i]
        idx = torch.minimum(i[:, None] + torch.arange(H, device=self.dev)[None], self.A["ep_end"][i][:, None])
        return self.A["a"][idx].transpose(1, 2)                             # [B,N,H]

    def labels(self, i):
        A = self.A
        end = A["ep_end"][i]
        kt = torch.minimum(i[:, None] + torch.tensor(KNOT_TICKS, device=self.dev)[None], end[:, None])   # [B,K]
        contact_k = A["contact"][kt]                                        # [B,K,M]
        pose = A["pose"][i]
        c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        ev = A["ev"][i]
        wp = torch.where((ev == 0)[:, None], A["wpa"][i], A["wpb"][i])
        dx, dy = wp[:, 0] - pose[:, 0], wp[:, 1] - pose[:, 1]
        goal = torch.stack([c * dx + s * dy, -s * dx + c * dy], -1) / 2.0
        p2 = A["pose"][torch.minimum(i + H, end)]
        ex, ey = p2[:, 0] - pose[:, 0], p2[:, 1] - pose[:, 1]
        dyaw = torch.remainder(p2[:, 2] - pose[:, 2] + math.pi, 2 * math.pi) - math.pi
        disp = torch.stack([(c * ex + s * ey) / 0.5, (-s * ex + c * ey) / 0.5, dyaw], -1)
        fall = A["fell"][i] & ((end - i) <= H)
        lab = dict(contact_k=contact_k, goal=goal, goal_valid=ev < 3, disp=disp, subtask=ev.clamp(max=3), fall=fall)
        if "com_support" in A:
            lab.update(com_support=A["com_support"][i], com_support_valid=A["com_support_valid"][i])
        return lab

    def realizer_batch(self, i, j):
        tj = torch.minimum(i + j, self.A["ep_end"][i])
        jj = tj - i
        b = self.ctx_batch(tj)
        return b, jj.float() * TICK_DT, self.A["a"][tj], self.A["amask"][tj]

    def hold_still(self, i):
        """Hold-still reference action at row i: the current joint position in action units."""
        sc = self.S["node_static"][self.A["body"][i]][..., 14].clamp(min=1e-6)       # action_scale column
        return self.A["q"][i] / sc * self.A["amask"][i]

    def sample(self, B, rng: np.random.Generator, test=False):
        pool = self.test_idx if test else self.train_idx
        return torch.from_numpy(rng.choice(pool, B)).to(self.dev)


def rep_step(E, R, P, data, i, j, specs, beta, train=True, qd_drop=0.0):
    """One stage-A loss. `specs`: the resolved factor list (probe queries + relational factors); each readout term is
    weighted by its own `FactorSpec.weight` inside `readout_loss` (`()` = realizer + KL only, no probe loss), the
    relational factors' pair estimates by `estimates_loss`."""
    b = data.train_batch(i)
    mu, lv, rc = E.encode(b, data.beh(i))
    z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if train else mu
    br, ph, a1, am = data.realizer_batch(i, j)
    if train and qd_drop > 0:                    # counter the proprioceptive (joint-velocity) shortcut, D-056
        keep = (torch.rand(len(i), 1, device=br["qd"].device) >= qd_drop).float()
        br = dict(br); br["qd"] = br["qd"] * keep
    pred = R(z, br, ph)
    m = am.float()
    l_real = (((pred - a1) ** 2) * m).sum() / m.sum()
    mm = b["asm_mask"][:, None, :, None].float().expand_as(mu)
    kl = (0.5 * (mu ** 2 + lv.exp() - 1 - lv) * mm).sum() / mm.sum()
    lab = data.labels(i)
    out = legged_probe_read(P, z, b["asm_mask"], b["body_asm"])
    l_sem, logs = readout_loss(*_terms(out, lab, b, specs)) if specs else (torch.zeros((), device=z.device), {})
    if rc is not None and rc.estimates:
        el, elogs, _ = estimates_loss(rc, specs)
        l_sem = l_sem + el
        logs.update(elogs)
    loss = l_real + l_sem + beta * kl
    logs.update(real=float(l_real.detach()), kl=float(kl.detach()), sem=float(l_sem.detach()))
    return loss, logs, (z, out, lab, b, pred, a1, m)


def _terms(out, lab, b, specs):
    """`readout_loss` / `readout_metrics` argument tuple (predictions, labels, specs, masks) for one probe read."""
    pred, labels, masks = probe_terms(out, lab, b)
    return pred, labels, specs, masks


def _agg(dst, res):
    for k, (x, n) in res.items():
        s_, n_ = dst.get(k, (0, 0)); dst[k] = (s_ + x, n_ + n)


def _probe_metrics(out, lab, b, specs) -> dict:
    """The five legacy packet-probe metrics, plus `readout_metrics` of any further query the specs read out
    (`leg.com_support`: mean absolute error of the stability margin at the body row)."""
    res = probe_metrics(out, lab, b)
    extra = [s for s in specs if s.name == "leg.com_support" and s.control != "off"]
    if extra and "com_support" in out:
        pred, labels, _, masks = _terms(out, lab, b, extra)
        res.update(readout_metrics(pred, labels, extra, masks))
    return res


def _fin(d):
    return {k: (x / n if n else None) for k, (x, n) in d.items()}


@torch.no_grad()
def eval_rep(E, R, P, data, specs=(), n_batches=40, seed=99):
    for m in (E, R, P):
        m.eval()
    rng = np.random.default_rng(seed)
    agg, sh, real, zero = {}, {}, [], []
    unit = _unit_probe_specs(specs)
    for _ in range(n_batches):
        i = data.sample(256, rng, test=True)
        j = torch.from_numpy(rng.integers(0, MAX_J + 1, 256)).to(data.dev)
        _, _, (z, out, lab, b, pred, a1, m) = rep_step(E, R, P, data, i, j, unit, 0.0, train=False)
        real.append(float((((pred - a1) ** 2) * m).sum() / m.sum()))
        zero.append(float(((a1 ** 2) * m).sum() / m.sum()))
        _agg(agg, _probe_metrics(out, lab, b, specs))
        perm = torch.randperm(len(i), device=data.dev)
        _agg(sh, _probe_metrics(legged_probe_read(P, z[perm], b["asm_mask"], b["body_asm"]), lab, b, specs))
    for m in (E, R, P):
        m.train()
    return dict(split="held_out_episodes", realize_mse=float(np.mean(real)), zero_action_mse=float(np.mean(zero)),
                probes=_fin(agg), probes_shuffled_z=_fin(sh))



def rng_state(rng) -> dict:
    """Every RNG a legged trainer draws from (numpy batch sampling; torch CPU; torch CUDA: posterior noise, qd dropout), so a
    resumed run continues the uninterrupted one exactly (W8; same as the arm fix 3d6e927)."""
    st = dict(rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda_rng"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: dict, rng) -> bool:
    """Restore rng_state(); returns False (INEXACT) for checkpoints written without the CUDA state."""
    rng.bit_generator.state = st["rng"]
    torch.set_rng_state(st["torch_rng"].cpu())
    if "cuda_rng" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in st["cuda_rng"]])
        return True
    return not torch.cuda.is_available()


def cuda_peak_mb() -> float | None:
    return round(torch.cuda.max_memory_reserved() / 2**20, 1) if torch.cuda.is_available() else None

def _data_physics(cfg: dict | None):
    """Physics record of the training data (first shard carrying provenance), or None (legacy data / no data)."""
    if not cfg or not cfg.get("data"):
        return None
    found = {}
    for body in cfg.get("bodies") or []:
        for sh in sorted((Path(cfg["data"]) / body).glob("s*.json")):
            if sh.name.endswith(".manifest.json"):
                continue
            try:
                ph = (json.loads(sh.read_text()).get("provenance") or {}).get("physics")
            except Exception:  # noqa: BLE001
                ph = None
            if ph:
                found[json.dumps(ph, sort_keys=True)] = ph
                break
        else:
            return None                         # a body without recorded physics: do not guess
    return next(iter(found.values())) if len(found) == 1 else None   # bodies differ (e.g. go2 impratio): None


def _stamp_specs(kw: dict):
    """The factor list a checkpoint is built on: explicit `specs`, else the config's own (a rep checkpoint's
    `latent.factors`, a BC checkpoint's `model.factors`), else None (a partial / non-net save: no stamp)."""
    if kw.get("specs") is not None:
        return tuple(kw["specs"])
    cfg = kw.get("cfg")
    if not cfg:
        return None
    if "E" in kw:
        return _legged_specs(cfg["latent"])
    if "model" in kw:
        return legged_specs((cfg.get("model") or {}).get("factors"))
    return None


def _save(path, **kw):
    """Legged checkpoint writer (bare dict, as before) plus `_provenance`: weights fingerprints of every state_dict
    entry (E/R/P/flow/model), git sha, flags and version IDs (incl. `versions["factors"]`, the relation-factor
    structure hash, stamped through `stamp_versions`); a `factors` entry (the resolved list, `to_json`) that makes the
    checkpoint self-describing; and a `<name>.json` sidecar with the file digest. `specs=` overrides the factor list."""
    from rrp.core.provenance import make_provenance, is_state_dict, file_digest
    path = Path(path)
    cfg = kw.get("cfg")
    res = kw.get("result") or {}
    kind = "bc" if "model" in kw and "E" not in kw else "learned"
    specs = _stamp_specs(kw)
    kw.pop("specs", None)
    if specs is not None:
        kw["factors"] = to_json(specs)
    versions = {k: res[k] for k in ("latent_space_version",) if k in res}
    if cfg:
        versions.update({k: str(cfg[k]) for k in ("data", "representation", "name") if k in cfg})
    versions = stamp_versions(versions, specs)
    prov = make_provenance(f"{kind}:{path.parent.name}/{path.name}", physics=_data_physics(cfg),
                           weights={k: v for k, v in kw.items() if is_state_dict(v)}, versions=versions,
                           flags=legged_flags(cfg))
    kw["_provenance"] = prov.to_dict()
    torch.save(kw, str(path) + ".tmp")
    digest = file_digest(Path(str(path) + ".tmp"))
    Path(str(path) + ".tmp").rename(path)
    path.with_suffix(".json").write_text(json.dumps(dict(path=str(path), sha256_16=digest, step=kw.get("step"),
                                                         provenance=prov.to_dict()), indent=1, default=str))


def train_rep(cfg, out: Path):
    _assert_sealed(cfg)
    dev = _dev()
    torch.manual_seed(cfg.get("seed", 0))
    rng = np.random.default_rng(cfg.get("seed", 0))
    data = LeggedData(Path(cfg["data"]), cfg["bodies"], dev)
    lc = cfg["latent"]
    specs = _legged_specs(lc)
    E, R, P = (m.to(dev) for m in build_legged_rep(lc, specs))
    params = [p for m in (E, R, P) for p in m.parameters()]
    steps, lr = cfg["steps"], cfg.get("lr", 3e-4)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    step0, last = 0, out / "rep_last.pt"
    if last.exists():                            # exact resume (D-040: the first run lost 9k steps)
        st = torch.load(str(last), map_location=dev, weights_only=False)
        E.load_state_dict(st["E"]); R.load_state_dict(st["R"]); P.load_state_dict(remap_legged_probe_state(st["P"]))
        opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
        step0 = st["step"]; exact = restore_rng(st, rng)
        print(f"resumed at step {step0} ({'exact: RNG restored' if exact else 'INEXACT: no CUDA RNG state in checkpoint'})", flush=True)
    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    B = cfg.get("batch_size", 256)
    ck, sn = cfg.get("ckpt_every", 500), cfg.get("snap_every", 0)
    for step in range(step0 + 1, steps + 1):
        i = data.sample(B, rng)
        j = torch.from_numpy(rng.integers(0, MAX_J + 1, B)).to(dev)
        loss, logs, _ = rep_step(E, R, P, data, i, j, specs, lc.get("beta_kl", 1e-3), qd_drop=lc.get("qd_dropout", 0.0))
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sch.step()
        if step % 200 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=float(loss.detach()), gn=float(gn), cuda_peak_mb=cuda_peak_mb(), **logs)) + "\n")
            log.flush()
        if step % ck == 0 and step < steps:
            _save(last, E=E.state_dict(), R=R.state_dict(), P=P.state_dict(), opt=opt.state_dict(),
                  sch=sch.state_dict(), step=step, **rng_state(rng))
        if sn and step % sn == 0 and step < steps:
            _save(out / f"snap_s{step}.pt", E=E.state_dict(), R=R.state_dict(), P=P.state_dict(), cfg=cfg,
                  result=dict(latent_space_version=f"legged-ls-{cfg['name']}", step=step))
    res = dict(steps=steps, wall_s=time.time() - t0, bodies=cfg["bodies"], n_rows=data.n,
               n_train_rows=len(data.train_idx), n_heldout_rows=len(data.test_idx),
               latent_space_version=f"legged-ls-{cfg['name']}", eval=eval_rep(E, R, P, data, specs))
    _save(out / "representation.pt", E=E.state_dict(), R=R.state_dict(), P=P.state_dict(), cfg=cfg, result=res)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def fit_probe(cfg, out: Path):
    """Post-hoc MEASUREMENT probe on detached frozen z (identical for sem / nosem); metadata-only control."""
    dev = _dev()
    rcfg, E, R, _, rres, specs = load_legged_rep(Path(cfg["representation"]), dev)
    _assert_sealed(rcfg)
    data = LeggedData(Path(rcfg["data"]), rcfg["bodies"], dev)
    res = {}
    # the measurement probe answers the five legacy packet queries (identical for sem / nosem); any further readout the
    # representation was trained with (`leg.com_support`) is measured with its own spec, at the spec's own weight
    probe_specs = tuple(s for s in specs if s.name.startswith("probe.legged.") or s.name == "leg.com_support")
    if not probe_specs:
        probe_specs = resolve(["preset:probes:legged-v1"], family="legged")
    unit_specs = _unit_probe_specs(probe_specs)
    for mode in ("z", "metadata_only"):
        P = legged_probe(dz=rcfg["latent"]["dz"], metadata_only=(mode == "metadata_only"), seed=5,
                         factors=probe_specs).to(dev)
        opt = torch.optim.AdamW(P.parameters(), lr=3e-4, weight_decay=1e-4)
        rng = np.random.default_rng(5)
        for step in range(cfg.get("steps", 6000)):
            i = data.sample(256, rng)
            with torch.no_grad():
                b = data.ctx_batch(i)
                mu, _ = E(b, data.beh(i))
            loss, _ = readout_loss(*_terms(legged_probe_read(P, mu, b["asm_mask"], b["body_asm"]), data.labels(i), b,
                                           unit_specs))
            opt.zero_grad(); loss.backward(); opt.step()
        P.eval()
        agg, sh = {}, {}
        rng2 = np.random.default_rng(1005)
        with torch.no_grad():
            for _ in range(40):
                i = data.sample(256, rng2, test=True)
                b = data.ctx_batch(i)
                mu, _ = E(b, data.beh(i))
                lab = data.labels(i)
                _agg(agg, _probe_metrics(legged_probe_read(P, mu, b["asm_mask"], b["body_asm"]), lab, b, probe_specs))
                _agg(sh, _probe_metrics(legged_probe_read(P, mu[torch.randperm(len(i), device=dev)], b["asm_mask"],
                                                          b["body_asm"]), lab, b, probe_specs))
        res[mode] = dict(heldout=_fin(agg), heldout_shuffled_z=_fin(sh))
        if mode == "z":
            out.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state=P.state_dict(), mode=mode), out / "probe_posthoc.pt")
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_posthoc.json").write_text(json.dumps(dict(representation=cfg["representation"], **res), indent=1))
    return res


def train_flow(cfg, out: Path):
    dev = _dev()
    torch.manual_seed(cfg.get("seed", 0))
    rng = np.random.default_rng(cfg.get("seed", 0))
    rcfg, E, R, P, rres, specs = load_legged_rep(Path(cfg["representation"]), dev)
    _assert_sealed(rcfg)
    data = LeggedData(Path(rcfg["data"]), rcfg["bodies"], dev)
    dz = rcfg["latent"]["dz"]
    F_ = LeggedFlow(dz=dz, D=cfg.get("width", 256), layers=cfg.get("layers", 4), factors=specs).to(dev)
    # target standardization over valid (knot, assembly) entries
    with torch.no_grad():
        zs = []
        for _ in range(20):
            i = data.sample(256, rng)
            b = data.ctx_batch(i)
            mu, _ = E(b, data.beh(i))
            zs.append(mu[b["asm_mask"][:, None, :].expand(-1, mu.shape[1], -1)])
        z = torch.cat(zs)
        F_.z_mean.copy_(z.mean(0)); F_.z_std.copy_(z.std(0).clamp(min=1e-3))
    steps, lr = cfg["steps"], cfg.get("lr", 3e-4)
    opt = torch.optim.AdamW(F_.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    unit = _unit_probe_specs(specs)          # the probe term is weighted once, by `w` below (not per query)
    # `packet_semantic_weight` stays a flat, top-level FLOW-stage params key ON DISK (distinct from `rcfg["latent"]`'s
    # own semantic weight/factors, which is stage A's -- this is stage B's own aux-loss weight through the FROZEN P):
    # sweep-flags follow-up (2026-09-30) moves the READ onto `nets/semantic_latent.py::packet_semantic_weight`, the
    # ONE conversion point shared with `harness/train/latent_train.py` / `harness/train/joint_adapt.py`, but does not
    # rewrite any on-disk `flow_*.json` (arm's OR legged's) to a different key -- that full architecture (`params.on`
    # letting the SAME `probe.<family>.*` spec carry a flow-path weight, docs/relations.md 10's R2 brief) needs
    # `catalog.py` / `nets/flow.py` / `nets/probes.py` wiring, out of this row's owned files. `joint_adapt.py`'s own
    # sibling read stays doubly permanent regardless (its own comment): it reads an ALREADY-TRAINED checkpoint's
    # saved config, the "reading an old pickle format forever" class.
    w = packet_semantic_weight(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    step0, last = 0, out / "flow_last.pt"
    if last.exists():
        st = torch.load(str(last), map_location=dev, weights_only=False)
        F_.load_state_dict(st["flow"]); opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
        step0 = st["step"]; exact = restore_rng(st, rng)
        print(f"resumed at step {step0} ({'exact: RNG restored' if exact else 'INEXACT: no CUDA RNG state in checkpoint'})", flush=True)
    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    B = cfg.get("batch_size", 256)
    ck, sn = cfg.get("ckpt_every", 500), cfg.get("snap_every", 0)
    for step in range(step0 + 1, steps + 1):
        if step > step0 + 1 and step % ck == 1 and step - 1 < steps:
            _save(last, flow=F_.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), step=step - 1,
                  specs=specs, **rng_state(rng))
        if sn and step > 1 and (step - 1) % sn == 0:
            _save(out / f"snap_s{step - 1}.pt", flow=F_.state_dict(), cfg=cfg, specs=specs,
                  result=dict(latent_space_version=rres["latent_space_version"], step=step - 1))
        i = data.sample(B, rng)
        b = data.train_batch(i)
        with torch.no_grad():
            zt, _ = E(b, data.beh(i))
        lab = data.labels(i)
        fn = (lambda zc: readout_loss(*_terms(legged_probe_read(P, zc, b["asm_mask"], b["body_asm"]), lab, b, unit))) \
            if w > 0 else None
        loss, logs = F_.loss(b, zt, fn, w, cfg.get("packet_tau_min", 0.6))
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(F_.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 200 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=float(loss.detach()), gn=float(gn), cuda_peak_mb=cuda_peak_mb(), **logs)) + "\n")
            log.flush()
    res = dict(steps=steps, wall_s=time.time() - t0, representation=cfg["representation"],
               latent_space_version=rres["latent_space_version"], eval=eval_flow(F_, E, R, P, data, specs))
    _save(out / "policy.pt", flow=F_.state_dict(), cfg=cfg, specs=specs, result=res)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


@torch.no_grad()
def eval_flow(F_, E, R, P, data, specs=(), n_batches=30, seed=7, nfe=8):
    """Held-out episodes: packet probes on oracle E targets vs FREE samples (public context only) vs shuffled;
    realization error of R driven by free samples vs by oracle targets (open-loop, recorded states)."""
    F_.eval()
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=data.dev).manual_seed(seed)
    aggs = {k: {} for k in ("oracle", "free", "free_shuffled")}
    real = {"oracle": [], "free": [], "zero": []}
    for _ in range(n_batches):
        i = data.sample(256, rng, test=True)
        b = data.ctx_batch(i)
        zt, _ = E(b, data.beh(i))
        zf = F_.sample(b, nfe=nfe, generator=g)
        lab = data.labels(i)
        for name, z in (("oracle", zt), ("free", zf), ("free_shuffled", zf[torch.randperm(len(i), device=data.dev)])):
            _agg(aggs[name], _probe_metrics(legged_probe_read(P, z, b["asm_mask"], b["body_asm"]), lab, b, specs))
        j = torch.from_numpy(rng.integers(0, MAX_J + 1, len(i))).to(data.dev)
        br, ph, a1, am = data.realizer_batch(i, j)
        m = am.float()
        for name, z in (("oracle", zt), ("free", zf)):
            real[name].append(float((((R(z, br, ph) - a1) ** 2) * m).sum() / m.sum()))
        real["zero"].append(float(((a1 ** 2) * m).sum() / m.sum()))
    F_.train()
    return dict(split="held_out_episodes", realize_mse={k: float(np.mean(v)) for k, v in real.items()},
                **{k: _fin(v) for k, v in aggs.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rep", "flow", "probe"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override (smoke runs)")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    if a.steps:
        cfg["steps"] = a.steps
    fn = dict(rep=train_rep, flow=train_flow, probe=fit_probe)[a.stage]
    res = fn(cfg, Path(a.out))
    print(json.dumps(res, indent=1, default=str)[:4000])
