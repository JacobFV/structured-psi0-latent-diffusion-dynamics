"""ComputerWorld pointer policies: split, teacher-demo collection, training and packet diagnostics (track pointer;
research/tracks/cworld.md "pointer policy"). `rrp train pointer <cmd> ...`:

    split    write the seed lists of research/splits/cworld_pointer_v1.json (held-out variants are declared there)
    collect  scripted-teacher demos (DART pointer noise on move ticks for a fraction of episodes) -> .npz (peer store)
    rep      E (encoder) + R (learned system 0) + P (packet probe) on demo chunks; --variant semfix | nosem
    flow     system i (rectified flow) on the frozen representation's posterior means (--target latent), or on the
             engineered encoding (--target eng, for the SCRIPTED engineered system 0)
    bc       BC baseline with the same public inputs and demos
    probe    post-hoc packet probes (and the metadata-only control) on frozen packets: UI probes for semfix vs nosem
    edit     causal packet edits: probe-guided retargeting of a received packet, realized by system 0 in closed loop
    video    labelled demo video of episodes of any pointer policy (peer: rendering)

Sources: demos are `scripted_teacher` (privileged labels: teacher target widget, destination, phase); every trained
model reads only rrp.policies.pointer.public_features. Weights and datasets stay in the peer store (never committed).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

SPLIT_PATH = "research/splits/cworld_pointer_v1.json"
TASKS = ("cw/calc_sum", "cw/open_type", "cw/drag_window", "cw/fill_form")


# ------------------------------------------------------------------------------------------------ split
def load_split(path: str = SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text())


def heldout_goal(task: str, goal: dict, split: dict) -> bool:
    h = split["heldout_variants"]
    if task == "cw/calc_sum":
        return [goal["a"], goal["b"]] in h["calc_pairs"]
    if task == "cw/open_type":
        return goal["text"] in h["words"]
    if task == "cw/fill_form":
        return goal["name"] in h["names"]
    return False


def excluded_seeds(split: dict, task: str) -> set[int]:
    s = split["seeds"][task]
    return set(s["dev"]) | set(s["sealed_id"]) | set(s.get("sealed_heldout", []))


def cmd_split(a):
    """Fill the seed lists: per task the first N seeds from each declared range whose goal is in/out of the held-out
    variants (goals are sampled at env reset, so this needs the ComputerWorld wheel)."""
    from rrp.envs.base import make_env
    split = load_split(a.split)
    rng = split["seed_ranges"]
    split["seeds"] = {}
    for task in TASKS:
        env = make_env("computerworld", task=task, body="cw_pointer", seed=0)
        out = {}
        for key, start, n, want_heldout in (("dev", rng["dev"][0], a.n_dev, False),
                                            ("sealed_id", rng["sealed_id"][0], a.n_sealed, False),
                                            ("sealed_heldout", rng["sealed_heldout"][0], a.n_heldout, True)):
            if want_heldout and task == "cw/drag_window":
                continue
            got, s = [], start
            while len(got) < n:
                env.reset(s)
                env._initial.clear()
                if heldout_goal(task, env.goal, split) == want_heldout:
                    got.append(s)
                s += 1
            out[key] = got
        split["seeds"][task] = out
        env.close()
    Path(a.split).write_text(json.dumps(split, indent=1) + "\n")
    print(json.dumps({t: {k: len(v) for k, v in d.items()} for t, d in split["seeds"].items()}))


# ------------------------------------------------------------------------------------------------ collect
def phase_of(groups: dict, prev_btn: bool, prev_xy) -> int:
    """PHASES index of a teacher tick: type, press, release, drag, move, idle."""
    if "key" in groups and groups["key"][0] >= 0:
        return 5
    b = groups["button"][0] >= 0.5
    if b and not prev_btn:
        return 2
    if prev_btn and not b:
        return 3
    moved = abs(groups["pointer"][0] - prev_xy[0]) + abs(groups["pointer"][1] - prev_xy[1]) > 1e-9
    if b:
        return 4
    return 1 if moved else 0


def collect_episode(task: str, seed: int, *, dart_px: float, rng: random.Random, max_ticks: int = 400) -> dict | None:
    """One scripted-teacher episode. Labels are the teacher's clean commands; executed pointer commands get
    N(0, dart_px) noise on intermediate move ticks (DART; the teacher's goto corrects from wherever the pointer is; the
    arriving tick is never perturbed, else the goto would never terminate)."""
    from rrp.core.action import NativeCommand
    from rrp.envs.base import make_env
    from rrp.policies.pointer import EventHistory, public_features, screen_half
    from rrp.policies.teachers.computerworld import CWTeacher
    from rrp.tasks.spec import get_task
    env = make_env("computerworld", task=task, body="cw_pointer", seed=seed)
    T = get_task(task)
    half = screen_half(env.spec)
    tt = CWTeacher(env, task)
    hist = EventHistory()
    rows, tabs, tab_keys = [], [], {}
    obs = env.observe()
    ok = False
    for tick in range(max_ticks):
        f = public_features(obs, half, hist, tick)
        c = tt.act()
        if c is None:
            break
        g = {k: list(v) for k, v in c.groups.items()}
        q = obs.measured_node_state.qpos[:2]
        ph = phase_of(g, hist.button, q)
        ex = {k: list(v) for k, v in g.items()}
        if dart_px > 0 and ph == 1 and tt.target_px is not None and \
                env.frame.m_to_px(*g["pointer"]) != tuple(tt.target_px):     # intermediate move ticks only
            ex["pointer"] = [g["pointer"][0] + rng.gauss(0, dart_px) * env.frame.m_per_px,
                             g["pointer"][1] + rng.gauss(0, dart_px) * env.frame.m_per_px]
        wkey = (f["wch"].tobytes(), f["wf"].tobytes(), f["wmask"].tobytes(), f["wbound"].tobytes())
        if wkey not in tab_keys:
            tab_keys[wkey] = len(tabs)
            tabs.append({k: f[k] for k in ("wch", "wrole", "wbound", "wf", "wmask")})
        tslot = env.slots.slots.get(tt.target, -1) if tt.target else -1
        tpx = tt.target_px
        txy = (np.array(env.frame.px_to_m(*tpx)) / half) if tpx is not None else np.array([np.nan, np.nan])
        key = int(round(g.get("key", [-1])[0]))
        rows.append(dict(tab=tab_keys[wkey], ptr=f["ptr"], btn=f["btn"], hist=f["hist"],
                         cmd_xy=np.array(g["pointer"]) / half, cmd_btn=float(g["button"][0] >= 0.5), cmd_key=key + 1,
                         slot=tslot if tslot < 80 else -1, txy=txy, phase=ph))
        hist.push(tick, ex, half)
        st = env.step(NativeCommand(controller_version="cw_pointer.v1", groups=ex, source="scripted_teacher"))
        obs = st.observation
        j = T.judge(env, st.time, T.max_seconds)
        if j.done:
            ok = j.outcome == "success"
            break
    goal, instr = dict(env.goal), obs.instruction
    env.close()
    if not ok:
        return None
    return dict(rows=rows, tabs=tabs, goal=goal, instr=instr)


def cmd_collect(a):
    from rrp.policies.pointer import codes, LI
    split = load_split(a.split)
    lo, hi = split["seed_ranges"]["train"]
    excl = excluded_seeds(split, a.task)
    rng = random.Random(a.seed)
    eps, s, skipped, failed = [], a.start, 0, 0
    t0 = time.time()
    from rrp.envs.base import make_env
    probe = make_env("computerworld", task=a.task, body="cw_pointer", seed=0)
    while len(eps) < a.episodes:
        assert lo <= s < hi, "ran out of training seeds"
        probe.reset(s)
        probe._initial.clear()                    # the env caches one snapshot per seed; do not grow it
        if s in excl or heldout_goal(a.task, probe.goal, split):
            skipped += 1
            s += 1
            continue
        ep = collect_episode(a.task, s, dart_px=a.dart_px if rng.random() < a.dart_frac else 0.0, rng=rng)
        if ep is None:
            failed += 1
        else:
            ep["seed"] = s
            eps.append(ep)
        s += 1
    probe.close()
    # flatten: tables, ticks, episodes
    tabs, ticks, ep_rows = [], [], []
    for e_i, ep in enumerate(eps):
        base = len(tabs)
        tabs += ep["tabs"]
        t_start = len(ticks)
        for r in ep["rows"]:
            ticks.append(dict(r, tab=r["tab"] + base, ep=e_i))
        ep_rows.append((ep["seed"], t_start, len(ticks), codes(ep["instr"], LI), json.dumps(ep["goal"])))
    out = dict(
        wch=np.stack([t["wch"] for t in tabs]), wrole=np.stack([t["wrole"] for t in tabs]),
        wbound=np.stack([t["wbound"] for t in tabs]), wf=np.stack([t["wf"] for t in tabs]).astype(np.float16),
        wmask=np.stack([t["wmask"] for t in tabs]),
        tab=np.array([t["tab"] for t in ticks], np.int32), ep=np.array([t["ep"] for t in ticks], np.int32),
        ptr=np.stack([t["ptr"] for t in ticks]).astype(np.float32), btn=np.array([t["btn"] for t in ticks], np.float32),
        hist=np.stack([t["hist"] for t in ticks]).astype(np.float32),
        cmd_xy=np.stack([t["cmd_xy"] for t in ticks]).astype(np.float32),
        cmd_btn=np.array([t["cmd_btn"] for t in ticks], np.float32), cmd_key=np.array([t["cmd_key"] for t in ticks],
                                                                                     np.int16),
        slot=np.array([t["slot"] for t in ticks], np.int16), txy=np.stack([t["txy"] for t in ticks]).astype(np.float32),
        phase=np.array([t["phase"] for t in ticks], np.int8),
        ep_seed=np.array([e[0] for e in ep_rows], np.int64), ep_start=np.array([e[1] for e in ep_rows], np.int64),
        ep_end=np.array([e[2] for e in ep_rows], np.int64), ep_instr=np.stack([e[3] for e in ep_rows]),
        ep_goal=np.array([e[4] for e in ep_rows]), task=np.array(a.task))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    meta = dict(task=a.task, episodes=len(eps), ticks=len(ticks), tables=len(tabs), seeds=[int(eps[0]["seed"]),
                int(eps[-1]["seed"])], skipped_heldout_or_eval=skipped, teacher_failures=failed, dart_px=a.dart_px,
                dart_frac=a.dart_frac, source="scripted_teacher", wall_s=round(time.time() - t0, 1))
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


# ------------------------------------------------------------------------------------------------ dataset
class Demos:
    """All collected ticks on one device. sample t -> public features at t, the 7-tick demo chunk t..t+6 (masked at the
    episode end) and probe labels at the knots' late ticks (j = 0, 2, 4, 6)."""
    H = 7
    LATE = (0, 2, 4, 6)

    def __init__(self, files: list[str], device, val_frac: float = 0.05, seed: int = 0):
        import torch
        parts = [dict(np.load(f, allow_pickle=False)) for f in files]
        cat = {}
        tab_off, tick_off, ep_off = 0, 0, 0
        for p in parts:
            p = dict(p)
            p["tab"] = p["tab"] + tab_off
            p["ep"] = p["ep"] + ep_off
            p["ep_start"] = p["ep_start"] + tick_off
            p["ep_end"] = p["ep_end"] + tick_off
            p["ep_task"] = np.full(len(p["ep_seed"]), TASKS.index(str(p["task"])), np.int8)
            tab_off += len(p["wch"])
            tick_off += len(p["tab"])
            ep_off += len(p["ep_seed"])
            for k, v in p.items():
                if k == "task":
                    continue
                cat.setdefault(k, []).append(v)
        d = {k: np.concatenate(v) for k, v in cat.items()}
        self.goals = list(d.pop("ep_goal"))
        self.ep_seed, self.ep_task = d["ep_seed"], d["ep_task"]
        N = len(d["tab"])
        # chunk indices (t + j within the episode, else -1)
        ep_end = d["ep_end"][d["ep"]]
        j = np.arange(self.H)[None]
        idx = np.arange(N)[:, None] + j
        idx = np.where(idx < ep_end[:, None], idx, -1)
        rng = np.random.default_rng(seed)
        n_ep = len(d["ep_seed"])
        val_ep = np.zeros(n_ep, bool)
        val_ep[rng.permutation(n_ep)[:max(1, int(val_frac * n_ep))]] = True
        self.val_mask = val_ep[d["ep"]]
        T = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x), device=device, dtype=dt)
        self.t = dict(wch=T(d["wch"].astype(np.int64)), wrole=T(d["wrole"].astype(np.int64)),
                      wbound=T(d["wbound"].astype(np.int64)), wf=T(d["wf"].astype(np.float32)), wmask=T(d["wmask"]),
                      tab=T(d["tab"].astype(np.int64)), ep=T(d["ep"].astype(np.int64)), ptr=T(d["ptr"]), btn=T(d["btn"]),
                      hist=T(d["hist"]), cmd_xy=T(d["cmd_xy"]), cmd_btn=T(d["cmd_btn"]),
                      cmd_key=T(d["cmd_key"].astype(np.int64)), slot=T(d["slot"].astype(np.int64)),
                      txy=T(np.nan_to_num(d["txy"], nan=0.0)), txy_ok=T(~np.isnan(d["txy"][:, 0])),
                      phase=T(d["phase"].astype(np.int64)), instr=T(d["ep_instr"].astype(np.int64)),
                      tick=T((np.arange(N) - d["ep_start"][d["ep"]]).astype(np.float32)), chunk=T(idx.astype(np.int64)))
        self.N, self.device = N, device
        self.train_idx = T(np.nonzero(~self.val_mask)[0].astype(np.int64))
        tick_task = self.ep_task[d["ep"]][~self.val_mask]
        cnt = np.bincount(tick_task, minlength=len(TASKS)).astype(np.float64)
        self.train_w = T((1.0 / cnt[tick_task]).astype(np.float32))       # every task equally likely per sample
        self.val_idx = T(rng.permutation(np.nonzero(self.val_mask)[0]).astype(np.int64))   # every task in any prefix
        self.half = None

    def sample(self, n: int):
        """Task-balanced training sample indices."""
        return self.train_idx[torch.multinomial(self.train_w, n, replacement=True)]

    def batch(self, ix):
        """-> (public batch b, demo chunk a, probe labels lab) for sample indices ix (a 1-D long tensor)."""
        import torch
        t = self.t
        tab = t["tab"][ix]
        b = dict(wch=t["wch"][tab], wrole=t["wrole"][tab], wbound=t["wbound"][tab], wf=t["wf"][tab], wmask=t["wmask"][tab],
                 instr=t["instr"][t["ep"][ix]], ptr=t["ptr"][ix], btn=t["btn"][ix], tick=t["tick"][ix],
                 hist=t["hist"][ix])
        ci = t["chunk"][ix]                                        # [B,H]
        valid = ci >= 0
        cc = ci.clamp(min=0)
        from rrp.policies.pointer import STEP_M
        # normalized screen units -> pointer steps (60 px): the realizer's output unit; half extents from wf scale
        half = self.half
        dxy = (t["cmd_xy"][cc] - t["ptr"][cc]) * half / STEP_M
        a = dict(dxy=dxy * valid[..., None], xy=t["cmd_xy"][cc] * valid[..., None], btn=t["cmd_btn"][cc] * valid,
                 key=t["cmd_key"][cc] * valid, valid=valid, ptr=t["ptr"][cc], pbtn=t["btn"][cc])
        late = ci[:, list(self.LATE)]
        lv = late >= 0
        lc = late.clamp(min=0)
        lab = dict(slot=torch.where(lv, t["slot"][lc], -1),
                   rel=(t["txy"][lc] - t["ptr"][ix][:, None]), rel_ok=lv & t["txy_ok"][lc],
                   phase=torch.where(lv, t["phase"][lc], -1))
        return b, a, lab


def probe_loss(out, lab, lv_min: float):
    """Probe objectives on the packet: target slot CE, relative target Gaussian NLL (log-variance bounded below by
    lv_min), phase CE; ignored where the label is missing (-1 / not ok)."""
    import torch.nn.functional as F
    out = {k: v.float() for k, v in out.items()}
    B, K, Nw = out["slot"].shape
    sl = lab["slot"].reshape(-1)
    L_slot = F.cross_entropy(out["slot"].reshape(-1, Nw), sl.clamp(min=0), reduction="none")
    m = (sl >= 0).float()
    L_slot = (L_slot * m).sum() / m.sum().clamp(min=1)
    mu, lv = out["rel"][..., :2], out["rel"][..., 2:].clamp(lv_min, 6)
    nll = 0.5 * (((lab["rel"] - mu) ** 2) / lv.exp() + lv + math.log(2 * math.pi)).sum(-1)
    ro = lab["rel_ok"].float()
    L_rel = (nll * ro).sum() / ro.sum().clamp(min=1)
    ph = lab["phase"].reshape(-1)
    L_ph = F.cross_entropy(out["phase"].reshape(-1, out["phase"].shape[-1]), ph.clamp(min=0), reduction="none")
    pm = (ph >= 0).float()
    L_ph = (L_ph * pm).sum() / pm.sum().clamp(min=1)
    L = L_slot + L_rel + L_ph
    return L, dict(slot=float(L_slot.detach()), rel=float(L_rel.detach()), phase=float(L_ph.detach()))


@torch.no_grad()
def probe_metrics(out, lab) -> dict:
    """(sum, count) pairs: slot top-1 accuracy, relative-position error (normalized units -> px), phase accuracy."""
    sl, ph = lab["slot"], lab["phase"]
    ms, mp, mr = sl >= 0, ph >= 0, lab["rel_ok"]
    px = torch.tensor([480.0, 320.0], device=lab["rel"].device)       # normalized screen units -> px (960 x 640)
    err = ((out["rel"][..., :2] - lab["rel"]) * px).norm(dim=-1)
    return dict(slot_acc=(int(((out["slot"].argmax(-1) == sl) & ms).sum()), int(ms.sum())),
                phase_acc=(int(((out["phase"].argmax(-1) == ph) & mp).sum()), int(mp.sum())),
                rel_err=(float((err * mr).sum()), int(mr.sum())))


def action_loss(xy_pred_steps, bl, kl, a):
    """Tick losses in pointer-step units (smooth L1), button BCE, key CE; masked by a['valid']."""
    import torch.nn.functional as F
    xy_pred_steps, bl, kl = xy_pred_steps.float(), bl.float(), kl.float()
    v = a["valid"].float()
    n = v.sum().clamp(min=1)
    Lxy = (F.smooth_l1_loss(xy_pred_steps, a["dxy_target"], beta=0.05, reduction="none").sum(-1) * v).sum() / n
    Lb = (F.binary_cross_entropy_with_logits(bl, a["btn"], reduction="none") * v).sum() / n
    Lk = (F.cross_entropy(kl.reshape(-1, kl.shape[-1]), a["key"].reshape(-1), reduction="none") * v.reshape(-1)).sum() / n
    return Lxy, Lb, Lk


@torch.no_grad()
def action_metrics(xy_pred_steps, bl, kl, a) -> dict:
    from rrp.policies.pointer import MAX_STEP_PX
    v = a["valid"]
    e = (xy_pred_steps - a["dxy_target"]).norm(dim=-1) * MAX_STEP_PX
    return dict(xy_px=(float((e * v).sum()), int(v.sum())),
                btn_acc=(int((((bl > 0).float() == a["btn"]) & v).sum()), int(v.sum())),
                key_acc=(int(((kl.argmax(-1) == a["key"]) & v).sum()), int(v.sum())),
                keypress_acc=(int(((kl.argmax(-1) == a["key"]) & v & (a["key"] > 0)).sum()), int((v & (a["key"] > 0)).sum())))


def _agg(acc: dict, m: dict):
    for k, (s, c) in m.items():
        s0, c0 = acc.get(k, (0.0, 0))
        acc[k] = (s0 + s, c0 + c)


def _fin(acc: dict) -> dict:
    return {k: (s / c if c else None) for k, (s, c) in acc.items()}


def check_no_leak(data: "Demos", split: dict) -> None:
    """Split guard: every training episode comes from the train seed range, is not a dev/sealed seed and does not use
    a held-out variant."""
    lo, hi = split["seed_ranges"]["train"]
    for seed, ti, g in zip(data.ep_seed, data.ep_task, data.goals):
        task = TASKS[int(ti)]
        if not lo <= int(seed) < hi or int(seed) in excluded_seeds(split, task) or heldout_goal(task, json.loads(str(g)),
                                                                                                split):
            raise RuntimeError(f"split leak: {task} seed {seed} goal {g}")


# ------------------------------------------------------------------------------------------------ training helpers
def _setup(a):
    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Demos(sorted(a.data), dev, seed=a.seed)
    check_no_leak(data, load_split(a.split))
    import rrp.envs.computerworld as cwm
    fr = cwm.ScreenFrame()
    data.half = torch.tensor([fr.width * fr.m_per_px / 2, fr.height * fr.m_per_px / 2], device=dev)
    return dev, data


def _amp(dev):
    """bf16 autocast on CUDA (losses are computed in fp32)."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda"))


def _sched(opt, step, total, lr, warm=500):
    f = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
    for g in opt.param_groups:
        g["lr"] = lr * max(f, 0.02)


def _save(path, *, kind, state: dict, config: dict, versions: dict, metrics: dict):
    import torch
    from rrp.core.provenance import weights_digest
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob = dict(kind=kind, state={k: m.state_dict() for k, m in state.items()}, config=config, versions=versions,
                digests={k: weights_digest(m.state_dict()) for k, m in state.items()}, metrics=metrics,
                saved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    torch.save(blob, path)
    Path(path).with_suffix(".json").write_text(json.dumps({k: v for k, v in blob.items() if k != "state"}, indent=1,
                                                          default=str))


def _chunk_targets(a, half):
    """Realizer targets: dxy in steps relative to the pointer measured at each tick (already in a['dxy'])."""
    a["dxy_target"] = a["dxy"]
    return a


def _rep_forward(E, R, a_b, a, dev, sample=True):
    import torch
    mu, lv = E(a_b, a)
    z = mu + (0.5 * lv).exp() * torch.randn_like(mu) if sample else mu
    B, H = a["valid"].shape
    dt = 0.1
    zz = z[:, None].expand(B, H, *z.shape[1:]).reshape(B * H, *z.shape[1:])
    ph = (torch.arange(H, device=dev, dtype=torch.float32) * dt)[None].expand(B, H).reshape(-1)
    dxy, bl, kl = R(zz, ph, a["ptr"].reshape(-1, 2), a["pbtn"].reshape(-1))
    return mu, lv, z, dxy.reshape(B, H, 2), bl.reshape(B, H), kl.reshape(B, H, -1)


def cmd_rep(a):
    import torch
    from rrp.policies.pointer import nets
    from rrp.policies.system0 import bundle_versions
    dev, data = _setup(a)
    N = nets()
    arch = dict(E=dict(dz=a.dz), R=dict(dz=a.dz), P=dict(dz=a.dz))
    E, R, P = N["PointerEncoder"](**arch["E"]).to(dev), N["PointerRealizer"](**arch["R"]).to(dev), \
        N["PointerProbe"](**arch["P"]).to(dev)
    params = list(E.parameters()) + list(R.parameters()) + list(P.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    w_sem = a.w_sem if a.variant == "semfix" else 0.0
    log = []
    t0 = time.time()
    for step in range(a.steps):
        _sched(opt, step, a.steps, a.lr)
        ix = data.sample(a.batch)
        b, ch, lab = data.batch(ix)
        ch = _chunk_targets(ch, data.half)
        with _amp(dev):
            mu, lv, z, dxy, bl, kl = _rep_forward(E, R, b, ch, dev)
            po = P(z, b) if w_sem > 0 else None
        mu, lv = mu.float(), lv.float()
        Lxy, Lb, Lk = action_loss(dxy, bl, kl, ch)
        kl_div = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).mean()
        loss = a.w_xy * Lxy + Lb + Lk + a.beta * kl_div
        logs = dict(xy=float(Lxy.detach()), btn=float(Lb.detach()), key=float(Lk.detach()), kl=float(kl_div.detach()))
        if w_sem > 0:
            pl, pl_logs = probe_loss(po, lab, a.lv_min)
            loss = loss + w_sem * pl
            logs.update({f"p_{k}": v for k, v in pl_logs.items()})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            ev = _eval_rep(E, R, P if w_sem > 0 else None, data, dev)
            log.append(dict(step=step, wall=round(time.time() - t0, 1), **logs, **{f"val_{k}": v for k, v in ev.items()}))
            print(json.dumps(log[-1]), flush=True)
    lsv, rcv = bundle_versions(f"cw_pointer_latent.v1-{a.variant}-dz{a.dz}", E.state_dict(), R.state_dict())
    cfg = dict(variant=a.variant, arch=arch, w_sem=w_sem, lv_min=a.lv_min, beta=a.beta, w_xy=a.w_xy, steps=a.steps,
               batch=a.batch, lr=a.lr, seed=a.seed, data=sorted(a.data), tasks=list(TASKS), target="latent")
    _save(a.out, kind="pointer_rep", state=dict(E=E, R=R, P=P), config=cfg,
          versions=dict(latent_space_version=lsv, realizer_compat_version=rcv), metrics=log[-1])


@torch.no_grad()
def _eval_rep(E, R, P, data, dev, n=4096) -> dict:
    import torch
    E.eval(); R.eval()
    acc = {}
    vi = data.val_idx[:n]
    for s in range(0, len(vi), 1024):
        b, ch, lab = data.batch(vi[s:s + 1024])
        ch = _chunk_targets(ch, data.half)
        mu, lv, z, dxy, bl, kl = _rep_forward(E, R, b, ch, dev, sample=False)
        _agg(acc, action_metrics(dxy, bl, kl, ch))
        if P is not None:
            P.eval()
            _agg(acc, probe_metrics(P(mu, b), lab))
            P.train()
    E.train(); R.train()
    return _fin(acc)


def _frozen_mu(E, data, dev, bs=2048):
    """Posterior means of the frozen encoder for every sample: [N, K, 1, dz]."""
    import torch
    out = []
    E.eval()
    with torch.no_grad():
        for s in range(0, data.N, bs):
            ix = torch.arange(s, min(s + bs, data.N), device=dev)
            b, ch, _ = data.batch(ix)
            out.append(E(b, ch)[0])
    return torch.cat(out)


def eng_targets(data, dev, bs=4096):
    """Engineered packets (rrp.policies.pointer encoding) of every sample's demo chunk: [N, K, 1, 14]."""
    import torch
    from rrp.policies.pointer import ENG_DIM, SLOT_W, packet_ticks, N_KEYCLS
    ticks = packet_ticks(0.1)
    out = torch.zeros(data.N, 4, 1, ENG_DIM, device=dev)
    for s in range(0, data.N, bs):
        ix = torch.arange(s, min(s + bs, data.N), device=dev)
        _, ch, _ = data.batch(ix)
        for j, (k, sl) in enumerate(ticks):
            o = sl * SLOT_W
            v = ch["valid"][:, j].float()
            xy = ch["xy"][:, j] * data.half
            out[ix, k, 0, o:o + SLOT_W] = torch.stack([xy[:, 0], xy[:, 1], torch.zeros_like(v),
                                                       ch["btn"][:, j] * 2 - 1, ch["key"][:, j].float() / (N_KEYCLS - 1),
                                                       torch.zeros_like(v), torch.ones_like(v)], -1) * v[:, None]
    return out


def cmd_flow(a):
    import torch
    from rrp.policies.pointer import load_pointer_bundle, nets, ENG_DIM, ENG_VERSION
    dev, data = _setup(a)
    N = nets()
    if a.target == "eng":
        Z = eng_targets(data, dev)
        dz, versions, P, variant = ENG_DIM, dict(latent_space_version=ENG_VERSION, realizer_compat_version=ENG_VERSION), \
            None, "eng"
    else:
        rb = load_pointer_bundle(a.representation, dev)
        Z = _frozen_mu(rb["modules"]["E"], data, dev)
        dz, versions, variant = Z.shape[-1], dict(rb["versions"]), rb["config"]["variant"]
        P = rb["modules"]["P"] if rb["config"]["w_sem"] > 0 else None
        if P is not None:
            for p in P.parameters():
                p.requires_grad_(False)
    arch = dict(S=dict(dz=dz))
    S = N["PointerFlow"](**arch["S"]).to(dev)
    tr = Z[data.train_idx]
    S.z_mean.copy_(tr.reshape(-1, dz).mean(0))
    S.z_std.copy_(tr.reshape(-1, dz).std(0).clamp(min=1e-3))
    opt = torch.optim.AdamW(S.parameters(), lr=a.lr, weight_decay=1e-4)
    w_sem = a.w_sem if P is not None else 0.0
    log, t0 = [], time.time()
    for step in range(a.steps):
        _sched(opt, step, a.steps, a.lr)
        ix = data.sample(a.batch)
        b, _, lab = data.batch(ix)
        with _amp(dev):
            loss, logs = S.loss(b, Z[ix], probe_fn=(lambda zc: probe_loss(P(zc, b), lab, rb["config"]["lv_min"]))
                            if P is not None else None, w_sem=w_sem)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0)
        opt.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            ev = _eval_flow(S, Z, data, dev, rb["modules"]["R"] if a.target != "eng" else None)
            log.append(dict(step=step, wall=round(time.time() - t0, 1), **logs, **{f"val_{k}": v for k, v in ev.items()}))
            print(json.dumps(log[-1]), flush=True)
    cfg = dict(variant=variant, target=a.target, representation=a.representation, arch=arch, w_sem=w_sem,
               steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed, data=sorted(a.data), tasks=list(TASKS))
    _save(a.out, kind="pointer_flow", state=dict(S=S), config=cfg, versions=versions, metrics=log[-1])


@torch.no_grad()
def _eval_flow(S, Z, data, dev, R=None, n=2048) -> dict:
    """Validation: generated-packet error vs the target packet (standardized units) and, with R, the realized
    first-tick action from the generated packet vs the demo."""
    S.eval()
    acc = {}
    vi = data.val_idx[:n]
    for s in range(0, len(vi), 1024):
        ix = vi[s:s + 1024]
        b, ch, _ = data.batch(ix)
        zh = S.sample(b, nfe=8)
        e = (((zh - Z[ix]) / S.z_std) ** 2).mean(dim=(1, 2, 3))
        acc["z_mse_std"] = (acc.get("z_mse_std", (0, 0))[0] + float(e.sum()), acc.get("z_mse_std", (0, 0))[1] + len(e))
        if R is not None:
            ch = _chunk_targets(ch, data.half)
            import torch
            B, H = ch["valid"].shape
            zz = zh[:, None].expand(B, H, *zh.shape[1:]).reshape(B * H, *zh.shape[1:])
            ph = (torch.arange(H, device=dev, dtype=torch.float32) * 0.1)[None].expand(B, H).reshape(-1)
            dxy, bl, kl = R(zz, ph, ch["ptr"].reshape(-1, 2), ch["pbtn"].reshape(-1))
            _agg(acc, {f"gen_{k}": v for k, v in action_metrics(dxy.reshape(B, H, 2), bl.reshape(B, H),
                                                                kl.reshape(B, H, -1), ch).items()})
    S.train()
    return _fin(acc)


def cmd_bc(a):
    import torch
    from rrp.policies.pointer import nets, STEP_M
    dev, data = _setup(a)
    N = nets()
    arch = dict(BC=dict())
    BC = N["PointerBC"]().to(dev)
    opt = torch.optim.AdamW(BC.parameters(), lr=a.lr, weight_decay=1e-4)
    log, t0 = [], time.time()
    for step in range(a.steps):
        _sched(opt, step, a.steps, a.lr)
        ix = data.sample(a.batch)
        b, ch, _ = data.batch(ix)
        with _amp(dev):
            xy, bl, kl = BC(b)
        ch["dxy_target"] = ch["xy"] * data.half / STEP_M            # absolute position, in pointer-step units
        Lxy, Lb, Lk = action_loss(xy.float() * data.half / STEP_M, bl, kl, ch)
        loss = a.w_xy * Lxy + Lb + Lk
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(BC.parameters(), 1.0)
        opt.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            BC.eval()
            acc = {}
            with torch.no_grad():
                vi = data.val_idx[:4096]
                for s in range(0, len(vi), 1024):
                    b, ch, _ = data.batch(vi[s:s + 1024])
                    xy, bl, kl = BC(b)
                    ch["dxy_target"] = ch["xy"] * data.half / STEP_M
                    _agg(acc, action_metrics(xy * data.half / STEP_M, bl, kl, ch))
            BC.train()
            log.append(dict(step=step, wall=round(time.time() - t0, 1), xy=float(Lxy.detach()), btn=float(Lb.detach()), key=float(Lk.detach()),
                            **{f"val_{k}": v for k, v in _fin(acc).items()}))
            print(json.dumps(log[-1]), flush=True)
    cfg = dict(variant="bc", arch=arch, steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed, data=sorted(a.data),
               tasks=list(TASKS), w_xy=a.w_xy)
    _save(a.out, kind="pointer_bc", state=dict(BC=BC), config=cfg, versions=dict(bc="cw_pointer_bc.v1"),
          metrics=log[-1])


def cmd_probe(a):
    """Post-hoc probes on frozen packets (E posterior means, or packets generated by a flow): the same probe recipe
    for every representation, plus the metadata-only control (no z). Reports held-out-episode metrics."""
    import torch
    from rrp.policies.pointer import load_pointer_bundle, nets
    dev, data = _setup(a)
    N = nets()
    rb = load_pointer_bundle(a.representation, dev)
    Z = _frozen_mu(rb["modules"]["E"], data, dev)
    if a.flow:
        fb = load_pointer_bundle(a.flow, dev)
        S = fb["modules"]["S"]
        with torch.no_grad():
            for s in range(0, data.N, 2048):
                ix = torch.arange(s, min(s + 2048, data.N), device=dev)
                Z[ix] = S.sample(data.batch(ix)[0], nfe=8)
    res = {}
    for name, meta in (("probe", False), ("metadata_only", True)):
        torch.manual_seed(a.seed)
        P = N["PointerProbe"](dz=Z.shape[-1], metadata_only=meta).to(dev)
        opt = torch.optim.AdamW(P.parameters(), lr=a.lr, weight_decay=1e-4)
        for step in range(a.steps):
            _sched(opt, step, a.steps, a.lr)
            ix = data.sample(a.batch)
            b, _, lab = data.batch(ix)
            with _amp(dev):
                po = P(Z[ix], b)
            loss, _ = probe_loss(po, lab, -8.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        P.eval()
        acc = {}
        with torch.no_grad():
            for s in range(0, len(data.val_idx), 2048):
                ix = data.val_idx[s:s + 2048]
                b, _, lab = data.batch(ix)
                _agg(acc, probe_metrics(P(Z[ix], b), lab))
        r = _fin(acc)
        r["rel_err_px"] = r.pop("rel_err")
        res[name] = r
        print(name, json.dumps(r), flush=True)
    out = dict(representation=a.representation, flow=a.flow, variant=rb["config"]["variant"], steps=a.steps,
               val_samples=int(len(data.val_idx)), **res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------ causal edits
def cmd_edit(a):
    """Probe-guided retargeting (causal use of the packet). At the first packet of an episode (pointer at its start,
    target = the teacher's first widget), z is edited by gradient steps on a post-hoc probe's target-slot logit toward
    another visible widget, with an L2 anchor to the original z. System 0 then realizes ONLY the edited packet (no
    replanning) for 7 ticks; we measure whether the pointer ends inside the new target's box vs the original's. The
    control applies a random edit of the same norm. The probe is trained on frozen E means (cmd_probe recipe)."""
    import torch
    from rrp.envs.base import make_env
    from rrp.envs.computerworld import scene_widgets
    from rrp.policies.pointer import (EventHistory, LearnedSystem0, PointerSystemI, collate_public, load_pointer_bundle,
                                      nets, pointer_packet, public_features, screen_half)
    dev, data = _setup(a)
    N = nets()
    rb, fb = load_pointer_bundle(a.representation, dev), load_pointer_bundle(a.flow, dev)
    Z = _frozen_mu(rb["modules"]["E"], data, dev)
    torch.manual_seed(a.seed)
    P = N["PointerProbe"](dz=Z.shape[-1]).to(dev)
    opt = torch.optim.AdamW(P.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(a.probe_steps):
        _sched(opt, step, a.probe_steps, 1e-3)
        ix = data.sample(512)
        b, _, lab = data.batch(ix)
        loss, _ = probe_loss(P(Z[ix], b), lab, -8.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    P.eval()
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
                env = make_env("computerworld", task=task, body="cw_pointer", seed=seed)
                si.reset([env])
                obs = env.observe()
                half = screen_half(env.spec)
                f = public_features(obs, half, EventHistory(), 0)
                b = collate_public([f], dev)
                p0 = si.packets([env])[0]
                z0 = torch.from_numpy(np.asarray(p0.z, np.float32))[None].to(dev)
                with torch.no_grad():
                    orig = int(P(z0, b)["slot"][0, -1].argmax())
                ws = [w for w in scene_widgets(env.scene()) if w["visible"] and w["box"] and w["role"] in
                      ("button", "textbox")]
                slots = {env.slots.slots[w["key"]]: w for w in ws if env.slots.slots.get(w["key"], 99) < 80}
                cand = [s for s in slots if s != orig and s in slots]
                if orig not in slots or not cand:
                    env.close()
                    continue
                new = random.Random(seed * 7 + 1).choice(cand)
                z = z0.clone()
                if mode != "none":
                    zv = z0.clone().requires_grad_(True)
                    o2 = torch.optim.Adam([zv], lr=a.edit_lr)
                    for _ in range(a.edit_steps):
                        s_ = P(zv, b)["slot"][0]                          # [K, NW]
                        L = torch.nn.functional.cross_entropy(s_, torch.full((s_.shape[0],), new, device=dev)) \
                            + a.anchor * ((zv - z0) ** 2).mean()
                        o2.zero_grad()
                        L.backward()
                        o2.step()
                    delta = (zv.detach() - z0)
                    if mode == "random":
                        r = torch.randn_like(delta)
                        delta = r / r.norm() * delta.norm()
                    z = z0 + delta
                with torch.no_grad():
                    pr = P(z, b)["slot"][0, -1].argmax().item()
                pk = pointer_packet(env, obs, z[0].cpu().numpy(), lsv=lsv, rcv=rcv, source="learned",
                                    name=f"edit:{mode}")
                s0 = LearnedSystem0(R, env, latent_space_version=lsv, realizer_compat_version=rcv, device=dev)
                s0.receive(pk, now=float(env.time), graph_version=0)
                ctr = lambda w: np.array([(w["box"][0] + w["box"][2]) / 2, (w["box"][1] + w["box"][3]) / 2])
                p0 = np.array([env.pointer.u, env.pointer.v], float)
                for _ in range(7):
                    c = s0.tick(env)
                    env.step(c)
                u, v = env.pointer.u, env.pointer.v
                p1 = np.array([u, v], float)
                inside = lambda w: w["box"][0] <= u < w["box"][2] and w["box"][1] <= v < w["box"][3]
                rows.append(dict(task=task, seed=seed, mode=mode, orig_slot=orig, new_slot=new, probe_after=pr,
                                 ptr_px=[u, v], at_new=inside(slots[new]), at_orig=inside(slots[orig]),
                                 d_new=[float(np.linalg.norm(p0 - ctr(slots[new]))), float(np.linalg.norm(p1 - ctr(slots[new])))],
                                 d_orig=[float(np.linalg.norm(p0 - ctr(slots[orig]))),
                                         float(np.linalg.norm(p1 - ctr(slots[orig])))],
                                 z_delta=float((z - z0).norm())))
                env.close()
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


# ------------------------------------------------------------------------------------------------ video
class FrameHook:
    """Rollout hook: the ComputerWorld screen (env.render, the privileged renderer; frames are for humans only) at reset
    and after every tick, downscaled, with a caption bar naming the controller source."""

    def __init__(self, caption: str, scale: float = 0.5):
        self.caption, self.scale, self.frames = caption, scale, {}

    def _grab(self, i, env, note=""):
        from PIL import Image, ImageDraw
        im = Image.fromarray(env.render()[..., :3])
        im = im.resize((int(im.width * self.scale), int(im.height * self.scale)))
        bar = Image.new("RGB", (im.width, 44), (20, 20, 20))
        d = ImageDraw.Draw(bar)
        d.text((6, 4), self.caption, fill=(255, 255, 255))
        d.text((6, 24), f"{env.task_name} seed {env.seed}  t={env.time:.1f}s  {note}", fill=(255, 220, 120))
        out = Image.new("RGB", (im.width, im.height + 44))
        out.paste(bar, (0, 0))
        out.paste(im, (0, 44))
        self.frames.setdefault(i, []).append(np.asarray(out))

    def on_reset(self, i, env, obs):
        self._grab(i, env)

    def on_step(self, i, env, act, step):
        self._grab(i, env)

    def on_end(self, i, env, ep):
        tag = ep.outcome + (f" ({ep.failure_reason})" if ep.failure_reason else "")
        self._grab(i, env, tag.upper())
        self.frames[i] += [self.frames[i][-1]] * 10         # hold the final frame 1 s
        return {}


def cmd_video(a):
    """Render labelled episodes of one policy (JSON spec as for rrp eval) into one mp4 (peer only: rendering)."""
    import imageio
    from rrp.harness.eval.evaluate import evaluate
    from rrp.policies.base import make_policy
    name, _, kw = a.policy.partition("=")
    pol = make_policy(name, **(json.loads(kw) if kw else {}))
    frames, rows = [], []
    for task, seed in (x.split("@") for x in a.episodes):
        h = FrameHook(f"{pol.info.source.upper()}  {pol.info.name}  {pol.info.variant or ''}  {a.caption}")
        ep = evaluate(pol, "computerworld", task, "cw_pointer", [int(seed)], batch=1, hooks=[h])[0]
        frames += h.frames[0]
        rows.append(dict(task=task, seed=int(seed), outcome=ep.outcome, failure_reason=ep.failure_reason,
                         steps=ep.steps))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=10, quality=6)
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(policy=a.policy, source=pol.info.source,
                                                                episodes=rows), indent=1))
    print(json.dumps(rows))


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="rrp train pointer", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, steps=20000, lr=3e-4):
        p.add_argument("--data", nargs="+", required=True, help="collected .npz files (one per task)")
        p.add_argument("--out", required=True)
        p.add_argument("--steps", type=int, default=steps)
        p.add_argument("--batch", type=int, default=512)
        p.add_argument("--lr", type=float, default=lr)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default=None)
        p.add_argument("--log-every", type=int, default=1000)
        p.add_argument("--split", default=SPLIT_PATH)

    p = sub.add_parser("split")
    p.add_argument("--split", default=SPLIT_PATH)
    p.add_argument("--n-dev", type=int, default=50)
    p.add_argument("--n-sealed", type=int, default=100)
    p.add_argument("--n-heldout", type=int, default=50)
    p.set_defaults(fn=cmd_split)
    p = sub.add_parser("collect")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--start", type=int, default=0, help="first training seed tried")
    p.add_argument("--dart-frac", type=float, default=0.5)
    p.add_argument("--dart-px", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default=SPLIT_PATH)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("rep")
    common(p)
    p.add_argument("--variant", required=True, choices=("semfix", "nosem"))
    p.add_argument("--dz", type=int, default=16)
    p.add_argument("--w-sem", type=float, default=1.0)
    p.add_argument("--lv-min", type=float, default=-4.0, help="semfix: bounded probe NLL (as the arm's semfix)")
    p.add_argument("--beta", type=float, default=3e-3)
    p.add_argument("--w-xy", type=float, default=5.0)
    p.set_defaults(fn=cmd_rep)
    p = sub.add_parser("flow")
    common(p, steps=30000)
    p.add_argument("--target", choices=("latent", "eng"), default="latent")
    p.add_argument("--representation", help="rep checkpoint (target latent)")
    p.add_argument("--w-sem", type=float, default=0.5)
    p.set_defaults(fn=cmd_flow)
    p = sub.add_parser("bc")
    common(p, steps=50000)
    p.add_argument("--w-xy", type=float, default=5.0)
    p.set_defaults(fn=cmd_bc)
    p = sub.add_parser("probe")
    common(p, steps=8000, lr=1e-3)
    p.add_argument("--representation", required=True)
    p.add_argument("--flow", help="probe packets generated by this flow instead of E means")
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("edit")
    common(p, steps=0)
    p.add_argument("--representation", required=True)
    p.add_argument("--flow", required=True)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--probe-steps", type=int, default=6000)
    p.add_argument("--edit-steps", type=int, default=60)
    p.add_argument("--edit-lr", type=float, default=0.05)
    p.add_argument("--anchor", type=float, default=0.1)
    p.set_defaults(fn=cmd_edit)
    p = sub.add_parser("video")
    p.add_argument("--policy", required=True, help="NAME or NAME=JSON kwargs (as rrp eval)")
    p.add_argument("--episodes", nargs="+", required=True, help="task@seed ...")
    p.add_argument("--caption", default="")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_video)
    a = ap.parse_args(argv)
    return a.fn(a)
