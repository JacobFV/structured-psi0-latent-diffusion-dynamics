"""The adapting trainers of the humanoid transfer matrix (HA, research/tracks/humanoid.md section 4).

Level 2 (new controller training) adapts a source-trained system to a target body from a NESTED target-demo pack (`pack` stage:
the first N demos of the body, N in {5, 20, 100}); Level 1 (existing-controller transfer) fine-tunes / retrains a tracker with PPO on
an env-sample budget. Every trainer here has ONE acquisition record (demos, teacher ticks, env samples, updates) that the transfer
driver compares against the pack's; the update count is the pack's matched one (150 / 300 / 600), checked by the pipeline stage.

| trainer | starts from | moves | writes |
|---|---|---|---|
| `adapt_refit` | a source representation (E, R, P) | system 0 (R) only: E and P stay, so the latent space is unchanged | `representation.pt` (a full rep: the source's E / P, the refit R) |
| `adapt_flow` | a source flow (system i) and a representation | the flow only (z statistics kept); E, R, P frozen | `policy.pt` (flow) over the given representation |
| `adapt_bc` | a source BC policy | the whole policy (positive control, SFT warm start) | `policy.pt` (model) |
| `adapt_ppo_plan` / `ppo_acquisition` | a tracker actor (fine-tune) or nothing (scratch) | PPO through `rrp train tracker-warp` | the argv, the sample budget check and `acquisition.json` |

Every demo of the pack trains (no held-out episodes: the pack IS the adaptation data, and the evaluation is the closed-loop cell
on the evaluation scenes), so `LeggedData` is built with a hold-out period no seed reaches and the offline eval is the training
loss at the end of the run (`train_*` keys), never presented as a held-out number. Sealed bodies train only on target-adaptation
seeds (SealedSplit). Sources: refit / flow are `learned`, BC is `bc`, the PPO actor is `learned_tracker`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ADAPT_STAGES = ("adapt_refit", "adapt_flow", "adapt_bc", "adapt_ppo")
NO_HOLDOUT = 10 ** 9            # LeggedData holds out episodes with seed % N == N - 1; no adaptation seed reaches it
PPO_MODES = ("finetune", "scratch")


def load_pack_data(cfg: dict, dev):
    """LeggedData over the pack directory of cfg (`data`, `bodies`) with EVERY episode in the training set; the sealed guard
    runs first (a sealed body is read only on its target-adaptation seeds)."""
    from rrp.core.sealed import SealedSplit
    from rrp.harness.train.legged_latent_train import LeggedData
    SealedSplit.load().assert_dataset_allowed(cfg["data"], cfg["bodies"])
    data = LeggedData(Path(cfg["data"]), list(cfg["bodies"]), dev, holdout_every=NO_HOLDOUT)
    if len(data.test_idx) or len(data.train_idx) != data.n:
        raise ValueError(f"{cfg['data']}: an adaptation pack trains on every demo (held out rows: {len(data.test_idx)})")
    return data


def _opt(params, steps: int, lr: float):
    import torch
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    return opt, torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)


def _tail(xs: list, k: int = 20):
    return float(np.mean(xs[-k:])) if xs else None


def _head(xs: list, k: int = 20):
    return float(np.mean(xs[:k])) if xs else None


def _adapt_cfg(cfg: dict, src: dict) -> dict:
    """The checkpoint config of an adapted run: the source's config with this run's data / name / steps."""
    out = dict(src)
    out.update({k: cfg[k] for k in ("name", "data", "bodies", "steps", "seed", "lr", "batch_size") if k in cfg})
    out["adapt"] = dict(cfg.get("adapt") or {})
    return out


# ---------------------------------------------------------------------------------------------------- system 0 refit
def adapt_refit(cfg: dict, out: Path) -> dict:
    """System-0 refit: R of the source representation on the pack's demos, through the stage-A realizer loss with the frozen E
    encoding the demonstrated chunks (E and P do not move, so flows trained on the source latent space still read this one).
    cfg: representation, data, bodies, steps, [lr 1e-4, batch_size, seed, qd_dropout 0.5, name, adapt]."""
    import torch
    from rrp.harness.train.legged_latent_train import MAX_J, REALIZER_GROUPS, _save, load_legged_rep, rep_step
    from rrp.policies.bundles import _dev
    dev = _dev()
    torch.manual_seed(cfg.get("seed", 0))
    rng = np.random.default_rng(cfg.get("seed", 0))
    rcfg, E, R, P, rres, specs = load_legged_rep(Path(cfg["representation"]), dev)
    data = load_pack_data(cfg, dev)
    R.train()
    for p in R.parameters():
        p.requires_grad_(True)
    steps, B = int(cfg["steps"]), cfg.get("batch_size", 256)
    opt, sch = _opt(R.parameters(), steps, cfg.get("lr", 1e-4))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "train_log.jsonl", "w")
    real, t0 = [], time.time()
    for step in range(1, steps + 1):
        i = data.sample(B, rng)
        j = torch.from_numpy(rng.integers(0, MAX_J + 1, B)).to(dev)
        loss, logs, _ = rep_step(E, R, P, data, i, j, (), 0.0, train=True, qd_drop=cfg.get("qd_dropout", 0.5))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
        opt.step(); sch.step()
        real.append(logs["real"])
        if step % 50 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=float(loss.detach()), real=logs["real"])) + "\n"); log.flush()
    R.eval()
    upper = bool(rres.get("upper_trained")) or data.upper_trained
    res = dict(steps=steps, wall_s=time.time() - t0, bodies=list(cfg["bodies"]), n_rows=data.n, n_train_rows=len(data.train_idx),
               n_heldout_rows=0, latent_space_version=rres["latent_space_version"], upper_trained=upper,
               action_groups=[g for g in REALIZER_GROUPS if g == "legs" or upper], moved="R (system 0) only; E and P frozen",
               train_realize_mse_first=_head(real), train_realize_mse_last=_tail(real),
               eval="none: every pack demo trains; the closed-loop cell is the evaluation")
    _save(out / "representation.pt", E=E.state_dict(), R=R.state_dict(), P=P.state_dict(),
          cfg=dict(_adapt_cfg(cfg, rcfg), latent=rcfg["latent"], representation=cfg["representation"]), result=res)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------------------------------------------- flow warm start
def adapt_flow(cfg: dict, out: Path) -> dict:
    """System i warm start: the source flow (`init`) trained further on the pack against the frozen E of `representation` (the
    source or the refit representation: E is identical), packet-semantic term through the frozen P as in stage B.
    cfg: init, representation, data, bodies, steps, [lr 1e-4, batch_size, seed, packet_semantic_weight, packet_tau_min, name, adapt]."""
    import torch
    from rrp.harness.train.legged_latent_train import _save, _terms, _unit_probe_specs, legged_probe_read, load_legged_rep, readout_loss
    from rrp.policies.bundles import _dev
    from rrp.policies.legged import load_legged_flow
    from rrp.policies.nets.semantic_latent import packet_semantic_weight
    dev = _dev()
    torch.manual_seed(cfg.get("seed", 0))
    rng = np.random.default_rng(cfg.get("seed", 0))
    rcfg, E, R, P, rres, specs = load_legged_rep(Path(cfg["representation"]), dev)
    F_, src = load_legged_flow(Path(cfg["init"]), dev, specs, rcfg["latent"]["dz"])
    data = load_pack_data(cfg, dev)
    F_.train()
    for p in F_.parameters():
        p.requires_grad_(True)
    steps, B = int(cfg["steps"]), cfg.get("batch_size", 256)
    opt, sch = _opt(F_.parameters(), steps, cfg.get("lr", 1e-4))
    unit = _unit_probe_specs(specs)
    w = packet_semantic_weight({**src["cfg"], **cfg})
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "train_log.jsonl", "w")
    losses, t0 = [], time.time()
    for step in range(1, steps + 1):
        i = data.sample(B, rng)
        b = data.train_batch(i)
        with torch.no_grad():
            zt, _ = E(b, data.beh(i))
        lab = data.labels(i)
        fn = (lambda zc: readout_loss(*_terms(legged_probe_read(P, zc, b["asm_mask"], b["body_asm"]), lab, b, unit))) if w > 0 else None
        loss, logs = F_.loss(b, zt, fn, w, cfg.get("packet_tau_min", src["cfg"].get("packet_tau_min", 0.6)))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(F_.parameters(), 1.0)
        opt.step(); sch.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=losses[-1], **logs)) + "\n"); log.flush()
    F_.eval()
    res = dict(steps=steps, wall_s=time.time() - t0, representation=cfg["representation"], init=cfg["init"],
               latent_space_version=rres["latent_space_version"], upper_trained=rres.get("upper_trained", False),
               action_groups=rres.get("action_groups"), n_rows=data.n, n_heldout_rows=0, moved="flow (system i) only",
               train_loss_first=_head(losses), train_loss_last=_tail(losses),
               eval="none: every pack demo trains; the closed-loop cell is the evaluation")
    fcfg = dict(_adapt_cfg(cfg, src["cfg"]), representation=cfg["representation"], init=cfg["init"])
    _save(out / "policy.pt", flow=F_.state_dict(), cfg=fcfg, specs=specs, result=res)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------------------------------------------- BC SFT warm start
def adapt_bc(cfg: dict, out: Path) -> dict:
    """BC positive control: the source BC policy (`init`) fine-tuned on the pack (whole policy; source `bc`).
    cfg: init, data, bodies, steps, [lr 1e-4, batch_size, seed, name, adapt]."""
    import torch
    from rrp.harness.train.legged_bc import bc_batch
    from rrp.harness.train.legged_latent_train import _save
    from rrp.policies.bundles import _dev
    from rrp.policies.legged import load_legged_bc
    dev = _dev()
    torch.manual_seed(cfg.get("seed", 0))
    rng = np.random.default_rng(cfg.get("seed", 0))
    model, src = load_legged_bc(Path(cfg["init"]), dev)
    data = load_pack_data(cfg, dev)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    steps, B = int(cfg["steps"]), cfg.get("batch_size", 256)
    opt, sch = _opt(model.parameters(), steps, cfg.get("lr", 1e-4))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "train_log.jsonl", "w")
    losses, t0 = [], time.time()
    for step in range(1, steps + 1):
        i = data.sample(B, rng)
        b, a, am = bc_batch(data, i)
        loss = model.loss(b, a, am)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=losses[-1])) + "\n"); log.flush()
    model.eval()
    res = dict(steps=steps, wall_s=time.time() - t0, bodies=list(cfg["bodies"]), init=cfg["init"], n_rows=data.n, n_heldout_rows=0,
               upper_trained=data.upper_trained or bool((src.get("result") or {}).get("upper_trained")),
               action_groups=data.action_groups, moved="whole BC policy (SFT warm start)",
               train_loss_first=_head(losses), train_loss_last=_tail(losses),
               eval="none: every pack demo trains; the closed-loop cell is the evaluation",
               source="learned (behaviour cloning of scripted_teacher -> frozen tracker targets; POSITIVE CONTROL, SFT warm start)")
    _save(out / "policy.pt", model=model.state_dict(), cfg=_adapt_cfg(cfg, src["cfg"]), step=steps, result=res)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------------------------------------------- Level 1: PPO
def adapt_ppo_plan(adapt: dict, args: dict, *, out: str, init: str | None = None, resume: bool = True) -> dict:
    """The `rrp train tracker-warp` argv of a Level-1 run and its accounting, WITHOUT any simulation.

    adapt: {task, body, budget (env samples), mode: finetune | scratch}; args: extra trainer options ({dest: value}, True = bare
    flag; a `recipe` key is the trainer's `--recipe`). The budget is reproduced exactly by the trainer: samples = iters * horizon *
    nworld (one env step of every world per row), so `iters` is DERIVED from the budget and a budget that is not a multiple of
    horizon * nworld is refused (an unmatched budget is not the same acquisition). `finetune` warm-starts from `init` (an exported
    actor.pt); `scratch` has none. A run over `groups` (shared tracker) is not a per-body budget and is refused. The sealed guard runs here too
    (`assert_train_allowed` with the trainer's env seed: a sealed body trains only on a target-adaptation seed)."""
    from rrp.harness.train.warp_tracker_ppo import build_args
    mode, body, budget = adapt.get("mode"), adapt["body"], int(adapt["budget"])
    if mode not in PPO_MODES:
        raise ValueError(f"adapt_ppo: mode {mode!r} (finetune | scratch)")
    if mode == "finetune" and not init:
        raise ValueError("adapt_ppo finetune needs an `init` actor (an exported actor.pt)")
    if mode == "scratch" and init:
        raise ValueError("adapt_ppo scratch starts from no actor (drop `init`)")
    args = dict(args)
    for k in ("iters", "body", "out", "init_shared", "resume"):            # `groups`: one group of the adapted body only (checked below)
        if k in args:
            raise ValueError(f"adapt_ppo: option {k!r} is set by the stage (iters from the budget), not by options.args")
    argv = ["--body", body, "--out", out]
    if args.get("recipe"):
        argv += ["--recipe", str(args.pop("recipe"))]
    for k, v in args.items():
        flag = "--" + str(k).replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v not in (False, None):
            argv += [flag, str(v)]
    if init:
        argv += ["--init-shared", str(init)]
    eff = build_args(argv)                                    # recipe + flags merged: the trainer's own effective values
    nworld = int(eff.nworld)
    if eff.groups:
        # D-147 (2026-10-03): a shared (morph) tracker fine-tunes on ONE body as exactly one group of exactly that body; any other
        # groups run is not a per-body budget
        gr = json.loads(eff.groups) if isinstance(eff.groups, str) else eff.groups
        if not (isinstance(gr, list) and len(gr) == 1 and list(gr[0][0]) == [body]):
            raise ValueError(f"adapt_ppo: groups {gr} is not a per-body budget; use exactly one group [[[{body!r}], nworld]]")
        nworld = int(gr[0][1])
    from rrp.core.sealed import SealedSplit
    SealedSplit.load().assert_train_allowed([body], [int(eff.seed)], what=f"adapt_ppo {adapt.get('task')}/{mode}")
    per_iter = nworld * int(eff.horizon)
    if budget <= 0 or budget % per_iter:
        raise ValueError(f"adapt_ppo: budget {budget} env samples is not a multiple of horizon*nworld = {eff.horizon}*{eff.nworld} = "
                         f"{per_iter} (one PPO iteration); choose nworld / horizon that divide it")
    iters = budget // per_iter
    argv += ["--iters", str(iters)] + (["--resume"] if resume else [])
    return dict(argv=argv, iters=iters, nworld=nworld, horizon=int(eff.horizon), samples_per_iter=per_iter,
                task=adapt.get("task"), body=body, budget=budget, mode=mode, train_task=eff.task, seed=int(eff.seed))


def ppo_acquisition(plan: dict) -> dict:
    """The acquisition record of a Level-1 run: the env samples it consumed (units of the `samples` level), no demos, no teacher
    ticks, no offline updates. Checked by the driver against `{env_samples: budget}`."""
    return dict(task=plan["task"], body=plan["body"], budget=plan["budget"], demos=0, teacher_ticks=0,
                env_samples=plan["iters"] * plan["samples_per_iter"], updates=0, ppo_iters=plan["iters"], nworld=plan["nworld"],
                horizon=plan["horizon"], mode=plan["mode"])
