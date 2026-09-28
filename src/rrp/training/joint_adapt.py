"""Joint system-i flow + system-0 adaptation on target demos (armdiag, D-135 addendum / D-136).

Added AFTER D-135: the sealed protocol's single-module adaptations (flow SFT / system-0 refit) each leave the other
module unadapted on a new arm. This adapts BOTH on the same budgeted demo episodes (nested_budget_indices, as flow SFT,
refit and BC SFT) with a TOTAL optimizer-update count `steps` (matched to BC SFT, which updates its whole policy per
step). The encoder E and packet probes P stay frozen, so the latent space (and its semantics) is unchanged.

mode:
  split  steps//2 flow-SFT updates (as sft_latent_flow), then steps - steps//2 realizer updates (as refit_realizer)
  joint  `steps` updates of ONE AdamW over flow + realizer parameters (sum of the two losses; per-module grad clip 1.0)
gen_frac: fraction of each realizer batch whose packet is the CURRENT flow's free sample (NFE `nfe`, detached) at the
  demo state instead of E's posterior sample (generator-consistent realizer training, like the source gen-DAgger refits
  but on the demo states only). 0 = E packets only.
Outputs in out_dir: policy.pt (adapted flow) and representation.pt (E, adapted R, P bundle), result.json.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from rrp.controllers.bundles import load_representation
from rrp.controllers.latent_realizer import make_realizer
from rrp.data.latent import LatentData
from rrp.models.checkpoint import load_checkpoint, save_checkpoint
from rrp.models.flow import FlowPolicy, PolicyConfig
from rrp.models.latent_probes import probe_loss
from rrp.models.semantic_latent import assembly_tokens

JOINT_ADAPT_VERSION = "joint_adapt_v1"


def split_steps(steps: int, mode: str) -> tuple[int, int]:
    """(flow updates, realizer updates) as counted against the total budget; joint mode shares every update."""
    if mode == "split":
        return steps // 2, steps - steps // 2
    if mode == "joint":
        return steps, steps
    raise ValueError(f"mode {mode!r} (split|joint)")


def joint_adapt(flow_ckpt: Path, rep_path: Path, packed_dir: Path, budget: int, *, seed: int, out_dir: Path, steps: int,
                mode: str = "joint", gen_frac: float = 0.0, lr: float = 1e-4, batch_size: int = 128, nfe: int = 8) -> dict:
    from rrp.evaluation.adaptation import nested_budget_indices
    from rrp.models.latent_batch import assembly_batch
    from rrp.training.latent_train import _bundle, _dev
    dev = _dev()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    st = load_checkpoint(Path(flow_ckpt), map_location=dev)
    cfgj = st["config"]
    lcfg, E, _, P, _ = load_representation(Path(cfgj["representation"]), dev)
    pcfg = PolicyConfig(**dict(cfgj["policy"], horizon=lcfg.knots, latent_dim=lcfg.dz, aux=False))
    flow = FlowPolicy(pcfg).to(dev)
    flow.load_state_dict(st["model"])
    st0 = load_checkpoint(Path(rep_path), map_location=dev)
    lcfg_r, E_r, R_old, P_r, rep_res = load_representation(Path(rep_path), dev)
    if not all(torch.equal(x, y) for x, y in zip(E.state_dict().values(), E_r.state_dict().values())):
        raise ValueError("flow and representation bundle use different encoders (latent spaces)")
    R = make_realizer(lcfg.dz, lcfg.realizer_layers, st0["config"].get("realizer_arch")).to(dev)
    R.load_state_dict(R_old.state_dict())
    R.anchor = bool(st0["config"].get("realizer_anchor", False))
    R.drop_qd = bool(st0["config"].get("realizer_drop_qd", False))
    for p_ in R.parameters():
        p_.requires_grad_(True)
    data = LatentData(Path(packed_dir), zero_prev_action=bool(st0["config"].get("zero_prev_action", False)),
                      anchor=R.anchor, drop_qd=R.drop_qd)
    eps = sorted(set(data.ep.tolist()))
    chosen = [eps[i] for i in nested_budget_indices(len(eps), [budget], seed)[budget]]
    pool = [int(i) for i in np.nonzero(np.isin(data.ep, chosen))[0]]
    w_sem = cfgj.get("packet_semantic_weight", 0.0)
    nf_steps, nr_steps = split_steps(steps, mode)
    kt = torch.tensor(lcfg.knot_times, device=dev)
    B = min(batch_size, max(8, len(pool)))
    if mode == "joint":
        opt = torch.optim.AdamW([{"params": flow.parameters()}, {"params": R.parameters()}], lr=lr, weight_decay=1e-4)
        opt_f = opt_r = opt
    else:
        opt_f = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-4)
        opt_r = torch.optim.AdamW(R.parameters(), lr=lr, weight_decay=1e-4)
    log = []
    t0 = time.time()
    flow.train(); R.train()

    def flow_loss(batch, ab, am, zt, lab):
        smask = batch.bank_mask["scene"] & lab["slot_valid"].bool()
        S = smask.shape[1]
        fn = (lambda zc: probe_loss(P(zc, am, S), lab, smask, lv_min=lcfg.probe_lv_min)) if w_sem > 0 else None
        loss, _ = flow.loss(ab, zt, am[:, None, :].expand(-1, lcfg.knots, -1), None, packet_loss_fn=fn,
                            packet_weight=w_sem)
        return loss

    def real_loss(ab, am, mu, logvar, j, r):
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        ng = int(round(gen_frac * z.shape[0]))
        if ng > 0:
            flow.eval()
            with torch.no_grad():
                g = torch.Generator(device=dev).manual_seed(rng.randrange(1 << 30))
                zg = flow.sample(flow.prepare(ab), lcfg.knots, nfe=nfe, generator=g)
            flow.train()
            z = torch.cat([zg[:ng], z[ng:]], 0)
        phase = torch.as_tensor(j * lcfg.control_dt, dtype=z.dtype, device=dev)
        pred = R(z, am, kt, phase, r["node"], r["node_mask"], r["local"], node_asm=r.get("node_asm"))
        m = (r["v1"] & r["node_mask"]).float()
        return ((pred - r["a1"]) ** 2 * m).sum() / m.sum().clamp(min=1)

    total = nf_steps if mode == "joint" else nf_steps + nr_steps
    for step in range(total):
        do_f = mode == "joint" or step < nf_steps
        do_r = mode == "joint" or step >= nf_steps
        sel, tgt, j = data.sample(B, rng, lcfg.max_phase_ticks if do_r else 0, pool)
        batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        with torch.no_grad():
            af, am, ai = assembly_tokens(batch)
            mu, logvar = E(batch, a, v, af, am, ai)
        ab = assembly_batch(batch)
        lf = flow_loss(batch, ab, am, mu, lab) if do_f else None
        lr_ = real_loss(ab, am, mu, logvar, torch.as_tensor(j, device=dev), r) if do_r else None
        loss = sum(x for x in (lf, lr_) if x is not None)
        (opt_f if do_f else opt_r).zero_grad()
        if mode == "joint":
            opt.zero_grad()
        loss.backward()
        if do_f:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        if do_r:
            torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
        if mode == "joint":
            opt.step()
        elif do_f:
            opt_f.step()
        else:
            opt_r.step()
        if (step + 1) % 50 == 0:
            log.append(dict(step=step + 1, flow=None if lf is None else float(lf.detach()),
                            real=None if lr_ is None else float(lr_.detach())))
    res = dict(version=JOINT_ADAPT_VERSION, mode=mode, gen_frac=gen_frac, budget=budget, seed=seed,
               demo_episodes=len(chosen), demo_control_transitions=len(pool), optimizer_updates=total,
               flow_updates=nf_steps, realizer_updates=nr_steps, lr=lr, batch_size=B, nfe=nfe, wall_s=time.time() - t0,
               changed_modules="system_i_flow+system0_realizer", frozen=["target_encoder", "packet_probes"],
               latent_space_version=rep_res["latent_space_version"], source_flow=str(flow_ckpt),
               source_representation=str(rep_path), log=log)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flow.eval(); R.eval()
    save_checkpoint(out_dir / "policy.pt", model=flow, optimizer=None, step=total,
                    versions=dict(st["versions"], adapted=True), config=cfgj,
                    extra=dict(result=dict(st["extra"]["result"], joint_adapt=res)))
    from rrp.controllers.latent_realizer import bundle_versions
    rres = dict(rep_res, joint_adapt={k: v for k, v in res.items() if k != "log"},
                realizer_compat_version=bundle_versions(lcfg.version(), E.state_dict(), R.state_dict())[1])
    save_checkpoint(out_dir / "representation.pt", model=_bundle(E, R, P), optimizer=None, step=total,
                    versions=dict(latent=rep_res["latent_space_version"]),
                    config=dict(st0["config"], joint_adapt={k: v for k, v in res.items() if k != "log"}),
                    extra=dict(result=rres))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True)
    ap.add_argument("--rep", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--mode", choices=["split", "joint"], default="joint")
    ap.add_argument("--gen-frac", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if torch.cuda.is_available():
        from rrp.contracts.workload import apply_cap
        apply_cap()
    res = joint_adapt(Path(a.flow), Path(a.rep), Path(a.pack), a.budget, seed=a.seed, out_dir=Path(a.out),
                      steps=a.steps, mode=a.mode, gen_frac=a.gen_frac)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, default=str))
    print("last_log", res["log"][-1] if res["log"] else None)


if __name__ == "__main__":
    main()
