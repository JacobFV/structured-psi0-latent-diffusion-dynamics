"""armdiag (D-135 follow-up): OFFLINE decomposition of the latent route's failure to adapt to a new arm.
NON-SEALED DIAGNOSTIC: reads only the target DEMO pack (the adaptation data, seed 1,000,000+; allowed) and source packs;
never touches the sealed eval scenes. Run on the peer (torch; GPU optional).

For one lineage (flow_src, rep_src) and its target adaptations (flow_sft, rep_refit) it reports, on the demo pack rows
of the episodes NOT chosen by the adaptation budget (held-out) and on the chosen ones (train):
  * realizer error with z from each source {E posterior mean (oracle), flow_src sample, flow_sft sample} x realizer
    {R_src, R_refit}: normalized 1-step action MSE, arm-joint MSE, and EE (TCP) 1-step displacement error through the
    public Jacobian columns of the node features (cm, cosine vs the demo's displacement);
  * generator error: ||z_gen - E mu||^2 / ||E mu||^2, and packet-probe rel_pos(cube - TCP) error from z_gen vs label;
  * node-feature ranges of the target body vs the source pack (static morphology columns and dynamic columns).
Output: one JSON (--out).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from rrp.controllers.bundles import load_representation
from rrp.controllers.latent_realizer import make_realizer
from rrp.data.latent import LatentData
from rrp.evaluation.adaptation import nested_budget_indices
from rrp.models.checkpoint import load_checkpoint
from rrp.models.flow import FlowPolicy, PolicyConfig
from rrp.models.latent_batch import assembly_batch
from rrp.models.latent_probes import probe_metrics
from rrp.models.semantic_latent import assembly_tokens

JP = slice(35, 38)          # node feature columns: TCP linear Jacobian column of the joint (base frame, m/rad)
GRIP_COL = 25               # static column: gripper actuator flag
DELTA = 0.5                 # ActionSpace.delta_scale (rad per normalized unit)
STATIC_NAMES = (["type_hinge", "type_slide", "ax_x", "ax_y", "ax_z", "rng_lo", "rng_hi", "rng_w", "depth", "n_desc",
                 "log_mass", "link_len", "log_kp", "log_force", "mimic"] + [f"asm{i}" for i in range(10)] + ["is_grip"])
DYN_NAMES = ["q", "qd", "anchor_or_pa", "anc_x", "anc_y", "anc_z", "wax_x", "wax_y", "wax_z", "jp_x", "jp_y", "jp_z",
             "jr_x", "jr_y", "jr_z", "lev_x", "lev_y", "lev_z"]
COLS = STATIC_NAMES + DYN_NAMES


def load_flow(path, dev):
    st = load_checkpoint(Path(path), map_location=dev)
    cfgj = st["config"]
    lcfg, E, R, P, _ = load_representation(Path(cfgj["representation"]), dev)
    pcfg = PolicyConfig(**dict(cfgj["policy"], horizon=lcfg.knots, latent_dim=lcfg.dz, aux=False))
    m = FlowPolicy(pcfg).to(dev)
    m.load_state_dict(st["model"])
    return m.eval(), lcfg, E, P, cfgj["representation"]


def load_R(path, dev):
    lcfg, E, R, P, res = load_representation(Path(path), dev)
    return R.eval(), E, res["latent_space_version"]


def ee_disp(nodes, a, mask):
    """TCP 1-step displacement [B,3] (m) from normalized arm-joint actions through the node Jacobian columns."""
    arm = mask & (nodes[..., GRIP_COL] < 0.5)
    return (nodes[..., JP] * (DELTA * a * arm)[..., None]).sum(1)


@torch.no_grad()
def decompose(flows, reals, E, P, lcfg, data, pool, dev, n_batches, seed, nfe):
    rng = random.Random(seed)
    kt = torch.tensor(lcfg.knot_times, device=dev)
    acc = {}

    def add(k, v):
        acc.setdefault(k, []).append(float(v))
    for _ in range(n_batches):
        sel, tgt, j = data.sample(128, rng, lcfg.max_phase_ticks, pool)
        batch, a, v, lab, r = data.fetch(sel, tgt, dev)
        af, am, ai = assembly_tokens(batch)
        mu, _ = E(batch, a, v, af, am, ai)
        zs = {"oracle_E": mu}
        ab = assembly_batch(batch)
        for fname, fm in flows.items():
            g = torch.Generator(device=dev).manual_seed(rng.randrange(1 << 30))
            zs[fname] = fm.sample(fm.prepare(ab), lcfg.knots, nfe=nfe, generator=g)
            add(f"gen_rel_mse/{fname}", ((zs[fname] - mu) ** 2).sum() / (mu ** 2).sum())
        smask = batch.bank_mask["scene"] & lab["slot_valid"].bool()
        for zname, z in zs.items():
            pm = probe_metrics(P(z, am, smask.shape[1]), lab, smask)
            for q in ("rel_pos_err_m", "subtask", "focused_on"):
                if q in pm and pm[q][1]:
                    add(f"probe_{q}/{zname}", pm[q][0] / pm[q][1])
        ph = torch.as_tensor(j * lcfg.control_dt, dtype=mu.dtype, device=dev)
        m = (r["v1"] & r["node_mask"])
        mf = m.float()
        armm = (m & (r["node"][..., GRIP_COL] < 0.5)).float()
        ee_demo = ee_disp(r["node"], r["a1"], m)
        add("zero_action_mse", ((r["a1"] ** 2) * mf).sum() / mf.sum())
        add("ee_demo_norm_cm", 100 * ee_demo.norm(dim=-1).mean())
        for rname, R in reals.items():
            node = r["node"]
            for zname, z in zs.items():
                pred = R(z, am, kt, ph, node, r["node_mask"], r["local"], node_asm=r.get("node_asm"))
                add(f"mse/{rname}<-{zname}", ((pred - r["a1"]) ** 2 * mf).sum() / mf.sum())
                add(f"mse_arm/{rname}<-{zname}", ((pred - r["a1"]) ** 2 * armm).sum() / armm.sum())
                ee = ee_disp(node, pred, m)
                add(f"ee_err_cm/{rname}<-{zname}", 100 * (ee - ee_demo).norm(dim=-1).mean())
                cos = torch.nn.functional.cosine_similarity(ee, ee_demo, dim=-1)
                big = ee_demo.norm(dim=-1) > 1e-3
                if big.any():
                    add(f"ee_cos/{rname}<-{zname}", cos[big].mean())
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


def node_ranges(packed_dir, max_rows=40000, seed=0):
    arr = {p.stem: np.load(p, mmap_mode="r") for p in Path(packed_dir).glob("*.npy")}
    n = len(arr["t"])
    idx = np.sort(np.random.default_rng(seed).choice(n, min(n, max_rows), replace=False))
    node = np.asarray(arr["node"][idx]).astype(np.float32)
    nn_ = np.asarray(arr["n_nodes"][idx])
    rid = np.asarray(arr["robot_id"][idx]) if "robot_id" in arr else np.zeros(len(idx), int)
    valid = np.arange(node.shape[1])[None] < nn_[:, None]
    out = {"n_nodes": sorted(set(nn_.tolist())), "per_robot_static": {}}
    flat = node[valid]
    out["min"] = flat.min(0).round(4).tolist()
    out["max"] = flat.max(0).round(4).tolist()
    out["mean"] = flat.mean(0).round(4).tolist()
    out["std"] = flat.std(0).round(4).tolist()
    for r_ in sorted(set(rid.tolist())):
        i0 = int(np.nonzero(rid == r_)[0][0])
        out["per_robot_static"][int(r_)] = node[i0, :nn_[i0], :26].round(3).tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-src", required=True)
    ap.add_argument("--rep-src", required=True)
    ap.add_argument("--flow-sft")
    ap.add_argument("--rep-refit", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--pack", required=True, help="target demo pack (stride 1)")
    ap.add_argument("--source-pack", help="source pack for the in-distribution reference and node ranges")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--budget-seed", type=int, required=True)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from rrp.contracts.workload import apply_cap
        apply_cap()
    torch.manual_seed(0)
    fsrc, lcfg, E, P, rep_E = load_flow(a.flow_src, dev)
    flows = {"flow_src": fsrc}
    if a.flow_sft:
        flows["flow_sft"] = load_flow(a.flow_sft, dev)[0]
    Rsrc, E_rep, lsv = load_R(a.rep_src, dev)
    reals = {"R_src": Rsrc}
    for spec in a.rep_refit:
        nm, p = spec.split("=", 1)
        R_, _, lsv_ = load_R(p, dev)
        if lsv_ != lsv:
            raise SystemExit(f"{nm}: latent space {lsv_} != {lsv}")
        reals[nm] = R_
    # encoder identity check: the flow's Stage A E and the deployed bundle's E must be the same latent space
    same_E = all(torch.equal(x, y) for x, y in zip(E.state_dict().values(), E_rep.state_dict().values()))
    rep_cfg = load_checkpoint(Path(a.rep_src), map_location="cpu")["config"]
    data = LatentData(Path(a.pack), zero_prev_action=True, anchor=bool(rep_cfg.get("realizer_anchor")),
                      drop_qd=bool(rep_cfg.get("realizer_drop_qd")))
    eps = sorted(set(data.ep.tolist()))
    chosen = set(eps[i] for i in nested_budget_indices(len(eps), [a.budget], a.budget_seed)[a.budget])
    train = [int(i) for i in np.nonzero(np.isin(data.ep, list(chosen)))[0]]
    held = [int(i) for i in np.nonzero(~np.isin(data.ep, list(chosen)))[0]]
    res = dict(label="NON-SEALED DIAGNOSTIC (target demo pack only; no eval scenes)", args=vars(a),
               encoder_flow_eq_bundle=same_E, latent_space_version=lsv, episodes=len(eps),
               train_rows=len(train), heldout_rows=len(held))
    res["target_heldout"] = decompose(flows, reals, E, P, lcfg, data, held, dev, a.batches, 1, a.nfe) if held else None
    res["target_train"] = decompose(flows, reals, E, P, lcfg, data, train, dev, a.batches, 2, a.nfe)
    res["node_cols"] = COLS
    res["target_nodes"] = node_ranges(a.pack)
    if a.source_pack:
        sdata = LatentData(Path(a.source_pack), zero_prev_action=True, anchor=data.anchor, drop_qd=data.drop_qd)
        spool = random.Random(3).sample(range(sdata.n), 20000)
        res["source_ref"] = decompose({"flow_src": fsrc}, {"R_src": Rsrc}, E, P, lcfg, sdata, spool, dev, a.batches, 3,
                                      a.nfe)
        res["source_nodes"] = node_ranges(a.source_pack)
        res["source_robot_ids"] = json.loads((Path(a.source_pack) / "meta.json").read_text()).get("robot_ids")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("target_heldout", "target_train", "source_ref") if k in res}, indent=1))


if __name__ == "__main__":
    main()
