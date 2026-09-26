"""Localize the latent route's failure on BC-visited states with a STATELESS oracle (no shadow teacher FSM).

Pass 1: the plain BC controller (learned:<ckpt>, deployable positive control) runs the ladder's matched dev scenes;
        every executed command is logged exactly.
Pass 2: each episode is replayed open loop from the log (deterministic sim; checked by the final outcome). At every
        replan tick t (every 8 ticks) the frozen Stage-A encoder E encodes the chunk BC ACTUALLY EXECUTED next
        (commands t..t+H-1, normalized with q0 at t, fp16-rounded like the pack) -> z_bc (ORACLE DIAGNOSTIC, valid
        off the teacher trajectory because it needs no teacher state). Not executed:
          system 0 (shadow) realizes z_bc every tick -> error vs BC's executed command (normalized MSE, arm/gripper),
          with the hold-still reference (measured q as the command) for scale;
          system i (flow, optional) samples z_gen at the same state -> |z_gen - z_bc| / |z_bc| and system 0's
          action from z_gen vs BC's command.
Usage: ladder_localize.py --policy <bc.pt> --policy-label <tag> --rep <representation.pt> [--flow <flow.pt>]
                          --robot panda_pg2 --n 30 --out <file.json>
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--policy-label", required=True)
    ap.add_argument("--rep", required=True)
    ap.add_argument("--flow")
    ap.add_argument("--robot", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=3_000_000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan", type=int, default=8)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero-qd", action="store_true", help="DIAGNOSTIC: zero the joint-velocity node column (27) at "
                    "system-0 input, to test whether system 0 copies the current velocity (causal confusion)")
    a = ap.parse_args()
    if a.zero_qd:
        import rrp.control.latent_realizer as LR
        _orig = LR.realizer_node_feats

        def _zq(s0, pi):
            nf = _orig(s0, pi).copy(); nf[:, 27] = 0; return nf
        LR.realizer_node_feats = _zq
    if a.seed_start < 3_000_000:
        sys.exit("dev seeds must be >= 3,000,000")
    dev = "cpu"
    from rrp.evaluation.ladder import (LadderConfig, load_models, run_ladder, summarize, OraclePacketPolicy,
                                       make_packet, PrevActionFeaturizer, _featurizer, _compare, Meter)
    from rrp.learning.latent_grpo import feasible_seeds, batched_ticks
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.control.latent_realizer import LatentSystem0
    from rrp.contracts.action import NativeCommand

    seeds = feasible_seeds(a.robot, a.seed_start, a.n)
    # ---- pass 1: BC rollouts with exact command log
    cfg = LadderConfig(route="learned", robot=a.robot, seeds=seeds, policy=a.policy, policy_label=a.policy_label,
                       replan_ticks=a.replan, max_steps=a.max_steps, nfe=a.nfe, compare_oracle=False, device=dev)
    t0 = time.time()
    log = {}
    rows1 = run_ladder(cfg, cmd_log=log)
    print("pass1", summarize(rows1)["success"], "/", len(rows1), f"{time.time() - t0:.0f}s", flush=True)

    # ---- pass 2: replay + stateless oracle + shadow system 0 (+ generator)
    cfg2 = LadderConfig(route="generated" if a.flow else "oracle", robot=a.robot, seeds=seeds, representation=a.rep,
                        flow=a.flow, nfe=a.nfe, device=dev)
    models, ids = load_models(cfg2)
    E, R, lcfg, res = models["E"], models["R"], models["lcfg"], models["res"]
    orc = OraclePacketPolicy(E, lcfg, res, dev)
    H = lcfg.horizon

    class Replay:                       # stands in for the shadow teacher's look-ahead: BC's executed future
        def __init__(self, cmds):
            self.cmds = cmds

        def lookahead(self, s, H_):
            t = s.step_count
            return [c for c in self.cmds[t:t + H_] if c is not None] or [None]

    robot = workbench_robots()[a.robot]()
    S, s0o, s0g, st, meters = [], [], [], [], []
    for k, sd in enumerate(seeds):
        s = Session(BUILDERS["pick_place"](robot, sd, n_distractors=sd % 3), seed=sd)
        f = s._rrp_featurizer = PrevActionFeaturizer(_featurizer(s), "zero")
        orc.shadows[id(s)] = Replay(log[k])
        S.append(s)
        meters.append(Meter(s))
        mk = lambda: LatentSystem0(R, f, latent_space_version=res["latent_space_version"],
                                   realizer_compat_version=res["realizer_compat_version"], device=dev)
        s0o.append(mk()); s0g.append(mk())
        st.append(dict(err_o_arm=[], err_o_grip=[], err_g_arm=[], err_g_grip=[], hold_arm=[], zrel=[], z_by_t=[],
                       err_o_by_j={}, err_g_by_j={}, hold_by_j={}, tcp={}))
    arm_n = grip_n = None
    Tmax = max(len(v) for v in log.values())
    for step in range(Tmax):
        act = [k for k in range(len(S)) if step < len(log[k])]
        if not act:
            break
        if step % a.replan == 0:
            ks = [k for k in act if log[k][step] is not None]
            if ks:
                zo = orc.encode([S[k] for k in ks])
                pg = models["flow"].packets([S[k] for k in ks]) if a.flow else [None] * len(ks)
                for j, k in enumerate(ks):
                    p = make_packet(S[k], _featurizer(S[k]), zo[j], lcfg.knot_times, orc.lsv, orc.rcv, orc.validity,
                                    source="target_encoder_oracle", policy_version="bc_executed_chunk_oracle")
                    s0o[k].receive(p, now=float(S[k].data.time), graph_version=S[k].runtime.graph_version)
                    if pg[j] is not None:
                        s0g[k].receive(pg[j], now=float(S[k].data.time), graph_version=S[k].runtime.graph_version)
                        c = _compare(models, S[k], pg[j].z, zo[j], dev)
                        st[k]["zrel"].append(c["z_dist_rel"])
                        st[k]["z_by_t"].append((step, c["z_dist_rel"], c["act_gen_vs_oracle_arm"]))
        co = batched_ticks([s0o[k] for k in act], [S[k] for k in act])
        cg = batched_ticks([s0g[k] for k in act], [S[k] for k in act]) if a.flow else [None] * len(act)
        for k, c_o, c_g in zip(act, co, cg):
            s, cmd = S[k], log[k][step]
            f = _featurizer(s)
            if arm_n is None:
                arm_n = [i for i, g in enumerate(f.aspace.is_gripper) if not g]
                grip_n = [i for i, g in enumerate(f.aspace.is_gripper) if g]
            if cmd is not None:
                q0z = np.zeros(len(f.aspace.node_group))
                lb = f.aspace.normalize([cmd], q0z)[0]
                qm = s.data.qpos[meters[k].qadr].copy()
                hold = f.aspace.normalize([dict(cmd, arm=qm.tolist())], q0z)[0]
                st[k]["hold_arm"].append(float(np.mean((lb[arm_n] - hold[arm_n]) ** 2)))
                j = int(round((float(s.data.time) - s0o[k].packet.valid_from) / s.dt)) if s0o[k].packet else -1
                for tag, c in (("o", c_o), ("g", c_g)):
                    if c is None:
                        continue
                    ca = f.aspace.normalize([c.groups], q0z)[0]
                    ea = float(np.mean((lb[arm_n] - ca[arm_n]) ** 2))
                    st[k][f"err_{tag}_arm"].append(ea)
                    st[k][f"err_{tag}_grip"].append(float(np.mean((lb[grip_n] - ca[grip_n]) ** 2)) if grip_n else 0.0)
                    st[k][f"err_{tag}_by_j"].setdefault(j, []).append(ea)
                    if tag == "o":        # TCP-space bias of system 0 vs BC's command (FK of the commanded arm q)
                        mt = meters[k]
                        tn = mt.tcp(); tb = mt.tcp_of(np.asarray(cmd["arm"], float)); tr = mt.tcp_of(np.asarray(c.groups["arm"], float))
                        cube = s.data.xpos[mt.cube].copy(); u = cube - tn; u = u / (np.linalg.norm(u) + 1e-9)
                        sb, sr = tb - tn, tr - tn
                        st[k]["tcp"].setdefault(0 if j == 0 else 1, []).append(
                            [float(np.dot(tr - tb, u)), *(tr - tb).tolist(), float(np.linalg.norm(sb)), float(np.linalg.norm(sr)),
                             float(np.dot(sr, sb) / (np.dot(sb, sb) + 1e-12))])
                st[k]["hold_by_j"].setdefault(j, []).append(st[k]["hold_arm"][-1])
            s.step(None if cmd is None else NativeCommand(controller_version=s.controller_version(), groups=cmd,
                                                          source="learned"))   # exact replay of BC's logged commands
    out_rows = []
    for k, s in enumerate(S):
        priv = bool(s.privileged_success())
        m = lambda v: float(np.mean(v)) if v else None
        out_rows.append(dict(seed=seeds[k], bc_success=rows1[k]["privileged_success"], replay_success=priv,
                             replay_consistent=priv == rows1[k]["privileged_success"], steps=len(log[k]),
                             sys0_bcoracle_err_arm=m(st[k]["err_o_arm"]), sys0_bcoracle_err_grip=m(st[k]["err_o_grip"]),
                             sys0_gen_err_arm=m(st[k]["err_g_arm"]), sys0_gen_err_grip=m(st[k]["err_g_grip"]),
                             z_gen_vs_bcoracle_rel=m(st[k]["zrel"]), hold_still_ref_arm=m(st[k]["hold_arm"]),
                             err_o_by_j={j: m(v) for j, v in sorted(st[k]["err_o_by_j"].items())},
                             err_g_by_j={j: m(v) for j, v in sorted(st[k]["err_g_by_j"].items())},
                             hold_by_j={j: m(v) for j, v in sorted(st[k]["hold_by_j"].items())},
                             tcp_bias={("j0" if g == 0 else "j1+"): dict(zip(["along_cube_m", "dx", "dy", "dz", "bc_step_m", "sys0_step_m",
                                                                                "gain_proj"], np.mean(v, 0).tolist()))
                                       for g, v in st[k]["tcp"].items()},
                             z_by_t=st[k]["z_by_t"]))
    pool = lambda key: float(np.mean([r[key] for r in out_rows if r[key] is not None])) \
        if any(r[key] is not None for r in out_rows) else None
    summ = dict(robot=a.robot, n=len(seeds), bc_label=f"learned:{a.policy_label}", bc_success=sum(r["bc_success"] for r in out_rows),
                replay_consistent=sum(r["replay_consistent"] for r in out_rows),
                sys0_bcoracle_err_arm=pool("sys0_bcoracle_err_arm"), sys0_bcoracle_err_grip=pool("sys0_bcoracle_err_grip"),
                sys0_gen_err_arm=pool("sys0_gen_err_arm"), sys0_gen_err_grip=pool("sys0_gen_err_grip"),
                z_gen_vs_bcoracle_rel=pool("z_gen_vs_bcoracle_rel"),
                hold_still_ref_arm=pool("hold_still_ref_arm"),
                note="z_bc = E(BC's executed next chunk): ORACLE DIAGNOSTIC; system 0 / generator not executed",
                err_o_by_j={j: float(np.mean([r["err_o_by_j"][j] for r in out_rows if j in r["err_o_by_j"]])) for j in range(a.replan)},
                hold_by_j={j: float(np.mean([r["hold_by_j"][j] for r in out_rows if j in r["hold_by_j"]])) for j in range(a.replan)},
                tcp_bias={g: {q: float(np.mean([r["tcp_bias"][g][q] for r in out_rows if g in r["tcp_bias"]]))
                              for q in ["along_cube_m", "dx", "dy", "dz", "bc_step_m", "sys0_step_m", "gain_proj"]}
                          for g in ("j0", "j1+")},
                checkpoints=ids, wall_s=time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=summ, rows=out_rows), indent=1, default=str))
    print(json.dumps(summ, default=str)[:1500])


if __name__ == "__main__":
    main()
