"""CLI for the controller-facing semantic latent path (R38)."""
from __future__ import annotations

import json
import time
from pathlib import Path


def cmd_rep(a):
    from rrp.learning.latent_train import train_representation
    cfg = json.loads(open(a.config).read())
    d = Path(cfg["out_dir"]); d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(train_representation(cfg, d), indent=1, default=str))


def register(sub):
    p = sub.add_parser("latent", help="controller-facing semantic latent (R38)").add_subparsers(dest="lat_cmd", required=True)
    r = p.add_parser("train-representation")
    r.add_argument("--config", required=True)
    r.set_defaults(fn=cmd_rep)
    register_more(p)
    register_probe_cmd(p)
    register_cell(p)
    register_latency(p)
    register_counterfactuals(p)
    register_eval_binding(p)
    from rrp import cli_dual_latent
    cli_dual_latent.register(p)
    register_grpo(p)
    register_causal(p)
    register_semantic(p)


def cmd_flow(a):
    from rrp.learning.latent_train import train_latent_flow
    cfg = json.loads(open(a.config).read())
    d = Path(cfg["out_dir"]); d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(train_latent_flow(cfg, d), indent=1, default=str))


def _load(a):
    import torch
    from rrp.policy.latent_runner import LatentPolicy
    from rrp.learning.latent_train import load_representation
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from rrp.ops.gpu import apply_cap
        apply_cap()
    pol = LatentPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=a.nfe)
    from rrp.learning.checkpoint import load_checkpoint
    rep = load_checkpoint(a.checkpoint, map_location="cpu")["config"]["representation"]
    lcfg, E, R, P, _ = load_representation(Path(rep), dev)
    return pol, R, P, dev


def cmd_eval(a):
    from rrp.evaluation.latent_eval import evaluate_latent
    from rrp.evaluation.statistics import wilson
    pol, R, P, dev = _load(a)
    summ = {}
    for robot in a.robots.split(","):
        res = evaluate_latent(pol, R, P, robot, list(range(a.seed_start, a.seed_start + a.episodes)), method=a.method,
                              batch=a.batch, out_path=Path(a.out), device=dev, replan_ticks=a.replan)
        att = [r for r in res if r.outcome != "infeasible"]
        k = sum(r.privileged_success for r in att)
        probes = {}
        for r in att:
            for q, (x, n) in r.probe_counts.items():
                s_, n_ = probes.get(q, (0, 0)); probes[q] = (s_ + x, n_ + n)
        summ[robot] = dict(attempted=len(att), successes=k, wilson95=wilson(k, len(att)),
                           outcomes={o: sum(r.outcome == o for r in res) for o in {r.outcome for r in res}},
                           system_i_calls=sum(r.system_i_calls for r in att), system0_ticks=sum(r.system0_ticks for r in att),
                           free_sample_packet_probes={q: (x / n if n else None) for q, (x, n) in probes.items()})
        print(robot, json.dumps(summ[robot]), flush=True)
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))


def cmd_eval_binding(a):
    """Closed-loop binding test on paired scenes: identical initial scene, each cube assigned as patient in turn.
    Reports success, wrong-object manipulation and whether behavior follows the assignment across a pair."""
    from rrp.evaluation.latent_eval import evaluate_latent, paired_keys, paired_scene_fn
    from rrp.evaluation.statistics import wilson
    pol, R, P, dev = _load(a)
    summ = {}
    for robot in a.robots.split(","):
        keys = paired_keys(a.seed_start, a.scenes)
        res = []
        for i in range(0, len(keys), a.batch):          # scene_fn per key (builders differ per key)
            grp = keys[i:i + a.batch]
            fns = {k: paired_scene_fn(k) for k in grp}
            res += evaluate_latent(pol, R, P, robot, grp, method=a.method, batch=len(grp), out_path=Path(a.out),
                                   device=dev, replan_ticks=a.replan, scene_fn=lambda rb, k: fns[k](rb, k))
        att = [r for r in res if r.outcome != "infeasible"]
        ok = [r for r in att if r.privileged_success]
        assigned = [r for r in att if "cube" in r.extra["moved"]]
        wrong = [r for r in att if any(b != "cube" for b in r.extra["moved"])]
        scenes = {}
        for r in att:
            scenes.setdefault(r.extra["scene_seed"], []).append(r)
        full = [v for v in scenes.values() if len(v) == 2 + v[0].extra["scene_seed"] % 2]
        k = len(ok)
        summ[robot] = dict(attempted=len(att), successes=k, wilson95=wilson(k, len(att)),
                           moved_assigned_object=len(assigned) / max(len(att), 1),
                           moved_wrong_object=len(wrong) / max(len(att), 1),
                           scenes_complete=len(full),
                           scenes_all_assignments_succeed=sum(all(r.privileged_success for r in v) for v in full),
                           scenes_all_follow_assignment=sum(all("cube" in r.extra["moved"] and
                                                                not any(b != "cube" for b in r.extra["moved"])
                                                                for r in v) for v in full),
                           outcomes={o: sum(r.outcome == o for r in res) for o in {r.outcome for r in res}})
        print(robot, json.dumps(summ[robot]), flush=True)
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))


def cmd_disturb(a):
    from rrp.evaluation.latent_eval import disturbance_test
    pol, R, P, dev = _load(a)
    rows = disturbance_test(pol, R, a.robots.split(",")[0], list(range(a.seed_start, a.seed_start + a.episodes)), device=dev)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows))
    for k in ("final_dev_closed_loop_m", "final_dev_open_loop_delta_m", "final_dev_servo_absolute_m"):
        print(k, round(float(sum(r[k] for r in rows) / len(rows)), 4))


def register_eval_binding(p):
    e = p.add_parser("eval-binding", help="closed-loop paired-binding test (same scene, different assigned object)")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--robots", required=True)
    e.add_argument("--scenes", type=int, default=10)
    e.add_argument("--seed-start", type=int, default=3100000)
    e.add_argument("--method", default="latent")
    e.add_argument("--nfe", type=int, default=8)
    e.add_argument("--replan", type=int, default=8)
    e.add_argument("--batch", type=int, default=15)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_eval_binding)


def register_more(p):
    f = p.add_parser("train-flow")
    f.add_argument("--config", required=True)
    f.set_defaults(fn=cmd_flow)
    for name, fn in (("evaluate", cmd_eval), ("disturbance", cmd_disturb)):
        e = p.add_parser(name)
        e.add_argument("--checkpoint", required=True)
        e.add_argument("--robots", required=True)
        e.add_argument("--episodes", type=int, default=20)
        e.add_argument("--seed-start", type=int, default=3000000)
        e.add_argument("--method", default="latent")
        e.add_argument("--nfe", type=int, default=8)
        e.add_argument("--replan", type=int, default=8)
        e.add_argument("--batch", type=int, default=16)
        e.add_argument("--out", required=True)
        e.set_defaults(fn=fn)


def cmd_fit_probes(a):
    from rrp.learning.latent_train import fit_probes_on_frozen
    res = fit_probes_on_frozen(Path(a.representation), Path(a.packed_dir), Path(a.out), steps=a.steps,
                               metadata_only=a.metadata_only, binding_cf=a.binding_cf)
    print(json.dumps(res, indent=1))


def register_probe_cmd(p):
    f = p.add_parser("fit-probes", help="measurement probe on frozen detached z (or metadata-only control)")
    f.add_argument("--representation", required=True)
    f.add_argument("--packed-dir", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--steps", type=int, default=6000)
    f.add_argument("--metadata-only", action="store_true")
    f.add_argument("--binding-cf", type=float, default=0.0,
                   help="fraction of each probe batch appended as counterfactual-binding copies (focus follows binding)")
    f.set_defaults(fn=cmd_fit_probes)


def cmd_cell(a):
    import torch
    from rrp.evaluation.latent_campaign import run_latent_cell
    if torch.cuda.is_available():
        from rrp.ops.gpu import apply_cap
        apply_cap()
    proto = json.loads(open(a.protocol).read())
    print(json.dumps(run_latent_cell(proto, a.method, a.seed, base_flow_config=a.base_flow_config,
                                     target_packed=a.target_packed), indent=1))


def register_cell(p):
    c = p.add_parser("cell", help="resumable latent_slice1 campaign cell")
    c.add_argument("--protocol", default="configs/eval/latent_slice1.json")
    c.add_argument("--method", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--base-flow-config", required=True)
    c.add_argument("--target-packed", default="artifacts/packed/latent_targets")
    c.set_defaults(fn=cmd_cell)


def cmd_latency(a):
    from rrp.evaluation.latency import latent_latency_suite
    print(json.dumps(latent_latency_suite(a.checkpoint, Path(a.out), direct_ckpt=a.direct, direct_config=a.direct_config, reps=a.reps), indent=1))


def register_latency(p):
    c = p.add_parser("latency")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--direct")
    c.add_argument("--reps", type=int, default=40)
    c.add_argument("--direct-config", help="time the direct-action architecture with random init (no checkpoint)")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_latency)


def cmd_counterfactuals(a):
    import torch
    from rrp.evaluation.latent_counterfactuals import counterexample, counterexample_v1, embodiment_swap
    from rrp.learning.latent_train import load_representation
    dev = "cuda" if a.gpu and torch.cuda.is_available() else "cpu"
    _, E, _, P, res = load_representation(Path(a.representation), dev)
    if a.probe:                                   # measurement probe fitted post hoc on frozen z (fair across variants)
        from rrp.model.latent_probes import PacketProbe
        st = torch.load(a.probe, map_location=dev, weights_only=False)
        P = PacketProbe(**st["cfg"]).to(dev).eval()
        P.load_state_dict(st["state"])
    ds = Path(a.dataset)
    out = dict(representation=a.representation, probe=a.probe or "representation (jointly trained)",
               latent_space_version=res["latent_space_version"],
               source="encoded teacher targets (not generated packets)",
               counterexample=counterexample(E, P, ds, robot=a.robot, n=a.n, dev=dev, per_episode=a.per_episode),
               counterexample_v1=counterexample_v1(E, P, ds, robot=a.robot, n=a.n, dev=dev),
               embodiment_swap=embodiment_swap(E, P, ds, arm=a.arm, seeds=range(a.seed_start, a.seed_start + a.n), dev=dev))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("counterexample")} | {
        "counterexample": {k: v for k, v in out["counterexample"].items() if k != "rows"},
        "counterexample_v1": {k: v for k, v in out["counterexample_v1"].items() if k != "rows"}}, indent=1))


def register_counterfactuals(p):
    c = p.add_parser("counterfactuals", help="section-6 counterexample + compatible embodiment swap on frozen E/P")
    c.add_argument("--representation", required=True)
    c.add_argument("--dataset", default="artifacts/datasets/pick_place_primary_v3dart")
    c.add_argument("--robot", default="panda_pg2")
    c.add_argument("--arm", default="parm6")
    c.add_argument("--n", type=int, default=20)
    c.add_argument("--seed-start", type=int, default=0)
    c.add_argument("--per-episode", type=int, default=1, help="counterexample samples per episode")
    c.add_argument("--gpu", action="store_true")
    c.add_argument("--probe", help="post-hoc probe .pt (fit-probes output); default: the representation's own P")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_counterfactuals)


def cmd_grpo(a):
    from rrp.learning.flow_sde import SDEConfig
    from rrp.learning.grpo import GRPOConfig
    from rrp.learning.latent_grpo import LatentGRPORunConfig, RewardConfig, train_latent_grpo
    g = GRPOConfig(group_size=a.group_size, lr=a.lr, epochs=a.epochs, minibatch=a.minibatch, kl_coef=a.kl_coef,
                   clip=a.clip, trainable=a.trainable,
                   sde=SDEConfig(nfe=a.nfe, noise_level=a.noise_level, first_step="clamp", last_step=a.last_step))
    cfg = LatentGRPORunConfig(checkpoint=a.checkpoint, robot=a.robot, out_dir=a.out, iters=a.iters,
                              groups_per_iter=a.groups_per_iter, train_seed_start=a.train_seed_start,
                              eval_seed_start=a.eval_seed_start, eval_episodes=a.eval_episodes, eval_every=a.eval_every,
                              eval_batch=a.eval_batch, max_steps=a.max_steps, nfe=a.nfe, seed=a.seed,
                              allow_target=a.allow_target, prefix_steps=a.teacher_prefix_steps,
                              reward=RewardConfig(shaping_events=a.shaping_events, shaping_dist=a.shaping_dist,
                                                  shaping_reach=a.shaping_reach), grpo=g)
    res = train_latent_grpo(cfg)
    print(json.dumps(dict(evals=res["evals"], accounting=res["accounting"]), indent=1))


def register_grpo(p):
    c = p.add_parser("grpo", help="packet-policy GRPO: RL fine-tunes system i only (system 0/encoder/probes frozen)")
    c.add_argument("--checkpoint", required=True, help="system-i flow checkpoint (e.g. an SFT-adapted policy.pt)")
    c.add_argument("--robot", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--allow-target", action="store_true", help="required for sealed target bodies (campaign only)")
    c.add_argument("--iters", type=int, default=20)
    c.add_argument("--groups-per-iter", type=int, default=8)
    c.add_argument("--group-size", type=int, default=8)
    c.add_argument("--train-seed-start", type=int, default=3_100_000)
    c.add_argument("--eval-seed-start", type=int, default=3_000_000)
    c.add_argument("--eval-episodes", type=int, default=64)
    c.add_argument("--eval-every", type=int, default=5)
    c.add_argument("--eval-batch", type=int, default=32)
    c.add_argument("--max-steps", type=int, default=300)
    c.add_argument("--lr", type=float, default=1e-6)
    c.add_argument("--epochs", type=int, default=2)
    c.add_argument("--minibatch", type=int, default=64)
    c.add_argument("--clip", type=float, default=0.2)
    c.add_argument("--kl-coef", type=float, default=0.05)
    c.add_argument("--trainable", default="action_expert", choices=["action_expert", "all"])
    c.add_argument("--nfe", type=int, default=8)
    c.add_argument("--noise-level", type=float, default=0.5)
    c.add_argument("--last-step", default="deterministic", choices=["deterministic", "stochastic"])
    c.add_argument("--shaping-events", type=float, default=0.0, help="weight of PUBLIC task-event progress reward")
    c.add_argument("--shaping-dist", type=float, default=0.0, help="weight of PRIVILEGED cube-zone distance reward")
    c.add_argument("--shaping-reach", type=float, default=0.0,
                   help="weight of PRIVILEGED min TCP-to-cube distance reward (curriculum, reward only)")
    c.add_argument("--teacher-prefix-steps", type=int, default=0,
                   help="curriculum: SCRIPTED TEACHER controls the first N ticks (labelled; also evaluated without)")
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=cmd_grpo)
# ------------------------------------------------------------------ causal edits / composition (acceptance track)
TARGET_BODIES = ("xarm7_pg2", "xarm7_tf3", "panda_tf3")      # D-025: never used for development


def _causal_common(a, window_conds, episode_conds):
    import torch
    from rrp.evaluation import latent_causal as lc
    robots = a.robots.split(",")
    if a.seed_start < 3_000_000 or any(r in TARGET_BODIES for r in robots):
        raise SystemExit("dev rule (D-025): source/dev bodies and dev seeds >= 3,000,000 only")
    pol, R, P, dev = _load(a)
    from rrp.learning.checkpoint import load_checkpoint
    rep = Path(load_checkpoint(a.checkpoint, map_location="cpu")["config"]["representation"])
    probe = a.probe or (str(rep.parent / "probe_posthoc.pt") if (rep.parent / "probe_posthoc.pt").exists() else None)
    P = lc.load_probe(probe, P, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seed_start, a.seed_start + a.episodes))
    meta = dict(checkpoint=a.checkpoint, source=f"learned:{a.checkpoint}", representation=str(rep),
                edit_probe=probe or "representation (jointly trained)", robots=robots, seeds=[seeds[0], seeds[-1]],
                delta_m=a.delta_cm / 100, nfe=a.nfe, replan=a.replan, t=time.time(),
                note="probe defines edit directions only; evidence = system-0 behaviour vs the unedited packet")
    summ = dict(meta=meta)
    conds = a.conditions.split(",") if a.conditions else None
    if a.protocol in ("window", "both"):
        rows = []
        for r in robots:
            rows += lc.window_protocol(pol, R, P, r, seeds, window=a.window, replan=a.replan, delta_m=a.delta_cm / 100,
                                       conditions=tuple(conds or window_conds), edit_steps=a.edit_steps, dev=dev)
        (out / "window_rows.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
        summ["window"] = lc.summarize_window(rows, a.delta_cm / 100)
        summ["window"]["per_robot"] = {r: lc.summarize_window([x for x in rows if x.get("robot") == r],
                                                              a.delta_cm / 100) for r in robots}
    if a.protocol in ("episode", "both"):
        rows = []
        for r in robots:
            rows += lc.episode_protocol(pol, R, P, r, seeds[:a.episode_seeds or None],
                                        conditions=tuple(conds or episode_conds), replan=a.replan,
                                        max_steps=a.max_steps, delta_m=a.delta_cm / 100, edit_steps=a.edit_steps,
                                        dev=dev)
        (out / "episode_rows.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
        summ["episode"] = lc.summarize_episodes(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)[:20000])


def cmd_causal(a):
    from rrp.evaluation import latent_causal as lc
    _causal_common(a, lc.WINDOW_CONDS, lc.EPISODE_CONDS)


def cmd_composition(a):
    _causal_common(a, ("control_replay", "rel+x", "rel+y", "rel+xy", "sum_xy"), ("control", "chain_A_then_B"))


def _causal_args(c):
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--robots", default="panda_pg2,parm6_pg2,ur5e_pg2")
    c.add_argument("--episodes", type=int, default=16, help="dev seeds per robot")
    c.add_argument("--episode-seeds", type=int, default=0, help="episode protocol: first N of the seeds (0 = all)")
    c.add_argument("--seed-start", type=int, default=3000000)
    c.add_argument("--protocol", choices=["window", "episode", "both"], default="both")
    c.add_argument("--conditions", help="comma list (default: all for this command)")
    c.add_argument("--probe", help="edit-direction probe (.pt from fit-probes); default <rep dir>/probe_posthoc.pt")
    c.add_argument("--delta-cm", type=float, default=5.0)
    c.add_argument("--window", type=int, default=8)
    c.add_argument("--edit-steps", type=int, default=80)
    c.add_argument("--max-steps", type=int, default=240)
    c.add_argument("--nfe", type=int, default=8)
    c.add_argument("--replan", type=int, default=8)
    c.add_argument("--out", required=True)


def register_causal(p):
    c = p.add_parser("causal", help="causal edits of the RECEIVED packet (window + episode protocols)")
    _causal_args(c)
    c.set_defaults(fn=cmd_causal)
    c = p.add_parser("composition", help="packet composition: edit superposition (window) + two-object chain")
    _causal_args(c)
    c.set_defaults(fn=cmd_composition)


# ------------------------------------------------------------------ semantic interventions (correction item 4)
def cmd_semantic(a):
    import torch
    from rrp.evaluation import latent_semantic_edits as se
    from rrp.evaluation import latent_causal as lc
    from rrp.learning.latent_train import load_representation
    robots = a.robots.split(",")
    if a.seed_start < 3_000_000 or any(r in TARGET_BODIES for r in robots):
        raise SystemExit("dev rule (D-025): source/dev bodies and dev seeds >= 3,000,000 only")
    dev = "cuda" if a.gpu and torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from rrp.ops.gpu import apply_cap
        apply_cap()
    if a.route == "teacher":
        rep = Path(a.representation)
        _, _, R, P, _ = load_representation(rep, dev)
        src = se.TeacherSource()
        label = "scripted_teacher (privileged expert native commands; reference rung)"
    elif a.route == "oracle":
        rep = Path(a.representation)
        lcfg, E, R, P, res = load_representation(rep, dev)
        if a.oracle_expert == "bc":
            from rrp.policy.runner import LearnedPolicy
            src = se.OracleSource(E, lcfg, res, rep, dev, expert="bc",
                                  bc=LearnedPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=a.nfe, execute_prefix=8))
            label = (f"ORACLE DIAGNOSTIC target_encoder_oracle:E({rep}) + STATELESS BC expert demo "
                     f"(learned:{a.checkpoint} chunk for the edited context)")
        else:
            src = se.OracleSource(E, lcfg, res, rep, dev)
            label = f"ORACLE DIAGNOSTIC target_encoder_oracle:E({rep}) + scripted_teacher demo"
    elif a.route == "bc":
        from rrp.policy.runner import LearnedPolicy
        from rrp.evaluation.ladder import sha256_file
        rep = Path(a.representation) if a.representation else None
        R = P = None
        if rep:
            _, _, R, P, _ = load_representation(rep, dev)
        src = se.BCSource(LearnedPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=a.nfe, execute_prefix=8),
                          a.checkpoint)
        label = (f"learned:{a.checkpoint} (direct-action BC reference controller, NOT the latent path; "
                 f"sha256 {sha256_file(a.checkpoint)[:16]})")
    else:
        from rrp.learning.checkpoint import load_checkpoint
        from rrp.policy.latent_runner import LatentPolicy
        rep = Path(load_checkpoint(a.checkpoint, map_location="cpu")["config"]["representation"])
        _, _, R, P, _ = load_representation(rep, dev)
        src = se.GeneratedSource(LatentPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=a.nfe))
        label = f"learned:{a.checkpoint}"
    probe = None
    if rep is not None:
        probe = a.probe or (str(rep.parent / "probe_posthoc.pt") if (rep.parent / "probe_posthoc.pt").exists() else None)
        P = lc.load_probe(probe, P, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"semantic_rows_{a.route}.jsonl"
    if a.scene == "paired":
        seeds = se.paired_edit_keys(a.seed_start, a.episodes)
        conds = a.conditions or ",".join(se.PAIRED_CONDITIONS)
        scenes = ("pick_place_paired: episode key = 10*scene_seed + patient (patient alternates with the seed), "
                  "n_objects = 2 + seed % 2; rebind target = distractor0 (first other cube in physical order); "
                  "rebind edit = task entity descriptor + public binding only")
    else:
        seeds = list(range(a.seed_start, a.seed_start + a.episodes))
        conds = a.conditions or ",".join(se.CONDITIONS)
        scenes = "pick_place with n_distractors = max(1, seed % 3)"
    rows = []
    for r in robots:
        rows += se.semantic_suite(src, R, P, r, seeds, tuple(conds.split(",")), max_steps=a.max_steps, dev=dev,
                                  out_path=rows_path, scene=a.scene)
    summ = dict(meta=dict(route=a.route, source=label, representation=str(rep), checkpoint=a.checkpoint,
                          orthogonal_probe=probe, robots=robots, seeds=[seeds[0], seeds[-1]], t=time.time(),
                          scenes=scenes, conditions=conds),
                summary=se.summarize_semantic(rows))
    (out / f"semantic_summary_{a.route}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def cmd_arm(a):
    import torch
    from rrp.evaluation import latent_semantic_edits as se
    from rrp.evaluation import latent_causal as lc
    from rrp.learning.latent_train import load_representation
    if a.seed_start < 3_000_000:
        raise SystemExit("dev rule (D-025): dev seeds >= 3,000,000 only")
    dev = "cuda" if a.gpu and torch.cuda.is_available() else "cpu"
    if a.route == "generated":
        from rrp.learning.checkpoint import load_checkpoint
        from rrp.evaluation.dual_latent_eval import DualLatentPolicy
        rep = Path(load_checkpoint(a.checkpoint, map_location="cpu")["config"]["representation"])
        _, _, R, P, _ = load_representation(rep, dev)
        src = se.GeneratedSource(DualLatentPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=a.nfe))
        label = f"learned:{a.checkpoint}"
    else:
        rep = Path(a.representation)
        lcfg, E, R, P, res = load_representation(rep, dev)
        if a.route == "oracle":
            src = se.OracleSource(E, lcfg, res, rep, dev)
            label = f"ORACLE DIAGNOSTIC target_encoder_oracle:E({rep}) + scripted_teacher demo"
        else:
            src = se.TeacherSource()
            label = "scripted_teacher (privileged expert native commands; reference rung)"
    probe = a.probe or (str(rep.parent / "probe_posthoc.pt") if (rep.parent / "probe_posthoc.pt").exists() else None)
    P = lc.load_probe(probe, P, dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seed_start, a.seed_start + a.episodes))
    conds = tuple((a.conditions or ",".join(se.ARM_CONDITIONS)).split(","))
    rows = se.arm_suite(src, R, P, a.pairs.split(","), seeds, conds, max_steps=a.max_steps, dev=dev,
                        out_path=out / f"arm_rows_{a.route}.jsonl")
    summ = dict(meta=dict(route=a.route, source=label, representation=str(rep), checkpoint=a.checkpoint,
                          orthogonal_probe=probe, pairs=a.pairs, seeds=[seeds[0], seeds[-1]], conditions=conds,
                          t=time.time(), scenes="assign_pick_place dev scenes; assigned arm = left for even seeds, "
                                                "right for odd; edit rebinds the actor of take/place"),
                summary=se.summarize_arm(rows))
    (out / f"arm_summary_{a.route}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def register_semantic(p):
    c = p.add_parser("arm-edits", help="manipulator-assignment edits (dual-arm assign scenes) + matched controls")
    c.add_argument("--route", choices=["teacher", "oracle", "generated"], required=True)
    c.add_argument("--representation")
    c.add_argument("--checkpoint")
    c.add_argument("--pairs", default="panda_pg2__ur5e_pg2,parm5_pg2__parm5_pg2,parm5_pg2__parm6_tf3")
    c.add_argument("--episodes", type=int, default=6)
    c.add_argument("--seed-start", type=int, default=3000000)
    c.add_argument("--conditions")
    c.add_argument("--probe")
    c.add_argument("--max-steps", type=int, default=400)
    c.add_argument("--nfe", type=int, default=8)
    c.add_argument("--gpu", action="store_true")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_arm)
    c = p.add_parser("semantic-edits", help="valid semantic packet edits (binding / goal) + irrelevant-edit controls")
    c.add_argument("--route", choices=["teacher", "oracle", "generated", "bc"], required=True,
                   help="bc: direct-action BC reference controller (--checkpoint), context-conditioned, no packet")
    c.add_argument("--representation", help="oracle route: frozen representation.pt")
    c.add_argument("--checkpoint", help="generated route: flow policy checkpoint")
    c.add_argument("--robots", default="panda_pg2")
    c.add_argument("--episodes", type=int, default=12)
    c.add_argument("--seed-start", type=int, default=3000000)
    c.add_argument("--conditions", help="default: all conditions of the scene family")
    c.add_argument("--oracle-expert", choices=["teacher", "bc"], default="teacher",
                   help="oracle route demo source; bc = stateless BC chunk (--checkpoint = BC policy)")
    c.add_argument("--scene", choices=["pick_place", "paired"], default="pick_place",
                   help="paired: binding-paired scenes (rebinding is physically valid; approach-level metrics)")
    c.add_argument("--probe")
    c.add_argument("--max-steps", type=int, default=300)
    c.add_argument("--nfe", type=int, default=8)
    c.add_argument("--gpu", action="store_true")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_semantic)
