"""Refactor goldens (D-140): digests of fixed, seeded inputs through the featurizers, teachers, physics, system i
and system 0. Recorded on the pre-refactor code (main c14ca55); every refactor stage must keep them byte-identical.
Re-record ONLY for an intended behaviour change: RRP_GOLDEN_RECORD=1 pytest tests/unit/test_golden.py (and say so
in research/decisions.md). Random-weight nets: plumbing only, no number here is a result."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parents[1] / "data" / "golden.json"


def _h(*parts, decimals=None) -> str:
    m = hashlib.sha256()
    for p in parts:
        if isinstance(p, dict):
            for k in sorted(p):
                m.update(k.encode()); m.update(_h(p[k], decimals=decimals).encode())
            continue
        if hasattr(p, "detach"):
            p = p.detach().cpu().numpy()
        a = np.asarray(p)
        if a.dtype.kind in "fc" and decimals is not None:
            a = np.round(a.astype(np.float64), decimals)
        if a.dtype == object:
            m.update(repr(p).encode())
        else:
            m.update(str(a.dtype).encode()); m.update(str(a.shape).encode()); m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()[:24]


def _pi(pi) -> dict:
    """All array-valued / numeric fields of a PolicyInput (ids and timestamps excluded)."""
    out = {}
    for k, v in sorted(vars(pi).items()):
        if isinstance(v, np.ndarray) and v.dtype != object:
            out[k] = v
        elif isinstance(v, (int, float, bool)) and not isinstance(v, str):
            out[k] = np.asarray(v)
    return out


_REC: dict = {}


@pytest.fixture(scope="module")
def golden():
    rec = os.environ.get("RRP_GOLDEN_RECORD") == "1"
    ref = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    yield (lambda k, v: _REC.__setitem__(k, v)) if rec else (lambda k, v: _check(ref, k, v))
    if rec:
        ref.update(_REC)
        GOLDEN.write_text(json.dumps(ref, indent=1, sort_keys=True) + "\n")


def _check(ref, k, v):
    assert k in ref, f"no golden for {k}; record with RRP_GOLDEN_RECORD=1"
    assert ref[k] == v, f"golden {k} changed"


def _arm_session(seed=3):
    from rrp.envs.mujoco.fixtures import make_pick_place_session
    return make_pick_place_session(seed=seed)


def test_arm_featurizer_teacher_physics(golden):
    from rrp.policies.features.featurizer import featurizer_for
    from rrp.policies.teachers.arm_smooth import make_arm_teacher
    s = _arm_session()
    obs = s.observe()
    f = featurizer_for(s)
    golden("arm.featurizer.reset", _h(_pi(f(obs))))
    golden("arm.obs.reset", _h(obs.measured_node_state.qpos, [d.position_estimate for d in obs.object_descriptors]))
    t = make_arm_teacher(s)
    cmds, qs = [], []
    for _ in range(25):
        c = t.act()
        st = s.step(c)
        cmds.append({k: np.asarray(v) for k, v in c.groups.items()})
        qs.append(st.qpos)
    golden("arm.teacher.commands", _h(*cmds))
    golden("arm.teacher.qpos", _h(np.stack(qs)))
    golden("arm.featurizer.t25", _h(_pi(f(s.observe(), prev_action=None))))
    tr = s.truth()
    golden("arm.truth.t25", _h(tr.object_poses, np.asarray(sorted(tr.predicates.items()), dtype=object)))


def _flow(seed=0, dz=8, K=4):
    import torch
    from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
    torch.manual_seed(seed)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=K, latent_dim=dz))
    with torch.no_grad():
        m.out.weight.normal_(0, 0.2)
        m.set_target_norm(torch.linspace(-1, 1, dz), torch.linspace(0.5, 2, dz))
    return m.eval()


def test_arm_latent_system_i_and_system0(golden):
    import torch
    from rrp.policies.latent import LatentPolicy
    from rrp.policies.system0 import LatentRealizer, LatentSystem0
    from rrp.policies.features.featurizer import featurizer_for
    s = _arm_session()
    pol = LatentPolicy(_flow(), knot_times=(0.1, 0.3, 0.5, 0.7), latent_space_version="ls-g", realizer_compat_version="rz-g",
                       device="cpu", nfe=4, seed=3)
    pk = pol.packets([s])[0]
    golden("latent.packet.z", _h(pk.z, decimals=5))
    golden("latent.packet.meta", _h(np.asarray([pk.knot_times, [pk.valid_until - pk.valid_from]], dtype=object),
                                    np.asarray([a.handle for a in pk.assemblies] + [str(pk.assembly_mask)], dtype=object)))
    torch.manual_seed(1)
    R = LatentRealizer(8, width=32, layers=1)
    s0 = LatentSystem0(R, featurizer_for(s), latent_space_version="ls-g", realizer_compat_version="rz-g")
    s0.receive(pk, now=float(s.data.time))
    cmds, qs = [], []
    for _ in range(6):
        c = s0.tick(s, s.controller_version())
        cmds.append({k: np.asarray(v) for k, v in c.groups.items()})
        qs.append(s.step(c).qpos)
    golden("latent.system0.commands", _h(*cmds, decimals=5))
    golden("latent.system0.qpos", _h(np.stack(qs), decimals=6))


def test_arm_bc_chunk(golden):
    import torch
    from rrp.policies.bc import LearnedPolicy
    from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
    s = _arm_session()
    torch.manual_seed(2)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=8, latent_dim=1))
    with torch.no_grad():
        m.out.weight.normal_(0, 0.2)
    pol = LearnedPolicy(m, None, "cpu", nfe=4, execute_prefix=4, seed=5)
    ch = pol.chunks([s])[0]
    golden("bc.chunk", _h({g.group: g.values for g in ch.command_groups}, decimals=5))


def test_dual_featurizer_and_teacher(golden):
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.dual import DualSession
    from rrp.envs.mujoco.dual_scenarios import build_support_insert
    from rrp.policies.features.multi import MultiFeaturizer
    from rrp.policies.teachers.dual import TEACHERS
    W = workbench_robots()
    s = DualSession(build_support_insert([W["parm5l_pg2"](), W["parm6_pg2"]()], 3), seed=3)
    f = MultiFeaturizer(s.model, s.scenario.robots)
    golden("dual.featurizer.reset", _h(_pi(f(s.observe()))))
    t = TEACHERS["support_insert"](s)
    cmds, qs = [], []
    for _ in range(15):
        c = t.act()
        st = s.step(c)
        cmds.append({f"{r}:{k}": np.asarray(v) for r, nc in sorted(c.items()) for k, v in nc.groups.items()})
        qs.append(st.qpos)
    golden("dual.teacher.commands", _h(*cmds))
    golden("dual.teacher.qpos", _h(np.stack(qs)))


def test_packet_wire_format(golden):
    from rrp.core.latent_action import LatentActionChunk
    p = LatentActionChunk(latent_space_version="ls", realizer_compat_version="rz",
                          z=np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 7, knot_times=[0.1, 0.3],
                          assemblies=[dict(handle=f"asm:{'0' * 16}:r/{i}", robot_index=0) for i in range(3)],
                          assembly_mask=[True, True, False], entity_registry=[dict(handle="ent:0")],
                          observation_id="o", graph_version=1, runtime_version=2, robot_spec_hash="h",
                          generated_at=1.0, valid_from=1.0, valid_until=1.8, source="learned", policy_version="v")
    golden("packet.bytes", hashlib.sha256(p.to_bytes()).hexdigest()[:24])


def _saved_actor(tmp, name, obs_dim, act_dim, meta, seed):
    import torch
    from rrp.envs.mujoco.tracker_nets import mlp
    torch.manual_seed(seed)
    net = mlp(obs_dim, (16, 8), act_dim)
    with torch.no_grad():
        net[-1].weight.mul_(30.0)          # saturating actions: exercises target_margin clipping
    g = torch.Generator().manual_seed(seed)
    st = dict(actor=net.state_dict(), obs_mean=torch.randn(obs_dim, generator=g) * 0.1,
              obs_var=torch.rand(obs_dim, generator=g) + 0.5,
              meta=dict(dict(obs_dim=obs_dim, act_dim=act_dim, hidden=[16, 8], control_dt=0.02, iter=7), **meta))
    p = Path(tmp) / f"{name}.pt"
    torch.save(st, str(p))
    return p


def _tracker_actions(tracker, env, extra=None, steps=12):
    import mujoco
    b, d = env.b, env.data[0]
    if extra:
        tracker.extra_fn = lambda data: np.linspace(-1, 1, extra).astype(np.float32) * float(data.time + 1)
    out = []
    for k in range(steps):
        a = tracker.act(d, np.array([0.3, 0.0, 0.1]) * (k % 3 != 0))   # zero command every 3rd tick: clock_gate
        out.append(a)
        d.ctrl[:] = 0
        mujoco.mj_step(b.model, d)
    return out


@pytest.mark.parametrize("case", ["plain", "options", "extra_obs", "morph_v1"])
def test_learned_tracker_saved_actor_options(golden, tmp_path, case):
    """W13 deployment options carried in saved actor files (clock_gate, target_margin, ref_ff, extra_obs_dim,
    obs_format morph_v1) must keep producing the same actions (lead note, D-140 S4)."""
    from rrp.bodies.legged import legged_body
    from rrp.envs.mujoco.legged_core import LeggedEnv
    from rrp.envs.mujoco.legged_tracker import LearnedTracker
    body = "phum_3"                     # biped: ref_ff and clock_gate apply
    env = LeggedEnv(lambda: legged_body(body), 1, 5)
    b = env.b
    extra = 5 if case == "extra_obs" else 0
    if case == "morph_v1":
        from rrp.envs.mujoco.morph_obs import NS, OBS_DIM
        meta = dict(obs_format="morph_v1", train_bodies=["phum_1"], clock_gate=True, target_margin=0.03, body="shared")
        p = _saved_actor(tmp_path, case, OBS_DIM, NS, meta, 4)
    else:
        meta = dict(body=body)
        if case == "options":
            meta.update(clock_gate=True, target_margin=0.05, ref_ff=0.2, ref_ff_vmax=0.5)
        if extra:
            meta.update(extra_obs_dim=extra, clock_gate=True)
        p = _saved_actor(tmp_path, case, b.obs_dim + extra, b.n, meta, 3)
    tr = LearnedTracker(p, b, body)
    golden(f"tracker.{case}.version", tr.version)
    golden(f"tracker.{case}.actions", _h(np.stack(_tracker_actions(tr, env, extra)), decimals=6))


def _old_ladder_wilson(k, n, z=1.96):
    import math
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


def test_ladder_wilson_is_bitwise_unchanged():
    from rrp.harness.eval.ladder import wilson
    for n in range(0, 61):
        for k in range(0, n + 1):
            assert wilson(k, n) == _old_ladder_wilson(k, n)
            assert wilson(k, n, 1.959964) == _old_ladder_wilson(k, n, 1.959964)
    assert type(wilson(3, 10)) is tuple


def test_legacy_pickle_paths_load(tmp_path):
    """Dataset pickles written before D-140 name `rrp.data.features.PolicyInput`; read_episode remaps them."""
    import gzip
    import pickle
    from rrp.harness.data.collect import read_episode
    from rrp.policies.features.featurizer import PolicyInput
    pi = PolicyInput({}, {}, np.zeros((1, 2)), np.zeros(1), np.zeros((0, 5)), np.zeros((0, 4)), {}, np.zeros(1))
    b = pickle.dumps(dict(x=pi), protocol=0).replace(PolicyInput.__module__.encode(), b"rrp.data.features")
    assert b"rrp.data.features" in b
    p = tmp_path / "e.public.pkl.gz"
    p.write_bytes(gzip.compress(b))
    assert isinstance(read_episode(p)["x"], PolicyInput)


# ------------------------------------------------------------------ D-140 S4: Policy adapters reproduce the old paths
class _Rec:
    def __init__(self):
        self.cmds, self.qpos, self.packets, self.chunks = [], [], [], []

    def on_step(self, i, env, act, step):
        c = act.command
        if isinstance(c, dict):
            self.cmds.append({f"{r}:{k}": np.asarray(v) for r, nc in sorted(c.items()) for k, v in nc.groups.items()})
        elif c is not None:
            self.cmds.append({k: np.asarray(v) for k, v in c.groups.items()})
        self.qpos.append(step.qpos)
        if act.packet is not None:
            self.packets.append(act.packet)
        if act.chunk is not None:
            self.chunks.append(act.chunk)


def _run(make, policy, task, ticks, dt=0.05):
    from rrp.harness.rollout import rollout
    from rrp.tasks.spec import get_task
    rec = _Rec()
    rollout(make, policy, get_task(task), [3], batch=1, max_seconds=ticks * dt - 1e-9, hooks=[rec])
    return rec


def test_teacher_adapter_matches_old_path(golden):
    from rrp.policies.base import make_policy
    rec = _run(lambda sd: _arm_session(sd), make_policy("teacher:pick_place"), "pick_place", 25)
    assert len(rec.cmds) == 25
    golden("arm.teacher.commands", _h(*rec.cmds))
    golden("arm.teacher.qpos", _h(np.stack(rec.qpos)))


def test_dual_teacher_adapter_matches_old_path(golden):
    from rrp.envs.base import make_env
    from rrp.policies.base import make_policy
    rec = _run(lambda sd: make_env("mujoco/dual", task="support_insert", body=["parm5l_pg2", "parm6_pg2"], seed=sd),
               make_policy("teacher:support_insert"), "support_insert", 15)
    golden("dual.teacher.commands", _h(*rec.cmds))
    golden("dual.teacher.qpos", _h(np.stack(rec.qpos)))


def test_latent_adapter_matches_old_path(golden):
    import torch
    from rrp.policies.latent import LatentPolicy, LatentStackPolicy
    from rrp.policies.system0 import LatentRealizer
    si = LatentPolicy(_flow(), knot_times=(0.1, 0.3, 0.5, 0.7), latent_space_version="ls-g", realizer_compat_version="rz-g",
                      device="cpu", nfe=4, seed=3)
    torch.manual_seed(1)
    pol = LatentStackPolicy(si, LatentRealizer(8, width=32, layers=1))
    rec = _run(lambda sd: _arm_session(sd), pol, "pick_place", 6)
    assert len(rec.packets) == 1 and pol.s0[0].stats.ticks == 6
    golden("latent.packet.z", _h(rec.packets[0].z, decimals=5))
    golden("latent.system0.commands", _h(*rec.cmds, decimals=5))
    golden("latent.system0.qpos", _h(np.stack(rec.qpos), decimals=6))


def test_bc_adapter_matches_old_path(golden):
    import torch
    from rrp.policies.bc import BCPolicy, LearnedPolicy
    from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
    torch.manual_seed(2)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=8, latent_dim=1))
    with torch.no_grad():
        m.out.weight.normal_(0, 0.2)
    rec = _run(lambda sd: _arm_session(sd), BCPolicy(LearnedPolicy(m, None, "cpu", nfe=4, execute_prefix=4, seed=5)),
               "pick_place", 6)
    assert len(rec.chunks) == 2                       # prefix 4: a new chunk at ticks 0 and 4
    golden("bc.chunk", _h({g.group: g.values for g in rec.chunks[0].command_groups}, decimals=5))


# ---------------------------------------------------------------- S5: episode-level goldens of the ported eval loops
def _tiny_bc():
    import torch
    from rrp.policies.bc import LearnedPolicy
    from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
    torch.manual_seed(2)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=8, latent_dim=1))
    with torch.no_grad():
        m.out.weight.normal_(0, 0.2)
    return LearnedPolicy(m, None, "cpu", nfe=4, execute_prefix=4, seed=5)


def _ep_key(seed, outcome, priv, pub, steps, calls, chunk_rej, cmd_rej, sim_time, events, fell):
    return np.asarray([seed, outcome, bool(priv), bool(pub), steps, calls, chunk_rej, cmd_rej, round(sim_time, 6),
                       sorted(events.items()), bool(fell)], dtype=object)


def test_arm_eval_loop_episodes(golden):
    """Recorded from harness.eval.runner.evaluate before S5 deleted it; the rollout port (BCPolicy + arm hooks) must
    reproduce outcomes (timeout x2, infeasible seed 50), chunk counts, rejections, events and sim time."""
    from rrp.harness.eval.hooks import arm_hooks
    from rrp.harness.eval.evaluate import evaluate
    from rrp.policies.bc import BCPolicy
    eps = evaluate(BCPolicy(_tiny_bc()), "mujoco/arm", "pick_place", "parm5_pg2", [3, 4, 50],
                   scene=lambda sd: {"n_distractors": sd % 3}, max_steps=6, batch=3, hooks=arm_hooks())
    rows = [_ep_key(e.seed, e.outcome, e.success_privileged, e.success_public, e.steps, e.metrics["chunks"],
                    e.metrics["chunk_rejections"], e.metrics["command_rejections"], e.metrics["sim_time"],
                    e.metrics["events"], e.failure_reason == "dropped_off_table") for e in eps]
    golden("loop.arm.bc", _h(*rows))


def test_oracle_encoder_is_the_former_inline_encoder():
    """D-140 S4: the one oracle encoder (policies.oracle.encode_demos) equals the former single-item inline code of
    latent_semantic_edits.OracleSource, and batching does not change an item's z (the ladder oracle batches)."""
    import torch
    from rrp.policies.features.featurizer import featurizer_for
    from rrp.policies.nets.batch import collate_inputs
    from rrp.policies.nets.semantic_latent import LatentConfig, TargetEncoder, assembly_tokens
    from rrp.policies.oracle import ShadowTeacher, encode_demos
    torch.manual_seed(4)
    cfg = LatentConfig(width=32, heads=2, ctx_layers=1, enc_layers=1, dz=8, horizon=6)
    E = TargetEncoder(cfg).eval()
    items = []
    for sd in (3, 4):
        s = _arm_session(sd)
        f = featurizer_for(s)
        items.append((f, f(s.observe()), ShadowTeacher(s).lookahead(s, 6)[:4 + sd - 3]))   # 4 and 5 demo rows (padded)
    f, pi, cmds = items[0]
    H, n = 6, len(cmds)
    with torch.no_grad():                                  # verbatim copy of the pre-S4 OracleSource.packet encoding
        a = f.aspace.normalize(cmds + [cmds[-1]] * (H - n), pi.q0).astype(np.float16).astype(np.float32)
        b = collate_inputs([pi])
        af, am, ai = assembly_tokens(b)
        N = b.node_feats.shape[1]
        at = np.zeros((1, H, N), np.float32); vt = np.zeros((1, H, N), bool)
        at[0, :, :a.shape[1]] = a; vt[0, :n, :a.shape[1]] = True
        mu, _ = E(b, torch.from_numpy(at), torch.from_numpy(vt), af, am, ai)
        ref = mu[0, :, :int(am[0].sum())].float().numpy()
    one = encode_demos(E, items[:1], H, "cpu")[0][0]
    both = encode_demos(E, items, H, "cpu")[0]
    assert np.array_equal(one, ref)
    assert np.allclose(both[0], ref, atol=1e-5) and both[1].shape == ref.shape


def _tiny_latent(dual=False):
    import torch
    from rrp.policies.latent import DualLatentPolicy, LatentPolicy
    from rrp.policies.nets.probes import ReadoutProbe
    from rrp.policies.system0 import LatentRealizer
    cls = DualLatentPolicy if dual else LatentPolicy
    si = cls(_flow(), knot_times=(0.1, 0.3, 0.5, 0.7), latent_space_version="ls-g", realizer_compat_version="rz-g",
             device="cpu", nfe=4, seed=3)
    torch.manual_seed(1)
    R = LatentRealizer(8, width=32, layers=1)
    torch.manual_seed(4)
    # D-144 R1 follow-up (archived research/tracks/rel-r1c.md): PacketProbe -> ReadoutProbe(probes:arm-packet-v1). Same seed,
    # same module names/parameter creation order (docs/relations.md 4, nets/probes.py docstring) -> byte-identical
    # state dict / outputs to the retired PacketProbe, so this golden's recorded hashes are unaffected (verified).
    return si, R, ReadoutProbe(8, 4, specs=["preset:probes:arm-packet-v1"], width=32, heads=2).eval()


def _probe_key(d):
    return sorted((q, round(float(x), 5), int(n)) for q, (x, n) in d.items())


def test_arm_latent_eval_loop_episodes(golden):
    """Recorded from harness.eval.latent_eval.evaluate_latent before S5b deleted it (tiny flow + realizer + probe, seeds
    3 and infeasible 50, 10 ticks = 2 packets); the port (LatentStackPolicy + latent_hooks) must reproduce outcomes,
    packet/system-0 counts, probe sums, events, sim time and object displacement."""
    from rrp.harness.eval.hooks import latent_hooks
    from rrp.harness.eval.hooks import arm_scene
    from rrp.harness.eval.evaluate import evaluate
    from rrp.policies.latent import LatentStackPolicy
    si, R, P = _tiny_latent()
    pol = LatentStackPolicy(si, R)
    eps = evaluate(pol, "mujoco/arm", "pick_place", "parm5_pg2", [3, 50], scene=arm_scene, max_steps=10, batch=2,
                   hooks=latent_hooks(pol, P))
    rows = [np.asarray([_ep_key(e.seed, e.outcome, e.success_privileged, e.success_public, e.steps, e.metrics["packets"],
                                e.metrics["packet_rejections"], e.metrics["fallback_holds"], e.metrics["sim_time"],
                                e.metrics["events"], False).tolist(),
                        e.metrics["system0_ticks"], _probe_key(e.metrics["probe_counts"]),
                        sorted((b, round(d, 6)) for b, d in e.metrics["final_disp_m"].items()),
                        e.metrics["moved"]], dtype=object) for e in eps]
    golden("loop.arm.latent", _h(*rows))


def test_dual_latent_eval_loop_episodes(golden):
    """Recorded from harness.eval.dual_latent_eval.evaluate_dual_latent before S5b deleted it (tiny dual stack, seeds 3
    and infeasible 2, 10 ticks, with and without the swap_slots edit); the port (LatentStackPolicy + dual_latent_hooks)
    must reproduce outcomes after settling, counts, per-slot and slot-swapped probe sums, events and sim time."""
    from rrp.harness.eval.hooks import dual_latent_hooks
    from rrp.harness.eval.evaluate import evaluate
    from rrp.policies.latent import LatentStackPolicy
    for edit in (None, "swap_slots"):
        si, R, P = _tiny_latent(dual=True)
        pol = LatentStackPolicy(si, R)
        eps = evaluate(pol, "mujoco/dual", "support_insert", "parm5l_pg2__parm6_pg2", [3, 2], max_steps=10, batch=2,
                       hooks=dual_latent_hooks(pol, "support_insert", P, packet_edit=edit))
        rows = [np.asarray([_ep_key(e.seed, e.outcome, e.success_privileged, e.success_public, e.steps,
                                    e.metrics["packets"], e.metrics["packet_rejections"], e.metrics["fallback_holds"],
                                    e.metrics["sim_time"], e.metrics["events"], False).tolist(),
                            e.metrics["system0_ticks"], _probe_key(e.metrics.get("probe_counts", {})),
                            _probe_key(e.metrics.get("probe_counts_slotswap", {}))], dtype=object) for e in eps]
        golden(f"loop.dual.latent.{edit}", _h(*rows))


def _row_h(row) -> str:
    """Digest of a result row dict (floats rounded to 5 decimals; key order irrelevant)."""
    import json as _j

    def rnd(x):
        if isinstance(x, dict):
            return {str(k): rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return round(float(x), 5)
        if isinstance(x, np.generic):
            return x.item()
        return x
    return hashlib.sha256(_j.dumps(rnd(row), sort_keys=True).encode()).hexdigest()[:24]


def _edit_sources(dual=False):
    import torch
    from rrp.harness.eval import latent_semantic_edits as se
    from rrp.policies.nets.semantic_latent import LatentConfig, TargetEncoder
    si, R, P = _tiny_latent(dual=dual)
    torch.manual_seed(4)
    lcfg = LatentConfig(width=32, heads=2, ctx_layers=1, enc_layers=1, dz=8, horizon=6)
    E = TargetEncoder(lcfg).eval()
    res = dict(latent_space_version="ls-g", realizer_compat_version="rz-g")
    srcs = dict(generated=se.GeneratedSource(si), teacher=se.TeacherSource(),
                oracle=se.OracleSource(E, lcfg, res, "tiny", "cpu"))
    if not dual:
        srcs["bc"] = se.BCSource(_tiny_bc(), "bc:tiny")
    return srcs, R, P


def test_semantic_edit_loop_rows(golden):
    """harness.eval.latent_semantic_edits.run_condition rows (every route x edit condition, seed 3, 9 ticks = 2 packets) must stay
    byte-identical through the S5 port onto harness.rollout."""
    from rrp.bodies.catalog import workbench_robots
    from rrp.harness.eval import latent_semantic_edits as se
    srcs, R, P = _edit_sources()
    robot = workbench_robots()["parm5_pg2"]()
    rows = []
    for name, src in srcs.items():
        for cond in se.PAIRED_CONDITIONS:
            rows.append(_row_h(se.run_condition(src, R, P, robot, "parm5_pg2", 3, cond, max_steps=9)))
    golden("loop.semantic_edits.arm", _h(np.asarray(rows, dtype=object)))


def test_arm_assignment_edit_loop_rows(golden):
    """run_arm_condition rows (dual assignment edits: every route x ARM_CONDITIONS, seed 3, 9 ticks), byte-identical."""
    from rrp.harness.eval import latent_semantic_edits as se
    srcs, R, P = _edit_sources(dual=True)
    rows = []
    for name, src in srcs.items():
        for cond in se.ARM_CONDITIONS:
            rows.append(_row_h(se.run_arm_condition(src, R, P, "parm5l_pg2__parm6_pg2", 3, cond, max_steps=9)))
    golden("loop.semantic_edits.dual", _h(np.asarray(rows, dtype=object)))


def test_grpo_episode_loop_rows(golden):
    """harness.train.latent_grpo.run_episodes rows (a repeated GRPO group seed, a teacher-prefix curriculum with a
    shared prefix, rewards with every shaping term) must stay byte-identical through the port onto harness.rollout."""
    from rrp.harness.train.latent_grpo import RewardConfig, run_episodes
    rw = RewardConfig(shaping_events=0.2, shaping_dist=0.3, shaping_reach=0.1)
    rows = []
    for prefix in (0, 4):
        si, R, _ = _tiny_latent()
        inner, n = si.packets, [0]

        def packets(sessions, _inner=inner, _n=n):          # deterministic stand-in for the SDE actor's records
            out = _inner(sessions)
            si.last_records = [dict(call=_n[0], j=j) for j in range(len(sessions))]
            _n[0] += 1
            return out
        si.packets = packets
        rows += [_row_h(r) for r in run_episodes(si, R, "parm5_pg2", [3, 3, 4], max_steps=10, prefix_steps=prefix,
                                                 reward=rw)]
    golden("loop.grpo.episodes", _h(np.asarray(rows, dtype=object)))


def test_train_drive_loop(golden):
    """harness.train.rollout.drive (GRPO/EXPO/adapt episode driver): lock-step SDE-actor chunks, a pause at a
    boundary and a resumed second call, per-state step allowances, chunk records, on_step observers, finalize."""
    import torch
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.envs.mujoco.session import Session
    from rrp.harness.train.flow_sde import SDEConfig
    from rrp.harness.train.rollout import EpisodeState, SDEPolicy, drive, finalize
    from rrp.policies.nets.flow import FlowPolicy, PolicyConfig
    torch.manual_seed(0)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=16))
    torch.nn.init.normal_(m.out.weight, std=0.05)
    pol = SDEPolicy(m, None, "cpu", SDEConfig(nfe=4), execute_prefix=4, version="t@0", seed=0)
    robot = workbench_robots()["parm5_pg2"]()
    sts = [EpisodeState(Session(BUILDERS["pick_place"](robot, sd, n_distractors=1), seed=sd), sd, n)
           for sd, n in ((3, 10), (4, 12), (5, 7))]
    trace = []
    drive(pol, sts, stop_fn=lambda st: st.seed == 4 and st.steps == 5,
          on_step=lambda st, r: trace.append((st.seed, st.steps, r.qpos.round(6).tolist())))
    assert sts[1].paused
    sts[1].paused = False
    drive(pol, sts, on_step=lambda st, r: trace.append((st.seed, st.steps, r.qpos.round(6).tolist())))
    # observation ids are "o<seed>_<step>_<n>" with n a process-global serial (differs with test order and with the
    # number of observe() calls, e.g. rollout's initial observation); the golden keeps the meaningful prefix
    oid = lambda o: o.rsplit("_", 1)[0]
    rows = [_row_h(dict(steps=st.steps, done=st.done, paused=st.paused, outcome=st.outcome, fell=st.fell,
                        calls=st.calls, rej=st.rejections, cmd_rej=st.cmd_rejections,
                        chunks=[(c["step"], c["executed_rows"], oid(c["observation_id"]), c["sim_time"])
                                for c in st.chunks], fin=finalize(st))) for st in sts]
    golden("loop.train.drive", _h(np.asarray(rows + [_row_h(trace)], dtype=object)))


# ---------------------------------------------------------------- relation-bias path (D-144 F1)
# The goldens above run with zero-init structural-bias weights, so they cannot see the relation path. These drive it
# with seeded NON-zero weights at every site, over the real arm + dual relations, in every legacy bias mode; recorded on
# the pre-registry code (StructuralBias / transform_relations) and kept byte-identical through the factor migration.
# `rewired` zeroes the act>act weights: the registry's one intended change is that `rewired` also rewires act>act.

def _rel_batch():
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.dual import DualSession
    from rrp.envs.mujoco.dual_scenarios import build_support_insert
    from rrp.policies.features.featurizer import featurizer_for
    from rrp.policies.features.multi import MultiFeaturizer
    from rrp.policies.nets.batch import collate_inputs
    s = _arm_session()
    W = workbench_robots()
    d = DualSession(build_support_insert([W["parm5l_pg2"](), W["parm6_pg2"]()], 3), seed=3)
    return collate_inputs([featurizer_for(s)(s.observe()), MultiFeaturizer(d.model, d.scenario.robots)(d.observe())])


def _randomize_bias(m, seed=11, zero_node=False):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in sorted(m.named_parameters()):
            if n.endswith("bias.w") or n.endswith("bias_c.w") or n.endswith("bias_e.w") or n.endswith("sb.w"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.7)
                if zero_node and n.endswith("bias_e.w"):
                    p.zero_()


def _rel_flow(batch, seed=0, **kw):
    import torch
    from rrp.policies.nets.flow import FlowPolicy
    m = FlowPolicy(_rel_cfg(**kw)).eval()
    _randomize_bias(m, zero_node=kw.get("bias_mode") == "rewired")
    gen = torch.Generator().manual_seed(7)
    cache = m.prepare(batch, rewire_gen=gen)
    B, N = cache.node_mask.shape
    g2 = torch.Generator().manual_seed(9)
    z = torch.randn((B, 4, N, 3), generator=g2)
    with torch.no_grad():
        v = m.velocity(z, torch.full((B,), 0.3), cache)
    return _h(cache.ctx, v, *[a for a in cache.act_bias if a is not None], decimals=6)


def _rel_cfg(**kw):
    import torch
    from rrp.policies.nets.flow import PolicyConfig
    torch.manual_seed(0)
    return PolicyConfig.from_dict(dict(dict(width=32, heads=2, ctx_layers=2, blocks=2, horizon=4, latent_dim=3), **kw))


@pytest.mark.parametrize("mode", ["true", "none", "zero", "reversed", "rewired"])
def test_relation_bias_modes(golden, mode):
    golden(f"rel.flow.{mode}", _rel_flow(_rel_batch(), bias_mode=mode))


def test_relation_bias_unstructured_and_slots(golden):
    b = _rel_batch()
    golden("rel.flow.unstructured.true", _rel_flow(b, structured=False))
    golden("rel.flow.unstructured.none", _rel_flow(b, structured=False, bias_mode="none"))
    golden("rel.flow.slot_handles", _rel_flow(b, slot_handles=True))


def test_relation_bias_latent_path_and_encoder(golden):
    import torch
    from rrp.policies.nets.latent_batch import assembly_batch
    from rrp.policies.nets.semantic_latent import LatentConfig, TargetEncoder, assembly_tokens
    b = _rel_batch()
    golden("rel.flow.latent_assemblies", _rel_flow(assembly_batch(b)))
    torch.manual_seed(0)
    E = TargetEncoder(LatentConfig(width=32, heads=2, ctx_layers=2, enc_layers=1, knots=2, dz=4, horizon=4)).eval()
    _randomize_bias(E)
    B, N = b.node_feats.shape[:2]
    a = torch.randn((B, 4, N), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        mu, lv = E(b, a, torch.ones(B, 4, N, dtype=torch.bool), *assembly_tokens(b))
    golden("rel.target_encoder", _h(mu, lv, decimals=6))


def test_relation_bias_psi0_dims(golden):
    import torch
    from rrp.policies.psi0 import nets as N
    torch.manual_seed(0)
    morph, enc = N.Morph(), N.DimEncoder(32, 2, 2).eval()
    _randomize_bias(enc)
    state = torch.randn((2, 40), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        golden("rel.psi0.dims", _h(enc(morph, state), decimals=6))
