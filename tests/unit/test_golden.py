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
    from rrp.envs.fixtures import make_pick_place_session
    return make_pick_place_session(seed=seed)


def test_arm_featurizer_teacher_physics(golden):
    from rrp.features.featurizer import featurizer_for
    from rrp.teachers.arm_smooth import make_arm_teacher
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
    from rrp.models.flow import FlowPolicy, PolicyConfig
    torch.manual_seed(seed)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=K, latent_dim=dz, aux=False))
    with torch.no_grad():
        m.out.weight.normal_(0, 0.2)
        m.set_target_norm(torch.linspace(-1, 1, dz), torch.linspace(0.5, 2, dz))
    return m.eval()


def test_arm_latent_system_i_and_system0(golden):
    import torch
    from rrp.controllers.latent_runner import LatentPolicy
    from rrp.controllers.latent_realizer import LatentRealizer, LatentSystem0
    from rrp.features.featurizer import featurizer_for
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
    from rrp.controllers.policy_runner import LearnedPolicy
    from rrp.models.flow import FlowPolicy, PolicyConfig
    s = _arm_session()
    torch.manual_seed(2)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=8, latent_dim=1, aux=False))
    with torch.no_grad():
        m.out.weight.normal_(0, 0.2)
    pol = LearnedPolicy(m, None, "cpu", nfe=4, execute_prefix=4, seed=5)
    ch = pol.chunks([s])[0]
    golden("bc.chunk", _h({g.group: g.values for g in ch.command_groups}, decimals=5))


def test_dual_featurizer_and_teacher(golden):
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.dual import DualSession
    from rrp.envs.dual_scenarios import build_support_insert
    from rrp.features.multi import MultiFeaturizer
    from rrp.teachers.dual import TEACHERS
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
    from rrp.contracts.latent_action import LatentActionChunk
    p = LatentActionChunk(latent_space_version="ls", realizer_compat_version="rz",
                          z=np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 7, knot_times=[0.1, 0.3],
                          assemblies=[dict(handle=f"asm:{'0' * 16}:r/{i}", robot_index=0) for i in range(3)],
                          assembly_mask=[True, True, False], entity_registry=[dict(handle="ent:0")],
                          observation_id="o", graph_version=1, runtime_version=2, robot_spec_hash="h",
                          generated_at=1.0, valid_from=1.0, valid_until=1.8, source="learned", policy_version="v")
    golden("packet.bytes", hashlib.sha256(p.to_bytes()).hexdigest()[:24])


def _saved_actor(tmp, name, obs_dim, act_dim, meta, seed):
    import torch
    from rrp.envs.tracker_nets import mlp
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
    from rrp.envs.legged_core import LeggedEnv
    from rrp.envs.legged_tracker import LearnedTracker
    body = "phum_3"                     # biped: ref_ff and clock_gate apply
    env = LeggedEnv(lambda: legged_body(body), 1, 5)
    b = env.b
    extra = 5 if case == "extra_obs" else 0
    if case == "morph_v1":
        from rrp.envs.morph_obs import NS, OBS_DIM
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
    from rrp.evaluation.ladder import wilson
    for n in range(0, 61):
        for k in range(0, n + 1):
            assert wilson(k, n) == _old_ladder_wilson(k, n)
            assert wilson(k, n, 1.959964) == _old_ladder_wilson(k, n, 1.959964)
    assert type(wilson(3, 10)) is tuple


def test_legacy_pickle_paths_load(tmp_path):
    """Dataset pickles written before D-140 name `rrp.data.features.PolicyInput`; read_episode remaps them."""
    import gzip
    import pickle
    from rrp.data.collect import read_episode
    from rrp.features.featurizer import PolicyInput
    pi = PolicyInput({}, {}, np.zeros((1, 2)), np.zeros(1), np.zeros((0, 5)), np.zeros((0, 4)), {}, np.zeros(1))
    b = pickle.dumps(dict(x=pi), protocol=0).replace(PolicyInput.__module__.encode(), b"rrp.data.features")
    assert b"rrp.data.features" in b
    p = tmp_path / "e.public.pkl.gz"
    p.write_bytes(gzip.compress(b))
    assert isinstance(read_episode(p)["x"], PolicyInput)
