"""RP1 (D-146, docs/architecture.md 14.1): `run_ladder` runs on `harness.rollout` + hooks. The rows of every rung
(R0 teacher, R1 oracle, R2 generated, plain BC `learned`, the BC-look-ahead oracle) on the smallest procedural arm
(random-weight nets: plumbing only, no number is a result) were recorded from the private ladder loop BEFORE the port
(`loop.ladder.*` in tests/data/golden.json) and must stay byte-identical (floats rounded to 5 decimals)."""
import hashlib
import json

import numpy as np
import pytest

from tests.unit.test_golden import _flow, _tiny_bc, golden  # noqa: F401  (golden is the recording fixture)

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

ROBOT = "parm5_pg2"


def _digest(x) -> str:
    def rnd(v):
        if isinstance(v, dict):
            return {str(k): rnd(w) for k, w in v.items()}
        if isinstance(v, (list, tuple)):
            return [rnd(w) for w in v]
        if isinstance(v, np.ndarray):
            return rnd(v.tolist())
        if isinstance(v, (float, np.floating)):
            return round(float(v), 5)
        if isinstance(v, np.generic):
            return v.item()
        return v
    return hashlib.sha256(json.dumps(rnd(x), sort_keys=True, default=str).encode()).hexdigest()[:24]


def _models(with_flow=False, with_bc=False):
    from rrp.policies.latent import LatentPolicy
    from rrp.policies.nets.probes import ReadoutProbe
    from rrp.policies.nets.semantic_latent import LatentConfig, TargetEncoder
    from rrp.policies.system0 import LatentRealizer
    torch.manual_seed(4)
    lcfg = LatentConfig(width=32, heads=2, ctx_layers=1, enc_layers=1, dz=8, horizon=6)
    E = TargetEncoder(lcfg).eval()
    torch.manual_seed(1)
    R = LatentRealizer(8, width=32, layers=1)
    P = ReadoutProbe(8, 4, specs=["preset:probes:arm-packet-v1"], width=32, heads=2).eval()
    res = dict(latent_space_version="ls-g", realizer_compat_version="rz-g")
    flow = None
    if with_flow:
        flow = LatentPolicy(_flow(), knot_times=(0.1, 0.3, 0.5, 0.7), latent_space_version="ls-g",
                            realizer_compat_version="rz-g", device="cpu", nfe=4, seed=3)
        flow.noise_scale = 1.0
    return dict(E=E, R=R, P=P, lcfg=lcfg, res=res, flow=flow, learned=_tiny_bc() if with_bc else None)


def _rows(rows) -> list:
    return [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]


def _run(cfg, models, **kw):
    from rrp.harness.eval.ladder import run_ladder
    return _rows(run_ladder(cfg, None, models, {}, **kw))


def _cfg(route, seeds=(3, 4), **kw):
    from rrp.harness.eval.ladder import LadderConfig
    return LadderConfig(route=route, robot=ROBOT, seeds=list(seeds), max_steps=kw.pop("max_steps", 10),
                        keep_ticks=kw.pop("keep_ticks", True), **kw)


def test_ladder_teacher_route(golden):
    log: dict = {}
    rows = _run(_cfg("teacher", prev_action="own", object_shift=(4, 0.02, 0.0)),
                dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None, learned=None), cmd_log=log)
    golden("loop.ladder.teacher", _digest([rows, {k: [None if c is None else {g: c[g] for g in sorted(c)} for c in v]
                                                  for k, v in sorted(log.items())}]))


def test_ladder_teacher_route_to_the_end(golden):
    """A whole teacher episode: the shadow FSM finishes, the loop settles one tick and judges."""
    rows = _run(_cfg("teacher", seeds=(3,), max_steps=300, keep_ticks=False),
                dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None, learned=None))
    golden("loop.ladder.teacher_full", _digest(rows))


def test_ladder_teacher_route_perturbed(golden):
    from rrp.envs.mujoco.perturb import PhysicsPerturbation
    rows = _run(_cfg("teacher", seeds=(3,), perturb=PhysicsPerturbation(mass_scale=1.2, friction_scale=0.8,
                                                                        latency_ms=100.0)),
                dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None, learned=None))
    golden("loop.ladder.teacher_perturbed", _digest(rows))


def test_ladder_oracle_route(golden):
    collect: dict = {}
    rows = _run(_cfg("oracle", prev_action="own", oracle_reanchor=True), _models(), collect=collect)
    buf = [np.asarray(collect[k]).astype(np.float32) for k in ("mu", "lv")]
    golden("loop.ladder.oracle", _digest([rows, [b.tolist() for b in buf],
                                          [[r[0], r[1], r[3], r[5].tolist(), r[4].tolist(), r[2].tolist()]
                                           for r in collect["rows"]]]))


def test_ladder_generated_route(golden):
    rows = _run(_cfg("generated", chunk_blend="crossfade"), _models(with_flow=True))
    golden("loop.ladder.generated", _digest(rows))


def test_ladder_learned_and_bc_oracle_routes(golden):
    rows = _run(_cfg("learned", seeds=(3,), replan_ticks=4), _models(with_bc=True))
    rows += _run(_cfg("oracle", seeds=(3,), oracle_expert="bc", replan_ticks=4), _models(with_bc=True))
    golden("loop.ladder.learned_bc", _digest(rows))


def test_run_ladder_is_a_rollout(monkeypatch):
    """No private loop: run_ladder calls harness.rollout with the ladder's hooks (the shift hook only when asked)."""
    import rrp.harness.rollout as R
    seen = []
    real = R.rollout

    def spy(make_env, policy, task, seeds, **kw):
        seen.append((type(policy).__name__, [type(h).__name__ for h in kw["hooks"]], kw["max_steps"], kw["batch"]))
        return real(make_env, policy, task, seeds, **kw)
    monkeypatch.setattr(R, "rollout", spy)
    _run(_cfg("teacher", seeds=(3,), max_steps=3, keep_ticks=False, object_shift=(1, 0.01, 0.0)),
         dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None, learned=None), cmd_log={})
    assert seen == [("LadderPolicy", ["LadderTrace", "MotionRecord", "PrevAction", "CommandLog", "ObjectShift"], 3, 1)]


def test_hooks_registry_names_every_ladder_hook():
    from rrp.harness.eval import hooks as H
    assert {"feasibility", "session_record", "settle", "displacement", "object_shift", "motion", "command_log",
            "prev_action", "frame"} <= set(H.HOOKS)
    assert isinstance(H.HOOKS["object_shift"](2, 0.1, 0.0), H.ObjectShift)
    assert isinstance(H.HOOKS["settle"](), H.Settle) and isinstance(H.HOOKS["prev_action"](), H.PrevAction)
    assert isinstance(H.HOOKS["feasibility"](), H.Feasibility)


def test_object_shift_moves_the_cube_at_the_start_of_its_tick_and_not_after_the_episode_ended():
    from types import SimpleNamespace as NS
    from rrp.harness.eval.hooks import ObjectShift

    class Env:
        def __init__(self):
            self.data = NS(xpos={7: np.zeros(3)})
            self.model = NS(body=lambda n: NS(id=7))
            self.moved = []

        def teleport_object(self, body, p, source):
            self.moved.append((body, p.tolist(), source))
    alive = {0: True, 1: False}
    h = ObjectShift(2, 0.1, -0.2, alive=lambda i: alive[i])
    envs = [Env(), Env()]
    for i, e in enumerate(envs):
        h.on_reset(i, e, None)
    for _ in range(2):
        for i, e in enumerate(envs):
            h.on_step(i, e, None, None)
    assert envs[0].moved == [("cube", [0.1, -0.2, 0.0], "ladder_disturbance")] and envs[1].moved == []
    e0 = Env()
    ObjectShift(0, 0.1, 0.0).on_reset(0, e0, None)          # tick 0: at reset, before the first act
    assert len(e0.moved) == 1
