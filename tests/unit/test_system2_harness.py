"""System II harness (rrp.evaluation.system2, D-126 #31): target -> legged public context math, labels, default-off."""
import numpy as np
import pytest

from rrp.harness.eval import system2 as s2


def _ctx():
    return np.arange(22, dtype=np.float32) + 1.0          # distinct entries; event one-hot slot replaced below


def test_apply_target_math_on_known_vectors():
    c = _ctx()
    h = s2.apply_target(c, s2.System2Target(source="mock:t", halt=True))
    assert h is not c and np.array_equal(c, _ctx())       # pure: input untouched
    assert h[16:20].tolist() == [0, 0, 1, 0] and np.array_equal(h[:16], c[:16]) and np.array_equal(h[20:], c[20:])
    m = s2.apply_target(c, s2.System2Target(source="mock:t", mirror_goal=True))
    exp = c.copy(); exp[9] = -exp[9]; exp[13] = -exp[13]
    assert np.array_equal(m, exp)
    t = s2.System2Target(source="mock:t", order=("cyan", "orange"))
    assert s2.apply_target(c, t) is c                     # order enters via the task graph by default
    w = s2.apply_target(c, t, binding_in_ctx=True)
    assert np.array_equal(w[8:12], c[12:16]) and np.array_equal(w[12:16], c[8:12])
    assert np.array_equal(w[:8], c[:8]) and np.array_equal(w[16:], c[16:])
    assert s2.apply_target(c, s2.System2Target(source="mock:t", order=s2.DECLARED_ORDER), binding_in_ctx=True) is c
    b = s2.apply_target(np.stack([c, c]), s2.System2Target(source="mock:t", halt=True))       # batched [B, 22]
    assert (b[:, 16:20] == np.array([0, 0, 1, 0])).all()


def test_apply_target_matches_legged_eval_edits(monkeypatch):
    torch = pytest.importorskip("torch")
    lle = pytest.importorskip("rrp.policies.legged")
    monkeypatch.setattr(lle, "public_context", lambda s, osc: _ctx())

    class _Ad:
        s = None

        def osc(self):
            return 0.0

    fake = type("C", (), {"dev": torch.device("cpu")})()
    for mode, tgt in (("halt", dict(halt=True)), ("mirror_goal", dict(mirror_goal=True))):
        ref = lle.LatentLeggedController._ctx(fake, _Ad(), mode)[0].numpy()
        assert np.array_equal(ref, s2.apply_target(_ctx(), s2.System2Target(source="mock:t", **tgt))), mode


def test_default_off_is_identity():
    c = _ctx()
    assert s2.make_system2("off") is None and s2.make_system2(None) is None
    assert s2.apply_target(c, None) is c
    p = s2.System2ContextProvider(None)
    assert not p.active and p(c) is c and p.record() is None
    torch = pytest.importorskip("torch")
    tc = torch.from_numpy(c)[None]
    assert p(tc) is tc
    assert s2.System2ContextProvider(s2.System2Target(source="default:x", order=s2.DECLARED_ORDER))(tc) is tc
    sc, prov, rec = s2.system2_episode(None, "go2", 10000)
    assert sc is None and rec is None and not prov.active


def test_mock_grounding_to_target_to_context():
    torch = pytest.importorskip("torch")
    m = s2.MockSystem2("halt_on_stop", fn=lambda text, img: dict(order=("cyan", "orange"), halt="stop now" in text))
    t = m.ground("walk to cyan then orange")
    assert t.source == "mock:halt_on_stop" and t.order == ("cyan", "orange") and not t.halt and not t.diagnostic
    assert t.binding() == ("cyan", "orange") and not t.context_edits()
    t2 = m.ground("stop now")
    prov = s2.System2ContextProvider(t2)
    out = prov(torch.from_numpy(_ctx())[None])
    assert isinstance(out, torch.Tensor) and out.shape == (1, 22) and out[0, 16:20].tolist() == [0, 0, 1, 0]
    assert prov.record()["source"] == "mock:halt_on_stop" and prov.record()["harness_version"] == "s2h-1"
    assert m.calls == [("walk to cyan then orange", False), ("stop now", False)]


def test_oracle_is_labelled_diagnostic_and_only_it_gets_truth():
    o = s2.make_system2("oracle")
    t = o.ground("go to cyan first", privileged_truth=("cyan", "orange"))
    assert t.diagnostic and t.source.startswith("oracle_diagnostic") and t.info.get("privileged")
    with pytest.raises(ValueError):
        o.ground("x")                                     # no truth -> refuses
    with pytest.raises(ValueError):
        s2.System2Target(source="oracle_x", order=("cyan", "orange"))    # oracle label without diagnostic flag
    with pytest.raises(ValueError):
        s2.System2Target(source="")
    seen = []

    class Spy(s2.MockSystem2):
        def ground(self, instruction, image=None, *, privileged_truth=None):
            seen.append(privileged_truth)
            return super().ground(instruction, image)
    s2._ground(Spy(order=("orange", "cyan")), "x", None, ("cyan", "orange"))
    assert seen == [None]                                 # non-oracle systems never receive the truth
    d = s2.make_system2("default").ground("go to cyan first")
    assert d.source == "default:declared_task_binding" and d.order == s2.DECLARED_ORDER and not d.diagnostic


def test_vlm_adapter_with_injected_model_and_ground_eval():
    class FakeVLM:                                        # stands in for rrp.models.system2_vlm.System2 (no weights)
        def run(self, images, texts):
            sc = np.array([1.0 if t.index("orange") < t.index("cyan") else -1.0 for t in texts])
            return dict(score=sc, feat=np.zeros((len(texts), 4)), gen=["x"] * len(texts), top1=["x"] * len(texts))

    v = s2.VLMSystem2(ckpt="psi0_test", model=FakeVLM())
    assert v.source == "learned:psi0_test#zero_shot" and v.needs_image and not v.needs_truth
    assert s2.make_system2("vlm:/weights/psi0").source == "learned:/weights/psi0#zero_shot"   # lazy: nothing loaded
    img = np.zeros((4, 4, 3), np.uint8)
    items = []
    for k, (a, b) in enumerate([("orange", "cyan"), ("cyan", "orange")] * 3):
        text, truth, ti = s2.make_instruction(0, "color", {}, template=k % 4)
        text = s2.TEMPLATES["color"][ti].format(a=a, b=b)
        items.append(dict(split="test", seed=k, family="color", template=ti, held_out_template=False, text=text,
                          truth=[a, b], img=img))
    res = ground_res = s2.ground_eval(v, items, log=None)
    assert ground_res["summary"]["per_family"]["color"]["acc_test"] == 1.0
    assert all(r["system2_source"] == v.source and r["harness_version"] == "s2h-1" for r in res["rows"])
    assert "correct_blank" in res["rows"][0]
    d = s2.ground_eval(s2.DefaultSystem2(), [dict(i, img=None) for i in items], log=None)
    assert d["summary"]["per_family"]["color"]["acc_test"] == 0.5


@pytest.mark.menagerie                  # builds go2 (Menagerie assets; skipped in a fresh clone)
def test_episode_wiring_binds_public_graph_and_privileged_truth():
    pytest.importorskip("mujoco")
    sd = next(s for s in range(10000, 10040) if s2.make_instruction(s, "color", {})[1][0] == "cyan")
    sc, prov, rec = s2.system2_episode(s2.DefaultSystem2(), "go2", sd)
    assert rec["truth"] == ["cyan", "orange"] and rec["binding"] == ["orange", "cyan"] and not rec["binding_correct"]
    ent = {o.sim_body: o.task_entity for o in sc.objects}
    assert ent == {"waypoint_b": "waypoint_a", "waypoint_a": "waypoint_b"}      # privileged evaluator follows truth
    sc, prov, rec = s2.system2_episode(s2.OracleSystem2(), "go2", sd)
    assert rec["binding_correct"] and rec["target"]["diagnostic"]
