"""RunConfig (W5): every legacy config round-trips unchanged; flags are required; derived out; overlays and matrices."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rrp.core.runconfig import (NON_RUN_CONFIGS, RunConfig, RunConfigError, RunIndex, expand_matrix,
                                     iter_legacy_configs, load_legacy, overlay, render)

ROOT = Path(__file__).resolve().parents[2]


def test_every_legacy_config_round_trips():
    paths = iter_legacy_configs(ROOT)
    assert len(paths) >= 250
    for p in paths:
        orig = json.loads(p.read_text())
        rc = load_legacy(p, ROOT)
        native = rc.to_native(RunIndex())
        assert native == orig and list(native) == list(orig), p
        again = RunConfig.model_validate_json(rc.to_json())      # serialised RunConfig keeps the meaning too
        assert again.to_native(RunIndex()) == orig, p
    others = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").rglob("*.json") if p not in paths)
    assert set(others) <= set(NON_RUN_CONFIGS)


def test_legacy_classification_and_flags():
    rc = load_legacy(ROOT / "configs/ladder/armseed2/sfjf2/rz_sfjf2_gendag1_noqd.json", ROOT)
    assert (rc.family, rc.stage, rc.variant, rc.seed) == ("arm", "refit", "semfix", 2761)
    assert rc.flags.realizer_drop_qd is True and rc.flags.zero_prev_action is True
    assert len(rc.inputs["dagger"]) == 51          # gendag1 omits gen1/ur5e_tf3: a plain ordered list, not a product
    rc2 = load_legacy(ROOT / "configs/ladder/armseed2/sfjf2/rz_sfjf2_gendag3_noqd.json", ROOT)
    assert rc2.inputs["dagger"].runs[0] == "runs/ladder_dagger_sfjf2_bc1" and len(rc2.inputs["dagger"].files) == 13
    assert rc.out == "artifacts/runs/ladder_rz_sfjf2_gendag1_noqd"
    rep = load_legacy(ROOT / "configs/ladder/armseed2/nsjf2/rep-ladder_latent_nosem_b1fix_anchor_s2.json", ROOT)
    assert (rep.stage, rep.variant) == ("train_rep", "nosem")
    # D-144 sweep-flags: `probe_lv_min` no longer applies to arm `train_rep` (retired from FLAG_SPEC; `latent.factors`
    # instead, codemodded onto this exact file -- see nets/semantic_latent.py / tests/unit/test_relations_r2_latent.py).
    assert rep.flags.probe_lv_min is None and "probe_lv_min" not in rep.legacy.absent_flags
    lg = load_legacy(ROOT / "configs/legged_fixsem/rep_fixsem_go2_s1.json", ROOT)
    assert (lg.family, lg.stage, lg.variant) == ("legged", "train_rep", "semfix")
    assert lg.flags.qd_dropout == 0.5 and lg.flags.zero_prev_action is None and lg.flags.contact_version == "contact_v1"


def _new(**kw):
    d = dict(schema_version="runconfig-1", family="arm", stage="refit", variant="semfix", seed=2, lineage="arm-semfix",
             tag="bcdag1", inputs={"representation": "runs/x:representation.pt"},
             flags=dict(zero_prev_action=True, realizer_anchor=True, realizer_drop_qd=False, probe_lv_min=None,
                        qd_dropout=None, contact_version="contact_v1"),
             params={"steps": 10})
    d.update(kw)
    return RunConfig.model_validate(d)


def test_flags_required_and_applicability():
    rc = _new()
    assert rc.out == "artifacts/runs/pipeline/arm-semfix/refit-bcdag1_s2"
    nat = rc.to_native(RunIndex({"runs/x": "artifacts/runs/legacy_x"}))
    assert nat["representation"] == "artifacts/runs/legacy_x/representation.pt"
    assert nat["zero_prev_action"] is True and nat["realizer_drop_qd"] is False and nat["out_dir"] == rc.out
    f = rc.flags.model_dump()
    with pytest.raises(Exception, match="must be stated"):
        _new(flags=dict(f, realizer_drop_qd=None))
    with pytest.raises(Exception, match="does not apply"):
        _new(flags=dict(f, qd_dropout=0.5))
    with pytest.raises(Exception):
        _new(flags={k: v for k, v in f.items() if k != "zero_prev_action"})      # missing field: no default
    with pytest.raises(Exception, match="must be set in flags"):
        _new(params={"zero_prev_action": False})
    with pytest.raises(Exception, match="derived"):
        _new(params={"out_dir": "elsewhere"})


def test_variant_must_match_recipe():
    # D-144 sweep-flags: arm `train_rep` variant checking is `latent.factors`-only now (`probe_lv_min` retired from
    # FLAG_SPEC[("arm", "train_rep")]; see core/runconfig.py::_check_variant and tests/unit/test_relations_r2_latent.py
    # for the full old-style-vs-factors-style coverage, incl. the legged-only old-style path).
    flags = dict(zero_prev_action=True, realizer_anchor=True, realizer_drop_qd=None, probe_lv_min=None,
                 qd_dropout=None, contact_version="contact_v1")
    base = dict(stage="train_rep", tag=None,
               params={"latent": {"factors": [{"name": "probe.arm.visible", "weight": 1.0}]}}, flags=flags)
    _new(**base, variant="sem")
    with pytest.raises(Exception, match="does not match"):
        _new(**base, variant="semfix")                       # semfix needs the bounded floor (> -8)
    with pytest.raises(Exception, match="does not match"):
        _new(**base, variant="nosem")


def test_overlay_matrix_render():
    assert overlay({"a": {"b": 1, "c": 2}, "l": [1]}, {"a": {"b": 3}, "l": [2], "x.y": 4}) == \
        {"a": {"b": 3, "c": 2}, "l": [2], "x": {"y": 4}}
    assert overlay({"a": 1, "b": 2}, {"a": None}) == {"b": 2}
    assert render("{1701 + 1000 * (seed - 1)}", {"seed": 2}) == 2701
    assert render("run_{v}_s{seed}", {"v": "sem", "seed": 3}) == "run_sem_s3"
    pts = expand_matrix({"params": {"seed": "{1706 + 1000*(seed-1)}"}}, {"variant": ["sem", "nosem"], "seed": [1, 2]},
                        per_value={"variant": {"nosem": {"params": {"w": 0}}}})
    assert [(p["variant"], p["seed"], p["params"]["seed"], p["params"].get("w")) for p in pts] == \
        [("sem", 1, 1706, None), ("sem", 2, 2706, None), ("nosem", 1, 1706, 0), ("nosem", 2, 2706, 0)]
    with pytest.raises(RunConfigError):
        render("{__import__('os')}", {})
