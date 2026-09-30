"""RunConfig (W5): flags are required; derived out; serialised round trip; overlays and matrices. (D-145 P2 retired the
legacy-config round trip with `load_legacy` and the `configs/` tree.)"""
from __future__ import annotations

import json

import pytest

from rrp.core.runconfig import RunConfig, RunConfigError, RunIndex, expand_matrix, overlay, render


def _new(**kw):
    d = dict(schema_version="runconfig-1", family="arm", stage="refit", variant="semfix", seed=2, lineage="arm-semfix",
             tag="bcdag1", inputs={"representation": "runs/x:representation.pt"},
             flags=dict(zero_prev_action=True, realizer_anchor=True, realizer_drop_qd=False, probe_lv_min=None,
                        qd_dropout=None, contact_version="contact_v1"),
             params={"steps": 10})
    d.update(kw)
    return RunConfig.model_validate(d)


def test_serialised_round_trip_keeps_the_frozen_legacy_key():
    """`legacy` is always null but stays in the serialised form: it is part of every existing run's config_hash."""
    rc = _new()
    d = json.loads(rc.to_json())
    assert d["legacy"] is None and RunConfig.model_validate_json(rc.to_json()).config_hash() == rc.config_hash()
    with pytest.raises(Exception):
        _new(legacy={"path": "x"})


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
