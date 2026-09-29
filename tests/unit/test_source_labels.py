"""D-126 canonical source labels (sl-1): readers accept both row formats, the default-off switch keeps rows
byte-identical, new writes are strict, and the widened wire Literals accept old and canonical values."""
from __future__ import annotations

import json
import typing

import numpy as np
import pytest

from rrp.core.action import ActionChunk, GroupCommand, LEGACY_SOURCES, NativeCommand
from rrp.core.latent_action import AssemblyHandle, LatentActionChunk
from rrp.core.provenance import (SOURCE_LABEL_VERSION, SOURCE_LABELS_ENV, Source, SourceLabel,
                                      canonical_source_labels, parse_legacy_source, row_source, stamp_source_label)

CANONICAL = [s.value for s in Source if s is not Source.UNKNOWN]
OLD_LATENT = ("learned", "target_encoder_oracle", "debug", "replay")


# ------------------------------------------------------------------ switch
def test_switch_default_off_and_env(monkeypatch):
    monkeypatch.delenv(SOURCE_LABELS_ENV, raising=False)
    assert canonical_source_labels() is False
    monkeypatch.setenv(SOURCE_LABELS_ENV, "legacy")
    assert canonical_source_labels() is False
    monkeypatch.setenv(SOURCE_LABELS_ENV, "canonical")
    assert canonical_source_labels() is True
    assert canonical_source_labels(False) is False            # explicit argument wins
    monkeypatch.setenv(SOURCE_LABELS_ENV, "canonicl")
    with pytest.raises(ValueError):
        canonical_source_labels()                               # typos are errors, not silently off


# ------------------------------------------------------------------ writers
def test_stamp_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv(SOURCE_LABELS_ENV, raising=False)
    row = dict(route="teacher", seed=0, source="scripted_teacher(privileged)", x=1.5)
    before = json.dumps(row)
    assert json.dumps(stamp_source_label(dict(row), "scripted_teacher", "privileged")) == before


def test_stamp_on_adds_only_two_keys_and_is_strict():
    row = dict(seed=0, source="learned(system-i flow)")
    out = stamp_source_label(dict(row), "learned", "runs/f/flow.pt", enabled=True)
    assert out == dict(row, source_label="learned:runs/f/flow.pt", source_label_version=SOURCE_LABEL_VERSION)
    with pytest.raises(ValueError):                             # learned needs a checkpoint
        stamp_source_label(dict(row), "learned", None, enabled=True)
    with pytest.raises(ValueError):                             # legacy alias is not a canonical kind
        stamp_source_label(dict(row), "teacher", enabled=True)
    with pytest.raises(ValueError):                             # unknown only for legacy reads
        stamp_source_label(dict(seed=0), "unknown", enabled=True)
    with pytest.raises(ValueError):                             # contradicts the legacy source
        stamp_source_label(dict(row), "oracle", "x", enabled=True)


def _teacher_ladder_row(**kw) -> dict:
    from rrp.harness.eval.ladder import LadderConfig, run_ladder
    cfg = LadderConfig(route="teacher", robot="parm5_pg2", seeds=[0], max_steps=4, compare_oracle=False, **kw)
    r = run_ladder(cfg, None, dict(E=None, R=None, P=None, lcfg=None, res=None, flow=None, learned=None), {})[0]
    r.pop("wall_s")
    return r


def test_ladder_row_switch_off_matches_pre_change_and_on_adds_label(monkeypatch):
    """A real 4-step teacher-route ladder rollout. Off: the same keys, key order and source string as before
    D-126 (captured from origin/main 733b02a). On: exactly the two sl-1 keys more, everything else identical."""
    monkeypatch.delenv(SOURCE_LABELS_ENV, raising=False)
    off = _teacher_ladder_row()
    assert list(off) == PRE_D126_LADDER_KEYS
    assert off["source"] == "scripted_teacher(privileged)"
    assert json.dumps(_teacher_ladder_row(source_labels=False), default=str) == json.dumps(off, default=str)
    on = _teacher_ladder_row(source_labels=True)
    assert on.pop("source_label") == "scripted_teacher:privileged"
    assert on.pop("source_label_version") == SOURCE_LABEL_VERSION
    assert json.dumps(on, default=str) == json.dumps(off, default=str)


def test_route_source_legacy_strings_unchanged():
    from rrp.harness.eval.ladder import LadderConfig, route_source
    c = lambda route, **kw: LadderConfig(route=route, robot="r", seeds=[0], **kw)
    assert route_source(c("teacher"))[0] == "scripted_teacher(privileged)"
    assert route_source(c("generated", flow="f.pt")) == ("learned(system-i flow)", "learned", "f.pt")
    assert route_source(c("learned", policy="p.pt", policy_label="bcx"))[0] == "learned:bcx"
    assert route_source(c("oracle"))[0] == "target_encoder_oracle(ORACLE DIAGNOSTIC: teacher future actions)"
    assert route_source(c("oracle", oracle_expert="bc", policy="p.pt"))[0] == \
        "target_encoder_oracle(ORACLE DIAGNOSTIC: chunk of learned:p.pt at the current state)"
    for route, kw in (("teacher", {}), ("oracle", {}), ("oracle", dict(oracle_expert="bc", policy="p.pt")),
                      ("generated", dict(flow="f.pt")), ("learned", dict(policy="p.pt"))):
        legacy, kind, detail = route_source(c(route, **kw))
        row = stamp_source_label(dict(source=legacy), kind, detail, enabled=True)   # strict + consistent
        assert row_source(row).kind is parse_legacy_source(legacy).kind


# ------------------------------------------------------------------ readers
@pytest.mark.parametrize("legacy,kind,detail", [
    ("scripted_teacher(privileged)", Source.SCRIPTED_TEACHER, None),
    ("learned(system-i flow)", Source.LEARNED, None),
    ("target_encoder_oracle(ORACLE DIAGNOSTIC: chunk of learned:x at the current state)", Source.ORACLE, None),
    ("learned:pol+edit:swap_slots", Source.LEARNED, "pol+edit:swap_slots"),
    ("scripted_teacher:pick_place_v2_minjerk", Source.SCRIPTED_TEACHER, "pick_place_v2_minjerk"),
    ("teacher", Source.SCRIPTED_TEACHER, None),
    ("bc:ck.pt", Source.BC, "ck.pt"),
])
def test_row_source_reads_legacy_rows(legacy, kind, detail):
    lab = row_source(dict(source=legacy))
    assert lab.kind is kind and lab.detail == detail


def test_row_source_prefers_canonical_and_handles_bad():
    row = dict(source="learned(system-i flow)", source_label="learned:runs/f.pt", source_label_version="sl-1")
    assert str(row_source(row)) == "learned:runs/f.pt"
    with pytest.raises(ValueError):
        row_source(dict(source="learned", source_label="learned"))      # sl-1 labels are strict
    with pytest.raises(ValueError):
        row_source(dict(source="some free text"))
    unk = SourceLabel(kind=Source.UNKNOWN)
    assert row_source(dict(source="some free text"), default=unk) is unk
    assert row_source(dict(seed=1), default=unk) is unk


def test_pipeline_source_counts_both_formats(tmp_path):
    from rrp.harness.pipelines.arm import _source_counts
    p = tmp_path / "rows.jsonl"
    rows = [dict(source="learned(system-i flow)"), dict(source="target_encoder_oracle(ORACLE DIAGNOSTIC: x)"),
            dict(source="learned(system-i flow)", source_label="learned:f.pt", source_label_version="sl-1"),
            dict(source="free text")]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert _source_counts(p) == {"learned": 2, "oracle": 1, "unparsed": 1}


# ------------------------------------------------------------------ wire Literals
def test_wire_literals_accept_old_and_canonical_and_round_trip():
    native = set(typing.get_args(typing.get_type_hints(NativeCommand)["source"]))
    latent = set(typing.get_args(LatentActionChunk.model_fields["source"].annotation))
    assert set(LEGACY_SOURCES) <= native and set(CANONICAL) <= native and "unknown" not in native
    assert set(OLD_LATENT) <= latent and set(CANONICAL) <= latent and "unknown" not in latent
    g = GroupCommand(group="arm", values=np.zeros((2, 3)), mask=np.ones((2, 3), bool))
    for s in sorted(set(LEGACY_SOURCES) | set(CANONICAL)):
        nc = NativeCommand(controller_version="c1", groups={"arm": [0.0]}, source=s)
        assert NativeCommand.model_validate_json(nc.model_dump_json()) == nc
        ac = ActionChunk(observation_id="o", graph_version=0, runtime_version=0, robot_spec_hash="h",
                         controller_version="c", policy_version="p", codec_version=None, start_time=0.0, dt=0.1,
                         horizon=2, command_groups=[g], sampling_seed=None, source=s)
        assert ActionChunk.model_validate_json(ac.model_dump_json()).source == s
    for s in sorted(set(OLD_LATENT) | set(CANONICAL)):
        p = LatentActionChunk(latent_space_version="ls", realizer_compat_version="rz", z=np.zeros((2, 1, 3), np.float32),
                              knot_times=[0.0, 0.1], assemblies=[AssemblyHandle(handle="asm:0123456789abcdef:arm",
                                                                                robot_index=0)],
                              assembly_mask=[True], observation_id="o", graph_version=0, runtime_version=0,
                              robot_spec_hash="h", generated_at=0.0, valid_from=0.0, valid_until=1.0, source=s,
                              policy_version="p")
        assert LatentActionChunk.from_bytes(p.to_bytes()).source == s
    for bad in ("unknown", "learned:x.pt", "oracle_diagnostic"):
        with pytest.raises(Exception):
            NativeCommand(controller_version="c1", groups={}, source=bad)


def test_old_serialized_payload_round_trips_byte_identical():
    old = '{"controller_version":"c1","groups":{"arm":[0.5,1.0]},"source":"teacher","chunk_ref":null}'
    assert NativeCommand.model_validate_json(old).model_dump_json() == old


PRE_D126_LADDER_KEYS = [
    "route", "robot", "seed", "outcome", "privileged_success", "public_success", "steps", "packets", "rejected",
    "fallback_holds", "events", "stage_reached", "failed_stage", "min_tcp_cube_m", "final_teacher_phase",
    "track_q_rad", "track_tcp_m", "cmd_step_rad", "lab_err_arm", "lab_step_arm", "lab_err_grip", "by_phase",
    "lab_err_by_j", "oracle_cmp", "oracle_cmp_by_phase", "ticks", "replans", "interventions", "source",
    "checkpoints", "motion"]
