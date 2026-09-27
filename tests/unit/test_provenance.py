"""W3 provenance and contracts: Provenance record, Source mapping, manifest writer/reader, fingerprinted legged
checkpoints and compatibility IDs, explicit zero_prev_action."""
from __future__ import annotations

import hashlib
import json
import sys
import typing
from pathlib import Path

import mujoco
import pytest
import torch

from rrp.contracts.provenance import (FEATURIZER_VERSION, MissingFlagError, Provenance, Source, legacy_provenance,
                                      make_provenance, parse_source, physics_provenance, read_provenance,
                                      resolve_zero_prev_action, source_label, training_flags, weights_digest)

ROOT = Path(__file__).resolve().parents[2]
XML = """<mujoco><option timestep="0.001" cone="elliptic" impratio="10" integrator="implicitfast" noslip_iterations="3"/>
<worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>"""


def test_physics_provenance_from_model():
    m = mujoco.MjModel.from_xml_string(XML)
    p = physics_provenance(m, contact_version="contact_v2")
    assert (p.timestep, p.cone, p.impratio, p.integrator, p.noslip_iterations) == (0.001, "elliptic", 10.0,
                                                                                    "implicitfast", 3)
    assert p.mujoco_version == mujoco.__version__ and p.contact_version == "contact_v2"
    assert physics_provenance(mujoco.MjModel.from_xml_string("<mujoco/>")).contact_version == "contact_v1"
    assert physics_provenance(mujoco.MjModel.from_xml_string("<mujoco/>")).cone == "pyramidal"
    # contact track (W1) embeds the version in the compiled model; it is picked up, and contradictions fail
    tagged = mujoco.MjModel.from_xml_string('<mujoco><custom><text name="contact_version" data="contact_v2"/>'
                                            '</custom></mujoco>')
    assert physics_provenance(tagged).contact_version == "contact_v2"
    with pytest.raises(ValueError):
        physics_provenance(tagged, contact_version="contact_v1")


def test_provenance_json_roundtrip_and_legacy():
    m = mujoco.MjModel.from_xml_string(XML)
    p = make_provenance("learned:runs/x/policy.pt", model=m, featurizer_version=FEATURIZER_VERSION,
                        weights=dict(E={"w": torch.ones(3)}), versions=dict(latent_space_version="ls-1"),
                        flags=dict(zero_prev_action=True, probe_lv_min=-4.0))
    q = Provenance.from_json(p.to_json())
    assert q == p and q.fingerprinted and not q.legacy and len(q.weights["E"]) == 12 and q.bundle_fingerprint
    assert q.source_label.kind is Source.LEARNED and q.source_label.detail == "runs/x/policy.pt"
    assert read_provenance({"provenance": p.to_dict()}) == p
    old = read_provenance({"source": "scripted_teacher", "featurizer": "feat-v2"})     # pre-W3 meta
    assert old.legacy and not old.fingerprinted and old.featurizer_version == "feat-v2"
    with pytest.raises(Exception):
        Provenance.from_json(dict(p.to_dict(), unknown_field=1))                        # strict schema


def test_source_mapping_covers_legacy_strings():
    from rrp.contracts.action import Source as ActionSource
    from rrp.contracts.latent_action import LatentActionChunk
    for s in typing.get_args(ActionSource):
        parse_source(s)
    for s in typing.get_args(LatentActionChunk.model_fields["source"].annotation):
        parse_source(s)
    assert parse_source("teacher").kind is Source.SCRIPTED_TEACHER
    assert parse_source("target_encoder_oracle").kind is Source.ORACLE
    assert parse_source("oracle").kind is Source.ORACLE
    assert parse_source("privileged_oracle_packet:legged-ls-a").kind is Source.ORACLE
    assert parse_source("scripted_controller").kind is Source.CPG_TRACKER
    assert parse_source("learned_tracker").kind is Source.LEARNED_TRACKER
    assert str(parse_source("scripted_teacher:arc_only")) == "scripted_teacher:arc_only"
    assert not parse_source("oracle").deployable and parse_source("bc:x.pt").deployable
    # new writes are strict
    assert source_label("learned", "runs/a/policy.pt") == "learned:runs/a/policy.pt"
    with pytest.raises(ValueError):
        parse_source("learned", strict=True)            # learned needs a checkpoint
    with pytest.raises(ValueError):
        parse_source("teacher", strict=True)            # legacy alias not accepted for new writes
    with pytest.raises(ValueError):
        parse_source("encoded teacher targets")         # unknown free string


def test_featurizer_constant_is_single():
    from rrp.data.collect import FEATURIZER_VERSION as A
    from rrp.learning.behavior import FEAT_VERSION as B
    assert A is FEATURIZER_VERSION and B is FEATURIZER_VERSION


def test_weights_digest_matches_arm_bundle_algorithm():
    from rrp.control.latent_realizer import weights_digest as wd_old_name
    sd = torch.nn.Linear(4, 3).state_dict()
    h = hashlib.sha256()                                  # the D-038 algorithm, verbatim
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    assert weights_digest(sd) == wd_old_name(sd) == h.hexdigest()[:12]


def test_manifest_writer_and_legacy_readers(tmp_path):
    from rrp.data.manifest import read_manifest, write_manifest, dataset_provenance
    phys = physics_provenance(mujoco.MjModel.from_xml_string(XML)).to_dict()
    eps = [dict(episode_id="a", status="success", physics=phys), dict(episode_id="b", status="failure", physics=phys)]
    prov = dataset_provenance(eps, source="scripted_teacher", featurizer_version=FEATURIZER_VERSION)
    write_manifest(tmp_path, "t", eps, extra=dict(source="scripted_teacher"), provenance=prov)
    m = read_manifest(tmp_path)
    assert m["hash_ok"] and m["provenance"] == prov and m["provenance"].physics.cone == "elliptic"
    eps[1]["physics"] = dict(phys, impratio=1.0)                 # mixed physics is recorded, never guessed
    assert dataset_provenance(eps, source="scripted_teacher").physics is None
    # old legged manifest (no hash, no provenance)
    old = tmp_path / "old"; old.mkdir()
    (old / "manifest.json").write_text(json.dumps(dict(body="go2", n=1, success=1, fell=0,
                                                       episodes=[dict(episode_id="x", status="success")])))
    m = read_manifest(old)
    assert m["hash_ok"] is None and m["provenance"].legacy and m["provenance"].source == "scripted_teacher"
    # old hashed arm manifest (pre-W3 write_manifest body)
    body = dict(name="o", n_episodes=1, status_counts={"success": 1}, episodes=[dict(status="success",
                                                                                     source="scripted_teacher")])
    body["manifest_hash"] = hashlib.sha256(json.dumps(body, indent=1, sort_keys=True).encode()).hexdigest()[:16]
    (old / "manifest.json").write_text(json.dumps(body))
    m = read_manifest(old)
    assert m["hash_ok"] and m["provenance"].legacy


def test_legged_checkpoints_fingerprinted_and_legacy(tmp_path):
    from rrp.learning.legged_latent_train import _save, checkpoint_provenance
    from rrp.evaluation.legged_latent_eval import legged_bundle_versions
    E, R = torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)
    cfg = dict(name="t", latent=dict(dz=2, width=4, probe_lv_min=-4.0, semantic_weight=1.0))
    _save(tmp_path / "representation.pt", E=E.state_dict(), R=R.state_dict(), cfg=cfg,
          result=dict(latent_space_version="legged-ls-t"))
    st = torch.load(tmp_path / "representation.pt", weights_only=False)
    p = checkpoint_provenance(st, tmp_path / "representation.pt")
    assert p.fingerprinted and set(p.weights) == {"E", "R"} and p.flags["probe_lv_min"] == -4.0
    assert p.flags["prev_action_input"] is False
    side = json.loads((tmp_path / "representation.json").read_text())
    assert side["provenance"]["weights"] == p.weights and len(side["sha256_16"]) == 16
    # a bare pre-W3 checkpoint still loads and is marked legacy / unfingerprinted
    torch.save(dict(E=E.state_dict(), R=R.state_dict(), cfg=cfg), tmp_path / "old.pt")
    q = checkpoint_provenance(torch.load(tmp_path / "old.pt", weights_only=False), tmp_path / "old.pt")
    assert q.legacy and not q.fingerprinted and q.notes == "unfingerprinted"
    # tampering is detected
    st["R"]["weight"] += 1.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        checkpoint_provenance(st, "x")
    # compatibility IDs: from the loaded weights, so a refit realizer gets a new system-0 ID
    lsv, rcv = legged_bundle_versions("legged-ls-t", E.state_dict(), R.state_dict())
    assert lsv.startswith("legged-ls-t-w") and rcv.startswith("legged-rz-osc-v1-" + lsv + "-r")
    R2 = torch.nn.Linear(2, 2)
    lsv2, rcv2 = legged_bundle_versions("legged-ls-t", E.state_dict(), R2.state_dict())
    assert lsv2 == lsv and rcv2 != rcv
    assert legged_bundle_versions(lsv, E.state_dict(), R.state_dict()) == (lsv, rcv)   # idempotent


def test_save_checkpoint_provenance(tmp_path):
    from rrp.learning.checkpoint import save_checkpoint, load_checkpoint, checkpoint_provenance
    m = torch.nn.Linear(2, 2)
    save_checkpoint(tmp_path / "p.pt", model=m, step=1, versions=dict(featurizer=FEATURIZER_VERSION),
                    config=dict(zero_prev_action=True, realizer_drop_qd=False))
    p = checkpoint_provenance(load_checkpoint(tmp_path / "p.pt"), tmp_path / "p.pt")
    assert p.weights["model"] == weights_digest(m.state_dict()) and p.flags["zero_prev_action"] is True
    assert p.featurizer_version == FEATURIZER_VERSION
    old = dict(model=m.state_dict(), versions={}, config={})
    q = checkpoint_provenance(old, tmp_path / "p.pt")
    assert q.legacy and q.flags["zero_prev_action"] is None


def test_zero_prev_action_explicit(capsys):
    assert resolve_zero_prev_action({"zero_prev_action": True}, where="t", new_run=True) is True
    with pytest.raises(MissingFlagError):
        resolve_zero_prev_action({}, where="t", new_run=True)
    with pytest.warns(UserWarning, match="LEGACY"):
        assert resolve_zero_prev_action({}, where="t", new_run=False) is False
    assert "WARNING" in capsys.readouterr().err
    with pytest.raises(ValueError):
        resolve_zero_prev_action({"zero_prev_action": "yes"}, where="t", new_run=True)
    assert training_flags({"latent": {"probe_lv_min": -3}})["zero_prev_action"] is None


def test_migration_insert_and_repo_configs_explicit():
    sys.path.insert(0, str(ROOT / "scripts"))
    import migrate_zero_prev_action as mig
    for txt in ('{"a": 1}', '{\n "a": [1, 2]\n}', '  {"name": "x",\n "b": {"c": 1}}'):
        assert json.loads(mig.insert_key(txt)) == dict(json.loads(txt), zero_prev_action=False)
    missing = []
    for p in sorted((ROOT / "configs").rglob("*.json")):
        rel = p.relative_to(ROOT).as_posix()
        cfg = json.loads(p.read_text())
        if mig.in_scope(rel, cfg) and "zero_prev_action" not in cfg and not rel.startswith(mig.PROTECTED):
            missing.append(rel)
    assert missing == [], missing


def test_legacy_provenance_unknown_source_is_marked():
    p = legacy_provenance("encoded teacher targets for x")
    assert p.legacy and p.source == "unknown" and "unparsed legacy source" in p.notes
    with pytest.raises(ValueError):
        source_label("unknown")
