"""Build artifacts/assets/legged_catalog.json: per-body status ladder with reasons.

Stages: source_verified -> imported -> physics_validated -> controller_validated -> teacher_validated
(policy_evaluated is owned by the learning track). Each stage records ok / reason / evidence.
  source_verified    : menagerie dir + LICENSE + MJCF at the pinned commit (sha256 of the MJCF);
                       procedural: generator + params hash.
  imported           : builds as a legged Module, compiles into a waypoint_contact scene, RobotSpec
                       compiles (joints/actuators/sensors typed), IMU + foot touch sensors present.
  physics_validated  : positive masses/inertias, finite 3 s rollout from the default pose under the
                       default-pose PD hold, no deep initial penetration; standing under the static
                       hold is recorded as information (humanoids are not expected to balance on it).
  controller_validated: frozen tracker eligibility (artifacts/trackers/<body>/eligibility_*.json).
  teacher_validated  : waypoint teacher success rate >= 0.8 on the frozen tracker
                       (artifacts/assets/legged_teacher/<body>.json).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from .importers import MENAGERIE, MENAGERIE_SHA
from .legged import ALL_LEGGED, LEGGED_ASSETS, PROCEDURAL, legged_body, standalone_model

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "artifacts" / "assets" / "legged_catalog.json"
TRACKERS = REPO / "artifacts" / "trackers"
TEACH = REPO / "artifacts" / "assets" / "legged_teacher"


def _stage(ok, reason=None, **ev):
    return dict(ok=bool(ok), reason=reason, **ev)


def physics_check(model, meta) -> dict:
    from rrp.control.legged_core import LeggedBinding
    b = LeggedBinding(model, meta)
    d = mujoco.MjData(model)
    b.set_default(d)
    mujoco.mj_forward(model, d)
    bodies = b.robot_bodies
    masses = model.body_mass[bodies]
    inert = model.body_inertia[bodies]
    pos_mass = bool((masses[masses > 0] > 0).all() and masses.sum() > 0)
    # bodies with mass must have positive inertia
    pos_inertia = bool(all((inert[i] > 0).all() for i in range(len(bodies)) if masses[i] > 0))
    pen = min([d.contact[i].dist for i in range(d.ncon)] or [0.0])
    stood, finite = True, True
    z0 = float(d.qpos[b.qa + 2])
    for k in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, d)
        if not np.isfinite(d.qpos).all():
            finite = False
            break
    fell = (not finite) or d.qpos[b.qa + 2] < b.min_h or b.tilt(d) > b.tilt_limit
    ok = pos_mass and pos_inertia and finite and pen > -0.01
    return _stage(ok, None if ok else f"mass_ok={pos_mass} inertia_ok={pos_inertia} finite={finite} "
                  f"initial_penetration={pen:.4f}",
                  total_mass_kg=float(model.body_subtreemass[b.root_bid]), n_policy_joints=b.n, n_feet=b.nf,
                  nominal_height_m=b.nominal_height(), initial_min_contact_dist=float(pen),
                  stands_under_static_pd_hold_3s=bool(not fell), final_height=float(d.qpos[b.qa + 2]),
                  start_height=z0, timestep=float(model.opt.timestep))


def body_entry(key: str) -> dict:
    e = dict(id=key, stages={})
    t0 = time.time()
    if key in LEGGED_ASSETS:
        info = LEGGED_ASSETS[key]
        d = MENAGERIE / info["dir"]
        f = d / info["file"]
        lic = d / "LICENSE"
        ok = f.exists() and lic.exists()
        e.update(source="menagerie", kind=info["kind"], family=info["family"], dir=info["dir"], file=info["file"],
                 license=info["license"], commit=MENAGERIE_SHA, synthetic=False)
        e["stages"]["source_verified"] = _stage(ok, None if ok else "missing asset/LICENSE",
                                                mjcf_sha256=hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None,
                                                license_file=lic.exists())
    else:
        mod = PROCEDURAL[key]()
        e.update(source="procedural", kind=mod.meta["family"], family=mod.meta["lineage"][1], synthetic=True,
                 license="project", generator="rrp.morphology.legged.procedural_legged")
        ph = hashlib.sha256(json.dumps(mod.meta["params"], sort_keys=True, default=str).encode()).hexdigest()[:16]
        e["stages"]["source_verified"] = _stage(True, None, params_hash=ph)
    if not e["stages"]["source_verified"]["ok"]:
        return e
    try:
        from rrp.sim.legged import build_waypoint_contact
        mod = legged_body(key)
        sc = build_waypoint_contact(mod, 0, body_key=key)
        rs = sc.robots[0].robot_spec
        imu_ok = any(s.kind == "imu" for s in rs.sensors)
        touch_ok = sum(s.kind == "touch" for s in rs.sensors) == len(mod.meta["legged"]["foot_bodies"])
        e["lineage"] = rs.lineage
        e["stages"]["imported"] = _stage(imu_ok and touch_ok, None if (imu_ok and touch_ok) else "sensors missing",
                                         spec_hash=rs.spec_hash, n_joints=len(rs.joints), n_actuators=len(rs.actuators),
                                         n_sensors=len(rs.sensors), family=rs.family,
                                         controller_contracts=[c.kind for c in rs.controller_contracts],
                                         actuator_adapter=(mod.meta.get("actuator_adapter") or {}).get("kind"),
                                         policy_actuators=len(mod.meta["legged"]["policy_actuators"]),
                                         held_actuators=len(mod.meta["legged"]["held_actuators"]))
    except Exception as ex:  # noqa: BLE001 - failure is evidence
        e["stages"]["imported"] = _stage(False, f"{type(ex).__name__}: {ex}")
        return e
    try:
        model, _, meta = standalone_model(mod)
        e["stages"]["physics_validated"] = physics_check(model, meta)
    except Exception as ex:  # noqa: BLE001
        e["stages"]["physics_validated"] = _stage(False, f"{type(ex).__name__}: {ex}")
        return e
    # controller: frozen eligibility files (prefer learned)
    elig = {}
    for kind in ("learned", "cpg"):
        p = TRACKERS / key / f"eligibility_{kind}.json"
        if p.exists():
            elig[kind] = json.loads(p.read_text())
    if elig:
        best = next((elig[k] for k in ("learned", "cpg") if k in elig and elig[k]["eligible"]), None)
        e["stages"]["controller_validated"] = _stage(
            best is not None, None if best else "; ".join(f"{k}: gate failed {v['gate']}" for k, v in elig.items()),
            trackers={k: dict(eligible=v["eligible"], gate=v["gate"], tracker_sha=v["tracker_sha"],
                              frozen_at=v["frozen_at"]) for k, v in elig.items()},
            selected=(best or {}).get("tracker_kind"))
    else:
        e["stages"]["controller_validated"] = _stage(False, "no tracker trained/validated for this body (not attempted)")
    reps = {}
    for tp in sorted(TEACH.glob(f"{key}*.json")):
        t = json.loads(tp.read_text())
        if t.get("body") == key:
            reps[tp.stem] = dict(success_rate=t["success_rate"], n=t["n"], fell=t["fell"],
                                 tracker_source=t["tracker_source"], file=str(tp.relative_to(REPO)))
    if reps:
        best = max(reps.values(), key=lambda r: r["success_rate"])
        ok = best["success_rate"] >= 0.8
        e["stages"]["teacher_validated"] = _stage(ok, None if ok else f"best teacher success {best['success_rate']:.2f} < 0.8",
                                                  selected_tracker_source=best["tracker_source"] if ok else None,
                                                  runs=reps)
    else:
        e["stages"]["teacher_validated"] = _stage(False, "teacher not run (requires a validated controller)")
    e["catalog_wall_s"] = time.time() - t0
    return e


def main():
    rows = [body_entry(k) for k in ALL_LEGGED]
    ladder = ["source_verified", "imported", "physics_validated", "controller_validated", "teacher_validated"]
    for r in rows:
        top = None
        for s in ladder:
            if r["stages"].get(s, {}).get("ok"):
                top = s
            else:
                break
        r["status"] = top or "candidate"
    out = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), menagerie_commit=MENAGERIE_SHA, ladder=ladder,
               lineage_notes=["anymal_b/c are near-duplicates: only anymal_c imported",
                              "go2 and a1 share the unitree_quadruped family key (not independent holdouts)",
                              "procedural bodies share procedural_legged/v1; sprawl4/6/8 + hexapod6_long are one "
                              "lineage family (topology diagnostics), pquad4 is the mammal-layout sub-family",
                              "psi0 contracts (rrp.control.psi_contracts) are NOT validated on any of these bodies: "
                              "menagerie g1.xml has no Dex3 hands (SONIC source body is G1+Dex3), h1.xml has no "
                              "Inspire hands (AMO source body)"],
               bodies=rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    for r in rows:
        print(f"{r['id']:14s} {r['kind']:10s} {r['status']:22s} " + " ".join(
            f"{s}={'Y' if r['stages'].get(s, {}).get('ok') else 'n'}" for s in ladder))


if __name__ == "__main__":
    main()
