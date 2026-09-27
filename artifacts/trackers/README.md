# Legged tracker artifacts

`artifacts/trackers/<body>/actor.pt` is the contact_v1 tracker and `artifacts/trackers/<body>/contact_v2/actor.pt` the contact_v2 tracker
(weights are gitignored; sha256 prefixes are recorded in `research/tracks/contact.md`). `meta.json`, `validation_*.json` and
`train_log_every10.jsonl` next to them are the committed provenance.

## Actuator torque limits (D-107, 2026-09-27)
Body models now use the manufacturers' sourced torque limits by default (`actuator_limits = sourced_v1`, see
`rrp.physics.actuator.SOURCED`). **Every t1 and g1 tracker trained before D-107 was trained with the legacy gains table**
(`legacy_gains_v0`: t1 2-3x too strong on all leg joints; g1 hip roll 139 N m instead of 88). Loading one into a default scene raises
`TrackerMismatch`. To run them, set:

    RRP_ACTUATOR_LIMITS=legacy_gains_v0

go2, anymal_c, h1 and the procedural bodies have identical limits in both versions, so their trackers are unaffected.
Actors trained after D-107 record `actuator_limits` in their meta.

## Installed t1 contact_v2 tracker (2026-09-27, lead decision; W8 pins this)
`artifacts/trackers/t1/contact_v2/actor.pt` = **w8d**, sha256 `36e9146792743115878c34e0bbf7cc46ccb5419417921358da3658c8377fc591`,
trained with `actuator_limits = sourced_v1` (loads under the default limits). Label: "t1 sourced-limits w8d: waypoint 20/20; lab gate fails forward
0.72 (slip 0.137 passes); arc no-fall 0.83 at 30 ms". Source run: `artifacts/runs/contact_t1_w8d`; evidence in `research/tracks/contact.md`.
Superseded installs (backed up, not deleted): w8c (sha 863d2469430231f6, installed briefly the same day) and the legacy-limit turn-trained t1 (sha 0d77322c0248e019,
needs `RRP_ACTUATOR_LIMITS=legacy_gains_v0`).
