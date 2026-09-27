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
