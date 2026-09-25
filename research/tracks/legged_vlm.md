# track legged_vlm — legged/humanoid breadth and VLM system II on the latent-packet path

Worktree `~/work/rrp-wt/legged_vlm`, branch `track/legged_vlm`, peer dir `/dev/shm/rrp-brandonin/wt/legged_vlm`.

## inventory (2026-09-25)
- Legged code from the pre-correction track exists (report `research/reports/legged_breadth.md`, D-023).
- Frozen tracker actors that SURVIVED (git-tracked): `artifacts/trackers/go2/actor.pt` (learned PPO iter 1199),
  `artifacts/trackers/hexapod6/actor.pt` (learned iter 774; fails full-stance halt). CPG trackers need no weights
  (pquad4, hexapod6, hexapod6_long, sprawl4, sprawl8).
- LOST (were only on peer tmpfs, not found on host or peer): anymal_c, g1, t1, h1 actors; all legged teacher datasets
  (`artifacts/datasets/legged_waypoint_contact*`), the VLM weights cache and feature cache.

## log
- 12:3x t1 tracker retrain with the recorded v3 recipe (same args as `artifacts/trackers/t1/meta.json`, seed 1),
  lease 1790363836_501e49, out `/dev/shm/rrp-brandonin/legged_runs_lv/t1` (state: running). This is a RE-RUN of a
  lost checkpoint; it is gated again by `tracker_validation` before any use.
- Tick-level collector `rrp.data.legged_latent_collect` (50 Hz native labels = clean tracker joint targets, DART
  execution noise sigma in {0,0.1,0.2,0.3} action units, randomized teacher gains). Smoke 4/4 go2 episodes.
  Full collection: lease 1790364032_2c1515, bodies go2(learned) pquad4 hexapod6 sprawl4 sprawl8 hexapod6_long (CPG),
  seeds 0-599 each -> `artifacts/datasets/legged_latent_v1/<body>/s*.npz` (peer disk). state: running.
