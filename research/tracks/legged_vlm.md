# track legged_vlm — legged/humanoid breadth and VLM system II on the latent-packet path

Worktree `~/work/rrp-wt/legged_vlm`, branch `track/legged_vlm`, peer dir `/dev/shm/rrp-brandonin/wt/legged_vlm`.

## inventory (2026-09-25)
- Legged code from the pre-correction track exists (report `research/reports/legged_breadth.md`, D-023).
- Frozen tracker actors that SURVIVED (git-tracked): `artifacts/trackers/go2/actor.pt` (learned PPO iter 1199),
  `artifacts/trackers/hexapod6/actor.pt` (learned iter 774; fails full-stance halt). CPG trackers need no weights
  (pquad4, hexapod6, hexapod6_long, sprawl4, sprawl8).
- NOT in the live checkouts: anymal_c, g1, t1, h1 actors and the old legged teacher datasets. FOUND in the host
  archive `~/.archive/relational-robot-policy-2026-09-25-original/artifacts/trackers/*/actor.pt` (iters 1274 / 3599 /
  2999 / 2999, matching legged_breadth.md; go2 actor sha identical to the tracked one). Copied (not committed) to the
  peer store `artifacts/trackers/<body>/actor.pt`. The VLM weights/feature caches were lost (re-downloaded below).

## log
- 12:3x t1 tracker retrain with the recorded v3 recipe (same args as `artifacts/trackers/t1/meta.json`, seed 1),
  lease 1790363836_501e49, out `/dev/shm/rrp-brandonin/legged_runs_lv/t1` (state: running). This is a RE-RUN of a
  lost checkpoint; it is gated again by `tracker_validation` before any use. STOPPED at iter ~400 once the original
  t1 actor was found in the archive (not needed).
- Tick-level collector `rrp.data.legged_latent_collect` (50 Hz native labels = clean tracker joint targets, DART
  execution noise sigma in {0,0.1,0.2,0.3} action units, randomized teacher gains). Smoke 4/4 go2 episodes.
  Full collection: lease 1790364032_2c1515, bodies go2(learned) pquad4 hexapod6 sprawl4 sprawl8 hexapod6_long (CPG),
  seeds 0-599 each -> `artifacts/datasets/legged_latent_v1/<body>/s*.npz` (peer disk). state: running.
- INCIDENT (host, ~12:49): copying the 4.0 GB psi0 snapshot to host disk (~/work/rrp-data/hf) pushed host free disk
  below the watchdog reserve (393.7 GB; host disk was already within ~4 GB of it because of the peer->host data
  mirror). The host watchdog shed my own download lease AND another track's job `binding_v1_reeval`
  (lease 1790363904_b7ef3b, ran 31 min, stopped_by=disk_below_reserve). I deleted the host copy immediately
  (free disk back above reserve). All VLM work now runs on the peer only (weights on peer disk
  ~/rrp-peer-data/cache/hf). The binding track needs to relaunch that job.
- psi0 System-II weights: peer HF_HOME=/home/brandonin/rrp-peer-data/cache/hf, USC-PSI-Lab/psi-model@4c6f977,
  subfolder psi0/pre.fast.2605160748.ckpt.ego390k; model.safetensors sha256 b2ab7b35...06f0 (matches the HF LFS oid;
  verified with sha256sum). Packages (transformers 4.57.1 etc.) in an isolated --target dir
  /home/brandonin/rrp-peer-data/vlm_pkgs (the shared peer venv is NOT modified); use PYTHONPATH=src:<that dir>.
- xet download stalled at 75 kB; HF_HUB_DISABLE_XET=1 downloaded 4.0 GB in 7.5 min.
