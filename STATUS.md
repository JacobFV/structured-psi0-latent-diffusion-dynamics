# project status: structured-psi0-latent-diffusion-dynamics (formerly relational robot policy)

Updated 2026-09-30 (D-145 P8). Overall: **in progress, experiments paused for the refactor (D-140), one track open.**
The repository layout is `schema.toml` (docs/architecture.md section 13); legacy material is under `.old/` (index
`.old/README.md`); the previous long-form status, evidence summary and work log are `.old/STATUS.md`.

Where to read: plan `docs/strategy.md` (D-094), open questions `docs/experiments_roadmap.md`, what is shown
`research/reports/evidence_matrix.md`, why `research/decisions.md` (latest entries D-138..D-145), rules `AGENTS.md`.

## tracks (`schema.toml [tracks]`; note `research/tracks/<track>.md`, recipes `recipes/<track>/`)
| track | workstream | state | next step | decisions |
|---|---|---|---|---|
| relations | relation-factor experiments | open (planned; recipes render and plan dry, nothing has run) | train `recipes/relations/relations_{geo,ix,task}.yaml` against the `base` control on the peer; open items are in the note | D-144 |
| humanoid | W13 humanoid transfer | paused (owner top priority when resumed) | RESUME section of the note: `rrp run-dag recipes/humanoid/...` | D-138, D-139 |
| armdiv | W7 training-arm diversity | paused | RESUME section: `rrp run-dag recipes/armdiv/...` (semfix s1 with `--retry-failed` first) | D-137 |
| psi0 | W10 Psi0 / SIMPLE | paused: structured-arm integration bug (Psi0 + structure 0/20 vs direct 19/20) to diagnose | RESUME section of the note; recipes `recipes/psi0/` | D-141 |
| pointer | ComputerWorld pointer policy | paused: follow-ups | RESUME section of the note; recipes `recipes/pointer/` | D-142 |

Closed tracks (W1 contact v2, W2 to W6 hygiene / provenance / restructure / pipeline / robustness, W7 arm expert, W8
legged regeneration, W9 target bodies, W11, W12 and the earlier ladder / acceptance / baselines / binding lines): notes in
`.old/research/tracks/`, decisions in `research/decisions.md`, raw results in `artifacts/runs/` (frozen run names listed in
`schema.toml`). Their DAG templates stay as generic family recipes in `recipes/templates/`. The loopback workbench is
retired (D-145 addendum): the visualization room (`viz/room`, D-131) is the only viz surface.

## running work
Nothing is running (all experiments paused, D-140). Before starting anything: `rrp ops status` on both nodes and
`systemctl --user list-units 'rrp-*'`; attach to owned jobs, never duplicate. After a host reboot restart the watchdog
(AGENTS.md).

## resolved environment
- repo `~/work/relational-robot-policy` (GitHub JacobFV/structured-psi0-latent-diffusion-dynamics; host `Dell-gb10-1`, aarch64 GB10); parallel agents use `~/work/rrp-wt/<track>`.
- peer `gb10-direct` (all of it, D-008/D-026); workspace `/dev/shm/rrp-brandonin/{repo,venv,cache,bin}` (RAM-backed); shared repo dir is for the lead only, every other agent uses `wt/<track>`.
- host venv `.venv` (CPU only); peer venv `/dev/shm/rrp-brandonin/venv` (torch cu130, mujoco). Host broker caps and the peer admission limits: `ops/resources.local.json`; disk reserve 100 GB (D-086).
- watchdogs: `rrp-watchdog-host.service`, `rrp-watchdog-peer.service` (user services in `rrp-control.slice`).

## how to run anything (always under a lease)
```bash
cd ~/work/relational-robot-policy
PYTHONPATH=src python3 -m rrp.cli ops status
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 1 --mem 2G --label NAME -- <cmd>        # host: light jobs only
export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track>; ops/bin/peer_sync.sh push       # peer: own dir, then
ops/bin/peer_run.sh --gpu --gpu-mem 12G --cpu 4 --mem 24G --label <track>_x --max-seconds N --detach -- PY -m rrp.cli run-dag recipes/<track>/<name>.yaml
```

## known limits (details: evidence matrix, `.old/STATUS.md`)
- Legged results through D-092 use contact v1; contact v2 trackers exist for anymal_c, go2 and t1 (t1 under a labelled D-112 gate exception, D-113 / D-139).
- Arm results before D-110 used grasp v1; v6 (teacher v2 + grasp v2.1) is the current arm baseline (D-134). The latent route does not transfer to a new arm and never beats BC there (D-135).
- Host: no heavy compute (D-115, D-127). Peer: disk about 97% used; memory pressure sheds leases (declare at least 1.35 x peak, D-117).

## resume
Read this file, `docs/strategy.md`, the latest entries of `research/decisions.md` and the track note; then the note's
RESUME `rrp run-dag` lines (idempotent ledgers under `artifacts/runs/<track>/_dags/`).
