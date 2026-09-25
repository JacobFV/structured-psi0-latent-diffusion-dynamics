# project status — structured-psi0-latent-diffusion-dynamics (formerly relational robot policy)

**ACTIVE: architecture correction (R38) on branch correction/controller-facing-latent — see research/corrections/controller-facing-semantic-latent.md. Old direct-action path = baseline only.**

Updated: 2026-09-25 (session 2). Overall: **in_progress** (not complete).

## resolved environment
- repo: `~/work/relational-robot-policy` (GitHub: JacobFV/structured-psi0-latent-diffusion-dynamics, renamed 2026-09-25) (host `Dell-gb10-1`, aarch64 GB10). Handoff in `docs/handoff/`.
- peer: `gb10-direct` (hostname promaxgb10-4dfb, direct link). Workspace (RAM-backed, D-003): `/dev/shm/rrp-brandonin/{repo,venv,cache,bin}`.
- host venv: `.venv` (CPU only). peer venv: `/dev/shm/rrp-brandonin/venv` (torch cu130, mujoco).
- enforced parents: host `rrp.slice` 7.11 CPU / 20.0 GiB / swap 0 / host GPU off; peer `rrp.slice` ~15.9 CPU / ~88 GiB / 1 GPU slot. See `configs/resources.local.json`.
- watchdogs: `rrp-watchdog-host.service`, `rrp-watchdog-peer.service` (user services inside rrp-control.slice).

## how to run anything (always under a lease)
```bash
cd ~/work/relational-robot-policy
PYTHONPATH=src python3 -m rrp.cli ops status
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 1 --mem 2G --label NAME -- <cmd>
# peer:
scripts/peer_sync.sh push
ssh gb10-direct 'cd /dev/shm/rrp-brandonin/repo && PATH=/dev/shm/rrp-brandonin/bin:$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$PWD python3 -m rrp.cli ops run --gpu --gpu-mem 8G --cpu 4 --mem 16G --label NAME -- /dev/shm/rrp-brandonin/venv/bin/python ...'
```

## completed / verified
- P02/P03 resource safety (cgroup enforcement, broker, watchdog; peer = whole machine per D-008, 3 GPU slots D-009).
- P01, P04-P06 contracts + task runtime; P07/P08 sim + labelled scripted teacher; P10 service; P11 UI (15/15 browser tests, artifacts/video/workbench-demo.webm).
- P09 (arms): procedural arm family (5/6/7 DOF), menagerie panda/fr3/ur5e/sawyer/xarm7/z1/lite6, two gripper modules with adapter standoffs; teacher ledger artifacts/assets/teacher_validation_pick_place_20seeds.txt.
- Frozen split research/splits/primary_v1.json; dataset artifacts/datasets/pick_place_primary_v1 (peer): 3791 success / 106 teacher failures / 453 infeasible.

- Dual-arm track (verified, scripted_teacher only): support_insert + handover scenarios (`src/rrp/sim/dual_scenarios.py`, `tasks/handover.json`), DualSession public estimators, dual teachers, ALOHA import, MultiFeaturizer (feat-multi-v1+feat-v2); teacher v2 validation 16 SI pairs / 9 handover pairs x 30 seeds (`artifacts/assets/dual_teacher_validation/*_v3*`, D-019); dataset `artifacts/datasets/support_insert_primary_v2` (teacher v2; v1 archived; split `research/splits/primary_v1_support_insert.json`); functional composition report `research/reports/functional_composition.md`; details `research/reports/dual_arm_tasks.md`.

## now (2026-09-25)
- Stage A complete (research/reports/latent_slice1_progress.md). Probe analysis + loss-gap diagnosis: D-031.
- RUNNING on peer: `scripts/latent_chain_v2.sh` (flow_latent_sem_v2 -> flow_latent_nosem_v2 -> 20-episode eval on
  panda_pg2, parm6_tf3, parm5s_tf3, parm5l_pg2 -> disturbance -> labelled videos), then systemd unit `rrp-chain-v3`
  runs the same chain for flow_latent_sem_v3. Log: /dev/shm/rrp-brandonin/repo/ops/logs/latent_chain_v{2,3}.out.
- Resume (idempotent; training resumes from policy_last.pt, finished steps skipped): check `pgrep -af latent_chain` and
  `systemctl --user status rrp-chain-v3` on the peer first; if neither runs, `scripts/peer_sync.sh push` then
  `ssh gb10-direct 'cd /dev/shm/rrp-brandonin/repo && nohup setsid bash scripts/latent_chain_v2.sh latent_sem_v2 latent_nosem_v2 latent_sem_v3 > ops/logs/latent_chain_resume.out 2>&1 &'`.
- After a peer reboot: push source, `scripts/peer_bootstrap.sh` (restores venv, menagerie on disk, data links),
  copy `~/rrp-peer-data/artifacts-snapshot-20260921/runs/*` into repo/artifacts/runs/ if missing (snapshot is on the PEER).
- Host `.venv` is a self-referencing symlink (broken); host tests currently run on the peer.
- Remaining after the chain: counterexample, embodiment-swap, latency, composition, causal-edit acceptance tests;
  four-way baseline comparison; target adaptation (xarm7_pg2, xarm7_tf3, panda_tf3) then packet-policy GRPO;
  migrate VLM, dual-arm, legged work. Competent closed-loop performance of the corrected policy is NOT established.

## current work (parallel tracks)
- lead: codec + structured/unstructured BC policies training on peer (artifacts/runs/codec_dev_v1, dev_structured_direct, dev_unstructured_direct).
- agent tracks: legged/humanoid breadth; dual-arm support_insert/handover; GRPO/EXPO-FT; psi0 VLM backbone + object QA.

## budgets used (see ops/resource-ledger.jsonl on each node)
- peer GPU device-hours: ~0.001 of 96. CPU core-hours: <0.1 of 512. Calendar day 1 of 7.

## blockers
- dual-arm: panda+tf3 insertion-arm target teacher-weak (3/30, 8/30, D-019); sawyer support / sawyer->panda_tf3 handover pairs fail (recorded); public insertion-depth estimator can false-positive on jams (D-018).
- none yet.

## resume
Read this file, `research/tasks.json`, `research/decisions.md`. Check `ops status` on both nodes and `systemctl --user list-units 'rrp-*'` before starting anything; attach to running owned jobs instead of duplicating.
