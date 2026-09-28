# related repositories

This repository (**rrp**; GitHub: [JacobFV/structured-psi0-latent-diffusion-dynamics](https://github.com/JacobFV/structured-psi0-latent-diffusion-dynamics),
local checkout `~/work/relational-robot-policy`) is the multi-body research platform and the installable core (`rrp.core`).
The Ψ₀ line lives in a second repository:

| | rrp (this repo) | psi1z |
|---|---|---|
| GitHub | [JacobFV/structured-psi0-latent-diffusion-dynamics](https://github.com/JacobFV/structured-psi0-latent-diffusion-dynamics) (public) | [JacobFV/psi1z](https://github.com/JacobFV/psi1z) (not public; owner-added remote, D-119) |
| local | `~/work/relational-robot-policy` (+ agent worktrees `~/work/rrp-wt/<track>`) | `~/work/psi1z` |
| scope | architecture, evidence, arm / dual-arm / legged research, physics (contact, grasp, actuators), robustness and gates, pipeline and run-dag, provenance, ops broker and watchdog | Ψ₀ / SIMPLE benchmark only (workstream W10): G1 + hands body, SIMPLE env wrapper, Ψ₀ trunk features, structured packet head for Ψ₀, packet → Ψ₀ action mapping, eval entrypoints, upstream patches |
| decisions log | [research/decisions.md](../research/decisions.md) (D-xxx) | [research/decisions.md](https://github.com/JacobFV/psi1z/blob/main/research/decisions.md) (P-xxx) |
| notes / resume | [STATUS.md](../STATUS.md), `research/tracks/<track>.md` | [research/notes.md](https://github.com/JacobFV/psi1z/blob/main/research/notes.md) |
| Python | 3.11 and 3.12 | 3.11 (Ψ₀ / Isaac Sim 5.1 envs under `~/work/ext`) |

## names and terms (read this first to avoid confusion)
| name | what it is | what it is NOT |
|---|---|---|
| **rrp** | this repository and its Python package (`import rrp`, `rrp.core`) | not a separate project from the GitHub repo below |
| **structured-psi0-latent-diffusion-dynamics** | the GitHub name of THIS repo (renamed 2026-09-25) | not the Ψ₀ upstream, not psi1z |
| **relational-robot-policy** | the local folder name of THIS repo (`~/work/relational-robot-policy`), kept from before the rename | not a different repo |
| **Ψ₀ / psi0** | the UPSTREAM model and code from physical-superintelligence-lab (github.com/physical-superintelligence-lab/Psi0, arXiv 2603.12263): a humanoid VLA plus the SIMPLE benchmark; used unmodified from `~/work/ext` | not our code; we never edit it (patches live as files in psi1z) |
| **psi1z** | OUR repo for the Ψ₀ line: adapters that fine-tune Ψ₀ with our structure and evaluate it in SIMPLE | not a fork of Ψ₀; not a replacement for rrp |
| **system i / system 0** | OUR architecture: system i (flow model) generates the latent packet z; system 0 (realizer) turns z into joint commands | "system 0" has nothing to do with Ψ₀ (psi-zero) despite the similar name |
| **packet / z** | the structured latent packet z[knots × assemblies × 64] passed from system i to system 0 | not Ψ₀'s action tokens |
| **SIMPLE** | Ψ₀'s humanoid benchmark (MuJoCo physics + Isaac Sim rendering) | not our MuJoCo scenes (arm/legged) |
| **D-xxx** | decisions in rrp `research/decisions.md` | — |
| **P-xxx** | decisions in psi1z `research/decisions.md` | not rrp decisions (the crosswalk below maps them) |
| **W1…W11** | workstreams in rrp `docs/strategy.md` (W10 = the Ψ₀ line, carried out in psi1z) | — |
| `~/work/rrp-wt/psi0`, branch `track/psi0`, peer dir `wt/psi0` | HISTORICAL: early W10 scratch in rrp before psi1z existed (D-100); nothing there is current | not where W10 lives now |
| contact_v1/v2, grasp_v1/v2/v2.1, sourced_v1 | physics versions in rrp (legged contact, arm grasp contact, actuator limits); recorded in provenance | unrelated to Ψ₀ versions |

## dependency direction (never the reverse)
psi1z → rrp. psi1z installs rrp as a library, pinned by git sha in
[psi1z/pyproject.toml](https://github.com/JacobFV/psi1z/blob/main/pyproject.toml) and uses only the stable API in
[docs/core_api.md](core_api.md) (`rrp.core`, version 1.0). It registers its `g1_simple` family through the extension hook
(`register_family` / the `rrp.families` entry point) without editing rrp. rrp never imports psi1z.
Shared pieces (packet contract, system 0 base, probes and bounded NLL, statistics, packet-edit harness, provenance, RunConfig /
Pipeline / run-dag, broker client) are changed **here** and then psi1z bumps its pin. psi1z never copies rrp code.
Both repos lease compute from the same broker (`rrp ops run`, peer broker under `/dev/shm/rrp-brandonin/repo/ops`); the rules
in [AGENTS.md](../AGENTS.md) (D-106 peer admission, D-115 host = no heavy compute, D-117 memory declarations) apply to both.

## where to look for what
| question | rrp | psi1z |
|---|---|---|
| overall plan and workstream status | [docs/strategy.md](strategy.md) (W10 row) | [README.md](https://github.com/JacobFV/psi1z/blob/main/README.md) |
| current evidence | [research/reports/evidence_matrix.md](../research/reports/evidence_matrix.md) | step tables in [research/notes.md](https://github.com/JacobFV/psi1z/blob/main/research/notes.md) |
| training/physics problem checklist | [docs/robot_training_considerations.md](robot_training_considerations.md) | inherits it |
| repo structure / API | [docs/repo_structure_audit.md](repo_structure_audit.md), [docs/core_api.md](core_api.md) | `src/psi1z/` |
| operating rules | [AGENTS.md](../AGENTS.md) | README "rules" (points here) |

## decision crosswalk (rrp D-xxx ↔ psi1z P-xxx)
| rrp | psi1z | topic |
|---|---|---|
| D-098 | P-001, P-004 | W10 opened; Isaac Sim licence accepted by the owner; matched fine-tune design (Ψ₀ direct vs Ψ₀ + structure) |
| D-100 | P-001 | third repo psi1z; rrp becomes an installable core |
| D-104 | P-002, P-003, P-005 | gate 0: Isaac 5.1 on aarch64; path-tracing render; released XMovePick does not reproduce; reproduction first |
| D-106 | P-006 | peer admission and allocation after the 07:50 memory emergency |
| D-109 | P-007, P-009, P-010 | TabletopGraspMP reproduces; ISO recalibration on training frames; structured-arm vx fix (stage A v2e); per-task reproduction gate |
| D-111, D-116 | P-008 | RAM-store and memory accounting; true Isaac job footprint |
| D-115 | P-011, P-013, P-014 | host = no heavy compute; step-2 training moved to the peer |
| D-117 | P-016 | memory declarations ≥ 1.35 × peak (memory.high = 0.8 × declared) |
| D-119 | P-017 | psi1z GitHub remote (owner); pushes allowed after pre-push check |
| D-120 | P-012, P-015 | step 1: 3/6 released checkpoints reproduce (TabletopGraspMP, BendPickMP, HandoverTeleop); XMovePick closed out; early stops |
| D-109 | P-018 | step-2 packet-edit design on TabletopGrasp: hand-binding edit (left hand; demos use right in 96/100), 10 cm goal shift, random-direction controls |
| D-117, D-121 | P-019 | Isaac eval declarations (20G + 20G GPU from a measured 14.1 + 16.0 GB peak); W10 yields the peer to the W7 v6 BC expert |
| W11 (strategy) | pin in pyproject | rrp.core 1.0, Python 3.11 support, extension hooks |
Keep this table current: every psi1z P-entry that reflects a lead decision cites its rrp D-number, and every rrp D-entry about W10
cites the P-numbers.
