# experiments roadmap: open questions we intend to test

Owner request (2026-09-27, D-126). Everything here is still UNCERTAIN: a question we are interested in but have not answered.
Each item says what it would settle, what it depends on, a rough peer cost, and its code status. Code status is tracked
separately from experiment status: per D-126, everything that is purely a code problem is implemented now behind ablation
flags, so experiments can launch as soon as compute allows. Related docs: docs/strategy.md (workstreams, allocation),
docs/intentions_backlog.md (stated-but-unimplemented items, D-123), research/reports/evidence_matrix.md (what IS established).

Status legend (experiment): running · queued · planned · open (needs design) · done → decision.
Code: ✅ exists · 🔧 implementing (D-126) · ❌ not a code problem / needs research design.

## A. Arm: semantic packet under realistic physics
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 1 | Do the arm semantic results (D-095) survive grasp_v2? Re-eval of existing routes | – | ~free | done → D-127 (yes; semfix 311/480 vs nosem 40/480; frozen sem s1 collapses) | ✅ |
| 2 | Is v6dart data good enough: v6 BC expert ≥ v2 expert (D-121 condition)? | – | ~3 h | running | ✅ |
| 3 | Do semfix vs nosem arm results hold on realistic physics + teacher v2 + v6dart? | 2 | 24–30 h | queued | ✅ |
| 4 | Can upstream recipe choices (Stage A β, BC-DAgger schedule, refit lengths) rescue nosem? (D-099 fairness) | 3 | ~1 day | planned | 🔧 ablation overlays |
| 5 | Does small-amplitude descent-phase DART restore near-grasp recovery without penetration? (D-121 fallback) | 2 | ~0.5 day | conditional | 🔧 |
| 6 | Can GRPO (success reward, anchor/forgetting evals) lift the latent route past the scripted ceiling vs BC at equal budget? | 3 | 6–10 h | planned | 🔧 stage + evals |
| 7 | Does overlapping-chunk blending remove chunk-boundary steps without losing success (BC and system 0)? | – | 0.5–1 day | planned | 🔧 |
| 8 | Does limit-aware IK remove procedural-arm joint-limit contact so the gate can be enforced? | – | ~1 day + data | planned | 🔧 |

## B. Cross-body transfer (the untested core claim)
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 9 | Does the latent route transfer to the sealed target bodies (xarm7_pg2/tf3, panda_tf3): zero-shot and with small adaptation budgets? | 3 | 1–2 days | planned (priority) | 🔧 DAG nodes |
| 10 | Fair baselines on those bodies: BC seeds 1702/1703, SFT budgets on the same data | – | ~1 day | planned (priority) | 🔧 DAG nodes |
| 11 | Legged held-out body transfer (heldout + refit stages) | – | ~1 day | planned | 🔧 DAG template |
| 12 | Do anchor-relative packets transfer better than base-frame ones? (W12 H2) | 20 | in W12 | planned | ✅ (W12 A) |

## C. Legged and humanoid
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 13 | Can we train humanoid trackers under sourced limits that pass the gates (t1 lab gate, h1 clock-driven gait, g1 without stomp/overspin), and does the halt effect hold on t1 then? | – | 2–4 days | planned | 🔧 options (clock ref gait, yaw-progress cap) |
| 14 | Do trackers and latent routes survive actuator dynamics + latency as default (sourced speeds)? | 13 | 1–2 days | planned | 🔧 switch |
| 15 | Does a terrain curriculum (and MJX for throughput) move the 8 cm terrain break-point? | – | 2–4 days | planned | 🔧 curriculum; MJX prototype |
| 16 | Is the semantic robustness cost (D-112: −4.6 points) general beyond anymal_c (go2, later t1)? | – | ~1 day/body | planned | ✅ |
| 17 | Why does semfix steer 2× more on anymal_c but not go2? Does it replicate? | – | ~1 day | open | ✅ |

## D. Feature-centric coordination (W12)
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 18 | Grasp fix for pegs/bars and dual teacher v3 (smooth, contact-confirmed, limit-aware, phase-gated DART), then data | – | 1–2 days | queued (W7 grasp part) | 🔧 |
| 19 | H1 go/no-go: base-frame vs anchor-relative supervision vs anchor-input system 0 (seed 0) | 18, 3 | 9–24 GPU-h | planned | ✅ |
| 20 | Full W12 matrix (+event-aligned knots, anchor-input control, 3 seeds) with anchor-shift and contact-sequence edits | 19 | 2–6 days | planned | ✅ |
| 21 | Legged stance drift with the new contact metrics on existing W8 checkpoints | – | ~0.5 day | planned | ✅ |
| 22 | Coordination tasks: pivot against a surface, carry a tray level, legged foothold stepping | – | design + data | open | 🔧 task defs + scenes |

## E. Ψ₀ line (psi1z, W10)
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 23 | TabletopGrasp: released vs Ψ₀ direct fine-tune vs Ψ₀ + our structure; hand-binding and goal edits | – | ~1 day | running | ✅ |
| 24 | BendPickMP, HandoverTeleop step 2, with a grasp-region affordance head | 23 | 2–3 days | planned | ✅ head (psi1z P-022, default off; needs re-replay labels) |
| 25 | Can a better render (materials/lighting) reproduce the walking tasks (XMovePick, LocoPick, XMoveBendPick)? | – | open-ended | open | ❌ research |
| 26 | Higher DR levels (1–2) on reproduced tasks | 23 | ~1 day/task | planned | ✅ eval flags (psi1z P-023) |

## F. Representation and deployment credibility
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 27 | Do routes still work with estimated (not truth+noise) base velocity? | – | 1–2 days | planned | ✅ `--base-state-source estimator` (bse-1) |
| 28 | Automated per-group privileged-information audit (ablate input groups, measure causal use) | – | ~1 day | planned | ✅ `rrp.evaluation.privileged_audit` (paudit-1) |
| 29 | Does an OOD packet detector + fallback catch the packet edits that make t1 fall? | – | 1–2 days | planned | ✅ `--packet-ood`, `rrp.training.packet_ood_fit` (pood-1) |
| 30 | Safety layer (clamp, rate limit, safe stop, fall recovery): cost to success, benefit under perturbation | – | 1–2 days | planned | ✅ `--safety` (safety-1) |
| 31 | VLM system II: language → packet, real evaluation | – | 3+ days | open | ✅ harness only: `rrp.evaluation.system2` (s2h-1), `--system2` |
| 32 | Minutes-long runs (drift) and latency on current routes | – | ~0.5 day | planned | ✅ `--eval-mode long`, `--measure-latency`; `rrp latent latency --representation` |

Section F code (D-126, 2026-09-28): commands and flags in research/tracks/d126_deploy.md "exact configs". All options
default off; defaults golden-identical to pre-D-126 rows.

## G. New task families
| # | question | depends on | cost | experiment | code |
|---|---|---|---|---|---|
| 33 | First dual-arm learned models (M=2 "across bodies") | 18 | 2–3 days | planned | 🔧 dual pipeline stages |
| 34 | Loco-manipulation (walk to a table, then pick) | 13 | 3+ days | open | 🔧 task + scene |
| 35 | Richer arm objects (sizes, masses, friction, shapes) | – | 1–2 days + data | planned | 🔧 scenario flags |

## ordering once current runs finish (lead)
1–3 (running/queued) → 9–10 (core claim) → 23 (queued) → 13 (unblocks humanoid) → 18–19 (W12 go/no-go). The rest fits
around peer capacity, which is about one major experiment per 1–2 days on the single peer GPU.
When an item is answered, record the decision number here and move the result into the evidence matrix.
