# .old/research — notes of closed tracks (D-145)

Path map: `research/tracks/<x>.md` -> `.old/research/tracks/<x>.md`. The decisions that cite each note stay whole in
`research/decisions.md`. Raw results stay in `artifacts/` (the evidence store). Later purge units append rows.

Rule: none of these is a track in `schema.toml [tracks]` (D-145); their decisions stay whole in `research/decisions.md`. The open items of the rel-* notes and sweep-flags are collected in `research/tracks/relations.md`; nothing else in this group is resumable without a fresh decision.

| note | what it was | decisions citing it | state at closure | moved in |
|---|---|---|---|---|
| `tracks/acceptance.md` | track: acceptance (causal edits, composition, latency on the corrected latent path) | D-041, D-059, D-062, D-074, D-077 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/armdiag.md` | armdiag: why the latent route does not adapt to xarm7 (D-135 follow-up) | D-135, D-136 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/armexpert.md` | W6/W7 arm expert program: v4dart data and the v2 BC expert (step 2), grasp contact v2, v5dart regeneration, GRPO / target-body work, backlog | D-097, D-102, D-110, D-121 | closed (owner, D-145): the arm templates stay in `recipes/templates/` as generic family recipes; instance DAGs are in `.old/dags/` | P4d |
| `tracks/baselines.md` | track: baselines (latent_slice1 four-way comparison, baseline methods) | D-035, D-050, D-065 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/binding.md` | track `binding`: make the Stage-A packet follow the supplied task binding (D-032) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/bodies.md` | track: sprint_bodies (demo sprint 2026-09-25) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/contact.md` | contact track: realistic foot-floor contact (contact_v2) and real walking gaits | D-093, D-101, D-103, D-107 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/core.md` | core track (W11: rrp as an installable core for psi1z, D-100) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/d126_arm.md` | track d126arm (D-126 arm/training code items: roadmap #4, #5, #6, #7, #8, #9, #10, #35) | D-128 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/d126_deploy.md` | D-126 deployment credibility and infra (track notes, append-only) | D-129 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/d126_legged.md` | D-126 legged/physics implementation (code only; nothing trained) | D-128 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/d126deploy-ladder.md` | d126deploy-ladder (D-126 sub-track: ladder / legged summaries into the library) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/d126deploy_sys2.md` | track d126deploy-sys2 — system II harness (roadmap #31, harness only; D-126 sub-agent) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/demo.md` | track: demo (sprint_demo, 2026-09-25 19:00 → 2026-09-26 05:00 PDT) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/dualarm.md` | track: dualarm — dual-arm tasks on the corrected latent-packet architecture | D-043 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/grpo.md` | track: grpo — packet-policy GRPO for system i (corrected architecture) | D-042 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/hygiene.md` | hygiene track (W2, docs/strategy.md) — 2026-09-26 | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/ladder.md` | track: ladder — closed-loop failure localization (correction 2026-09-25, item 3) | D-044, D-052, D-072, D-081, D-089, D-091, D-095, D-099 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/legged8.md` | W8 legged semantic-supervision study on contact_v2 (anymal_c, go2, t1 waves, incidents, resource declarations) | D-105, D-113 | closed (owner, D-145): `recipes/templates/legged_lineage.yaml` / `legged_heldout.yaml` stay; the per-body result tables moved with P5 (see `tracks/legged8/` below); the room's radar sources are in `artifacts/runs/research_tables/legged8/` | P4d |
| `tracks/legged_vlm.md` | track legged_vlm — legged/humanoid breadth and VLM system II on the latent-packet path | D-040, D-067, D-069, D-071, D-073, D-085, D-087, D-088 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/pipeline.md` | pipeline track (W5: unified pipeline + DAG orchestration, audit phases 3-4) | D-096 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/provenance.md` | provenance track (W3: provenance and contracts, phase 1) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-geo.md` | track rel-geo: wire geometry/interaction factors so they actually reach attention | D-144 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r1.md` | track: rel-r1 (relation-factor fanout, unit R1: arm/dual probes -> ReadoutProbe) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r10.md` | rel-r10 track (D-144 fanout unit R10: relgen stage + shards + mix) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r11.md` | rel-r11 track (D-144 fanout unit R11: responsive, steerable scheduler + composition + interference) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r12.md` | R12: arm / dual token-set fields (D-144 fanout, docs/relations.md section 10) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r13.md` | rel-r13 track (D-144 fanout unit R13: geometry factors) | D-144 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r14.md` | R14: geometry labels + parts (D-144 fanout, docs/relations.md section 10) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r15.md` | track: rel-r15 (relation-factor fanout, unit R15: membership / graph factors) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r16.md` | R16: contact / grasp / handover (D-144 fanout, docs/relations.md section 10) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r17.md` | track: rel-r17 (D-144 relation-factor fan-out, unit R17: support / force flow) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r18.md` | track: rel-r18 (relation-factor fanout, unit R18: task / temporal / epistemic) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r19.md` | track: rel-r19 (D-144 relation-factor fan-out, unit R19: legged labels + foothold) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r1c.md` | track: rel-r1c (relation-factor fanout, R1 deferred-scope follow-up: delete nets/latent_probes.py for real) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r2.md` | track: rel-r2 (relation-factor fanout, unit R2: arm latent config + flags -> specs) | D-144 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r20.md` | track: rel-r20 (D-144 relation-factor fan-out, unit R20: UI factors) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r21.md` | track: rel-r21 (relation-factor fanout, unit R21: viz factor / attention inspection) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r2c.md` | track: rel-r2c (relation-factor fanout, R2 deferred-scope follow-up) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r3.md` | track rel-r3: arm system 0 on RelBlock + route.own_assembly | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r4.md` | R4: legged nets on RelBlock / route.own_assembly / ReadoutProbe | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r5.md` | track: rel-r5 (relation-factor fanout, unit R5: Ψ₀ nets) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r6.md` | track: rel-r6 (relation-factor fanout, unit R6: pointer nets) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r7.md` | track: rel-r7 (D-144 relation-factor fan-out, unit R7: StateView on the MuJoCo sessions) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r8.md` | track rel-r8: StateView — Warp, ComputerWorld, SIMPLE (D-144 fanout row R8) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/rel-r9.md` | track: rel-r9 (relation-factor fanout, unit R9: generic transforms) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/restructure.md` | restructure track (W4: package restructure with shims, phase 2 of the audit) | (named in the note only) | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/robust.md` | track robust (W6): robustness sweeps + motion-quality gates | D-108, D-112 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/room.md` | room track (D-131): live visualization room | D-133 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/sweep-flags.md` | sweep-flags: close rel-r2c's open question 1 (probe_lv_min/semantic_weight off live paths for arm+dual+legged, | D-144, D-144 | closed (D-145: not in `schema.toml [tracks]`) | P5 |
| `tracks/w12.md` | W12 feature-centric coordination: anchor-relative packets, the dual scripted-teacher audit (section 7), the D-126 dual teacher v3 / dataset gate / pipeline (section 10) | D-125, D-130 | closed (owner, D-145): `recipes/templates/dual_lineage.yaml` stays as the family recipe | P4d |

## result files, scripts, reports

| path | what it was | citing | state | moved in |
|---|---|---|---|---|
| `tracks/ladder/{armnosem,armv6,sprint_final}/` | closed-loop ladder raw summaries (`summary.json`, jsonl) tabulated in `tracks/ladder.md` (arm no-sem seed 2 / ablation / ladder_v1, arm v6, sprint final) | D-091, D-095, D-099, D-134, D-135 | frozen evidence, unread by live code. EXCEPTION: `armv6/compare_v6.json`, `armv6/targets_v6.json` are read by the room radar and live in `artifacts/runs/research_tables/ladder/armv6/` | P5 |
| `tracks/armdiag/{closedloop,d136,offline,stepA,joint300x300_result.json}` | arm latent-adaptation diagnosis summaries (D-135 / D-136) | D-135, D-136 (via the note) | frozen evidence. EXCEPTION: `d136_compare.json` (room radar source) is `artifacts/runs/research_tables/armdiag/d136_compare.json` | P5 |
| `tracks/legged8/` | W8 per-body compare tables, gates and clipscale probes on contact_v2 | D-105, D-113, D-124 | frozen evidence. EXCEPTION: `legged8_compare_{anymal_c,go2}.{json,md}`, `summary_contact_v2.{json,md}` (room radar sources) are `artifacts/runs/research_tables/legged8/` | P5 |
| `scripts/2026-09-{26,27,28}/` | one-off analysis scripts (armexpert compare / trace / gate replay, ladder peeks, W4/W5 parity, d126 deploy smoke) and two smoke configs | (named in the notes only) | not re-runnable as recipes; results are in `artifacts/` | P5 |
| `reports/` | superseded prose reports: latent_slice1 progress / baselines tables, dual-arm task notes, functional composition, legged breadth, VLM status, GRPO ratio probes | D-023, D-065 | superseded by `research/reports/evidence_matrix.md` and `docs/relations.md` | P5 |
| `naming.md` | decoder from historical lineage codes (jointfix, sfjf, lv4, ...) to variant x seed x stage | (cited by the notes) | still the decoder for pre-schema run directory names in `artifacts/runs/` | P5 |

`research/pairs/assign_pick_place_v1.json` was not legacy: it moved to `research/splits/assign_pick_place_v1.json` (the paired-scene spec of the manipulator-assignment edit).

