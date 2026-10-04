# Fresh audit of `rrp_report.tex` before publication (2026-10-04)

AGENTS.md requires a fresh audit before published conclusions. This audit read the paper end to end (last merged
f0c7cef9, snapshot 2 Oct) and checked every quantitative claim and result statement against the recorded evidence:
`research/decisions.md` (D-085, D-087, D-124, D-134..D-147 and addenda through 2026-10-04), `research/registry.jsonl`,
`research/tracks/{psi0,pointer,armdiv,relations,humanoid}.md`, `~/work/rrp-data/campaign/STATUS.md` and
`~/work/rrp-data/campaign/logs/psi0_progress.log`. Registry counts in the text were re-checked against code
(`rrp.harness.data.relgen`: 18 LABELS, 7 PARTS, 6 TRANSFORMS; `figures/table_relations_counts.json`: 76 rows, 75 implemented).

Verdicts: **ok** = matches evidence as written; **fixed** = corrected or relabelled in this revision; **removed** = deleted.
Tally: **ok 33, fixed 27, removed 1** (61 claims; rows 34 and 52 count as ok, with a label or state added).

## Banner, abstract, front matter

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| 1 | Banner count 251 embodiments; breakdown 150 arms (108 procedural), 19 dual, 80 legged (64 phum), G1, pointer; 150 trained | `figures/embodiments.json`, `table_embodiments.tex` (rows sum to 251 / 150) | ok |
| 2 | Banner "total unique **robot** embodiments!" | the 251 include the ComputerWorld pointer (not a robot); all are simulated | fixed: "251 unique simulated embodiments", pointer marked "a UI agent, not a robot", "none is a physical robot" |
| 3 | 76 factors, 75 implemented | `figures/table_relations_counts.json` | ok |
| 4 | Semantic supervision raises arm competence +0.23 [0.18, 0.28] | D-134 | ok |
| 5 | Task-context halt carried into legged behaviour | D-124 | ok |
| 6 | "packet edits steer a pointer" | D-142: edits partial (probe reads new target 8-10/25) | fixed: "partially steer" |
| 7 | Latent route does not beat BC on a new arm | D-135, D-136 | ok |
| 8 | "this report states its gates and **no campaign result**" | D-147 addenda 10-03/10-04 record G3 FAIL, T7 tabletop 7/20, T9 halted, sealed-body failures | fixed: abstract now states the recorded campaign outcomes, mostly negative |
| 9 | (new) dated Status note: living report, campaign ongoing | owner request | fixed (added) |

## Sections 2-5 (design, data, environments)

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| 10 | Packet shapes per family (Table 1) | code (unchanged since f0c7cef9 audit of tables) | ok |
| 11 | Clocks: arm replan 0.4 s, System 0 20 Hz, physics 500 Hz; legged 50 Hz, 0.8 s packet | Fig. 1 facts file / code | ok |
| 12 | 13 operators listed | `docs/relations.md` (D-144: 13 operators) | ok |
| 13 | LABELS 18, PARTS 7 (names), TRANSFORMS 6 (names) | `rrp.harness.data.relgen` registries, counted 2026-10-04 | ok |
| 14 | Curriculum implemented and plumbing-smoked; T9 in progress, no curriculum result | STATUS relations rows | ok |
| 15 | Table 2 envs: SIMPLE step-1 reproduced on 3 of 6 tasks | registry `w10_step1_repro`, psi0.md | ok |
| 16 | Table 2 arm pool "24 procedural pa2s*" | D-137 addendum (first 24 admitted seeds enter the pool) | ok |
| 17 | Fig. 2 / Fig. 5 show factor values on simulator states, not trained attention; sim-truth labels marked training-only | captions, `make_figures.py` | ok |
| 18 | Fig. 3: arm/dual frames are scripted-teacher (privileged); others reset states | caption | ok |

## Section 6: recorded results

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| 19 | Snapshot "2 Oct 2026" | campaign advanced to 4 Oct | fixed: 4 Oct 2026 |
| 20 | "three environments and one third-party humanoid policy" | the Psi0 direct / structured arms are OUR fine-tunes; released is upstream | fixed: "our fine-tunes of one third-party pretrained policy"; `upstream` source label defined |
| 21 | semfix 425/480 vs nosem 314/480, 8/8 cells | D-134 | ok |
| 22 | Latent below BC on panda 135/180 vs 58/60; BC 60/60 parm6 | D-134 | ok |
| 23 | Binding seed-dependent for both | D-134 | ok |
| 24 | Unbounded NLL scaled updates 0.002-0.004 vs 0.70-0.92; 81/90, 83/90, 21/90 | D-085, D-087 | ok |
| 25 | Halt: anymal_c -0.40 [-0.50, -0.30], go2 -0.66 [-0.72, -0.59], 3/3 seeds, p = 0.05; irrelevant <= 0.02 m; success 78 vs 83, 87 vs 87; t1 BC 0/30 | D-124 | ok |
| 26 | Pointer edits 163 px -> 41 / 22 px; random 102-105; nosem 128-141 vs 137-144; probe reads edit 8-10/25; success unchanged | D-142 | ok |
| 27 | xarm7 zero-shot 0/200; BC SFT 69-140 / 143-192; refit <= 38; flow SFT 0; panda_tf3 154 / 32 / 199 | D-135 | ok |
| 28 | Joint adaptation beats refit 12/12 (595 vs 130), below BC 12/12 (595 vs 842), gaps -0.07..-0.34, nosem -0.30..-0.80, 595 vs 151, post-hoc | D-136 | ok |
| 29 | D-135 addendum: protocol (one module at a time), ~2x slower per update than BC | D-135 addendum | ok |
| 30 | Pointer sealed_id BC 398, semfix 372, nosem 368, eng 185; McNemar 2.2e-7, 0.68 | D-142 result | ok |
| 31 | `eng` described as "the scripted System-0 route" | eng = LEARNED System 1 driving the SCRIPTED System 0 | fixed: source of each half stated |
| 32 | Unseen words/names 0/50 typing, 0/50 form; calc 45-50/50; "no copy" | D-142 (v1 packet) | fixed: scoped to the v1 packet; table row says "v1 packet" |
| 33 | "a copy head is the pre-registered test" (forward-looking) | T8 round 1 completed 10-03 | fixed: round 1 result added (free head with v2 key code types held-out-bucket strings on DEV; copy head worse on drag 5.3 vs 37.3) and labelled dev-only |
| 34 | Psi0 D-141: released 20/20 [0.84, 1.00], direct 19/20 [0.76, 0.99], structured 0/20 [0.00, 0.16] | D-141 | ok; released relabelled "upstream, not ours" (fixed label) |
| 35 | Offline diagnosis: System 0 barely read the packet; torso state never in training | D-141 addendum | ok |
| 36 | Packet-use gate PASS gap 0.0782 (0.0843 vs 0.0060) | D-147 T7 addendum, STATUS | ok |
| 37 | "the closed-loop comparison is queued" | psi0.md 2026-10-03: structured 7/20 [0.18, 0.57] vs direct 19/20, McNemar p = 0.0018 | fixed: result stated; heading changed from "a failed diagnosis, not a negative result" to "fixed, but below direct fine-tuning" |
| 38 | Table 4 "Closed-loop structured evaluations are queued, not results" | same | fixed: new row "closed loop after the fix" |
| 39 | (new) G3 lineage gate FAIL 362/480 = 0.754 [0.714, 0.791] vs 425/480; -0.131 [-0.179, -0.083]; failed_hypothesis; no sealed cell | D-147 addendum 2026-10-03 (T6) | fixed (added text + table row) |
| 40 | (new) BendPickMP released (upstream) 20/20, direct 15/20 [0.53, 0.89], gate 0.1483, structured in progress | psi0.md source-label note; `campaign/logs/psi0_progress.log` 10-03 19:27 and 10-04 06:40 | fixed (added, structured labelled in progress) |
| 41 | (new) T9 v3dart halted at floor 0/360 vs 0/360 | D-147 addendum 2026-10-03 (T9) | fixed (added, labelled uninformative) |
| 42 | (new) Sealed bodies: n1 0/40, toddlerbot_2xc 0/40; 2xc adapted 0/20 walk, 0/20 turn | STATUS T3 row; D-147 addendum 2026-10-04 | fixed (added) |
| 43 | (new) Legged vs legged-none fair-input caveat; legged-tokens control running | D-146 amendment 2026-10-04; D-147 addendum (legged-tokens) | fixed (added; no legged-vs-none claim) |

## Section 7: campaign, Table 5, Fig. 4

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| 44 | Ten-node campaign T0-T9; broker <= 2 GPU jobs per machine, humanoid first | D-147, STATUS | ok |
| 45 | Humanoid: "terrain trackers failed round 1 ... terrain tasks currently unsupported" | round 2 also failed; terrain closed for D-147 (addendum 10-03); h_walk/h_turn the only T4 tasks | fixed |
| 46 | T6 H1 "65 distinct arm kinematics" | pool = 65 arm x gripper KEYS (13 v6 + 48 procedural + 4 menagerie), D-137 addendum | fixed: "65 arm x gripper keys" |
| 47 | T6 "BC half passed" (G3 pending) | G3 lineage half FAILED | fixed |
| 48 | T7 "No numerical threshold is recorded for the closed-loop comparison" | D-147 T7 addendum 10-03 fixes a reading rule (structured > direct, McNemar p < 0.05) | fixed |
| 49 | T8 rules P-SEEDS / P-UI / P-COPY / P-KEY | D-147, pointer.md | ok; round-1 state added |
| 50 | T9 "No numerical success threshold is recorded for T9" | T9 re-plan (v6) pre-registers a Newcombe criterion + competence >= 0.5 validity gate | fixed |
| 51 | Table 5 T0 verified | STATUS | ok |
| 52 | Table 5 T1 round 1 numbers (t1 2/20, g1 0/20, h1 10/20, gap t1 0/20, h1 20/20 with stand falls) | STATUS T1 rows | ok; round 2 + closure added (fixed state) |
| 53 | Table 5 T2 joint margins 0.0183 / 0.0191 / 0.0160; "teacher check not yet run" | STATUS, T4 gate: h_carry blocked (g1/h1 no plan) | fixed (h_carry blocked_external, steps_ub h1 only) |
| 54 | Table 5 T3 "planned", T4 "planned" | T3 installed under exception; T4 gate completed (h_walk 0.95, h_turn 1.00) | fixed |
| 55 | Table 5 T5 "planned" | T5 running with the fair-input caveat | fixed |
| 56 | Table 5 T6 "running", T7 "phase 2 running", T8 "copy running", T9 "factor-set training running" | STATUS 10-04 | fixed (all four rows rewritten) |
| 57 | Table 5 T6 BC gate 117/120 for both BC experts | STATUS armdiv rows | ok |
| 58 | Table 5 T8 collection 4 x 6,000, oracle 50/50 (privileged reference) | STATUS pointer rows | ok |
| 59 | Table 5 T8 "0 teacher failures" | STATUS | removed (true but cut for space; kept in STATUS) |
| 60 | Fig. 4 roadmap gate states (2 Oct: T6 lineages / T7 phase 2 / T9 factor sets "now"; no failed state) | STATUS 10-04 | fixed: `make_figures.py roadmap` regenerated with a `failed` state (T1, T6 G3, T9 v3dart) and the 4 Oct front |

## Section 8: limitations

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| 61 | (i), (v), (vi), (vii), (ix) | as above | fixed: G3 failure and sealed-body failures in (i); v1 vs v2 in (v); 7/20 vs 19/20 in (vi); fair-input caveat in (vii); "living report" in (ix). (ii), (iii), (iv), (viii) ok |

## Source labels

Every Psi0 released number is now labelled **upstream, not ours**; direct and structured are **ours**. Teacher checks are
labelled **scripted teacher (privileged)** over a learned tracker; oracle packets as **privileged reference**; the pointer
`eng` route as learned System 1 -> **scripted** System 0. Every in-flight node (T3 Level-1 n1 / 2xm, T5, BendPick
structured, HandoverTeleop, P-SEEDS, P-UI, T9 v6) is labelled *in progress* / *running* / *planned*, never as a result.

## Build checks

`tectonic` build: 22 pages (was 21), 0 overfull boxes, 0 "float too large" (the enlarged results table was re-balanced),
PDF ~3.0 MB. Pages 1, 6, 7, 9, 10 rendered to PNG and inspected.
