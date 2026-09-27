# naming map: historical lineage codes → variant × seed × stage

Target scheme (docs/repo_structure_audit.md, W4/W5): `variant ∈ {sem, nosem, semfix}` × `seed s{n}` × `stage ∈ {rep, flow,
flowft, gdagN, rz, dagN-{bc|gen}}`, per body family. Existing paths are NOT renamed (running jobs and chain markers depend on
them); this table is the decoder. Verified against config contents on 2026-09-26 (W2); "uncertain" marks inferences.

Variants:
- **sem**: Stage A `latent.semantic_weight` 1 with the probe log-variance floor at its default −8 (the D-085 defect:
  unbounded NLL starves system 0). Flow `packet_semantic_weight` 1.0 on the arm, **0.5 on legged**, so "sem" is not one
  recipe across families.
- **semfix**: sem + bounded NLL, `latent.probe_lv_min: -4` (D-085/D-086). Also written `sfjf`, `sf`, `fixsem`, `lv4`, "fixed sem".
- **nosem**: capacity-matched, `semantic_weight` 0 and flow `packet_semantic_weight` 0.
- Every arm lineage from `jointfix` on also has `zero_prev_action: true` (bug B-1 fix, D-044/D-045) and
  `realizer_anchor: true`, pack `latent_pp_v3dart_s1_H16`, 13 training bodies.

## arm lineages (pick_place)

| code | variant | seed | canonical | defined | example config |
|---|---|---|---|---|---|
| `jointfix`, `jf` (Stage A `ladder_latent_sem_b1fix_anchor`) | sem | s1 | `arm/sem/s1` (the frozen sem route) | D-047, D-072, D-078 | `configs/ladder/flow_jointfix.json` |
| `sfjf` (dirs `armsemfix`, Stage A `ladder_latent_semfix_b1fix_anchor`) | semfix | s1 | `arm/semfix/s1` | D-086, D-091 | `configs/ladder/armsemfix/flow_sfjf.json` |
| `nsjf` (dir `armnosem`) | nosem | s1 | `arm/nosem/s1` | D-089, D-091 | `configs/ladder/armnosem/flow_nsjf.json` |
| `sejf2` | sem | s2 | `arm/sem/s2` | ladder.md "ARM SEED-2 REPLICATION" (R0) | `configs/ladder/armseed2/sejf2/flow_sejf2.json` |
| `sfjf2` | semfix | s2 | `arm/semfix/s2` | same | `configs/ladder/armseed2/sfjf2/flow_sfjf2.json` |
| `nsjf2` | nosem | s2 | `arm/nosem/s2` | same | `configs/ladder/armseed2/nsjf2/flow_nsjf2.json` |
| `nszn` | nosem, recipe ablation: system-0 z-noise 0 (instead of 0.3); qd still removed | s1 | `arm/nosem-zn0/s1` | ladder.md "ARM NOSEM RECIPE ABLATION" | `configs/ladder/armnosemabl/nszn/rz_nszn_gendag1_noqd.json` |
| `nsqd` | nosem, recipe ablation: joint velocity KEPT (`realizer_drop_qd` false, no qd dropout); z-noise 0.3 | s1 | `arm/nosem-qd/s1` | same | `configs/ladder/armnosemabl/nsqd/rz_nsqd_gendag1_qd.json` |
| `nszq` | nosem, both ablations (z-noise 0 AND qd kept) | s1 | `arm/nosem-zn0-qd/s1` | same | `configs/ladder/armnosemabl/nszq/rz_nszq_gendag1_qd.json` |
| `b1fix` | any | — | flag: `zero_prev_action` (B-1 fix) | D-044, D-045 | `configs/ladder/rz_sem_v1_b1fix.json` |
| `anchor` | any | — | flag: `realizer_anchor` (column 28 = joint displacement since packet anchor) | D-045, ladder.md | `configs/ladder/rep-latent_sem_b1fix_anchor.json` |
| `latent_{sem,nosem}_v1`, `flow_latent_sem_v{1,2,3}` | sem / nosem | s1 | pre-B-1 lineage (contaminated; v2 = standardized flow target D-031, v3 = +`packet_tau_min`; all reuse the v1 encoder) | D-031, D-045 | `configs/latent/rep-latent_sem_v1.json` |
| `flow_jointfix_nosem` | sem encoder + nosem flow | s1 | Stage-B-only ablation; no results found (uncertain purpose) | commit 9fce481 | `configs/ladder/flow_jointfix_nosem.json` |

Recipe ablations (`nszn`, `nsqd`, `nszq`) differ from `nsjf` ONLY in the gendag1/2/3 refits; every upstream stage
(Stage A, flow 20k, flow_ft, bc1-3, gen1, bcdag1/bcdag1_long/bcdag2) is config-identical and is reused from `nsjf`.
Suffix `_qd` (instead of `_noqd`) on their refits = joint velocity kept.

Seed 1 → seed 2: every training seed +1000 (Stage A 1706 → 2706; flows 1701/1702/1703/1860 → 2701/…/2860; refits
1711/1712/1715/1761/1790/1820 → 2711/…/2820) and every DAgger collection seed +1,100,000 (`SOFF`). Nothing else differs.

### arm stages and suffixes

| code | canonical stage | meaning | defined | example |
|---|---|---|---|---|
| `rep-…` | rep | Stage A: encoder E + system 0 R + probes P | D-047 | `rep-latent_sem_b1fix_anchor.json` |
| `flow_<lin>` | flow | system i, 20k steps | ladder.md | `flow_jointfix.json` |
| `_ft`, `flowjfft10k` | flowft | flow +10k steps, lr 1e-4, from snap_final_s20000 | D-067 | `flow_jointfix_ft.json` |
| `gdag1` | gdag1 | generator DAgger round 1 (+4k, 50% pack) | D-067, D-070 | `flow_jointfix_gdag1.json` |
| `gdag2` | gdag2 | round 2, stopped at ~300 steps | ladder.md | `flow_jointfix_gdag2.json` |
| `gdag2h` | gdag2 (pack-free) | `h` = pack-free (`gen_dagger_frac` 1.0), 1.5k steps from gdag1; the FROZEN flow | D-078, D-080 | `flow_jointfix_gdag2h.json` |
| `gdag3h` | gdag3 (pack-free) | one more round; negative | D-081 | `flow_jointfix_gdag3h.json` |
| `gdag2h1k` | gdag2 variant | sibling scored 25/30 but not frozen (no best-of selection); no config, "1k" steps inferred (uncertain) | D-078 | eval rows only |
| `pfA`–`pfD` | flow variants | post-freeze retrains mixing the pack back in; all negative | D-081 | `flow_jointfix_pfA.json` |
| `rz_…` | rz | system-0 refit on the frozen encoder (`scripts/ladder_refit.py`) | D-046 | `rz_jointfix_bcdag1.json` |
| `bcdag1/2/3`, `_long`, `jfbcdag*` | rz on dag-bc | refit on BC-expert DAgger buffers (stateless BC `direct1701_u12000` labels); `_long` = 16k steps | D-053, D-055, D-066 | `rz_jointfix_bcdag1_long.json` |
| `gendag1/2/3` | rz on dag-bc + dag-gen | refit also on generated-packet (R2) buffers, z-noise 0.3 | D-063, D-066, D-068 | `rz_jointfix_gendag1.json` |
| `_noqd` | flag | joint-velocity input removed from system 0 (`realizer_drop_qd`) | D-056 | `rz_jointfix_gendag3_noqd.json` |
| `gendag3_noqd` | rz (final) | final system 0 of every arm lineage | D-072, D-078 | `armsemfix/rz_sfjf_gendag3_noqd.json` |
| `qdd` | flag | qd dropout 0.5 instead of removal | ladder.md | `rz_jointfix_gendag1_qdd.json` |
| `gendag4`, `gendag4b` | rz | 4th round; negative | D-080, ladder.md | `rz_jointfix_gendag4b_noqd.json` |
| `dagger1/2`, `_df08`, `jfdag1` | rz on shadow-teacher DAgger | stateful shadow teacher as expert: confounded, withdrawn | D-048..D-050 | `rz_jointfix_dagger2_df08.json` |
| `rz_sem_v1_*`, `rz_nosem_v1_b1fix` | rz | refits on the pre-fix v1 encoders; fail | D-046 | `rz_sem_v1_b1fix_ft.json` |
| `bc1..bc4`, `gen1..gen3`, `gdag1..gdag3` (buffers) | dagN-bc, dagN-gen, gdag contexts | DAgger buffers `artifacts/runs/ladder_dagger_[<lin>_]<buf>`: bcN = BC-expert oracle-route states; genN = generated-packet states; gdagN = `.genctx.pkl` contexts for generator DAgger. Seed-1 collection seeds 3.2M–4.1M | ladder.md | listed inside rz/flow configs |

Lease labels and eval tags: `a<TG>_…` (e.g. `asf_rz_gendag2`, `ans_col_bc1_1`) with TG ∈ {sf, ns, sf2, ns2, se2} (ablations use the full code: `anszn_…`, eval tag `flownszngdag2h_rznszngendag3`); eval rows
`generated_zero_flow<TG>gdag2h_rz<TG>gendag3_noqd_s<seedset>.summary.json` (TG empty = jointfix sem).

## binding, dual arm, data

| code | variant / stage | meaning | defined | example |
|---|---|---|---|---|
| `binding_latent_{sem,nosem}_v2` | sem/nosem rep | Stage A + counterfactual binding augmentation (`binding_cf` 0.5) | binding.md | `configs/latent/rep-binding_latent_sem_v2.json` |
| `binding_paired_*_v3` | sem/nosem rep | paired pack `binding_combined_v1_H16`; B-1 contaminated, stopped | D-045 | `rep-binding_paired_sem_v3.json` |
| `bindv4`, `bv4`, `binding_paired_*_v4` | sem/nosem, s1 | v3 + B-1 fix + anchor; never competent | D-045, D-059, D-080 | `configs/ladder/rz_bindv4sem_bcdag3_noqd.json` |
| `dualarm_latent_{sem,nosem}_v1` | M=2 rep/flow | never finished | D-043 | `configs/latent/rep-dualarm_latent_sem_v1.json` |
| `pick_place_primary_v1/v2/v3/v3dart` | data | v1 feat-v1; v2 feat-v2 (D-012); v3 continuous grasp yaw (D-016); v3dart = v3 + DART noise (D-022) | D-012, D-016, D-022 | `configs/data/pick_place_primary_v3dart.json` |
| `latent_pp_v3dart_s1_H16` | pack | pick_place pack, stride 1, 16-step chunks | configs | — |

## evaluation routes and tags

| code | meaning | defined |
|---|---|---|
| R0 / R1 / R2 | R0 scripted teacher; R1 oracle packet E(expert chunk) → system 0 (DIAGNOSTIC); R2 generated packet → system 0 (deployable) | ladder.md, `rrp.evaluation.ladder` |
| `oracle_zero_*`, `generated_zero_*`, `teacher_shadow_*` | `<route>_<prev-action input>_<tag>`; `zero` = deployment input (prev action zeroed), not a zero packet | ladder.md |
| `orcbc` | stateless R1 with the BC expert (`--oracle-expert bc`) | D-052, D-053 |
| `reanchor`, `K1`/`K8`, `_replan1` | shadow teacher re-anchored every K ticks; K1 and replan1 invalid | D-049 |
| `_s3000000/_s3000100/_s3000200`, `_fresh3000100` | dev vs fresh seed sets (two spellings of the same set) | D-064, D-078 |
| `direct1701_u12000/u18000`, `codec1701_u*`, `bc18k` | BC positive controls, seed 1701, snapshot at update 12k/18k | D-050, D-058, D-083 |
| `latent_slice1`, `_b1fix` | sealed four-way protocol / its B-1-fixed baseline runs | `configs/eval/latent_slice1.json`, D-064 |
| `sprint_*`, `acceptance_sprint_sem_*` | demo-sprint agents and their outputs | D-050..D-080 |
| `gen_jf_{parm6,panda,parm6_ext}` | task-context edit suite on the deployable jointfix route; `_ext` = 39 new seeds | D-074, D-075, D-077 |
| `acceptance_arm{nosem,sfjf}_gen_*`, `semcompl` | the same suite per lineage; semcompl = sem panda completed to 48 seeds | D-091 |

## legged / humanoid (all contact model v1 through D-092)

| code | variant | seed | canonical | defined | example |
|---|---|---|---|---|---|
| `legged_{rep,flow}_{sem,nosem}_<body>_v2` | sem / nosem | s0 | `legged/<body>/{sem,nosem}/s0` rep, flow (qd dropout 0.5, dz 32) | D-067, D-070 | `configs/legged_latent/flow_sem_go2_v2.json` |
| `_v2s1/_v2s2/_v2s3`, `t1s1` | sem / nosem | s1–s3 | t1 training seeds (two spellings) | D-082, D-084 | `rep_sem_t1_v2s2.json` |
| `legged_vlm_{rep,flow}_*_v1` | sem / nosem | s0 | 6-body Stage A, stopped | D-040 | `configs/legged_latent/rep_sem_v1.json` |
| `legged_bc_<body>_v1` | BC | s0 | positive control | D-067, D-073 | `configs/legged_bc/bc_t1_v1.json` |
| `dag1/dag2`, `t1s1_dag1` | rz | — | system-0 refit on stateless-BC DAgger | D-079, D-082 | `configs/legged_dagger/rz_sem_t1s1_dag1.json` |
| `lv4`, `sem_lv4`, `t1_diag`, `t1diag_*` | semfix | s0, s1, s3 | T1 DIAGNOSIS: `lv4` = `probe_lv_min −4`; `flow_sem_w0` = psw 0; `rz_*_ctl` = realization-only refit; `rz_*_genz` = 50% generated packets | D-085, D-087 | `configs/t1_diag/rep_sem_lv4_s3.json` |
| `legged_fixsem`, `fixsem` | semfix | s0 | go2/hexapod6 rerun with lv4 | D-088 | `configs/legged_fixsem/rep_sem_go2_lv4.json` |
| `legged_fixrep`, `fixrep` | semfix + nosem | s1, s2 | replication | D-090 | `configs/legged_fixsem/flow_fixsem_hexapod6_s2.json` |
| `t1_edits`, `t1edits_{fixsem,nosem}_s{0,1,3}` | semfix / nosem | s0, s1, s3 | context-halt suite on the D-087 models (no training) | D-092 | — |
| `legged_ladder`, `r1`, `r1qd0`, `r1t`, `r2`, `r2ctx` | eval | — | R1 stateless-BC oracle; qd zeroed; privileged teacher-encoded packets; R2; task-context edits on R2 | D-069..D-071 | `scripts/legged_ladder.sh` |
| `snap_s4000`, `snap_s8000`, `policy`, `_orig` | eval | — | flow snapshot at 4k/8k vs final flow; original (pre-DAgger) system 0 | D-070, D-084 | — |
| `bigrand`, `trainseed` | eval / video | — | extra random z edits at norm 16/25; video term for the training seed | D-088, D-092 | — |
| contact `v1` / `v2` | physics | — | contact model version (v2: elliptic cone, higher impratio, noslip, compliant sole) | D-093 | `research/tracks/contact.md` |

## inconsistencies to fix in W4/W5 (not renamed now)
1. The bounded-NLL fix has six names: `semfix`, `sfjf`/`sf`/`asf_`, `fixsem`, `lv4`, "fixed sem", "bounded-NLL sem".
2. The frozen arm sem lineage is `jointfix`/`jf` in seed 1 and `sejf2`/`se2` in seed 2; its Stage A name has no `jf`.
3. `gdag` (flow generator-DAgger rounds and context buffers) vs `gendag` (system-0 refits on generated-packet buffers) vs
   `gen1..3` (those buffers); `h` means pack-free, not half. `dagger1/2` (shadow teacher) vs `bcdag1/2` (BC expert).
4. `gendag1_noqd` (configs) vs `gendag1noqd` (tags); `_fresh3000100` vs `_s3000100` for the same seed set.
5. Seed-2 arm Stage A configs carry an extra `ladder_` (`rep-ladder_latent_semfix_b1fix_anchor_s2.json`).
6. Legged seeds are `t1_v2s1` in rep/flow but `t1s1` in DAgger/eval, and `_s1` in fixrep.
7. "v1/v2" means different things for datasets, flows, binding, legged and contact physics; `flow_latent_sem_v2/v3` reuse the v1 encoder.
8. The arm nosem driver keeps its own script (`armnosem_chain.sh`) and label prefix `ans_`.
9. `rz_bindv4sem_noqd.json` packs from `latent_pp_v3dart_s1_H16`, unlike the other bindv4 configs (`binding_combined_v1_H16`); intent unknown.
10. Legged sem uses `packet_semantic_weight` 0.5, arm sem 1.0.
