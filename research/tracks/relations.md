# track: relations (open) — relation-factor experiments (D-144)

State: **budget_exhausted** — WOUND DOWN (owner, 2026-10-05; D-147 addendum "campaign wound down"): campaign stopped for the Ψ₀.1 pivot, state at stop recorded there. Before the stop: T9 RE-PLANNED a second time, on the v6 semfix lineage (D-147 addendum 2026-10-03 night, lead): the armdiv
G3 lineage gate FAILED (v8div semfix 362/480 = 0.754 < 0.835), so T9 on v8div is MOOT per its pre-registration (nothing of it ran
beyond the CPU plumbing smoke). Pre-registration: "T9 v6 pre-registration" below; recipes `recipes/relations/relations_v6.yaml`
(template `recipes/templates/relations_v6_factor.yaml`). The first T9 comparison (v3dart, F0-only route) was halted earlier: floor,
uninformative ("T9 halted v3dart comparison"). Progress log: `~/work/rrp-data/campaign/logs/relations_progress.log`.

## T9 halted v3dart comparison (2026-10-01..03; recorded 2026-10-03, failed_hypothesis of the DESIGN, not of the factors)

What ran (`recipes/relations/relations_{geo,ix,task}.yaml` = `recipes/templates/relations_factor.yaml`, worktree
`camp-relations`, code 15bf78da/3c84df9d): one shared stage A (v3dart pack `latent_pp_v3dart_s1_H16`, 13 bodies, 15,000 steps,
probes: focused_on 1.0, held_by .997, visible .973), relgen shards for geo / ix / task (200 teacher episodes on parm6_pg2; task
needed one code fix 16cd9bfe), then flow F0 (20,000 steps) for `base` (preset:arm) and `geo` x seeds 1, 2, evaluated as
**F0 + the un-refit stage-A system 0** on the R2 dev bodies (parm6_tf3, panda_pg2; seeds 3,000,000/100, 30 each) and the two
held-out bodies (parm5s_tf3, parm5l_pg2; 30 each). ix F0 was never admitted (below); task F0 never started.

Result (`rrp suite relations-compare`, tables archived with the runs):

| set | n (paired) | success | failed at approach / grasp / lift / transport | mean closest tcp-cube | final flow loss s1 / s2 |
|---|---|---|---|---|---|
| base | 360 | 0 | 272 / 70 / 17 / 1 | 0.055 m | 0.258 / 0.255 |
| geo | 360 | 0 | 287 / 53 / 12 / 8 | 0.061 m | 0.271 / 0.257 |

Paired geo - base: 0 discordant pairs (McNemar p = 1); furthest stage -0.017 [-0.083, 0.050]; closest tcp-cube +0.007 m
[0.004, 0.009] (geo slightly farther). geo competence (scheduler EMA, depth 1): geo.depth3d 0.10, geo.normal_align 0.54.
**Reading: a floor; it says nothing about factors.** Both arms fail the same way at the same stage, so the contrast has no room.

Cause analysis (evidence, not a new experiment):
1. Stage / route (main cause). The evaluated route was F0 with the stage-A system 0: no DAgger, no system-0 refits. The deployable
   arm route of every competent lineage is the DAgger-refit chain (v6: F0 -> ... -> Fgdag2h + rzgendag3). In the archived v6 semfix
   lineage the SAME F0 stage evaluated with the DAgger-refit system 0 (`prog20k` = F0 + rzgendag1, dev seed 3,000,000) reached
   99/120 (s1 26/30 panda, 26/30 parm6; s2 17/30, 30/30) with 0-5 approach failures per 30
   (`.old/research/tracks/ladder/armv6/summaries/arm6-semfix/eval_r2-prog20k_*`). T9's F0 with the un-refit system 0 failed at
   approach in 78% of episodes (559/720): the policy hovers near the cube (closest 5.5-6 cm; final teacher phase `pregrasp`)
   and never descends: the covariate-shift failure the DAgger refits exist to remove.
2. Data. v3dart (13 bodies, the pre-v6 expert data) instead of the current v7div pack / v8div labeller; T9's evals also ran
   without `grasp_contact: v2.1` (the v6 / v8div physics). Grasp-stage failures are 16-19% of the rest, so this is secondary, but it
   makes the comparison off-protocol for the arm track.
3. Approach failures dominate in BOTH arms and the paired outcomes are identical: no factor effect can be read at a floor.
4. Resources: F0 declared 24G + 16G GPU, then 40G + 16G after one throttled resume, against measured RSS peaks of 1.96 - 2.51 GiB
   (base s1, geo s1, base s2; the 19.2 / 25.6 GiB of geo s2 were the PSI-shed segment and its resume, page cache included);
   two ix F0 leases at 56 GiB each exceeded the peer's aggregate admission limit (131 > 114 GB) for 6 h and were never admitted.
Disposition: halted by the lead; coordinators `rrp-camp-relations-main*` stopped (units failed/inactive). Artifacts archived
2026-10-03: peer `runs/relations` (190 files, 1.2 GB; checkpoints, evals, relgen shards, schedule logs) ->
`~/work/rrp-data/peer-archive/runs/relations-v3dart-halted/` (sha256 tree equal, `SHA256SUMS`; host-side dag ledgers and tables in
`host-side/`), removed from peer /dev/shm; T9's 24 GB peer copy of the v3dart pack removed (host original sha256-equal).

The relation-factor registry (D-144, `docs/relations.md`, catalog `research/relations_catalog.md`) is built: token model,
`FactorSite` in the shared attention block, `ReadoutProbe`, `relgen` labels / parts / transforms, curriculum scheduler,
deploy guard, room panels. What is missing is the EXPERIMENT: no factor set has been trained against the `preset:arm`
control. That is this track.

Per-unit history of the build (R1-R21, rel-geo, sweep-flags, closures) is in `research/decisions.md` D-144 and its
addenda; the unit notes `research/tracks/rel-*.md` and `sweep-flags.md` are closed-track material (moved to `.old/research/tracks/` by P5).
Their open items are collected below, so nothing needs the old notes.

## T9 v8div pre-registration (written 2026-10-03 BEFORE any T9-v8div run; committed before the smoke and before G3) -- MOOT

**Outcome (2026-10-03 20:35):** the armdiv G3 lineage gate FAILED (v8div semfix 362/480 = 0.754 vs v6 semfix 425/480 = 0.885,
threshold 0.835; per seed 180 / 182 of 240; recorded failed_hypothesis by the armdiv owner, research/tracks/armdiv.md). Per item 7
below, T9 on v8div is moot: no v8div T9 lineage ran (only the CPU plumbing smoke, deleted). The coordinator
`rrp-camp-relations-b-main` was disarmed before acting (its verdict pattern did not match the `stage2c` tag, so it was still waiting).
The plumbing fixes this pre-registration produced (amendment A1, compare-tool validity) carry over to the v6 re-plan.

1. **Question.** On the competent arm route (v8div semfix lineage), does adding one relation-factor set to the flow, with its
   probe supervision scheduled by the responsive curriculum (docs/relations.md 5.5), change deployable success at equal data
   and updates? Sets: `geo`, `ix`, `task` (arm-buildable members, as before); `all` (their union) conditionally (item 8).
2. **Control `none`** = the armdiv lineage `arm8div-semfix` seeds 1, 2 (`recipes/armdiv/arm_lineage_v8div.yaml`, FROZEN, T6;
   default factor list = `preset:arm`, verified equal to the explicit `['preset:arm']`). It is the same recipe and seeds, so it is
   read, not re-run (D-140): `artifacts/runs/armdiv/arm8div-semfix/`.
3. **Treatments** `relations8-<set>` seeds 1, 2 (instance `recipes/relations/relations_v8div.yaml` -> template
   `recipes/templates/relations_v8div_factor.yaml`, which `extends` the v8div recipe unchanged).
   - Varied: `params.policy.factors` = preset:arm + the set on F0, Fft, Fgdag1, Fgdag2h (the architecture must match for the
     strict `init_from`); `inputs.relgen` + `params.curriculum` on F0 (ramp 20,000) and Fft (ramp 10,000) only (the pack-trained
     flows; Fgdag2h has no pack rows, so a curriculum would be inert there; Fgdag1 carries the architecture only). Scheduled
     factors: geo.depth3d, geo.normal_align (share_max 0.25); ix.contact, ix.held_by, ix.support (0.2); task.next_contact (0.25);
     interval 1000. Relgen: 200 scripted-teacher episodes on parm6_pg2 (a pool body) under grasp_contact v2.1, seed_start
     5,000,000 + 100,000 (seed - 1), a snapshot every 8 ticks, at most 6 per episode. Relgen labels are privileged teacher-snapshot
     labels (StateView), training targets only; the deployed flow reads the same observation tokens as the control (deploy guard).
   - Shared with the control (consumed by run id, the representations pinned by sha256; the DAgger buffers cannot be pinned and
     their `pipeline_manifest.json` digests are recorded at launch): stage A, rzbcdag1long, rzbcdag2 representations and the
     bc1 / bc2 / bc3 buffers. These nodes do not depend on the flow; reusing them makes the prefix identical, not merely matched.
   - Matched: v7div pack (pinned), DAgger labeller bcv7div 1701 (pinned), every step count / batch / lr / seed / DAgger seed /
     eval set / grasp_contact. `tests/unit/test_relations_v8div_recipe.py` asserts that each planned node equals the control node
     except the factor list, curriculum, relgen input, pins and names. Data accounting: relgen rows REPLACE up to share_max of a
     batch's main rows (equal optimizer updates and batch size; up to 20-25% fewer main rows on F0 / Fft), so the factor's data
     cost is inside the contrast.
   - Code: the control trains on the armdiv peer code (c41decde + resource-only changes); T9 trains on main at launch. Their
     train / eval-path difference is the default-off compute block and warp eval backend (D-147 unit enable: nothing switched).
     Every T9 node records its src_tree; a train-path change beyond those is reported as a caveat.
4. **Evaluation (dev seeds only; no sealed target is touched).** PRIMARY = the G3 source-body cells: `finalevals` parm6_tf3 and
   panda_pg2 x seed starts 3,000,000 / 3,000,100 / 3,000,200 x 30 + `heldout` parm5s_tf3 and parm5l_pg2 x 3,000,000 x 30 =
   240 per seed, **480 per set pooled over seeds 1, 2**. SECONDARY = `newarms` (pa2s0_pg2, pa2s3_tf3, ur10e_pg2, vx300s_tf3 x 30 =
   240 pooled). Success = the privileged evaluator's success (source: learned flow + learned system 0; labels in every table).
5. **Metrics.** Success k/n with Wilson 95% per set, body and pooled; **contrast set - control with the Newcombe 95% interval**
   (pooled primary = the test; per body and newarms descriptive) plus the Bonferroni (1 - 0.05/3) Newcombe interval for the
   headline; paired McNemar (training seed x body x env seed) and graded paired deltas (furthest stage, closest tcp-cube) as
   secondary; per-factor probe competence by composition depth from `schedule.jsonl` (F0 and Fft); interference table = per set
   x body success / stage deltas against the control, plus the scheduler's within-set factor-pair interference where recorded;
   final training record (flow loss, last 2000 steps) next to the control's. Command (one place, tested):
   `rrp suite relations-compare --v8div --root artifacts/runs/relations --control artifacts/runs/armdiv/arm8div-semfix --sets geo,ix,task --seeds 1,2`.
6. **Success criterion (per set, fixed now).**
   - Validity: every scheduled factor's competence at the last Fft scheduler decision >= 0.5 in both seeds. Otherwise the set is
     reported "factor not learned": its contrast is still reported but is not evidence about the factor's use.
   - **HELPS** iff the Newcombe 95% lower bound of (pooled primary k/480 - control k/480) > 0; **HURTS** (interference) iff the
     upper bound < 0; otherwise **NO DETECTABLE EFFECT** (the interval is the result; near the G3 level ~0.88 the design resolves
     roughly +-0.045). A headline "factor set X improves the arm route" additionally needs the Bonferroni interval to exclude 0.
   - Ceiling guard: if the control's pooled primary rate is > 0.95, the primary has no room and the newarms contrast is the
     reading (same rules), decided now. The control floor is excluded by the G3 gate itself (>= 401/480).
   - Probes / attention are diagnostics. The causal evidence is the matched contrast (the factor set is the only difference);
     a deploy-time factor knockout needs code that does not exist and is NOT part of this pre-registration.
7. **Gate, order, compute.** Starts only after the armdiv G3 lineage gate PASSES (`G3 PASS` in
   `~/work/rrp-data/campaign/logs/armdiv_progress.log`); if G3 fails, T9 on v8div is moot (recorded, nothing runs). Peer only,
   ONE GPU lease at a time (`--max-parallel-gpu 1`; the broker's shared non-humanoid slot), CPU nodes in parallel within broker
   caps; order seed 1 geo, ix, task, then seed 2. A node cut at a lease segment end (`max_seconds_exceeded`, resumable from
   `policy_last.pt`), an admission wait, or an infrastructure kill (watchdog shed, slice-level OOM) is resumed with
   `--retry-failed` and unchanged settings (not an attempt); any other failure stops that lineage for review. Budget: six
   lineage suffixes, estimated ~8 GPU-h each (~50 GPU-h); cap 80 GPU lease-hours including resumes; at the cap the remaining
   cells are `budget_exhausted` and reported as such.
8. **Combination `all`** (`relations_v8div_all.yaml`, share_max 0.1 per scheduled factor): runs only if the six single-set
   lineages complete within 60 GPU lease-hours (then cap +20 h); otherwise it is not run and recorded so.
9. **Memory declarations (D-117, >= 1.35 x measured peak; resources are not in any config hash).** Measured: T9 v3dart F0 RSS
   1.96 - 2.51 GiB; v8div flow F0 7.2 GiB at the 9G declaration while throttled by memory.high (true peak unknown: page cache of
   the 21 GB pack read through mmap); v8div refits 6.4 - 6.9 GiB at 8G (throttled); v8div bc collections 6.3 - 7.0 GiB at 22G
   (unthrottled); T9 relgen 1.79 GiB; T9 dev evals 4.8 - 5.2 GiB; GPU: v6 flows 1.12 GiB. Declared: flows 14G + 2G GPU, refits
   10G + 3G GPU, collections 10G (was 22G), relgen 3G, finalevals / heldout 8G, newarms 12G (armdiv's). The largest concurrent
   T9 footprint is one GPU node (<= 16 GiB incl. GPU) plus CPU collections (10 GiB each), down from 56 GiB per F0. The CPU smoke
   measures the flow / refit peaks on the v7div pack; before launch the flow declaration is set to max(14G, 1.35 x the largest
   unthrottled v8div / smoke flow peak), recorded here (a resource change, not a protocol change).
10. **Smoke (plumbing only, before G3).** `relations_v8div_smoke.yaml`: seed 1, all three sets, tiny steps, 4 bodies, CPU only
   (no GPU lease), consuming the control's completed seed-1 outputs. Its outputs are not results and are deleted after the
   check; a plumbing fix it forces is recorded here as an amendment before the real run.
11. Pins: seed 1 filled 2026-10-03 (control nodes completed: train_rep_s1 108e09ab..., refit-bcdag1_long_s1 87beb0c2...,
   refit-bcdag2_s1 78dbfb23...); seed 2 filled when those control nodes complete (a PENDING pin refuses the stage).

12. **Amendment A1 (2026-10-03, from the smoke, before any real run; plumbing, not protocol).** The CPU smoke's ix F0 (20-step
   curriculum interval) failed: `RelationBatches.loss` raised because one drawn shard batch carried no valid `ix.support` label
   (sparse pair labels). `estimates_loss` documents an absent label as "no term, never an error"; the guard meant to catch a
   factor the net never writes an estimate for. Fix (`harness/data/mix.py`): a scheduled factor without a term is tolerated per
   batch (logged `relgen_nolabel`; a batch with no term at all adds a zero factor loss); the error is raised only for a factor
   that has NEVER produced a term after `SILENT_STEPS_MAX` = 20 active shard steps (test updated accordingly). Smoke otherwise
   48/48 nodes (seven watchdog `live_limit_reduced` stops resumed with `--retry-failed`). Measured smoke peaks (CPU, v7div pack):
   F0 4.5-5.2 GiB, other flows 1.5-2.9, refits 0.9-2.2, collections 3.3-3.5, evals 1.0-1.1, relgen 0.4-0.6 GiB; all within the
   declarations above (flows 14G). Re-run of the smoke F0s with a 20-step interval: schedule records carry competence and the
   shares respond (ix at step 40: contact 0.92, held_by 0.95, support 0.99 -> shares 0.20 / 0.16 / 0.07; geo depth3d 0.06,
   normal_align 0.34; task next_contact 1.00). CAVEAT on the validity gate (criterion unchanged): for the pair / class factors
   (ix.*, task.next_contact) competence is a hit rate on imbalanced labels and is near 1 after 40 steps, so the >= 0.5 gate is
   weak for them; the raw per-factor losses (`probe_<f>` in the training logs) are reported next to it. The analysis tool now
   also counts a scheduled factor that was never observed as not learned (a tested fix found on the smoke layout).

## T9 v6 pre-registration (written 2026-10-03 night BEFORE any T9-v6 run; lead: "re-plan T9 on the competent pipeline")

Same design, criterion and validity gate as the v8div pre-registration above, on the v6 semfix lineage; differences stated here.
1. **Pipeline.** The archived v6 lineage recipe (`.old/dags/arm_lineage_v6.yaml` -> `arm_lineage_v2` -> `recipes/templates/arm_lineage.yaml`):
   v6dart pack `packed/latent_pp_v6dart_s1_H16` (meta.json sha256 dee32ff1...1204, pinned), DAgger labeller bcv6_direct1701 final
   (a29810bd...767b3, pinned), grasp_contact v2.1 in every simulated stage. v6 semfix reached 425/480 on the G3 cells (D-134).
   Template `recipes/templates/relations_v6_factor.yaml`, instance `recipes/relations/relations_v6.yaml` (118 nodes).
2. **Shared prefix.** The flow-independent nodes are the RECORDED v6 lineage `runs/armv6/arm6-semfix` (peer store, offloaded to
   peer disk, intact): stage A, rzbcdag1long, rzbcdag2 representations pinned by sha256 (s1 6cc278b3 / 0239535c / 0020bdbc,
   s2 b7d28fe7 / cca73c18 / 461cb103: equal to the recorded v6 checkpoints); bc1-3 buffer manifests s1 d3996c04 / 261a9d5f / 175d5250,
   s2 b3da97b1 / 101120b0 / b6565097.
3. **Control (CHANGE from the v8div design, and why).** PRIMARY control = `relations6-none` seeds 1, 2: the v6 flow-dependent suffix
   re-run on CURRENT code with the default factor list, same prefix. The recorded v6 suffix trained on code 520916e (2026-09-28);
   T9's factor runs train on current main, and the refactor since then touched the trainers (D-145 / D-146; the v8div lineage on
   current code reached 0.754 where v6 had 0.885, a gap whose code / data split is unknown). Reading the recorded v6 as the control
   would confound factor and code. The same-code re-run costs ~2 x 2 GPU-h (v6 measured 1.7-2.2 GPU-h per suffix). The recorded v6
   lineage is the SECONDARY reference: `none` vs recorded v6 is reported as a reproduction check (Newcombe 95%), never a verdict.
   `tests/unit/test_relations_v6_recipe.py` proves `none` renders the recorded v6 suffix configs (`tests/data/armv6_semfix_suffix_runconfigs.json`,
   from the v6 run manifests) up to identity, pins and the legacy flags bias_mode "true" / structured true (= the default preset:arm),
   and that every factor-set node equals the `none` node except factor list, curriculum, relgen input and names.
4. **Treatments, relgen, curriculum, data accounting**: exactly as v8div items 3 (geo / ix / task factor lists; curriculum on F0
   ramp 20,000 and Fft ramp 10,000; relgen 200 teacher episodes on parm6_pg2 under grasp v2.1), with amendment A1 (sparse labels) in the code.
5. **Evaluation (dev seeds only, no sealed target).** PRIMARY = the G3 source-body cells: `finalevals` parm6_tf3, panda_pg2 x
   3,000,000 / 3,000,100 / 3,000,200 x 30 + `heldout` parm5s_tf3, parm5l_pg2 x 3,000,000 x 30 = 480 per set pooled over seeds 1, 2.
   (v6 has no newarms node; there is no secondary body group.)
6. **Criterion (fixed now, unchanged).** Validity: every scheduled factor's competence at the last Fft decision >= 0.5 in both seeds
   (a scheduled factor never observed = not learned; caveat for imbalanced pair / class labels as in amendment A1, raw `probe_<f>`
   losses reported). HELPS iff the Newcombe 95% lower bound of (set k/480 - none k/480) > 0; HURTS iff the upper bound < 0; else
   NO DETECTABLE EFFECT; headline additionally needs the 1 - 0.05/3 interval to exclude 0. Ceiling guard: if `none` pooled > 0.95,
   the success contrast has no room; the paired graded deltas (furthest stage, closest tcp-cube) are then reported, descriptively
   only. If `none` is BELOW 0.5 (current code does not reproduce a competent route), the contrasts are still reported but labelled
   "control not competent"; that outcome is itself a finding about the code drift, not about factors.
   Command: `rrp suite relations-compare --v6 --root artifacts/runs/relations --reference artifacts/runs/armv6/arm6-semfix --sets geo,ix,task --seeds 1,2`.
7. **Order, compute, budget.** Peer only, ONE GPU lease at a time (`--max-parallel-gpu 1`), CPU nodes in parallel within broker
   caps; the DAG schedules all eight suffixes (none, geo, ix, task x s1, s2) as admission allows. Resumable stops (lease segment end,
   admission wait, watchdog shed / live_limit_reduced / disk reserve) are relaunched with `--retry-failed` and unchanged settings;
   anything else stops for review. Budget 80 GPU lease-hours (estimate ~20). Combination `all`: only if the eight suffixes complete
   within 60 GPU lease-hours; its recipe is then committed as a further amendment BEFORE it runs.
8. **Memory (>= 1.35 x measured peak).** Flows 12G + 2G GPU (v6 flow 6.6G incl. page cache; T9 smoke F0 on the larger v7div pack
   5.2G), refits 8G + 3G (v6 1.7G / 0.9G), collections 22G (v6 measured up to 15.8G), evals 8G, relgen 3G.
9. **Smoke.** The v8div smoke already exercised every stage of this pipeline with factors (same code path, other pack); no new smoke.
   The first live F0 doubles as the check: its `config.json` must equal the recorded v6 F0 `config.json` except name, factors,
   curriculum, relgen (`none`: except name) -- checked and recorded before the remaining suffixes are read.

## recipes (v3dart instances: HALTED, superseded by `relations_v8div*.yaml`; kept for the record)

`recipes/templates/relations_factor.yaml`: one shared representation (stageA, semantic probes, trained once so every
factor set sees the same input information) + probes read once, then per factor set (`fset` axis) x seed: flow `F0` with
`params.policy.factors = [preset:arm, preset:<set>]`, the deployable route on the R2 dev bodies (`dev`) and the two
held-out bodies (`heldout`). `base` (preset:arm only) is the paired control at equal steps and seeds.

| instance | factor set vs `base` | resolves to (beyond the 17 `edge.*` + `msg.incidence`) |
|---|---|---|
| `recipes/relations/relations_geo.yaml` | `preset:geo` | `geo.pos3d geo.depth3d geo.orient geo.normal_align geo.above` |
| `recipes/relations/relations_ix.yaml` | `preset:ix` | `ix.contact ix.held_by ix.handover ix.support ix.force_flow` |
| `recipes/relations/relations_task.yaml` | `preset:task` | `task.next_contact` (`time.same_track` is `planned`, D-146 round 2) |

`preset:ui` (ComputerWorld widget tree) is a pointer-track factor set: it is a `factors:` var of the pointer recipes
(`recipes/pointer/`), not an arm instance. Legged / humanoid sets (`legged-r19`: `leg.foothold leg.com_support`) run through
`recipes/templates/legged_lineage.yaml` once a body set is chosen; no instance yet.

Relgen reaches the trainer (R2 RG, below): per set the F0 node takes the `relgen` shards through `inputs.relgen` and a
`params.curriculum`; the `base` control and the shared stageA take none. Factor lists are the ones the arm family can build:
`resolve(family="arm")` rejects `geo.normal_align` as `given` (no `normal` ctx field; it is `source: probe` here), `ix.handover`
(no `handover_pairs` label), so those two are not in the F0 lists (the full presets failed at model build, which no run had
reached before; `time.same_track` is already out of `preset:task`, D-146 round 2). State: planned for the real recipes, smoke on the peer (R2 RG note).

## resume (v3dart instances, HALTED: do not run; T9 resume is in the pre-registration above)

Host = git / editing / unit suite only. Everything below runs on the peer with `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/relations`
after `ops/bin/peer_sync.sh push`.
```
rrp run-dag recipes/relations/relations_geo.yaml --dry-run          # 14 nodes: stageA, probes_rep, then F0/dev/heldout x (base, geo) x seeds 1, 2
rrp run-dag recipes/relations/relations_geo.yaml --max-parallel-gpu 1
rrp run-dag recipes/relations/relations_ix.yaml --max-parallel-gpu 1     # the base control (lineage relations-base) is the same run set in all three instances: it is skipped once complete
rrp run-dag recipes/relations/relations_task.yaml --max-parallel-gpu 1
```
Dry-run node list of one instance (`relations_geo`): `stageA`, `probes_rep` (global, lineage `relations-shared`), then for
each `fset` in base, geo and seed in 1, 2: `F0@<fset>.s<seed>` -> `dev@...` and `heldout@...` (out
`artifacts/runs/relations/relations-<fset>/<stage>[-dev]_s<seed>`). `base` runs are identical across the three instances
(same config hash): the second and third instance find them complete and skip them.

## open items (collected from the rel-* notes and sweep-flags)

1. ~~`relations_data` is not a stage~~ (R10) closed by R1 below: a registered stage (`arm`, `legged`, `pointer`, `psi0`,
   `relations`) and the trainer hook `relation_batches`; the arm trainers call the hook (R2 RG note); HL legged and the
   observation gap for non-arm families remain.
2. **Probe-sourced `edges:support-v1`** (rel-geo, R17): `ix.force_flow` reads the support graph from the PRIVILEGED label
   only (`nets/batch.py::support_edges`, a pure function, not on `RelCtx.edges`); the deployable half (from `ix.support`'s own
   bilinear pair score) needs a read hook in `relations/ops.py` (`BilinearOp` returns q/k features, `FieldReadouts` skips
   bilinear). Until then `ix.force_flow` cannot be deployed with a probe source.
3. **Pair readouts unconsumed** (R18): `task.next_contact`'s `ReadoutDef(address="pair", reads="tokens")` is metadata; neither
   `ReadoutProbe` nor `FieldReadouts` reads a pair address. `candidate_interaction_edges` is not wired into `RelCtx`.
4. **`ix.handover` is a per-frame proxy** (R16): "both hands on the object at once"; the episode-level release-then-grip
   event belongs in `relations_data` calling the label across a trajectory.
5. **Legged / humanoid foothold not on a live env** (R19): `terrain_steps` / `foothold_cell` entities are declarative; no live
   env's `state_view()` populates them (compose wiring, R11-adjacent). `warp_tracker_ppo.py`'s `extra_obs` display string
   still implies the height scan reaches the actor (one-line wording fix).
6. **`ui.drag_to` label** (R20): "focused widget = dragged widget" is a proxy; a mid-drag predicate from the `cw/drag_window`
   judge state would sharpen it.
7. **`base_cmd` reads the whole packet** (R5): the legacy restricted key-mask (locomotion knot tokens only) is not reproduced by
   `ReadoutProbe` (one key mask per call). Add a per-query key mask only if the restriction is a real requirement.
8. **Per-sample-conditioned query address** (R4): legged `body` probes use `address="asm"` + gather (M queries instead of 1);
   a shared `asm@index_field` primitive in `nets/probes.py` would remove the trick if R5/R6-style needs recur.
9. **`packet_semantic_weight` on-disk key** (rel-r2c / sweep-flags): four readers share one conversion point; retiring the key
   itself needs a registered `flow.packet_semantic` factor (`catalog.py`, `nets/flow.py`, `nets/probes.py`). Not required by
   anything today; `joint_adapt.py`'s flat read stays (stage `adapt` is a permanent legacy-only stage).
10. **Curriculum policy** (R11): `relgen/curriculum.py` ships the scheduler, promotion by probe competence and `rrp steer`; no
    run has used it. First use belongs to the geo -> ix -> task ladder once item 1 lands.

## F1: relation runtime contract (ready-f1; audit D2 / D5; docs/relations.md section 11)

Closes open items 2 and 3 above for the arm / dual nets (pair readouts are consumed; the probe-sourced support graph is
published by `ix.support`'s own pair logits) and makes "resolves" mean "runs".
- `relations.base`: `FamilyTokens` / `FAMILIES` / `register_family`; `resolve(..., family=, env_caps=, env=, training=)`
  refuses what the net cannot run (no applicable site, unfilled `given` field, label the collate cannot attach, readout
  the net does not implement, StateView caps, no scene part / transform for `mix > 0`); `estimates_loss` supervises the
  pair estimates and field probes a forward wrote (Gaussian NLL / bce / soft_ce, `(sum, count)` metrics);
  `stamp_versions` / `require_factors` for checkpoints. `gaussian_nll` lives here (`nets.probes` re-exports it).
- `relations.catalog`: families `arm`, `dual`, `psi0`, `pointer`, and `legged` / `humanoid` as their nets run today
  (the `act>knots` routing site + `probe.legged.*`); unit HL extends those two with `legged-rel-v1` and the ctx sites.
- `relations.ops`: every bilinear factor writes `rc.estimates[("pair", name)]`; closure / hop ops apply only at square self
  sites (`kin.ancestor` / `kin.sibling` crashed at `act>ctx` before); `site_field` gt path reads the factor's label name.
- `nets.batch`: field builders are Batch-only and vectorised; `relation_token_sets(family, batch, labels, deploy, fields,
  pad_ctx)`; labels are refused when `deploy=True`. `nets.flow` builds its ctx / act sets through it (the arm net had no
  fields at all before, so every `geo.*` / `id.*` factor was unrunnable there) and adds `estimates_loss` to `loss`.
- `relgen`: `load_families()`, raising `label_def` / `part_def` / `transform_def`, `rrp factors coverage` writing
  `artifacts/runs/relations/coverage/coverage.json` (factor x family x env: resolves / label runnable / part available /
  training resolve).
- Not done (lead questions): other checkpoint writers (`pointer.py`, `bundles.py`, `latent.py`, `bc.py`) still stamp by hand
  and do not `require_factors`; `harness/pipelines/relations.py` still uses `TRANSFORMS.get` (silent skip); no `@gt`
  EdgeSet builder for `ix.force_flow` source gt; the `dual` family is checked by the shared arm test only.
- Architect review fixes: the field builders now run inside the forward, so every tensor they create is placed on the
  batch's device (the branch built them on CPU: any GPU forward, default preset included, would have raised in
  `_global_kind`; test on the `meta` device); `nets.batch` was missing its `F` import (the image-token pad path raised
  NameError) and padded pair labels along one token dim only; `relations.base` is torch-free at import again
  (`rrp factors list`, the DAG parent); the `est_weight` loss argument is gone (`FactorSpec.weight` is the one knob);
  the env-caps literals of `rrp factors coverage` are now actually tested against the env sources.

## R1: `relations_data` stage + trainer hook (ready-r1; audit D1)

Closes open item 1. Nothing about the training loop changed in F0 (no new params or inputs); a trainer opts in with one line.
- Stage `relations_data` (`harness/pipelines/relations.py`, registered for `relations arm legged pointer psi0`, source
  `privileged_teacher`): resolves `params.factors` (or `policy.factors` / `latent.factors`), rolls the driver (`options.policy`,
  default `teacher:<task>`) over `options.seeds` / `n_episodes (+ seed_start)` on `options.env / task / body` through the ordinary
  `evaluate` with a `SnapshotCollector` rollout hook. The hook labels from `env.state_view()` at reset and every
  `snapshot_every` ticks (at most `max_snapshots` per episode), at capture time, so a lazy view is never labelled from a later
  state. Shards land under `<out>/relgen/<factor>/<version>/` (not `artifacts/relgen`: `schema.toml` forbids it); the shard id is
  `seed<seed>-<hash of the rows' provenance>`, so rerunning replaces its rows instead of doubling them. The stage manifest carries
  per-shard `manifest_hash`, row count and npz digests; a stage that produces no rows raises. DAG adoption (config hash + pins)
  is the resume-by-hash. The `relations_factor` template has a `relgen` node per `geo / ix / task` instance (dry-run planned for
  the three relations recipes).
- Shard IO (`write_shard`, `load_shard_rows`, `RelgenError`) lives in `harness/data/mix.py` (the data layer must not import pipelines: `test_layering`); the stage imports it from there.
- Names are checked: a factor label not in `LABELS` (except `probe.*`, whose labels are family-level readouts) or a `gen` that
  is neither a TRANSFORM nor a scene PART raises `RelgenError` (F1 note: the silent `TRANSFORMS.get` skip is gone).
- Trainer hook (`harness/data/mix.py`): `batches = relation_batches(rc, out_dir, main_batches)`. Needs `params.curriculum`
  (`shards`: the `relations_data` `<run>/relgen` dir(s); optional `factors`; any `SchedulerConfig` key such as `interval`,
  `share_min`, `share_max`) and `batch_size`. Each batch is `{"step", "main", "relgen", "counts", "active", "labels"}`; `labels`
  is ready for `batch.extra["relation_labels"]`. The trainer feeds `batches.observe(step, {factor: {"competence": ..}})` or
  `batches.observe_estimates(step, metrics_of_estimates_loss)`. Every decision appends `<out_dir>/schedule.jsonl` with the
  steers applied and metrics observed since the last one, so `Scheduler.replay_records(cfg, seed, records, steps, n)` reproduces
  the composition exactly (tested); `start_step > 0` resumes and re-decides from the earlier records.
- `rrp steer <run>` with no op lists `<run>/steer.jsonl` as pending / applied at step N / REJECTED; `rrp steer <run> <op>`
  appends as before. An unparseable line is logged as a rejected steer, not applied.
- `rrp run-dag` now loads the pipeline modules before planning (`cli/dag.py`), otherwise a stage a module registers on import
  (this one) is unknown to the recipe planner.
- OBSERVATION GAP: closed for the arm family by the R2 RG note below (shard rows now carry the featurizer input).
- Not run: no peer smoke of the real collector path (the host does no simulation); the tests drive `SnapshotCollector` with a
  fake env and the stage with a fake collector.

## R2 RG: relgen reaches the arm trainers (readiness round 2)

Closes the R1 observation gap for the arm family and the hook wiring (open item 1).
- Forwardable rows: a shard row stores the snapshot's `PolicyInput` arrays plus its ctx token entity ids (`relgen-shard-3`, `input_kind` per row; a
  shard-1 / shard-2 manifest or a row without `policy_input` is refused, no reader for the old layout). `SnapshotCollector` takes the
  observation from `on_reset(obs)` / `on_step(step.observation)` and featurizes it once per episode with the env's
  `featurizer_for`; morph tokens map to assemblies through the inverse `feat.bindings`, scene tokens to `view.token_entity`,
  task/interact tokens carry no entity. `relations_data_stage` collects inside `apply_run_context(rc)` (the run's kinfeat).
  `label_episode` calls `transform_def(name)`, so an unknown generator name raises instead of being skipped.
- `collate_rows(rows, family)` (`harness/data/mix.py`) builds one Batch through `collate_inputs` and scatters each row's labels
  into the padded ctx layout (`extra["relation_labels"]["ctx"]`); a non-ctx label raises. `RelationBatches` (from
  `relation_batches(cfg, out_dir, specs, batch_size=, family=, main=, start_step=)`; None without `params.curriculum`) gives
  `draw(dev) -> (record, shard batch | None)` and `loss(rc, step)` = `estimates_loss` over the shard rows, then
  `observe_estimates(step, metrics)`. A scheduled factor whose net wrote no estimate (source `given`, or no readout) raises
  ("give the factor `source: probe`"): it would otherwise train at zero loss without anyone noticing. `inputs.relgen` without
  `params.curriculum`, and `prefetch`, raise.
- Trainers (`latent_train.py` stage A and flow, `behavior.py`): each step the main batch is the scheduler's `counts["main"]`
  rows of the trainer's own data (flow / stage A) and the factor loss is `(Bm*L_main + n_shard*L_factor) / B`; the shard rows go
  through the same context forward (`E.context` for stage A, `FlowPolicy.prepare(assembly_batch(b)).rc` for flow and BC) with NO
  action / flow loss, so the action loss of a mix row is zero and the factor loss is on. Logs gain `relgen`, `n_relgen` and the
  `probe_<factor>` terms; `schedule.jsonl` / `steer.jsonl` are in the run dir and `start_step` resumes.
- Recipes: `relations_factor.yaml` per set F0 has `inputs.relgen: '@relgen:relgen'` and `params.curriculum` (geo:
  depth3d, normal_align; ix: contact, held_by, support; task: next_contact; `interval` 1000 = the snapshot cadence so a resume
  loses no metrics, `ramp_steps` 20000). Stage A is not fed in the template (its E has only `preset:arm`); the trainer supports
  it through `encoder_factors`. `recipes/relations/relations_geo_smoke.yaml` is the plumbing smoke.
- Tests (`tests/unit/test_relgen_trainers.py`, `test_relgen_stage.py`; CPU, about 13 s): 3 steps each of stage A, flow and BC on
  a tiny pack plus real shards from fixture sessions: factor loss > 0 and differentiable, no labels in the action-loss batch,
  `schedule.jsonl` steps `[0, 2]`, `Scheduler.replay_records` reproduces the composition; the recipes feed F0 and the F0 lists
  build in the arm family.
- Peer smoke (`relations_geo_smoke`, `RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/r2-rg`): relgen on 6 teacher episodes x 3
  snapshots wrote 72 rows (geo.depth3d / normal_align / orient / pos3d 18 each; geo.above is not a producer) in 11 s; F0 (120
  steps, batch 16, width 64, the armdiv semfix stage-A representation, v7div pack) completed in 11 s with `schedule.jsonl` at
  steps 0..100 (shares 0.4 -> 0.16 as the full-world floor ramps) and the step-100 record `n_relgen 7`, `relgen 327.7`,
  `probe_geo.depth3d 329.2`, `probe_geo.normal_align -1.4`, flow 1.6. Plumbing only: a 120-step net is not a result. The first
  launch found that `rrp stage run` validates the RunConfig before loading the pipeline modules, so the stage `relations_data`
  is unknown to a leased job (`base.py`, not RG's): the smoke used a peer-side `sitecustomize` preloading
  `rrp.harness.pipelines.relations` (RRP_PEER_PYTHONPATH, not in the repo).
- Open (round 2): (3) `FlowPolicy.loss` discards the `estimates_loss` metrics, so the trainer does a second prepare + estimates
  pass on the shard rows only (`nets/flow.py` is RC's); (5) task / interact tokens carry no entity id.
- Round 3 (relgen-train, D-146 round-3 addendum), closing the round-2 items 1, 2, 4 and 6 in this file:
  - (1) One calibration rule, `RelationBatches.loss`: the shard factor term is `factor_loss_scale * mean_f(loss_f / ref_f)`,
    `ref_f = max(|loss_f at its first observation|, 1)` per factor (`factor_calibration.json` in the run dir, restored on
    resume), one run-config `params.factor_loss_scale`, default `mix.FACTOR_LOSS_SCALE = 0.2` = 0.1 x the flow loss at init (a
    unit-variance rectified-flow target: 2). The raw term stays in the log as `relgen_raw` (the 330 of depth3d); fixture tests:
    raw / flow > 100 at a scripted 330, calibrated / flow = 0.1 (`test_legged_relgen.py`), calibrated step-0 term <= 0.2 in every
    trainer run. The curriculum `weight` key is not the knob. The in-data `estimates_loss` of the pack rows (legged `rep_step` /
    `LeggedBC.loss`) is unchanged: only shard rows are calibrated.
  - (2) Every readout factor has a competence: `<f>_acc` the hit rate, `<f>_mae` the fraction of the initial mean absolute error
    removed (`curriculum.estimate_competence`; the first observation sets the reference, so it reads 0), both in `schedule.jsonl`.
  - (4) The legged BC trainer reads `cfg["relgen"]` / `params.curriculum` through the same `relation_batches` path as the other
    trainers (no `curriculum.shards` needed). The arm `train_bc` stage (`pipelines/arm.py`) still drops `inputs`, so arm BC takes
    shards through `params.curriculum.shards` until that stage passes `cfg["relgen"]` (not this unit's file).
  - (6) Legged / humanoid: `refuse_relgen` is gone. `write_shard` takes `inputs["legged_batch"]` (the arrays of
    `LeggedData.ctx_batch` at one tick, no batch axis, no `foothold_cell`) and `input_kind` says which input a row holds;
    `collate_rows(rows, "legged")` stacks them into the batch dict and turns the `foothold_next` label (ctx tokens `[glob | N
    joints | M limbs | M feet | C cells]`) back into the `foothold_cell` [B, M] key (-2 absent, -1 planted, >= 0 the cell) the
    legged graph builds it from. `train_rep` (E.encode), `train_flow` (F.prepare) and `legged_bc.train` (prepare) forward them
    with the action loss on the pack rows only; BC's batch keeps its B pack rows and the shard share weights the loss.
  - Not done (not this unit's files): no producer writes `legged_batch` rows yet (`pipelines/relations.py` has no legged
    featurizer; `foothold_next` is all-invalid until a backend emits terrain cells); `leg.com_support` is a readout through P on z
    and cannot be scheduled from shard rows (the error says so); `docs/relations.md` section 10 row for legged still says refused.
