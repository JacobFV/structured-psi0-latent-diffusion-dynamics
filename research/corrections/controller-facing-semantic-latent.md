# correction: the controller-facing action latent must carry the semantics

status: AUTHORITATIVE (user correction received 2026-09-21). Supersedes the earlier interpretation wherever they conflict.
pre-correction revision: `272c689` (tag `pre-correction-272c689`); migration branch `correction/controller-facing-latent`
(worktree `~/work/rrp-migration`).

## central requirement
System i generates a structured continuous latent packet `z[b, k, m, d_z]` (knots x controllable assemblies x width);
the conceptual objectives (entities/attention, manipulator subtasks, interaction state, relative geometry, compositional
task semantics, contact frames, uncertainty, object-language) are supervised ON THAT TENSOR; system 0 receives the SAME
tensor (only declared serialization/dtype transforms) and realizes it online with morphology + current proprioception +
declared local sensors + phase. Acceptance diagnostic = `probe(received_latent_packet, query)`.

## audit of the old path (at 272c689, traced end to end)
- training target: `learning/behavior.py` trains `FlowPolicy` on per-actuator normalized native actions
  (`latent_dim=1`) or on the ACTION-ONLY codec latent (`model/codec.py`: encoder sees only demonstrated actions + node
  features — cannot distinguish same trajectory / different task meaning).
- semantics: `model/flow.py::SemanticReadout` reads action-expert HIDDEN states (+ optional projected `z_hat`) with
  queries built from CONTEXTUAL scene-slot features. Semantics were shown inside system i, not in any transmitted packet.
- sampler/output: `policy/runner.py` samples `z`, takes `z[..., 0]` or codec-decodes, denormalizes to joint targets,
  emits a native `ActionChunk`; `sim/native.py` ChunkExecutor queues native rows; `control/joint_targets.py` tracks them.
  Same pattern in `learning/rollout.py` (GRPO/EXPO), `evaluation/runner.py`, `service/sessions.py` (learned mode),
  `sim/dual.py`, `sim/legged.py`. => system 0 received native references only; nothing semantic crossed the boundary.

## what stays (reused, not rewritten)
simulation + sensors/tracker, public/privileged separation, teachers (arm/dual/legged), datasets (pick_place v3/v3dart,
support_insert v2, legged), morphology import/surgery/validation, task runtime/receipts/edits, featurizer (feat-v2 banks
remain the SYSTEM-I context), controllers/trackers as the native layer under system 0, broker/watchdog, workbench/UI,
evaluation/statistics/registry, flow-SDE GRPO math (re-targeted to the latent policy).

## baselines kept under explicit names (never presented as the corrected architecture)
- `baseline_direct_action` (+ hidden-state auxiliaries): old FlowPolicy latent_dim=1, runner -> native ActionChunk.
- `baseline_action_only_codec`: old `model/codec.py`.
- new `latent_nosem`: same new latent/system-0 path, no semantic supervision (capacity-matched).
- new `latent_sem`: the corrected architecture.

## affected runs
Every learned-policy run before this correction tested the old path only: dev_structured_partial, dev3/dev4/dev5/dev6
(pick_place; dev4/5 closed-loop results in decisions D-021/D-022), si_dev_*, codec_dev_*, adapt_* (GRPO/EXPO on
direct-action policies), VLM policy configs (not run). Their results remain valid FOR THE OLD PATH ONLY and are never
reported as evidence for the latent packet. dev6 x3, codec_v3, si_dev x2 were lost in the 2026-09-21 peer reboot
(pre-correction; no rerun scheduled). No old-path run will be scheduled except as a named baseline in the four-way
comparison.

## new path (implemented on the migration branch)
- `contracts/latent_action.py`: versioned `LatentActionChunk` (z values, knot times, opaque assembly handles/masks,
  obs/graph/runtime revisions, robot spec hash, latent_space + realizer compatibility versions, generation time,
  validity interval, source, sampling provenance, opaque entity registry). Strict schema: no semantic metadata fields.
- `model/semantic_latent.py`: contextual target encoder E(observed history/context banks, supplied task context,
  morphology, demonstrated behavior) -> z_target; privileged labels supervise via packet probes only (audited: no
  privileged INPUTS to E).
- `model/latent_probes.py`: packet-only probes P(z, query_type, opaque handle); controls: query/metadata-only, shuffled z.
- `control/latent_realizer.py`: online system 0: z + morphology node features + current proprio + declared local
  sensors (touch, grip width) + elapsed phase (+ versioned recurrent state) -> native joint references every control
  tick; existing joint tracker underneath. Trained with DART off-nominal states.
- system-i flow policy generates normalized z over (knots x assemblies); semantic losses on z_hat_clean with frozen
  probes; evaluation on noise-started terminal samples.
- clocks (separate, declared): system-i replan 0.4 s; latent knots 4 x 0.2 s; system-0 feedback 20 Hz; servo/physics
  500 Hz. NFE is not a clock.

## revised execution order
1. contracts + failing boundary tests (independent consumer, no bypass, metadata controls, output gradient, validity).
2. representation stage: train E + system-0 realizer + packet probes (latent_sem and latent_nosem) on pick_place
   v3dart; freeze latent-space v1 + realizer interface.
3. system-i flow over z (sem / nosem) + runner/sim bridge emitting LatentActionChunk; closed-loop eval; packet probes on
   free samples (oracle-target vs one-step clean vs free-sampled reported separately).
4. disturbance test (packet held fixed), causal interventions, counterexample test, embodiment-swap test.
5. GUI packet panel/freeze-latent/disturbance; replay of packets.
6. predeclared first-slice protocol; four-way comparison; supervised adaptation; then GRPO on frozen latent meaning.
7. dual-arm, VLM (system ii) attaches upstream of system i, legged/humanoid on the roadmap.
