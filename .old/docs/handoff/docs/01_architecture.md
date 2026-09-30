# model architecture and computational contract

This is the proposed implementation, not a claim that upstream ψ₀ already has these components.

## data flow and four clocks

Use `b` for batch, `s` for actual observation/control time, `h` for future offset inside a chunk, `l` for transformer depth, and `τ` for flow integration time. These indices are never interchangeable. A task-event index is not a timestep.

At observation time s, compile public body specification, current sensor observation/history, supplied task graph, and runtime belief into typed context. Encode once and build layer-specific KV banks. Each of K flow evaluations updates only the noisy action tensor. Decode the final latent chunk to native controller commands. Execute a bounded prefix, refresh observations, and replan. The low-level controller continues at its own declared frequency while the policy computes the next chunk.

## morphology and controller entities

`RobotSpec` contains rigid links, joints, transmissions/actuators, equality/mimic constraints, geometry, sensors, and attachment ports. Graph nodes are typed; do not conflate 43 generalized coordinates with 43 independent actuators. Controller entities can be individual joints, an arm group, or a low-level whole-body command channel. Exactly one controller owns each physical actuator at a time.

Static encoder inputs: joint type/axis, parent transform, limits, geometry features, mass/inertia where valid, controller capabilities, and sensor mount descriptors. Dynamic inputs: measured q/qdot, base estimates, declared effort/tactile observations, timestamps and masks. Separate static pooling from state encoding. A known kinematic model may compute body-relative poses from measured state; this is not privileged object-state access.

Coarse manipulator/body nodes are pooled sets with typed role geometry. Give each aggregate an explicitly defined frame (palm, tool center point, root), not an arbitrary average rotation. A whole-body center-of-mass point and a body-root frame are different queries. Distances use meters; orientations use a declared representation and geodesic error.

Node identity is episode-consistent structural addressing, not a product-name embedding table. An identical symmetric limb can be distinguished by task role and relative pose. Equivariance tests permute graph, node state, controller map, and noise together; unrelated noise draws need only distributional equivariance.

## semantic banks

- task/events: event instances, ordered role bindings, prerequisite/concurrency/resource edges, desired deltas, phase and execution receipts.
- scene/entities: visual tokens and tracked object slots, descriptors, estimates, observation masks and provenance.
- morphology/state: graph nodes/coarse assemblies and current body state/capabilities.
- interaction/frames: contacts, held objects, support constraints, reference frames, covariance, validity and failure memory.

Uncertainty belongs on representations, not in an independent expensive network by default. Each bank uses shared width with type embeddings and optional low-rank type adapters. Pack its projected KV with metadata rather than launching four serial full-width attentions per action block. Preserve an unstructured content-access path.

Context-token caps apply to compressible visual/event summaries, not permission to delete required node identities or role bindings. Preserve direct per-node morphology/state conditioning even when a bank is compressed; log actual effective context sizes and reject unsupported truncation.

Use the VLM's relevant visual/text hidden sequence through a bounded resampler; do not keep only the last language token. Morphology remains directly available even if also projected into system ii. Reused checkpoints must have explicit model revision, tokenizer, processor, hidden widths and tapped layers. Test real images and task text through the actual adapter; an all-zero visual feature stub does not pass.

## flow target and codec

A codec encoder consumes training demonstrations `a[b,h,n,u]`, body/state, and optional training-only effect labels. The decoder consumes only `z[b,h,n,d_z]`, body/state and controller metadata. It does not receive the demonstrated actions or future truth. Heterogeneous command widths use masks; latent width is shared across entities. An explicit typed vector group handles a multi-coordinate whole-body controller without pretending its coordinates are independent joints.

Train masked native-command reconstruction plus physical-effect reconstruction and compatible semantic alignment. Evaluate codec reconstruction on held-out bodies and closed-loop teacher-latent replay. Keep a bottleneck but do not choose its width purely by visualization. Check noncollapse via shuffled-latent decoding, changed-target effects, and latent reconstruction baselines. Effects need not uniquely determine controls; do not collapse different valid contact strategies into one averaged command.

Freeze the codec for the first policy and RL campaigns. A direct-action flow policy is a required comparison. New-body codec/decoder fine-tuning is an allowed separate deployment cost, not zero-shot.

Use this convention:

```text
ε ~ normal(0, identity) on valid latent coordinates
z_τ = (1 - τ) ε + τ z_target ; τ in [0,1]
v_target = z_target - ε
loss_flow = masked_mean(||vθ(z_τ, τ, context) - v_target||²)
sample: integrate dz/dτ = vθ from τ=0 to 1
```

Only valid coordinates enter losses and RL likelihoods. Invalid padded values cannot alter valid outputs. Do not noise node identity, graph structure, task events, or the current sensor state.

## fast path

For a=h*n action tokens and c context tokens, static-context attention has quadratic terms a²+ac rather than (a+c)². This is not a latency prediction: projections, MLPs, normalization, graph-bias construction, memory bandwidth, kernel selection and the VLM may dominate.

Naive node tokenization multiplies action-token count compared with one token per timestep. Use factorized temporal attention per entity plus entity attention per timestep: approximately n*h²+h*n²+h*n*c attention entries. Compare a coarse-entity hierarchy when n is large. Do not force every passive link into the diffused stream. Use masks/bucketing and separately test larger-than-training graphs.

Cache clean per-layer KV only when context is independent of τ and z, encoder dropout is disabled at inference, and model/context versions are unchanged. Frozen weights are insufficient if activations depend on changing noisy actions. Training can share differentiable projected context within one minibatch graph; do not retain stale autograd graphs across optimizer steps. For frozen perception, cached embeddings include image augmentation and processor hashes.

Cache key: robot/controller spec hash, observation/belief version, task-graph version, event/runtime version, frame/contact version, model/codec/projection revision, precision, and attention backend. A graph edit, new contact estimate, model update, or changed body invalidates the affected cache and queued commands. Bidirectional semantic refresh is a separate, measured architecture with explicit invalidation.

Benchmark naive concatenation, cached packed banks, and factorized action attention on identical shapes. Test 1/2/4/8/16 function evaluations, warm/cold context, variable bodies, batch=1 live inference, and batched rollout throughput. Count classifier-free guidance or other extra forwards if introduced. Keep synchronized p50/p95/p99 and end-to-end observation-to-command measurements. Do not use dense diagnostic attention weights in the production path unless measured affordable.

## semantic supervision and interventions

Objects: canonical slots across prompt/view/time, identity matching without target leakage, null and multi-object answers. Visibility, gaze alignment, task focus and physical interaction are different relations, not a monotone implication chain. Occluded manipulation is valid.

Manipulators: separate instance identity, capabilities/physical morphology, task role, dynamic state, and current interaction. An event may bind multiple manipulators; a hand can support one event while waiting for another. Task readouts should return a masked set of associated events and role-conditioned representations, not just an unstructured weighted average that loses conjunction and role multiplicity.

Do not confine auxiliary supervision to system ii or to clean memory. Provide query-conditioned readouts from selected system-i action hidden states (and predicted clean action-latent projections for future effects), with verified gradients into the action expert. Those readouts bind to the same canonical object/event handles. This is how the proposed concepts shape the denoising representation itself. Evaluate selected flow times, distinguish current-state labels from future-effect targets, and do not add future truth to conditioning. Probe extraction can be training-only or on-demand rather than an extra full pass per sampler step.

Object QA: system-i object readout bound to a canonical slot -> learned projector -> small prefix token set -> separate frozen text transformer -> answer NLL given question. Gradients pass through frozen decoder operations into projector/slots. Use counterfactual questions and blank/shuffled-object controls to detect language-prior shortcutting. Do not put the QA decoder in every sampler call. Contrastive text alignment is an optional ablation, not the complete semantic target.

Spatial probes: scalar distance, relative translation and orientation, camera projection/depth, held-by/contact, effect deltas. Explicit reference frames and observability masks; translation and orientation errors are not mixed in arbitrary units. Use low-rank semantic projections; hard partitioning every neuron dimension is unnecessary. Orthogonality is an ablation and does not establish disentanglement.

Contact frames: point/origin, observed normal, optional feature/gravity-defined tangent, contacted object binding, participants, age, slip, covariance, validity. A point plus a normal leaves tangent yaw undetermined; never invent a fully certain frame. Track the contacted surface's motion; old frames expire or gain uncertainty. Separate the mechanical constraint from improved estimation: a touch does not remove every degree of freedom. Active anchors may directly inform control as well as latent supervision.

Unknown is not false. Pose beliefs use distributions or calibrated residual models; relation heads use probabilities plus observation masks/provenance. Calibrate under held-out bodies and occlusion. Report reliability, Brier/NLL, coverage-risk and error-conditioned behavior. Do not multiply correlated head probabilities and call it calibrated end-to-end confidence.

## defaults and scale-up

Use `config/model_profiles.json` as initial capacity/shape targets, not claims of memory fit. Profile on the peer before allocation. Small end-to-end learning precedes medium and target-scale profiles. The large profile can be reduced only with measured memory/latency reasons and an explicit scope label. Preserve a real VLM-backed evaluation even when most structural sweeps use cached/frozen features or small models.
