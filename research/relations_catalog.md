# relations catalog (D-144): the owner's ~600 candidate relation structures, distilled

Source: owner-pasted brainstorm (2026-09-30, raw, overlapping; not committed). Design: `docs/relations.md`. Every
candidate is classified as a **property** (unary field), a **pair relation** (field × operator), a **graph** (edge
set with a graph operator), an **epistemic** structure (confidence / data transform) or a **meta** property of
relations (metadata or design choice). Each implementable item is ONE declarative `FactorDef` over the primitives
below plus, when it needs supervision, ONE label function (`LabelDef` against `StateView`) and optionally a scene part.

Primitives (docs/relations.md 3.1–3.2):
- **field kinds**: position (pos3d, cam_uvd, screen), frame / orientation, direction (normal, axis, tangent), shape
  (extent, curvature, scale), scalar / vector property (mass, friction, stiffness, hazard, affordance score), id /
  membership (entity, part, assembly, track, material, category), graph (edge sets), time / phase, belief (probability
  over hypotheses), symbol (text / instruction tokens).
- **operators**: `diff`, `sqdiff+diff` (PaPE), `rel_rot`, `align`, `order`, `same`, `sim`, `edge`, `hop`,
  `ancestor`, `flow`, `bilinear`, `unary`.
- **algebra**: arity, directed / symmetric / antisymmetric, transitive, bool / signed / weighted / prob / vector,
  static / dynamic, local / global.
- **forms**: `bias`, `aug`, `gate`, `mask` (+ `message`, `embed`, `readout`).
- **conditioning**: gates task / instruction / goal / embodiment / history; confidence scaling.
- **sources**: given (public / estimated) | probe | estimator:<name> (distilled) | gt (supervision only).

Wave tags (a scope label for the entry, NOT a progress state; progress states are the vocabulary in `STATUS.md`): **W1** first wave (docs/relations.md section 6; envs can label now), **W2** next (needs a new scene part or
label), **P** planned (entry declarable, no env / label yet), **X** out of scope for our envs (no physics / sensing for
it), **M** meta (not a factor).

## A. geometry and pose (fields: position / frame; ops: sqdiff+diff, diff, rel_rot, align, order)

| family | candidates (examples) | decomposition | wave | label (StateView caps) | envs |
|---|---|---|---|---|---|
| metric position | 2d / 3d position, world / camera / object / manipulator / task-frame position, relative translation, x/y/z displacement, range, bearing, azimuth, elevation, euclidean distance, nearest / farthest neighbor | pos3d × sqdiff+diff × aug (frame world or query; nearest-neighbor = emergent from distance term) | W1 `geo.pos3d` | `pos3d` (poses) | arm, dual, legged, cw |
| depth | depth, camera-frame position, image-plane distance, ordering along view ray | cam_uvd × sqdiff+diff × aug, source probe | W1 `geo.depth3d` | `cam_uvd` (poses, camera, depth_render) | arm, dual |
| orientation | relative rotation, relative SE(3) pose, quaternion difference, rotation-axis difference, local frames | orient × rel_rot × aug (+ pos3d frame query) | W1 `geo.orient` | `orient` (poses) | arm, dual, simple |
| surface directions | surface normal, relative normal, tangent / bitangent, normal / tangent alignment, parallel, perpendicular, coplanar, collinear | normal / axis × align × aug | W1 `geo.normal_align` (contact normals); W2 surface normals from depth | `contact_normal` (contacts); `surface_normal` (depth_render) | arm, dual |
| qualitative spatial | above, below, left-of, right-of, front-of, behind, ordering along gravity / motion / task axis | pos3d × order(axis) × bias (antisymmetric) | W1 `geo.above`; others = same entry with `params.axis` | derived from pos3d | arm, dual, legged, cw |
| shape and scale | scale, relative scale, aspect ratio, thickness, width, height, length, volume, area, curvature, convexity, planarity, shape similarity | extent / shape × sim or unary | P | `extent` (poses + geom sizes) | arm, dual |
| region topology | containment, enclosure, overlap, intersection, interior / exterior / boundary membership, adjacency, separation, touching, near-contact, clearance, penetration depth | bbox / extent pair op → edge (estimated via bilinear) | W2 `ix.contains` (needs `bin` part); near-contact = W1 `ix.contact` with margin | `contains_pairs` (poses, extents) | arm, dual, cw (`ui.contains` W1) |
| visibility | visibility, occlusion, partial occlusion, line of sight, mutual visibility, ray intersection | unary (visible) / pair (occludes) × edge | W1 as readout `probe.arm.visible`; pair occlusion P | ray test (camera, poses) | arm, dual |
| symmetry / lattice | reflective / rotational / translational symmetry, repeated structure, lattice, pattern | same(part-class) / edge `mirror` | W1 `kin.mirror` (morphology); scene symmetry P | – | simple, arm |

## B. identity, membership, correspondence (fields: id / membership; ops: same, sim, hop, ancestor)

| family | candidates | decomposition | wave | label | envs |
|---|---|---|---|---|---|
| same-X | same object / part / subpart / component / rigid body / articulated body / semantic instance / category / class / material / texture / color region / functional region / motion group / support group / assembly | id field × same × aug (fixed codes) | W1 `id.same_body`, `id.same_assembly`; W2 `id.same_material` (needs material ids) | public ids / `materials` | all |
| correspondence | same point / patch / object across views or time, track identity, temporal persistence, object permanence, dense / sparse correspondence | track_id × same across time tokens | W1 declared `time.same_track` (used when a net has history tokens) | public tracker ids; gt `track` (poses over time) | arm, dual |
| segmentation | instance / semantic / panoptic / part / affordance / contact segmentation membership | membership id × same (patch ↔ entity) | P (needs image-patch tokens with masks) | segmentation render | arm |
| hierarchy | part-whole, parent / child / sibling / ancestor / descendant / peer, assembly hierarchy, component-subcomponent, scene graph, compositional graph, graph distance, centrality, articulation point, bridge, hub | parent_id / edges × ancestor / hop | W1 `kin.ancestor`, `kin.sibling` (morphology); scene hierarchy W2 | public morphology; scene parents (poses) | arm, legged, simple |
| equivalence | repeated-part, homologous, substitute, interchangeable part | same(part-class) | P | – | – |

## C. embodiment and kinematics (graphs over morphology tokens; public)

| family | candidates | decomposition | wave | envs |
|---|---|---|---|---|
| kinematic tree | kinematic-tree, kinematic chain membership, joint-to-link, link-to-link, kinematic dependency, end-effector relation | edges(arm-rel-v1 `kin_parent`, g1 `parent`/`child`) × edge / ancestor | W1 (legacy `edge.kin_parent`, `edge.kin_child`, `kin.ancestor`) | arm, simple, legged |
| ownership | manipulator ownership, left / right arm, body membership, effector binding, node-in-assembly | assembly_id × same; edges `node_in_assembly` | W1 (legacy + `route.own_assembly`, `id.same_assembly`) | all |
| joints | joint axis / origin / limit / state / velocity / torque / stiffness / damping, DOF, constrained / free DOF, articulation state | unary fields on joint tokens (already in node features) | M (inputs, not relations) | – |
| mirror / morphology correspondence | mirror, homologous limbs, cross-embodiment correspondence, morphology graph, action / state remapping | edge `mirror`; same(limb-class) | W1 `edge.mirror` (Ψ₀); cross-embodiment P | simple |
| locomotion | footholds, COM ↔ support polygon, stability margin, tipping, balancing, center of pressure | pos3d × sqdiff+diff gate task; unary `com_in_support` | W1 `leg.foothold`, `leg.com_support` | legged, humanoid |
| sensor–actuator | sensor-actuator, sensor-frame, efference copy, tactile-vision correspondence | edges sensor→assembly (legacy `node_in_assembly` on interact tokens) | W1 (legacy) | arm |

## D. physical interaction (dynamic pairs / graphs; labels from contacts, poses, gravity)

| family | candidates | decomposition | wave | label (caps) | envs |
|---|---|---|---|---|---|
| contact | touching, contact point / patch / manifold / area, collision partner / point / normal, contact graph, contact phase | hidden × bilinear × aug, source probe; contact normal → A | W1 `ix.contact` | `contact_pairs` (contacts) | arm, dual, legged |
| grasp | hand-object, finger-object, gripper-object, grasp point / region / axis / aperture, antipodal, pinch / power grasp, held-by | bilinear (held_by); unary grasp region | W1 `ix.held_by`; grasp region = Ψ₀ `probe.psi0.grasp_*` | `held_pairs` (contacts) | arm, dual, simple |
| support / stacking | support-object, stacked-on, resting-on, leaning-on, load-bearing, load-supported-by, support graph, support group | bilinear (support) + flow closure | W1 `ix.support` | `support_pairs` (contacts, poses, gravity) | arm, dual |
| force transfer | force chain / propagation, upstream / downstream force, load path, stress path, torque transfer, force graph | support / contact graph × flow × bias (transitive, directed by gravity or force) | W1 `ix.force_flow`; impulse-directed W2 | `support_pairs` closure; `contact_forces` (forces) | arm, dual |
| collisions over time | collision sequence / chain, time-to-collision, swept-volume intersection, dynamic clearance, shock propagation | contact over time × flow(time) | W2 | contacts over an episode | arm, dual, legged |
| handover | handover, handoff location, agent-agent interaction (robot–robot) | bilinear across manipulators over time | W1 `ix.handover` | `handover_pairs` | dual |
| attachment / articulation | rigid / compliant / detachable attachment, hinge / slider / ball / screw joint relation, part ↔ joint axis, drawer-handle, door-handle, lid-container, cap-bottle | bilinear + align(joint axis) | W2 `ix.articulation` (needs a `drawer` / `door` part) | `articulation_pairs` (joints) | arm |
| mating / insertion | peg-hole, key-slot, plug-socket, insertion point / axis / depth, mating axis, alignment constraint, assembly / disassembly constraint | bilinear + align(axis) | W2 `ix.mating` (support_insert task exists; needs a `peg_hole` part) | `mating_pairs` (poses) | arm |
| containment of content | container-content, receptacle-object, fill / empty | bilinear | W2 `ix.contains` (`bin` part) | `contains_pairs` | arm, dual |
| tool use | tool-object, tool-part, tool-tip, tool-working-surface, tool→target | bilinear gate task | W2 `ix.tool_target` (needs a tool part) | `tool_pairs` (contacts) | arm |
| physical properties | mass, inertia, friction (static / dynamic / rolling), stiffness, damping, elasticity, hardness, roughness, slipperiness, density, fragility, deformability | unary fields (key prior) or sim(material) | P `prop.*` (labels exist in MuJoCo: `inertia`, `materials`); used first as readouts | `inertia`, `materials` | arm, dual |
| mechanics | lever / moment arm, mechanical advantage, gear / pulley / tendon / cable / belt routing, backlash, slack, spring, damper | – | X (no such mechanisms in our scenes) | – | – |
| fields and flows | stress / strain / deformation / pressure / velocity / momentum / energy fields | – | X (rigid-body sims) except velocity: `vel` unary P | – | – |
| fluids, thermal, electrical, magnetic, optical, chemical | buoyancy, drag, heat flow, circuits, magnetism, reflectance, corrosion, ... | – | X | – | – |

## E. affordances and function (object → region / action; mostly unary scores or task-gated pairs)

| family | candidates | decomposition | wave | envs |
|---|---|---|---|---|
| affordance scores | graspability, liftability, pushability, openability, insertability, stackability, carryability, containability, ... (all "-ability") | unary field × unary op (key prior), readout from probe | P `aff.<verb>` (labels need per-object scripted affordance tables; start with graspable / supportable from scene parts) | arm, dual |
| affordance pairs | object → affordance region, handle-object, button-device, knob-device, switch-device, receptacle-object, fixture / clamp / locator | bilinear gate task | W2 with the matching part | arm, cw (button-device = `ui.label_for` analogue) |
| compatibility | tool-action, material-action, manipulator-action, object-action, goal-action compatibility, functional similarity / complementarity | sim / bilinear gate task | P | – |
| safety regions | safe / unsafe touch, no-go region, pinch / crush / edge hazard, preferred grasp / support region | unary field (region label) | P | arm |

## F. task, procedure and causality (public task structure; task-conditioned)

| family | candidates | decomposition | wave | label | envs |
|---|---|---|---|---|---|
| role / argument | agent-patient, instrument, source, destination, location, predicate-argument, subject-object, role relation | edges(arm-rel-v1: actor_of, patient_of, target_of, destination_of, support_of, role_points_to, pred_arg) | W1 (legacy) | public task graph | arm, dual |
| prerequisites | prerequisite, successor, dependency, enables, maintained, blocks, unlocks, output_to, produced, consumed_by, co-requirement, mutual exclusion, critical path | edges(enables, maintained, output_to, produced, consumed_by) + ancestor closure | W1 (legacy); closure W2 `task.precedes` | public | arm, dual |
| next interaction | next / previous / first / terminal / alternative contact point, sequential contact points, grasp→place, lift→transport→place, approach→contact→release, pickup / place location, source→destination, manipulation graph, plan branch | candidate edges × bilinear gate task, soft_ce (reveal) | W1 `task.next_contact` | `next_contact` (task runtime + contacts) | arm, dual, cw (drag) |
| phase | phase-of-action, pre-contact / contact / release / recovery phase, event boundary | unary readout (contact_phase) | W1 as readout (`probes:anchor-v1` contact_phase) | contacts over time | arm |
| causality | cause→effect, action-effect, intervention effect, counterfactual dependence, object-state transition, pre / postcondition, reveals / conceals / opens-access-to | edge (task graph) + flow(time) + cf_swap data | W2 `task.cause_effect` (labels from task receipts + interventions; data via cf_swap) | receipts, interventions | arm, dual |
| relevance / salience | task relevance, goal / subgoal relevance, distractor, salience (object, part, task, risk, novelty) | unary × gate task (readout = focus) | W1 as readout `probe.arm.focused_on`; `rel.task_relevance` gated unary W2 | public focus (active event binding) | arm, dual, cw |
| costs / values | cost, reward, utility, risk, information gain, exploration value | – | M (critics / planners, not attention factors) | – | – |

## G. temporal and motion

| family | candidates | decomposition | wave | envs |
|---|---|---|---|---|
| order in time | before, after, overlap, adjacency, temporal distance, duration | time × sqdiff+diff (1-d) / order | W1 via knot `time` fields (system 0 already keys knots by knot time − phase) | all |
| periodicity / phase | periodicity, phase relation, synchronization, anti-synchronization | time/phase × rel_rot on S¹ (cos / sin features = align) | W2 `time.phase_sync` (legged gait legs) | legged |
| motion | co-motion, relative motion, rigid / articulated / independent motion, optical / scene / object / point / contact flow | vel field × sim / diff | P (velocity labels exist) | arm, dual, legged |
| futures | future point / contact / object / manipulator trajectory, predicted / observed / intended / planned motion | readouts on the packet (observed_effect, goal_effect) | W1 as readouts (`probe.arm.observed_effect`, `goal_effect`) | arm, dual |

## H. user interface (ComputerWorld; public from semantic.v1 unless noted)

| candidates | decomposition | wave | label |
|---|---|---|---|
| label ↔ widget, containment tree, focus order, drag source → target, z-order depth, window membership | edges(ui-rel-v1) × edge × bias; drag = bilinear gate task (probe) | W1 `ui.label_for`, `ui.contains`, `ui.focus_next`, `ui.above`, `ui.drag_to` | ui_tree (public); drag target from the teacher (privileged) |

## I. language and grounding

| candidates | decomposition | wave |
|---|---|---|
| text-object / text-part / text-region / text-contact alignment, reference resolution, coreference, deixis, pointing, symbol-object correspondence, instruction-action alignment | bilinear between text tokens and entity tokens; label = task binding (public, `task`) or synthetic | W2 `lang.refers_to` (needs instruction tokens in the arm ctx; Ψ₀ / cw have text) |
| semantic similarity / opposition / entailment / synonymy / category / concept hierarchy | sim(text embedding) | P |

## J. epistemic (not attention factors: confidence scaling, probabilistic edges, data transforms)

| candidates | how it is expressed | status |
|---|---|---|
| confidence, aleatoric / epistemic uncertainty, measurement confidence, estimator confidence, pseudo-label confidence | `.var` fields + `confidence: true` scaling; Gaussian readouts | W1 (mechanism) |
| ambiguity, multimodality, hypothesis membership / compatibility / conflict, alternative explanation, occluded hypothesis | soft EdgeSets (prob) over candidates; bilinear readouts with `soft_ce` | W1 (`task.next_contact`) |
| belief update / revision / persistence / decay, surprise, prediction error, contradiction, hypothesis revival, awareness collapse / expansion, attentional commitment / entropy | transforms `reveal`, `surprise`; recovery metric; attention-entropy diagnostic | W1 (transforms R9) |
| novelty, anomaly, missing / corrupted data, sensor noise / bias / drift / latency | transforms `noise`, `occlude`; packet OOD (existing `policies.packet_ood`) | W1 (transforms) |
| data / supervision provenance, source reliability, teacher-student, distillation, synthetic-real, sim-to-real, domain gap | `LabelProv` on every label; `sources` of every factor | M (provenance, section 7 of the design) |

## K. meta (properties OF relations; metadata or design choices, never entries)

transitive / symmetric / antisymmetric / reflexive / irreflexive, one-to-one / one-to-many / many-to-many, directed /
undirected, signed / weighted / probabilistic / deterministic, continuous / discrete, local / global, short / long
range, static / dynamic, latent / explicit / observed / inferred / predicted / counterfactual / hypothetical,
task- / instruction- / action- / memory- / surprise-induced → `Algebra` fields, `sources`, `gates`.
Relation gating / strength / polarity / confidence / sparsity / rank / persistence / competition / composition /
chaining / closure → forms, gates, confidence, `heads`, ops `ancestor` / `flow`. Translation / rotation / permutation /
scale / reflection equivariance, viewpoint / embodiment / material / task invariance → choices of field frame
(`frame: query` for SE(3)), token-set permutation equivariance (attention), and decoupling / composition data
(docs/relations.md 5.2–5.5). Head / layer / relation-head specialization → diagnostics (`FactorSite.contributions`).
Calibration relations (camera-camera, hand-eye, intrinsics / extrinsics, temporal calibration) → env provenance, not
factors. Conditioned-attention items (task-, instruction-, goal-, embodiment-, history-, memory-, user-, modality-,
context-conditioned) → the `gates` of any entry.

## composable now (curriculum levels, docs/relations.md 5.5)

| env | parts available after W1/W2 units | dynamics that compose |
|---|---|---|
| `mujoco/arm`, `mujoco/dual` | `table_objects`, `camera_depth`, `grasp_target`, `stack`, `handover` (dual) | {depth, orient, above} × {contact, held} × {support, force_flow} × {next_contact}; handover (dual) |
| `mujoco/legged`, humanoid | `terrain_steps` | {foot contact} × {foothold} × {com_support} |
| `computerworld` | the app scenes | {label_for, contains, focus_next, above, drag_to} |
| wave 2 parts | `drawer` / `door` (articulation), `peg_hole` (mating), `bin` (containment), tool | adds articulation, mating, contains, tool_target |
