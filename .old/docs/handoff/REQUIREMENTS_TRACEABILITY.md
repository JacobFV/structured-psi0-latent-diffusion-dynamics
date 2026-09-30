# requirements traceability

All statuses begin unimplemented/unexecuted. The agent writes proof-linked results into `artifacts/requirements.json`; this table describes obligations, not accomplishments.

| id | requirement | tasks | contract | required evidence |
|---|---|---|---|---|
| R01 | autonomous end-to-end implementation, testing and actual research | P01–P25 | `docs/00_scope_and_acceptance.md` | proof-linked release matrix; no scaffold-only closure |
| R02 | two GB10s; existing SSH peer discovery without guessing identity | P02,P03 | `docs/03_resource_safety.md` | redacted distinct-host capability receipts |
| R03 | aggregate host use ≤50% CURRENTLY FREE resources | P02,P03 | `docs/03_resource_safety.md` | effective parent quotas, measured baseline, load-change and child tests |
| R04 | fresh implementation with meaningful ψ₀ pretrained integration | P14,P16 | `docs/01_architecture.md` | source-locked backbone; real image features and learned rollout |
| R05 | minimize new-body deployment cost; zero-shot plus small fine-tuning | P17,P21,P23 | `docs/07_research_protocol.md` | zero-update and adaptation curves; controller/integration accounting |
| R06 | ψ₀-style low-level tracking and controller-specific action contracts | P08,P12 | `docs/04_simulation_robots_and_surgery.md` | separate validated original/SONIC/native command contracts |
| R07 | privileged training labels without inference leakage | P01,P06,P07,P16 | `docs/02_interfaces.md` | serialized public/private separation and leakage tests |
| R08 | supplied compositional task hypergraph | P04,P05 | `docs/02_interfaces.md` | ordered role/incidence, predicates/effects, actual output binding |
| R09 | GUI to watch, request and directly edit task graph | P10,P11 | `docs/05_workbench.md` | actual browser commands, versioned edits, stale chunk rejection |
| R10 | GUI to interact with robots, objects, targets and manipulators | P07,P10,P11 | `docs/05_workbench.md` | physics-backed selection/movement/replay browser evidence |
| R11 | many humanoids, quadrupeds, hexapods, arms, dual/mobile bodies | P09,P12,P22 | `docs/04_simulation_robots_and_surgery.md` | catalogue plus distinct executed family/controller/policy ledger |
| R12 | physically synthetic hand/arm/body swaps | P09,P17 | `docs/04_simulation_robots_and_surgery.md` | validated attachments, physics lineage and heldout combinations |
| R13 | structural node initialization and coarse manipulator/body nodes | P09,P14 | `docs/01_architecture.md` | hierarchical pooling/FK and geometry/identity tests |
| R14 | variable-node flow action trajectory; node identity separate from noise | P13,P14 | `docs/01_architecture.md` | variable n, permutation, masks and independent actuator tests |
| R15 | graph structure directly available to system i | P14 | `docs/01_architecture.md` | graph bias/input tests independent of VLM encoding |
| R16 | four clean context banks, one noised action stream | P14,P23 | `docs/01_architecture.md` | cached/uncached equivalence; packed versus naive latency |
| R17 | object visibility versus looking/focus/action distinctions | P15 | `docs/01_architecture.md` | canonical slot probes, occluded-action and multiobject tests |
| R18 | object embeddings shared across prompts/views | P15,P16 | `docs/01_architecture.md` | entity correspondence and prompt-invariance controls |
| R19 | object QA projection into separate frozen text transformer | P15,P16 | `docs/06_learning_and_adaptation.md` | live slot/projector gradients, frozen decoder, blank/shuffle controls |
| R20 | held object query for each manipulator | P15 | `docs/01_architecture.md` | held-by relation per manipulator including handover/dual hold |
| R21 | relative object/manipulator/camera/body geometry | P13,P15 | `docs/01_architecture.md` | frame-aware poses/distances and analytic FK baseline |
| R22 | queryable per-manipulator compositional task representation | P05,P15 | `docs/02_interfaces.md` | role/set readout, bimanual overlap and role intervention |
| R23 | task-role invariance under compatible morphology swaps | P17,P22 | `docs/06_learning_and_adaptation.md` | paired feasible swap alignment without capability erasure |
| R24 | predicates, desired deltas, dependencies and functional composition | P04,P05,P15,P22 | `docs/02_interfaces.md` | real output-to-input dependence versus supplied stage sequence |
| R25 | topoformer progress informs typed routing and input audits | P05,P14,P22 | `docs/09_sources_and_topoformer.md` | read-only source lock; provenance/failure-memory counterfactual tests |
| R26 | contact-centric active reference frames | P07,P15,P22 | `docs/01_architecture.md` | contact creation/slip/expiry/tangent uncertainty and causal ablation |
| R27 | uncertainty and missing/false distinction | P06,P15,P22 | `docs/01_architecture.md` | calibration, null channels, risk/coverage and partial observation |
| R28 | GRPO fine-tuning with minimal experience | P18,P23 | `docs/06_learning_and_adaptation.md` | valid path likelihood math and actual online update/outcomes |
| R29 | shared-prefix/event-boundary branching | P06,P18,P23 | `docs/06_learning_and_adaptation.md` | complete continuation snapshots and correct credit/accounting |
| R30 | EXPO-FT off-policy adaptation comparison | P19,P23 | `docs/06_learning_and_adaptation.md` | verified source-derived objective and actual short learning run |
| R31 | trainable action latent with an explicit valid target | P13,P14 | `docs/01_architecture.md` | codec reconstruction/rollout gates plus direct-action comparator |
| R32 | matched baselines and no fake positive research results | P20,P21,P24 | `docs/07_research_protocol.md` | same public information, independent seeds, raw counts and failures |
| R33 | sampler and end-to-end latency measured on real machine | P14,P23 | `docs/07_research_protocol.md` | synchronized components, NFE/node-size sweep and memory telemetry |
| R34 | pretrained/medium/larger profiles after small tests | P16,P23 | `docs/01_architecture.md` | actual VLM-backed run and larger profile receipt or explicit resource boundary |
| R35 | resume without duplicating or destroying other work | P03,P06,P20,P25 | `docs/10_autonomy_and_recovery.md` | lease expiry, ownership, checkpoint/restart and owned-only shutdown |
| R36 | reproducible docs, checkpoints, data, reports and meeting artifacts | P24,P25 | `docs/08_tests_and_release.md` | real commands, local paths+hashes, videos/brief and clean-environment check |
| R37 | concepts shape system-i denoising hidden/action subspaces, not only context | P15,P16,P22 | `docs/01_architecture.md` | action-expert auxiliary gradients and causal control interventions |
