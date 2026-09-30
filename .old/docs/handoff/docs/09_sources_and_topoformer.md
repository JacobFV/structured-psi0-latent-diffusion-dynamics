# source audit and what to transfer from topoformer

Sources are leads and factual constraints, not blanket permission to reuse code/assets. Resolve current versions once at project inception, inspect license/dependencies, pin revisions, and record any differences from this audit. Do not repeatedly track a moving main branch during a frozen experiment. Do not clone a multi-gigabyte repository including every historical artifact just to read three files: fetch selected files or use a filtered read-only checkout.

## primary references

- ψ₀ paper: https://arxiv.org/abs/2603.12263 ; original HTML https://arxiv.org/html/2603.12263v1 ; code https://github.com/physical-superintelligence-lab/Psi0
- ψ₀ SONIC release: https://github.com/physical-superintelligence-lab/Psi0/blob/main/examples/psi0_for_sonic.md ; inspected blob `988f77230eab5d8d7c48aeaed3d0b55914e8ec68`.
- Z-1: https://arxiv.org/abs/2606.31846 ; methodology https://arxiv.org/html/2606.31846v1
- EXPO-FT: https://arxiv.org/abs/2605.25477 ; methodology https://arxiv.org/html/2605.25477v2
- unified latent-space reference: https://arxiv.org/abs/2601.15419 ; methodology https://arxiv.org/html/2601.15419v1
- SIMPLE: https://github.com/physical-superintelligence-lab/SIMPLE
- morphology assets: https://github.com/google-deepmind/mujoco_menagerie ; each asset has its own license and model notes.
- additional description discovery: https://github.com/robot-descriptions/robot_descriptions.py ; verify each actual model source rather than counting descriptions as controllers.
- task graph UI: https://reactflow.dev/learn/advanced-use/computing-flows
- physical/actuator modeling: https://mujoco.readthedocs.io/en/stable/modeling.html
- native editable model API: https://mujoco.readthedocs.io/en/stable/python.html ; verify installed-version support rather than assuming examples from a newer release.
- NVIDIA platform: https://docs.nvidia.com/dgx/dgx-spark/system-overview.html
- NVIDIA unified-memory optimization: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html
- existing two-node networking reference: https://build.nvidia.com/spark/connect-two-sparks ; do not execute its privileged reconfiguration instructions on already-connected machines.
- MPS limitations/limits: https://docs.nvidia.com/deploy/mps/when-to-use-mps.html and https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html
- NVIDIA GPUDirect caveat: https://nvidia.custhelp.com/app/answers/detail/a_id/5780 ; introspect actual platform support.

The original ψ₀ action contract and its current SONIC interface differ. The inspected release note describes 36-dimensional prior commands versus an 80-dimensional SONIC command grouping and revised state/feature handling. Source-specific adapters must enforce those differences, not infer interchangeability from the common model name.

Z-1 is a post-training reference built on another base VLA; port its shared-prefix/branching and flow-SDE objective carefully rather than claiming its reported performance transfers to ψ₀. EXPO-FT is an off-policy adaptation comparison with edit and base-policy learning, not simply an always-frozen residual. The unified-latent paper motivates shared motion/control representation; it does not by itself establish our task/event semantics or zero-update arbitrary-URDF deployment. Read their exact methods before implementation.

## topoformer revisions inspected for this package

Canonical repository is https://github.com/JacobFV/topoformer ; the user's pasted `/repo` suffix is not a required Git path.

At package preparation, the GitHub branch read returned:

```text
main / campaign/extended-02: e80ddacbe731a1ae064d869962bd63720e5f86c0
campaign/extended-03: a19ff5d8275a659961a2494e69b0f9558f1f3d3a
historical extended-01: 65d44ae6acb17cd53360e5350f9c297b0c5954d0
```

The branch existence of extended-03 is verified, NOT its results. Read its latest relevant report/status if it informs this project, and distinguish proposed work from executed evidence. Do not interfere with its running jobs or working directory.

Required reference reads:

```text
research/stages/stage-01/design.md
src/topoformer/attention.py
research/campaigns/extended-01/composition/P01-protocol.md
research/campaigns/extended-01/composition/C04-claim-prerequisites-and-next.md
research/campaigns/extended-02/final-report.md
research/campaigns/extended-02/diagnostics/encoded-counterfactuals.md
```

The extended-02 final report at the pinned main commit was inspected here (blob `4d92b5c39df46e3395d9533ebeb5f05fb9935379`). It identifies missing provenance and rejection-history distinctions in actual encoded inputs, distinguishes supplied stage sequencing from untested functional composition, and does not establish a structural-attention advantage. These are constraints on interpretation, not a negative verdict on this new architecture.

## transfer these ideas, not unsupported claims

Use query-row/key-column read orientation and zero-bias equivalence; typed ordered roles; explicit instance identity/version/provenance; trained input discrimination exposed by counterfactuals; separate supplied structure from learned scheduling; and compare sophisticated workspaces with competent simpler baselines.

Implement a fast attention backend only after reference-output/gradient tests. A dense diagnostic kernel that returns all weights is not automatically an optimized sampler kernel. A correct per-event type/argument decoder is not full planning; a working supplied event sequence is not functional data-dependent composition.

Do not import the entire topoformer research monorepo as a heavy runtime dependency. Copy/adapt only license-permitted small interfaces or reimplement the documented math in our repository with attribution, and pin any consumed code. Keep reference checkouts read-only.

## new literature pass

Perform a focused primary-source search on morphology-conditioned policies, graph/transformer control, compositional task policies, object-centric representations, contact-rich reference frames, latent-action codecs, and VLA RL. Write a related-work table: exact claim; source; what is supplied; what is learned; body/task regime; sample/compute accounting; and our difference. Do not spend days collecting papers before the first runnable vertical slice. Novelty is a documented comparison, never guaranteed by this handoff.
