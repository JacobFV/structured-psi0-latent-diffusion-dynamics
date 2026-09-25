# correction 2026-09-25: prove causal packet semantics before expanding (mother-agent feedback, relayed by the user)

Authoritative. It overrides earlier plans where they conflict. Quoted verbatim:

> The controller-facing latent correction is now implemented in the right architectural direction. Do not redesign the stack again. Prioritize proving that task semantics in the transmitted packet causally control behavior.
>
> 1. Finish and review the binding-track diagnostic repair. Preserve the original result but label its asymmetric graph edit and slot-prior limitations. Use symmetric rebinding, address permutations, and balanced binding conditions.
>
> 2. Extend binding diversity THROUGH THE WHOLE PIPELINE. Stage-A semantic-only counterfactuals are insufficient while stage B and system 0 see fixed bindings. Generate physically consistent paired tasks with identical initial scenes and different assigned objects/manipulators, with valid corresponding demonstrations. Train representation, generator, and realizer on these pairs. Keep same-trajectory/different-meaning examples separately; do not give them contradictory action labels.
>
> 3. Before expanding expensive sweeps, localize closed-loop failure on a small source task:
>    expert native commands → tracker;
>    expert-encoded packet → system 0 → tracker;
>    generated packet → system 0 → tracker.
>    Use matched scenes, frozen checkpoint identities, and explicitly label the expert-encoded route as an oracle diagnostic. Report success, tracking error, approach/grasp failures, and fixed-packet disturbance recovery.
>
> 4. Packet dependence is not semantic control. Zero/shuffle sensitivity and probe accuracy are insufficient. On a competent checkpoint, demonstrate that valid changes in object binding, manipulator assignment, or requested physical effect produce the corresponding behavior. Include irrelevant edits as controls.
>
> 5. Separate goal_effect, predicted_effect, and observed_effect. The present desired_delta probe uses future_disp, which is realized motion rather than necessarily intended task change. Keep that predictor but rename it accurately and add explicit task-goal supervision.
>
> 6. Fingerprint the frozen representation/checkpoint bundle in latent-space compatibility IDs; a configuration hash alone is insufficient. Add a mismatch-rejection test.
>
> 7. Reserve shared disk before downloads/transfers; do not let peripheral work interrupt the binding/control critical path. Reconcile STATUS, README, branch locations, resource authorization, and resume commands. Preserve all previous resource accounting and failed runs.
>
> Continue development autonomously. Publish one current evidence matrix distinguishing implementation, oracle-target behavior, generated-packet behavior, semantic interventions, and held-out transfer. A functioning source controller and causally meaningful packet semantics take priority over more robot names or additional training variants.

## owners (D-037)
| item | owner |
|---|---|
| 1 diagnostic repair review, 2 whole-pipeline binding diversity (single-arm objects), 5 goal/predicted/observed effect | binding track |
| 2 paired manipulator-assignment tasks (dual-arm: same scene, different arm assigned) | dualarm track (re-scoped) |
| 3 closed-loop failure localization ladder | ladder track (new) |
| 4 semantic interventions with irrelevant-edit controls, on a competent checkpoint | acceptance track (re-scoped) |
| 6 bundle fingerprint in compatibility IDs + mismatch-rejection test | lead |
| 7 disk reservation (done, D-036), priorities, STATUS/README reconciliation, evidence matrix | lead |
Peripheral work (legged/VLM breadth, extra baseline/training variants) must not take resources from the critical path.
