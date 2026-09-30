# track d126deploy-sys2 — system II harness (roadmap #31, harness only; D-126 sub-agent)

State: verified (unit tests only; no model trained, no VLM run, no closed-loop result — nothing here is a system II claim).

- `rrp.evaluation.system2` (SYSTEM2_HARNESS_VERSION `s2h-1`): instruction -> System2 (`OracleSystem2` DIAGNOSTIC,
  `DefaultSystem2`, `MockSystem2`, `VLMSystem2` = frozen psi0 System-II VLM, lazy) -> `System2Target` (order / halt /
  mirror_goal + source label) -> `apply_target` (legged public ctx) / `scenario_for` (public task-graph binding;
  privileged evaluator = instruction truth) -> `ground_eval`, `closed_loop_eval`; CLI `python -m rrp.evaluation.system2`.
- Opt-in hook `System2ContextProvider` + `system2_episode(system, body, seed)`; default off (`make_system2("off") is None`).
- Moved code (identical source, verified by AST compare): instruction/scene helpers -> `rrp.evaluation.system2`;
  VLM wrapper -> `rrp.models.system2_vlm` (torch lazy). `rrp.research.system2` re-exports the same objects;
  `rrp.research.system2_eval` unchanged.
- Tests: `tests/unit/test_system2_harness.py` (ctx math incl. equality with LatentLeggedController._ctx halt/mirror_goal
  edits, default-off identity, oracle labelling + truth only to the oracle, mock/VLM(injected) grounding, episode wiring).
- Not wired into `legged_latent_eval` (another agent owns it); wiring proposal in the D-126 report.
