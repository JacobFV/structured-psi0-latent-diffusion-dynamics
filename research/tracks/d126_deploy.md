# D-126 deployment credibility and infra (track notes, append-only)

## source labels
Backlog row "Canonical source labels everywhere" (sub-agent, branch `track/d126deploy-source`). State: **verified**
(unit tests + a real 4-step teacher-route ladder rollout on parm5_pg2, host, tiny).

- Switch: env `RRP_SOURCE_LABELS=canonical` or explicit `source_labels=True` (LadderConfig field,
  `evaluate_dual_latent(..., source_labels=)`, `run_audit_episode(..., source_labels=)`,
  `run_policy_quality_episode(..., ckpt=, source_labels=)`). Default OFF. Version tag `sl-1`.
- On: new rows gain `source_label` (strict canonical string) and `source_label_version: "sl-1"`; `source` unchanged.
  Canonical labels: arm ladder teacher `scripted_teacher:privileged`, oracle `oracle:teacher_future` or
  `oracle:learned_chunk:<label>`, generated `learned:<flow ckpt>`, learned `learned:<policy label>`; dual latent eval
  `learned:<policy>[+edit:<e>]`; dual teacher audit `scripted_teacher:dual_<task>`; arm teacher_quality policy rows
  `learned:<ckpt>` (or `bc:<ckpt>` if the label parses as bc). Teacher rows already wrote canonical strings.
- Byte identity (off): the parm5_pg2 teacher ladder row (seed 0, 4 steps, wall_s dropped) was byte-identical before
  (origin/main 733b02a) and after the change (`cmp`); `test_source_labels.py` freezes its key order and source string.
- Readers: `row_source` (contracts/provenance); `pipelines/arm._source_counts` (manifest `source_kinds`) uses it;
  `robustness.summarize_shard` adds `source_labels` only when some row has one (old summaries unchanged).
  Other readers read the unchanged legacy `source` key, so they need no change. The service/UI labels by mode and
  does not filter by row source.
- Wire: `contracts.action.Source` / `LatentActionChunk.source` widened with canonical kinds; `check_packet` and
  `contracts/channels.py` do not read the source (unaffected).
- Left: `evaluation/legged_latent_eval.py` (not touched, parallel edit) writes `row["source"]` in `run_episode`
  (`src = ...; row = dict(body=..., source=src, ...)`). Its legacy strings already parse with `row_source`
  (scripted_teacher[:arc_only], privileged_oracle_packet:<lsv> -> oracle, oracle_diagnostic:... -> oracle, ctl.policy_version
  = learned:/bc:<ckpt>). Writer change for the parent: after `row = dict(...)` add
  `lab = parse_legacy_source(src); stamp_source_label(row, "oracle" if lab.kind is Source.ORACLE else lab.kind,
  (f"privileged_packet:{ctl.lsv}" if src.startswith("privileged_oracle_packet") else lab.detail))`
  (import from rrp.contracts.provenance; default off, so rows stay byte-identical). The arm ladder `learned` route is
  a plain FlowPolicy BC baseline but its legacy string says `learned:`; relabelling it `bc:` is a semantic change
  left for a decision (the sl-1 stamp keeps kind `learned` so it agrees with the legacy string).
