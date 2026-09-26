#!/usr/bin/env bash
# Demo sprint: copy the SMALL raw result files that docs/demo/index.html cites from the shared peer store into
# docs/demo/raw/artifacts/runs/<same path>. Only JSON/JSONL summaries; never weights. Extra paths may be passed as args.
set -euo pipefail
PEER=${ROBOT_PEER:-gb10-direct}
S=/dev/shm/rrp-brandonin/repo/artifacts/runs
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
D=$ROOT/docs/demo/raw/artifacts/runs
mkdir -p $D
FILES=(
  'ladder_v1/*/*.summary.json'
  'lead_teacher_reanchor/*.json'
  'acceptance_semantic_teacher/semantic_summary_teacher.json'
  'acceptance_semantic_sem_v1rep/semantic_summary_oracle.json'
  'acceptance_causal_sem_v1i/window_summary.json'
  'acceptance_latency/sem_v1_interrupted.json'
  'dualarm_teacher_ref/*.jsonl'
  'legged_vlm_teacher_ref/eval_dev.summary.json'
  'latent_sem_v1/probe_posthoc.json' 'latent_nosem_v1/probe_posthoc.json' 'latent_nosem_v1/probe_metadata_only.json'
  'latent_sem_v1/counterfactuals_posthoc.json' 'latent_nosem_v1/counterfactuals_posthoc.json'
  'binding_v1_reeval/*.json'
  'grpo_latent_ref_v2s22k/result.json' 'grpo_latent_v2s22k_p40_v1/result.json'
  'ladder_smoke/t0_check_sem_panda.json'
  "$@"
)
cd $D; L=$(mktemp)
ssh $PEER "cd $S && ls -d ${FILES[*]} 2>/dev/null | grep -v '\.pt$'" > $L || true
rsync -a --files-from=$L $PEER:$S/ $D/
rm -f $L
find $D -type f | wc -l; du -sh $D
