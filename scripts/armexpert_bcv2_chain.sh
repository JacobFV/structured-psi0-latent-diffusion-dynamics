#!/usr/bin/env bash
# W7 step 2 (D-097): stateless direct-action BC expert trained on the v2-teacher pack, evaluated exactly like sprint_bc.
# Usage: SEED=1701 PLACE=host|peer scripts/armexpert_bcv2_chain.sh
#   train  : rrp campaign baseline-cell --train-only (policy-small-structured, packed stride 2, exact resume, B-1 fix),
#            --source-pack latent_pp_v4dart_s1_H16, snapshot at 12000 updates (the u12000 DAgger-expert point).
#            Bounded relaunches (<= 4 leases; exact resume continues an interrupted run).
#   evals  : (CPU leases) ladder `learned` route, panda_pg2 + parm6_tf3, 30 feasible from 3,000,000 and 3,000,200, for
#            u12000 and final; held-out source bodies parm5s_tf3/parm5l_pg2 (protocol harness, 50 from 2,000,000) for
#            u12000 and final; budget-0 zero-shot cells panda_tf3 / xarm7_pg2 / xarm7_tf3 (100 each) for final;
#            motion metrics of the BC rollouts (teacher_quality --policy).
# Every step checks its exit code; a failed step stops the chain (no unbounded retries, D-061).
set -uo pipefail
cd "$(dirname "$0")/.."
SEED=${SEED:?}; PLACE=${PLACE:?}
ROOT=${ROOT:-artifacts/runs/armexpert_bcv2}
PACK=${PACK:-artifacts/packed/latent_pp_v4dart_s1_H16}
SRC=$ROOT/baseline_direct_action/seed$SEED/source
EV=${EV:-artifacts/runs/armexpert_bcv2_eval}
GC=${GC:-v1}                     # grasp contact version for every evaluation (must match the pack's)
TAGP=${TAGP:-bcv2}
HPY=$HOME/work/relational-robot-policy/.venv/bin/python
log() { echo "$(date '+%F %T') [bcv2 s$SEED] $*"; }

run_job() {   # $1 label $2 cpu $3 mem $4 max_s $5 gpu(0/1) -- cmd...   (python = "PY")
  local label=$1 cpu=$2 mem=$3 maxs=$4 gpu=$5; shift 5
  local g=(); [ "$gpu" = 1 ] && g=(--gpu --gpu-mem 16G)
  set -- env RRP_GRASP_CONTACT=$GC "$@"
  if [ "$PLACE" = peer ]; then
    RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/armexpert scripts/peer_run.sh "${g[@]}" --cpu "$cpu" --mem "$mem" \
      --label "$label" --max-seconds "$maxs" -- "$@" > /tmp/armexpert_$label.out 2>&1
  else
    local a=(); for x in "$@"; do [ "$x" = PY ] && a+=("$HPY") || a+=("$x"); done
    PYTHONPATH=src "$HPY" -m rrp.cli ops run "${g[@]}" --cpu "$cpu" --mem "$mem" --label "$label" \
      --max-seconds "$maxs" -- env PYTHONPATH=src OMP_NUM_THREADS=1 "${a[@]}" > /tmp/armexpert_$label.out 2>&1
  fi
  local rc=$?
  tail -2 /tmp/armexpert_$label.out
  grep -q '"returncode": 0' /tmp/armexpert_$label.out && return 0
  return $(( rc == 0 ? 1 : rc ))
}

have() {   # file exists on the placement's store
  if [ "$PLACE" = peer ]; then ssh gb10-direct "test -s /dev/shm/rrp-brandonin/wt/armexpert/$1"; else test -s "$1"; fi
}

# ---------------------------------------------------------------- input check: the pack's grasp physics == the eval's
if [ "$PLACE" = peer ]; then pg=$(ssh gb10-direct "python3 -c \"import json;print(json.load(open('/dev/shm/rrp-brandonin/wt/armexpert/$PACK/meta.json')).get('grasp_contact_version') or 'grasp_v1')\"")
else pg=$(python3 -c "import json;print(json.load(open('$PACK/meta.json')).get('grasp_contact_version') or 'grasp_v1')"); fi
[ "$pg" = "grasp_$GC" ] || { log "FAILED: pack $PACK has $pg, evaluation would use grasp_$GC"; exit 1; }
log "pack $PACK grasp contact $pg"

# ---------------------------------------------------------------- train (bounded relaunches)
for attempt in 1 2 3 4; do
  have $SRC/policy.pt && break
  log "train attempt $attempt ($PLACE)"
  run_job ax${TAGP}_train_s$SEED ${TRAIN_CPU:-3} 24G 21000 1 PY -m rrp.cli campaign baseline-cell --method baseline_direct_action \
    --seed $SEED --target source --train-only --root $ROOT --source-pack $PACK --snapshot-steps 12000 \
    || log "train lease ended without success (attempt $attempt); exact resume on the next lease"
done
have $SRC/policy.pt || { log "FAILED: no policy.pt after 4 leases"; exit 1; }
log "trained"

# ---------------------------------------------------------------- evaluations (CPU)
for ck in u12000 final; do
  f=$SRC/policy_u12000.pt; [ $ck = final ] && f=$SRC/policy.pt
  tag=${TAGP}_direct${SEED}_$ck
  for r in panda_pg2 parm6_tf3; do
    for ss in 3000000 3000200; do
      have $EV/$r/learned_${tag}_s$ss.summary.json && continue
      run_job ax${TAGP}_lad_${SEED}_${ck}_${r}_$ss 2 6G 7200 0 env CUDA_VISIBLE_DEVICES= PY scripts/ladder.py --route learned \
        --policy $f --policy-label learned:$tag --robot $r --n 30 --seed-start $ss --tag ${tag}_s$ss --out $EV/$r \
        || { log "FAILED ladder $ck $r $ss"; exit 1; }
    done
  done
  [ $ck = final ] && continue          # final: the protocol source cell below (same harness)
  have $EV/heldout/$tag.summary.json || run_job ax${TAGP}_ho_${SEED}_$ck 2 6G 7200 0 env CUDA_VISIBLE_DEVICES= PY \
    research/scripts/2026-09-26/armexpert_bc_heldout.py $f $EV/heldout/$tag.jsonl $SEED \
    || { log "FAILED heldout $ck"; exit 1; }
done
have $ROOT/baseline_direct_action/seed$SEED/cells/source.json || run_job ax${TAGP}_src_$SEED 2 6G 7200 0 env CUDA_VISIBLE_DEVICES= \
  PY -m rrp.cli campaign baseline-cell --method baseline_direct_action --seed $SEED --target source --root $ROOT \
  --source-pack $PACK --eval-device cpu || { log "FAILED source cell"; exit 1; }
for t in panda_tf3 xarm7_pg2 xarm7_tf3; do
  have $ROOT/baseline_direct_action/seed$SEED/cells/${t}_b0.json && continue
  run_job ax${TAGP}_b0_${SEED}_$t 2 6G 10800 0 env CUDA_VISIBLE_DEVICES= PY -m rrp.cli campaign baseline-cell \
    --method baseline_direct_action --seed $SEED --target $t --budget 0 --root $ROOT --source-pack $PACK --eval-device cpu \
    || { log "FAILED b0 $t"; exit 1; }
done
have $EV/motion/${TAGP}_s$SEED.jsonl || run_job ax${TAGP}_mot_$SEED 2 6G 7200 0 env CUDA_VISIBLE_DEVICES= PY -m rrp.evaluation.teacher_quality \
  --bodies panda_pg2,parm6_tf3,parm5s_tf3,parm5l_pg2 --seeds 3000000-3000039 --workers 2 --chunk 20 \
  --policy learned:${TAGP}_direct${SEED}_u12000=$SRC/policy_u12000.pt --policy learned:${TAGP}_direct${SEED}_final=$SRC/policy.pt \
  --out $EV/motion/${TAGP}_s$SEED.jsonl || { log "FAILED motion"; exit 1; }
log "DONE"
