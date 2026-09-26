#!/usr/bin/env bash
# Generic arm-lineage driver (copy of scripts/armnosem_chain.sh, parametrized; the nosem chain keeps its own script).
# LIN=sfjf: SEM with the bounded semantic NLL (D-085 fix: Stage A latent.probe_lv_min -4; configs/ladder/armsemfix/).
# Same recipe, seeds, DAgger seeds, bodies and buffer compositions as the frozen sem lineage. Runs ON THE PEER.
# Resume: rerun; done nodes are skipped. A .failed node blocks dependants (manual decision, no retry; D-061).
# Usage: LIN=sfjf bash scripts/arm_lineage_chain.sh [node ...]
set -uo pipefail
P=/dev/shm/rrp-brandonin
cd $P/wt/ladder
export PATH=$P/bin:$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$PWD RRP_OPS_ROOT=$P/repo MUJOCO_GL=egl
PY=$P/venv/bin/python
LIN=${LIN:?set LIN=sfjf}
case $LIN in
  sfjf) C=configs/ladder/armsemfix; REPN=ladder_latent_semfix_b1fix_anchor; REPCFG=rep-latent_semfix_b1fix_anchor.json; TG=sf ;;
  *) echo "unknown LIN $LIN"; exit 2 ;;
esac
ST=artifacts/runs/ladder_arm${LIN}_state; mkdir -p $ST
RUNS=artifacts/runs
BODIES="panda_pg2 parm5_pg2 parm5_tf3 parm5l_tf3 parm5s_pg2 parm6_pg2 parm6_tf3 parm7_pg2 parm7_tf3 sawyer_pg2 sawyer_tf3 ur5e_pg2 ur5e_tf3"
# body groups of the sem collections (4 jobs x 3 CPU)
G1="panda_pg2 parm5_pg2 parm5_tf3 parm5l_tf3"; G2="parm5s_pg2 parm6_pg2 parm6_tf3 parm7_pg2"; G3="parm7_tf3 sawyer_pg2 sawyer_tf3"; G4="ur5e_pg2 ur5e_tf3"
REP0=$RUNS/$REPN/representation.pt
F0=$RUNS/ladder_flow_${LIN}/snap_final_s20000.pt
FFT=$RUNS/ladder_flow_${LIN}_ft/policy.pt
FG1=$RUNS/ladder_flow_${LIN}_gdag1/policy.pt
FG2H=$RUNS/ladder_flow_${LIN}_gdag2h/policy.pt
rz() { echo $RUNS/ladder_rz_${LIN}_$1/representation.pt; }
log() { echo "$(date '+%F %T') [$1] ${*:2}" | tee -a $ST/chain.log; }

need() {  # wait (bounded: 16 h) for dependency markers; abort if any failed
  local t0=$(date +%s)
  for d in "$@"; do
    until [ -f $ST/$d.done ]; do
      [ -f $ST/$d.failed ] && return 1
      [ $(( $(date +%s) - t0 )) -gt 57600 ] && return 1
      sleep 30
    done
  done
}
ops() { python3 -m rrp.cli ops run "$@"; }   # non-detached: exits non-zero iff the job's rc != 0

node() {  # node <name> <deps (comma list or -)> <function...>
  local n=$1 deps=$2; shift 2
  [ -f $ST/$n.done ] && { log $n "already done"; return 0; }
  [ -f $ST/$n.failed ] && { log $n "marked failed; not retrying"; return 1; }
  if [ "$deps" != - ] && ! need ${deps//,/ }; then log $n "dependency failed/timeout ($deps)"; touch $ST/$n.failed; return 1; fi
  log $n "start"
  if "$@" >> $ST/$n.out 2>&1; then log $n "done"; touch $ST/$n.done; else log $n "FAILED rc=$?"; touch $ST/$n.failed; return 1; fi
}

# ---- job bodies ----
train_gpu() {  # label cfg kind(rep|flow|rz) gpu_mem max_s
  local label=$1 cfg=$2 kind=$3 gm=$4 mx=$5
  case $kind in
    rep)  ops --gpu --gpu-mem $gm --cpu 4 --mem 24G --label $label --max-seconds $mx -- $PY -m rrp.cli latent train-representation --config $cfg ;;
    flow) ops --gpu --gpu-mem $gm --cpu 4 --mem 24G --label $label --max-seconds $mx -- $PY -m rrp.cli latent train-flow --config $cfg ;;
    rz)   ops --gpu --gpu-mem $gm --cpu 3 --mem 20G --label $label --max-seconds $mx -- $PY scripts/ladder_refit.py $cfg ;;
  esac
}
collect() {  # name rep seed [flow] [genctx]: 4 parallel one-shot jobs, then verify all 13 buffers
  local name=$1 rep=$2 seed=$3 flow=${4:-} gc=${5:-} out=$RUNS/ladder_dagger_${LIN}_$1 pids=() i=0 ok=0
  for g in "$G1" "$G2" "$G3" "$G4"; do
    i=$((i+1))
    ops --cpu 3 --mem 8G --label a${TG}_col_${name}_$i --max-seconds 10800 -- env EXPERT=bc ${flow:+FLOW=$flow} ${gc:+GENCTX=1} SEED=$seed \
      bash scripts/ladder_dagger_collect.sh $rep $out 24 $g &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || ok=1; done
  for r in $BODIES; do
    [ -f $out/$r.npz ] || { echo "missing $out/$r.npz"; ok=1; }
    [ -n "$gc" ] && { [ -f $out/$r.npz.genctx.pkl ] || { echo "missing $out/$r.npz.genctx.pkl"; ok=1; }; }
  done
  return $ok
}
r2eval() {  # tag flow rep seedstart robots...: R2 evaluations (CPU)
  local tag=$1 flow=$2 rep=$3 s=$4; shift 4
  local rs="$*"
  ops --cpu 3 --mem 8G --label a${TG}_r2_${tag}_$s --max-seconds 10800 -- bash -c "export CUDA_VISIBLE_DEVICES=; set -e; sha256sum $flow $rep; for r in $rs; do $PY scripts/ladder.py --route generated --flow $flow --rep $rep --prev-action zero --robot \$r --n 30 --seed-start $s --tag zero_${tag}_s$s --out artifacts/runs/ladder_v1/\$r; done"
}
orcbc() {  # rep tag: stateless R1 oracle diagnostic
  ops --cpu 3 --mem 10G --label a${TG}_orcbc_$2 --max-seconds 10800 -- bash -c "set -e; bash scripts/ladder_eval_orcbc.sh $1 $2; for r in panda_pg2 parm6_tf3; do test -f artifacts/runs/ladder_v1/\$r/oracle_zero_$2_orcbc.summary.json; done"
}
semedit() {  # robot seedstart episodes shardname: task-context semantic-edit suite (D-074/075/077 command)
  ops --cpu 1 --mem 2G --label a${TG}_sem_${4//\//_} --max-seconds 14400 -- env OMP_NUM_THREADS=1 $PY -m rrp.cli latent semantic-edits --route generated \
    --checkpoint $F0 --representation $(rz gendag1_noqd) --robots $1 --episodes $3 --seed-start $2 --max-steps 400 \
    --conditions control,goal_shift,rebind_desc,irrelevant_distractor,orthogonal_matched,control_replay \
    --out artifacts/runs/acceptance_arm${LIN}_gen_$4
}
semedits_all() {
  local pids=() ok=0
  for i in 0 1 2 3 4 5; do semedit parm6_tf3 $((3000000 + 10*i)) 10 parm6/shard$i & pids+=($!); done   # wave 1 (9 CPU)
  for i in 0 1 2; do semedit panda_pg2 $((3000000 + 8*i)) 8 panda/shard$i & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || ok=1; done; pids=()
  for i in 6 7 8 9 10 11; do semedit parm6_tf3 $((3000000 + 10*i)) 10 parm6/shard$i & pids+=($!); done  # wave 2
  for i in 3 4 5; do semedit panda_pg2 $((3000000 + 8*i)) 8 panda/shard$i & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || ok=1; done
  return $ok
}
final_evals() {
  local pids=() ok=0 T=flow${TG}gdag2h_rz${TG}gendag3_noqd
  for s in 3000000 3000100 3000200; do r2eval $T $FG2H $(rz gendag3_noqd) $s parm6_tf3 panda_pg2 & pids+=($!); done
  r2eval $T $FG2H $(rz gendag3_noqd) 3000000 parm5s_tf3 parm5l_pg2 & pids+=($!)
  for p in "${pids[@]}"; do wait $p || ok=1; done
  return $ok
}
prog_evals() {  # progression rows on the dev seeds (same pairs as the sem progression table)
  local pids=() ok=0
  r2eval flow${TG}20k_rz${TG}gendag1_noqd $F0 $(rz gendag1_noqd) 3000000 parm6_tf3 panda_pg2 & pids+=($!)
  r2eval flow${TG}gdag1_rz${TG}gendag3_noqd $FG1 $(rz gendag3_noqd) 3000000 parm6_tf3 panda_pg2 & pids+=($!)
  orcbc $(rz gendag3_noqd) ${LIN}gendag3noqd & pids+=($!)
  for p in "${pids[@]}"; do wait $p || ok=1; done
  return $ok
}

flow0() {  # as the sem watcher did: snap_final_s20000.pt = copy of the final policy.pt (warm-start path in the ft/gdag1 configs)
  train_gpu a${TG}_flow0 $C/flow_${LIN}.json flow 16G 14400 && cp $RUNS/ladder_flow_${LIN}/policy.pt $F0
}

# ---- DAG ----
all_nodes() {
  node stageA - train_gpu a${TG}_stageA $C/$REPCFG rep 16G 14400 &
  node F0 stageA flow0 &
  node Fft F0 train_gpu a${TG}_flowft $C/flow_${LIN}_ft.json flow 16G 10800 &
  node bc1 stageA collect bc1 $REP0 3200000 &
  node rzbcdag1 bc1 train_gpu a${TG}_rz_bcdag1 $C/rz_${LIN}_bcdag1.json rz 8G 7200 &
  node rzbcdag1long bc1 train_gpu a${TG}_rz_bcdag1long $C/rz_${LIN}_bcdag1_long.json rz 8G 10800 &
  node bc2 rzbcdag1 collect bc2 $(rz bcdag1) 3300000 &
  node bc3 rzbcdag1long collect bc3 $(rz bcdag1_long) 3400000 &
  node gen1 rzbcdag1long,F0 collect gen1 $(rz bcdag1_long) 3500000 $F0 &
  node rzbcdag2 rzbcdag1long,bc2 train_gpu a${TG}_rz_bcdag2 $C/rz_${LIN}_bcdag2.json rz 8G 10800 &
  node rzgendag1 rzbcdag2,gen1,bc3 train_gpu a${TG}_rz_gendag1 $C/rz_${LIN}_gendag1_noqd.json rz 8G 7200 &
  node semedits rzgendag1,F0 semedits_all &
  node gen2 rzgendag1,F0 collect gen2 $(rz gendag1_noqd) 3700000 $F0 &
  node rzgendag2 gen2 train_gpu a${TG}_rz_gendag2 $C/rz_${LIN}_gendag2_noqd.json rz 8G 7200 &
  node gen3 rzgendag2,Fft collect gen3 $(rz gendag2_noqd) 3800000 $FFT &
  node gdag1 rzgendag2,Fft collect gdag1 $(rz gendag2_noqd) 3900000 $FFT 1 &
  node rzgendag3 gen3 train_gpu a${TG}_rz_gendag3 $C/rz_${LIN}_gendag3_noqd.json rz 8G 7200 &
  node Fgdag1 gdag1,F0 train_gpu a${TG}_flowgdag1 $C/flow_${LIN}_gdag1.json flow 16G 7200 &
  node gdag2 rzgendag3,Fgdag1 collect gdag2 $(rz gendag3_noqd) 4000000 $FG1 1 &
  node Fgdag2h gdag2,Fgdag1 train_gpu a${TG}_flowgdag2h $C/flow_${LIN}_gdag2h.json flow 16G 7200 &
  node finalevals Fgdag2h,rzgendag3 final_evals &
  node progevals rzgendag3,Fgdag1,rzgendag1 prog_evals &
  wait
  log chain "all nodes returned"
}
if [ $# -eq 0 ]; then all_nodes; else for n in "$@"; do "$n"; done; fi
