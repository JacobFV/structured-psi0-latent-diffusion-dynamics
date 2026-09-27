#!/usr/bin/env bash
# ARM NOSEM RECIPE ABLATION (lead, after D-095): nosem seed-1 lineage with the system-0 recipe changed.
#   LIN=nszn: z-noise 0 (instead of 0.3); LIN=nsqd: joint velocity kept (no realizer_drop_qd, no qd dropout);
#   LIN=nszq: both. Configs: configs/ladder/armnosemabl/<lin>/.
# z-noise and qd removal enter ONLY the gendag1/2/3 refits, so every upstream stage is config-identical to seed-1 nosem
# (nsjf) and is REUSED from it: Stage A, flow 20k, flow_ft, buffers bc1-3 + gen1, refits bcdag1/bcdag1_long/bcdag2.
# This driver runs everything from gendag1 on (same DAG, seeds, collection seeds, bodies as the nsjf chain):
#   rz gendag1 -> gen2 -> rz gendag2 -> gen3 + gdag1 ctx -> rz gendag3, flow gdag1 -> gdag2 ctx -> flow gdag2h -> evals;
#   edit suite on flow 20k -> rz gendag1. Runs ON THE PEER from its own code dir (PEERWT, default wt/armabl).
# One-shot jobs, exit-code/output checks, .done/.failed markers; no retries (D-061). Resume: rerun (done nodes skipped).
# Usage: LIN=<nszn|nsqd|nszq> bash scripts/arm_nosem_ablation.sh
set -uo pipefail
P=/dev/shm/rrp-brandonin
WT=${PEERWT:-$P/wt/armabl}
cd $WT
export PATH=$P/bin:$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$PWD RRP_OPS_ROOT=$P/repo MUJOCO_GL=egl
PY=$P/venv/bin/python
LIN=${LIN:?}
case $LIN in
  nszn) SUF=_noqd ;;
  nsqd|nszq) SUF=_qd ;;
  *) echo "unknown LIN $LIN"; exit 2 ;;
esac
C=configs/ladder/armnosemabl/$LIN
RUNS=artifacts/runs
ST=$RUNS/ladder_arm${LIN}_state; mkdir -p $ST
G1="panda_pg2 parm5_pg2 parm5_tf3 parm5l_tf3"; G2="parm5s_pg2 parm6_pg2 parm6_tf3 parm7_pg2"; G3="parm7_tf3 sawyer_pg2 sawyer_tf3"; G4="ur5e_pg2 ur5e_tf3"
BODIES="$G1 $G2 $G3 $G4"
F0=$RUNS/ladder_flow_nsjf/snap_final_s20000.pt          # shared with nsjf (config-identical upstream)
FFT=$RUNS/ladder_flow_nsjf_ft/policy.pt
FG1=$RUNS/ladder_flow_${LIN}_gdag1/policy.pt
FG2H=$RUNS/ladder_flow_${LIN}_gdag2h/policy.pt
rz() { echo $RUNS/ladder_rz_${LIN}_$1$SUF/representation.pt; }
log() { echo "$(date '+%F %T') [$1] ${*:2}" | tee -a $ST/chain.log; }
need() {
  local t0=$(date +%s)
  for d in "$@"; do
    until [ -f $ST/$d.done ]; do
      [ -f $ST/$d.failed ] && return 1
      [ $(( $(date +%s) - t0 )) -gt 57600 ] && return 1
      sleep 30
    done
  done
}
ops() { python3 -m rrp.cli ops run "$@"; }
node() {
  local n=$1 deps=$2; shift 2
  [ -f $ST/$n.done ] && { log $n "already done"; return 0; }
  [ -f $ST/$n.failed ] && { log $n "marked failed; not retrying"; return 1; }
  if [ "$deps" != - ] && ! need ${deps//,/ }; then log $n "dependency failed/timeout ($deps)"; touch $ST/$n.failed; return 1; fi
  log $n "start"
  if "$@" >> $ST/$n.out 2>&1; then log $n "done"; touch $ST/$n.done; else log $n "FAILED rc=$?"; touch $ST/$n.failed; return 1; fi
}
train() {  # label cfg kind
  case $3 in
    flow) ops --gpu --gpu-mem 16G --cpu 4 --mem 24G --label $1 --max-seconds 14400 -- $PY -m rrp.cli latent train-flow --config $2 ;;
    rz)   ops --gpu --gpu-mem 8G --cpu 3 --mem 20G --label $1 --max-seconds 7200 -- $PY scripts/ladder_refit.py $2 ;;
  esac
}
collect() {  # name rep seed flow [genctx]
  local name=$1 rep=$2 seed=$3 flow=$4 gc=${5:-} out=$RUNS/ladder_dagger_${LIN}_$1 pids=() i=0 ok=0
  for g in "$G1" "$G2" "$G3" "$G4"; do
    i=$((i+1))
    ops --cpu 3 --mem 16G --label a${LIN}_col_${name}_$i --max-seconds 10800 -- env EXPERT=bc FLOW=$flow ${gc:+GENCTX=1} SEED=$seed \
      bash scripts/ladder_dagger_collect.sh $rep $out 24 $g &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || ok=1; done
  for r in $BODIES; do
    [ -f $out/$r.npz ] || { echo "missing $out/$r.npz"; ok=1; }
    [ -n "$gc" ] && { [ -f $out/$r.npz.genctx.pkl ] || { echo "missing genctx $r"; ok=1; }; }
  done
  return $ok
}
r2eval() {
  local tag=$1 flow=$2 rep=$3 s=$4; shift 4
  local rs="$*"
  ops --cpu 3 --mem 8G --label a${LIN}_r2_${tag}_$s --max-seconds 10800 -- bash -c "export CUDA_VISIBLE_DEVICES=; set -e; sha256sum $flow $rep; for r in $rs; do $PY scripts/ladder.py --route generated --flow $flow --rep $rep --prev-action zero --robot \$r --n 30 --seed-start $s --tag zero_${tag}_s$s --out artifacts/runs/ladder_v1/\$r; done"
}
orcbc() {
  ops --cpu 3 --mem 10G --label a${LIN}_orcbc --max-seconds 10800 -- bash -c "set -e; bash scripts/ladder_eval_orcbc.sh $1 $2; for r in panda_pg2 parm6_tf3; do test -f artifacts/runs/ladder_v1/\$r/oracle_zero_$2_orcbc.summary.json; done"
}
semedit() {
  ops --cpu 1 --mem 2G --label a${LIN}_sem_${4//\//_} --max-seconds 14400 -- env OMP_NUM_THREADS=1 $PY -m rrp.cli latent semantic-edits --route generated \
    --checkpoint $F0 --representation $(rz gendag1) --robots $1 --episodes $3 --seed-start $2 --max-steps 400 \
    --conditions control,goal_shift,rebind_desc,irrelevant_distractor,orthogonal_matched,control_replay \
    --out artifacts/runs/acceptance_arm${LIN}_gen_$4
}
semedits_all() {
  local pids=() ok=0
  for i in 0 1 2 3 4 5; do semedit parm6_tf3 $((3000000 + 10*i)) 10 parm6/shard$i & pids+=($!); done
  for i in 0 1 2; do semedit panda_pg2 $((3000000 + 8*i)) 8 panda/shard$i & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || ok=1; done; pids=()
  for i in 6 7 8 9 10 11; do semedit parm6_tf3 $((3000000 + 10*i)) 10 parm6/shard$i & pids+=($!); done
  for i in 3 4 5; do semedit panda_pg2 $((3000000 + 8*i)) 8 panda/shard$i & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || ok=1; done
  return $ok
}
final_evals() {
  local pids=() ok=0 T=flow${LIN}gdag2h_rz${LIN}gendag3
  for s in 3000000 3000100 3000200; do r2eval $T $FG2H $(rz gendag3) $s parm6_tf3 panda_pg2 & pids+=($!); done
  r2eval $T $FG2H $(rz gendag3) 3000000 parm5s_tf3 parm5l_pg2 & pids+=($!)
  for p in "${pids[@]}"; do wait $p || ok=1; done
  return $ok
}
prog_evals() {
  local pids=() ok=0
  r2eval flow${LIN}20k_rz${LIN}gendag1 $F0 $(rz gendag1) 3000000 parm6_tf3 panda_pg2 & pids+=($!)
  r2eval flow${LIN}gdag1_rz${LIN}gendag3 $FG1 $(rz gendag3) 3000000 parm6_tf3 panda_pg2 & pids+=($!)
  orcbc $(rz gendag3) ${LIN}gendag3 & pids+=($!)
  for p in "${pids[@]}"; do wait $p || ok=1; done
  return $ok
}
all_nodes() {
  node rzgendag1 - train a${LIN}_rz_gendag1 $C/rz_${LIN}_gendag1$SUF.json rz &
  node semedits rzgendag1 semedits_all &
  node gen2 rzgendag1 collect gen2 $(rz gendag1) 3700000 $F0 &
  node rzgendag2 gen2 train a${LIN}_rz_gendag2 $C/rz_${LIN}_gendag2$SUF.json rz &
  node gen3 rzgendag2 collect gen3 $(rz gendag2) 3800000 $FFT &
  node gdag1 rzgendag2 collect gdag1 $(rz gendag2) 3900000 $FFT 1 &
  node rzgendag3 gen3 train a${LIN}_rz_gendag3 $C/rz_${LIN}_gendag3$SUF.json rz &
  node Fgdag1 gdag1 train a${LIN}_flowgdag1 $C/flow_${LIN}_gdag1.json flow &
  node gdag2 rzgendag3,Fgdag1 collect gdag2 $(rz gendag3) 4000000 $FG1 1 &
  node Fgdag2h gdag2,Fgdag1 train a${LIN}_flowgdag2h $C/flow_${LIN}_gdag2h.json flow &
  node finalevals Fgdag2h,rzgendag3 final_evals &
  node progevals rzgendag3,Fgdag1 prog_evals &
  wait
  log chain "all nodes returned"
}
all_nodes
