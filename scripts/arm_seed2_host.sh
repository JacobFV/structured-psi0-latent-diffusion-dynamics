#!/usr/bin/env bash
# ARM SEED-2 takeover driver (lead 17:15: peer CPU oversubscribed, host idle). Runs on the HOST, one instance per lineage.
# Takes over the peer driver (scripts/arm_lineage_chain.sh) after its units were stopped at the gen2/semedits stage:
#  - ADOPTS nodes whose jobs were already launched on the peer (flow_ft, gen2 collections, edit-suite wave 1): waits for
#    those exact jobs' exit codes (no relaunch, no duplication);
#  - GPU training of the remaining nodes stays on the PEER GPU (same configs/labels as the peer driver);
#  - DAgger collections, edit-suite wave 2 and all evaluations run on the HOST CPU (checkpoints pulled from / outputs
#    pushed to the peer store, which stays the single artifact store). Same commands, seeds and shards as the peer driver.
# Markers stay in the PEER state dir. One-shot jobs; a failed job marks the node failed (no retry). Host admission is
# waited for (bounded: 3 h) when the broker refuses for capacity; that is not a retry of a started job.
# Usage: LIN=<nsjf2|sfjf2|sejf2> bash scripts/arm_seed2_host.sh
set -uo pipefail
cd ~/work/rrp-wt/ladder
export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/ladder PYTHONPATH=src
HPY=/home/brandonin/work/relational-robot-policy/.venv/bin/python
PPY=/dev/shm/rrp-brandonin/venv/bin/python
PR=/dev/shm/rrp-brandonin/repo/artifacts/runs
PLOGS=/dev/shm/rrp-brandonin/repo/ops/logs
LIN=${LIN:?}
case $LIN in
  nsjf2) C=configs/ladder/armseed2/nsjf2; TG=ns2; REPN=ladder_latent_nosem_b1fix_anchor_s2 ;;
  sfjf2) C=configs/ladder/armseed2/sfjf2; TG=sf2; REPN=ladder_latent_semfix_b1fix_anchor_s2 ;;
  sejf2) C=configs/ladder/armseed2/sejf2; TG=se2; REPN=ladder_latent_sem_b1fix_anchor_s2 ;;
  *) echo "unknown LIN"; exit 2 ;;
esac
SOFF=1100000
PLACE=${PLACE:-host}     # host | peer: where collections, edit shards and evaluations run (GPU training is always on the peer)
RUNS=artifacts/runs; mkdir -p $RUNS
PST=$PR/ladder_arm${LIN}_state
HST=$RUNS/ladder_arm${LIN}_hoststate; mkdir -p $HST
G1="panda_pg2 parm5_pg2 parm5_tf3 parm5l_tf3"; G2="parm5s_pg2 parm6_pg2 parm6_tf3 parm7_pg2"; G3="parm7_tf3 sawyer_pg2 sawyer_tf3"; G4="ur5e_pg2 ur5e_tf3"
BODIES="$G1 $G2 $G3 $G4"
F0=$RUNS/ladder_flow_${LIN}/snap_final_s20000.pt
FFT=$RUNS/ladder_flow_${LIN}_ft/policy.pt
FG1=$RUNS/ladder_flow_${LIN}_gdag1/policy.pt
FG2H=$RUNS/ladder_flow_${LIN}_gdag2h/policy.pt
rz() { echo $RUNS/ladder_rz_${LIN}_$1/representation.pt; }
log() { local m="$(date '+%F %T') [$1] ${*:2} (host takeover)"; echo "$m" >> $HST/chain.log; ssh gb10-direct "echo '$m' >> $PST/chain.log"; }
mark() { ssh gb10-direct "touch $PST/$1.$2"; touch $HST/$1.$2; }
pmarks() { ssh gb10-direct "ls $PST" 2>/dev/null; }
need() {
  local t0=$(date +%s) l
  while true; do
    l=$(pmarks); local ok=1
    for d in "$@"; do
      echo "$l" | grep -qx "$d.failed" && return 1
      echo "$l" | grep -qx "$d.done" || ok=0
    done
    [ $ok = 1 ] && return 0
    [ $(( $(date +%s) - t0 )) -gt 72000 ] && return 1
    sleep 60
  done
}
node() {
  local n=$1 deps=$2; shift 2
  local l; l=$(pmarks)
  echo "$l" | grep -qx "$n.done" && { log $n "already done"; return 0; }
  echo "$l" | grep -qx "$n.failed" && { log $n "marked failed; not retrying"; return 1; }
  if [ "$deps" != - ] && ! need ${deps//,/ }; then log $n "dependency failed/timeout ($deps)"; mark $n failed; return 1; fi
  log $n "start"
  if "$@" >> $HST/$n.out 2>&1; then log $n "done"; mark $n done; else log $n "FAILED"; mark $n failed; return 1; fi
}
pull() { mkdir -p $RUNS/$1 $RUNS/$REPN && rsync -a gb10-direct:$PR/$1/ $RUNS/$1/ && rsync -a gb10-direct:$PR/$REPN/ $RUNS/$REPN/; }
push() { ssh gb10-direct "mkdir -p $PR/$1" && rsync -a $RUNS/$1/ gb10-direct:$PR/$1/; }
hops() {  # host lease; waits (bounded) only while the broker refuses admission for capacity
  local i out rc
  for i in $(seq 1 180); do
    out=$($HPY -m rrp.cli ops run "$@" 2>&1); rc=$?
    echo "$out"
    [ $rc = 0 ] && return 0
    if ! echo "$out" | grep -q '"lease_id"' && echo "$out" | grep -qE "aggregate limit|gpu owners|CapacityError"; then sleep 60; continue; fi
    return $rc
  done
  return 1
}
adopt() {  # wait for already-launched peer jobs (by label) and require rc 0
  local lab f t0=$(date +%s) ok=0
  for lab in "$@"; do
    f=$(ssh gb10-direct "ls -t $PLOGS/*_$lab.log 2>/dev/null | head -1")
    [ -z "$f" ] && { echo "no job log for $lab"; ok=1; continue; }
    until ssh gb10-direct "[ -f ${f%.log}.rc ]"; do
      [ $(( $(date +%s) - t0 )) -gt 36000 ] && { echo "timeout $lab"; return 1; }
      sleep 60
    done
    rc=$(ssh gb10-direct "cat ${f%.log}.rc"); echo "$lab rc=$rc ($f)"; [ "$rc" = 0 ] || ok=1
  done
  return $ok
}
peer_has_bufs() {  # name [genctx]
  local d=$PR/ladder_dagger_${LIN}_$1 n
  n=$(ssh gb10-direct "ls $d/*.npz 2>/dev/null | wc -l"); [ "$n" -ge 13 ] || { echo "only $n buffers in $d"; return 1; }
  [ -z "${2:-}" ] || { n=$(ssh gb10-direct "ls $d/*.genctx.pkl 2>/dev/null | wc -l"); [ "$n" -ge 13 ] || { echo "only $n genctx"; return 1; }; }
}

# ---- adopted in-flight nodes ----
adopt_Fft() { adopt a${TG}_flowft && ssh gb10-direct "test -f $PR/ladder_flow_${LIN}_ft/policy.pt"; }
adopt_gen2() { adopt a${TG}_col_gen2_1 a${TG}_col_gen2_2 a${TG}_col_gen2_3 a${TG}_col_gen2_4 && peer_has_bufs gen2; }
semedit_host() {  # robot seedstart episodes shard
  hops --cpu 1 --mem 2G --label a${TG}_sem_${4//\//_} --max-seconds 14400 -- env OMP_NUM_THREADS=1 $HPY -m rrp.cli latent semantic-edits --route generated \
    --checkpoint $F0 --representation $(rz gendag1_noqd) --robots $1 --episodes $3 --seed-start $2 --max-steps 400 \
    --conditions control,goal_shift,rebind_desc,irrelevant_distractor,orthogonal_matched,control_replay \
    --out $RUNS/acceptance_arm${LIN}_gen_$4
}
adopt_semedits() {  # wave 1 already on the peer; wave 2 (never launched) on the host
  local ok=0 pids=() w1=""
  for i in 0 1 2 3 4 5; do w1="$w1 a${TG}_sem_parm6_shard$i"; done
  for i in 0 1 2; do w1="$w1 a${TG}_sem_panda_shard$i"; done
  pull ladder_flow_${LIN} && pull ladder_rz_${LIN}_gendag1_noqd || return 1
  for i in 6 7 8 9 10 11; do semedit_host parm6_tf3 $((3000000 + 10*i)) 10 parm6/shard$i & pids+=($!); done
  for i in 3 4 5; do semedit_host panda_pg2 $((3000000 + 8*i)) 8 panda/shard$i & pids+=($!); done
  adopt $w1 || ok=1
  for p in "${pids[@]}"; do wait $p || ok=1; done
  for b in parm6 panda; do push acceptance_arm${LIN}_gen_$b || ok=1; done
  return $ok
}
# ---- PLACE=peer variants (17:30: host jobs were shed by the host memory-PSI watchdog; other projects hold ~60 GB) ----
semedit_peer() {
  scripts/peer_run.sh --cpu 1 --mem 2G --label a${TG}_sem_${4//\//_} --max-seconds 14400 -- env OMP_NUM_THREADS=1 PY -m rrp.cli latent semantic-edits --route generated \
    --checkpoint $F0 --representation $(rz gendag1_noqd) --robots $1 --episodes $3 --seed-start $2 --max-steps 400 \
    --conditions control,goal_shift,rebind_desc,irrelevant_distractor,orthogonal_matched,control_replay \
    --out artifacts/runs/acceptance_arm${LIN}_gen_$4
}
collect_peer() {  # name repdir seed flowdir flowfile [genctx]
  local name=$1 repd=$2 seed=$3 fd=$4 ff=$5 gc=${6:-} d=ladder_dagger_${LIN}_$1 pids=() i=0 ok=0
  for g in "$G1" "$G2" "$G3" "$G4"; do
    i=$((i+1))
    scripts/peer_run.sh --cpu 3 --mem 8G --label a${TG}_col_${name}_$i --max-seconds 14400 -- env EXPERT=bc FLOW=$RUNS/$fd/$ff ${gc:+GENCTX=1} SEED=$seed \
      bash scripts/ladder_dagger_collect.sh $RUNS/$repd/representation.pt $RUNS/$d 24 $g &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || ok=1; done
  peer_has_bufs $name $gc || ok=1
  return $ok
}
r2peer() {  # tag flowdir flowfile repdir seedstart robots...
  local tag=$1 fd=$2 ff=$3 repd=$4 s=$5; shift 5
  local rs="$*"
  scripts/peer_run.sh --cpu 3 --mem 8G --label a${TG}_r2_${tag}_$s --max-seconds 14400 -- bash -c "export CUDA_VISIBLE_DEVICES=; set -e; sha256sum $RUNS/$fd/$ff $RUNS/$repd/representation.pt; for r in $rs; do $PPY scripts/ladder.py --route generated --flow $RUNS/$fd/$ff --rep $RUNS/$repd/representation.pt --prev-action zero --robot \$r --n 30 --seed-start $s --tag zero_${tag}_s$s --out $RUNS/ladder_v1/\$r; done"
}
orcbc_peer() {  # repdir tag
  scripts/peer_run.sh --cpu 3 --mem 10G --label a${TG}_orcbc_$2 --max-seconds 14400 -- bash -c "set -e; bash scripts/ladder_eval_orcbc.sh $RUNS/$1/representation.pt $2; for r in panda_pg2 parm6_tf3; do test -f artifacts/runs/ladder_v1/\$r/oracle_zero_$2_orcbc.summary.json; done"
}

# ---- new nodes ----
train_peer() {  # label cfg kind
  case $3 in
    flow) scripts/peer_run.sh --gpu --gpu-mem 16G --cpu 4 --mem 24G --label $1 --max-seconds 14400 -- PY -m rrp.cli latent train-flow --config $2 ;;
    rz)   scripts/peer_run.sh --gpu --gpu-mem 8G --cpu 3 --mem 20G --label $1 --max-seconds 7200 -- PY scripts/ladder_refit.py $2 ;;
  esac
}
collect_host() {  # name repdir seed flowdir flowfile [genctx]
  local name=$1 repd=$2 seed=$3 fd=$4 ff=$5 gc=${6:-} d=ladder_dagger_${LIN}_$1 pids=() i=0 ok=0
  pull $repd && pull $fd || return 1
  for g in "$G1" "$G2" "$G3" "$G4"; do
    i=$((i+1))
    hops --cpu 2 --mem 4G --label a${TG}_col_${name}_$i --max-seconds 14400 -- env EXPERT=bc FLOW=$RUNS/$fd/$ff ${gc:+GENCTX=1} SEED=$seed PY=$HPY \
      bash scripts/ladder_dagger_collect.sh $RUNS/$repd/representation.pt $RUNS/$d 24 $g &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || ok=1; done
  for r in $BODIES; do
    [ -f $RUNS/$d/$r.npz ] || { echo "missing $d/$r.npz"; ok=1; }
    [ -n "$gc" ] && { [ -f $RUNS/$d/$r.npz.genctx.pkl ] || { echo "missing genctx $r"; ok=1; }; }
  done
  [ $ok = 0 ] && push $d
}
r2host() {  # tag flowdir flowfile repdir seedstart robots...
  local tag=$1 fd=$2 ff=$3 repd=$4 s=$5; shift 5
  local rs="$*"
  pull $fd && pull $repd || return 1
  hops --cpu 2 --mem 4G --label a${TG}_r2_${tag}_$s --max-seconds 14400 -- bash -c "export CUDA_VISIBLE_DEVICES=; set -e; sha256sum $RUNS/$fd/$ff $RUNS/$repd/representation.pt; for r in $rs; do $HPY scripts/ladder.py --route generated --flow $RUNS/$fd/$ff --rep $RUNS/$repd/representation.pt --prev-action zero --robot \$r --n 30 --seed-start $s --tag zero_${tag}_s$s --out $RUNS/ladder_v1/\$r; done" || return 1
  for r in $rs; do ssh gb10-direct "mkdir -p $PR/ladder_v1/$r" && rsync -a $RUNS/ladder_v1/$r/generated_zero_${tag}_s$s.* gb10-direct:$PR/ladder_v1/$r/ || return 1; done
}
orcbc_host() {  # repdir tag
  pull $1 || return 1
  hops --cpu 2 --mem 4G --label a${TG}_orcbc_$2 --max-seconds 14400 -- bash -c "set -e; export CUDA_VISIBLE_DEVICES=; for r in panda_pg2 parm6_tf3; do $HPY scripts/ladder.py --route oracle --oracle-expert bc --policy artifacts/runs/baselines_bc_ckpts/direct1701_u12000.pt --policy-label direct1701_u12000 --tag zero_$2_orcbc --prev-action zero --robot \$r --n 30 --rep $RUNS/$1/representation.pt --out $RUNS/ladder_v1/\$r; done" || return 1
  for r in panda_pg2 parm6_tf3; do rsync -a $RUNS/ladder_v1/$r/oracle_zero_$2_orcbc.* gb10-direct:$PR/ladder_v1/$r/ || return 1; done
}
final_evals() {
  local pids=() ok=0 T=flow${TG}gdag2h_rz${TG}gendag3_noqd
  for s in 3000000 3000100 3000200; do r2host $T ladder_flow_${LIN}_gdag2h policy.pt ladder_rz_${LIN}_gendag3_noqd $s parm6_tf3 panda_pg2 & pids+=($!); done
  r2host $T ladder_flow_${LIN}_gdag2h policy.pt ladder_rz_${LIN}_gendag3_noqd 3000000 parm5s_tf3 parm5l_pg2 & pids+=($!)
  for p in "${pids[@]}"; do wait $p || ok=1; done
  return $ok
}
prog_evals() {
  local pids=() ok=0
  r2host flow${TG}20k_rz${TG}gendag1_noqd ladder_flow_${LIN} snap_final_s20000.pt ladder_rz_${LIN}_gendag1_noqd 3000000 parm6_tf3 panda_pg2 & pids+=($!)
  r2host flow${TG}gdag1_rz${TG}gendag3_noqd ladder_flow_${LIN}_gdag1 policy.pt ladder_rz_${LIN}_gendag3_noqd 3000000 parm6_tf3 panda_pg2 & pids+=($!)
  orcbc_host ladder_rz_${LIN}_gendag3_noqd ${LIN}gendag3noqd & pids+=($!)
  for p in "${pids[@]}"; do wait $p || ok=1; done
  return $ok
}

if [ "$PLACE" = peer ]; then
  semedit_host() { semedit_peer "$@"; }
  collect_host() { collect_peer "$@"; }
  r2host() { r2peer "$@"; }
  orcbc_host() { orcbc_peer "$@"; }
fi
semedits_fill() {  # run ONLY shards without a completed summary (partial rows deleted first); one launch per shard
  local ok=0 pids=() spec b st n e dir
  for spec in $(for i in $(seq 0 11); do echo parm6:$i:$((3000000 + 10*i)):10; done; for i in $(seq 0 5); do echo panda:$i:$((3000000 + 8*i)):8; done); do
    IFS=: read b i st n <<< "$spec"; dir=acceptance_arm${LIN}_gen_$b/shard$i
    ssh gb10-direct "test -f $PR/$dir/semantic_summary_generated.json" && continue
    echo "rerun $dir"; ssh gb10-direct "rm -rf $PR/$dir"; rm -rf $RUNS/$dir
    e=parm6_tf3; [ $b = panda ] && e=panda_pg2
    semedit_host $e $st $n $b/shard$i & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || ok=1; done
  [ "$PLACE" = host ] && for b in parm6 panda; do push acceptance_arm${LIN}_gen_$b || ok=1; done
  for b in parm6:12 panda:6; do n=$(ssh gb10-direct "ls $PR/acceptance_arm${LIN}_gen_${b%%:*}/shard*/semantic_summary_generated.json 2>/dev/null | wc -l"); [ $n -ge ${b##*:} ] || { echo "only $n complete ${b%%:*} shards"; ok=1; }; done
  return $ok
}

all_nodes() {
  node Fft F0 adopt_Fft &
  node gen2 rzgendag1,F0 adopt_gen2 &
  node semedits rzgendag1,F0 semedits_fill &
  node rzgendag2 gen2 train_peer a${TG}_rz_gendag2 $C/rz_${LIN}_gendag2_noqd.json rz &
  node gen3 rzgendag2,Fft collect_host gen3 ladder_rz_${LIN}_gendag2_noqd $((3800000 + SOFF)) ladder_flow_${LIN}_ft policy.pt &
  node gdag1 rzgendag2,Fft collect_host gdag1 ladder_rz_${LIN}_gendag2_noqd $((3900000 + SOFF)) ladder_flow_${LIN}_ft policy.pt 1 &
  node rzgendag3 gen3 train_peer a${TG}_rz_gendag3 $C/rz_${LIN}_gendag3_noqd.json rz &
  node Fgdag1 gdag1,F0 train_peer a${TG}_flowgdag1 $C/flow_${LIN}_gdag1.json flow &
  node gdag2 rzgendag3,Fgdag1 collect_host gdag2 ladder_rz_${LIN}_gendag3_noqd $((4000000 + SOFF)) ladder_flow_${LIN}_gdag1 policy.pt 1 &
  node Fgdag2h gdag2,Fgdag1 train_peer a${TG}_flowgdag2h $C/flow_${LIN}_gdag2h.json flow &
  node finalevals Fgdag2h,rzgendag3 final_evals &
  node progevals rzgendag3,Fgdag1,rzgendag1 prog_evals &
  wait
  log chain "all nodes returned"
}
all_nodes
