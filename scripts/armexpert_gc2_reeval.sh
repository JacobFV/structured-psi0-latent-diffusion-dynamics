#!/usr/bin/env bash
# D-110 (4): re-evaluate the EXISTING arm routes under grasp_v2 with the recorded commands (scripts/arm_lineage_chain.sh
# r2eval: ladder.py --route generated --prev-action zero --n 30) and only RRP_GRASP_CONTACT changed. BC direct1701 final
# (learned route, as sprint_bc) is run under BOTH grasp versions (its v1 rows exist only for the dev set).
# One small host lease per (route, body, seed set); a finished summary is never redone; <= PASSES passes (host sheds).
# Usage: GC=v2 PAR=3 [PLACE=peer] scripts/armexpert_gc2_reeval.sh   (D-115: heavy compute on the peer only)
set -uo pipefail
cd "$(dirname "$0")/.."
GC=${GC:-v2}; PAR=${PAR:-3}; PASSES=${PASSES:-6}
PY=$HOME/work/relational-robot-policy/.venv/bin/python
O=artifacts/runs/armexpert_gc2eval/grasp_$GC
R=artifacts/runs
declare -a JOBS=()
for L in "jointfix|" "sfjf|sf" "nsjf|ns" "sejf2|se2" "sfjf2|sf2" "nsjf2|ns2"; do
  lin=${L%|*}; tg=${L#*|}; tag=flow${tg}gdag2h_rz${tg}gendag3_noqd
  for s in 3000000 3000100 3000200; do for r in panda_pg2 parm6_tf3; do JOBS+=("gen|$lin|$tag|$r|$s"); done; done
  for r in parm5s_tf3 parm5l_pg2; do JOBS+=("gen|$lin|$tag|$r|3000000"); done
done
for s in 3000000 3000100 3000200; do for r in panda_pg2 parm6_tf3; do JOBS+=("bc|direct1701|bc_direct1701_final|$r|$s"); done; done
for r in parm5s_tf3 parm5l_pg2; do JOBS+=("bc|direct1701|bc_direct1701_final|$r|3000000"); done
if [ -n "${KIND:-}" ]; then f=(); for j in "${JOBS[@]}"; do [ "${j%%|*}" = "$KIND" ] && f+=("$j"); done; JOBS=("${f[@]}"); fi
PLACE=${PLACE:-peer}
export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/armexpert
PR=$RRP_PEER_REPO
exists() { if [ $PLACE = peer ]; then ssh gb10-direct "test -s $PR/$1"; else test -s "$1"; fi; }
admit() { [ $PLACE = peer ] && return 0; for _ in $(seq 1 240); do PYTHONPATH=src $PY -m rrp.cli ops status 2>/dev/null | grep -q '"admission_stopped": false' && return 0; sleep 10; done; return 1; }
summ() { local kind=$1 tag=$3 r=$4 s=$5; [ $kind = gen ] && echo $O/$r/generated_zero_${tag}_s$s.summary.json || echo $O/$r/learned_${tag}_s$s.summary.json; }
for pass in $(seq 1 $PASSES); do
  have=$(if [ $PLACE = peer ]; then ssh gb10-direct "cd $PR && find $O -name '*.summary.json' -size +0" 2>/dev/null; else find $O -name '*.summary.json' -size +0 2>/dev/null; fi)
  todo=(); for j in "${JOBS[@]}"; do IFS='|' read -r k l t r s <<< "$j"; grep -qxF "$(summ $k $l $t $r $s)" <<< "$have" || todo+=("$j"); done
  [ ${#todo[@]} -eq 0 ] && { echo "all done after pass $((pass-1))"; exit 0; }
  echo "$(date +%T) pass $pass: ${#todo[@]} jobs"
  i=0
  for j in "${todo[@]}"; do
    IFS='|' read -r k lin tag r s <<< "$j"
    [ $PLACE = peer ] && ssh gb10-direct "mkdir -p $PR/$O/$r" || mkdir -p $O/$r
    if [ $k = gen ]; then
      A=(scripts/ladder.py --route generated --flow $R/ladder_flow_${lin}_gdag2h/policy.pt --rep $R/ladder_rz_${lin}_gendag3_noqd/representation.pt
         --prev-action zero --robot $r --n 30 --seed-start $s --tag zero_${tag}_s$s --out $O/$r)
    else
      A=(scripts/ladder.py --route learned --policy $R/latent_slice1_b1fix/baseline_direct_action/seed1701/source/policy.pt
         --policy-label learned:direct1701_final --robot $r --n 30 --seed-start $s --tag ${tag}_s$s --out $O/$r)
    fi
    admit || { echo "admission stopped for 40 min"; exit 2; }
    mkdir -p $O/$r
    if [ $PLACE = peer ]; then
      ( for _try in $(seq 1 90); do     # wait (<= 90 min) for peer admission (memory-capped broker, D-106)
          scripts/peer_run.sh --cpu 2 --mem 2G --label axgc2e_${GC}_${tag:0:20}_${r}_$s --max-seconds 7200 -- \
            env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 RRP_GRASP_CONTACT=$GC PY "${A[@]}" > $O/$r/${tag}_s$s.log 2>&1
          grep -q "AdmissionStopped\|CapacityError" $O/$r/${tag}_s$s.log || break
          sleep 60
        done ) &
    else
      PYTHONPATH=src $PY -m rrp.cli ops run --cpu 2 --mem 2G --label axgc2e_${GC}_${tag:0:20}_${r}_$s --max-seconds 5400 -- \
        env PYTHONPATH=src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 RRP_GRASP_CONTACT=$GC $PY "${A[@]}" > $O/$r/${tag}_s$s.log 2>&1 &
    fi
    i=$((i+1)); [ $((i % PAR)) -eq 0 ] && wait
  done
  wait
done
have=$(if [ $PLACE = peer ]; then ssh gb10-direct "cd $PR && find $O -name '*.summary.json' -size +0"; else find $O -name '*.summary.json' -size +0; fi)
n=0; for j in "${JOBS[@]}"; do IFS='|' read -r k l t r s <<< "$j"; grep -qxF "$(summ $k $l $t $r $s)" <<< "$have" || { echo "INCOMPLETE $j"; n=1; }; done
exit $n
