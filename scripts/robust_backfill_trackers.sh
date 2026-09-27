set -u
cd ~/work/rrp-wt/robust
PY=~/work/relational-robot-policy/.venv/bin/python
O=artifacts/runs/robust/gates/trackers; A=artifacts/runs/robust/gates/actors; mkdir -p $O
for spec in anymal_c:anymal_c_installed: go2:go2_installed: t1:t1_w8d_installed: t1:t1_w8c: g1:g1_src: g1:g1_r1_installed:legacy_gains_v0; do
  IFS=: read body name lim <<< "$spec"
  [ -f $O/$name/gate_report.json ] && continue
  mkdir -p $O/$name
  env ${lim:+RRP_ACTUATOR_LIMITS=$lim} PYTHONPATH=src $PY -m rrp.evaluation.tracker_validation --body $body --contact v2 --actor $A/$name/actor.pt \
     --seeds 5 --robust --out $O/$name/validation.json --gate-dir $O/$name > $O/$name/log.txt 2>&1
  echo "$name rc=$?"
done
