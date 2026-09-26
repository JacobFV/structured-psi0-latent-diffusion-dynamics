#!/usr/bin/env bash
# Sprint BC learning curve (host, lightweight loop): every EVERY updates of the B-1-fixed baseline sources, snapshot
# policy_last.pt (sha-verified against policy_last.json), then run the closed-loop ladder `learned` route on PEER CPU:
# panda_pg2 + parm6_tf3, 30 matched dev seeds from 3,000,000 (same as the ladder), failure-stage breakdown.
# Also evaluates the final policy.pt once it appears. Label: learned:<method>_s1701@<step>.
# Outputs (peer shared store): artifacts/runs/baselines_bc_ladder/<robot>/learned_<tag>.{jsonl,summary.json}
# Usage: EVERY=3000 scripts/bc_ckpt_watch.sh
set -uo pipefail
cd "$(dirname "$0")/.."
EVERY=${EVERY:-3000}
PEER=gb10-direct; P=/dev/shm/rrp-brandonin; PR=$P/wt/baselines
HOSTSRC=$HOME/work/relational-robot-policy/artifacts/runs/latent_slice1_b1fix/baseline_direct_action/seed1701/source
PEERSRC=$P/repo/artifacts/runs/latent_slice1_b1fix/baseline_action_only_codec/seed1701/source
SNAP=artifacts/runs/baselines_bc_ckpts
STATE=$HOME/work/rrp-wt/baselines-logs/bc_watch_state; mkdir -p $STATE $SNAP
export RRP_PEER_REPO=$PR

launch() {  # $1 tag  $2 peer ckpt path (relative to PR)
  local tag=$1 ck=$2
  scripts/peer_run.sh --cpu 4 --mem 8G --label baselines_bcl_$tag --max-seconds 7200 --detach -- bash -c \
    "export CUDA_VISIBLE_DEVICES=; for r in panda_pg2 parm6_tf3; do $P/venv/bin/python scripts/ladder.py --route learned --policy $ck --policy-label $tag --robot \$r --n 30 --tag $tag --out artifacts/runs/baselines_bc_ladder/\$r; done" \
    && touch $STATE/$tag.launched
}

while true; do
  # direct-action source on the HOST
  for f in policy_last policy; do
    [ -f $HOSTSRC/$f.pt ] || continue
    if [ $f = policy ]; then step=final; else step=$(python3 -c "import json;print(json.load(open('$HOSTSRC/policy_last.json'))['step'])" 2>/dev/null) || continue; fi
    tag=direct1701_u$step
    [ -e $STATE/$tag.launched ] && continue
    if [ $f = policy_last ] && [ -e $STATE/direct_last ] && [ $((step - $(cat $STATE/direct_last))) -lt $EVERY ]; then continue; fi
    cp $HOSTSRC/$f.pt $SNAP/$tag.pt.tmp || continue
    if [ $f = policy_last ]; then
      want=$(python3 -c "import json;print(json.load(open('$HOSTSRC/policy_last.json'))['sha256_16'])")
      [ "$(sha256sum $SNAP/$tag.pt.tmp | cut -c1-16)" = "$want" ] || { rm -f $SNAP/$tag.pt.tmp; continue; }
    fi
    mv $SNAP/$tag.pt.tmp $SNAP/$tag.pt
    ssh $PEER "mkdir -p $PR/$SNAP" && scp -q $SNAP/$tag.pt $PEER:$PR/$SNAP/$tag.pt || continue
    launch $tag $SNAP/$tag.pt && { [ $f = policy_last ] && echo $step > $STATE/direct_last; echo "$(date) launched $tag"; }
  done
  # codec baseline source on the PEER (for comparison)
  for f in policy_last policy; do
    ssh $PEER "test -f $PEERSRC/$f.pt" || continue
    if [ $f = policy ]; then step=final; else step=$(ssh $PEER "python3 -c \"import json;print(json.load(open('$PEERSRC/policy_last.json'))['step'])\"") || continue; fi
    tag=codec1701_u$step
    [ -e $STATE/$tag.launched ] && continue
    if [ $f = policy_last ] && [ -e $STATE/codec_last ] && [ $((step - $(cat $STATE/codec_last))) -lt $EVERY ]; then continue; fi
    ssh $PEER "mkdir -p $PR/$SNAP && cp $PEERSRC/$f.pt $PR/$SNAP/$tag.pt" || continue
    if [ $f = policy_last ]; then
      ok=$(ssh $PEER "python3 -c \"import json,hashlib;h=hashlib.sha256(open('$PR/$SNAP/$tag.pt','rb').read()).hexdigest()[:16];print(h==json.load(open('$PEERSRC/policy_last.json'))['sha256_16'])\"")
      [ "$ok" = True ] || { ssh $PEER "rm -f $PR/$SNAP/$tag.pt"; continue; }
    fi
    launch $tag $SNAP/$tag.pt && { [ $f = policy_last ] && echo $step > $STATE/codec_last; echo "$(date) launched $tag"; }
  done
  [ -e $STATE/direct1701_ufinal.launched ] && [ -e $STATE/codec1701_ufinal.launched ] && { echo done; exit 0; }
  sleep 300
done
