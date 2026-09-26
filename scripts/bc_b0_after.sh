#!/usr/bin/env bash
# Sprint BC: once a B-1-fixed seed-1701 baseline source has FINISHED (policy.pt + result.json), run its budget-0
# zero-shot transfer cells (xarm7_pg2, xarm7_tf3, panda_tf3; sealed protocol: 100 episodes, seeds 2,000,000+) on PEER
# CPU. The direct-action source trains on the host, so its source dir is first copied into the peer store (same
# relative path). Source-competence cells are run by the training units themselves (not duplicated here).
# Usage (host, background): scripts/bc_b0_after.sh <baseline_direct_action|baseline_action_only_codec>
set -uo pipefail
cd "$(dirname "$0")/.."
M=${1:?method}
PEER=gb10-direct; P=/dev/shm/rrp-brandonin; PR=$P/wt/baselines
REL=artifacts/runs/latent_slice1_b1fix/$M/seed1701/source
export RRP_PEER_REPO=$PR
if [ $M = baseline_direct_action ] && [ ! -e $HOME/work/relational-robot-policy/$REL/MOVED_TO_PEER.txt ]; then
  H=$HOME/work/relational-robot-policy/$REL
  until [ -f $H/policy.pt ] && [ -f $H/result.json ]; do sleep 300; done
  ssh $PEER "mkdir -p $P/repo/$REL" && rsync -a $H/policy.pt $H/policy.json $H/result.json $H/config.json $PEER:$P/repo/$REL/ 2>/dev/null \
    || rsync -a $H/policy.pt $H/result.json $H/config.json $PEER:$P/repo/$REL/ || exit 1
else
  until ssh $PEER "test -f $P/repo/$REL/policy.pt && test -f $P/repo/$REL/result.json"; do sleep 300; done
fi
echo "$(date) $M source finished; launching b0 cells"
for T in xarm7_pg2 xarm7_tf3 panda_tf3; do
  scripts/peer_run.sh --cpu 3 --mem 8G --label baselines_b0_${M#baseline_}_$T --max-seconds 14400 --detach -- \
    env CUDA_VISIBLE_DEVICES= PY -m rrp.cli campaign baseline-cell --method $M --seed 1701 --target $T --budget 0 \
    --root artifacts/runs/latent_slice1_b1fix
done
