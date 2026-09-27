#!/usr/bin/env bash
# Bounded parallel tracker validation (contact track). Each line of $1: "<body> <actor path|v1|v2> <physics v1|v2> <tag>".
# Writes artifacts/runs/contact_v2/val/<body>_<tag>_phys<physics>.json; exits non-zero if any job failed.
set -u
LIST=$1; PAR=${2:-6}; SEEDS=${SEEDS:-5}
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
OUT=artifacts/runs/contact_v2/val; mkdir -p "$OUT"
run_one() {
  set -u
  body=$1; actor=$2; phys=$3; tag=$4
  case $actor in v1) a=artifacts/trackers/$body/actor.pt;; v2) a=artifacts/trackers/$body/contact_v2/actor.pt;; *) a=$actor;; esac
  o=$OUT/${body}_${tag}_phys${phys}.json
  PYTHONPATH=src timeout 3000 "$PY" -m rrp.control.tracker_validation --body "$body" --kind learned --actor "$a" \
     --contact "$phys" --seeds "$SEEDS" --out "$o" > "$o.log" 2>&1
  rc=$?; echo "$body $tag phys$phys rc=$rc"; return $rc
}
export -f run_one; export PY OUT SEEDS
grep -v '^#' "$LIST" | grep -v '^$' | xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _
