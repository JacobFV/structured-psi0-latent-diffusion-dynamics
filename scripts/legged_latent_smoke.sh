set -e
PY=/dev/shm/rrp-brandonin/venv/bin/python
D=/dev/shm/rrp-brandonin/lv_smoke2
[ -f $D/go2/s0-39.npz ] || $PY -m rrp.data.legged_latent_collect --body go2 --seeds 0-39 --out $D > /dev/null
python3 - <<'P'
import json
c=json.load(open('configs/legged_latent/rep_sem_v1.json')); c.update(data='/dev/shm/rrp-brandonin/lv_smoke2', bodies=['go2'], steps=300)
json.dump(c,open('/dev/shm/rrp-brandonin/lv_smoke2/rep.json','w'))
f=json.load(open('configs/legged_latent/flow_sem_v1.json')); f.update(representation='/dev/shm/rrp-brandonin/lv_smoke2/rep/representation.pt', steps=300)
json.dump(f,open('/dev/shm/rrp-brandonin/lv_smoke2/flow.json','w'))
json.dump(dict(representation='/dev/shm/rrp-brandonin/lv_smoke2/rep/representation.pt', steps=200),open('/dev/shm/rrp-brandonin/lv_smoke2/probe.json','w'))
P
$PY -m rrp.learning.legged_latent_train rep --config $D/rep.json --out $D/rep | tail -30
$PY -m rrp.learning.legged_latent_train probe --config $D/probe.json --out $D/rep | tail -5
$PY -m rrp.learning.legged_latent_train flow --config $D/flow.json --out $D/flow | tail -30
$PY -m rrp.evaluation.legged_latent_eval --flow $D/flow/policy.pt --bodies go2 --seeds 10000-10001 --max-s 6 --out $D/eval.jsonl --video-dir $D/video --video-n 1 | tail -20
$PY -m rrp.evaluation.legged_latent_eval --flow $D/flow/policy.pt --bodies go2 --seeds 10000 --max-s 4 --edit probe_yaw:0.5 --out $D/eval_edit.jsonl | tail -3
$PY -m rrp.evaluation.legged_latent_eval --bodies go2 --seeds 10000-10001 --out $D/eval_teacher.jsonl | tail -8
