#!/usr/bin/env bash
# Rebuild the peer workspace after a reboot (RAM-backed code/venv; large data on the peer disk).
set -euo pipefail
P=/dev/shm/rrp-brandonin
D=$HOME/rrp-peer-data          # persistent disk (user authorized the whole peer, D-008/D-026)
mkdir -p $P/repo $P/bin $P/cache/tmp $D/packed $D/datasets
chmod 700 $P $D
cd $P/repo
export PATH=$P/bin:$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$P/repo UV_CACHE_DIR=$D/uv-cache TMPDIR=$P/cache/tmp
python3 -m rrp.cli ops init --role peer --window 3 >/dev/null
python3 - <<PY
import json
p="configs/resources.local.json"; c=json.load(open(p)); c["peer"]["unrestricted"]=True; c["peer"]["lease_expiry_s"]=120
json.dump(c,open(p,"w"),indent=1)
PY
# whole-project safety ceiling (protects OS/sshd; not a per-job limit): MemoryMax 100G, no swap for the project
systemctl --user start rrp.slice
systemctl --user set-property --runtime rrp.slice CPUQuota= MemoryHigh=infinity MemoryMax=100G MemorySwapMax=0 TasksMax=infinity
python3 -m rrp.cli ops start-watchdog >/dev/null || true
if [ ! -x $P/venv/bin/python ]; then
  uv venv -q --python /usr/bin/python3.12 $P/venv
  uv pip install -q --python $P/venv/bin/python --index-url https://download.pytorch.org/whl/cu130 torch torchvision
  uv pip install -q --python $P/venv/bin/python mujoco numpy pydantic jsonschema pytest pytest-timeout fastapi uvicorn httpx imageio imageio-ffmpeg pillow matplotlib
fi
$P/venv/bin/python -c "import torch,mujoco;print('ok',torch.__version__,torch.cuda.is_available(),mujoco.__version__)"
