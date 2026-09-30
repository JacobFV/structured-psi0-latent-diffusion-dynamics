#!/usr/bin/env bash
# Ψ₀ / SIMPLE third-party stack under $RRP_PSI0_EXT (default ~/work/ext), never inside this repo (W10, D-140; replaces
# psi1z install_{simple_env,simple_deps,psi_env}.sh, upgrade_torch_psi.sh, build_cyclonedds.sh, download_*.sh).
# Upstream clones stay pristine: nothing is pip-installed from them; their src dirs go on sys.path via .pth files.
#
#   psi0_ext.sh clone                  Ψ₀ @ 4f3720d with SIMPLE @ 803db7e (https submodules, Git LFS skipped)
#   psi0_ext.sh simple-env             venvs/simple: py3.11, torch 2.7 cu128, isaacsim 5.1.0 (first aarch64 wheel), SIMPLE
#                                      deps, rrp (no deps) + pydantic, and the .pth line that installs rrp.envs.simple.compat
#   psi0_ext.sh cyclonedds             cyclonedds 0.10.5 from source (no aarch64 wheel; unitree_sdk2py needs it)
#   psi0_ext.sh psi-env                venvs/psi: py3.11, Ψ₀ deps, torch 2.14 cu130 (sm_121 JIT), rrp[psi0] editable
#   psi0_ext.sh fetch-base             public base VLM (ego200k.he30k) + post-trained action header
#   psi0_ext.sh fetch-ckpt TASK...     released SIMPLE checkpoints (6.25 GB each) + run configs, by SIMPLE task name
#   psi0_ext.sh fetch-data TASK...     SIMPLE training data (LeRobot) and eval configs (dr-level-0..2)
# Declare downloads through the broker (`rrp ops run --disk ...`, D-036). Isaac Sim licence accepted by the owner (D-098).
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1
EXT=${RRP_PSI0_EXT:-$HOME/work/ext}
PH=${PSI_HOME:-$EXT/psi_home}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
HFM=https://huggingface.co/USC-PSI-Lab/psi-model/resolve/main
HFD=https://huggingface.co/datasets/USC-PSI-Lab/psi-data/resolve/main
LEROBOT="lerobot @ git+https://github.com/songlin/lerobot.git@09929d8057b044b53aecaf5c6d7eb71f99e8beb9"
get() { mkdir -p "$(dirname "$2")"; curl -sSL --retry 5 -C - -o "$2" "$1"; }
run_of() { PYTHONPATH=$REPO/src python3 -c "from rrp.tasks.spec import SIMPLE_TASKS; print(SIMPLE_TASKS['$1'][0])"; }

cmd=${1:?usage: psi0_ext.sh clone|simple-env|cyclonedds|psi-env|fetch-base|fetch-ckpt|fetch-data ...}; shift
case $cmd in
clone)
  [ -d $EXT/psi0 ] || git clone -q https://github.com/physical-superintelligence-lab/Psi0 $EXT/psi0
  git -C $EXT/psi0 checkout -q 4f3720d
  git -C $EXT/psi0 submodule update --init --recursive
  git -C $EXT/psi0/third_party/SIMPLE checkout -q 803db7e ;;
simple-env)
  V=$EXT/venvs/simple; [ -x $V/bin/python ] || uv venv $V --python 3.11
  P=$V/bin/python
  uv pip install --python $P "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python $P "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
  uv pip install --python $P --index-strategy unsafe-best-match \
    "dm-control==1.0.21" "mujoco==3.3.6" "python-fcl>=0.7.0.8" "rich>=13.9.4" "transforms3d>=0.4.2" "trimesh>=4.7.1" \
    "typer-slim>=0.17.3" typer "gymnasium>0.29.1" "python-dotenv>=0.9.9" "numpy==1.26.4" "scipy==1.15.3" "dm-tree>=0.1.8" \
    msgpack msgpack-numpy "websockets>=11.0" pillow tree pyzmq tyro pyyaml loguru termcolor onnxruntime pin \
    opencv-python-headless pandas matplotlib tqdm platformdirs requests huggingface_hub glfw imageio imageio-ffmpeg av \
    "pydantic>=2.7" "$LEROBOT"
  uv pip install --python $P "cyclonedds==0.10.2" || echo "WARN cyclonedds wheel missing: run 'psi0_ext.sh cyclonedds'"
  uv pip install --python $P -e "$REPO" --no-deps
  SP=$($P -c "import site;print(site.getsitepackages()[0])")
  echo 'import os; os.environ.get("RRP_SIMPLE_COMPAT") == "1" and __import__("rrp.envs.simple.compat", fromlist=["x"]).install()' > $SP/rrp_simple_compat.pth
  RRP_SIMPLE_COMPAT=1 $P -c "import simple, mujoco; print('simple', simple.__version__, 'mujoco', mujoco.__version__)"
  echo SIMPLE_ENV_OK ;;
cyclonedds)
  S=$EXT/src/cyclonedds; P=$EXT/cyclonedds
  [ -d $S ] || git clone -q --depth 1 --branch 0.10.5 https://github.com/eclipse-cyclonedds/cyclonedds $S
  cmake -S $S -B $S/build -DCMAKE_INSTALL_PREFIX=$P -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SSL=OFF -DBUILD_IDLC=ON > /dev/null
  cmake --build $S/build -j 4 --target install > /dev/null
  CYCLONEDDS_HOME=$P uv pip install --python $EXT/venvs/simple/bin/python "cyclonedds==0.10.5"
  $EXT/venvs/simple/bin/python -c "import cyclonedds.idl; print('cyclonedds ok')" ;;
psi-env)
  V=$EXT/venvs/psi; [ -x $V/bin/python ] || uv venv $V --python 3.11
  P=$V/bin/python
  # torch 2.7+cu128 (upstream pin) cannot JIT for the GB10 (sm_121): torch 2.14+cu130 (P-002 notes)
  uv pip install --python $P "torch==2.14.0" "torchvision==0.29.0" --index-url https://download.pytorch.org/whl/cu130
  uv pip install --python $P --index-strategy unsafe-best-match \
    "albumentations==1.4.18" "datasets==3.6.0" "dm-tree==0.1.8" "einops==0.8.1" "einx>=0.3.0" "h5py==3.14.0" \
    "imageio==2.34.2" "jsonlines>=4.0.0" "numpy<2.0.0" "numpydantic==1.6.7" "omegaconf>=2.3.0" opencv-python-headless \
    "plotly==6.2.0" "pydantic==2.10.6" "pydantic-yaml>=1.6.0" "tqdm==4.67.1" "typer>=0.19.1" "tyro==0.9.32" \
    "wandb>=0.20.0" "simplejpeg>=1.9.0" dotenv fastapi uvicorn "accelerate==1.7.0" "peft==0.17.1" \
    transforms3d pyarrow matplotlib av "$LEROBOT"
  uv pip install --python $P -e "$REPO[psi0]"
  SP=$($P -c "import site;print(site.getsitepackages()[0])")
  echo "$EXT/psi0/src" > $SP/psi0_upstream_src.pth
  $P -c "import psi, torch, rrp.policies.psi0.nets; print('psi', psi.__version__, 'torch', torch.__version__)"
  echo PSI_ENV_OK ;;
fetch-base)
  V=psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k
  for f in added_tokens.json chat_template.jinja config.json generation_config.json merges.txt preprocessor_config.json \
           special_tokens_map.json tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json model.safetensors; do
    get $HFM/$V/$f $PH/cache/checkpoints/$V/$f; done
  A=psi0/postpre.1by1.pad36.2601131206.ckpt.he30k
  get $HFM/$A/action_header.safetensors $PH/cache/checkpoints/$A/action_header.safetensors ;;
fetch-ckpt)
  for T in "$@"; do R=$(run_of $T); D=$PH/cache/checkpoints/psi0/simple-checkpoints/$R
    for f in argv.txt run_config.json envs.txt; do get $HFM/psi0/simple-checkpoints/$R/$f $D/$f; done
    get $HFM/psi0/simple-checkpoints/$R/checkpoints/ckpt_40000/model.safetensors $D/checkpoints/ckpt_40000/model.safetensors
    echo "done $T"; done ;;
fetch-data)
  for T in "$@"; do
    [ -d $PH/data/simple/$T ] || { get $HFD/simple/$T.zip $PH/data/simple/$T.zip && (cd $PH/data/simple && unzip -qo $T.zip && rm -f $T.zip); }
    [ -d $PH/data/simple-eval/$T ] || { get $HFD/simple-eval/$T.zip $PH/data/simple-eval/$T.zip && (cd $PH/data/simple-eval && unzip -qo $T.zip && rm -f $T.zip); }
    echo "done $T"; done ;;
*) echo "unknown command $cmd"; exit 2 ;;
esac
