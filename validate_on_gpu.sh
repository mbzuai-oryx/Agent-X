#!/usr/bin/env bash
# Run this INSIDE the srun session on sc3-c128 (8x H200). It validates the parts of the
# Agent-X install that need a real GPU, which cannot be checked from sc-vnc10 (no CUDA there).
#
#   srun --reservation=vllm-stuff --partition=gpuonly --nodelist=sc3-c128 --gres=gpu:1 --pty bash
#   bash tests/agentx_snova/validate_on_gpu.sh
#
# The envs live on the shared NFS mount, built on RHEL 8 (glibc 2.28) and executed here on
# RHEL 10 (glibc 2.39) - old-glibc-on-new-glibc is the compatible direction.

set -uo pipefail
CONDA=/import/snvm-sc-scratch2/rodrigom/miniforge3
export HF_HOME=/import/snvm-sc-scratch2/rodrigom/hf_cache
export XDG_CACHE_HOME=/import/snvm-sc-scratch2/rodrigom/.cache
export PIP_CACHE_DIR=/import/snvm-sc-scratch2/rodrigom/pip_cache

source "$CONDA/etc/profile.d/conda.sh"
conda activate agentlego

echo "=== host ==="
hostname -f; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo
echo "=== 1. does the vnc10-built env actually see the GPU here? ==="
python - <<'PY'
import torch
print("torch                ", torch.__version__)
print("cuda available       ", torch.cuda.is_available())
print("device count         ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0             ", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("compute capability   ", "sm_%d%d" % cap)
    print("arch list in wheel   ", torch.cuda.get_arch_list())
    # a real kernel launch - proves the cu121 wheel has usable kernels for this arch
    a = torch.randn(2048, 2048, device="cuda")
    print("matmul on GPU        ", float((a @ a).sum()) is not None)
PY

echo
echo "=== 2. mmcv CUDA ops (the thing `mim install` would have gotten wrong) ==="
python - <<'PY'
import torch
from mmcv.ops import nms
boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.],[20.,20.,30.,30.]], device="cuda")
scores = torch.tensor([0.9,0.8,0.7], device="cuda")
keep = nms(boxes, scores, 0.5)
print("mmcv.ops.nms on CUDA OK ->", keep[1].tolist())
PY

echo
echo "=== 3. Qwen-VL-Chat load (backs ImageDescription/CountGivenObject/RegionAttributeDescription) ==="
echo "    first run downloads ~20GB into $HF_HOME"
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
REV = 'f57cfbd358cb56b710d963669ad1bcfb44cdcdd8'   # pinned in benchmark.py
tok = AutoTokenizer.from_pretrained('Qwen/Qwen-VL-Chat', trust_remote_code=True, revision=REV)
print("tokenizer OK")
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-VL-Chat', device_map='cuda',
                                         trust_remote_code=True, revision=REV).eval()
print("model on", next(m.parameters()).device, "| dtype", next(m.parameters()).dtype)
PY

echo
echo "=== 4. tool server smoke (AGENTX_SETUP.md step 4, but --device cuda) ==="
echo "Run this in a tmux window and leave it up:"
cat <<'EOF'
  conda activate agentlego
  cd /import/snvm-sc-scratch2/rodrigom/fc-so-testing-suite/tests/agentx_snova/agentlego
  export $(grep -E '^(SERPER|MATHPIX)' ../.env | xargs)
  agentlego-server start --port 16181 --device cuda --no-setup --extra ./benchmark.py \
      Calculator OCR ImageDescription CountGivenObject RegionAttributeDescription \
      TextToBbox DrawBox AddText Plot Solver GoogleSearch MathOCR \
      TextToImage ImageStylization --host 0.0.0.0
EOF
echo
echo "Then, from another window:  curl -s localhost:16181/openapi.json | head -c 300"
