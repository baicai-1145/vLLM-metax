#!/usr/bin/env bash
set -euo pipefail
source /root/vLLM-metax/.venv/bin/activate
source /root/vLLM-metax/env.sh
TORCH_LIB=$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH=${TORCH_LIB}:${LD_LIBRARY_PATH}
export MODEL=${MODEL:-/root/models/DeepSeek-V4-Flash-W4A16-BF16Attn-MTP}
export TP=${TP:-4}
export GPU_MEM=${GPU_MEM:-0.9}
export ENFORCE_EAGER=${ENFORCE_EAGER:-0}
export CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-FULL}
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}
python /root/vLLM-metax/tools/tmp_deepseek_v4_mtp_generate.py
