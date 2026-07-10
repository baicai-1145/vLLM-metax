#!/usr/bin/env bash
set -euo pipefail
source /root/vLLM-metax/.venv/bin/activate
source /root/vLLM-metax/env.sh
TORCH_LIB=$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH=${TORCH_LIB}:${LD_LIBRARY_PATH}
export MODEL=/home/waas/models/DeepSeek-V4-Flash-W4A16-BF16Attn-MTP
export TP=4
export GPU_MEM=0.9
python /root/vLLM-metax/tools/tmp_deepseek_v4_mtp_generate.py
