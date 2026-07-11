#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/vLLM-metax
PREFIX_DIR="${DG_PREFIX:-/tmp/deep-gemm-metax-prefix}"
EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://repos.metax-tech.com/r/maca-pypi/simple}"
PYVER="$(/root/vLLM-metax/.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PREFIX_SITE="$PREFIX_DIR/lib/python${PYVER}/site-packages"

source "$ROOT/.venv/bin/activate"
source "$ROOT/env.sh"
TORCH_LIB="$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
mkdir -p "$PREFIX_DIR"

python -m pip install --prefix "$PREFIX_DIR" \
  --extra-index-url "$EXTRA_INDEX_URL" \
  'mctlassEx==0.1.1+metax3.7.2.0torch2.8' \
  'deep_gemm==0.0.2+maca3.7.1.103'
PYTHONPATH="$PREFIX_SITE:${PYTHONPATH:-}" python - <<'PY'
import importlib
for name in ('mctlassEx', 'deep_gemm'):
    mod = importlib.import_module(name)
    print(name, getattr(mod, '__file__', None), getattr(mod, '__version__', None))
print('has tf32_hc_prenorm_gemm:', hasattr(importlib.import_module('deep_gemm'), 'tf32_hc_prenorm_gemm'))
PY

echo "DeepGEMM MetaX packages installed into $PREFIX_DIR"
echo "Export PYTHONPATH=$PREFIX_SITE:\$PYTHONPATH before smoke tests"
