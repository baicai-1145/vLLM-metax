#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/vLLM-metax
SRC_DIR="${1:-/root/tilelang-metax}"
PREFIX_DIR="${TL_PREFIX:-/tmp/tilelang-metax-prefix}"
PYVER="$(/root/vLLM-metax/.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PREFIX_SITE="$PREFIX_DIR/lib/python${PYVER}/site-packages"
for path in \
  "$SRC_DIR/CMakeLists.txt" \
  "$SRC_DIR/3rdparty/tvm/include/tvm" \
  "$SRC_DIR/3rdparty/cutlass/include/cutlass" \
  "$SRC_DIR/3rdparty/composable_kernel/include/ck"
do
  if [[ ! -e "$path" ]]; then
    echo "TileLang-MetaX source tree is incomplete: missing $path" >&2
    exit 2
  fi
done
source "$ROOT/.venv/bin/activate"
source "$ROOT/env.sh"
TORCH_LIB="$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")"
export LD_LIBRARY_PATH="${TORCH_LIB}:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"
export PATH="/opt/maca/mxgpu_llvm/bin:${PATH}"
mkdir -p "$PREFIX_DIR"
python -m pip install --prefix "$PREFIX_DIR" \
  'cython>=3.1' \
  scikit-build-core \
  'patchelf>=0.17.2' \
  'z3-solver==4.15.4.0'
PYTHONPATH="$PREFIX_SITE:${PYTHONPATH:-}" \
  CMAKE_ARGS='-DUSE_MACA=ON -DUSE_CUDA=OFF -DUSE_ROCM=OFF' \
  python -m pip install --prefix "$PREFIX_DIR" --no-build-isolation --no-deps "$SRC_DIR"
echo "TileLang-MetaX installed into $PREFIX_DIR"
echo "Export PYTHONPATH=$PREFIX_SITE:\$PYTHONPATH before smoke tests"
