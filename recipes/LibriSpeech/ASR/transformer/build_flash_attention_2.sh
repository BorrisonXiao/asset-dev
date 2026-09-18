#!/bin/bash
set -euo pipefail

REPO=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm
PY=/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python
UV=/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm/_portability/bin/uv
SDIST="$REPO/artifacts/benchmarks/flash_attn2_build/source/flash_attn-2.7.4.post1.tar.gz"
TARGET="$REPO/artifacts/benchmarks/flash_attn2_env"
PY_HEADERS="$REPO/artifacts/benchmarks/flash_attn2_build/python/cpython-3.11.11-linux-x86_64-gnu/include/python3.11"

module purge
module load gcc/9.3.0
module load cuda/11.8.0
export CUDA_HOME=/apps/software/extern/cuda/11.8.0
export PATH="$TARGET/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="$REPO/artifacts/benchmarks/uv_cache_flash_build"
export CC
CC=$(command -v gcc)
export CXX
CXX=$(command -v g++)
export CPATH="$PY_HEADERS${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$PY_HEADERS${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$PY_HEADERS${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_THREADS=${NVCC_THREADS:-2}

"$UV" pip install \
  --python "$PY" \
  --target "$TARGET" \
  --no-build-isolation \
  --no-deps \
  "$SDIST"

"$PY" - <<'PY'
import flash_attn
import torch

print("flash_attn", flash_attn.__version__)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
PY
