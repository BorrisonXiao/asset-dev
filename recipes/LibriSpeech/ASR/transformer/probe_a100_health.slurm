#!/bin/bash
# Short CUDA health probe; writes logs only and touches no experiment output.
#SBATCH --job-name=a100_health
#SBATCH --time=00:05:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer/slurm_logs/%x_%j.out

set -euo pipefail
echo "node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,temperature.gpu --format=csv,noheader
/home/jhu/jsalt2026-ext-cxiao7/cxiao/envs/jointllm/bin/python - <<'PY'
import torch

assert torch.cuda.is_available()
device = torch.device("cuda")
props = torch.cuda.get_device_properties(device)
print(f"torch={torch.__version__} gpu={props.name} total_gib={props.total_memory / 2**30:.1f}")

# Exercise allocation, BF16 Tensor Cores, fused SDPA, and synchronization.
torch.manual_seed(0)
a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
for _ in range(20):
    c = a @ b
q = torch.randn(8, 4, 512, 64, device=device, dtype=torch.bfloat16)
for _ in range(20):
    y = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
checksum = float(c.float().mean() + y.float().mean())
torch.cuda.synchronize()
assert torch.isfinite(c).all() and torch.isfinite(y).all()
print(f"PASS checksum={checksum:.6f} peak_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")
PY
nvidia-smi --query-gpu=index,uuid,memory.used,temperature.gpu --format=csv,noheader
