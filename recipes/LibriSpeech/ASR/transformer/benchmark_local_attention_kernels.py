#!/usr/bin/env python3
"""A100 benchmark for exact 64-frame causal local attention kernels."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


WINDOW = 64


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=8)
    return parser.parse_args()


def local_mask(length, device):
    pos = torch.arange(length, device=device)
    delta = pos[:, None] - pos[None, :]
    return (delta >= 0) & (delta < WINDOW)


def flex_mask(length, device):
    def mask_mod(batch, head, query, key):
        del batch, head
        delta = query - key
        return (delta >= 0) & (delta < WINDOW)

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=length,
        KV_LEN=length,
        device=str(device),
        # This PyTorch/A100 Inductor kernel requires the sparse-mask block size
        # to be divisible by its selected 128-token Triton tile. The semantic
        # mask remains an exact 64-frame window inside partial blocks.
        BLOCK_SIZE=128,
    )


def synchronize():
    torch.cuda.synchronize()


def forward_backward(fn, q, k, v, grad):
    for tensor in (q, k, v):
        tensor.grad = None
    out = fn(q, k, v)
    out.backward(grad)
    return out


def timed(fn, q, k, v, grad, repeats):
    forward_backward(fn, q, k, v, grad)
    synchronize()
    samples = []
    peak = 0
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        synchronize()
        start = time.perf_counter()
        forward_backward(fn, q, k, v, grad)
        synchronize()
        samples.append(time.perf_counter() - start)
        peak = max(peak, torch.cuda.max_memory_allocated())
    return {
        "median_seconds": statistics.median(samples),
        "samples": samples,
        "peak_allocated_gib": peak / 1024**3,
    }


def compare(reference_fn, candidate_fn, q, k, v, grad):
    ref_inputs = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    cand_inputs = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    ref = forward_backward(reference_fn, *ref_inputs, grad)
    candidate = forward_backward(candidate_fn, *cand_inputs, grad)
    return {
        "max_output_abs_error": (ref - candidate).abs().max().item(),
        "max_q_grad_abs_error": (
            ref_inputs[0].grad - cand_inputs[0].grad
        ).abs().max().item(),
        "max_k_grad_abs_error": (
            ref_inputs[1].grad - cand_inputs[1].grad
        ).abs().max().item(),
        "max_v_grad_abs_error": (
            ref_inputs[2].grad - cand_inputs[2].grad
        ).abs().max().item(),
    }


def main():
    cfg = args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(3407)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    compiled_flex = torch.compile(flex_attention, dynamic=True)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "window": WINDOW,
        "batch": cfg.batch,
        "heads": 4,
        "head_dim": 64,
        "results": [],
    }

    for length in cfg.lengths:
        shape = (cfg.batch, 4, length, 64)
        q, k, v = [
            torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            for _ in range(3)
        ]
        grad = torch.randn(shape, device=device, dtype=dtype)
        mask = local_mask(length, device)
        block_mask = flex_mask(length, device)

        def dense(q_, k_, v_):
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(
                    q_, k_, v_, attn_mask=mask, is_causal=False
                )

        def automatic(q_, k_, v_):
            return F.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=mask, is_causal=False
            )

        def flash(q_, k_, v_):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q_, k_, v_, attn_mask=mask, is_causal=False
                )

        def flex(q_, k_, v_):
            return compiled_flex(q_, k_, v_, block_mask=block_mask)

        row = {"length": length, "kernels": {}}
        reference = timed(dense, q, k, v, grad, cfg.repeats)
        row["kernels"]["math_dense_mask"] = reference
        for name, fn in (
            ("sdpa_auto_mask", automatic),
            ("sdpa_forced_flash_mask", flash),
            ("compiled_flex", flex),
        ):
            try:
                correctness = compare(dense, fn, q, k, v, grad)
                timing = timed(fn, q, k, v, grad, cfg.repeats)
                timing["speedup_vs_math"] = (
                    reference["median_seconds"] / timing["median_seconds"]
                )
                timing["correctness"] = correctness
                row["kernels"][name] = timing
            except Exception as error:
                row["kernels"][name] = {
                    "supported": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        result["results"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
