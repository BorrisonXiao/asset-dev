#!/usr/bin/env python3
"""Benchmark attention backends for the project's Llama decoder on one GPU.

This intentionally leaves all training/evaluation entry points untouched.  It
times both the attention primitive with Llama-3.2-1B's GQA dimensions and the
actual locally cached Llama-3.2-1B-Instruct model used by the SpeechLLM recipe.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM

from speechbrain.nnet.adapters import AdaptedModel, LoRA

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}


def transformers_flash_attn_preflight() -> dict:
    """Verify that HF will dispatch to the external FA2 package.

    ``flash-attn`` lives in an isolated PYTHONPATH target so it cannot change
    the environment of running experiments.  Transformers snapshots the
    importlib package-to-distribution map during import; refresh that single
    entry here before checking availability.  This changes discovery only and
    never substitutes a different attention implementation.
    """
    import importlib.util

    from transformers import modeling_flash_attention_utils as fa_utils
    from transformers.utils import import_utils

    distributions = importlib.metadata.packages_distributions().get(
        "flash_attn", []
    )
    for package_map in (
        import_utils.PACKAGE_DISTRIBUTION_MAPPING,
        fa_utils.PACKAGE_DISTRIBUTION_MAPPING,
    ):
        package_map["flash_attn"] = list(distributions)

    for availability_check in {
        import_utils.is_flash_attn_2_available,
        fa_utils.is_flash_attn_2_available,
    }:
        cache_clear = getattr(availability_check, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()

    package_visible = (
        fa_utils.FLASH_ATTENTION_COMPATIBILITY_MATRIX[2][
            "pkg_availability_check"
        ]()
    )
    hf_available = import_utils.is_flash_attn_2_available()
    diagnostics = {
        "module_spec": str(importlib.util.find_spec("flash_attn")),
        "distributions": list(distributions),
        "package_visible_to_transformers": package_visible,
        "available_to_transformers": hf_available,
        "torch_cuda_available": torch.cuda.is_available(),
    }
    if flash_attn_func is None or not package_visible or not hf_available:
        raise RuntimeError(
            "External FlashAttention 2 failed the Transformers preflight: "
            f"{diagnostics}"
        )
    return diagnostics


def backend_context(name: str):
    if name == "auto":
        return nullcontext()
    return sdpa_kernel(backends=[BACKENDS[name]])


def timed_cuda(fn, warmup: int, repeats: int) -> tuple[list[float], float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    return times_ms, peak_gib


def summarize(times_ms: list[float], peak_gib: float) -> dict:
    ordered = sorted(times_ms)
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "p90_ms": ordered[max(0, int(0.9 * len(ordered)) - 1)],
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "peak_allocated_gib": peak_gib,
        "repeats": len(times_ms),
    }


def profile_auto_kernel(q, k, v, *, causal: bool, enable_gqa: bool) -> list[str]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, enable_gqa=enable_gqa
        )
        torch.cuda.synchronize()
    needles = ("scaled_dot_product", "flash", "efficient_attention", "cudnn")
    return sorted(
        event.key
        for event in prof.key_averages()
        if any(needle in event.key.lower() for needle in needles)
    )


def benchmark_attention_primitive() -> dict:
    """Test PyTorch SDPA and external FA2 at the decoder's GQA dimensions."""
    dtype = torch.bfloat16
    device = "cuda"
    cases = [
        ("prefill_b8_s256", 8, 256, 256, True, False),
        ("prefill_b4_s512", 4, 512, 512, True, False),
        ("decode_b8_q1_kv256", 8, 1, 256, False, False),
        ("train_b4_s256", 4, 256, 256, True, True),
    ]
    results = {}
    for case_name, batch, q_len, kv_len, causal, backward in cases:
        q = torch.randn(
            batch, 32, q_len, 64, device=device, dtype=dtype, requires_grad=backward
        )
        k = torch.randn(
            batch, 8, kv_len, 64, device=device, dtype=dtype, requires_grad=backward
        )
        v = torch.randn(
            batch, 8, kv_len, 64, device=device, dtype=dtype, requires_grad=backward
        )
        case = {
            "shape": {
                "batch": batch,
                "query_heads": 32,
                "kv_heads": 8,
                "q_len": q_len,
                "kv_len": kv_len,
                "head_dim": 64,
                "causal": causal,
                "backward": backward,
            },
            "auto_profiler_kernels": profile_auto_kernel(
                q.detach(), k.detach(), v.detach(), causal=causal, enable_gqa=True
            ),
            "backends": {},
        }
        with torch.no_grad():
            reference = F.scaled_dot_product_attention(
                q.detach(),
                k.detach(),
                v.detach(),
                is_causal=causal,
                enable_gqa=True,
            )
        for mode in (
            "auto",
            "flash",
            "external_flash_attention_2",
            "efficient",
            "cudnn",
            "math",
        ):
            try:
                def step():
                    if backward:
                        for tensor in (q, k, v):
                            tensor.grad = None
                    if mode == "external_flash_attention_2":
                        if flash_attn_func is None:
                            raise RuntimeError("flash-attn package is unavailable")
                        out = flash_attn_func(
                            q.transpose(1, 2),
                            k.transpose(1, 2),
                            v.transpose(1, 2),
                            dropout_p=0.0,
                            causal=causal,
                        ).transpose(1, 2)
                    else:
                        with backend_context(mode):
                            out = F.scaled_dot_product_attention(
                                q,
                                k,
                                v,
                                is_causal=causal,
                                enable_gqa=True,
                            )
                    if backward:
                        out.float().square().mean().backward()

                with torch.no_grad():
                    if mode == "external_flash_attention_2":
                        if flash_attn_func is None:
                            raise RuntimeError("flash-attn package is unavailable")
                        candidate = flash_attn_func(
                            q.detach().transpose(1, 2),
                            k.detach().transpose(1, 2),
                            v.detach().transpose(1, 2),
                            dropout_p=0.0,
                            causal=causal,
                        ).transpose(1, 2)
                    else:
                        with backend_context(mode):
                            candidate = F.scaled_dot_product_attention(
                                q.detach(),
                                k.detach(),
                                v.detach(),
                                is_causal=causal,
                                enable_gqa=True,
                            )

                repeats = 20 if backward else 50
                times, peak = timed_cuda(step, warmup=5, repeats=repeats)
                case["backends"][mode] = summarize(times, peak)
                diff = (candidate.float() - reference.float()).abs()
                case["backends"][mode]["max_abs_diff_vs_sdpa_auto"] = float(
                    diff.max()
                )
                case["backends"][mode]["mean_abs_diff_vs_sdpa_auto"] = float(
                    diff.mean()
                )
            except Exception as exc:  # unsupported backend is an expected result
                case["backends"][mode] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
            torch.cuda.empty_cache()
        results[case_name] = case
    return results


def full_model_step(model, embeds, mask, labels, mode: str, backward: bool):
    if backward:
        model.zero_grad(set_to_none=True)
        embeds.grad = None
    # The production recipe runs with ``precision: bf16``.  AdaptedModel keeps
    # trainable LoRA weights in fp32, so autocast is required to mirror that
    # path and avoid an artificial bf16-input/fp32-adapter dtype mismatch.
    with backend_context(mode), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(inputs_embeds=embeds, attention_mask=mask, labels=labels)
        if backward:
            output.loss.backward()
    return float(output.loss.detach())


def benchmark_full_llama(
    snapshot: str, attn_implementation: str, modes: tuple[str, ...]
) -> tuple[dict, AdaptedModel]:
    # Reset this before each implementation so LoRA initialization and synthetic
    # batches are matched exactly. LoRA's up projection starts at zero, as in the
    # real recipe.
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    base = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).cuda()
    model = AdaptedModel(
        model_to_adapt=base,
        adapter_class=LoRA,
        all_linear=True,
        adapter_kwargs={"rank": 16},
    ).cuda()

    results = {
        "model_type": type(base).__name__,
        "config": {
            "layers": base.config.num_hidden_layers,
            "hidden_size": base.config.hidden_size,
            "attention_heads": base.config.num_attention_heads,
            "kv_heads": base.config.num_key_value_heads,
            "attn_implementation": base.config._attn_implementation,
            "dtype": str(next(base.parameters()).dtype),
            "lora_rank": 16,
            "lora_all_linear": True,
        },
        "cases": {},
    }

    # Representative variable-length batches for compressed-audio teacher
    # forcing. The full vocabulary projection and LoRA backward are included.
    for name, batch, seq_len, backward, warmup, repeats in (
        ("teacher_forced_forward_b8_s256", 8, 256, False, 3, 12),
        ("teacher_forced_lora_b4_s192", 4, 192, True, 2, 8),
        ("teacher_forced_lora_b4_s384", 4, 384, True, 2, 6),
    ):
        model.train(backward)
        embeds = torch.randn(
            batch,
            seq_len,
            base.config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=backward,
        )
        lengths = torch.linspace(
            seq_len, max(seq_len // 2, 1), batch, device="cuda"
        ).round().long()
        positions = torch.arange(seq_len, device="cuda").unsqueeze(0)
        mask = (positions < lengths.unsqueeze(1)).long()
        labels = torch.randint(
            0, base.config.vocab_size, (batch, seq_len), device="cuda"
        )
        labels[:, : seq_len // 2] = -100  # approximate masked audio-prefix labels
        labels.masked_fill_(mask == 0, -100)
        case = {}
        reference_loss = None
        for mode in modes:
            try:
                def step():
                    nonlocal reference_loss
                    reference_loss = full_model_step(
                        model, embeds, mask, labels, mode, backward
                    )

                times, peak = timed_cuda(step, warmup=warmup, repeats=repeats)
                case[mode] = summarize(times, peak)
                case[mode]["last_loss"] = reference_loss
            except Exception as exc:
                case[mode] = {"error": f"{type(exc).__name__}: {exc}"}
            torch.cuda.empty_cache()
        results["cases"][name] = case

    return results, model


def benchmark_cached_generation(
    model: AdaptedModel, modes: tuple[str, ...]
) -> dict:
    model.eval()
    base = model.adapted_model
    batch, prefix_len, new_tokens = 8, 192, 64
    embeds = torch.randn(
        batch,
        prefix_len,
        base.config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    lengths = torch.linspace(
        prefix_len, prefix_len // 2, batch, device="cuda"
    ).round().long()
    positions = torch.arange(prefix_len, device="cuda").unsqueeze(0)
    # Match the batch-invariant decoder's left-packed prefixes.
    mask = (positions >= prefix_len - lengths.unsqueeze(1)).long()
    result = {}
    for mode in modes:
        try:
            generated_checksum = None

            def step():
                nonlocal generated_checksum
                with (
                    backend_context(mode),
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    generated = base.generate(
                        inputs_embeds=embeds,
                        attention_mask=mask,
                        do_sample=False,
                        use_cache=True,
                        min_new_tokens=new_tokens,
                        max_new_tokens=new_tokens,
                        eos_token_id=None,
                        pad_token_id=base.config.pad_token_id or 0,
                    )
                    generated_checksum = int(generated.long().sum())

            times, peak = timed_cuda(step, warmup=1, repeats=4)
            result[mode] = summarize(times, peak)
            result[mode]["generated_tokens_per_second"] = (
                batch * new_tokens * 1000.0 / result[mode]["median_ms"]
            )
            result[mode]["generated_token_checksum"] = generated_checksum
        except Exception as exc:
            result[mode] = {"error": f"{type(exc).__name__}: {exc}"}
        torch.cuda.empty_cache()
    return {
        "shape": {"batch": batch, "prefix_len": prefix_len, "new_tokens": new_tokens},
        "backends": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an NVIDIA GPU")

    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    flash_attn_preflight = transformers_flash_attn_preflight()
    result = {
        "environment": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": platform.node(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "flash_attn_package": flash_attn_func is not None,
            "flash_attn_version": (
                importlib.metadata.version("flash-attn")
                if flash_attn_func is not None
                else None
            ),
            "flash_attn_preflight": flash_attn_preflight,
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "attention_primitive": benchmark_attention_primitive(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    sdpa_full, sdpa_model = benchmark_full_llama(
        args.snapshot, "sdpa", ("auto", "flash")
    )
    result["full_llama"] = {"sdpa": sdpa_full}
    result["cached_generation"] = {
        "sdpa": benchmark_cached_generation(sdpa_model, ("auto", "flash"))
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    del sdpa_model
    gc.collect()
    torch.cuda.empty_cache()

    fa2_full, fa2_model = benchmark_full_llama(
        args.snapshot, "flash_attention_2", ("auto",)
    )
    result["full_llama"]["flash_attention_2"] = fa2_full
    result["cached_generation"]["flash_attention_2"] = (
        benchmark_cached_generation(fa2_model, ("auto",))
    )
    result["elapsed_seconds"] = time.time() - started
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
