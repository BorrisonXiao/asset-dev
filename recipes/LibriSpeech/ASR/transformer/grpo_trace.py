#!/usr/bin/env python3
"""Dump GRPO sampling traces for a few dev utterances (diagnostic, no training).

For each utterance: the boundary policy's per-frame stats (mean p, entropy), the
argmax kept-ratio, then K sampled segmentations with their kept-ratio rho_k, reward
(= -NLL under the loaded decoder), and the resulting GRPO group-relative advantage.
This is what one GRPO step "sees". Reveals whether the samples have any spread
(exploration) and whether reward tracks rho.

Load checkpoints via yaml overrides:
  --trace_seg_dir  <coldstart or joint save dir>   (globs latest CKPT/segmenter.ckpt)
  --trace_dec_dir  <a run's save dir with proj.ckpt + llm.ckpt>  (the reward decoder)
  --trace_n_utts N  --grpo_k K  --trace_out <file>
"""
import glob
import os
import sys

import torch

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from segmenter import group_advantage
from train_speechllm import dataio_prepare
from train_speechllm_with_segmenter import SegmenterASR


def _latest_ckpt(save_dir):
    ckpts = sorted(glob.glob(os.path.join(save_dir, "CKPT*")))
    if not ckpts:
        raise FileNotFoundError(f"no CKPT* in {save_dir}")
    return ckpts[-1]


def _rho(boundary, pad_mask):
    valid = ~pad_mask
    n_seg = (boundary == 1).sum(dim=1).float() + 1.0
    return (n_seg / valid.sum(dim=1).float().clamp(min=1.0))


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f, overrides)

    from librispeech_prepare import prepare_librispeech  # noqa
    sb.utils.distributed.run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )
    tokenizer = hparams["llm"].tokenizer
    train_data, valid_data, _, tokenizer, _, _ = dataio_prepare(hparams, tokenizer)

    asr = SegmenterASR(
        modules=hparams["modules"], hparams=hparams, run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr.tokenizer = tokenizer
    asr.txt_embedding = asr.raw_modules.llm.model.get_input_embeddings()
    asr.current_epoch = 99

    # Load checkpoints.
    seg_ckpt = os.path.join(_latest_ckpt(hparams["trace_seg_dir"]), "segmenter.ckpt")
    asr.modules.segmenter.load_state_dict(torch.load(seg_ckpt, map_location="cpu"))
    print(f"# segmenter <- {seg_ckpt}")
    dec_dir = hparams.get("trace_dec_dir")
    if dec_dir:
        c = _latest_ckpt(dec_dir)
        asr.modules.proj.load_state_dict(
            torch.load(os.path.join(c, "proj.ckpt"), map_location="cpu")
        )
        # llm.ckpt holds ONLY the LoRA adapter weights (frozen base already loaded
        # from the HF cache at init), so load non-strict and overlay the adapters.
        llm_sd = torch.load(os.path.join(c, "llm.ckpt"), map_location="cpu")
        miss, unexp = asr.modules.llm.load_state_dict(llm_sd, strict=False)
        print(f"# decoder  <- {c}  (loaded {len(llm_sd)} llm-adapter tensors; "
              f"{len(unexp)} unexpected, {len(miss)} base-missing[expected])")

    asr.modules.eval()
    K = int(hparams.get("grpo_k", 8))
    n_utts = int(hparams.get("trace_n_utts", 6))
    out_lines = []

    def emit(s):
        print(s)
        out_lines.append(s)

    emit(f"# GRPO trace: K={K} samples/utt, reward = -NLL (teacher-forced)")
    emit(f"# {'utt_id':<22} {'T':>4} {'p_mean':>6} {'entropy':>7} {'rho_am':>6} "
         f"| per-sample rho_k / reward_k -> advantage")

    loader = sb.dataio.dataloader.make_dataloader(
        valid_data, batch_size=1, shuffle=False
    )
    # bf16 autocast (the LLM is bf16; training gets this via Brain.fit_batch).
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i, batch in enumerate(loader):
            if i >= n_utts:
                break
            batch = batch.to(asr.device)
            feats, feat_lens = asr._encoder_features(batch)
            T = feats.size(1)
            from segment_pooling import lengths_to_padding_mask
            pad = lengths_to_padding_mask(feat_lens, T)
            logits = asr.modules.segmenter.boundary_logits(feats, pad)
            p = torch.sigmoid(logits.float())[~pad]
            ent = -(p * p.clamp(min=1e-8).log()
                    + (1 - p) * (1 - p).clamp(min=1e-8).log()).mean()
            argmax_b = asr.modules.segmenter.sample_boundary(logits, pad, "argmax")
            rho_am = float(_rho(argmax_b, pad)[0])

            rhos, rewards, bnds = [], [], []
            for _ in range(K):
                bk = asr.modules.segmenter.sample_boundary(logits, pad, "sample")
                dec = asr._run_decoder(feats, pad, bk, batch)
                nll = asr._utterance_nll(dec["llm_logits"], batch)
                rhos.append(float(_rho(bk, pad)[0]))
                rewards.append(float(-nll[0]))
                bnds.append(bk)
            rew = torch.tensor(rewards).unsqueeze(0)
            adv = group_advantage(rew)[0].tolist()

            uid = batch.id[0][:22]
            emit(f"  {uid:<22} {T:>4} {float(p.mean()):>6.3f} {float(ent):>7.4f} "
                 f"{rho_am:>6.3f}")
            for k in range(K):
                emit(f"      sample {k}: rho={rhos[k]:.3f}  reward={rewards[k]:+.3f}  "
                     f"adv={adv[k]:+.3f}")
            emit(f"      --> reward mean {sum(rewards)/K:+.3f}, "
                 f"std {float(rew.std()):.4f}, rho spread "
                 f"[{min(rhos):.3f}, {max(rhos):.3f}]")

    out = hparams.get("trace_out")
    if out:
        with open(out, "w") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"\n# saved -> {out}")
