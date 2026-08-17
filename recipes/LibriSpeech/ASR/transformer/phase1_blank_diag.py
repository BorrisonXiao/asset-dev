#!/usr/bin/env python3
"""Phase-1 blank/separator diagnostic (no training) on the with-blank model.

Loads the exp-001 checkpoint (Model A, trained WITH separator tokens), runs the
frozen SSL + trained projection on a dev subset, pools with the word_ctc
boundaries, and characterizes the separator ("blank") vs word audio tokens:

  * geometry: L2-norm, within-group variance, mean pairwise cosine, PCA
    concentration -> tight/low-variance/high-cosine == register-like (H2);
    spread == content-rich (H5).
  * linear probes (closed-form ridge): can the separator embedding predict its
    own duration (pause length) and whether it is leading-silence vs an
    inter-word gap? Above-chance == the separator token carries structure/
    content (argues against a pure content-free register; toward H5/H4).

This PRIORITIZES hypotheses; it does not reproduce the WER effect (that needs the
trained drop model). See plans/blank_symbol_investigation.md.
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault(
    "HF_HUB_CACHE",
    "/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub",
)

import torch  # noqa: E402
from hyperpyyaml import load_hyperpyyaml  # noqa: E402

import speechbrain as sb  # noqa: E402
from segment_pooling import (  # noqa: E402
    lengths_to_padding_mask,
    mean_pool_segments,
    segment_separator_mask,
)

RECIPE = "/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/jointllm/recipes/LibriSpeech/ASR/transformer"
EXP1 = f"{RECIPE}/results/speechllm_fixed_pooling/alignment/3407"
BOUND = "/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/datasets/wavlm_boundaries/word_ctc"
SEP = "/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/boundary_targets/word_ctc_sep"
N_UTTS = int(os.environ.get("N_UTTS", "300"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Build modules from the yaml and recover exp-001 weights."""
    with open(f"{RECIPE}/hparams/speechllm_fixed_pooling.yaml") as f:
        hp = load_hyperpyyaml(
            f,
            overrides={
                "output_folder": EXP1,
                "ssl_folder": f"{EXP1}/save/ssl_checkpoint",
                "blank_mode": "keep",
            },
        )
    # Recover trained weights (ssl frozen, proj trained, normalize).
    hp["checkpointer"].recover_if_possible(min_key="WER")
    ssl = hp["ssl"].to(DEVICE).eval()
    proj = hp["proj"].to(DEVICE).eval()
    normalize = hp["normalize"]
    return hp, ssl, proj, normalize


def read_utts(csv_path, data_folder, n):
    """Parse (id, wav_path) from a SpeechBrain csv, sorted short->long."""
    import csv

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            wav = r["wav"].replace("$data_root", data_folder)
            rows.append((r["ID"], float(r["duration"]), wav))
    rows.sort(key=lambda x: x[1])
    # spread across the duration range
    idx = torch.linspace(0, len(rows) - 1, n).long().tolist()
    return [(rows[i][0], rows[i][2]) for i in sorted(set(idx))]


def ridge_probe(X, y, lam=1.0):
    """Closed-form ridge; returns R^2 (regression) on a 70/30 split."""
    n = X.shape[0]
    ntr = int(n * 0.7)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[:ntr], perm[ntr:]
    Xtr = torch.cat([X[tr], torch.ones(len(tr), 1)], 1)
    Xte = torch.cat([X[te], torch.ones(len(te), 1)], 1)
    A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1])
    w = torch.linalg.solve(A, Xtr.T @ y[tr])
    pred = Xte @ w
    ss_res = ((y[te] - pred) ** 2).sum()
    ss_tot = ((y[te] - y[te].mean()) ** 2).sum().clamp(min=1e-8)
    r2 = (1 - ss_res / ss_tot).item()
    # also accuracy if y is binary
    acc = None
    if set(y.unique().tolist()) <= {0.0, 1.0}:
        acc = ((pred > 0.5).float() == y[te]).float().mean().item()
    return r2, acc


@torch.no_grad()
def main():
    hp, ssl, proj, normalize = load_model()
    data_folder = hp["data_folder"]
    utts = read_utts(f"{EXP1}/dev-clean.csv", data_folder, N_UTTS)
    print(f"device={DEVICE}  utts={len(utts)}")

    word_pre, sep_pre = [], []      # pre-proj (768) segment embeddings
    word_post, sep_post = [], []    # post-proj (2048) LLM-input embeddings
    sep_dur, sep_is_lead = [], []   # separator segment metadata for probes

    for k, (uid, wav_path) in enumerate(utts):
        sig = sb.dataio.dataio.read_audio(wav_path).to(DEVICE).unsqueeze(0)
        wav_lens = torch.ones(1, device=DEVICE)
        feats = ssl(normalize(sig, wav_lens), wav_lens)  # (1, T, 768)
        T = feats.size(1)
        b = torch.load(f"{BOUND}/{uid}.pt").view(-1).long().to(DEVICE)
        iss = torch.load(f"{SEP}/{uid}.pt").view(-1).long().to(DEVICE)
        if b.numel() != T:  # guard against rare frame mismatch
            continue
        pad = lengths_to_padding_mask(wav_lens, T)
        boundary = b.masked_fill(pad.squeeze(0), -1).unsqueeze(0)
        pooled, seg_pad = mean_pool_segments(feats, pad, boundary)  # (1,S,768)
        sep_seg = segment_separator_mask(iss.unsqueeze(0), boundary, pad)[0]  # (S,)
        S = (~seg_pad[0]).sum().item()
        proj_out = proj(pooled)[0]  # (S, 2048)
        pooled = pooled[0]
        seg_ids = torch.cumsum((boundary[0] >= 1).long(), 0) - 1
        for s in range(S):
            (sep_pre if sep_seg[s] else word_pre).append(pooled[s].float().cpu())
            (sep_post if sep_seg[s] else word_post).append(proj_out[s].float().cpu())
            if sep_seg[s]:
                sep_dur.append(float((seg_ids == s).sum()))
                sep_is_lead.append(1.0 if s == 0 else 0.0)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(utts)} utts")

    def stats(name, pre, post):
        P = torch.stack(pre)
        Q = torch.stack(post)
        # mean pairwise cosine via mean-normalized-vector norm (sample up to 2000)
        def mean_cos(M):
            M = M[torch.randperm(M.shape[0])[:2000]]
            Mn = M / M.norm(dim=1, keepdim=True).clamp(min=1e-8)
            mu = Mn.mean(0)
            return (mu @ mu).item()  # E[cos] = ||mean unit vector||^2

        def pca_top(M, k=5):
            M = M - M.mean(0)
            s = torch.linalg.svdvals(M[torch.randperm(M.shape[0])[:3000]])
            v = (s**2) / (s**2).sum()
            return v[:k].sum().item()

        print(f"\n[{name}] n={P.shape[0]}")
        print(f"  pre-proj (768): |x|={P.norm(dim=1).mean():.2f}±{P.norm(dim=1).std():.2f}"
              f"  var(trace)={P.var(0).sum():.2f}  mean_cos={mean_cos(P):.3f}"
              f"  top5PC_var={pca_top(P):.2f}")
        print(f"  post-proj(2048):|x|={Q.norm(dim=1).mean():.2f}±{Q.norm(dim=1).std():.2f}"
              f"  var(trace)={Q.var(0).sum():.2f}  mean_cos={mean_cos(Q):.3f}"
              f"  top5PC_var={pca_top(Q):.2f}")

    stats("WORD tokens", word_pre, word_post)
    stats("SEP  tokens", sep_pre, sep_post)

    # Probes on SEP embeddings (post-proj = what the LLM sees).
    Xs = torch.stack(sep_post)
    r2_dur, _ = ridge_probe(Xs, torch.tensor(sep_dur))
    r2_lead, acc_lead = ridge_probe(Xs, torch.tensor(sep_is_lead))
    base_lead = max(sum(sep_is_lead), len(sep_is_lead) - sum(sep_is_lead)) / len(sep_is_lead)
    print("\n[PROBES on SEP post-proj embeddings]")
    print(f"  duration (pause length) regression:  R^2={r2_dur:.3f}")
    print(f"  leading-silence vs inter-word gap:   acc={acc_lead:.3f} (baseline {base_lead:.3f})")
    print("\ninterpretation: high sep mean_cos + low var + low probe scores => register-like (H2);"
          "\n                spread + above-baseline probes => structure/content in sep tokens (H4/H5).")

    torch.save(
        {"word_pre": torch.stack(word_pre), "sep_pre": torch.stack(sep_pre),
         "word_post": torch.stack(word_post), "sep_post": torch.stack(sep_post),
         "sep_dur": torch.tensor(sep_dur), "sep_is_lead": torch.tensor(sep_is_lead)},
        f"{RECIPE}/results/blank_ablation/phase1_embeddings.pt",
    )
    print(f"\nsaved embeddings -> {RECIPE}/results/blank_ablation/phase1_embeddings.pt")


if __name__ == "__main__":
    sys.exit(main())
