# Literature Review — Adaptive Speech Downsampling for Decoder-Only ASR

**Instruction**: Review the segmenter training report and investigate how to minimize the kept ratio while retaining or improving downstream ASR, with particular attention to decoder–segmenter frequency bias.  
**Date**: 2026-08-11  
**Window**: 3y, with seminal older work  
**Depth**: standard

## Scope & sub-questions

The core question is how to train a low-rate acoustic prefix for a frozen-encoder, decoder-only ASR model without losing the decoder behavior learned at character-aligned rates.

1. Does prior work observe co-adaptation between an ASR decoder and a particular encoder output rate or alignment?
2. Which adaptive compression mechanisms work better than independent frame-level Bernoulli selection?
3. How do successful systems control the emitted-token count and avoid unstable discrete optimization?
4. Is progressive or multi-rate training known to improve robustness to sequence-length changes?
5. Which objectives best expose the rate–WER Pareto frontier rather than choosing a rate indirectly?
6. What diagnostics can distinguish decoder adaptation error, boundary error, positional shift, and true information loss?

Target setting: frozen wav2vec 2.0/WavLM features at 50 Hz, mean-pooled learned segments, a decoder-only Llama acoustic prefix, and a small trainable projection/LoRA interface. No local PDFs were present under `papers/` or `literature/`; the project report and code were inspected locally.

## Landscape summary

The most direct precedent is variable-frame-rate ASR trained with reinforcement learning. Jiang et al. initialize a controller using phone alignments, then optimize a sequence-level error-minus-baseline objective plus a frame-rate cost. Their variable-rate system slightly improves both rate and WER over fixed-rate baselines, but they explicitly observe that pushing an already low-rate model further can reduce accuracy. This resembles the current staged imitation-to-RL setup, while also suggesting two improvements: use a matched high-rate baseline in the reward, and avoid propagating noisy controller gradients through shared acoustic representations.

A larger body of work avoids independent frame decisions. CIF integrates continuous boundary weights until a unit fires; CTC compressors collapse blank/repeated regions; AdaTranS predicts boundary candidates and uses training-time count forcing; AdaTS merges adjacent, similar speech tokens. These methods constrain the action space to monotonic, locally coherent segments and usually train the compressor jointly with the consumer. This is a strong contrast to assigning one utterance-level GRPO advantage to every source frame.

Decoder-only speech models are especially relevant. Speech-LLaMA, CTC-prompt ASR, and CJST all compress acoustic features before presenting them as a decoder prefix. CJST's strongest compression rule is a conservative CTC-blank threshold rather than maximal collapse, and its study reports a small quality–compactness trade-off across compression modes. AdaTS uses a second training stage in which the sampler is introduced while the downstream decoder is trainable. These results support *interface co-adaptation*, but none isolates a decoder that was first trained at one exact acoustic-prefix rate and then evaluated across much lower rates.

The literature therefore does not establish “character warm-up causes decoder frequency lock-in” as a known phenomenon. It does establish adjacent phenomena: alignment error propagates into autoregressive decoding; count control is crucial; decoders compensate for extreme encoder reduction; and models trained under stochastic length variation tolerate multiple budgets better. The exact decoder-rate transfer question in this project remains a useful open ablation.

## Paper table

| Paper (Author, Year) | Venue | Method | Key result | Relevance | Status |
|---|---|---|---|---|---|
| Variable Frame Rate Acoustic Models Using Minimum Error Reinforcement Learning (Jiang, Zhang, Woodland, 2021) | Interspeech 2021 | Phone-alignment controller pretraining followed by minimum-phone-error RL and a frame-rate cost | On MGB3, variable-rate models slightly beat fixed low-rate systems in both rate and WER; benefits narrow at very low rates | Closest objective/training analogue; motivates matched-baseline rewards and controlled rate pressure | ✅ verified (venue/DOI) |
| Learning Adaptive Downsampling Encoding for Online E2E Speech Recognition (Na et al., 2019) | APSIPA ASC 2019 | Differentiable expectation over keep/skip decisions; an entropy *penalty* sharpens decisions | TIMIT PER 20.86 versus 22.44 for fixed downsampling; adaptive behavior is robust to speed changes | A lower-variance alternative to REINFORCE and a warning that the sign/role of entropy matters | ✅ verified (venue) |
| Trainable Dynamic Subsampling for E2E Speech Recognition (Zhang et al., 2019) | Interspeech 2019 | Dynamic-subsampling RNN learns whether to skip frames | TIMIT PER 16.8 versus 23.5/20.4 for static uni-/bidirectional baselines | Early evidence that content-adaptive selection can outperform a fixed stride | ✅ verified (venue/DOI) |
| End-to-end Speech Recognition with Adaptive Computation Steps (Li, Liu, Hattori, 2018) | arXiv; submitted to ICASSP 2019 | Encoder-side adaptive integration emits a representation once sufficient acoustic evidence accumulates | AIShell-1 CER 31.2 online and 18.7 offline versus 32.4 and 22.0 attention baselines | Treats output rate as label-synchronous accumulation rather than frame-wise selection | ✅ verified (arXiv) |
| CIF: Continuous Integrate-and-Fire for E2E Speech Recognition (Dong, Xu, 2020) | ICASSP 2020 | Continuous monotonic integration with target-length scaling and a quantity loss | Competitive ASR; ablations show count scaling, quantity loss, and the autoregressive decoder are all important | Strong template for differentiable count control and token-synchronous acoustic embeddings | ✅ verified (arXiv/venue) |
| CTC-based Compression for Direct Speech Translation (Gaido et al., 2021) | EACL 2021 | Group frames by CTC predictions, then average or confidence-weight each group | +1.3–1.5 BLEU and more than 10% memory reduction in direct ST; ASR-side compression is near-lossless at phone scale | Supports linguistically constrained candidates and weighted rather than plain mean pooling | ✅ verified (venue/DOI) |
| AdaTranS (Zeng, Li, Liu, 2023) | Findings of EMNLP 2023 | Boundary predictor, weighted shrinking, auxiliary alignment losses, and training-time forced output count | Better quality/speed/memory than other shrinking methods; removing count forcing or weighted pooling hurts | Direct recipe for exact-rate supervised warm-up before free inference | ✅ verified (venue/arXiv) |
| CJST (Zhou et al., 2025) | ICASSP 2025 | Joint speech/text training with CTC compression and several blank/repeat collapse modes | On LibriSpeech, robust blank-threshold compression reports 2.22/4.94 WER at prompt/text ratio 1.22; stronger collapse reaches 1.08 with 2.27/5.04 | Closest published decoder-only ASR compression study; favors conservative confidence-aware removal | ✅ verified (arXiv/DOI/venue) |
| Decoder-only Architecture for ASR with CTC Prompts (Tsunoo et al., 2023) | arXiv | CTC-predicted encoder features are compressed into prompts for an autoregressive decoder | Text augmentation improves over ordinary CTC by 0.3/1.4 WER on LibriSpeech clean/other | Shows a decoder-only model can refine a compact CTC-derived prompt | ✅ verified (arXiv) |
| On Decoder-only Architecture for Speech-to-Text and LLM Integration (Wu et al., 2023) | ASRU 2023 | Speech-LLaMA uses a CTC compressor and adapter to map continuous acoustic features into an LLM | Competitive multilingual speech-to-text results with a compact continuous prefix | Establishes the same broad acoustic-prefix architecture and pretraining order | ✅ verified (arXiv/venue PDF) |
| Extreme Encoder Output Frame Rate Reduction (Prabhavalkar et al., 2024) | ICASSP 2024 | Several funnel reduction layers progressively compress a large ASR encoder | One output per 2.56 s with limited degradation on large-scale voice search; a richer prediction network greatly recovers extreme-rate errors | Evidence that very low rates are possible and that decoder capacity/context becomes more important | ✅ verified (arXiv/venue) |
| AdaTS: Adaptive Token Sampling for Efficient Speech Language Models (Sannigrahi et al., 2026) | ICLR 2026 MM Intelligence Workshop | Score-and-merge groups adjacent redundant tokens using similarity and learns merge weights | Roughly 2× fewer speech tokens and 40% inference-efficiency improvement with comparable or better ASR/SQA/ST | Most direct modern support for merge-based policies and introducing compression while the decoder adapts | ✅ verified (OpenReview; workshop paper) |
| Length-Adaptive Transformer (Kim, Cho, 2021) | ACL-IJCNLP 2021 | LengthDrop trains shared weights under stochastic sequence lengths | One model supports multiple accuracy/efficiency budgets after length-aware training | Non-speech precedent for rate-jitter warm-up and a sandwich of high/low/random budgets | ✅ verified (venue/arXiv) |
| Transformer ASR with Time Reduction and Self-Knowledge Distillation (Haidar et al., 2021) | Interspeech 2021 | Adds internal time-reduction layers and fine-tunes with self-distillation | Improves LibriSpeech accuracy and efficiency over the authors' Transformer baselines | Supports high-rate-teacher/low-rate-student logit distillation | ✅ verified (venue/DOI) |
| CTC-synchronous Training for Monotonic Attention (Inaguma et al., 2020) | Interspeech 2020 | Synchronizes expected monotonic-attention boundaries to an auxiliary CTC alignment | Improves TEDLIUM2 and LibriSpeech, especially on long utterances | Demonstrates that boundary drift can propagate through an autoregressive decoder and alignment supervision can stabilize it | ✅ verified (arXiv/venue) |

## Themes & consensus

### 1. Linguistically constrained, monotonic compression is safer than free frame selection

CTC collapse, CIF accumulation, boundary prediction, and adjacent-token merging all reduce the policy search space. They preserve order, discourage pathological short/long segments, and give each decision a clearer local interpretation. Independent Bernoulli actions over every 50 Hz frame are unusually unconstrained by comparison.

### 2. The compressor and decoder normally adapt together

The successful decoder-only systems either pretrain a CTC compressor and then train the speech–LLM interface, or introduce the compressor during a stage where the downstream adapter/decoder remains trainable. This supports the general co-adaptation concern. It does not prove that character-rate initialization is harmful; it predicts that a low-rate interface should receive its own adaptation phase.

### 3. Count control and boundary quality are distinct

CIF's quantity loss and AdaTranS's forced count show that hitting a target count is itself a training problem. CTC-synchronous training and the current report's boundary-swap experiment show that two segmentations with similar counts can differ sharply in downstream quality. A rate-only regularizer cannot substitute for good boundary geometry.

### 4. Extreme compression shifts responsibility to the decoder

Prabhavalkar et al. find that a richer prediction network recovers much of the loss at extreme encoder frame reduction. This is consistent with a rate-dependent division of labor: the acoustic prefix supplies evidence while the autoregressive decoder supplies increasingly long-range label context. A frozen or lightly adapted decoder may therefore become the limiting component below a threshold.

### 5. Multi-budget exposure and distillation are established stabilizers

LengthDrop makes one Transformer robust to multiple length budgets; speech time-reduction work benefits from self-distillation. Together they motivate multi-rate boundary merging during decoder warm-up plus text-logit distillation from the high-rate character model.

## Open gaps & opportunities

1. **Decoder-rate transfer has not been cleanly isolated.** Existing work changes compressor, rate, pooling, and decoder training jointly. A fixed-correct-boundary cross-rate matrix can directly estimate how much loss is recoverable by adapting only projection/LoRA/decoder parameters. (p8–p12)
2. **Decoder-only acoustic prefixes create a positional confound.** Shortening the prefix changes the RoPE position of all subsequent text tokens unless positions are explicitly controlled. The surveyed compression papers do not report an ablation that separates this from acoustic information loss. (p8–p10)
3. **Segment duration is usually implicit or discarded.** Plain mean pooling makes a two-frame and a twelve-frame segment comparable in shape while changing vector variance and erasing duration/order. Duration embeddings and confidence-weighted statistics are low-cost unexplored fixes in this exact setup. (p6–p8, p12)
4. **Frame-level utterance-reward credit assignment is weak.** Variable-rate RL exists, but modern successful compressors use accumulation, CTC candidates, or local merges. Candidate-boundary actions plus counterfactual merge costs could greatly reduce variance. (p1–p7, p12)
5. **The objective should match the scientific goal.** “Minimize rate subject to no more than ε ASR degradation” is naturally an epsilon-constraint/dual problem. A fixed rate band can obscure the Pareto frontier and gives no principled tolerance to trade quality for tokens. (p1, p13)
6. **Per-utterance budgets remain underexplored.** A global average budget can allocate extra tokens to fast/noisy/hard utterances, but needs tail-risk controls so the model does not sacrifice difficult examples. Speech-rate-conditioned budgets, max-duration constraints, and CVaR-WER are promising.
7. **Boundary importance can be measured before RL.** Leave-one-boundary-out changes in teacher NLL give a supervised target for local merge value. No reviewed work combines this counterfactual signal with a decoder-only LLM prefix.

## References

1. Jiang, D., Zhang, C., Woodland, P. C. (2021). [Variable Frame Rate Acoustic Models Using Minimum Error Reinforcement Learning](https://www.isca-archive.org/interspeech_2021/jiang21b_interspeech.html). Interspeech. DOI: 10.21437/Interspeech.2021-2198.
2. Na, R., Hou, J., Guo, W., Song, Y., Dai, L. (2019). [Learning Adaptive Downsampling Encoding for Online End-to-End Speech Recognition](https://www.apsipa.org/proceedings/2019/pdfs/15.pdf). APSIPA ASC.
3. Zhang, S., Loweimi, E., Xu, Y., Bell, P., Renals, S. (2019). [Trainable Dynamic Subsampling for End-to-End Speech Recognition](https://doi.org/10.21437/Interspeech.2019-2778). Interspeech.
4. Li, M., Liu, M., Hattori, M. (2018). [End-to-end Speech Recognition with Adaptive Computation Steps](https://arxiv.org/abs/1808.10088). arXiv:1808.10088.
5. Dong, L., Xu, B. (2020). [CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition](https://arxiv.org/abs/1905.11235). ICASSP. arXiv:1905.11235.
6. Gaido, M., Cettolo, M., Negri, M., Turchi, M. (2021). [CTC-based Compression for Direct Speech Translation](https://aclanthology.org/2021.eacl-main.57/). EACL. DOI: 10.18653/v1/2021.eacl-main.57.
7. Zeng, X., Li, L., Liu, Q. (2023). [AdaTranS: Adapting with Boundary-based Shrinking for End-to-End Speech Translation](https://aclanthology.org/2023.findings-emnlp.154/). Findings of EMNLP. arXiv:2212.08911.
8. Zhou, W., Jia, J., Sari, L., Mahadeokar, J., Kalinli, O. (2025). [CJST: CTC Compressor based Joint Speech and Text Training for Decoder-Only ASR](https://arxiv.org/abs/2411.07607). ICASSP. DOI: 10.1109/ICASSP49660.2025.10888940.
9. Tsunoo, E., Futami, H., Kashiwagi, Y., Arora, S., Watanabe, S. (2023). [Decoder-only Architecture for Speech Recognition with CTC Prompts and Text Data Augmentation](https://arxiv.org/abs/2309.08876). arXiv:2309.08876.
10. Wu, J. et al. (2023). [On Decoder-only Architecture for Speech-to-Text and Large Language Model Integration](https://arxiv.org/abs/2307.03917). ASRU. arXiv:2307.03917.
11. Prabhavalkar, R. et al. (2024). [Extreme Encoder Output Frame Rate Reduction](https://arxiv.org/abs/2402.17184). ICASSP. arXiv:2402.17184.
12. Sannigrahi, S., Attanasio, G., Martins, A. (2026). [AdaTS: Adaptive Token Sampling for Efficient Speech Language Models](https://openreview.net/forum?id=9XelZ6s8qe). ICLR MM Intelligence Workshop.
13. Kim, G., Cho, K. (2021). [Length-Adaptive Transformer](https://aclanthology.org/2021.acl-long.508/). ACL-IJCNLP. arXiv:2010.07003.
14. Haidar, M. A., Xing, C., Rezagholizadeh, M. (2021). [Transformer-Based ASR Incorporating Time-Reduction Layer and Fine-Tuning with Self-Knowledge Distillation](https://www.isca-archive.org/interspeech_2021/haidar21_interspeech.html). Interspeech. DOI: 10.21437/Interspeech.2021-1743.
15. Inaguma, H., Mimura, M., Kawahara, T. (2020). [CTC-synchronous Training for Monotonic Attention Model](https://arxiv.org/abs/2005.04712). Interspeech. arXiv:2005.04712.
