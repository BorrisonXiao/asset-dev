# Literature Review — Order-preserving pooling for short speech segments

**Instruction**: Replace order-invariant segment averaging with an order-aware alternative; investigate RNN/LSTM and recommend options.  
**Date**: 2026-08-13  
**Window**: all (foundational sequence and segment models are load-bearing)  
**Depth**: focused-standard

## Scope & sub-questions

The core question is how to encode temporal order inside each predicted speech segment without losing the strong mean-pooled decoder initialization or adding material inference cost.

1. What information is mathematically removed by mean/statistics pooling?
2. Which sequence encoders have prior support for representing short spans?
3. Is a GRU/LSTM, endpoint feature, convolution, or tiny Transformer the best first test?
4. How can the current checkpoints be reused without shifting decoder inputs at initialization?
5. Which controls distinguish order information from extra capacity or retraining?

Constraints: WavLM/wav2vec2 features are frozen; retained ratio is about 0.23, so a typical segment is roughly 4.3 encoder frames; existing boundary-policy and best-character-decoder checkpoints should remain usable; the prototype must be disabled by default.

## Landscape summary

**Source facts.** Sum and mean are commutative reductions and represent a segment as a set rather than a sequence (Zaheer et al., 2017). Attention or statistics pooling does not automatically fix this: without a positional signal, its final reduction can remain permutation invariant (Lee et al., 2019; Okabe et al., 2018). Transformers likewise need position information to exploit order (Vaswani et al., 2017). Recurrent encoders update a state in temporal order; GRUs were introduced as variable-length phrase encoders (Cho et al., 2014), and segmental RNNs explicitly use bidirectional recurrent networks to embed the constituent sequence of a segment (Kong et al., 2016; Lu et al., 2016).

**Source facts.** The evidence does not imply that a terminal recurrent state should simply replace pooling. Maini et al. (2020) found worse early gradient flow and endpoint bias in terminal BiLSTM representations relative to pooled recurrent representations. Across NLP span tasks, the best representation varies by task and matters especially with a fixed pretrained encoder (Toshniwal et al., 2020); a later NER study found the simple ordered pair of span endpoints robust (Zaratiana et al., 2022). Convolutions are a parallel order-aware alternative (Gehring et al., 2017), though this does not establish a benefit for four-frame acoustic spans.

**Project-specific inference.** The `a→e` example is exactly correct for the pooling operator applied to fixed vectors. At the waveform level, WavLM and wav2vec2 features are already contextual and position-dependent, so reversing audio also changes the vectors before pooling (Baevski et al., 2020; Chen et al., 2022). Mean therefore weakens rather than necessarily erases all order information. This makes a small residual test preferable to replacing the representation wholesale.

**Recommendation.** Run a matched three-way ablation: endpoint residual, bidirectional GRU residual, and unchanged mean. Keep a bidirectional LSTM residual as a secondary capacity control. Project every learned path back to the original feature dimension and add it to the mean through a zero-initialized layer. An old checkpoint then starts with identical decoder inputs and learns only the useful correction.

## Paper table

| Paper (author, year) | Venue | Main finding | Relevance | Status |
|---|---|---|---|---|
| Deep Sets (Zaheer et al., 2017) | NeurIPS | Characterizes permutation-invariant set functions | Formal reason mean cannot distinguish a→e from e→a | ✅ venue |
| Long Short-Term Memory (Hochreiter & Schmidhuber, 1997) | Neural Computation | Gated memory addresses decaying error flow | Foundation for LSTM; its long-memory strength is unnecessary here | ✅ DOI |
| Learning Phrase Representations (Cho et al., 2014) | EMNLP | GRU maps variable-length phrases to vectors | Supports GRU as the lighter recurrent option | ✅ DOI/venue |
| Segmental Recurrent Neural Networks (Kong et al., 2016) | ICLR/arXiv | BiRNN embeddings for constituent tokens of spans | Closest general precedent | ✅ arXiv |
| Segmental RNNs for End-to-End Speech Recognition (Lu et al., 2016) | Interspeech | End-to-end recurrent segment features; 17.3% TIMIT PER | Direct speech precedent | ✅ DOI/venue |
| Why and when should you pool? (Maini et al., 2020) | Findings EMNLP | Pooling improved early gradient flow; terminal BiLSTMs had endpoint bias | Warns against a bare last-state replacement | ✅ DOI/venue |
| Cross-Task Analysis of Text Span Representations (Toshniwal et al., 2020) | RepL4NLP | Best span representation varies; effect is larger with fixed encoders | Motivates controlled ablations | ✅ DOI/venue |
| NER as Structured Span Prediction (Zaratiana et al., 2022) | UM-IoS | Endpoints robust across tested models and datasets | Motivates the cheapest ordered residual | ✅ DOI/venue |
| Attention Is All You Need (Vaswani et al., 2017) | NeurIPS | Adds positions because attention has no recurrence/convolution | Tiny attention requires within-segment positions | ✅ venue |
| Set Transformer (Lee et al., 2019) | ICML | Attention architecture designed to be set-invariant | Learned attention alone is not enough | ✅ venue |
| Attentive Statistics Pooling (Okabe et al., 2018) | Interspeech | Weighted mean/std improved speaker verification | Statistics may help but do not preserve order alone | ✅ venue |
| Convolutional Seq2Seq (Gehring et al., 2017) | ICML | Parallel order-aware convolutions optimized faster than recurrent MT baseline | Candidate if packed RNN overhead matters | ✅ venue |
| wav2vec 2.0 (Baevski et al., 2020) | NeurIPS | Context network produces contextual speech features | Qualifies strict invariance at waveform level | ✅ venue |
| WavLM (Chen et al., 2022) | IEEE JSTSP/arXiv | Uses a Transformer and relative-position bias | Same qualification for our decoder features | ✅ arXiv |

## Themes & consensus

### Order must enter before reduction

Mean, max, weighted mean, and mean-plus-standard-deviation are invariant to a permutation of supplied vectors unless their weights/features include position. An RNN, temporal convolution, ordered endpoints, or positional attention is required.

### Keep mean as a residual base

Prior work supports both recurrent span encoding and pooling, and also documents optimization advantages for pooling. The useful synthesis is “mean plus a learned ordered correction,” not “LSTM instead of mean.” A zero-initialized output projection also isolates representation gains from decoder distribution shift.

### Short segments change the ranking

At roughly four frames per segment, long memory is not the bottleneck. A GRU should precede an LSTM because it has fewer gates and parameters. Endpoints may suffice because upstream SSL vectors already contain broad context. A Transformer is excessive initially and needs positions plus a learned readout.

### Evidence is adjacent, not decisive

No paper evaluates frozen WavLM frames, learned character-like boundaries, a large autoregressive text decoder, and 4–5-frame spans together. The literature motivates an ablation; it does not guarantee an improvement.

## Recommended experiment

The prototype exposes four configurations while keeping `mean` as the default:

1. `endpoint_residual`: `mean + MLP(first, last, last-first, log length)`. Recommended cheapest diagnostic.
2. `bigru_residual`: one-layer BiGRU over each segment, projected back to the WavLM dimension and added to mean. Recommended expressive run.
3. `bilstm_residual`: matched BiLSTM capacity control, secondary priority.
4. `mean`: unchanged control.

Use the same boundaries, best character decoder, three seeds, optimizer, epochs, and rate controls. Train the residual with decoder CE during warmup and continue it during joint RL. Report WER, kept ratio, wall time, added parameters, and a reversal probe `||pool(x)-pool(reverse(x))||`. The decisive comparison is mean versus endpoint/BiGRU after equal training.

## Open gaps & opportunities

1. **Order or capacity?** Shuffle frames within each segment as a control. Loss of the gain under shuffling is stronger evidence than WER alone.
2. **Boundary error interaction.** Compare oracle-char and learned boundaries; endpoint-sensitive encoders may amplify boundary drift.
3. **Where is order already encoded?** Compare early/local SSL features with final WavLM features.
4. **Length-specific effects.** Stratify reversal sensitivity and decoder loss by segment length; one-frame spans cannot benefit.
5. **Parallel fallback.** If packed RNN calls are slow, test a depthwise temporal convolution before a positional Transformer.

## References

- [Baevski et al., 2020 — wav2vec 2.0](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html).
- [Chen et al., 2022 — WavLM](https://arxiv.org/abs/2110.13900), arXiv:2110.13900.
- [Cho et al., 2014 — Learning Phrase Representations](https://aclanthology.org/D14-1179/), DOI 10.3115/v1/D14-1179.
- [Gehring et al., 2017 — Convolutional Sequence to Sequence Learning](https://proceedings.mlr.press/v70/gehring17a.html).
- [Hochreiter & Schmidhuber, 1997 — Long Short-Term Memory](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory), DOI 10.1162/neco.1997.9.8.1735.
- [Kong et al., 2016 — Segmental Recurrent Neural Networks](https://arxiv.org/abs/1511.06018), arXiv:1511.06018.
- [Lee et al., 2019 — Set Transformer](https://proceedings.mlr.press/v97/lee19d.html).
- [Lu et al., 2016 — Segmental RNNs for Speech Recognition](https://www.isca-archive.org/interspeech_2016/lu16b_interspeech.html), DOI 10.21437/Interspeech.2016-40.
- [Maini et al., 2020 — Why and when should you pool?](https://aclanthology.org/2020.findings-emnlp.410/), DOI 10.18653/v1/2020.findings-emnlp.410.
- [Okabe et al., 2018 — Attentive Statistics Pooling](https://www.isca-archive.org/interspeech_2018/okabe18_interspeech.html).
- [Toshniwal et al., 2020 — Cross-Task Span Representations](https://aclanthology.org/2020.repl4nlp-1.20/), DOI 10.18653/v1/2020.repl4nlp-1.20.
- [Vaswani et al., 2017 — Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).
- [Zaheer et al., 2017 — Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).
- [Zaratiana et al., 2022 — NER as Structured Span Prediction](https://aclanthology.org/2022.umios-1.1/), DOI 10.18653/v1/2022.umios-1.1.
