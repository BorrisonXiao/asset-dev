# Proposed paper revision — not applied

Add a short **Emotion recognition and task adaptation** paragraph near the end of §4.2, with the five-row table in [emotion_paper_proposal.tex](emotion_paper_proposal.tex). The full six-row version remains in the analysis report. Retain native frames, fixed 10/5-Hz controls, the frozen ASR policy and GRPO; omit only the redundant 25-Hz row.

Proposed setup addition (can accompany the table caption):

> For single-task emotion recognition, we adapt the ASR model on CREMA-D using crowd voice-vote labels and select checkpoints by validation macro-F1. GRPO uses a label-likelihood margin reward and a 2–5-Hz rate band during final policy adaptation.

Proposed results paragraph:

> **Emotion recognition and task adaptation.** On CREMA-D, adapting the boundary policy with GRPO yields 43.8 ± 4.4% macro-F1 at 4.15 tokens/s, compared with 43.1 ± 0.4% at 8.56 tokens/s for a frozen ASR policy (three seeds). This is approximately 52% fewer tokens at similar mean macro-F1; fixed-rate baselines remain competitive. On 420 Expresso utterances matched by text across speaking styles, 94% of emotion boundaries fall within 20 ms of an ASR boundary, despite substantially coarser segmentation. Count-matched controls indicate selective retention of boundaries near forced-aligned vowel onsets and automatically estimated voicing changes. These observations suggest grouping around shared acoustic transitions; the specified rate band prevents interpreting the output frequency as an unconstrained task optimum.

“Selective retention” is the proposed description. Avoid claiming phoneme/word discovery, syllable or demi-syllable identity, improved preservation of prosody, an accuracy gain, or an optimal emotion frequency. If space is tight, keep the 94% same-audio containment result and omit the finer vowel/voicing interpretation from the paper, retaining it in the report.

Space plan:

1. Shorten **Compression and pooling at 100h** by roughly 40–50 words, dropping numerical repetitions already visible in the figure/table.
2. Merge **Input-dependent allocation** into one sentence in the existing boundary paragraph, saving about 25–35 words.
3. Shorten the efficiency paragraph's measurement details by about 20–25 words, retaining the full-path measurement definition and throughput result.
4. Keep the TIMIT and LibriSpeech phone-boundary evidence, but condense its protocol/baseline wording by roughly 20–30 words. Do not silently compare the historical grid F1 with this audit's continuous-time scores.

This should recover roughly 105–140 words plus some vertical space. The final layout must be compiled and checked after approval; no page-fit claim is made from word counts alone. No extra main-paper figure is necessary. The new scientific figures and examples can remain supporting material.

Revise the conclusion's final future-work sentence, which currently says additional speech tasks are still entirely future work. Describe the emotion result as an initial extension and reserve unconstrained rate selection and broader task generalization for future work.

Keep the abstract's current ASR/oracle sentence and its “approximately 25% fewer” wording. The new result is best introduced in the body at this stage. No manuscript source, bibliography, figure, or PDF has been changed by this audit.

If accepted, the method/setup text must also identify the restricted non-ASR adaptation (task FiLM, last policy block, pooler/projection/LoRA) and the classification-margin reward; it should not imply this is an unchanged ASR NLL objective or full-policy training from scratch.
