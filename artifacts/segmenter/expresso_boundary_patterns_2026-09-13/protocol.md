# Expresso boundary-pattern study — 2026-09-13

The author requests interpretation of individual boundary decisions: different task policies on the same utterance, the emotion policy across same-text expressive renditions, and acoustic/linguistic explanations. Recognition scores are secondary. Preserve the manuscript. This exploratory protocol is fixed before calculating the new pattern statistics; previous aggregate results are already known.

## Primary comparisons

1. Same audio: ASR parent, emotion seed 3408 (plus seeds 3407/3409), intent last-step and speaker-count last-step policies. All primary adapted examples include final-block updates. Show selected FiLM-only intent/count as controls, clearly labeled. Test coarsening, non-nesting, shared and task-specific retained ASR locations at 0/20/40 ms. Do not attribute rate differences solely to task: prescribed bands and shared initialization remain relevant.
2. Same speaker and text: all 60 groups, seven renditions each. Main affective contrasts use default/happy/sad/confused; whisper, laughing and enunciated are additional speaking styles, not interchangeable emotion labels. Compare word-relative and phone-relative cuts with monotone maps derived independently from MFA. Preserve gaps and report alignment exclusions; force-aligned edge correspondences are not discoveries. Include count-matched grids, within-phone randomization and default-boundary time-warps to expose rate/alignment artifacts.
3. Patterns: phone transitions/classes, vowel starts and lexical stress (not phrasal accent ground truth), rising/falling envelope edges, envelope peaks/valleys, voicing and pitch changes, phone duration, word/phrase position, and pause adjacency. Compare retained vs dropped parent cuts and use within-utterance or within-phone contrasts. Do not infer word/phoneme/syllable identity from isolated correspondence.
4. Additional controlled probe: 60 default utterances, resynthesis sham, duration-only and pitch-only manipulations, pitch flattening and envelope compression, with measured preservation of non-target cues. One frozen decoder-free A100 inference job; no training. Interpret policy sensitivity, not causal emotion recognition or unchanged human perception.

## Evidence and uncertainty

- Use raw saved cuts and automatic phone/word/pitch references, excluding compulsory utterance endpoints. Quantitative paired comparisons cover the full predefined panel.
- Timing conventions: frame index times 20 ms, plus +12.5-ms and 40-ms sensitivity. Same-text comparisons use phone correspondence and a stricter same-phone-sequence subset.
- Bootstrap speaker×text blocks (60), conditional on four speakers; summarize seed sensitivity explicitly. Style effects are exploratory, not independent population emotion estimates.
- Choose examples after calculation as representative near-median and explicitly labeled contrasting cases; do not substitute anecdotes for full-panel prevalence. Include exact event tables and aligned timelines that expose both agreement and disagreement.
- Source inspirations: Oganian & Chang (2019) on envelope-rate landmarks; Bhati et al. (2021) on transition-class dependence; Mo (2008) on duration/intensity versus prominence and phrasing; Nguyen et al. (2023), Expresso. These motivate measurements and do not establish mechanisms in our model.

Outputs are local artifacts under this directory (estimated below 500 MB). CPU analysis uses a 4-core Slurm allocation; controlled inference uses one A100. The revised HTML analysis will retain the existing URL and move the older performance audit to supporting material. No manuscript changes.
