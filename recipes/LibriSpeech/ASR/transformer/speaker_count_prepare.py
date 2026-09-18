#!/usr/bin/env python3
"""Generate the speaker-count mixture recipes and their multi-task manifest.

Speaker count is the study's fully non-lexical task and the only one whose data
does not exist as a corpus: the mixtures are drawn from LibriSpeech speakers
already on disk. This script writes the *plan* for each mixture -- sources,
crops, onsets, gains, the noise seed -- and never the audio;
``speaker_count_mixing.render_mixture`` reconstructs the waveform on demand at
training time. Persistent cost is a few MB of manifests, not the hundreds of GB
a materialized LibriMix would take.

**Speaker ids, not speaker-chapter ids.** The LibriSpeech manifests' ``spk_id``
column is a speaker-CHAPTER key (``2277-149874``); train-clean-100 has 585 of
them but only 251 speakers. Mixing two "different" chapters of one reader would
label a one-speaker mixture as two. The true speaker is the leading field, and
that is what is drawn from here and what the split disjointness is checked on.

**The label is true by construction, then verified.** Every source's activity
interval contains a common peak window, so the maximum concurrent count is the
requested count. ``max_concurrency`` recomputes it from the stored intervals and
this script raises on any disagreement.

**Short peaks are the point.** A configurable fraction of mixtures reach their
full count only for a 0.5-1.5 s window; the rest hold it for longer. Without the
short-peak cases, total energy or utterance length would predict the count and
the task would say nothing about whether the segmenter tracks speaker structure.

**Two placement modes.** ``placement: peak`` (default) designs the common peak
window as above. ``placement: random_offset`` is LibriCount-style: each source
gets an independent random onset and spans at least ~60% of the clip ("speakers
active for most of the recording"), nothing forces a common window, and the
recipe is accepted by rejection sampling only when the realized concurrency
reaches the requested count -- so the label keeps meaning "max concurrent
speakers" while the overlap structure is emergent rather than designed.

**Two zero-class styles.** ``zero_style: noise`` keeps the audible -35 dBFS
Gaussian floor; ``silence`` uses an inaudible -48 dBFS floor matching the
measured level of LibriCount's count-0 examples; ``mixed`` draws either with
equal probability, so the zero class covers both what our old mixtures sound
like and what the external test set sounds like.

Splits use disjoint LibriSpeech sources, which are speaker-disjoint by
construction: train from the three LS960 train splits, validation from
``dev-clean``, and an internal test from ``test-clean``. LibriCount10/HEAR
remains the *external* test and is not generated here.
"""

import argparse
import collections
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from covost2_prepare import write_csv  # noqa: E402
from multitask_data import SPEAKER_COUNT_CANDIDATES, task_id_of  # noqa: E402
from speaker_count_mixing import (  # noqa: E402
    energy_vad,
    full_overlap_interval,
    max_concurrency,
    mix_uri,
)

SOURCE_DATASET = "librispeech_mixtures"
TARGET_SR = 16000

DEFAULT_SOURCES = {
    "train": ("train-clean-100", "train-clean-360", "train-other-500"),
    "valid": ("dev-clean",),
    "test": ("test-clean",),
}
DEFAULT_COUNTS = {"train": 20000, "valid": 2000, "test": 2000}

# The pilot ranks 0-4 (multitask_data.SPEAKER_COUNT_PILOT_CANDIDATES); the full
# task goes to 10 after the gate. Both read their labels from the same tuple.
DEFAULT_MAX_COUNT = 4

# A source contributes at least this much speech, so no "speaker" is a click.
MIN_SOURCE_SECONDS = 0.6
SHORT_PEAK_RANGE = (0.5, 1.5)

# --- LibriCount-style (random_offset) placement ---
# Minimum span a source covers in random_offset mode ("speakers active for most
# of the recording", as in the original corpus). Sources shorter than this fill
# their whole length instead.
RANDOM_MIN_SPAN = 3.0
# Placement redraws before the generator gives up on a requested count. With
# spans >= RANDOM_MIN_SPAN a common overlap is near-certain; the cap exists so
# a degenerate pool fails loudly instead of looping forever.
MAX_PLACEMENT_ATTEMPTS = 64
PLACEMENT_MODES = ("peak", "random_offset", "libricount")

# Measured from LibriCount's own per-speaker VAD annotations (Zenodo 1216072,
# 520 clips per count): every speaker vocalizes for ~93.5% of the 5 s clip, and
# all N speakers are simultaneously active for 93.5/86.9/81.4/76.5% of it at
# counts 1/2/3/4. Our earlier random_offset mode only guaranteed a single
# instant of N-way overlap, which is why the validated CountNet reference
# under-predicted counts 3-4 on our mixtures (0.70/0.32 against its 0.92/0.79 on
# the real corpus). The libricount mode reproduces the measured statistics:
# every source spans the whole clip, and crops are chosen to be mostly voiced.
LIBRICOUNT_MIN_DUTY = 0.88
LIBRICOUNT_CROP_ATTEMPTS = 12

# --- zero-class styles ---
# ``silence`` matches the measured level of LibriCount's count-0 examples
# (about -48 dBFS); it is still a seeded Gaussian floor rather than digital
# zeros, because a literally-empty class would be trivially detectable.
ZERO_STYLES = ("noise", "silence", "mixed")
# Floor under mixtures that DO contain speech. It exists only so the render is
# never bit-silent between sources; it must sit far below the speech or it
# becomes broadband babble. At -35 dBFS against -25 dBFS equal-power sources
# (10 dB SNR) the validated CountNet reference dropped to 0.55 accuracy on
# SINGLE-speaker clips; at -60 dBFS it recovers to 0.98. LibriCount adds no
# floor at all. The zero class keeps an audible floor (ZERO_STYLES above),
# because there a silent class would be free.
SPEECH_FLOOR_DBFS = -60.0
ZERO_SILENCE_FRACTION = 0.5
SILENCE_LEVEL_DBFS = -48.0


def true_speaker(spk_id):
    """LibriSpeech speaker id from the manifests' speaker-chapter key."""
    return str(spk_id).split("-", 1)[0]


def read_source_pool(manifest_dir, splits, data_folder, min_seconds):
    """Utterances grouped by true speaker, for one split's source corpora."""
    pool = collections.defaultdict(list)
    for split in splits:
        path = os.path.join(manifest_dir, f"{split}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LibriSpeech manifest not found: {path}")
        with open(path, encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                duration = float(row["duration"])
                if duration < min_seconds:
                    continue
                wav = row["wav"].replace("$data_root/", f"{data_folder}/")
                pool[true_speaker(row["spk_id"])].append(
                    {
                        "wav": os.path.normpath(wav),
                        "duration": duration,
                        "utt": row["ID"],
                    }
                )
    if not pool:
        raise ValueError(f"No usable source utterances in {splits}")
    return dict(pool)


def _draw_width(rng, duration, short_peak):
    """Target width of the window where every source is simultaneously active."""
    if short_peak:
        return float(rng.uniform(*SHORT_PEAK_RANGE))
    return float(min(rng.uniform(2.0, duration), duration))


def _select_sources(rng, pool, speakers, width):
    """One utterance per speaker, preferring ones long enough to span the peak.

    A source shorter than the peak window cannot stay active across it, and its
    interval would then pull the realized concurrency below the requested count
    -- i.e. the label would be wrong. Preferring eligible utterances keeps that
    from happening; when a speaker has none, their longest is taken and the
    caller shrinks the window to fit.
    """
    chosen = []
    for speaker in speakers:
        utterances = pool[speaker]
        eligible = [u for u in utterances if u["duration"] >= width]
        if eligible:
            chosen.append(eligible[int(rng.integers(len(eligible)))])
        else:
            chosen.append(max(utterances, key=lambda u: u["duration"]))
    return chosen


def _place(rng, duration, src_durations, width):
    """Onsets and spans whose intervals all contain one common peak window.

    ``onset_i`` is drawn no earlier than ``peak_end - src_dur_i``, so every
    source can still be playing at ``peak_end``; the span is then drawn long
    enough to reach it. Maximum concurrency is therefore the number of sources
    by construction, while onsets, offsets and overlap fractions stay varied.
    """
    width = min(width, duration, min(src_durations))
    peak_start = float(rng.uniform(0.0, duration - width))
    peak_end = peak_start + width

    placements = []
    for src_dur in src_durations:
        lo = max(0.0, peak_end - src_dur)
        onset = float(rng.uniform(lo, peak_start)) if peak_start > lo else lo
        span_lo = peak_end - onset
        span_hi = min(src_dur, duration - onset)
        span = (
            float(rng.uniform(span_lo, span_hi))
            if span_hi > span_lo
            else span_lo
        )
        placements.append((onset, span))
    return placements, (peak_start, peak_end)


def _voice_duty(wave, sr):
    """Fraction of a crop that ``energy_vad`` marks as speech."""
    if wave.shape[0] == 0:
        return 0.0
    total = wave.shape[0] / sr
    return sum(b - a for a, b in energy_vad(wave, sr)) / total if total else 0.0


def _select_libricount_crop(rng, utterances, duration, min_duty):
    """A full-clip crop that is mostly voiced, mimicking LibriCount's sources.

    LibriCount's speakers are active ~93.5% of the clip. LibriSpeech utterances
    contain pauses, so a crop taken blindly is often far quieter than that and
    the resulting mixture genuinely contains fewer concurrent voices than its
    label claims. Candidate offsets are scored by voice duty and the best is
    kept; the utterance must be at least as long as the clip.
    """
    long_enough = [u for u in utterances if u["duration"] >= duration]
    if not long_enough:
        long_enough = [max(utterances, key=lambda u: u["duration"])]
    best = None
    for _ in range(LIBRICOUNT_CROP_ATTEMPTS):
        source = long_enough[int(rng.integers(len(long_enough)))]
        span = min(duration, source["duration"])
        offset = float(rng.uniform(0.0, max(0.0, source["duration"] - span)))
        info = sf.info(source["wav"])
        wave, sr = sf.read(
            source["wav"],
            start=int(round(offset * info.samplerate)),
            frames=int(round(span * info.samplerate)),
            dtype="float32",
            always_2d=False,
        )
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        duty = _voice_duty(wave, sr)
        if best is None or duty > best[0]:
            best = (duty, source, offset, span)
        if duty >= min_duty:
            break
    return best


def _place_random(rng, duration, src_durations):
    """LibriCount-style placement: independent random onsets, no designed peak.

    Each source spans at least ``RANDOM_MIN_SPAN`` seconds (or its whole length
    when shorter), so speakers stay "active for most of the recording" as in
    the original corpus, but nothing forces a common overlap window. Whether
    the requested count is actually realized is the *caller's* job (rejection
    sampling in ``build_recipe``); the label stays "max concurrent speakers"
    rather than "sources placed".
    """
    placements = []
    for src_dur in src_durations:
        span_hi = min(src_dur, duration)
        span_lo = min(RANDOM_MIN_SPAN, span_hi)
        span = (
            float(rng.uniform(span_lo, span_hi))
            if span_hi > span_lo
            else span_lo
        )
        onset = float(rng.uniform(0.0, max(0.0, duration - span)))
        placements.append((onset, span))
    return placements


def _assemble_sources(
    rng, speakers, chosen, placements, gain_jitter_db=9.0, equal_power_dbfs=None
):
    """Source dicts from chosen utterances and (onset, span) placements.

    ``gain_jitter_db`` randomizes per-source loudness so it cannot stand in for
    the count; ``equal_power_dbfs`` instead brings every source to a common RMS,
    which is LibriCount's "0 dB SNR" construction. Setting jitter to 0 and a
    level here reproduces the benchmark; the defaults keep the hardened design.
    """
    sources = []
    for speaker, source, (onset, span) in zip(speakers, chosen, placements):
        max_offset = max(0.0, source["duration"] - span)
        sources.append(
            {
                "speaker": str(speaker),
                "wav": source["wav"],
                "utt": source["utt"],
                # 6 decimals is finer than one sample at 16 kHz, so rounding
                # cannot move an interval endpoint enough to change the
                # realized concurrency.
                "src_offset": round(float(rng.uniform(0.0, max_offset)), 6),
                "onset": round(float(onset), 6),
                "dur": round(float(span), 6),
                "gain_db": round(
                    float(rng.uniform(-gain_jitter_db, 0.0))
                    if gain_jitter_db
                    else 0.0,
                    2,
                ),
                "norm_rms_dbfs": equal_power_dbfs,
            }
        )
    return sources


def _place_random_sources(
    rng, duration, speakers, chosen, count, gain_jitter_db=9.0,
    equal_power_dbfs=None,
):
    """Random-offset placements whose realized concurrency reaches ``count``.

    Redraws the placement (and the crop/gain draws that follow it) until the
    intervals realize the requested count; raises if ``MAX_PLACEMENT_ATTEMPTS``
    draws never do, which only happens for a pool too short to span the clip.

    Returns the assembled sources and the realized full-overlap interval.
    """
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        placements = _place_random(
            rng, duration, [c["duration"] for c in chosen]
        )
        sources = _assemble_sources(
            rng, speakers, chosen, placements, gain_jitter_db, equal_power_dbfs
        )
        if max_concurrency({"sources": sources}) == count:
            return sources, full_overlap_interval(
                {"count": count, "sources": sources}
            )
    raise RuntimeError(
        f"random_offset placement failed to realize {count} concurrent "
        f"speakers after {MAX_PLACEMENT_ATTEMPTS} attempts; the source pool "
        "is too short for this count and clip length"
    )


def build_recipe(
    rng,
    utt_key,
    pool,
    speakers_sorted,
    count,
    duration,
    noise_dbfs,
    target_rms_dbfs=-20.0,
    short_peak_fraction=0.5,
    placement="peak",
    zero_style="mixed",
    gain_jitter_db=9.0,
    equal_power_dbfs=None,
    speech_floor_dbfs=SPEECH_FLOOR_DBFS,
):
    """One mixture recipe: which speakers, which crops, placed where.

    ``placement`` selects the scheme: ``"peak"`` designs a common peak window
    (the original behavior); ``"random_offset"`` draws LibriCount-style
    independent onsets and is accepted by rejection sampling once the realized
    concurrency reaches the requested count. ``zero_style`` selects the count-0
    floor -- ``"noise"`` (-35 dBFS), ``"silence"`` (-48 dBFS, matching the
    measured LibriCount zero level), or ``"mixed"`` for either with equal
    probability; it is ignored for count > 0.
    """
    if placement not in PLACEMENT_MODES:
        raise ValueError(f"Unknown placement mode {placement!r}")
    if zero_style not in ZERO_STYLES:
        raise ValueError(f"Unknown zero style {zero_style!r}")
    sources = []
    peak = None
    if count:
        speakers = list(rng.choice(speakers_sorted, size=count, replace=False))
        if placement == "libricount":
            # Every source spans the whole clip, as in the real corpus, so the
            # concurrency is the source count by construction and the label
            # needs no rejection sampling.
            sources = []
            for speaker in speakers:
                duty, chosen_src, offset, span = _select_libricount_crop(
                    rng, pool[speaker], duration, LIBRICOUNT_MIN_DUTY
                )
                sources.append(
                    {
                        "speaker": str(speaker),
                        "wav": chosen_src["wav"],
                        "utt": chosen_src["utt"],
                        "src_offset": round(float(offset), 6),
                        "onset": 0.0,
                        "dur": round(float(span), 6),
                        "voice_duty": round(float(duty), 4),
                        "gain_db": round(
                            float(rng.uniform(-gain_jitter_db, 0.0))
                            if gain_jitter_db
                            else 0.0,
                            2,
                        ),
                        "norm_rms_dbfs": equal_power_dbfs,
                    }
                )
            peak = (0.0, float(duration))
        elif placement == "random_offset":
            chosen = _select_sources(rng, pool, speakers, RANDOM_MIN_SPAN)
            sources, peak = _place_random_sources(
                rng, duration, speakers, chosen, count, gain_jitter_db,
                equal_power_dbfs,
            )
        else:
            short_peak = bool(rng.random() < short_peak_fraction)
            width = _draw_width(rng, duration, short_peak)
            chosen = _select_sources(rng, pool, speakers, width)
            placements, peak = _place(
                rng, duration, [c["duration"] for c in chosen], width
            )
            sources = _assemble_sources(
                rng, speakers, chosen, placements, gain_jitter_db,
                equal_power_dbfs,
            )

    if count == 0:
        if zero_style == "mixed":
            style = (
                "silence"
                if rng.random() < ZERO_SILENCE_FRACTION
                else "noise"
            )
        else:
            style = zero_style
        floor_dbfs = (
            float(SILENCE_LEVEL_DBFS) if style == "silence" else float(noise_dbfs)
        )
    else:
        style = None
        floor_dbfs = float(speech_floor_dbfs)

    recipe = {
        "utt_key": utt_key,
        "count": int(count),
        "duration": float(duration),
        "sample_rate": TARGET_SR,
        "placement": placement,
        "zero_style": style,
        "noise": {
            "kind": "gaussian",
            "level_dbfs": floor_dbfs,
            "seed": int(rng.integers(2**31 - 1)),
        },
        # Normalizing the speech sum stops total energy standing in for the
        # count. Disabled (None) for the libricount profile on purpose: the
        # benchmark's 0 dB SNR construction keeps that cue, and removing it is
        # what made our mixtures far harder than the benchmark -- the validated
        # CountNet reference scores 0.927 on LibriCount and 0.564 on the
        # hardened profile.
        "target_rms_dbfs": (
            round(
                float(rng.uniform(target_rms_dbfs - 3.0, target_rms_dbfs + 3.0)),
                2,
            )
            if target_rms_dbfs is not None
            else None
        ),
        "peak_interval": (
            [round(peak[0], 6), round(peak[1], 6)] if peak else None
        ),
        "sources": sources,
    }

    achieved = max_concurrency(recipe)
    if achieved != count:
        raise AssertionError(
            f"{utt_key}: placed {count} sources but the intervals reach "
            f"{achieved} concurrent speakers. Placement is broken; the label "
            "would be wrong."
        )
    return recipe


def _vad_one(args):
    """Voice-activity spans, in mixture time, for every source of one recipe."""
    recipe, = args
    out = []
    for source in recipe["sources"]:
        info = sf.info(source["wav"])
        start = int(round(float(source["src_offset"]) * info.samplerate))
        frames = int(round(float(source["dur"]) * info.samplerate))
        wave, sr = sf.read(
            source["wav"], start=start, frames=frames, dtype="float32",
            always_2d=False,
        )
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        onset = float(source["onset"])
        out.append(
            [[round(onset + a, 6), round(onset + b, 6)]
             for a, b in energy_vad(wave, sr)]
        )
    return out


def apply_vad_labels(recipes, workers):
    """Relabel recipes by max simultaneously *vocalizing* speakers.

    LibriCount's ground truth is the max concurrent speakers by voice-activity
    detection, not the number of overlapping source placements. Read speech is
    full of internal pauses, so counting placements labels a mixture higher than
    a listener (or the benchmark) would: measured with the validated CountNet
    reference, placement labels produced a -0.31 prediction bias, i.e. the model
    consistently heard fewer speakers than our labels claimed.

    The realized count can only be known after measuring the audio, so this runs
    as a post-pass and *overwrites* the requested count. That necessarily
    unbalances the class distribution -- a mixture of 4 placed speakers whose
    pauses never coincide is genuinely a 3-speaker mixture -- so the caller
    rebalances afterwards.
    """
    changed = collections.Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for recipe, spans in zip(
            recipes, pool.map(_vad_one, [(r,) for r in recipes], chunksize=16)
        ):
            for source, active in zip(recipe["sources"], spans):
                source["active"] = active
            realized = max_concurrency(recipe)
            changed[(recipe["count"], realized)] += 1
            recipe["count"] = realized
            recipe["label_source"] = "vad"
    return changed


def _rebalance(recipes, rng):
    """Even out counts after VAD relabelling by subsampling to the rarest."""
    by_count = collections.defaultdict(list)
    for r in recipes:
        by_count[r["count"]].append(r)
    keep_n = min(len(v) for v in by_count.values())
    kept = []
    for count in sorted(by_count):
        group = by_count[count]
        idx = rng.choice(len(group), size=keep_n, replace=False)
        kept.extend(group[int(i)] for i in sorted(idx))
    kept.sort(key=lambda r: r["utt_key"])
    return kept, keep_n


def _manifest_row(recipe, recipes_path, source_key):
    """Project one mixture onto the multi-task schema."""
    speakers = sorted({s["speaker"] for s in recipe["sources"]})
    return {
        "ID": f"speaker_count-{recipe['utt_key']}",
        "duration": recipe["duration"],
        "wav": mix_uri(recipes_path, recipe["utt_key"]),
        "spk_id": "+".join(speakers) if speakers else "none",
        "wrd": SPEAKER_COUNT_CANDIDATES[recipe["count"]],
        "task_name": "speaker_count",
        "task_id": task_id_of("speaker_count"),
        "source_key": source_key,
        "source_dataset": SOURCE_DATASET,
        "source_lang": "en",
        "target_lang": "en",
        "utt_key": recipe["utt_key"],
    }


def prepare_split(
    split,
    source_splits,
    n_mixtures,
    manifest_dir,
    librispeech_manifest_dir,
    data_folder,
    out_dir,
    max_count,
    duration,
    noise_dbfs,
    target_rms_dbfs,
    short_peak_fraction,
    seed,
    placement="peak",
    zero_style="mixed",
    gain_jitter_db=9.0,
    equal_power_dbfs=None,
    speech_floor_dbfs=SPEECH_FLOOR_DBFS,
    vad_labels=False,
    workers=8,
):
    """Write one split's recipes JSONL and manifest CSV."""
    pool = read_source_pool(
        librispeech_manifest_dir,
        source_splits,
        data_folder,
        MIN_SOURCE_SECONDS,
    )
    speakers_sorted = sorted(pool)
    if len(speakers_sorted) < max_count:
        raise ValueError(
            f"{split}: {len(speakers_sorted)} speakers cannot fill mixtures of "
            f"up to {max_count} distinct speakers"
        )

    # Balanced over counts: an unbalanced count prior would let the model score
    # well by guessing the mode, exactly the failure the emotion task already
    # has to guard against.
    counts = [i % (max_count + 1) for i in range(n_mixtures)]

    recipes_path = os.path.join(out_dir, f"speaker_count_{split}_recipes.jsonl")
    recipes = []
    for index, count in enumerate(counts):
        # Independent, reproducible stream per mixture: regenerating one row
        # does not disturb any other.
        rng = np.random.default_rng([seed, index])
        recipes.append(
            build_recipe(
                rng,
                f"spkcount-{split}-{index:06d}",
                pool,
                speakers_sorted,
                count,
                duration,
                noise_dbfs,
                target_rms_dbfs,
                short_peak_fraction,
                placement,
                zero_style,
                gain_jitter_db,
                equal_power_dbfs,
                speech_floor_dbfs,
            )
        )

    vad_summary = None
    if vad_labels:
        moved = apply_vad_labels(recipes, workers)
        kept, per_count = _rebalance(recipes, np.random.default_rng(seed))
        vad_summary = {
            "relabelled": True,
            "before_after_counts": {
                f"{a}->{b}": n for (a, b), n in sorted(moved.items())
            },
            "unchanged": sum(n for (a, b), n in moved.items() if a == b),
            "recipes_before_rebalance": len(recipes),
            "per_count_after_rebalance": per_count,
        }
        recipes = kept
        print(
            f"[{split}] VAD relabelling: "
            f"{vad_summary['unchanged']}/{vad_summary['recipes_before_rebalance']} "
            f"kept their placed count; rebalanced to {per_count} per count "
            f"({len(recipes)} total)",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(recipes_path)), exist_ok=True)
    with open(recipes_path, "w", encoding="utf-8") as stream:
        for recipe in recipes:
            stream.write(json.dumps(recipe, sort_keys=True) + "\n")

    source_key = "speaker_count"
    manifest_path = os.path.join(manifest_dir, f"speaker_count_{split}.csv")
    write_csv(
        manifest_path,
        [_manifest_row(r, recipes_path, source_key) for r in recipes],
    )

    # In ``peak`` mode this is the share of *designed* brief peak windows; in
    # ``random_offset`` mode it is the share of mixtures whose *realized*
    # full-overlap interval comes out that brief. Either way it is the share of
    # mixtures where the full count holds for at most 1.5 s.
    short = sum(
        1
        for r in recipes
        if r["peak_interval"]
        and (r["peak_interval"][1] - r["peak_interval"][0])
        <= SHORT_PEAK_RANGE[1]
    )
    multi = sum(1 for r in recipes if r["count"] > 0)
    stats = {
        "mixtures": len(recipes),
        "hours": round(len(recipes) * duration / 3600.0, 3),
        "speakers_available": len(speakers_sorted),
        "placement": placement,
        "count_histogram": dict(
            sorted(collections.Counter(r["count"] for r in recipes).items())
        ),
        "short_peak_share": round(short / multi, 4) if multi else 0.0,
        "vad": vad_summary,
        "zero_style_histogram": dict(
            sorted(
                collections.Counter(
                    r["zero_style"] for r in recipes if r["count"] == 0
                ).items()
            )
        ),
        "recipes": recipes_path,
        "manifest": manifest_path,
    }
    print(
        f"[{split}] {stats['mixtures']} mixtures, {stats['hours']:.2f} h, "
        f"{stats['speakers_available']} speakers available, "
        f"{stats['short_peak_share']:.0%} short-peak\n"
        f"  {recipes_path}\n  {manifest_path}",
        flush=True,
    )
    return stats, recipes


def audit_render(recipes, n_samples, seed):
    """Render a random handful and confirm they are real, finite audio.

    A recipe that points at a missing or wrong-rate file is silent until the
    first training step touches it, which on a queued A100 job is an expensive
    place to find out.
    """
    rng = np.random.default_rng(seed)
    from speaker_count_mixing import render_mixture

    picked = rng.choice(
        len(recipes), size=min(n_samples, len(recipes)), replace=False
    )
    report = []
    for index in picked:
        recipe = recipes[int(index)]
        wave = render_mixture(recipe, as_tensor=False)
        expected = int(round(recipe["duration"] * recipe["sample_rate"]))
        if wave.shape[0] != expected:
            raise AssertionError(
                f"{recipe['utt_key']}: rendered {wave.shape[0]} samples, "
                f"expected {expected}"
            )
        if not np.isfinite(wave).all():
            raise AssertionError(f"{recipe['utt_key']}: non-finite samples")
        report.append((recipe["count"], float(np.sqrt((wave**2).mean()))))
    by_count = collections.defaultdict(list)
    for count, rms in report:
        by_count[count].append(rms)
    return {
        count: round(float(np.mean(values)), 5)
        for count, values in sorted(by_count.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--librispeech-manifest-dir",
        default="/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests",
    )
    parser.add_argument(
        "--data-folder",
        default="/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech",
        help="LibriSpeech root, used to resolve $data_root in the manifests",
    )
    parser.add_argument("--manifest-dir", required=True, help="CSV output dir")
    parser.add_argument(
        "--recipes-dir",
        default=None,
        help="recipes JSONL output dir (default: --manifest-dir)",
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument(
        "--max-count",
        type=int,
        default=DEFAULT_MAX_COUNT,
        help="largest concurrent speaker count (pilot 4, full 10)",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--noise-dbfs",
        type=float,
        default=-35.0,
        help="background floor, so the zero-speaker class is not free",
    )
    parser.add_argument(
        "--target-rms-dbfs",
        type=float,
        default=-20.0,
        help="speech level the mixture is normalized to (jittered +/-3 dB), so "
        "total energy cannot stand in for the speaker count",
    )
    parser.add_argument(
        "--short-peak-fraction",
        type=float,
        default=0.5,
        help="share of mixtures whose full count is reached for only 0.5-1.5 s",
    )
    parser.add_argument(
        "--placement",
        choices=PLACEMENT_MODES,
        default="peak",
        help="peak: designed common peak window; random_offset: LibriCount-style "
        "independent random onsets, accepted by rejection sampling once the "
        "realized concurrency reaches the requested count",
    )
    parser.add_argument(
        "--zero-style",
        choices=ZERO_STYLES,
        default="mixed",
        help="count-0 floor: noise (-35 dBFS), silence (-48 dBFS, matching the "
        "measured LibriCount zero level), or mixed (either, equally likely)",
    )
    parser.add_argument(
        "--profile",
        choices=("hardened", "libricount"),
        default="hardened",
        help="hardened = the original design (randomized gains, short peak "
        "windows, mixture level-normalized so global energy cannot predict the "
        "count). libricount = the benchmark's construction (equal-power 0 dB "
        "SNR sources, random offsets spanning most of the clip, silence-style "
        "zero class, NO mixture normalization). The validated CountNet "
        "reference scores 0.927 on real LibriCount and 0.564 on hardened data, "
        "so the profile must be reported with any number.",
    )
    parser.add_argument(
        "--equal-power-dbfs",
        type=float,
        default=None,
        help="bring every source to this RMS before its gain (0 dB SNR)",
    )
    parser.add_argument(
        "--gain-jitter-db",
        type=float,
        default=9.0,
        help="per-source random attenuation range; 0 disables",
    )
    parser.add_argument(
        "--no-normalize-mixture",
        action="store_true",
        help="leave total energy proportional to count, as LibriCount does",
    )
    parser.add_argument(
        "--speech-floor-dbfs",
        type=float,
        default=SPEECH_FLOOR_DBFS,
        help="floor under mixtures containing speech (the zero class uses "
        "--zero-style instead); must sit far below the sources",
    )
    parser.add_argument(
        "--vad-labels",
        action="store_true",
        help="label by max simultaneously VOCALIZING speakers (LibriCount's "
        "definition) instead of overlapping placements, then rebalance",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--audit-renders",
        type=int,
        default=24,
        help="mixtures to actually render as a smoke check (0 to skip)",
    )
    for split, n in DEFAULT_COUNTS.items():
        parser.add_argument(f"--n-{split}", type=int, default=n)
    args = parser.parse_args()

    # The profile is a preset over the individual knobs; anything passed
    # explicitly still wins, so an ablation can vary one factor at a time.
    if args.profile == "libricount":
        given = set(sys.argv[1:])
        if not any(a.startswith("--placement") for a in given):
            args.placement = "random_offset"
        if not any(a.startswith("--zero-style") for a in given):
            args.zero_style = "silence"
        if not any(a.startswith("--equal-power-dbfs") for a in given):
            args.equal_power_dbfs = -25.0
        if not any(a.startswith("--gain-jitter-db") for a in given):
            args.gain_jitter_db = 0.0
        if not any(a.startswith("--no-normalize-mixture") for a in given):
            args.no_normalize_mixture = True
    target_rms = None if args.no_normalize_mixture else args.target_rms_dbfs

    if args.max_count >= len(SPEAKER_COUNT_CANDIDATES):
        parser.error(
            f"--max-count {args.max_count} exceeds the task table's "
            f"{len(SPEAKER_COUNT_CANDIDATES)} count labels"
        )

    os.makedirs(args.manifest_dir, exist_ok=True)
    recipes_dir = args.recipes_dir or args.manifest_dir
    os.makedirs(recipes_dir, exist_ok=True)

    summary = {
        "librispeech_manifest_dir": args.librispeech_manifest_dir,
        "max_count": args.max_count,
        "duration": args.duration,
        "profile": args.profile,
        "noise_dbfs": args.noise_dbfs,
        "equal_power_dbfs": args.equal_power_dbfs,
        "gain_jitter_db": args.gain_jitter_db,
        "normalize_mixture": not args.no_normalize_mixture,
        "speech_floor_dbfs": args.speech_floor_dbfs,
        "vad_labels": args.vad_labels,
        "target_rms_dbfs": args.target_rms_dbfs,
        "short_peak_fraction": args.short_peak_fraction,
        "placement": args.placement,
        "zero_style": args.zero_style,
        "seed": args.seed,
        "splits": {},
    }
    speakers_by_split = {}
    for split in args.splits:
        stats, recipes = prepare_split(
            split,
            DEFAULT_SOURCES[split],
            getattr(args, f"n_{split}"),
            args.manifest_dir,
            args.librispeech_manifest_dir,
            args.data_folder,
            recipes_dir,
            args.max_count,
            args.duration,
            args.noise_dbfs,
            target_rms,
            args.short_peak_fraction,
            args.seed + 7919 * args.splits.index(split),
            placement=args.placement,
            zero_style=args.zero_style,
            gain_jitter_db=args.gain_jitter_db,
            equal_power_dbfs=args.equal_power_dbfs,
            speech_floor_dbfs=args.speech_floor_dbfs,
            vad_labels=args.vad_labels,
            workers=args.workers,
        )
        if args.audit_renders:
            stats["rendered_rms_by_count"] = audit_render(
                recipes, args.audit_renders, args.seed
            )
            print(
                f"  rendered {min(args.audit_renders, len(recipes))} mixtures, "
                f"RMS by count: {stats['rendered_rms_by_count']}",
                flush=True,
            )
        speakers_by_split[split] = {
            s["speaker"] for r in recipes for s in r["sources"]
        }
        summary["splits"][split] = stats

    # The split corpora are disjoint by LibriSpeech construction; check it
    # anyway, because a mixed-up --splits argument would silently put test
    # speakers in training.
    for a in speakers_by_split:
        for b in speakers_by_split:
            if a < b and speakers_by_split[a] & speakers_by_split[b]:
                raise ValueError(
                    f"Splits {a} and {b} share speakers: "
                    f"{sorted(speakers_by_split[a] & speakers_by_split[b])[:5]}"
                )
    summary["speakers_used"] = {
        split: len(members) for split, members in speakers_by_split.items()
    }

    sidecar = os.path.join(args.manifest_dir, "speaker_count_prep.json")
    with open(sidecar, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(f"Provenance written to {sidecar}", flush=True)
    total = sum(s["mixtures"] for s in summary["splits"].values())
    print(f"Prepared {total} speaker-count mixtures", file=sys.stderr)


if __name__ == "__main__":
    main()
