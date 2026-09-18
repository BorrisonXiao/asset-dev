"""Deterministic on-demand rendering of the speaker-count mixtures.

The speaker-count task is the study's fully non-lexical one: how many people are
talking. Its labels are free, which is exactly why it is worth having -- but the
audio has to come from somewhere, and materializing LibriMix-style mixtures would
cost hundreds of GB for a task whose mixtures we want to redraw freely.

So a mixture is stored as a *recipe*, not as audio: which LibriSpeech utterances,
which slice of each, where each one starts in the 5 s window, and at what gain.
``render_mixture`` turns a recipe back into a waveform, bit-for-bit the same
every time, from the LibriSpeech audio already on disk. Only manifests and
recipes are written.

Two consequences worth stating, because they are the reason for the design:

1. **The activity intervals are stored, not inferred.** The boundary-alignment
   analysis scores predicted boundaries against source onsets/offsets and against
   changes in concurrent count. Those labels come straight out of the recipe, so
   they cannot drift from the audio the model actually heard.
2. **The label is true by construction.** Every source's interval contains a
   common "peak window", so the maximum number of concurrently active speakers
   *is* the requested count. :func:`max_concurrency` recomputes it from the
   intervals as a check rather than trusting the generator.
3. **Global energy** is normalized away in the ``hardened`` profile, because
   unnormalized mixtures let a single RMS feature predict the count well above
   chance. The ``libricount`` profile deliberately does NOT normalize it: the
   benchmark mixes at 0 dB SNR per source, so total energy grows with count and
   that cue is part of the task as the field defines it. Which profile produced
   a number must be stated with the number.

A recipe is a plain dict (one JSON line per mixture)::

    {
        "utt_key": "spkcount-train-000001",
        "count": 3,
        "duration": 5.0,
        "sample_rate": 16000,
        "placement": "peak",            # or "random_offset" (LibriCount-style)
        "zero_style": None,             # "noise"/"silence" on count-0 recipes
        "noise": {"kind": "gaussian", "level_dbfs": -35.0, "seed": 1234},
        "target_rms_dbfs": -20.4,
        "peak_interval": [1.82, 2.91],  # designed (peak) or realized (random)
        "sources": [
            {
                "speaker": "2277",
                "wav": "/abs/path.flac",
                "src_offset": 1.23,
                "onset": 0.40,
                "dur": 2.10,
                "gain_db": -3.2,
            },
            ...,
        ],
    }

Nothing here is imported by the ASR-only recipes.
"""

import json
import logging
import os

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# ``wav`` values carrying a recipe pointer rather than a file path, so the
# mixtures travel through the ordinary CSV manifest schema untouched.
MIX_SCHEME = "mix:"

# Rendered mixtures are peak-limited to this amplitude. Summing several sources
# can exceed full scale; clipping would put a nonlinearity in the audio that
# correlates with the label, which is the one artifact this task cannot afford.
PEAK_LIMIT = 0.98

_RECIPE_CACHE = {}


def mix_uri(recipes_path, utt_key):
    """Manifest ``wav`` value pointing at one recipe."""
    return f"{MIX_SCHEME}{os.path.abspath(recipes_path)}#{utt_key}"


def parse_mix_uri(uri):
    """``(recipes_path, utt_key)`` from a ``mix:`` pointer."""
    if not uri.startswith(MIX_SCHEME):
        raise ValueError(f"Not a mixture URI: {uri!r}")
    body = uri[len(MIX_SCHEME) :]
    path, sep, utt_key = body.rpartition("#")
    if not sep:
        raise ValueError(
            f"Mixture URI {uri!r} has no '#<utt_key>' suffix; it must look "
            f"like '{MIX_SCHEME}/path/to/recipes.jsonl#spkcount-train-000001'"
        )
    return path, utt_key


def load_recipes(recipes_path):
    """All recipes in one JSONL, keyed by ``utt_key`` (cached per process).

    Cached because every dataloader worker resolves thousands of pointers into
    the same file, and the file is a few MB at most.
    """
    path = os.path.abspath(recipes_path)
    if path not in _RECIPE_CACHE:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mixture recipes not found: {path}")
        table = {}
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                recipe = json.loads(line)
                table[recipe["utt_key"]] = recipe
        _RECIPE_CACHE[path] = table
        logger.info("Loaded %d mixture recipes from %s", len(table), path)
    return _RECIPE_CACHE[path]


def clear_recipe_cache():
    """Drop the per-process recipe cache (tests, or a regenerated file)."""
    _RECIPE_CACHE.clear()


def activity_intervals(recipe):
    """``[(onset, offset, speaker), ...]`` in seconds, the evaluation labels.

    When a source carries ``active`` -- voice-activity sub-intervals measured
    from the audio, in mixture time -- those are returned instead of the placed
    span. That is the difference between "a source is scheduled here" and "this
    speaker is actually vocalizing here", and it is how LibriCount defines its
    label (max simultaneously vocalizing speakers, via py-webrtcvad). Counting
    placed spans instead makes labels systematically too high, because every
    read utterance contains internal pauses.
    """
    out = []
    for source in recipe["sources"]:
        speaker = source["speaker"]
        active = source.get("active")
        if active:
            out.extend(
                (float(a), float(b), speaker) for a, b in active if b > a
            )
        else:
            onset = float(source["onset"])
            out.append((onset, onset + float(source["dur"]), speaker))
    return out


def energy_vad(wave, sample_rate, frame_ms=20.0, hop_ms=10.0,
               rel_threshold_db=30.0, min_speech_ms=100.0, min_gap_ms=100.0):
    """Voice-activity spans of ``wave`` as ``[(start_s, end_s), ...]``.

    An energy VAD, not ``py-webrtcvad``: neither webrtcvad nor silero is
    installed here, and the sources are clean read speech at high SNR where a
    relative-energy threshold tracks a GMM VAD closely. It is nonetheless an
    approximation, so a generator using it is benchmark-*like*, not
    benchmark-identical; say which was used when reporting.

    The threshold is relative to the clip's own peak frame energy, so it does
    not depend on the absolute level a source was scaled to.
    """
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if wave.shape[0] < frame:
        return []
    n_frames = 1 + (wave.shape[0] - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    energy = (wave[idx].astype(np.float64) ** 2).mean(axis=1)
    peak = energy.max()
    if peak <= 0.0:
        return []
    db = 10.0 * np.log10(np.maximum(energy, peak * 1e-12) / peak)
    voiced = db > -abs(rel_threshold_db)

    spans = []
    start = None
    for i, flag in enumerate(voiced):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(voiced)))

    to_s = lambda f: f * hop / sample_rate  # noqa: E731
    merged = []
    for a, b in spans:
        if merged and to_s(a) - merged[-1][1] < min_gap_ms / 1000.0:
            merged[-1] = (merged[-1][0], to_s(b) + frame / sample_rate)
        else:
            merged.append((to_s(a), to_s(b) + frame / sample_rate))
    return [
        (a, b) for a, b in merged if (b - a) >= min_speech_ms / 1000.0
    ]


def max_concurrency(recipe):
    """Largest number of simultaneously active sources.

    Computed by sweeping interval endpoints rather than by trusting the
    generator, so a placement bug shows up as a label mismatch at prep time
    instead of as an unexplained accuracy ceiling months later.
    """
    events = []
    for onset, offset, _ in activity_intervals(recipe):
        events.append((onset, 1))
        # Ends sort before starts at equal time: two sources that merely touch
        # are not concurrent.
        events.append((offset, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    running = best = 0
    for _, delta in events:
        running += delta
        best = max(best, running)
    return best


def full_overlap_interval(recipe):
    """Longest interval on which every source is simultaneously active.

    ``placement: peak`` recipes design a common peak window, so they store that
    window directly. ``placement: random_offset`` recipes (LibriCount-style)
    guarantee the label by rejection sampling instead, and their stored
    ``peak_interval`` is this realized interval -- which always exists, since a
    recipe whose sources never all overlap is never accepted.

    Returns ``[start, end]`` in seconds, or ``None`` when the recipe has no
    sources.
    """
    count = recipe.get("count") or 0
    if count <= 0 or not recipe.get("sources"):
        return None
    events = []
    for onset, offset, _ in activity_intervals(recipe):
        events.append((onset, 1))
        events.append((offset, -1))
    events.sort(key=lambda item: (item[0], item[1]))  # ends before starts
    # Concurrency on the open segment after each event time. Events at one
    # instant are applied as a batch, so a start compensating an end at the
    # same time does not split a full-overlap interval.
    running = 0
    cursor = 0
    segments = []
    while cursor < len(events):
        t = events[cursor][0]
        while cursor < len(events) and events[cursor][0] == t:
            running += events[cursor][1]
            cursor += 1
        if cursor < len(events) and events[cursor][0] > t:
            segments.append((t, events[cursor][0], running))
    best = None
    best_len = 0.0
    start = end = None
    for seg_start, seg_end, level in segments:
        if level == count:
            if start is None:
                start = seg_start
            end = seg_end
            if end - start > best_len:
                best_len = end - start
                best = [start, end]
        else:
            start = end = None
    return best


def concurrency_change_points(recipe):
    """Times where the number of active speakers changes, deduplicated.

    The second half of the boundary-alignment analysis: a segmenter that tracks
    speaker structure should place boundaries at these instants, not only at
    source onsets.
    """
    times = set()
    for onset, offset, _ in activity_intervals(recipe):
        times.add(round(onset, 6))
        times.add(round(offset, 6))
    return sorted(times)


def _read_source(source, sample_rate):
    """One source's cropped, gain-scaled waveform as float32."""
    path = source["wav"]
    info = sf.info(path)
    if info.samplerate != sample_rate:
        raise ValueError(
            f"{path}: {info.samplerate} Hz, but the mixture is {sample_rate} Hz. "
            "The mixer does not resample; prepare the pool at the target rate."
        )
    start = int(round(float(source["src_offset"]) * sample_rate))
    frames = int(round(float(source["dur"]) * sample_rate))
    wave, _ = sf.read(
        path, start=start, frames=frames, dtype="float32", always_2d=False
    )
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if wave.shape[0] < frames:
        # A source shorter than its planned crop is padded rather than shifted:
        # moving it would invalidate the stored activity interval.
        wave = np.pad(wave, (0, frames - wave.shape[0]))
    # Equal-power mixing (LibriCount's "0 dB SNR"): each source is brought to a
    # common RMS *before* its gain, so no speaker dominates by virtue of how
    # loudly the original utterance happened to be recorded. Absent the field,
    # the raw source level is kept, which is the hardened profile.
    per_source = source.get("norm_rms_dbfs")
    if per_source is not None:
        rms = float(np.sqrt((wave.astype(np.float64) ** 2).mean()))
        if rms > 0.0:
            wave = wave * (float(10.0 ** (float(per_source) / 20.0)) / rms)
    return wave * float(10.0 ** (float(source["gain_db"]) / 20.0))


def render_mixture(recipe, as_tensor=True):
    """Render one recipe to a waveform.

    Arguments
    ---------
    recipe : dict
        A mixture recipe (see the module docstring).
    as_tensor : bool
        Return a ``torch.Tensor`` (what the dataio pipeline wants) rather than a
        numpy array.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Mono waveform of ``duration * sample_rate`` samples.
    """
    sample_rate = int(recipe.get("sample_rate", 16000))
    total = int(round(float(recipe["duration"]) * sample_rate))
    mixture = np.zeros(total, dtype=np.float32)

    for source in recipe["sources"]:
        wave = _read_source(source, sample_rate)
        onset = int(round(float(source["onset"]) * sample_rate))
        end = min(total, onset + wave.shape[0])
        if end > onset:
            mixture[onset:end] += wave[: end - onset]

    # Level-normalize the speech sum BEFORE the noise floor is added. Without
    # this, total energy grows with the number of speakers and a single RMS
    # feature predicts the label: measured 48% accuracy against a 20% chance
    # baseline (44% against 25% on counts 1-4). The task is supposed to be about
    # speaker structure, so the one cue that trivially encodes the answer is
    # removed. Local speech density still varies with count -- that is a real
    # cue for real speaker counting and is deliberately left alone.
    target = recipe.get("target_rms_dbfs")
    if target is not None and recipe["sources"]:
        rms = float(np.sqrt((mixture.astype(np.float64) ** 2).mean()))
        if rms > 0.0:
            mixture *= float(10.0 ** (float(target) / 20.0)) / rms

    noise = recipe.get("noise")
    if noise:
        mixture += _render_noise(noise, total)

    peak = float(np.abs(mixture).max()) if total else 0.0
    if peak > PEAK_LIMIT:
        mixture *= PEAK_LIMIT / peak

    if as_tensor:
        return torch.from_numpy(mixture)
    return mixture


def _render_noise(noise, total):
    """Background floor, so a zero-speaker mixture is not digital silence.

    Digital silence would make the ``zero`` class free, and a free class
    inflates aggregate accuracy without the model having learned anything about
    speakers. The floor is drawn from the recipe's own seed, so it is part of the
    reproducible audio rather than fresh randomness at load time.
    """
    kind = noise.get("kind", "gaussian")
    if kind != "gaussian":
        raise ValueError(f"Unknown mixture noise kind {kind!r}")
    level = float(10.0 ** (float(noise["level_dbfs"]) / 20.0))
    rng = np.random.default_rng(int(noise["seed"]))
    return rng.standard_normal(total).astype(np.float32) * level


def render_mix_uri(uri, as_tensor=True):
    """Render the mixture a ``mix:`` manifest path points at."""
    recipes_path, utt_key = parse_mix_uri(uri)
    table = load_recipes(recipes_path)
    if utt_key not in table:
        raise KeyError(
            f"{utt_key!r} is not in {recipes_path}. The manifest and the recipe "
            "file are out of sync; regenerate both with speaker_count_prepare.py."
        )
    return render_mixture(table[utt_key], as_tensor=as_tensor)


__all__ = [
    "MIX_SCHEME",
    "activity_intervals",
    "clear_recipe_cache",
    "concurrency_change_points",
    "full_overlap_interval",
    "load_recipes",
    "max_concurrency",
    "mix_uri",
    "parse_mix_uri",
    "render_mix_uri",
    "render_mixture",
]
