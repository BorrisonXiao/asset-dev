"""Tests for the speaker-count mixture recipes and their renderer.

The speaker-count label is asserted by construction -- every source stays active
across a shared peak window -- so the tests that matter are the ones that would
catch a placement or rendering change quietly making the label wrong: coverage of
the peak window, the concurrency sweep itself, byte-identical re-rendering, and
the level normalization that stops total energy from encoding the answer.
"""

import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

RECIPE_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "recipes",
    "LibriSpeech",
    "ASR",
    "transformer",
)
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)

import speaker_count_prepare as prep  # noqa: E402
from speaker_count_mixing import (  # noqa: E402
    MIX_SCHEME,
    PEAK_LIMIT,
    activity_intervals,
    clear_recipe_cache,
    concurrency_change_points,
    full_overlap_interval,
    max_concurrency,
    mix_uri,
    parse_mix_uri,
    render_mix_uri,
    render_mixture,
)

SR = 16000


def _intervals_recipe(intervals, duration=5.0):
    return {
        "utt_key": "t",
        "count": len(intervals),
        "duration": duration,
        "sample_rate": SR,
        "sources": [
            {
                "speaker": f"s{i}",
                "wav": "/dev/null",
                "src_offset": 0.0,
                "onset": onset,
                "dur": offset - onset,
                "gain_db": 0.0,
            }
            for i, (onset, offset) in enumerate(intervals)
        ],
    }


# ------------------------------------------------------------------ mix: URIs


def test_mix_uri_round_trips():
    uri = mix_uri("/data/recipes.jsonl", "spkcount-train-000007")
    assert uri.startswith(MIX_SCHEME)
    path, utt_key = parse_mix_uri(uri)
    assert path == "/data/recipes.jsonl"
    assert utt_key == "spkcount-train-000007"


def test_mix_uri_rejects_a_pointer_with_no_key():
    with pytest.raises(ValueError, match="no '#<utt_key>' suffix"):
        parse_mix_uri("mix:/data/recipes.jsonl")
    with pytest.raises(ValueError, match="Not a mixture URI"):
        parse_mix_uri("/data/audio.flac")


# ------------------------------------------------------- concurrency counting


def test_max_concurrency_counts_simultaneous_sources():
    assert max_concurrency(_intervals_recipe([])) == 0
    assert max_concurrency(_intervals_recipe([(0.0, 1.0)])) == 1
    # Fully overlapping.
    assert max_concurrency(_intervals_recipe([(0.0, 2.0), (0.5, 1.5)])) == 2
    # Staircase: three sources but never more than two at once.
    assert (
        max_concurrency(_intervals_recipe([(0.0, 1.0), (0.5, 2.0), (1.5, 3.0)]))
        == 2
    )


def test_touching_intervals_are_not_concurrent():
    # One source ending exactly where the next begins is a hand-off, not
    # overlap; counting it as two would label a sequential pair as simultaneous.
    assert max_concurrency(_intervals_recipe([(0.0, 1.0), (1.0, 2.0)])) == 1


def test_concurrency_change_points_are_sorted_and_deduplicated():
    points = concurrency_change_points(
        _intervals_recipe([(0.0, 1.0), (1.0, 2.0), (0.0, 2.0)])
    )
    assert points == [0.0, 1.0, 2.0]


def test_activity_intervals_expose_speaker_and_bounds():
    recipe = _intervals_recipe([(0.25, 1.75)])
    assert activity_intervals(recipe) == [(0.25, 1.75, "s0")]


# ----------------------------------------------------------------- placement


def test_placement_makes_every_source_span_the_peak_window():
    rng = np.random.default_rng(0)
    for trial in range(200):
        duration = 5.0
        n = int(rng.integers(1, 5))
        src_durations = list(rng.uniform(0.6, 8.0, size=n))
        width = float(rng.uniform(0.5, 4.0))
        placements, (peak_start, peak_end) = prep._place(
            rng, duration, src_durations, width
        )
        for (onset, span), src_dur in zip(placements, src_durations):
            # Inside the mixture window, within the source's own length, and
            # covering the peak -- the three properties the label rests on.
            assert onset >= -1e-9
            assert onset + span <= duration + 1e-9
            assert span <= src_dur + 1e-9
            assert onset <= peak_start + 1e-9
            assert onset + span >= peak_end - 1e-9


def test_placement_yields_the_requested_concurrency():
    rng = np.random.default_rng(1)
    for n in (1, 2, 3, 4):
        for _ in range(50):
            src_durations = list(rng.uniform(0.6, 6.0, size=n))
            placements, _ = prep._place(
                rng, 5.0, src_durations, float(rng.uniform(0.5, 3.0))
            )
            recipe = _intervals_recipe(
                [(onset, onset + span) for onset, span in placements]
            )
            assert max_concurrency(recipe) == n


# --------------------------------------- LibriCount-style placement & zero class


def test_full_overlap_interval_finds_the_longest_full_window():
    # Staircase: two full-concurrency windows, [0.5, 1.0] and [1.5, 2.0].
    recipe = _intervals_recipe([(0.0, 1.0), (0.5, 2.0), (1.5, 3.0)])
    recipe["count"] = 2
    assert full_overlap_interval(recipe) == pytest.approx([0.5, 1.0])
    # Nested intervals: the inner window is the full overlap.
    nested = _intervals_recipe([(0.0, 2.0), (0.5, 1.5)])
    assert full_overlap_interval(nested) == pytest.approx([0.5, 1.5])
    # Touching intervals never overlap, so there is no full-concurrency span.
    touching = _intervals_recipe([(0.0, 1.0), (1.0, 2.0)])
    assert full_overlap_interval(touching) is None
    # A count-0 recipe has nothing to overlap.
    empty = _intervals_recipe([])
    assert full_overlap_interval(empty) is None


def test_random_placement_recipes_verify_the_label(tiny_pool):
    # random_offset places sources independently, so the label is guaranteed by
    # rejection sampling instead of a designed peak window. Every accepted
    # recipe must still realize its count, and its stored peak_interval is the
    # realized full-overlap window.
    speakers = sorted(tiny_pool)
    for count in (1, 2, 3, 4):
        for trial in range(20):
            rng = np.random.default_rng([count, trial])
            recipe = prep.build_recipe(
                rng,
                f"m-r{count}-{trial}",
                tiny_pool,
                speakers,
                count,
                5.0,
                -35.0,
                placement="random_offset",
            )
            assert recipe["placement"] == "random_offset"
            assert max_concurrency(recipe) == count
            peak = recipe["peak_interval"]
            assert peak is not None and peak[1] > peak[0]
            for source in recipe["sources"]:
                # Every source covers the realized peak, or it would not be a
                # full-overlap window.
                assert source["onset"] <= peak[0] + 1e-9
                assert source["onset"] + source["dur"] >= peak[1] - 1e-9


def test_random_placement_sources_are_active_most_of_the_recording():
    # LibriCount sources stay "active for most of the recording": with
    # RANDOM_MIN_SPAN=3.0, a 3 s utterance fills its whole length and a longer
    # one covers at least 3 s of the 5 s clip.
    long_pool = {
        "2000": [{"wav": "/dev/null", "duration": 8.0, "utt": "long-0"}],
        "2001": [{"wav": "/dev/null", "duration": 8.0, "utt": "long-1"}],
    }
    for trial in range(50):
        rng = np.random.default_rng([trial, 7])
        recipe = prep.build_recipe(
            rng,
            f"m-l{trial}",
            long_pool,
            sorted(long_pool),
            2,
            5.0,
            -35.0,
            placement="random_offset",
        )
        for source in recipe["sources"]:
            assert source["dur"] >= 3.0 - 1e-9
            assert source["onset"] >= -1e-9
            assert source["onset"] + source["dur"] <= 5.0 + 1e-9


def test_random_placement_raises_when_overlap_is_impossible(monkeypatch):
    # A pool that can never realize the requested count must fail loudly at
    # prep time, not emit a wrong label.
    pool = {
        str(1000 + i): [
            {"wav": "/dev/null", "duration": 3.0, "utt": f"u{i}"}
        ]
        for i in range(4)
    }
    monkeypatch.setattr(
        prep,
        "_place_random",
        lambda rng, duration, src_durations: [
            (float(i), 1.0) for i in range(len(src_durations))
        ],
    )
    with pytest.raises(RuntimeError, match="failed to realize"):
        prep.build_recipe(
            np.random.default_rng(0),
            "m-hopeless",
            pool,
            sorted(pool),
            3,
            5.0,
            -35.0,
            placement="random_offset",
        )


def test_peak_mode_is_the_default_and_stores_the_designed_window(tiny_pool):
    # The original behavior is untouched: no placement argument means peak
    # mode, and peak_interval is the designed common window.
    recipe = prep.build_recipe(
        np.random.default_rng(3),
        "m-peak",
        tiny_pool,
        sorted(tiny_pool),
        2,
        5.0,
        -35.0,
        short_peak_fraction=0.0,
    )
    assert recipe["placement"] == "peak"
    peak = recipe["peak_interval"]
    width = peak[1] - peak[0]
    # short_peak_fraction=0 means a long window in [2.0, 5.0].
    assert width >= 2.0 - 1e-9


def test_zero_style_mixed_draws_both_floors(tiny_pool):
    styles = set()
    levels = set()
    for trial in range(200):
        recipe = prep.build_recipe(
            np.random.default_rng([11, trial]),
            f"m0-{trial}",
            tiny_pool,
            sorted(tiny_pool),
            0,
            5.0,
            -35.0,
            zero_style="mixed",
        )
        assert recipe["sources"] == []
        styles.add(recipe["zero_style"])
        levels.add(recipe["noise"]["level_dbfs"])
        if recipe["zero_style"] == "noise":
            assert recipe["noise"]["level_dbfs"] == -35.0
        else:
            assert recipe["noise"]["level_dbfs"] == -48.0
    assert styles == {"noise", "silence"}
    assert levels == {-35.0, -48.0}


def test_zero_style_single_modes_pin_the_floor(tiny_pool):
    for style, level in (("noise", -35.0), ("silence", -48.0)):
        for trial in range(10):
            recipe = prep.build_recipe(
                np.random.default_rng([22, trial]),
                f"m-{style}-{trial}",
                tiny_pool,
                sorted(tiny_pool),
                0,
                5.0,
                -35.0,
                zero_style=style,
            )
            assert recipe["zero_style"] == style
            assert recipe["noise"]["level_dbfs"] == level


def test_silence_zero_renders_at_the_measured_libricount_level(tiny_pool):
    # The silence style is not digital zeros (a free class); it is an
    # inaudible floor at the level measured on the HEAR LibriCount count-0
    # examples.
    recipe = prep.build_recipe(
        np.random.default_rng(0),
        "m0-silence",
        tiny_pool,
        sorted(tiny_pool),
        0,
        5.0,
        -35.0,
        zero_style="silence",
    )
    wave = render_mixture(recipe, as_tensor=False)
    rms = float(np.sqrt((wave.astype(np.float64) ** 2).mean()))
    assert rms == pytest.approx(10 ** (-48.0 / 20.0), rel=0.2)


def test_count_positive_recipes_carry_no_zero_style(tiny_pool):
    recipe = prep.build_recipe(
        np.random.default_rng(4),
        "m-2",
        tiny_pool,
        sorted(tiny_pool),
        2,
        5.0,
        -35.0,
        zero_style="mixed",
    )
    assert recipe["zero_style"] is None
    # A floor still accompanies every count > 0 mixture, but it is the SPEECH
    # floor, not the zero-class level passed above: at -35 dBFS against
    # equal-power sources the broadband noise made single-speaker clips read as
    # multi-speaker (count-1 accuracy 0.55 -> 0.98 once lowered). Assert the
    # constant rather than a literal so the two cannot drift apart again.
    assert recipe["noise"]["level_dbfs"] == prep.SPEECH_FLOOR_DBFS


def test_unknown_placement_and_zero_style_are_rejected(tiny_pool):
    with pytest.raises(ValueError, match="placement"):
        prep.build_recipe(
            np.random.default_rng(0),
            "m-x",
            tiny_pool,
            sorted(tiny_pool),
            1,
            5.0,
            -35.0,
            placement="bogus",
        )
    with pytest.raises(ValueError, match="zero style"):
        prep.build_recipe(
            np.random.default_rng(0),
            "m-x",
            tiny_pool,
            sorted(tiny_pool),
            0,
            5.0,
            -35.0,
            zero_style="bogus",
        )


def test_true_speaker_strips_the_chapter():
    # The LibriSpeech manifests key on speaker-CHAPTER; mixing two chapters of
    # one reader would label a one-speaker mixture as two.
    assert prep.true_speaker("2277-149874") == "2277"
    assert prep.true_speaker("2277") == "2277"


# --------------------------------------------------------------- build_recipe


@pytest.fixture
def tiny_pool(tmp_path):
    """Six speakers with one 3 s tone each, written as real 16 kHz WAVs."""
    pool = {}
    for index in range(6):
        speaker = f"{1000 + index}"
        path = tmp_path / f"{speaker}.wav"
        t = np.arange(int(3.0 * SR)) / SR
        wave = 0.2 * np.sin(2 * np.pi * (200 + 50 * index) * t)
        sf.write(str(path), wave.astype(np.float32), SR)
        pool[speaker] = [
            {"wav": str(path), "duration": 3.0, "utt": f"{speaker}-0"}
        ]
    return pool


def test_build_recipe_labels_match_the_intervals_for_every_count(tiny_pool):
    speakers = sorted(tiny_pool)
    for count in range(5):
        for trial in range(20):
            rng = np.random.default_rng([count, trial])
            recipe = prep.build_recipe(
                rng,
                f"m-{count}-{trial}",
                tiny_pool,
                speakers,
                count,
                5.0,
                -35.0,
            )
            assert recipe["count"] == count
            assert max_concurrency(recipe) == count
            assert len(recipe["sources"]) == count
            # Distinct speakers: a repeated speaker would inflate the count.
            assert len({s["speaker"] for s in recipe["sources"]}) == count


def test_zero_count_recipe_has_no_sources_but_keeps_a_noise_floor(tiny_pool):
    # zero_style="noise" pins the classic audible floor; the default is now
    # "mixed", which also admits the silence style (covered below).
    recipe = prep.build_recipe(
        np.random.default_rng(0),
        "m0",
        tiny_pool,
        sorted(tiny_pool),
        0,
        5.0,
        -35.0,
        zero_style="noise",
    )
    assert recipe["sources"] == []
    assert recipe["peak_interval"] is None
    assert recipe["zero_style"] == "noise"
    assert recipe["noise"]["level_dbfs"] == -35.0


def test_build_recipe_crops_stay_inside_the_source_file(tiny_pool):
    speakers = sorted(tiny_pool)
    for trial in range(50):
        rng = np.random.default_rng([99, trial])
        recipe = prep.build_recipe(
            rng, f"m-{trial}", tiny_pool, speakers, 3, 5.0, -35.0
        )
        for source in recipe["sources"]:
            assert source["src_offset"] >= 0.0
            assert source["src_offset"] + source["dur"] <= 3.0 + 1e-6


# ------------------------------------------------------------------ rendering


def _wav(tmp_path, name, seconds=3.0, freq=220.0, sample_rate=SR):
    path = tmp_path / name
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    sf.write(
        str(path),
        (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32),
        sample_rate,
    )
    return str(path)


def _recipe(tmp_path, n_sources=2, target=None, duration=2.0):
    sources = [
        {
            "speaker": f"s{i}",
            "wav": _wav(tmp_path, f"src{i}.wav", freq=220.0 * (i + 1)),
            "src_offset": 0.1 * i,
            "onset": 0.0,
            "dur": duration,
            "gain_db": -3.0 * i,
        }
        for i in range(n_sources)
    ]
    recipe = {
        "utt_key": "r",
        "count": n_sources,
        "duration": duration,
        "sample_rate": SR,
        "noise": {"kind": "gaussian", "level_dbfs": -60.0, "seed": 5},
        "peak_interval": [0.0, duration],
        "sources": sources,
    }
    if target is not None:
        recipe["target_rms_dbfs"] = target
    return recipe


def test_render_length_and_determinism(tmp_path):
    recipe = _recipe(tmp_path)
    a = render_mixture(recipe, as_tensor=False)
    b = render_mixture(recipe, as_tensor=False)
    assert a.shape == (int(2.0 * SR),)
    # Byte-identical: the mixtures are reproducible audio, not fresh randomness
    # at load time, so a rerun of an experiment hears the same thing.
    assert np.array_equal(a, b)


def test_render_returns_a_tensor_by_default(tmp_path):
    import torch

    out = render_mixture(_recipe(tmp_path))
    assert isinstance(out, torch.Tensor)
    assert torch.isfinite(out).all()


def test_level_normalization_hits_the_requested_rms(tmp_path):
    # Without this, total energy grows with the speaker count and a single RMS
    # feature predicts the label well above chance.
    for n in (1, 2, 3):
        wave = render_mixture(
            _recipe(tmp_path, n_sources=n, target=-20.0), as_tensor=False
        )
        rms_db = 20 * np.log10(
            float(np.sqrt((wave.astype(np.float64) ** 2).mean()))
        )
        assert rms_db == pytest.approx(-20.0, abs=0.5)


def test_unnormalized_recipes_still_render(tmp_path):
    # Recipes written before target_rms_dbfs existed must keep working.
    recipe = _recipe(tmp_path, n_sources=2)
    assert "target_rms_dbfs" not in recipe
    assert np.isfinite(render_mixture(recipe, as_tensor=False)).all()


def test_zero_source_mixture_is_noise_not_digital_silence(tmp_path):
    recipe = _recipe(tmp_path, n_sources=0)
    wave = render_mixture(recipe, as_tensor=False)
    rms = float(np.sqrt((wave.astype(np.float64) ** 2).mean()))
    # A free "zero" class would inflate aggregate accuracy without the model
    # having learned anything about speakers.
    assert rms > 0.0
    assert rms == pytest.approx(10 ** (-60.0 / 20.0), rel=0.2)


def test_render_is_peak_limited(tmp_path):
    recipe = _recipe(tmp_path, n_sources=2, target=0.0)
    wave = render_mixture(recipe, as_tensor=False)
    # Clipping would put a label-correlated nonlinearity in the audio.
    assert float(np.abs(wave).max()) <= PEAK_LIMIT + 1e-6


def test_source_shorter_than_its_crop_is_padded_not_shifted(tmp_path):
    recipe = _recipe(tmp_path, n_sources=1, duration=2.0)
    # Ask for more audio than the 3 s file holds past the offset.
    recipe["sources"][0]["src_offset"] = 2.5
    recipe["sources"][0]["dur"] = 2.0
    wave = render_mixture(recipe, as_tensor=False)
    # Length is preserved, so the stored activity interval stays truthful.
    assert wave.shape == (int(2.0 * SR),)


def test_sample_rate_mismatch_is_an_error_not_a_silent_resample(tmp_path):
    recipe = _recipe(tmp_path, n_sources=1)
    recipe["sources"][0]["wav"] = _wav(
        tmp_path, "wrong_rate.wav", sample_rate=8000
    )
    with pytest.raises(ValueError, match="does not resample"):
        render_mixture(recipe, as_tensor=False)


def test_render_mix_uri_resolves_through_a_recipes_file(tmp_path):
    clear_recipe_cache()
    recipe = _recipe(tmp_path, n_sources=2, target=-20.0)
    recipe["utt_key"] = "spkcount-valid-000001"
    path = tmp_path / "recipes.jsonl"
    path.write_text(json.dumps(recipe) + "\n", encoding="utf-8")

    direct = render_mixture(recipe, as_tensor=False)
    viaptr = render_mix_uri(
        mix_uri(str(path), recipe["utt_key"]), as_tensor=False
    )
    assert np.array_equal(direct, viaptr)

    with pytest.raises(KeyError, match="out of sync"):
        render_mix_uri(mix_uri(str(path), "spkcount-valid-999999"))
    clear_recipe_cache()


def test_missing_recipes_file_is_a_clear_error(tmp_path):
    clear_recipe_cache()
    with pytest.raises(FileNotFoundError, match="recipes not found"):
        render_mix_uri(mix_uri(str(tmp_path / "absent.jsonl"), "x"))
