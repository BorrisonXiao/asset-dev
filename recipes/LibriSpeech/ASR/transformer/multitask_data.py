"""Task schema, merged manifests, and task-homogeneous batching for multi-task runs.

Nothing here is imported by the ASR-only recipes; ``train_speechllm.py`` and
``train_speechllm_with_segmenter.py`` keep their single-task data path exactly
as it was. The pieces are:

``TASK_SPECS``
    The stable task table. ``task_id`` indexes the segmenter FiLM embedding and
    the task-routed decoder adapters, so the order must never change once a
    checkpoint exists. ``asr`` is task 0 precisely so an ASR warm start is the
    identity at step zero. A task either generates text (``asr``, ``st_en_de``)
    or carries a closed ``candidates`` label set that is ranked by
    log-likelihood (``emotion``, ``speaker_count``, ``intent``).

``build_multitask_manifest``
    Merges per-source CSVs (CoVoST ASR/ST views plus a LibriSpeech replay
    manifest) into one manifest carrying ``task_id``/``source_key`` columns.

``MultiSourceDynamicBatchSampler``
    Duration-budgeted batches drawn from ONE source per step, so every batch is
    task homogeneous. That keeps adapter routing unambiguous and keeps GRPO from
    ever comparing an ASR rollout reward against an ST rollout reward.
"""

import csv
import logging
import os
import random
from typing import NamedTuple

import numpy as np
import torch
from speaker_count_mixing import MIX_SCHEME, render_mix_uri

import speechbrain as sb
from speechbrain.dataio.sampler import DynamicBatchSampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task schema
# ---------------------------------------------------------------------------


class TaskSpec(NamedTuple):
    """One row of the task table.

    Attributes
    ----------
    task_id : int
        Indexes the segmenter FiLM embedding and the task-routed decoder
        adapters. The order must never change once a checkpoint exists.
    prompt : str
        Canonical decoder prompt. Wording is fixed for the core study (no
        paraphrase generalization claim is being made).
    candidates : tuple of str
        Empty for generative tasks. For classification tasks it is the closed
        label set, scored by candidate log-likelihood rather than free decoding
        (see ``candidate_scoring``), which is what makes an invalid answer
        impossible and removes the unequal-label-length confound.

    ``TaskSpec`` is a NamedTuple so ``spec[0]``/``spec[1]`` keep working for
    any code written against the original ``(task_id, prompt)`` tuples.
    """

    task_id: int
    prompt: str
    candidates: tuple = ()


# Six-way CREMA-D emotion. The label strings are the survey's, not the corpus
# codes (ANG/DIS/...): the decoder scores English words, not abbreviations.
EMOTION_CANDIDATES = (
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
)

# Speaker count carries the FULL 0-10 vocabulary even while the pilot runs at
# 0-4, so growing the pilot into the full task changes no task id and no
# checkpoint shape. Restrict the ACTIVE set per run via ``candidate_subsets``;
# the scorer only ever ranks the subset it is handed.
SPEAKER_COUNT_CANDIDATES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)

SPEAKER_COUNT_PILOT_CANDIDATES = SPEAKER_COUNT_CANDIDATES[:5]

# The 31 Fluent Speech Commands intents, each the non-empty slots of the
# (action, object, location) tuple joined in that order. Generated from
# ``data/train_data.csv`` and sorted, so the index order is reproducible.
INTENT_CANDIDATES = (
    "activate lamp",
    "activate lights",
    "activate lights bedroom",
    "activate lights kitchen",
    "activate lights washroom",
    "activate music",
    "bring juice",
    "bring newspaper",
    "bring shoes",
    "bring socks",
    "change language",
    "change language Chinese",
    "change language English",
    "change language German",
    "change language Korean",
    "deactivate lamp",
    "deactivate lights",
    "deactivate lights bedroom",
    "deactivate lights kitchen",
    "deactivate lights washroom",
    "deactivate music",
    "decrease heat",
    "decrease heat bedroom",
    "decrease heat kitchen",
    "decrease heat washroom",
    "decrease volume",
    "increase heat",
    "increase heat bedroom",
    "increase heat kitchen",
    "increase heat washroom",
    "increase volume",
)

# task_name -> TaskSpec. Ids 0 and 1 are frozen by every existing multi-task
# checkpoint; 2-4 are the non-ASR tasks from the 2026-08-31 survey. ``st_en_de``
# keeps id 1 even in runs that never sample it, so an ST checkpoint and a
# non-ASR checkpoint address the same slots.
TASK_SPECS = {
    "asr": TaskSpec(0, "Transcribe speech to text."),
    "st_en_de": TaskSpec(1, "Translate the English speech into German."),
    "emotion": TaskSpec(
        2,
        "Identify the emotion in the speaker's voice.",
        EMOTION_CANDIDATES,
    ),
    "speaker_count": TaskSpec(
        3,
        "How many people are speaking?",
        SPEAKER_COUNT_CANDIDATES,
    ),
    "intent": TaskSpec(
        4,
        "Identify the requested action.",
        INTENT_CANDIDATES,
    ),
}

NUM_TASKS = len(TASK_SPECS)

MULTITASK_FIELDS = [
    "ID",
    "duration",
    "wav",
    "spk_id",
    "wrd",
    "task_name",
    "task_id",
    "source_key",
    "source_dataset",
    "source_lang",
    "target_lang",
    "utt_key",
]


def task_id_of(task_name):
    """Stable integer id for a task name."""
    if task_name not in TASK_SPECS:
        raise KeyError(
            f"Unknown task {task_name!r}; known tasks: {sorted(TASK_SPECS)}"
        )
    return TASK_SPECS[task_name][0]


def prompt_of(task_name):
    """Canonical decoder prompt for a task name."""
    if task_name not in TASK_SPECS:
        raise KeyError(
            f"Unknown task {task_name!r}; known tasks: {sorted(TASK_SPECS)}"
        )
    return TASK_SPECS[task_name][1]


def candidates_of(task_name):
    """Closed label set for a task, or ``()`` if the task is generative."""
    if task_name not in TASK_SPECS:
        raise KeyError(
            f"Unknown task {task_name!r}; known tasks: {sorted(TASK_SPECS)}"
        )
    return tuple(TASK_SPECS[task_name].candidates)


def is_classification(task_name):
    """True when a task is answered by ranking a closed label set."""
    return bool(candidates_of(task_name))


def resolve_candidates(task_name, subsets=None):
    """Active label set for a task, optionally restricted by run config.

    ``subsets`` maps ``task_name -> sequence of labels`` and exists for the
    survey's staged speaker-count task: the pilot ranks ``zero``-``four`` while
    the task table keeps the full ``zero``-``ten`` vocabulary, so widening the
    pilot later changes no task id and no checkpoint shape. Every requested
    label must appear in the task's canonical set, in order to keep a typo from
    silently becoming a label the model can never be right about.
    """
    canonical = candidates_of(task_name)
    if not subsets or task_name not in subsets or subsets[task_name] is None:
        return canonical
    requested = tuple(subsets[task_name])
    if not requested:
        raise ValueError(f"Empty candidate subset for task {task_name!r}")
    unknown = [c for c in requested if c not in canonical]
    if unknown:
        raise ValueError(
            f"Candidate subset for {task_name!r} names labels outside the "
            f"task table: {unknown}. Known labels: {list(canonical)}"
        )
    return requested


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def _read_rows(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def librispeech_replay_rows(
    csv_path, data_folder, source_key="librispeech_asr"
):
    """Project a plain LibriSpeech manifest onto the multi-task schema.

    The LibriSpeech CSVs store ``$data_root/...`` paths resolved at load time by
    SpeechBrain's ``replacements``. A merged multi-task manifest mixes corpora
    with different roots, so the path is made absolute here and the merged
    manifest carries no ``$data_root`` dependency at all.
    """
    rows = []
    for row in _read_rows(csv_path):
        wav = row["wav"].replace("$data_root/", f"{data_folder}/")
        wav = os.path.normpath(wav)
        rows.append(
            {
                "ID": f"asr-{row['ID']}",
                "duration": row["duration"],
                "wav": wav,
                "spk_id": row.get("spk_id", ""),
                "wrd": row["wrd"],
                "task_name": "asr",
                "task_id": task_id_of("asr"),
                "source_key": source_key,
                "source_dataset": "librispeech",
                "source_lang": "en",
                "target_lang": "en",
                "utt_key": row["ID"],
            }
        )
    return rows


# Fixed so a subset is reproducible across runs and machines. It is not a
# tuning knob; changing it changes which utterances every held-out subset
# contains, and therefore which checkpoints get selected.
_SUBSET_SEED = 20260904


def _stride_subset(rows, limit):
    """Deterministic subset of ``rows``, immune to periodic manifest order.

    This used to take evenly spaced indices (``rows[int(i * total / limit)]``),
    which silently aliases whenever the manifest is periodic. The speaker-count
    manifests are interleaved by label with period 5; a 600-of-2000 subset
    strides by 3.33 and lands only on residues {0, 3, 1}, so the "balanced"
    validation set contained **three** of the five classes and never a single
    ``two`` or ``four``. Validation accuracy then measured an easier 3-way
    problem (0.90) than the 5-way test set (0.69) and checkpoint selection was
    driven by it. A 1000-of-2000 subset strides by exactly 2 and happens to
    cover all five, which is why the multi-task runs never showed it.

    A seeded permutation is just as reproducible and cannot alias against any
    ordering. Order within the returned subset follows the original manifest,
    so downstream sorting and bucketing behave as before.
    """
    total = len(rows)
    if limit >= total:
        return list(rows)
    picked = sorted(
        random.Random(_SUBSET_SEED).sample(range(total), int(limit))
    )
    return [rows[i] for i in picked]


def build_multitask_manifest(
    sources, out_csv, data_folder=None, max_duration=None
):
    """Merge per-source manifests into one multi-task CSV.

    Arguments
    ---------
    sources : list of dict
        Each entry needs ``csv`` and ``source_key``; set ``kind: librispeech``
        for a plain LibriSpeech manifest that must be projected onto the
        multi-task schema first, and ``limit`` to take a deterministic evenly
        spaced subset (used to keep held-out decoding affordable).
    out_csv : str
        Destination manifest path.
    data_folder : str, optional
        LibriSpeech root used to resolve ``$data_root`` in replay manifests.
    max_duration : float, optional
        Drop utterances longer than this many seconds, logging each one. The
        autoregressive boundary policy has a hard position limit, and a single
        over-long utterance raises deep inside the rollout -- which, if it only
        appears in a test split, surfaces hours into a run at final evaluation.
        Filtering here turns that into a loud, cheap, up-front decision.

    Returns
    -------
    dict
        ``source_key`` -> row count, for launcher-side validation.
    """
    merged = []
    counts = {}
    for source in sources:
        key = source["source_key"]
        path = source["csv"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing manifest for {key}: {path}")
        limit = source.get("limit")
        if source.get("kind") == "librispeech":
            if not data_folder:
                raise ValueError(
                    "A librispeech source requires data_folder to resolve "
                    "$data_root into an absolute path."
                )
            rows = librispeech_replay_rows(path, data_folder, source_key=key)
        else:
            rows = _read_rows(path)
            for row in rows:
                # The manifest on disk already carries the schema; the source
                # key is re-stamped so one CSV can serve several mixes.
                row["source_key"] = key
        if not rows:
            raise ValueError(f"Source {key} is empty: {path}")
        if max_duration is not None:
            kept, dropped = [], []
            for row in rows:
                if float(row["duration"]) > float(max_duration):
                    dropped.append(row)
                else:
                    kept.append(row)
            if dropped:
                logger.warning(
                    "Source %s: dropping %d of %d utterances longer than %.1f s "
                    "(the boundary policy's position limit). Longest dropped: "
                    "%s. This changes the evaluated set and must be disclosed.",
                    key,
                    len(dropped),
                    len(rows),
                    float(max_duration),
                    ", ".join(
                        f"{r['utt_key']} ({float(r['duration']):.1f}s)"
                        for r in sorted(
                            dropped,
                            key=lambda r: -float(r["duration"]),
                        )[:5]
                    ),
                )
            rows = kept
        if limit is not None and len(rows) > int(limit):
            # Evenly spaced over the (utt_key-sorted) manifest. Two views of the
            # same corpus therefore select the SAME underlying utterances, which
            # is what makes the paired same-audio ASR/ST analysis possible.
            rows = _stride_subset(rows, int(limit))
        counts[key] = len(rows)
        merged.extend(rows)

    seen = set()
    for row in merged:
        if row["ID"] in seen:
            raise ValueError(
                f"Duplicate utterance ID in merged manifest: {row['ID']}. "
                "IDs must be unique across task views."
            )
        seen.add(row["ID"])

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MULTITASK_FIELDS)
        writer.writeheader()
        for row in merged:
            writer.writerow({k: row[k] for k in MULTITASK_FIELDS})
    logger.info(
        "Wrote multi-task manifest %s: %s (total %d)",
        out_csv,
        ", ".join(f"{k}={v}" for k, v in counts.items()),
        len(merged),
    )
    return counts


# ---------------------------------------------------------------------------
# Task-homogeneous batching
# ---------------------------------------------------------------------------


class _LengthOnlyDataset(list):
    """Stand-in whose only use inside DynamicBatchSampler is ``len()``.

    ``DynamicBatchSampler`` requires a ``DynamicItemDataset`` only when it has to
    call ``length_func``. Supplying ``lengths_list`` bypasses that path entirely,
    which lets one sampler be built per source over a subset of a merged dataset
    while reusing the validated bucketing logic verbatim.
    """


class MultiSourceDynamicBatchSampler(torch.utils.data.Sampler):
    """Duration-budgeted batches drawn from one source per step.

    Each source gets its own ``DynamicBatchSampler`` over its slice of the
    merged dataset, so batches keep the same padding-efficient bucketing as the
    single-task recipes. At every step one source is drawn from ``probabilities``
    and its next batch is emitted, translated back to merged-dataset indices.
    A source that runs out is regenerated (reshuffled) and keeps supplying
    batches, so the small corpus simply repeats rather than starving the mix.

    Arguments
    ---------
    dataset : DynamicItemDataset
        The merged multi-task dataset.
    source_keys : list of str
        Sampling sources, in a fixed order.
    probabilities : list of float
        Sampling probability per source; normalized internally.
    max_batch_length : float
        Per-batch audio-second budget, as in the single-task recipes.
    num_buckets : int
        Length buckets per source.
    max_batch_ex : int, optional
        Hard cap on examples per batch.
    min_batch_ex : int
        Minimum examples per batch.
    batches_per_epoch : int, optional
        Batches per data pass. Defaults to the point at which the first source
        would be exhausted given its probability, which keeps an "epoch" a sane
        size; runs are step-limited regardless. Ignored when ``exhaustive``.
    seed : int
        Base RNG seed.
    epoch : int
        Starting epoch.
    length_key : str
        Dataset annotation key holding the example duration.
    exhaustive : bool
        Validation/test mode: emit every batch of every source exactly once, in
        a fixed interleaved order, ignoring ``probabilities``. Sampling a
        validation set probabilistically would make each task's metric
        denominator random from one validation point to the next, so held-out
        scores could not be compared across steps.
    """

    def __init__(
        self,
        dataset,
        source_keys,
        probabilities,
        max_batch_length,
        num_buckets=200,
        max_batch_ex=None,
        min_batch_ex=1,
        batches_per_epoch=None,
        seed=42,
        epoch=0,
        length_key="duration",
        exhaustive=False,
    ):
        if len(source_keys) != len(probabilities):
            raise ValueError(
                "source_keys and probabilities must have the same length, got "
                f"{len(source_keys)} and {len(probabilities)}"
            )
        if not source_keys:
            raise ValueError("At least one source is required")
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if np.any(probabilities <= 0):
            raise ValueError(
                f"Sampling probabilities must be positive, got {probabilities}"
            )
        self.source_keys = list(source_keys)
        self.probabilities = probabilities / probabilities.sum()
        self.exhaustive = bool(exhaustive)

        annotations = dataset.data
        data_ids = dataset.data_ids
        # Merged-dataset indices and durations, grouped by sampling source.
        self._source_indices = {key: [] for key in self.source_keys}
        source_lengths = {key: [] for key in self.source_keys}
        for index, data_id in enumerate(data_ids):
            row = annotations[data_id]
            key = row["source_key"]
            if key not in self._source_indices:
                continue
            self._source_indices[key].append(index)
            source_lengths[key].append(float(row[length_key]))

        for key in self.source_keys:
            if not self._source_indices[key]:
                raise ValueError(
                    f"Source {key!r} has no rows in the merged manifest"
                )

        self._seed = int(seed)
        self._epoch = int(epoch)
        self._samplers = {}
        for offset, key in enumerate(self.source_keys):
            lengths = source_lengths[key]
            self._samplers[key] = DynamicBatchSampler(
                _LengthOnlyDataset([0] * len(lengths)),
                max_batch_length=max_batch_length,
                num_buckets=num_buckets,
                lengths_list=lengths,
                shuffle=not self.exhaustive,
                batch_ordering=("ascending" if self.exhaustive else "random"),
                max_batch_ex=max_batch_ex,
                min_batch_ex=min_batch_ex,
                # Distinct streams so two sources never share a permutation.
                seed=self._seed + 1000 * (offset + 1),
                epoch=self._epoch,
            )

        # Reverse map for O(1) source lookup of an emitted batch.
        self._source_of_index = {}
        for key, indices in self._source_indices.items():
            for index in indices:
                self._source_of_index[index] = key

        self._source_batch_counts = {
            key: len(self._samplers[key]) for key in self.source_keys
        }
        if self.exhaustive:
            self._batches_per_epoch = sum(self._source_batch_counts.values())
        elif batches_per_epoch is not None:
            self._batches_per_epoch = int(batches_per_epoch)
        else:
            # Stop at the first source that would be exhausted at its own rate.
            self._batches_per_epoch = int(
                min(
                    self._source_batch_counts[key] / p
                    for key, p in zip(self.source_keys, self.probabilities)
                )
            )
        self._batches_per_epoch = max(1, self._batches_per_epoch)

        logger.info(
            "MultiSourceDynamicBatchSampler: %s; batches_per_epoch=%d",
            ", ".join(
                f"{key}: n={len(self._source_indices[key])} "
                f"batches={self._source_batch_counts[key]} p={p:.3f}"
                for key, p in zip(self.source_keys, self.probabilities)
            ),
            self._batches_per_epoch,
        )

    def set_epoch(self, epoch):
        """Reshuffle every source stream for a new data pass."""
        if self.exhaustive:
            return
        self._epoch = int(epoch)
        for offset, key in enumerate(self.source_keys):
            self._samplers[key].set_epoch(self._epoch + 100 * (offset + 1))

    def source_of_batch(self, batch):
        """Source key for an emitted batch (all members share it)."""
        try:
            return self._source_of_index[batch[0]]
        except KeyError:
            raise KeyError(f"Index {batch[0]} belongs to no known source")

    def __iter__(self):
        if self.exhaustive:
            yield from self._iter_exhaustive()
            return
        generator = np.random.default_rng(self._seed + self._epoch)
        iterators = {key: iter(self._samplers[key]) for key in self.source_keys}
        for _ in range(self._batches_per_epoch):
            choice = int(
                generator.choice(len(self.source_keys), p=self.probabilities)
            )
            key = self.source_keys[choice]
            try:
                local_batch = next(iterators[key])
            except StopIteration:
                # DynamicBatchSampler reshuffles on exhaustion; restart it so a
                # small source repeats instead of ending the pass early.
                iterators[key] = iter(self._samplers[key])
                local_batch = next(iterators[key])
            mapping = self._source_indices[key]
            yield [mapping[i] for i in local_batch]

    def _iter_exhaustive(self):
        """Every batch of every source exactly once, sources interleaved."""
        streams = []
        for key in self.source_keys:
            mapping = self._source_indices[key]
            streams.append(
                [[mapping[i] for i in batch] for batch in self._samplers[key]]
            )
        cursors = [0] * len(streams)
        remaining = sum(len(stream) for stream in streams)
        while remaining:
            for index, stream in enumerate(streams):
                if cursors[index] < len(stream):
                    yield stream[cursors[index]]
                    cursors[index] += 1
                    remaining -= 1

    def __len__(self):
        return self._batches_per_epoch


# ---------------------------------------------------------------------------
# Dataio
# ---------------------------------------------------------------------------


def _load_manifest(csv_path):
    """Read a merged manifest without SpeechBrain's ``$variable`` substitution.

    ``DynamicItemDataset.from_csv`` expands ``$name`` in every field via the
    ``replacements`` mechanism used for ``$data_root``. Merged multi-task
    manifests already carry absolute paths, so that expansion buys nothing --
    and it actively breaks on real text: MuST-C German contains prices like
    "$9 pro Pfund", which the loader reads as an undefined variable and raises
    ``KeyError: '9'``. Any currency amount in any language would do the same.
    """
    with open(csv_path, encoding="utf-8", newline="") as stream:
        data = {}
        for row in csv.DictReader(stream, skipinitialspace=True):
            data_id = row.pop("ID", None)
            if data_id is None:
                raise KeyError(f"{csv_path} has no ID column")
            if data_id in data:
                raise ValueError(f"Duplicate id in {csv_path}: {data_id}")
            # The samplers and `filtered_sorted` compare durations numerically;
            # left as strings, "10.5" would sort before "9.2".
            row["duration"] = float(row["duration"])
            data[data_id] = row
    return data


def multitask_dataio_prepare(hparams, tokenizer, csv_path, sorting="random"):
    """Build one multi-task DynamicItemDataset with per-task prompts.

    The text pipeline mirrors ``train_speechllm.dataio_prepare`` exactly, except
    the prompt is selected by ``task_name`` instead of a single global string,
    and ``task_id`` is exposed so the trainer can route FiLM and the decoder
    adapters. Batches are task homogeneous, so every element of a batch shares
    one prompt length -- the invariant ``_run_decoder`` already relies on.
    """
    bos_index = hparams["bos_index"]
    eos_index = hparams["eos_index"]
    start_of_audio_index = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
    end_of_audio_index = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")

    prompt_ids = {}
    for task_name in TASK_SPECS:
        prompt_ids[task_name] = (
            tokenizer(
                prompt_of(task_name),
                return_tensors="pt",
                add_special_tokens=False,
            )
            .input_ids.view(-1)
            .tolist()
        )
        logger.info(
            "Task %s (id=%d) prompt %r -> %d tokens",
            task_name,
            task_id_of(task_name),
            prompt_of(task_name),
            len(prompt_ids[task_name]),
        )

    @sb.utils.data_pipeline.takes("wrd", "task_name")
    @sb.utils.data_pipeline.provides(
        "wrd",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
        "prompt_len",
        "task_id",
        "task_name",
    )
    def text_pipeline(wrd, task_name):
        """Tokenize the target text behind this task's canonical prompt."""
        yield wrd
        tokens_list = tokenizer(wrd, add_special_tokens=False).input_ids
        yield tokens_list
        task_prompt = prompt_ids[task_name]
        yield torch.LongTensor(
            [start_of_audio_index]
            + [end_of_audio_index]
            + task_prompt
            + ([bos_index] if bos_index is not None else [])
            + tokens_list
        )
        yield torch.LongTensor(tokens_list + [eos_index])
        yield torch.LongTensor(tokens_list)
        yield len([start_of_audio_index] + [end_of_audio_index] + task_prompt)
        # A plain int, like prompt_len: PaddedBatch pads tensor-valued keys, so
        # a 0-dim tensor here would be routed into the padding path. Ints are
        # stacked into a (B,) tensor instead.
        yield task_id_of(task_name)
        yield task_name

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load one utterance; the merged manifest stores absolute paths.

        A ``mix:`` value is a synthesis recipe rather than a file. The
        speaker-count mixtures are rendered on demand from LibriSpeech audio
        already on disk -- deterministically, from the recipe's own seeds -- so
        the task costs a few MB of manifests instead of a materialized LibriMix.
        """
        if wav.startswith(MIX_SCHEME):
            return render_mix_uri(wav)
        return sb.dataio.dataio.read_audio(wav)

    output_keys = [
        "id",
        "sig",
        "wrd",
        "tokens_bos",
        "tokens_eos",
        "tokens",
        "prompt_len",
        "task_id",
        "task_name",
        "utt_key",
    ]

    dataset = sb.dataio.dataset.DynamicItemDataset(
        _load_manifest(csv_path),
        dynamic_items=[text_pipeline, audio_pipeline],
        output_keys=output_keys,
    )
    if sorting == "ascending":
        dataset = dataset.filtered_sorted(sort_key="duration")
    elif sorting == "descending":
        dataset = dataset.filtered_sorted(sort_key="duration", reverse=True)
    elif sorting != "random":
        raise ValueError(f"Unknown sorting {sorting!r}")
    return dataset
