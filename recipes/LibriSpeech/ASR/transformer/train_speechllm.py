#!/usr/bin/env python3
"""Recipe for training a SpeechLLM ASR system with LibriSpeech.

The system employs a speech SSL encoder, and a pre-trained LLM decoder.
The speech features are projected to the LLM embedding space using a linear layer projection.
The LLM is trained used the cross-entropy loss on the text tokens excluding the prompt.

An input sequence is typically constructed like this:
 <|start_of_audio|> audio features <|end_of_audio|> <prompt> <bos> <text> <eos>

This script supports both offline and online SSL/cached features mode.
To extract the features offline, run the `extract_ssl_feats.py` script, and use
the correct yaml file for this script.

python extract_ssl_feats.py hparams/extract_ssl_feats.yaml
    --data_folder path/to/LibriSpeech \
    --output_folder path/to/feats_cache \
    --ssl_hub path/to/wavlm-large \
    --feats_cache_dir path/to/feats_cache
    ...other_hparams...

python train_speechllm.py hparams/speechllm_ssl_feats.yaml
    --feats_cache_dir path/to/feats_cache \
    ...other_hparams...

This script also supports fixed-segmentation baselines via `boundary_source`
(see hparams/speechllm_fixed_pooling.yaml), which replace the fixed
feat_downsampler with mean-pooling between boundaries:
 - none (default): original behavior (e.g. ConcatDownsampler)
 - fixed_rate: mean-pool every `fixed_rate_k` frames
 - alignment: mean-pool between externally provided boundaries, loaded from
   per-utterance `<id>.pt` files in `boundary_target_dir` (per-frame {0,1}
   labels at the encoder frame rate)

Authors
-------
 * Adel Moumen, 2025
"""

import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from segment_pooling import (
    apply_blank_mode,
    fixed_rate_boundary_targets,
    lengths_to_padding_mask,
    mean_pool_segments,
    padding_mask_to_lengths,
    segment_separator_mask,
)
from speechbrain.integrations.hdf5.cached_item import CachedHDF5DynamicItem
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def get_multimodal_attention_mask(wav, wav_lens, txt, txt_lens, device):
    """Create attention mask for multimodal sequence.

    Arguments
    ---------
    wav : torch.Tensor
        Audio features tensor of shape (batch_size, L_audio, ...)
    wav_lens : torch.Tensor
        Relative lengths of audio features, shape (batch_size,)
    txt : torch.Tensor
        Text embeddings tensor of shape (batch_size, txt_len, ...)
        This is txt_embds which includes: [start_of_audio, end_of_audio, prompt, bos, text]
    txt_lens : torch.Tensor
        Relative lengths of text tokens, shape (batch_size,)
    device : torch.device
        Device to create the mask on

    Returns
    -------
    attention_mask : torch.Tensor
        Boolean attention mask of shape (batch_size, L_audio + txt_len).

        Important
        ---------
        The actual multimodal embedding order in this recipe is:

            [start_of_audio] + [audio_feats] + [end_of_audio + prompt + bos + text]

        i.e., the first text token (<|start_of_audio|>) is placed *before* audio.
        Therefore, we must build the mask with the same layout:
            position 0              -> <|start_of_audio|>
            positions [1 : 1+L_audio] -> audio feats
            positions [1+L_audio : ]  -> remaining text tokens (txt[:, 1:])
    """
    batch_size = wav.size(0)
    wav_len = wav.size(1)
    txt_len = txt.size(1)
    # Total length matches multimodal_embds: 1 (start token) + L_audio + (txt_len - 1)
    total_len = wav_len + txt_len
    attention_mask = torch.zeros(
        batch_size, total_len, dtype=torch.bool, device=device
    )
    for i in range(batch_size):
        # Match SpeechBrain convention (see S2SGreedySearcher): round relative lengths.
        actual_wav_len = int(torch.round(wav_lens[i] * wav_len).item())
        actual_txt_len = int(torch.round(txt_lens[i] * txt_len).item())

        # (1) start_of_audio token (always valid)
        attention_mask[i, 0] = True

        # (2) audio features
        attention_mask[i, 1 : 1 + actual_wav_len] = True

        # (3) remaining text tokens (exclude the start token already handled above)
        remaining_txt = max(actual_txt_len - 1, 0)
        attention_mask[i, 1 + wav_len : 1 + wav_len + remaining_txt] = True
    return attention_mask


# Define training procedure
class ASR(sb.core.Brain):
    def get_fixed_boundaries(self, boundary_source, pad_mask, batch):
        """Boundary decisions ``(B, T)`` in ``{1, 0, -1}`` from a non-learned source.

        Arguments
        ---------
        boundary_source : str
            "fixed_rate" (a boundary every `fixed_rate_k` frames) or
            "alignment" (per-frame labels from `batch.boundary_target`).
        pad_mask : torch.Tensor
            (B, T) boolean, True at padding.
        batch : PaddedBatch
            Current batch (used for `boundary_target` in alignment mode).

        Returns
        -------
        boundary : torch.Tensor
            (B, T) long with values {1, 0, -1} (pad == -1).
        """
        if boundary_source == "fixed_rate":
            return fixed_rate_boundary_targets(
                pad_mask, int(self.hparams.fixed_rate_k)
            )
        if boundary_source == "alignment":
            boundary, _ = batch.boundary_target
            T = pad_mask.size(1)
            if boundary.size(1) != T:
                raise ValueError(
                    f"boundary_target has {boundary.size(1)} frames but the "
                    f"encoder produced {T}. Alignment labels must be at the "
                    "encoder frame rate (one {0,1} label per encoder frame)."
                )
            return boundary.long().masked_fill(pad_mask, -1)
        raise ValueError(f"Unknown boundary_source '{boundary_source}'.")

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities.

        The forward pass processes either cached SSL features or raw audio waveforms,
        projects them to the LLM embedding space, and concatenates with text embeddings
        to form a multimodal sequence.

        Sequence structure:
        [start_of_audio] + [audio_features] + [end_of_audio + prompt + bos + text]

        Arguments
        ---------
        batch : PaddedBatch
            Batch containing audio/features, tokens, and metadata
        stage : sb.Stage
            Current stage (TRAIN, VALID, or TEST)

        Returns
        -------
        logits : torch.Tensor
            Model output logits of shape (batch_size, seq_len, vocab_size)
        hyps : list or None
            Decoded hypotheses (only during validation/test, None during training)
        """
        batch = batch.to(self.device)
        tokens_bos, tokens_bos_lens = batch.tokens_bos
        prompt_len = batch.prompt_len

        use_feats = bool(getattr(self.hparams, "use_feats", False))
        if use_feats:
            if getattr(batch, "feats", None) is None:
                raise ValueError(
                    "`use_feats=True` but the batch does not provide `feats`. "
                    "Check `feats_cache_dir` and the data pipeline."
                )
            audio_feats, audio_feats_lens = batch.feats
        else:
            wavs, wav_lens = batch.sig
            wavs = self.hparams.normalize(wavs, wav_lens)
            audio_feats = self.modules.ssl(wavs, wav_lens)
            audio_feats_lens = wav_lens
        boundary_source = getattr(self.hparams, "boundary_source", "none")
        if boundary_source and boundary_source != "none":
            # Fixed-segmentation baseline: mean-pool frames between boundaries
            # instead of the fixed feat_downsampler. Sequence lengths change
            # per utterance, so recompute relative lengths from the pooled mask.
            pad_mask = lengths_to_padding_mask(
                audio_feats_lens, audio_feats.size(1)
            )
            boundary = self.get_fixed_boundaries(
                boundary_source, pad_mask, batch
            )
            audio_down_feats, seg_pad_mask = mean_pool_segments(
                audio_feats, pad_mask, boundary
            )
            # Optional blank/separator-token ablation: drop or replace the CTC
            # separator/"blank" tokens before the decoder (no-op when 'keep').
            blank_mode = getattr(self.hparams, "blank_mode", "keep") or "keep"
            if blank_mode != "keep":
                sep_frame, _ = batch.separator_target
                if sep_frame.size(1) != boundary.size(1):
                    raise ValueError(
                        f"separator_target has {sep_frame.size(1)} frames but "
                        f"boundary has {boundary.size(1)}; they must align."
                    )
                sep_seg = segment_separator_mask(
                    sep_frame.long(), boundary, pad_mask
                )
                audio_down_feats, seg_pad_mask = apply_blank_mode(
                    audio_down_feats,
                    seg_pad_mask,
                    sep_seg,
                    blank_mode,
                    const_vec=getattr(self, "blank_const", None),
                )
            audio_down_lens = padding_mask_to_lengths(seg_pad_mask)
        else:
            # R^L*D -> R^(L/R)*(D*R)
            audio_down_feats = self.modules.feat_downsampler(audio_feats)
            audio_down_lens = audio_feats_lens
        # R^D' -> R^llm_emb_size
        projected_audio_feats = self.modules.proj(audio_down_feats)
        txt_embds = self.txt_embedding(tokens_bos)
        multimodal_embds = torch.cat(
            [
                txt_embds[:, 0].unsqueeze(1),  # B, D -> B, 1, D
                projected_audio_feats,
                txt_embds[:, 1:],
            ],
            dim=1,
        )
        # attention_mask should be all the true audio features + all the true text features
        attention_mask = get_multimodal_attention_mask(
            projected_audio_feats,
            audio_down_lens,
            txt_embds,
            tokens_bos_lens,
            self.device,
        )
        logits = self.modules.llm(
            inputs_embeds=multimodal_embds, attention_mask=attention_mask
        ).logits

        hyps = None
        if stage != sb.Stage.TRAIN:
            audio_and_prompt_len = projected_audio_feats.shape[1] + int(
                prompt_len[0].item()
            )
            inputs_embeds = multimodal_embds[:, :audio_and_prompt_len]
            hyps = self.modules.searcher(
                inputs_embeds,
                audio_down_lens,
                attention_mask[:, :audio_and_prompt_len],
            )
        return logits, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the cross-entropy loss given predictions and targets.

        The loss is computed only on text tokens, with audio feature positions
        masked out using ignore_index. During validation/test, also computes
        CER and WER metrics.

        Arguments
        ---------
        predictions : tuple
            (logits, hyps) from compute_forward
        batch : PaddedBatch
            Batch containing target tokens and metadata
        stage : sb.Stage
            Current stage (TRAIN, VALID, or TEST)

        Returns
        -------
        loss : torch.Tensor
            Cross-entropy loss value
        """
        logits, hyps = predictions
        tokens_eos, _ = batch.tokens_eos
        ids = batch.id

        num_audio_feats = logits.shape[1] - tokens_eos.shape[1]
        # We prepend `ignore_index` to the tokens_eos to ignore them in the loss.
        # This corresponds to the audio features.
        target_tokens = torch.cat(
            [
                torch.full(
                    (tokens_eos.shape[0], num_audio_feats),
                    self.hparams.ignore_index,
                    device=self.device,
                ),
                tokens_eos,
            ],
            dim=1,
        ).long()
        # compute the cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target_tokens.view(-1),
            ignore_index=self.hparams.ignore_index,
        )
        if stage != sb.Stage.TRAIN:
            # replace ignore_index with pad token
            target_tokens = target_tokens.masked_fill(
                target_tokens == self.hparams.ignore_index,
                self.tokenizer.pad_token_id,
            )
            preds = self.tokenizer.batch_decode(
                hyps[0], skip_special_tokens=True
            )
            preds_words = [pred.split(" ") for pred in preds]
            targets = self.tokenizer.batch_decode(
                target_tokens, skip_special_tokens=True
            )
            targets_words = [target.split(" ") for target in targets]
            self.cer_metric.append(ids, preds_words, targets_words)
            self.wer_metric.append(ids, preds_words, targets_words)
        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Initializes metrics for validation and test stages.

        Arguments
        ---------
        stage : sb.Stage
            Current stage (TRAIN, VALID, or TEST)
        epoch : int
            Current epoch number
        """
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Logs statistics, updates learning rate, and saves checkpoints.

        Arguments
        ---------
        stage : sb.Stage
            Current stage (TRAIN, VALID, or TEST)
        stage_loss : float
            Average loss for this stage
        epoch : int
            Current epoch number
        """
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.scheduler(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # Optional: SSL fine-tuning LR scheduling (only when SSL is unfrozen).
            if hasattr(self, "ssl_optimizer") and hasattr(
                self.hparams, "lr_annealing_ssl"
            ):
                old_lr_ssl, new_lr_ssl = self.hparams.lr_annealing_ssl(
                    stage_stats["WER"]
                )
                sb.nnet.schedulers.update_learning_rate(
                    self.ssl_optimizer, new_lr_ssl
                )

            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": old_lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        """Initialize optimizers for the model.

        Creates separate optimizers for the main model and optionally for the SSL encoder
        if it's not frozen. Registers optimizers with the checkpointer for resuming training.
        """
        self.optimizer = self.hparams.opt(self.hparams.model.parameters())
        self.optimizers_dict = {"model_optimizer": self.optimizer}

        ssl_frozen = getattr(self.hparams, "ssl_frozen", True)
        if not ssl_frozen:
            self.ssl_optimizer = self.hparams.opt_ssl(
                self.modules.ssl.parameters()
            )
            self.optimizers_dict["ssl_optimizer"] = self.ssl_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("model_optimizer", self.optimizer)
            if not ssl_frozen:
                self.checkpointer.add_recoverable(
                    "ssl_optimizer", self.ssl_optimizer
                )


def dataio_prepare(hparams, tokenizer):
    """Prepares the datasets and dynamic pipelines for the brain class.

    This function sets up the data pipelines for both training and evaluation.
    It handles two modes:
    1. Standard audio mode: loads raw audio files and processes them on-the-fly
    2. Cached features mode: loads pre-extracted SSL features from HDF5 cache

    Arguments
    ---------
    hparams : dict
        Hyperparameters dictionary containing data paths, token indices, etc.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for encoding text tokens

    Returns
    -------
    train_data : DynamicItemDataset
        Training dataset
    valid_data : DynamicItemDataset
        Validation dataset
    test_datasets : dict
        Dictionary of test datasets (keyed by split name)
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer (returned for convenience)
    train_batch_sampler : DynamicBatchSampler or None
        Batch sampler for training if dynamic batching is enabled
    valid_batch_sampler : DynamicBatchSampler or None
        Batch sampler for validation if dynamic batching is enabled
    """
    data_folder = hparams["data_folder"]
    # Cached-feats mode should be enabled ONLY via the explicit `use_feats` flag.
    # Do not use `hparams["ssl"]` as a boolean (it's a model object).
    use_feats = bool(hparams.get("use_feats", False))

    if use_feats:
        feats_cache_dir = hparams.get("feats_cache_dir", None)
        if not feats_cache_dir:
            raise ValueError(
                "`use_feats=True` requires `feats_cache_dir` to be set "
                "(directory produced by `extract_ssl_feats.py`)."
            )
    else:
        # On-the-fly SSL feature extraction requires an SSL encoder module.
        modules = hparams.get("modules", {})
        if not (isinstance(modules, dict) and "ssl" in modules):
            raise ValueError(
                "`use_feats=False` requires an SSL encoder under `modules.ssl` "
                "to extract features on-the-fly. Either set `use_feats=True` "
                "and provide `feats_cache_dir`, or add `ssl` to `modules`."
            )

    logger.info("use_feats=%s", use_feats)

    # The alignment-boundary pipeline serves both the fixed-pooling baseline
    # (`boundary_source: alignment`) and the segmenter warm-start
    # (`warmstart_target: alignment` in train_speechllm_with_segmenter.py).
    boundary_source = hparams.get("boundary_source", "none") or "none"
    needs_boundary_target = (
        boundary_source == "alignment"
        or hparams.get("warmstart_target", None) == "alignment"
    )
    if needs_boundary_target and not hparams.get("boundary_target_dir"):
        raise ValueError(
            "Alignment boundaries require `boundary_target_dir`: a directory "
            "of per-utterance `<id>.pt` files, each a 1-D {0,1} tensor with "
            "one label per encoder frame (1 = frame starts a new segment)."
        )

    # Blank/separator-token ablations (blank_mode != 'keep') need a per-frame
    # is_separator channel (`separator_target_dir`, from ctc_boundary_align's
    # word_ctc_sep) so the recipe can identify the separator/"blank" tokens.
    blank_mode = hparams.get("blank_mode", "keep") or "keep"
    needs_separator = needs_boundary_target and blank_mode != "keep"
    if needs_separator and not hparams.get("separator_target_dir"):
        raise ValueError(
            f"blank_mode='{blank_mode}' requires `separator_target_dir`: a "
            "directory of per-utterance `<id>.pt` files, each a 1-D {0,1} "
            "tensor with one is_separator label per encoder frame."
        )

    # Token indices and prompt setup
    bos_index = hparams["bos_index"]
    eos_index = hparams["eos_index"]
    pad_index = hparams["pad_token"]

    # Convert special tokens to IDs with error handling
    start_of_audio_token = "<|start_of_audio|>"
    end_of_audio_token = "<|end_of_audio|>"

    start_of_audio_index = tokenizer.convert_tokens_to_ids(start_of_audio_token)
    end_of_audio_index = tokenizer.convert_tokens_to_ids(end_of_audio_token)

    logger.info(
        f"Token indices - BOS: {bos_index}, EOS: {eos_index}, PAD: {pad_index}, "
        f"start_of_audio: {start_of_audio_index}, end_of_audio: {end_of_audio_index}"
    )
    logger.info(f"Prompt: '{hparams['prompt']}'")

    prompt_ids = (
        tokenizer(
            hparams["prompt"], return_tensors="pt", add_special_tokens=False
        )
        .input_ids.view(-1)
        .tolist()
    )

    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "prompt_len"
    )
    def text_pipeline(wrd):
        """Process text through tokenization pipeline.

        Creates the following sequence structure:
        tokens_bos: [<|start_of_audio|>, <|end_of_audio|>, prompt_tokens, <bos>?, text_tokens]
        (<bos> is omitted when bos_index is None, e.g. for tokenizers with no BOS token)
        tokens_eos: [text_tokens, <eos>]

        Arguments
        ---------
        wrd : str
            Word/transcription text

        Yields
        ------
        wrd : str
            Original word (unchanged)
        tokens_list : list
            List of token IDs for the text (without special tokens)
        tokens_bos : torch.LongTensor
            Token sequence with start_of_audio, end_of_audio, prompt, bos, and text
        tokens_eos : torch.LongTensor
            Token sequence with text and eos
        tokens : torch.LongTensor
            Token IDs for text only (same as tokens_list but as tensor)
        prompt_len : int
            Length of prompt tokens (start_of_audio + end_of_audio + prompt)
        """
        yield wrd
        tokens_list = tokenizer(wrd, add_special_tokens=False).input_ids
        yield tokens_list
        tokens_bos = torch.LongTensor(
            [start_of_audio_index]
            + [end_of_audio_index]
            + prompt_ids
            + ([bos_index] if bos_index is not None else [])
            + tokens_list
        )
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        prompt_len = len(
            [start_of_audio_index] + [end_of_audio_index] + prompt_ids
        )
        yield prompt_len

    extra_items = []
    if needs_boundary_target:
        boundary_dir = hparams["boundary_target_dir"]

        @sb.utils.data_pipeline.takes("id")
        @sb.utils.data_pipeline.provides("boundary_target")
        def boundary_pipeline(utt_id):
            """Load per-frame {0,1} boundary labels for one utterance."""
            path = os.path.join(boundary_dir, f"{utt_id}.pt")
            return torch.load(path).view(-1).long()

        extra_items.append(boundary_pipeline)

    if needs_separator:
        separator_dir = hparams["separator_target_dir"]

        @sb.utils.data_pipeline.takes("id")
        @sb.utils.data_pipeline.provides("separator_target")
        def separator_pipeline(utt_id):
            """Load per-frame {0,1} is_separator labels for one utterance."""
            path = os.path.join(separator_dir, f"{utt_id}.pt")
            return torch.load(path).view(-1).long()

        extra_items.append(separator_pipeline)

    # Define dynamic items based on mode
    # Note: build_dynamic_items is defined outside the if/else to avoid scope issues
    def build_dynamic_items():
        """Build dynamic items list based on whether we're using cached features or raw audio.

        Returns
        -------
        list
            List of dynamic item pipelines
        """
        if use_feats:
            feats_pipeline = CachedHDF5DynamicItem(
                hparams["feats_cache_dir"],
                file_mode="r",
                takes=["id"],
                provides=["feats"],
                compression="gzip",
            )
            return [text_pipeline, feats_pipeline] + extra_items
        else:

            @sb.utils.data_pipeline.takes("wav")
            @sb.utils.data_pipeline.provides("sig")
            def audio_pipeline(wav):
                """Load audio from file path.

                Arguments
                ---------
                wav : str
                    Path to audio file

                Returns
                -------
                sig : torch.Tensor
                    Audio waveform
                """
                sig = sb.dataio.dataio.read_audio(wav)
                return sig

            return [text_pipeline, audio_pipeline] + extra_items

    # Set output keys based on mode
    if use_feats:
        output_keys = [
            "id",
            "wrd",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "prompt_len",
            "feats",
        ]
    else:
        output_keys = [
            "id",
            "sig",
            "wrd",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "prompt_len",
        ]
    if needs_boundary_target:
        output_keys.append("boundary_target")
    if needs_separator:
        output_keys.append("separator_target")

    def _create_dataset(csv_path, sorting="ascending"):
        """Create a dataset from CSV file with optional sorting.

        Arguments
        ---------
        csv_path : str
            Path to CSV file containing dataset metadata
        sorting : str
            Sorting strategy: "ascending", "descending", or "random"

        Returns
        -------
        dataset : DynamicItemDataset
            Configured dataset with dynamic pipelines applied
        """
        assert sorting in ["ascending", "descending", "random"], (
            f"sorting must be one of ['ascending', 'descending', 'random'], got '{sorting}'"
        )

        dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_path,
            replacements={"data_root": data_folder},
            dynamic_items=build_dynamic_items(),
            output_keys=output_keys,
        )
        if sorting == "ascending":
            dataset = dataset.filtered_sorted(sort_key="duration")
            hparams["train_dataloader_opts"]["shuffle"] = False
        elif sorting == "descending":
            dataset = dataset.filtered_sorted(sort_key="duration", reverse=True)
            hparams["train_dataloader_opts"]["shuffle"] = False
        elif sorting == "random":
            pass
        return dataset

    # Create training dataset with sorting logic
    train_data = _create_dataset(
        hparams["train_csv"], sorting=hparams["sorting"]
    )
    valid_data = _create_dataset(hparams["valid_csv"], sorting="ascending")

    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = _create_dataset(csv_file, sorting="ascending")

    # Dynamic batch sampling
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **hparams["dynamic_batch_sampler_train"],
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **hparams["dynamic_batch_sampler_valid"],
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
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

    # here we create the datasets objects as well as tokenization and encoding
    tokenizer = hparams["llm"].tokenizer

    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.tokenizer = tokenizer
    asr_brain.txt_embedding = (
        asr_brain.raw_modules.llm.model.get_input_embeddings()
    )
    # Expose the learned const vector (used only by blank_mode: const) so
    # compute_forward's apply_blank_mode(const_vec=self.blank_const) can reach it.
    if hasattr(asr_brain.raw_modules, "blank_const_module"):
        asr_brain.blank_const = asr_brain.raw_modules.blank_const_module.vec
    # adding objects to trainer:
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            min_key="WER",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
