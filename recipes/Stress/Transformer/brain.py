import sys, os
import re
import torch
from torch.nn import functional as F
import torch.distributed as dist
import wandb
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataset import DynamicItemDataset
from transformers import AutoTokenizer
from jiwer import wer
import datasets

def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True  # single GPU, always log

def get_stress(text):
    m = re.findall(r"<stress>(.*?)</stress>", text)
    return ", ".join(m)

def get_stripped(text):
    return re.sub(r"</?stress>", "", text)

def pool_by_word_boundaries(feats, feat_lens, word_boundaries, frame_rate=50.0):
    """
    Average-pool SSL features into one vector per word, using word timing
    instead of a fixed stride.

    feats:          (B, T, D) SSL features
    feat_lens:      (B,) relative lengths in [0,1], as returned by speechbrain
                    for the padded batch (used to avoid pooling into pad frames)
    word_boundaries: list of length B; each element is a list of dicts like
                    {"start": float, "end": float, "text": str} (seconds)
    frame_rate:     frames/sec produced by the SSL encoder (e.g. WavLM at
                    16kHz with a 20ms stride -> 50Hz). Check against your
                    actual encoder config rather than assuming.

    Returns:
        pooled:    (B, W_max, D) zero-padded, one row per word
        word_lens: (B,) number of real (non-pad) words per sample
    """
    B, T, D = feats.shape
    device = feats.device

    per_sample = []
    for b in range(B):
        words = [w for w in word_boundaries[b] if w["text"].strip() != ""]
        n_valid = int(feat_lens[b].item() * T) if feat_lens is not None else T

        pooled_words = []
        for w in words:
            start_idx = int(round(w["start"] * frame_rate))
            end_idx = int(round(w["end"] * frame_rate))

            start_idx = max(0, min(start_idx, n_valid - 1))
            end_idx = max(start_idx + 1, min(end_idx, n_valid))

            pooled_words.append(feats[b, start_idx:end_idx].mean(dim=0))

        if not pooled_words:  # no words (e.g. silence-only) — fallback
            pooled_words = [feats[b, :n_valid].mean(dim=0)]

        per_sample.append(torch.stack(pooled_words, dim=0))  # (W_b, D)

    word_lens = torch.tensor([w.shape[0] for w in per_sample], device=device)
    W_max = int(word_lens.max().item())

    pooled = torch.zeros(B, W_max, D, device=device, dtype=feats.dtype)
    for b, wf in enumerate(per_sample):
        pooled[b, : wf.shape[0]] = wf

    return pooled, word_lens

model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Best practice for decoder-only generation

def dataio_pipeline(dataset):
    sbds = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(dataset)

    # 1. audio
    @sb.utils.data_pipeline.takes("audio")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(audio):
        samples = audio.get_all_samples()
        return samples.data[0]

    # 2. transcript → token ids
    @sb.utils.data_pipeline.takes("transcription", "gt_stress_indices")
    @sb.utils.data_pipeline.provides(
        "tokens_eos",
        "prompt_len",
        "transcript",
    )
    def text_pipeline(transcription, gt_stress_indices):
        words = transcription.split()
        transcript = ""
        for i, word in enumerate(words):
            if i in gt_stress_indices:
                transcript += f"<stress>{word}</stress> "
            else:
                transcript += word + " "
        transcript = transcript.strip()

        prompt_prefix = (
            "Given the audio and audio transcript, "
            "find the words that were stressed in the audio "
            f"<transcript>{get_stripped(transcript)}</transcript>: "
        )

        answer = get_stress(transcript)

        prompt_ids = tokenizer(
            prompt_prefix,
            add_special_tokens=True,
        ).input_ids

        answer_ids = tokenizer(
            answer,
            add_special_tokens=False,
        ).input_ids

        eos_ids = tokenizer(
            tokenizer.eos_token,
            add_special_tokens=False,
        ).input_ids

        tokens = torch.tensor(
            prompt_ids + answer_ids + eos_ids,
            dtype=torch.long,
        )

        return tokens, len(prompt_ids), transcript

    sbds.add_dynamic_item(audio_pipeline)
    sbds.add_dynamic_item(text_pipeline)
    sbds.set_output_keys([
        "id",
        "sig",
        "tokens_eos",
        "prompt_len",
        "transcript",
        "word_boundaries",
        "phone_boundaries"
    ])

    return sbds

class StressBrain(sb.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_val_acc = 0

    def on_stage_start(self, stage, epoch):
        self.epoch = epoch
        if stage == sb.Stage.VALID:
            self.val_wer = []
            self.val_stress = []

    def _llm(self):
        """Unwrap DDP if present, return the raw AdaptedModel."""
        m = self.modules.llm
        return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

    def _hf_model(self):
        """Returns HuggingFace LlamaForCausalLM, handling DDP."""
        return self._llm().adapted_model.model

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        transcripts = batch.transcript

        prompt_lens = batch.prompt_len

        # audio features from WavLM
        feats = self.modules.ssl(wavs)  # (B, T', 1024)

        if type(self.hparams.pooling_factor) is int:
            feats = F.avg_pool1d(
                feats.transpose(1, 2),  # (B, 1024, T)
                kernel_size=self.hparams.pooling_factor,
                stride=self.hparams.pooling_factor,
            ).transpose(1, 2)           # (B, T // 5, 1024)
        elif self.hparams.pooling_factor == "word_boundaries":
            feats, _ = pool_by_word_boundaries(
                feats, lens, batch.word_boundaries, frame_rate=self.hparams.ssl_frame_rate
            )  # (B, W_max, 1024)
        elif self.hparams.pooling_factor == "phone_boundaries":
            feats, _ = pool_by_word_boundaries(
                feats, lens, batch.phone_boundaries, frame_rate=self.hparams.ssl_frame_rate
            )  # (B, P_max, 1024)
        else:
            raise ValueError(f"Unknown pooling factor: {self.hparams.pooling_factor}")

        # project into LLM space
        audio_tokens = self.modules.proj(feats)  # (B, T', llm_emb_size)

        # text embeddings
        tokens, _ = batch.tokens_eos
        text_embeds = self._hf_model().model.embed_tokens(tokens)

        # prepend audio tokens to text embeddings
        inputs_embeds = torch.cat(
            [audio_tokens, text_embeds], dim=1
        )  # (B, T'+L, llm_emb_size)

        if stage == sb.Stage.VALID:
            prompt_evals = [
                f"Given the audio and audio transcript, find the words that were stressed in the audio <transcript>{get_stripped(t)}</transcript>: " for t in transcripts
            ]

            tokens_eval = tokenizer(prompt_evals, return_tensors="pt", padding=True).to(self.device)
            text_embeds_eval = self._hf_model().model.embed_tokens(tokens_eval.input_ids)
            input_embeds_eval = torch.cat([audio_tokens, text_embeds_eval], dim=1)
        else:
            input_embeds_eval = None

        return (
            inputs_embeds,
            tokens,
            prompt_lens,
            feats.shape[1],
            input_embeds_eval,
            transcripts,
        )

    def compute_objectives(self, predictions, batch, stage):
        (
            inputs_embeds,
            tokens,
            prompt_lens,
            audio_len,
            input_embeds_eval,
            transcripts,
        ) = predictions

        _, lens = batch.tokens_eos
        abs_lens = (lens * tokens.shape[1]).long()

        # mask audio tokens, supervise only on text
        labels = torch.cat(
            [
                torch.full((tokens.shape[0], audio_len), -100, device=self.device),
                tokens.clone(),
            ],
            dim=1,
        )
        for i, prompt_len in enumerate(prompt_lens):
            labels[
                i,
                audio_len : audio_len + int(prompt_len)
            ] = -100

        # 2. Mask out padding tokens at the end of the sequences
        for i, length in enumerate(abs_lens):
            labels[i, audio_len + length :] = -100

        outputs = self.modules.llm(inputs_embeds=inputs_embeds, labels=labels)

        if stage == sb.Stage.VALID:
            self.modules.llm.eval()
            with torch.no_grad():
                out = self._hf_model().generate(
                    inputs_embeds=input_embeds_eval, 
                    max_new_tokens=50,
                    max_length=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            preds = tokenizer.batch_decode(out, skip_special_tokens=True)
            
            for pred, gt in zip(preds, transcripts):
                gt_stress = get_stress(gt)
                pred = pred.strip()
                
                self.val_wer.append(wer(gt_stress, pred) if len(gt_stress) > 0 else (0 if pred == "" else 1))
                self.val_stress.append(1 if gt_stress == pred else 0)
                
                if len(self.val_stress) <= 3:
                    if self._is_main:
                        print(f"pred: {pred}")
                        print(f"gt  : {gt_stress}")
                        print("---")
                        wandb.log({
                            f"examples/pred_{len(self.val_stress)}": pred,
                            f"examples/gt_{len(self.val_stress)}": gt_stress,
                        }, step=self.epoch, commit=False)

        return outputs.loss

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            if self._is_main:
                wandb.log({"train/loss": stage_loss, "epoch": epoch}, step=epoch, commit=False)

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_scheduler(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            epoch_wer = sum(self.val_wer) / len(self.val_wer) if self.val_wer else 0
            epoch_stress = sum(self.val_stress) / len(self.val_stress) if self.val_stress else 0

            self.best_val_acc = max(self.best_val_acc, epoch_stress)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "wer_stress": epoch_wer,
                    "stress_acc": epoch_stress,
                },
            )

            if self._is_main:
                wandb.log({
                    "epoch": epoch,
                    "lr": old_lr,
                    "valid/loss": stage_loss,
                    "valid/wer_stress": epoch_wer,
                    "valid/stress_acc": epoch_stress,
                    "valid/best_stress_acc": self.best_val_acc,
                }, step=epoch, commit=True)

            self.hparams.checkpointer.save_and_keep_only(
                meta={"loss": stage_loss},
                min_keys=["loss"],  # keep checkpoint with lowest valid loss
            )

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)  # Initialize DDP

    import torch
    _orig_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load
    # TEMPORARY: only for trusted/local checkpoints
    import transformers.utils.import_utils as tf_import_utils
    tf_import_utils.check_torch_load_is_safe = lambda: None

    import transformers.modeling_utils as tf_modeling_utils
    tf_modeling_utils.check_torch_load_is_safe = lambda: None

    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    if is_main_process():
        run = wandb.init(
            project=hparams["project_name"],
            name=f"{hparams['experiment_name']}-seed{hparams['seed']}",
            config={
                "lora_rank": hparams["lora_rank"],
                "lora_alpha": hparams["lora_alpha"],
                "lora_dropout": hparams["lora_dropout"],
                "batch_size": hparams["batch_size"],
                "initial_lr": hparams["initial_lr"],
                "ssl_frozen": hparams["ssl_frozen"],
            }
        )
        artifact = wandb.Artifact(
            name="config",
            type="config",
        )
        artifact.add_file(hparams_file)
        run.log_artifact(artifact)

    # create output folders
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # data
    full_dataset = datasets.load_from_disk(hparams["dataset"])
    split = full_dataset.train_test_split(test_size=0.1, seed=42)

    train = split['train']
    test = split['test']

    valid_data = dataio_pipeline(test)
    train_data = dataio_pipeline(train)

    # brain
    brain = StressBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    brain._is_main = is_main_process()

    brain.valid_set = valid_data

    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs={
            "batch_size": hparams["batch_size"],
            "num_workers": hparams["num_workers"],
        },
        valid_loader_kwargs={"batch_size": 1, "num_workers": 0},
    )
    
    if is_main_process():
        wandb.finish()
