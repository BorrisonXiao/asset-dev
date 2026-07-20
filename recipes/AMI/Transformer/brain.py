from hyperpyyaml import load_hyperpyyaml
import sys
import torch
from torch.nn import functional as F
import wandb
import speechbrain as sb
from speechbrain.dataio.sampler import DynamicBatchSampler
from transformers import AutoTokenizer
import datasets



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Best practice for decoder-only generation


def dataio(ds_path):
    ds = datasets.load_from_disk(ds_path)
    sbds = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(ds)

    @sb.utils.data_pipeline.takes("audio")
    @sb.utils.data_pipeline.provides("sig", "sig_len")
    def audio_pipeline(audio):
        sig = torch.tensor(audio, dtype=torch.float32)
        return sig, len(sig)

    @sb.utils.data_pipeline.takes("speakers")
    @sb.utils.data_pipeline.provides(
        "tokens_eos",
        "prompt_len",
    )
    def text_pipeline(speakers):
        ans = len(speakers)

        prompt_prefix = (
            "Given the audio, identify the number of speakers in the audio clip."
            "The only valid answers are 0-4 inclusive: "
        )

        answer = str(ans)

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

        return tokens, len(prompt_ids)

    sbds.add_dynamic_item(audio_pipeline)
    sbds.add_dynamic_item(text_pipeline)
    sbds.set_output_keys(["sig", "sig_len", "tokens_eos", "prompt_len"])

    return sbds


def make_dynamic_batch_sampler(
    dataset,
    max_secs_per_batch=30,
    num_buckets=20,
    shuffle=True,
    batch_ordering="descending",
):
    max_batch_length = int(max_secs_per_batch * 16000)  # Assuming 16kHz sample rate
    hf_dataset = dataset.data

    # 2. Extract all lengths instantly via columnar access
    lengths = [int(dur * 16000) for dur in hf_dataset["duration"]]
    return DynamicBatchSampler(
        dataset,
        max_batch_length=max_batch_length,
        num_buckets=num_buckets,
        lengths_list=lengths,
        shuffle=shuffle,
        batch_ordering=batch_ordering,
    )


class StressBrain(sb.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_val_acc = 0

    def on_stage_start(self, stage, epoch):
        self.epoch = epoch
        self.acc_tracker = []
        if stage != sb.Stage.TRAIN:
            self.printed_examples = 0

    def _llm(self):
        """Unwrap DDP if present, return the raw AdaptedModel."""
        m = self.modules.llm
        return (
            m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m
        )

    def _hf_model(self):
        """Returns HuggingFace LlamaForCausalLM, handling DDP."""
        return self._llm().adapted_model.model

    def _pad_mask(self, rel_lens, max_len):
        """(B, max_len) mask: True for real positions, False for right-padding."""
        abs_lens = (rel_lens * max_len).round().long()
        positions = torch.arange(max_len, device=self.device)
        return positions[None, :] < abs_lens[:, None]

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        MIN_SAMPLES = 400
        if wavs.shape[-1] < MIN_SAMPLES:
            wavs = torch.nn.functional.pad(wavs, (0, MIN_SAMPLES - wavs.shape[-1]))

        prompt_lens = batch.prompt_len

        # audio features from WavLM
        feats = self.modules.ssl(wavs)  # (B, T', 1024)
        feats = F.avg_pool1d(
            feats.transpose(1, 2),  # (B, 1024, T)
            kernel_size=self.hparams.pooling_factor,
            stride=self.hparams.pooling_factor,
        ).transpose(1, 2)           # (B, T // 5, 1024)
        # project into LLM space
        audio_tokens = self.modules.proj(feats)  # (B, T', llm_emb_size)

        # text embeddings
        tokens, tok_lens = batch.tokens_eos
        text_embeds = self._hf_model().model.embed_tokens(tokens)

        # cross-modal input: audio tokens followed by text tokens
        inputs_embeds = torch.cat([audio_tokens, text_embeds], dim=1)

        # tell the LLM which positions are real vs. right-padding, so it never
        # attends to padded audio frames or padded text tokens
        audio_len, text_len = feats.shape[1], tokens.shape[1]
        attention_mask = torch.cat(
            [self._pad_mask(lens, audio_len), self._pad_mask(tok_lens, text_len)],
            dim=1,
        )

        return (inputs_embeds, attention_mask, tokens, prompt_lens, audio_len)

    def compute_objectives(self, predictions, batch, stage):
        inputs_embeds, attention_mask, tokens, prompt_lens, audio_len = predictions

        _, lens = batch.tokens_eos
        abs_lens = (lens * tokens.shape[1]).long()

        # Build labels (identical sequence layout for both passes)
        labels = torch.cat(
            [
                torch.full((tokens.shape[0], audio_len), -100, device=self.device),
                tokens.clone(),
            ],
            dim=1,
        )
        for i, prompt_len in enumerate(prompt_lens):
            labels[i, audio_len : audio_len + int(prompt_len)] = -100

        for i, length in enumerate(abs_lens):
            labels[i, audio_len + length :] = -100

        outputs = self.modules.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        # Track accuracy for all stages
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = outputs.logits[..., :-1, :].detach()

        for i in range(shift_labels.size(0)):
            valid_indices = (shift_labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                ans_idx = valid_indices[0]
                target = shift_labels[i, ans_idx].item()
                pred = shift_logits[i, ans_idx].argmax(dim=-1).item()
                self.acc_tracker.append(1 if pred == target else 0)

                if stage != sb.Stage.TRAIN and self.printed_examples < 10:
                    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                    if is_main:
                        t_char = tokenizer.decode([target]).strip()
                        n_char = tokenizer.decode([pred]).strip()
                        print(f"\n[Epoch {self.epoch} | Sample {self.printed_examples + 1}] "
                              f"Target: '{t_char}' Pred: '{n_char}' (Correct: {pred == target})")
                    self.printed_examples += 1

        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        acc = (sum(self.acc_tracker) / len(self.acc_tracker) * 100) if self.acc_tracker else 0.0
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        self.best_val_acc = max(self.best_val_acc, acc)

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            if is_main:
                wandb.log({"train/loss": stage_loss, "train/acc": acc, "epoch": epoch}, step=epoch, commit=False)

        elif stage == sb.Stage.VALID:
            # anneal the learning rate per the schedule defined in train.yaml
            _, new_lr = self.hparams.lr_scheduler(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            print(f"\n" + "=" * 60)
            print(f"EPOCH {epoch} VALIDATION SUMMARY")
            print(f"Valid Loss: {stage_loss:.4f}")
            print(f"Valid Acc:  {acc:.2f}%")
            print("=" * 60 + "\n")
            if is_main:
                wandb.log({"valid/loss": stage_loss, "valid/acc": acc, "best_val_acc": self.best_val_acc,
                           "lr": new_lr, "epoch": epoch}, step=epoch)
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_loss, "acc": acc, "epoch": epoch},
                min_keys=["loss"],
            )

        elif stage == sb.Stage.TEST:
            print(f"\n" + "=" * 60)
            print(f"FINAL TEST SUMMARY")
            print(f"Test Loss: {stage_loss:.4f}")
            print(f"Test Acc:  {acc:.2f}%")
            print("=" * 60 + "\n")
            if is_main:
                wandb.log({"test/loss": stage_loss, "test/acc": acc, "best_val_acc": self.best_val_acc, "epoch": epoch}, step=epoch)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)


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

    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_main:
        run = wandb.init(project=hparams.get("wandb_project", "jsalt_speakerctn"), config=hparams)
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
    train_data = dataio(hparams["train_hf"])
    valid_data = dataio(hparams["valid_hf"])

    # brain
    brain = StressBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    brain.valid_set = valid_data

    train_sampler = make_dynamic_batch_sampler(
        train_data,
        max_secs_per_batch=hparams["max_secs_per_batch"],
        num_buckets=hparams.get("num_buckets", 20),
        shuffle=True,
    )

    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs={
            "batch_sampler": train_sampler,
            "num_workers": hparams["num_workers"],
        },
        valid_loader_kwargs={"batch_size": 1, "num_workers": 0},
    )
