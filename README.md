# SpeechLLM

### Installation

Tested with Python 3.11, but >3.9 should work

```bash
git clone git@github.com:JSALT2026-OmniEnc/speechllm.git
cd speechllm
# Pytorch versions I used
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt 
pip install --editable .
pip install accelerate
pip install h5py
```

### Speech Encoder + LLM for ASR on LibriSpeech

1. End-to-end training:
```bash
python recipes/LibriSpeech/ASR/transformer/train_speechllm.py \
    recipes/LibriSpeech/ASR/transformer/hparams/speechllm_e2e.yaml \
    --seed 42 \
    --data_folder /path/to/LibriSpeech \
    --output_folder ./results/speechllm_e2e/42/ls960 \
    --ssl_hub microsoft/wavlm-large \
    --ssl_folder ssl_checkpoints \
    --ssl_feat_dims 1024 \
    --llm_path meta-llama/Llama-3.2-1B \
    --llm_emb_size 2048 \
    --bos_index 128000 \
    --eos_index 128001 \
    --pad_token 128004

# Useful arguments to override
# 1. Use prepared CSV for training and skip prep step
    # --skip_prep True \
    # --train_splits='["train-clean-100", "train-clean-360", "train-other-500"]' \
    # --csv_folder ./results \

# 2. Use local huggingface LLM checkpoint folder
    # --llm_path /path/to/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/ \
    # --llm_save_path None \

# 3. Reduce max batch size to fit on GPU
    # --max_batch_length_train 200 \
    # --grad_accumulation_factor 10 \

# 4. Use original LLM without LoRA adapters
    # --llm='!ref <backbone_llm>' \
```
2. Precompute SSL features, then train LLM:
```bash
python recipes/LibriSpeech/ASR/transformer/extract_ssl_feats.py \
    recipes/LibriSpeech/ASR/transformer/hparams/extract_ssl_feats.yaml \
    --data_folder /path/to/LibriSpeech \
    --output_folder ./results/feats_cache \
    --ssl_hub microsoft/wavlm-large \
    --ssl_folder ssl_checkpoints \
    --ssl_output_norm False \
    --feats_cache_dir /path/to/features_cache 
    # use different cache folders for different datasets!

python recipes/LibriSpeech/ASR/transformer/train_speechllm.py \
    recipes/LibriSpeech/ASR/transformer/hparams/speechllm_ssl_feats.yaml \
    --seed 42 \
    --feats_cache_dir /path/to/features_cache \
    --data_folder /path/to/LibriSpeech \
    --output_folder ./results/speechllm_ssl_feats/42/ls960 \
    --csv_folder ./results \
    --ssl_feat_dims 1024 \
    --llm_path meta-llama/Llama-3.2-1B \
    --llm_emb_size 2048 \
    --bos_index 128000 \
    --eos_index 128001 \
    --pad_token 128004
```

### Attribution

Using the Speechbrain toolkit at commit [da704dd9fd710df5c6c93ea2d31a028837d93d85](https://github.com/speechbrain/speechbrain/tree/da704dd9fd710df5c6c93ea2d31a028837d93d85)