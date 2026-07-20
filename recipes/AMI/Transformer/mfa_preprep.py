from datasets import load_dataset
import os
import torchaudio
from tqdm import tqdm


ds = load_dataset("edinburghcstr/ami", "sdm")
SUBSET = "train"
ds = ds[SUBSET]

MFA_PATH = "/home/krishna/mfa_data/my_corpus/"



os.makedirs(MFA_PATH, exist_ok=True)

empty = True
for _ in os.scandir(MFA_PATH):
    empty = False
    break

if not empty:
    print(f"Directory {MFA_PATH} is not empty. Please clear it before running this script.")
    exit(1)


for meeting_id, audio_id, text, audio, begin_time, end_time, mic_id, speaker_id in tqdm(zip(ds["meeting_id"], ds["audio_id"], ds["text"], ds["audio"], ds["begin_time"], ds["end_time"], ds["microphone_id"], ds["speaker_id"]), total=len(ds)):
    speaker_dir = os.path.join(MFA_PATH, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)
    audio_path = os.path.join(speaker_dir, f"{meeting_id}___{audio_id}.wav")
    text_path = os.path.join(speaker_dir, f"{meeting_id}___{audio_id}.lab")
    try:
        samples = audio.get_all_samples()
    except RuntimeError as e:
        print(f"Error processing audio for meeting {meeting_id}, audio {audio_id}: {e}")
        continue
    torchaudio.save(audio_path, samples.data, samples.sample_rate)

    with open(text_path, "w") as f:
        f.write(text)
