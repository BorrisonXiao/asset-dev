from praatio import textgrid

from datasets import load_dataset
import os
import torchaudio
from tqdm import tqdm


ds = load_dataset("edinburghcstr/ami", "sdm")
SUBSET = "train"
ds = ds[SUBSET]

MFA_PATH = "/home/krishna/mfa_data/my_corpus/"
ALIGNED_PATH = "/home/krishna/mfa_data/my_corpus_aligned/"


skipped = 0
avg_dur = 0

for meeting_id, audio_id, text, audio, begin_time, end_time, mic_id, speaker_id in tqdm(zip(ds["meeting_id"], ds["audio_id"], ds["text"], ds["audio"], ds["begin_time"], ds["end_time"], ds["microphone_id"], ds["speaker_id"]), total=len(ds)):
    speaker_dir_mfa = os.path.join(MFA_PATH, speaker_id)
    speaker_dir_aligned = os.path.join(ALIGNED_PATH, speaker_id)
    audio_path = os.path.join(speaker_dir_mfa, f"{meeting_id}___{audio_id}.wav")
    text_path = os.path.join(speaker_dir_mfa, f"{meeting_id}___{audio_id}.lab")
    grid_path = os.path.join(speaker_dir_aligned, f"{meeting_id}___{audio_id}.TextGrid")

    if not os.path.exists(grid_path):
        # print(f"TextGrid file {grid_path} does not exist. Skipping.")
        avg_dur += (end_time - begin_time)
        skipped += 1

print(f"Average duration of skipped files: {avg_dur / skipped:.2f} seconds. Total skipped duration: {avg_dur:.2f} seconds.")
print(f"Skipped {skipped} files due to missing TextGrid files.")
