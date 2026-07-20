from datasets import load_dataset
from tqdm import tqdm
import os
from praatio import textgrid


MFA_PATH = "/home/krishna/mfa_data/my_corpus/"
MFA_ALIGN = "/home/krishna/mfa_data/my_corpus_aligned/"

dataset = load_dataset("slprl/Stress-17K-raw")['train_full']

metadata = dataset['metadata']
audio = dataset['audio']
transcript = dataset['transcription']
tid = dataset['transcription_id']



skip = 0

word_boundaries = []
phone_boundaries = []

keepers = []

for idx, (i, m, a, t) in enumerate(tqdm(zip(tid, metadata, audio, transcript), total=len(tid))):
    speaker_dir = os.path.join(MFA_PATH, m['tts_metadata']['voice_name'])
    os.makedirs(speaker_dir, exist_ok=True)
    audio_path = os.path.join(speaker_dir, f"{i}.wav")
    text_path = os.path.join(speaker_dir, f"{i}.lab")
    grid_path = os.path.join(MFA_ALIGN, m['tts_metadata']['voice_name'], f"{i}.TextGrid")
    fileExists = os.path.exists(grid_path)
    if not fileExists:
        skip += 1
        continue
    grid_reader = textgrid.openTextgrid(grid_path, True)
    keepers.append(idx)

    words = []
    for word_interval in grid_reader.tiers[0].entries:
        words.append({
            "start": word_interval.start,
            "end": word_interval.end,
            "text": word_interval.label
        })

    phones = []
    for phone_interval in grid_reader.tiers[1].entries:
        phones.append({
            "start": phone_interval.start,
            "end": phone_interval.end,
            "text": phone_interval.label
        })

    word_boundaries.append(words)
    phone_boundaries.append(phones)

new_dataset = dataset.select(keepers)
# add to the original dataset
new_dataset = new_dataset.add_column("word_boundaries", word_boundaries)
new_dataset = new_dataset.add_column("phone_boundaries", phone_boundaries)

print(new_dataset)
new_dataset.save_to_disk("stress_full_mfa")



