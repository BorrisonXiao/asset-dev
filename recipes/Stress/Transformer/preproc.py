from datasets import load_dataset
from tqdm import tqdm
import torchaudio
import os


MFA_PATH = "/home/krishna/mfa_data/my_corpus/"

dataset = load_dataset("slprl/Stress-17K-raw")['train_full']

metadata = dataset['metadata']
audio = dataset['audio']
transcript = dataset['transcription']
tid = dataset['transcription_id']


os.makedirs(MFA_PATH, exist_ok=True)

empty = True
for _ in os.scandir(MFA_PATH):
    empty = False
    break

if not empty:
    print(f"Directory {MFA_PATH} is not empty. Please clear it before running this script.")
    exit(1)



for i, m, a, t in tqdm(zip(tid, metadata, audio, transcript)):
    speaker_dir = os.path.join(MFA_PATH, m['tts_metadata']['voice_name'])
    os.makedirs(speaker_dir, exist_ok=True)
    audio_path = os.path.join(speaker_dir, f"{i}.wav")
    text_path = os.path.join(speaker_dir, f"{i}.lab")
    try:
        samples = a.get_all_samples()
    except RuntimeError as e:
        print(f"Error processing audio for {i}")
        continue
    torchaudio.save(audio_path, samples.data, samples.sample_rate)

    with open(text_path, "w") as f:
        f.write(t)
