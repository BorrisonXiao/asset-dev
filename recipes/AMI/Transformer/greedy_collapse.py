import datasets
import numpy as np
from tqdm import tqdm
import random
import torch
from matplotlib import pyplot as plt

class SyntheticNoiseSegment:
    """Mock container mimicking your dataset's original audio objects."""
    def __init__(self, duration_sec, sample_rate=16000, noise_level=0.005):
        self.sample_rate = sample_rate
        num_samples = int(duration_sec * sample_rate)
        # Generate low-amplitude Gaussian white noise (sounds like static hiss)
        self.data = torch.randn(1, num_samples) * noise_level

    def get_all_samples(self):
        # We wrap this so .get_all_samples().data works seamlessly with your torch.cat code
        class DataHolder:
            def __init__(self, data):
                self.data = data
        return DataHolder(self.data)

def coalesce_windows(ds_path, max_speakers=4, max_seconds=25):
    ds = datasets.load_from_disk(ds_path)
    hist = np.zeros(5)
    coalesced_ds = []
    duration_accum = 0
    speakers_comb = set()
    audios_comb = []
    meeting_running = None
    durations = []
    for meeting, chron, speakers, texts, duration, audios in tqdm(
        zip(
            ds["meeting"],
            ds["window_chronology"],
            ds["speakers"],
            ds["texts"],
            ds["duration"],
            ds["audios"],
        ),
        total=len(ds["meeting"]),
        desc="Coalescing windows",
    ):
        if meeting_running is None:
            meeting_running = meeting

        if hist[0] < np.mean(hist[1:]):
            noise_duration = random.uniform(np.mean(durations), np.std(durations)) # Generate noise between 1.5 to 4.0 seconds
            noise_segment = SyntheticNoiseSegment(noise_duration)
            
            # Immediately write it to the dataset as a 0-speaker event
            coalesced_ds.append({
                "meeting": meeting_running,
                "speakers": [], # 0 speakers!
                "duration": noise_duration,
                "audio": noise_segment.data.squeeze(0),
            })
            hist[0] += 1 # Populate our 0 bin

        speakers_with = speakers_comb | set(speakers)

        # Option A: What if we split (keep current count)?
        hist_if_split = hist.copy()
        hist_if_split[len(speakers_comb)] += 1
        std_if_split = np.std(hist_if_split[1:])  # Ignore index 0 (no speakers) if it's not a valid target

        # Option B: What if we coalesce (add current speakers to the pool)?
        hist_if_coalesce = hist.copy()
        # (We don't increment the hist yet because the segment isn't finalized, 
        # but we check if the new combined length moves us toward a less-represented bin)
        target_bin = len(speakers_with)

        # 2. Decision Logic
        if len(audios_comb) == 0:
            coalesce_now = True
        elif duration_accum + duration > max_seconds:
            # If adding this segment would exceed our max duration, we must yield
            coalesce_now = False
        elif len(speakers_comb) >= max_speakers:
            # Hard stop if we hit our maximum speaker capacity
            coalesce_now = False
        else:
            # If the combined speaker count is currently underrepresented compared to our split count,
            # or if merging reduces (or keeps equal) the overall variance of our histogram:
            if target_bin < len(hist) and hist[target_bin] < hist[len(speakers_comb)]:
                coalesce_now = True
            elif target_bin < len(hist) and np.std(hist_if_coalesce[1:]) <= std_if_split:
                coalesce_now = True
            else:
                coalesce_now = False

        if coalesce_now and meeting_running == meeting: #Combine
            speakers_comb.update(speakers)
            duration_accum += duration
            audios_comb.append(
                audios[0]
            )  # for sdm, all audios are identical for all speakers.
        else: # yield
            if duration_accum <= max_seconds:
                audio = torch.cat(
                    [s.get_all_samples().data for s in audios_comb], dim=1
                ).squeeze(0)
                coalesced_ds.append(
                    {
                        "meeting": meeting_running,
                        "speakers": list(speakers_comb),
                        "duration": duration_accum,
                        "audio": audio,
                    }
                )
            else:
                print(f"Error: Coalesced duration {duration_accum:.2f}s exceeds max_seconds {max_seconds}s.")
            durations.append(duration_accum)
            hist[len(speakers_comb)] += 1
            meeting_running = meeting
            speakers_comb = set(speakers)
            duration_accum = duration
            audios_comb = [audios[0]]
    # Add the last accumulated meeting
    if len(audios_comb) > 0:
        audio = torch.cat(
            [s.get_all_samples().data for s in audios_comb], dim=1
        ).squeeze(0)
        coalesced_ds.append(
            {
                "meeting": meeting_running,
                "speakers": list(speakers_comb),
                "duration": duration_accum,
                "audio": audio,
            }
        )
    new_ds = datasets.Dataset.from_list(coalesced_ds)
    return new_ds


dataset = coalesce_windows("sdm_raw/ami_sdm_overlap_windows_train")
dataset.save_to_disk("ami_sdm_overlap_windows_balance_train")


speaker_counts = [len(s) for s in dataset["speakers"]]

bins = np.arange(5)
hist = np.bincount(speaker_counts, minlength=len(bins))

# 1. Extract the data from your new dataset
durations = np.array(dataset["duration"])
speaker_counts = np.array([len(s) for s in dataset["speakers"]])

# 2. Group durations by their corresponding speaker count (0, 1, 2, 3, 4)
grouped_durations = []
labels = []

for count in range(5):
    # Filter durations where the speaker count matches
    group = durations[speaker_counts == count]
    grouped_durations.append(group)
    labels.append(f"{count} Speakers\n(n={len(group)})")

# 3. Create the Vertical/Horizontal Box & Whisker Plot
plt.figure(figsize=(10, 6))

# Set vert=False to flip the plot vertically
plt.boxplot(
    grouped_durations, 
    labels=labels, 
    vert=False,  # This flips the axes!
    patch_artist=True,  # Allows filling boxes with color
    # showfliers=False,
    medianprops={"color": "black", "linewidth": 1.5},
    boxprops={"facecolor": "#a8dadc", "edgecolor": "#457b9d"}
)

plt.title("Distribution of Segment Durations by Speaker Count", fontsize=14, fontweight="bold")
plt.xlabel("Duration (Seconds)", fontsize=11)  # Swapped axes label
plt.ylabel("Number of Speakers in Segment", fontsize=11)  # Swapped axes label
plt.grid(axis='x', linestyle='--', alpha=0.7)  # Changed grid to vertical lines

# Save and display
plt.tight_layout()
plt.savefig("duration_box_plot_vertical.png")


plt.clf()

plt.bar(bins, hist)
plt.title("Window count per speaker")
plt.xlabel("Number of Speakers")
plt.ylabel("Number of Windows")
plt.savefig("speaker_count_histogram.png")
