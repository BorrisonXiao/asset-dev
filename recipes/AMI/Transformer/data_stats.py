import datasets
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

dataset = datasets.load_from_disk("coalesced_5/ami_sdm_overlap_windows_test_coalesced")

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
