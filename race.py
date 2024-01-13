import matplotlib
matplotlib.use('TkAgg')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

image_folder = "C:\\MiNI\\Semestr 5\\ML\\cropped_faces\\crop_part1\\crop_sampled"

# Function to categorize race based on filename parts
def categorize_race(race_num):
    if race_num == '0':
        return 'white'
    elif race_num == '1':
        return 'black'
    elif race_num == '2':
        return 'asian'
    elif race_num == '3':
        return 'indian'
    elif race_num == '4':
        return 'others'
    else:
        return 'unknown'

# Initialize dictionaries to store histograms for each color channel and race
race_histograms = defaultdict(lambda: defaultdict(list))

# Iterate through each image in the folder
for image_name in os.listdir(image_folder):
    parts = image_name.split('_')
    race = categorize_race(parts[2])

    # Read the image
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate histograms for each color channel
    for i, col in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        race_histograms[race][col].append(hist)

# Plot histograms for races and color channels
plt.figure(figsize=(15, 10))
max_frequency = 1000  # Set maximum y-axis value

# Channel colors mapping
channel_colors = {'r': 'red', 'g': 'green', 'b': 'blue'}

for idx, race in enumerate(race_histograms, start=1):
    plt.subplot(2, 3, idx)
    for col in ['r', 'g', 'b']:
        avg_hist = np.mean(race_histograms[race][col], axis=0)
        plt.plot(avg_hist, label=col.upper(), color=channel_colors[col])
        plt.title(f'Race: {race}')
        plt.xlim([0, 256])
        plt.ylim([0, max_frequency])

plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
