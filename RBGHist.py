import matplotlib
matplotlib.use('TkAgg')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

image_folder = "C:\\MiNI\\Semestr 5\\ML\\cropped_faces\\crop_part1\\crop_sampled"

# Function to categorize age
def categorize_age(age):
    if age <= 10:
        return '0-10'
    elif age <= 20:
        return '11-20'
    elif age <= 30:
        return '21-30'
    elif age <= 40:
        return '31-40'
    elif age <= 50:
        return '41-50'
    elif age <= 60:
        return '51-60'
    elif age <= 70:
        return '61-70'
    else:
        return '70+'

# Initialize dictionaries to store histograms for each color channel
age_histograms = defaultdict(lambda: defaultdict(list))

# Grupy wiekowe, dla których chcesz wyświetlić histogramy
desired_age_groups_1 = ['0-10', '11-20', '21-30', '31-40']  # Pierwsze cztery grupy
desired_age_groups_2 = ['41-50', '51-60', '61-70', '70+']    # Pozostałe grupy

# Iterate through each image in the folder
for image_name in os.listdir(image_folder):
    # Extract age from the filename
    age = int(image_name.split('_')[0])
    age_category = categorize_age(age)

    # Check if the current age category is in the desired groups
    if age_category in desired_age_groups_1 or age_category in desired_age_groups_2:
        # Read the image
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate histograms for each color channel
        for i, col in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            age_histograms[age_category][col].append(hist)

# Plot histograms for desired age groups and color channels
plt.figure(figsize=(15, 10))

# Ustalenie maksymalnej wartości dla osi Y
max_frequency = 1000  # Ustawienie maksymalnej wartości Y

# Mapowanie nazw kanałów na kolory
channel_colors = {'r': 'red', 'g': 'green', 'b': 'blue'}

# Pierwsze cztery grupy
for idx, age_category in enumerate(desired_age_groups_1, start=1):
    plt.subplot(2, 4, idx)
    for col in ['r', 'g', 'b']:
        avg_hist = np.mean(age_histograms[age_category][col], axis=0)
        plt.plot(avg_hist, label=col.upper(), color=channel_colors[col])
        plt.title(f'Age Group: {age_category}')
        plt.xlim([0, 256])
        plt.ylim([0, max_frequency])  # Ustawienie maksymalnej wartości osi Y

# Pozostałe grupy
for idx, age_category in enumerate(desired_age_groups_2, start=5):
    plt.subplot(2, 4, idx)
    for col in ['r', 'g', 'b']:
        avg_hist = np.mean(age_histograms[age_category][col], axis=0)
        plt.plot(avg_hist, label=col.upper(), color=channel_colors[col])
        plt.title(f'Age Group: {age_category}')
        plt.xlim([0, 256])
        plt.ylim([0, max_frequency])  # Ustawienie maksymalnej wartości osi Y

plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
