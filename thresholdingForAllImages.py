import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Path to the directory containing photos
photos_directory = "C:\\MiNI\\Semestr 5\\ML\\cropped_faces\\crop_part1\\crop_part1"

# Define age groups
age_groups = {
    '0-10': (0, 10),
    '11-20': (11, 20),
    '21-30': (21, 30),
    '31-40': (31, 40),
    '41-50': (41, 50),
    '51-60': (51, 60),
    '61-70': (61, 70),
    '70+': (70, float('inf'))
}

# Dictionary to store black and white pixel proportions for each age group
black_white_proportions = {age_group: {'black': 0, 'white': 0} for age_group in age_groups}
total_pixels_per_age_group = {age_group: 0 for age_group in age_groups}

# Iterate through each file in the photos directory
for filename in os.listdir(photos_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Get the full path of the image
        image_path = os.path.join(photos_directory, filename)

        # Extract age from the filename
        age = int(filename.split('_')[0])

        # Read the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a simple thresholding technique (adjust threshold value as needed)
        _, thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)

        # Calculate black and white pixel proportions
        for age_group, (start_age, end_age) in age_groups.items():
            if start_age <= age <= end_age:
                black_pixels = np.count_nonzero(thresholded_image == 0)
                white_pixels = np.count_nonzero(thresholded_image == 255)
                black_white_proportions[age_group]['black'] += black_pixels
                black_white_proportions[age_group]['white'] += white_pixels
                total_pixels_per_age_group[age_group] += (black_pixels + white_pixels)

# Normalize black and white pixel proportions for each age group
for age_group in black_white_proportions:
    total_pixels = total_pixels_per_age_group[age_group]
    black_white_proportions[age_group]['black'] /= total_pixels
    black_white_proportions[age_group]['white'] /= total_pixels

# Prepare data for plotting
age_group_labels = list(black_white_proportions.keys())
black_proportions = [proportions['black'] for proportions in black_white_proportions.values()]
white_proportions = [proportions['white'] for proportions in black_white_proportions.values()]

# Plotting the bar chart
bar_width = 0.35
index = np.arange(len(age_group_labels))

plt.figure(figsize=(10, 6))
plt.bar(index, black_proportions, bar_width, label='Black', color='black')
plt.bar(index + bar_width, white_proportions, bar_width, label='White', color='white', edgecolor='black')

plt.xlabel('Age Groups')
plt.ylabel('Pixel Proportions')
plt.title('Normalized Black and White Pixel Proportions in Different Age Groups')
plt.xticks(index + bar_width / 2, age_group_labels)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
