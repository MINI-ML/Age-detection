from PIL import Image
import cv2
import os
import numpy as np

def read_annotations(annotation_path):
    with open(annotation_path, 'r') as file:
        annotations = [line.strip().split() for line in file]
    return annotations

def detect_faces(face_cascade, gray, scaleFactor, minNeighbors):
    return face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(30, 30))

def evaluate_face_detection(images_folder, ground_truth_path, scaleFactor, minNeighbors):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read ground truth annotations
    ground_truth_annotations = read_annotations(ground_truth_path)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for image_info in ground_truth_annotations:
        filename = image_info[0]
        image_path = os.path.join(images_folder, filename)

        pil_image = Image.open(image_path)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces_detected = detect_faces(face_cascade, gray, scaleFactor, minNeighbors)

        # Extract ground truth coordinates for each feature
        gt_coordinates = [(int(float(image_info[i])), int(float(image_info[i+1]))) for i in range(1, len(image_info), 2)]

        # Check if any face is detected
        if len(faces_detected) > 0:
            # Check if any detection overlaps with the ground truth for any feature
            detected_overlap = any(
                any(x <= gt_x and x + w >= gt_x and y <= gt_y and y + h >= gt_y for (x, y, w, h) in faces_detected)
                for (gt_x, gt_y) in gt_coordinates
            )

            if detected_overlap:
                true_positives += 1
            else:
                false_positives += 1
        else:
            false_negatives += 1

    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return true_positives, false_positives, false_negatives, precision, recall, f1_score

    # print(f"True Positives: {true_positives}")
    # print(f"False Positives: {false_positives}")
    # print(f"False Negatives: {false_negatives}")
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall: {recall:.2f}")
    # print(f"F1 Score: {f1_score:.2f}")

def evaluate_parameters(images_folder, ground_truth_path):
    for scaleFactor in np.arange(1.1, 1.5, 0.1):
        for minNeighbor in range(2, 5):
            print(f"\nEvaluating with scaleFactor={scaleFactor} and minNeighbors={minNeighbor}")
            true_positives, false_positives, false_negatives, precision, recall, f1_score = (
                evaluate_face_detection(images_folder, ground_truth_path, scaleFactor, minNeighbor))
            print(f"True Positives: {true_positives}")
            print(f"False Positives: {false_positives}")
            print(f"False Negatives: {false_negatives}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1_score:.2f}")

def main():
    test_sets = ['A_test', 'B_test-low', 'C_newtest', 'rotated']
    # images_folder = 'C:\\Users\\glowa\\Desktop\\Studia\\IML\\test-images\\A_test'
    images_folder = 'C:\\Users\\glowa\\Desktop\\Studia\\IML\\test-images\\' + test_sets[2]
    ground_truth_path = images_folder + '\\ground_truths.txt'
    # ground_truth_path = 'C:\\Users\\glowa\\Desktop\\Studia\\IML\\test-images\\rotated\\ground_truths.txt'
    # ground_truth_path = 'C:\\Users\\glowa\\Desktop\\Studia\\IML\\test-images\\A_test\\ground_truths.txt'
    # evaluate_face_detection(images_folder, ground_truth_path)
    evaluate_parameters(images_folder, ground_truth_path)

main()
