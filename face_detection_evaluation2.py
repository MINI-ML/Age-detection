import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from mtcnn import MTCNN
import random
import statistics

num_of_classes = 0
cumMAPMTCNN = 0
cumHAARE = 0


def calculate_iou(boxA, boxB):
    # boxA, boxB w formacie [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Obliczenie obszaru wspólnego
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Obliczenie obszaru obu boxów
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Obliczenie IoU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou


def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    total_gt_boxes = len(gt_boxes)
    # print(len(pred_boxes))
    # Sortowanie predykcji po pewności
    # pred_boxes.sort(key=lambda x: x[4], reverse=True)

    # Liczenie precision i recall
    for pred_box in pred_boxes:
        iou_max = 0
        match = None

        for gt_box in gt_boxes:
            iou = calculate_iou(gt_box, pred_box[:4])
            # print(iou)
            if iou > iou_max:
                iou_max = iou
                match = gt_box

        if iou_max >= iou_threshold:
            true_positives += 1
            gt_boxes.remove(match)
        else:
            false_positives += 1
    precision = 0
    # print(true_positives)
    if true_positives != 0:
        precision = true_positives / (true_positives + false_positives)

    recall = 0

    return precision, recall


# def calculate_average_precision(gt_boxes, pred_boxes, iou_threshold=0.5):
#     precisions = []
#     recalls = []

#     for class_pred_boxes in pred_boxes:
#         precision, recall = calculate_precision_recall(gt_boxes, class_pred_boxes, iou_threshold)
#         precisions.append(precision)
#         recalls.append(recall)

#     # Tworzenie krzywej precision-recall
#     # ... (implementacja tworzenia krzywej precision-recall)

#     # Obliczanie Average Precision (AP)
#     # ... (implementacja obliczania AP dla danej klasy)

#     return average_precision

# Obliczenie mAP
# mAP = calculate_average_precision(ground_truth_boxes, predicted_boxes)
# print(f"Mean Average Precision (mAP): {mAP}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Inicjalizacja modelu MTCNN
detector = MTCNN()

y_true = []
y_scores = []
y_scoresMTCNN = []
# Otwórz plik
file_path = './WIDER_val/wider_face_val_bbx_gt.txt'  # Zmień na ścieżkę do twojego pliku
with open(file_path, 'r') as file:
    lines = file.readlines()

# Przetwarzanie danych
current_line = 0
while current_line < len(lines):
    file_name = lines[current_line].strip()  # Nazwa pliku obrazu
    num_boxes = int(lines[current_line + 1])  # Liczba bounding boxów
    # Losowanie liczby z zakresu od 1 do 8
    random_number = random.randint(1, 30)
    # print(random_number)
    # Sprawdzenie, czy wylosowana liczba to 1 (prawdopodobieństwo 1/8)
    if random_number == 1:
        num_of_classes += 1
        for i in range(num_boxes):
            box_info = lines[current_line + 2 + i].split()  # Informacje o boxie
            x, y, w, h = map(int, box_info[:4])  # Współrzędne i wymiary
            other_info = list(map(int, box_info[4:]))  # Pozostałe informacje

            # print(f"Box {i + 1}: (x={x}, y={y}), Width: {w}, Height: {h}, Other Info: {other_info}")
            # Ścieżka do obrazu

        # Otwórz obraz
        img = Image.open("./WIDER_val/images/" + file_name)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = cv2.imread("./WIDER_val/images/" + file_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Stwórz subplot i dodaj obraz
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Przewidź twarze na obrazie przy użyciu modelu MTCNN

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = detector.detect_faces(image_rgb)
        num_detected_facesMTCNN = len(detections)

        faces = face_cascade.detectMultiScale(gray)
        num_detected_faces = len(faces)
        # Zbierz rzeczywiste i przewidywane bounding boxy
        y_true = np.append(y_true, num_boxes)
        # y_scores = np.append(y_scores,num_detected_faces)
        # y_scoresMTCNN = np.append(y_scoresMTCNN,num_detected_facesMTCNN)
        y_scores.append(num_detected_faces / num_boxes)
        y_scoresMTCNN.append(num_detected_facesMTCNN / num_boxes)
        TrueBoxes = []
        # Dodaj bounding boxy
        for i in range(num_boxes):
            box_info = lines[current_line + 2 + i].split()  # Informacje o boxie
            x, y, w, h = map(int, box_info[:4])  # Współrzędne i wymiary
            TrueBoxes.append([x, y, x + w, y + h])
            # Utwórz prostokąt reprezentujący bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

            # Dodaj prostokąt do obrazu
            ax.add_patch(rect)
        boxesA = []
        boxesB = []
        for (x, y, w, h) in faces:
            boxesA.append([x, y, x + w, y + h, 1])
            # Utwórz prostokąt reprezentujący bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')

            # Dodaj prostokąt do obrazu
            ax.add_patch(rect)
        for face in detections:
            x, y, w, h = face['box']
            boxesB.append([x, y, x + w, y + h, 1])
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')

            # Dodaj prostokąt do obrazu
            ax.add_patch(rect)
        (precisonH, recallH) = calculate_precision_recall(TrueBoxes.copy(), boxesA)
        (precisonM, recallM) = calculate_precision_recall(TrueBoxes.copy(), boxesB)
        cumMAPMTCNN += precisonM
        cumHAARE += precisonH

        # Wyświetl obraz z bounding boxami
        print(f"Preicison for haarCascade: {(precisonH)}")
        print(f"Preicison for MTCNN: {(precisonM)}")
        # print(num_detected_faces/num_boxes)
        # print(num_detected_facesMTCNN/num_boxes)
        # plt.show()

    current_line += num_boxes + 2  # Przesunięcie do kolejnej nazwy pliku
# Konwersja do tablic numpy

# y_true = np.array(y_true)
# y_scores = np.array(y_scores)
# y_scoresMTCNN = np.array(y_scoresMTCNN)
# Obliczanie średniej precyzji (mAP)
# y_true = y_true.reshape(-1, 1)
# y_scores = y_scores.reshape(-1, 1)
# y_scoresMTCNN = y_scoresMTCNN.reshape(-1, 1)

# mAP = average_precision_score(y_true, y_scores, average='macro')
# mAPMTCNN = average_precision_score(y_true,y_scoresMTCNN, average='macro')

# print(y_scores)
print(num_of_classes)
print(f"Mean Average Precision for haarCascade: {cumHAARE / num_of_classes}")
print(f"Mean Average Precision for MTCNN: {cumMAPMTCNN / num_of_classes}")