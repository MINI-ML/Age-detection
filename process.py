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

num_of_classes=0
cumMAPMTCNN=0
cumHAARE=0

cumMAPMTCNN_Recall=0
cumHAARE_Recall=0

precisionsHaar = []
precisionsMTCNN = []
recallsHaar = []
recallsMTCNN = []

TPM = 0
FNM = 0
FPM= 0


TPH = 0
FNH = 0
FPH = 0

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
    # Sortowanie predykcji po pewności
    #pred_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Liczenie precision i recall
    for pred_box in pred_boxes:
        iou_max = 0
        match = None
        
        for gt_box in gt_boxes:
            iou = calculate_iou(gt_box, pred_box[:4])
            #print(iou)
            if iou > iou_max:
                iou_max = iou
                match = gt_box
        
        if iou_max >= iou_threshold:
            true_positives += 1
            gt_boxes.remove(match) 
        else:
            false_positives += 1
    precision=0
    recall = 0
    false_negatives = len(gt_boxes)  # Pozostałe niewykryte pudełka to false negatives

    if true_positives !=0:
        precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    
    return precision, recall,true_positives,false_positives,false_negatives



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = MTCNN()

y_true = []
y_scores = []
y_scoresMTCNN = []

file_path = './WIDER_val/wider_face_val_bbx_gt.txt' 
with open(file_path, 'r') as file:
    lines = file.readlines()


current_line = 0
while current_line < len(lines):
    file_name = lines[current_line].strip() 
    num_boxes = int(lines[current_line + 1]) 

    random_number = random.randint(1, 30)

    if random_number == 1: 
        num_of_classes+=1
        for i in range(num_boxes):
            box_info = lines[current_line + 2 + i].split()  
            x, y, w, h = map(int, box_info[:4])  
            other_info = list(map(int, box_info[4:])) 

        img = Image.open("./WIDER_val/images/"+file_name)

        image = cv2.imread("./WIDER_val/images/"+file_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1)
        ax.imshow(img)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = detector.detect_faces(image_rgb)
        num_detected_facesMTCNN = len(detections)


        faces = face_cascade.detectMultiScale(gray)
        num_detected_faces = len(faces)

        y_true = np.append(y_true,num_boxes)

        y_scores.append(num_detected_faces/num_boxes)
        y_scoresMTCNN.append(num_detected_facesMTCNN/num_boxes)
        TrueBoxes=[]

        for i in range(num_boxes):
            box_info = lines[current_line + 2 + i].split()  
            x, y, w, h = map(int, box_info[:4]) 
            TrueBoxes.append([x,y,x+w,y+h])

            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')


            ax.add_patch(rect)
        boxesA=[]
        boxesB=[]
        for (x, y, w, h) in faces:
            boxesA.append([x,y,x+w,y+h,1])
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')

            ax.add_patch(rect)    
        for face in detections:
                x, y, w, h = face['box']
                boxesB.append([x,y,x+w,y+h,1])
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)  
        #plt.show()
        (precisonH,recallH,tp1,fp1,fn1) = calculate_precision_recall(TrueBoxes.copy(),boxesA)
        (precisonM,recallM,tp2,fp2,fn2) = calculate_precision_recall(TrueBoxes.copy(),boxesB)

        TPH+=tp1
        FPH+=fp1
        FNH+=fn1

        TPM+=tp2
        FPM+=fp2
        FNM+=fn2

        cumMAPMTCNN+=precisonM
        cumHAARE+=precisonH

        cumMAPMTCNN_Recall+=recallM
        cumHAARE_Recall+=recallH

        precisionsHaar.append(precisonH)
        precisionsMTCNN.append(precisonM)

        
        recallsHaar.append(recallH)
        recallsMTCNN.append(recallM)
        # Wyświetl obraz z bounding boxami
        print(f"Preicison for haarCascade: {(precisonH)}") 
        print(f"Preicison for MTCNN: {(precisonM)}") 

        print(f"Recall for haarCascade: {(recallH)}") 
        print(f"Recall for MTCNN: {(recallM)}") 
        #print(num_detected_faces/num_boxes)
        #print(num_detected_facesMTCNN/num_boxes)
        plt.show()


    current_line += num_boxes + 2  # Przesunięcie do kolejnej nazwy pliku


precisionHaare = cumHAARE/num_of_classes
precisionMTCNN = cumMAPMTCNN/num_of_classes

recallHaare = cumHAARE_Recall/num_of_classes
recallMTCNN = cumMAPMTCNN_Recall/num_of_classes

F1Haare = 2*(precisionHaare*recallHaare)/(precisionHaare+recallHaare)
F1_Mtcnn = 2*(precisionMTCNN*recallMTCNN)/(precisionMTCNN+recallMTCNN)

print(num_of_classes)
print(f"Mean Average Precision for haarCascad: {precisionHaare}")
print(f"Mean Average Precision for MTCNN: {precisionMTCNN}")


pH = TPH/(TPH+FPH)
pM = TPM/(TPM+FPM)
print(f"Precision for haarCascad: {pH}")
print(f"Precision for MTCNN: {pM}")


rH = TPH/(TPH+FNH)
rM = TPM/(TPM+FNM)
print(f"Recall for haarCascad: {rH}")
print(f"Recall for MTCNN: {rM}")


f1H = (2*rH*pH)/(rH+pH)
f1M = (2*rM*pM)/(rM+pM)
print(f"F1 score for haarCascade: {f1H}")
print(f"F1 score for MTCNN: {f1M}")

# Stworzenie wykresu z punktami
plt.scatter(recallsHaar, precisionsHaar)


# Dodanie tytułu i etykiet osi
plt.title('Wykres Recall i Precision dla Haar')
plt.xlabel('Recall')
plt.ylabel('Precision')

# Wyświetlenie wykresu
plt.show()

plt.scatter(recallsMTCNN, precisionsMTCNN)
# Dodanie tytułu i etykiet osi
plt.title('Wykres Recall i Precision dla MTCNN')
plt.xlabel('Recall')
plt.ylabel('Precision')

# Wyświetlenie wykresu
plt.show()

# print("Haar:")
# print("TP "+TPH)
# print("FP "+FPH)
# print("FN "+FNH)
# print("MTCNN:")
# print("TP "+TPM)
# print("FP "+FPM)
# print("FN "+FNM)
