import numpy as np
import os
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image  # zmiana nazwy modułu image na keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Wczytanie wytrenowanego modelu ResNet50
model = ResNet50(weights='imagenet')

# Funkcja do przetwarzania obrazów
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Zmniejszenie rozmiaru obrazu do 64x64
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

data_folder = './crop_part1'

# Funkcja do wczytania danych z folderu
def load_data(data_folder):
    images = []
    ages = []
    for filename in os.listdir(data_folder):
       # if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(data_folder, filename))
            img = preprocess_image(img)
            age = int(filename.split('_')[0])  # Wiek jest pierwszym elementem w nazwie pliku
            images.append(img)
            ages.append(age)
    return np.array(images), np.array(ages)

def preprocess_data(x, y, limit):
    return x[:limit], y[:limit]
def preprocess_Train_data(x, y):
    return x[::8], y[::8]

def preprocess_Test_data(x, y):
    return x[::20], y[::20]
# Wczytanie danych
images, ages = load_data(data_folder)

train_images, train_labels= preprocess_Train_data(images, ages)
test_images, test_labels= preprocess_Test_data(images, ages)

# Estymacja wieku za pomocą ResNet50
for x, y in zip(test_images, test_labels):
    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Dekodowanie predykcji
    print(f"pred: {decoded_predictions}, true: {y}")
