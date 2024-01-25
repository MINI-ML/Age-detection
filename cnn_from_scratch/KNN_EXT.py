import os
import cv2
import sys
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image  # zmiana nazwy modułu image na keras_image

from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# Wczytanie modelu ResNet50 bez warstw klasyfikacyjnych (top layers)
model = ResNet50(weights='imagenet', include_top=False)

# Funkcja do wczytywania i przetwarzania obrazów
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Przetwarzanie obrazu i ekstrakcja cech
# processed_img = preprocess_image(img_path)
# features = model.predict(processed_img)

# Zapisanie wyekstrahowanych cech do pliku (opcjonalnie)
#np.save('wyekstrahowane_cechy.npy', features)


# Ścieżka do foldera z obrazami
folder_path = './crop_part1'

# Przygotowanie listy przechowującej obrazy i wiek
images = []
ages = []

# Iteracja przez pliki w folderze
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        # Wczytanie obrazu przy użyciu OpenCV
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        # Konwersja do skalowanej wersji o ustalonym rozmiarze (np. 100x100 pikseli)
        resized_image = cv2.resize(image, (100, 100))
        resized_image = resized_image/255.0
        # Dodanie obrazu do listy
        processed_img = preprocess_image(resized_image)
        orig_stdout = sys.stdout

        sys.stdout = open(os.devnull, 'w')

        features = model.predict(processed_img)
        sys.stdout = orig_stdout

        images.append(features)
        
        # Odczytanie wieku z nazwy pliku lub innych źródeł informacji (np. metadanych)
        age = int(filename.split('_')[0])  # Pobranie wieku z nazwy pliku
        ages.append(age)

# Przekształcenie list na tablice numpy
X = np.array(images)
y = np.array(ages)

# Zmiana kształtu tablicy na wymagany przez KNN (np. spłaszczenie obrazów)
X_flattened = X.reshape(X.shape[0], -1)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, train_size=0.2,test_size=0.1, random_state=42)

# Tworzenie modelu KNN
k = 5  # liczba sąsiadów
knn_model = KNeighborsRegressor(n_neighbors=k)

# Trenowanie modelu
knn_model.fit(X_train, y_train)

# Predykcja wieku na zbiorze testowym
predictions = knn_model.predict(X_test)

# Obliczenie błędu
mae = mean_absolute_error(y_test, predictions)
print(f"Średni błąd bezwzględny (MAE): {mae}")

# Wypisanie predykcji i prawdziwych wartości dla każdego zdjęcia
for i in range(len(predictions)):
    print(f"Obraz {i+1}: Predykcja wieku: {predictions[i]}, Prawdziwy wiek: {y_test[i]}")