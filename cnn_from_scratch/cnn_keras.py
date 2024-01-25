from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
import numpy as np
import math
from sklearn.model_selection import train_test_split

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import cv2


def class_labels_reassign(age):
    if 1 <= age <= 2:
        return 0
    elif 3 <= age <= 9:
        return 1
    elif 10 <= age <= 20:
        return 2
    elif 21 <= age <= 27:
        return 3
    elif 28 <= age <= 45:
        return 4
    elif 46 <= age <= 65:
        return 5
    else:
        return 6
    
# Inicjalizacja modelu sekwencyjnego
model = Sequential()

# Dodanie warstw konwolucyjnych i warstw max pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))  # Pierwsza warstwa konwolucyjna
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))  # Druga warstwa konwolucyjna
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))  # Druga warstwa konwolucyjna
model.add(MaxPooling2D(pool_size=(2, 2)))
# Warstwa spłaszczająca (flatten)
model.add(Flatten())

# Warstwy w pełni połączone (fully connected)
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(111, activation='softmax'))  # Warstwa wyjściowa - przykład binarnej klasyfikacji

# Kompilacja modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Funkcja do wczytania danych z folderu
def load_data(data_folder):
    images = []
    ages = []
    for filename in os.listdir(data_folder):
       # if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(data_folder, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img = cv2.resize(img, (200, 200))  
            age = int(filename.split('_')[0])  
            images.append(img)
            ages.append(age)
    return np.array(images), np.array(ages)

# Wczytanie danych
data_folder="./crop_part1"
images, ages = load_data(data_folder)

#train_images, train_labels= (images, ages)
#test_images, test_labels= preprocess_Test_data(images, ages)
train_images, test_images, train_labels, test_labels = train_test_split(images, ages, test_size=0.2, random_state=42)

# Wyświetlenie architektury modelu
model.summary()

history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

predictions = model.predict(test_images)

# Przykład dla pojedynczej próbki
for i in range(len(test_images)):
    pred_label = np.argmax(predictions[i])  # Pobranie indeksu klasy o najwyższym prawdopodobieństwie
    true_label = test_labels[i]  # Prawdziwa etykieta dla danej próbki
    
    print(f"Przewidywana etykieta: {pred_label}, Prawdziwa etykieta: {true_label}")

# Wyświetlenie historii treningu
import matplotlib.pyplot as plt

# Wykres dokładności (accuracy)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Wykres funkcji straty (loss)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()