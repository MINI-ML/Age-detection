import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Wczytanie danych z folderu
data_path = './crop_part1'
images = []
ages = []

for filename in os.listdir(data_path):
    img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_GRAYSCALE)  # Wczytanie zdjęcia jako obrazu w skali szarości
    if img is not None:
        img = cv2.resize(img, (64, 64))  # Dostosowanie rozmiaru zdjęcia
        images.append(img)
        age = int(filename.split('_')[0])  # Pobranie wieku z nazwy pliku
        ages.append(age)

# Przetworzenie danych
images = np.array(images)
images = images.reshape(images.shape[0], 64, 64, 1)  # Dopasowanie kształtu do wymagań modelu CNN
ages = np.array(ages)
ages = ages / 110.0  # Normalizacja wieku, np. do przedziału od 0 do 1

# Podział danych na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

# Budowa modelu CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))  # Wyjście to wiek

# Kompilacja modelu
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Trening modelu
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Testowanie modelu
loss, mae = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}, Test MAE: {mae}")

# Przewidywanie wieku dla zdjęć testowych i porównanie z prawdziwymi wartościami
predictions = model.predict(X_test)

for i in range(len(predictions)):
    print(f"Prawdziwy wiek: {y_test[i] * 110}, Przewidziany wiek: {predictions[i][0] * 110}")
