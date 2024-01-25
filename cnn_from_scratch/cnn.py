import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Ścieżka do folderu z danymi UTKFace
data_folder = './crop_part1'

# Funkcja do wczytania danych z folderu
def load_data(data_folder):
    images = []
    ages = []
    for filename in os.listdir(data_folder):
       # if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(data_folder, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konwersja kolorów BGR do RGB
            img = cv2.resize(img, (64, 64))  # Zmniejszenie rozmiaru obrazu do 64x64
            age = int(filename.split('_')[0])  # Wiek jest pierwszym elementem w nazwie pliku
            img = img / 255.0 
            img = img.reshape( 64 *64* 3,1)

            images.append(img)
            age= to_categorical(110)
            age= age.reshape(111,1)
            ages.append(age)
    return np.array(images), np.array(ages)

def preprocess_data(x, y, limit):
    return x[:limit], y[:limit]
def preprocess_Train_data(x, y):
    return x[::4], y[::4]
# Wczytanie danych
images, ages = load_data(data_folder)
#ages= ages.reshape(ages.shape[1],ages.shape[0])
# Normalizacja danych obrazów

# Podział danych na zbiory treningowe i testowe
#train_images, test_images, train_labels, test_labels = train_test_split(images, ages, test_size=0.2, random_state=42)
train_images, train_labels= preprocess_Train_data(images, ages)
test_images, test_labels= preprocess_data(images, ages,1000)

# Dekodowanie etykiet wiekowych do postaci kategorialnej
num_classes = 110

print(train_images.shape)
print(train_labels.shape)
# train_labels = to_categorical(train_labels, num_classes=num_classes)
# test_labels = to_categorical(test_labels, num_classes=num_classes)


# Model sieci neuronowej
network = [
    Dense(64 * 64*3, 128),  # Zmiana rozmiaru warstwy wejściowej na 48x48
    Tanh(),
    Dense(128, 111),  # Wyjście ma tylko 1 neuron, ponieważ przewidujemy wiek
]

# Funkcje kosztu i uczenia
def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def mse_prime(predictions, targets):
    return 2 * (predictions - targets) / predictions.size

# Trenowanie
train(network,mse, mse_prime, train_images, train_labels, epochs=100, learning_rate=0.1)

# Testowanie
for x, y in zip(test_images, test_labels):
    output = predict(network, x)
    print('predicted age:',np.argmax(output), '\ttrue age:', np.argmax(y))
