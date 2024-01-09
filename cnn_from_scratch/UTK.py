import numpy as np
import math

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import cv2
from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid , Softmax
from losses import cross_entropy, cross_entropy_prime
from network import train, predict

age_classification = [
    (0, 9), (10, 19), (20, 29), (30, 39), (40, 49),
    (50, 59), (60, 69), (70, 79), (80, 89), (90, 99), (100, 109) , (110,110)
]
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
            age = math.floor(age/10)
    
            # print("--argmax--")
            img = img.astype("float32") / 255
            img = img.reshape( 3,64,64)

            images.append(img)
            newAge =  np.zeros((12, 1))
            newAge[age] = 1
            #print(age)
            #print(newAge)
            ages.append(newAge)
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


# print(np.argmax(ages[0]))
# print(np.argmax(ages[1]))
# neural network
network = [
    Convolutional((3, 64, 64), 3, 5),
    Sigmoid(),
    Reshape((5, 62, 62), (5 * 62 * 62, 1)),
    Dense(5 * 62 * 62, 24),
    Sigmoid(),
    Dense(24, 12),
    Sigmoid()
]

# train
train(
    network,
    cross_entropy,
    cross_entropy_prime,
    train_images,
    train_labels,
    epochs=5,
    learning_rate=.1
)

# test
for x, y in zip(test_images, test_labels):
    output = predict(network, x)
    print(f"pred: {age_classification[np.argmax(output)]}, true: {age_classification[np.argmax(y)]}")
