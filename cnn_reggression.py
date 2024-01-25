from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import regularizers
import numpy as np
import math
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler , EarlyStopping
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import cv2
from keras import backend as K


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
    
model = Sequential()

# Dodanie warstw konwolucyjnych i warstw max pooling
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(200, 200, 3)))  # Pierwsza warstwa konwolucyjna
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3), activation='relu'))  # Druga warstwa konwolucyjna
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(128, (3, 3), activation='relu'))  # Druga warstwa konwolucyjna
model.add(MaxPooling2D(pool_size=(3,3)))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

def load_data(data_folder):
    images = []
    ages = []
    for filename in os.listdir(data_folder):
            img = cv2.imread(os.path.join(data_folder, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img = cv2.resize(img, (200, 200))  
            age = int(filename.split('_')[0])  
            images.append(img)
            age = age
            ages.append(age)
    return np.array(images), np.array(ages)


data_folder="./../crop_part1"
images, ages = load_data(data_folder)

train_images, test_images, train_labels, test_labels = train_test_split(images, ages, test_size=0.2, random_state=42)

initial_learning_rate = 0.01 
epochs = 10
batch_size = 128

def lr_scheduler(epoch, lr):
    new_learning_rate = lr * 0.9 
    return new_learning_rate

optimizer = Adam(learning_rate=initial_learning_rate)

lr_callback = LearningRateScheduler(lr_scheduler)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
def accuracy(y_true, y_pred, tolerance=5): 
    return K.mean(K.abs(y_pred - y_true) < tolerance)

model.compile(optimizer, loss='mean_squared_error', metrics=[accuracy])
model.summary()

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels), callbacks=[])

predictions = model.predict(test_images)

for i in range(len(test_images)):
    pred_label = predictions[i]
    true_label = test_labels[i]
    
    print(f"Przewidywana etykieta: {pred_label}, Prawdziwa etykieta: {true_label}")

model.save('reggresion_model.keras')

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()