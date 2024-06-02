# Azizu Ahmad Rozaki Riyanto
# azizu.rozaki@gmail.com

import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import pathlib
import cv2
from PIL import Image

np.random.seed(0)
tf.random.set_seed(0)

dataset_dir = "/kaggle/input/animal/raw-img"

category = os.listdir(dataset_dir)

length = len(category)
print(f"This Dataset have : {length} label")

print(f"\nThe label is: ")
for label in category:
    print(label)

totalData = 0

for label in category:
    # Create the full path to the label's directory
    label_dir = os.path.join(dataset_dir, label)

    # List all files in the label's directory
    file_list = os.listdir(label_dir)

    # Count the number of images for the label
    num_images = len(file_list)
    totalData = totalData + num_images

    # Print the label and the number of images
    print(f"Label {label} : {num_images}")

print(f"\nTotal Dataset : {totalData}")

for i in category:
    dir = os.path.join(dataset_dir, i)
    y = len(os.listdir(dir))
    print(i,'=', y)

    img_name = os.listdir(dir)
    for j in range(10):
        img_path = os.path.join(dir, img_name[j])
        img = Image.open(img_path)
        print(img.size)
    print('=========')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input as densePI

size = (250 ,250)
batch = 32
valSplit = 0.2

dataGenerator = ImageDataGenerator(
#     rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.3,
    zoom_range=0.3,
    fill_mode = 'nearest',
    preprocessing_function = densePI,
    validation_split = valSplit
)
train_data = dataGenerator.flow_from_directory(
    directory = dataset_dir,
    target_size= size,
    class_mode='categorical',
    batch_size = batch,
    shuffle = True,
    subset="training",
    seed = 0
)

valid_data = dataGenerator.flow_from_directory(
    directory = dataset_dir,
    target_size= size,
    class_mode='categorical',
    batch_size = batch,
    shuffle = False,
    subset="validation",
    seed = 0
)

x,y = train_data.next()
i,j = valid_data.next()

print(f"Train shape : {x.shape},{y.shape} \n")
print(f"Valid shape : {i.shape},{j.shape} \n")

from tensorflow.keras.optimizers.experimental import Adamax
from tensorflow.keras.models import Model
import time
from sklearn.metrics import classification_report, confusion_matrix ,multilabel_confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

def grafik(hist):

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.plot(loss, 'b', label='Training loss')
    plt.plot(val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f"loss.png"))
    plt.show()


    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(acc, 'b', label='Training acc')
    plt.plot(val_acc, 'r', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'lower right')
    plt.savefig(os.path.join(folder_path, f"accuracy.png"))
    plt.show()

input_shape = (250, 250, 3)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

for layer in base_model.layers :
    layer.trainable = True

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer= Adamax(learning_rate = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

def lr_schedule(epoch, initial_lr=0.0001):
    return initial_lr * 0.9 ** epoch

folder_path = f"run/"

callbacks = [
            tf.keras.callbacks.ModelCheckpoint(os.path.join(folder_path, f"best_model.h5"), save_best_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir='logs'),
            tf.keras.callbacks.LearningRateScheduler(lr_schedule)
            ]

# Hitung waktu training
start_time = time.time()

# Training
history = model.fit(
    train_data,
    epochs=50,
    validation_data=(valid_data),
    callbacks=callbacks,
    batch_size=32,
)

# Hitung waktu training
end_time = time.time()

hsl = end_time - start_time

# Menampilkan lama proses training
print(f"Training Time : {hsl}")
print(round(hsl/60) , ' Menit')
print(round(hsl % 60), ' Detik')

from sklearn.metrics import accuracy_score

model = load_model('/kaggle/working/run/best_model.h5')

y_pred = model.predict(valid_data)
y_true = valid_data.classes
y_pred = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_true, y_pred)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")

classification_rep = classification_report(y_true, y_pred, target_names=valid_data.class_indices)
print(classification_rep)

grafik(history)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with tf.io.gfile.GFile('modeltflite.tflite', 'wb') as f:
    f.write(tflite_model)

