import tensorflow as tf
from tensorflow import keras
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

bb_path = pathlib.Path("/Users/philipp/ALASKA2")
img_count = len(list(bb_path.glob("train/*/*.jpg")))
img_width = 512
img_height = 512
batch_size = 64
epochs = 15
steps_per_epoch = np.ceil(img_count / (batch_size * 20))

kernel_hp = np.array(
    [[[ -1], [  2], [ -2], [  2], [ -1]],
     [[  2], [ -6], [  8], [ -6], [  2]],
     [[ -2], [  8], [-12], [  8], [ -2]],
     [[  2], [ -6], [  8], [ -6], [  2]],
     [[ -1], [  2], [ -2], [  2], [ -1]]]) * (1.0/12.0)

def apply_hpf(image):
  image = ndimage.convolve(image, kernel_hp)
  return image

if __name__ == "__main__":
  train_image_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    preprocessing_function=apply_hpf
  )

  train_data_gen = train_image_gen.flow_from_directory(
    batch_size=batch_size,
    directory="/Users/philipp/ALASKA2/train",
    shuffle=True,
    target_size=(512, 512),
    class_mode="binary",
    color_mode="grayscale"
  )

  print(train_data_gen)

  model = keras.Sequential([
    keras.layers.Conv2D(64, (7, 7), strides=2, padding="same", activation="relu", input_shape=(512, 512, 1)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(16, (5, 5), padding="same", activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
  ])

  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  print(model.summary())

  model.fit(train_data_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, shuffle=False)
