import tensorflow as tf
from tensorflow import keras
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
BB_HOME = pathlib.Path("/tmp/BOSSbase")

class_names = np.array([item.name for item in BB_HOME.glob("train/*") if item.name != ".DS_Store"])
image_count = len(list(BB_HOME.glob("train/*/*.pgm")))

BATCH_SIZE = 32
EPOCHS = 10
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

kernel_hp = np.array(
    [[[ -1], [  2], [ -2], [  2], [ -1]],
     [[  2], [ -6], [  8], [ -6], [  2]],
     [[ -2], [  8], [-12], [  8], [ -2]],
     [[  2], [ -6], [  8], [ -6], [  2]],
     [[ -1], [  2], [ -2], [  2], [ -1]]]) * (1.0/12.0)

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  if parts[-2] == "cover": return np.array([0, 1])
  else: return np.array([1, 0])
  #return parts[-2] == class_names

def get_image(file_path):
  image = keras.preprocessing.image.load_img(file_path, color_mode="grayscale")
  image = keras.preprocessing.image.img_to_array(image) / 255.0
  image = ndimage.convolve(image, kernel_hp)
  image = np.delete(image, [0, 1, 510, 511], 0)
  image = np.delete(image, [0, 1, 510, 511], 1)
  return image

def process_path(file_path):
  label = get_label(file_path)
  image = get_image(file_path)
  return image, label

def set_shapes(img, label):
  img.set_shape((508, 508, 1))
  label.set_shape((2))
  return img, label

if __name__ == "__main__":
  ds_train_list = tf.data.Dataset.list_files(os.path.join(BB_HOME, "train/*/*.pgm"))
  train_dataset = ds_train_list.map(lambda x: tf.numpy_function(process_path, [x], [tf.float32, tf.int64]), num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.map(lambda i, l: set_shapes(i, l))

  print("\nDATASET:", train_dataset.element_spec)

  for image, label in train_dataset.take(1):
    print("\nIMAGE SHAPE:", image)
    print("\nLABEL SHAPE:", label)

  model = keras.Sequential([
    keras.layers.Conv2D(64, (7, 7), strides=2, padding="same", activation="relu", input_shape=(508, 508, 1)),
    keras.layers.Conv2D(16, (5, 5), padding="same", activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation="softmax")
  ])

  model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
  )

  print("\nMODEL:", model.summary())
  train_dataset = train_dataset.shuffle(image_count).repeat(EPOCHS).batch(BATCH_SIZE)
  model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
  