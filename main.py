import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import jpegio as jio
import os
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

input_size = 15000 # for each category
tt_ratio = 0.9

train_path = "/tmp/ALASKA2/train/"
cover_label = "Cover_75"
stego_label = "UERD_75"

cover_path = train_path + cover_label
cover_list = [os.path.abspath(os.path.join(cover_path, p)) for p in os.listdir(cover_path)]
cover_list = cover_list[:input_size]

stego_path = train_path + stego_label
stego_list = [os.path.abspath(os.path.join(stego_path, p)) for p in os.listdir(stego_path)]
stego_list = stego_list[:input_size]

input_list = cover_list + stego_list
random.shuffle(input_list)

img_count = len(input_list)
train_img_count = (int)(img_count * tt_ratio)
input_list_train = input_list[:train_img_count]
input_list_test = input_list[train_img_count:]

print("\nTOTAL IMAGES:", img_count)
print("\nTRAINING DATA:", len(input_list_train))
print("\nTEST DATA:", len(input_list_test))

img_width = 512
img_height = 512
batch_size = 32
epochs = 25
steps_per_epoch = train_img_count // batch_size

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  if parts[-2] == cover_label: 
    label = np.array([0.])
  else: 
    label = np.array([1.])
  return label

def get_img_as_ycbcr(file_path):
  image = Image.open(file_path)
  image.draft("YCbCr", None)
  image.load()
  image = np.array(image) / 255.
  return image

def get_img_as_dct_coef(file_path):
  jpeg = jio.read(file_path)
  image = np.dstack(jpeg.coef_arrays)
  image = np.absolute(image) / 128.
  return image

def process_path(file_path):
  file_path = file_path.decode("utf-8")
  label = get_label(file_path)
  image = get_img_as_dct_coef(file_path)
  return image, label

def set_shapes(image, label):
  image.set_shape((512, 512, 3))
  label.set_shape((1))
  return image, label

if __name__ == "__main__":
  train_dataset_list = tf.data.Dataset.from_tensor_slices(input_list_train)
  train_dataset = train_dataset_list.map(lambda x: tf.numpy_function(process_path, [x], [tf.float64, tf.float64]), num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.map(lambda i, l: set_shapes(i, l))

  test_dataset_list = tf.data.Dataset.from_tensor_slices(input_list_test)
  test_dataset = test_dataset_list.map(lambda x: tf.numpy_function(process_path, [x], [tf.float64, tf.float64]), num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.map(lambda i, l: set_shapes(i, l))

  print("\nTRAIN DATASET:", train_dataset.element_spec)
  print("\nTEST DATASET:", test_dataset.element_spec)

  model = keras.Sequential([
    # type 1
    keras.layers.Conv2D(64, 7, padding="same", kernel_initializer="he_normal", strides=2, input_shape=(img_height, img_width, 3)),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.BatchNormalization(),

    # type 2
    keras.layers.Conv2D(16, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(3, strides=2, padding="same"),

    keras.layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(3, strides=2, padding="same"),

    keras.layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(3, strides=2, padding="same"),

    keras.layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(3, strides=2, padding="same"),

    keras.layers.Conv2D(256, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(3, strides=2, padding="same"),

    # type 3
    keras.layers.Conv2D(512, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(2048),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1024),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1, activation="sigmoid")
  ])

  model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
  )

  print(model.summary())
  
  # train the model
  train_dataset = train_dataset.repeat(epochs).batch(batch_size)
  model.fit(train_dataset, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch)
  
  # evaluate the model
  test_dataset = test_dataset.batch(batch_size)
  model.evaluate(test_dataset, batch_size=batch_size)