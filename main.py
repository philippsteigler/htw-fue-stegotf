import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

input_size = 25000 # for each category
train_path = "/Users/philipp/ALASKA2/train/"
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
img_width = 512
img_height = 512
batch_size = 32
epochs = 50
steps_per_epoch = img_count // batch_size

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  if parts[-2] == cover_label: 
    label = np.array([0.])
  else: 
    label = np.array([1.])
  return label

def load_img_as_ycbcr(file_path):
  image = Image.open(file_path)
  image.draft("YCbCr", None)
  image.load()
  image = np.array(image) / 255.
  return image

def load_img_as_dct_coef(file_path):
  jpeg = jio.read(file_path)
  image = np.dstack(jpeg.coef_arrays)
  image = np.absolute(image) / 128.
  return image

def get_img(file_path):
  image = load_img_as_dct_coef(file_path)
  return image

def process_path(file_path):
  file_path = file_path.decode("utf-8")
  label = get_label(file_path)
  image = get_img(file_path)
  return image, label

def set_shapes(image, label):
  image.set_shape((512, 512, 3))
  label.set_shape((1))
  return image, label

if __name__ == "__main__":
  ds_train_list = tf.data.Dataset.from_tensor_slices(input_list)
  train_dataset = ds_train_list.map(lambda x: tf.numpy_function(process_path, [x], [tf.float64, tf.float64]), num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.map(lambda i, l: set_shapes(i, l))

  print("\nDATASET:", train_dataset.element_spec)

  """
  for image, label in train_dataset.take(1):
    plt.figure(figsize=(10,10))
    plt.title(label.numpy())
    plt.imshow(image)
    plt.show()
  """

  model = keras.Sequential([
    keras.layers.Conv2D(64, 7, padding="same", kernel_initializer="he_normal", strides=2, input_shape=(img_height, img_width, 3)),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(16, 5, padding="same", kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(16, 5, padding="same", kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.BatchNormalization(),

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

  print("\nMODEL:", model.summary())

  model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
  )
  
  train_dataset = train_dataset.repeat(epochs).batch(batch_size)
  model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)
  