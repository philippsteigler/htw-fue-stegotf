import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_path = "/Users/philipp/ALASKA2/train"
img_count = len(os.listdir(train_path + "/cover") + os.listdir(train_path + "/stego"))
img_width = 512
img_height = 512
batch_size = 128
epochs = 15
steps_per_epoch = img_count // batch_size

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  if parts[-2] == "cover": 
    label = np.array([0.])
  else: 
    label = np.array([1.])
  return label

def get_img(file_path): 
  image = Image.open(file_path).convert("YCbCr")
  image = np.array(image) / 255.
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
  ds_train_list = tf.data.Dataset.list_files(os.path.join(train_path, "*/*.jpg"))
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
    keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=(img_height, img_width, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(5, strides=2, padding="same"),

    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(5, strides=2, padding="same"),

    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(5, strides=2, padding="same"),

    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(5, strides=2, padding="same"),

    keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(5, strides=2, padding="same"),

    keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(256, activation="relu"),
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
  