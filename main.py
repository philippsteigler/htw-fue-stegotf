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
batch_size = 32
epochs = 10
steps_per_epoch = np.ceil(img_count / batch_size)

kernel_hp = np.array(
    [[[ -1], [  2], [ -2], [  2], [ -1]],
     [[  2], [ -6], [  8], [ -6], [  2]],
     [[ -2], [  8], [-12], [  8], [ -2]],
     [[  2], [ -6], [  8], [ -6], [  2]],
     [[ -1], [  2], [ -2], [  2], [ -1]]]) * (1.0/12.0)

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  if parts[-2] == "Cover": return 0
  else: return 1

def get_image(file_path):
  image = keras.preprocessing.image.load_img(file_path, color_mode="grayscale")
  image = keras.preprocessing.image.img_to_array(image) / 255.0
  image = ndimage.convolve(image, kernel_hp)
  image = np.delete(image, [0, 1, img_width - 2, img_width - 1], 0)
  image = np.delete(image, [0, 1, img_height - 2, img_height - 1], 1)
  return image

def process_path(file_path):
  label = get_label(file_path)
  image = get_image(file_path)
  return image, label

def set_shapes(img, label):
  img.set_shape((img_width - 4, img_height - 4, 1))
  label.set_shape(())
  return img, label

if __name__ == "__main__":
  ds_train_list = tf.data.Dataset.list_files(os.path.join(bb_path, "train/*/*.jpg"))
  train_dataset = ds_train_list.map(lambda x: tf.numpy_function(process_path, [x], [tf.float32, tf.int64]), num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.map(lambda i, l: set_shapes(i, l))

  print("\nDATASET:", train_dataset.element_spec)
  
  '''
  for image, label in train_dataset.take(10):
    print(image.numpy())
    print("Label: ", label.numpy())
    image = image.numpy().reshape((image.shape[0], -1))
    plt.imshow(image, "gray")
    plt.show()
  '''

  model = keras.Sequential([
    keras.layers.Conv2D(64, (7, 7), padding="same", activation="relu", input_shape=(img_width - 4, img_height - 4, 1)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(16, (5, 5), padding="same", activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1)
  ])

  model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
  )

  print("\nMODEL:", model.summary())
  train_dataset = train_dataset.repeat(epochs).batch(batch_size)
  model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)
