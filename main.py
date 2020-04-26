import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
BB_HOME = pathlib.Path("/Users/philipp/BOSSbase")

class_names = np.array([item.name for item in BB_HOME.glob("train/*") if item.name != ".DS_Store"])
iamge_count = len(list(BB_HOME.glob("train/*/*.pgm")))

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return np.where(parts[-2] == class_names)

def get_image(file_path):
  image = tf.keras.preprocessing.image.load_img(file_path, color_mode="grayscale")
  image = np.array(image)
  return tf.convert_to_tensor(image)

def process_path(file_path):
  label = get_label(file_path)
  image = get_image(file_path)
  return image, label

if __name__ == "__main__":
  ds_train_list = tf.data.Dataset.list_files(os.path.join(BB_HOME, "train/*/*.pgm"))
  ds_test_list = tf.data.Dataset.list_files(os.path.join(BB_HOME, "test/*/*.pgm"))
  train_dataset = ds_train_list.map(lambda x: tf.numpy_function(process_path, [x], [tf.uint8, tf.int64]), num_parallel_calls=AUTOTUNE)
  test_dataset = ds_test_list.map(lambda x: tf.numpy_function(process_path, [x], [tf.uint8, tf.int64]), num_parallel_calls=AUTOTUNE)

  for image, label in train_dataset.take(4):
      print("Image shape: ", image.numpy().shape)
      print("Label: ", class_names[label.numpy()])
