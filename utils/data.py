import tensorflow as tf
from PIL import Image
import numpy as np
import random
import os

# Image Squenece to fit data into model during training
class AlaskaSequence(tf.keras.utils.Sequence):
  def __init__(self, x_set, class_names, batch_size):
    self.x = x_set
    self.class_names = class_names
    self.batch_size = batch_size

  def __len__(self):
    return len(self.x) // self.batch_size

  def get_img(self, fn):
    image = Image.open(fn)
    image.draft("YCbCr", None)
    image.load()
    image = np.array(image) / 255.
    return image

  def get_label(self, fn):
    cn = os.path.basename(os.path.dirname(fn))
    idx = self.class_names.index(cn)
    label = np.zeros(len(self.class_names), dtype=np.int8)
    label[idx] = 1
    return label

  def __getitem__(self, idx):
    batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    x = np.array([self.get_img(fn) for fn in batch_x])
    y = np.array([self.get_label(fn) for fn in batch_x])
    return x, y

# Helper function to load classes and images
def load_files(train_path):
  class_names = []
  train_files = []
  valid_files = []
  for subdir in os.listdir(train_path):
    if os.path.isdir(os.path.join(train_path, subdir)):
      class_names.append(subdir)
      current_dir = os.listdir(os.path.join(train_path, subdir))
      image_count = len(current_dir)
      print("Loading class '" + subdir + "' with " + str(image_count) + " images...")
      for image in current_dir[:int(image_count*0.9)]:
        train_files.append(os.path.abspath(os.path.join(train_path, subdir, image)))
      for image in current_dir[int(image_count*0.9):]:
        valid_files.append(os.path.abspath(os.path.join(train_path, subdir, image)))
  
  random.shuffle(train_files)
  random.shuffle(valid_files)

  return class_names, train_files, valid_files