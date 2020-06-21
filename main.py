import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import jpegio as jio
import os
import random
import math

AUTOTUNE = tf.data.experimental.AUTOTUNE

input_size = 10000 # for each category
tt_ratio = 0.9

base_path = "/Users/philipp/ALASKA2"
train_path = base_path + "/train/"
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

total_img_count = len(input_list)
train_img_count = (int)(total_img_count * tt_ratio)

train_filenames = input_list[:train_img_count]
test_filenames = input_list[train_img_count:]

print("\nTOTAL IMAGES:", total_img_count)
print("\nTRAINING DATA:", len(train_filenames))
print("\nTEST DATA:", len(test_filenames))

img_width = 512
img_height = 512
batch_size = 32
epochs = 20
steps_per_epoch = train_img_count // batch_size

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  if parts[-2] == cover_label: 
    label = np.array([0.])
  else: 
    label = np.array([1.])
  return label

def get_img_as_ycbcr(file_path):
  jpeg = Image.open(file_path)
  jpeg.draft("YCbCr", None)
  jpeg.load()
  jpeg = np.array(jpeg) / 255.

  y = jpeg[:, :, 0]
  y = y[:, :, np.newaxis]
  return y

def get_img_as_ycbcr_raw(file_path):
  jpegStruct = jio.read(file_path)

  [col, row] = np.meshgrid(range(8), range(8))
  T = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))
  T[0, :] = T[0, :] / np.sqrt(2)

  img_dims = np.array(jpegStruct.coef_arrays[0].shape)
  n_blocks = img_dims // 8
  broadcast_dims = (n_blocks[0], 8, n_blocks[1], 8)
  
  dct_coeffs = jpegStruct.coef_arrays[0]
  QM = jpegStruct.quant_tables[0]
  
  t = np.broadcast_to(T.reshape(1, 8, 1, 8), broadcast_dims)
  qm = np.broadcast_to(QM.reshape(1, 8, 1, 8), broadcast_dims)
  dct_coeffs = dct_coeffs.reshape(broadcast_dims)
  
  a = np.transpose(t, axes=(0, 2, 3, 1))
  b = (qm * dct_coeffs).transpose(0, 2, 1, 3)
  c = t.transpose(0, 2, 1, 3)
          
  z = a @ b @ c
  z = z.transpose(0, 2, 1, 3)
  z = z.reshape(img_dims)
  
  y = z[:, :, np.newaxis] 
  y = y / 128.
  return y

def process_path(file_path):
  file_path = file_path.decode("utf-8")
  label = get_label(file_path)
  image = get_img_as_ycbcr_raw(file_path)
  return image, label

def set_shapes(image, label):
  image.set_shape((512, 512, 1))
  label.set_shape((1))
  return image, label

def load_dataset(filenames):
  dataset_list = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset_list.map(lambda x: tf.numpy_function(process_path, [x], [tf.float64, tf.float64]), num_parallel_calls=AUTOTUNE)
  dataset = dataset.map(lambda i, l: set_shapes(i, l))
  return dataset

def dctr_filter(shape, dtype=None):
  B = np.zeros(shape=(8, 8, 8, 8)) # k, l; m, n
  for k in range(0, 8):
    for l in range(0, 8):
      if k == 0: wk = 1 / math.sqrt(2)
      else: wk = 1
      if l == 0: wl = 1 / math.sqrt(2)
      else: wl = 1

      for m in range(0, 8):
        for n in range(0, 8):
          B[k][l][m][n] = (wk * wl / 4) * math.cos((math.pi * k * (2*m + 1)) / 16) * math.cos((math.pi * l * (2*n + 1)) / 16)

  B = B.reshape(64, 8, 8)
  B = np.expand_dims(B, -1)
  B = B.transpose(1, 2, 3, 0)
  assert B.shape == shape
  return tf.keras.backend.variable(B, dtype="float32")

def generate_model():
  model = keras.Sequential([
    # type 1
    keras.layers.Conv2D(64, 8, padding="same", kernel_initializer=dctr_filter, strides=2, use_bias=False, trainable=False, input_shape=(img_height, img_width, 1)),
    keras.layers.Lambda(lambda x: tf.abs(x)),

    # type 2
    keras.layers.Conv2D(16, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.MaxPool2D(3, strides=2, padding="same"),

    keras.layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.MaxPool2D(3, strides=2, padding="same"),

    keras.layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.MaxPool2D(3, strides=2, padding="same"),

    keras.layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.MaxPool2D(3, strides=2, padding="same"),

    keras.layers.Conv2D(256, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.MaxPool2D(3, strides=2, padding="same"),

    # type 3
    keras.layers.Conv2D(512, 3, padding="same", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(512),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(256),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1, activation="sigmoid")
  ])

  model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"])

  return model

def create_cp_callback():
  checkpoint_path = base_path + "/TF/checkpoints/train_final/cp-{epoch:04d}.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      save_weights_only=True,
      verbose=1)

  return cp_callback

def get_latest_chechkpoint():
  return tf.train.latest_checkpoint(base_path + "/TF/checkpoints/train_final")

if __name__ == "__main__":
  train_dataset = load_dataset(train_filenames)
  test_dataset = load_dataset(test_filenames)

  print("\nTRAIN DATASET:", train_dataset.element_spec)
  print("\nTEST DATASET:", test_dataset.element_spec)

  # generate the model
  model = generate_model()
  #model.load_weights(get_latest_chechkpoint())
  print(model.summary())
  
  # train the model
  train_dataset = train_dataset.repeat(epochs).batch(batch_size)
  cp_callback = create_cp_callback()
  model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[cp_callback])

  # save trained model with weights
  model.save(base_path + "/TF/model")
  
  # evaluate the model
  test_dataset = test_dataset.batch(batch_size)
  model.evaluate(test_dataset)
  