import tensorflow as tf
from tensorflow import keras
#from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
#import cv2

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_path = pathlib.Path("/Users/philipp/ALASKA2")
img_count = len(list(train_path.glob("train/*/*.jpg")))
img_width = 512
img_height = 512
batch_size = 128
epochs = 10
steps_per_epoch = img_count // batch_size

"""
kernel_hp = np.array(
    [[[ -1], [  2], [ -2], [  2], [ -1]],
     [[  2], [ -6], [  8], [ -6], [  2]],
     [[ -2], [  8], [-12], [  8], [ -2]],
     [[  2], [ -6], [  8], [ -6], [  2]],
     [[ -1], [  2], [ -2], [  2], [ -1]]]) * (1.0/12.0)

def apply_hpf(image):
  image = ndimage.convolve(image, kernel_hp)
  cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
  return image

def apply_hpf_dft(image):
  dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
  dft_shift = np.fft.fftshift(dft)
  magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

  crow, ccol = int(img_height / 2), int(img_width / 2)
  mask = np.ones((img_height, img_width, 2), np.uint8)
  r = 40
  center = [crow, ccol]
  x, y = np.ogrid[:img_height, :img_width]
  mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
  mask[mask_area] = 0

  fshift = dft_shift * mask
  fshift_mask_mag = np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

  f_ishift = np.fft.ifftshift(fshift)
  image_back = cv2.idft(f_ishift)
  image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])

  cv2.normalize(image_back, image_back, 0, 1, cv2.NORM_MINMAX)
  image_back = np.reshape(image_back, (img_height, img_width, -1))
  return image_back
"""

if __name__ == "__main__":
  train_image_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
    #preprocessing_function=apply_hpf_dft
  )

  train_data_gen = train_image_gen.flow_from_directory(
    batch_size=batch_size,
    directory="/Users/philipp/ALASKA2/train",
    shuffle=True,
    target_size=(img_height, img_width),
    #color_mode="grayscale",
    class_mode="binary"
  )

  print(train_data_gen)

  """
  image_batch, label_batch = next(train_data_gen)
  for i in range(0, 11):
    image = image_batch[i]
    print(image)
    print(np.amin(image), np.amax(image))
    plt.title(label_batch[i])
    plt.imshow(image)
    plt.show()
  """

  """
  model = keras.Sequential([
    keras.layers.Conv2D(64, 7, padding="same", activation="relu", input_shape=(img_height, img_width, 3)),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(16, 5, padding="same", activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
  ])
  """

  model = keras.Sequential([
    keras.layers.Conv2D(32, 7, strides=2, padding="same", activation="relu", input_shape=(img_height, img_width, 3)),
    keras.layers.Conv2D(16, 5, padding="same", activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
  ])

  model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
  )

  print(model.summary())

  model.fit(train_data_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, shuffle=False)
