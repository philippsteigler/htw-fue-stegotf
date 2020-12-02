import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#import keras
#import efficientnet.keras as efn
import tensorflow as tf
from tensorflow import keras
AUTOTUNE = tf.data.experimental.AUTOTUNE
import tensorflow.keras.applications.efficientnet as efn

train_path = "/home/aw4/ALASKA"
images_per_class = 1000
classes = []
train_filenames = []
valid_filenames = []

img_width = 512
img_height = 512
batch_size = 16
epochs = 10

def load_filenames():
  for subdir in os.listdir(train_path):
    if os.path.isdir(os.path.join(train_path, subdir)):
      classes.append(subdir)
      for image in os.listdir(os.path.join(train_path, subdir))[:int(images_per_class*0.8)]:
        train_filenames.append(os.path.abspath(os.path.join(train_path, subdir, image)))
      for image in os.listdir(os.path.join(train_path, subdir))[int(images_per_class*0.8):images_per_class]:
        valid_filenames.append(os.path.abspath(os.path.join(train_path, subdir, image)))

  # Check if valid images are not in training list
  for image in train_filenames:
    if image in valid_filenames:
      print("WARNING: Found duplicate in training and validation list: ", image)

  random.shuffle(train_filenames)
  random.shuffle(valid_filenames)

  print("Classes: ", classes)
  print("Training: Found %d images belonging to %d classes" % (len(train_filenames), len(classes)))
  print("--Files (example):", train_filenames[:4])
  print("Validation: Found %d images belonging to %d classes" % (len(valid_filenames), len(classes)))
  print("--Files (example):", valid_filenames[:4])

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  classname = parts[-2].numpy().decode("utf-8")
  index = classes.index(classname)
  label = np.zeros(len(classes), dtype=np.int8)
  label[index] = 1
  return label

def get_img(file_path): 
  image = Image.open(file_path)
  image.draft("YCbCr", None)
  image.load()
  image = np.array(image) / 255.
  return image

def process_path(file_path):
  file_path = file_path.decode("utf-8")
  label = get_label(file_path)
  image = get_img(file_path)
  return image, label

def set_shapes(image, label):
  image.set_shape((img_height, img_width, 3))
  label.set_shape((len(classes)))
  return image, label

def get_model():
  # load EfficientNet as base
  conv_base = efn.EfficientNetB0(
    weights = "imagenet",
    include_top = False,
    classes = len(classes),
    input_shape = (img_height, img_width, 3)
  )

  model = keras.Sequential()
  model.add(conv_base)

  # add custom top layers for classification
  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(len(classes), activation="softmax"))

  # finally compile the model
  model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[
      keras.metrics.CategoricalAccuracy(name="CatCross"),
      keras.metrics.AUC(name="AUC"),
      keras.metrics.TruePositives(name="TP"),
      keras.metrics.FalsePositives(name="FP"),
      keras.metrics.TrueNegatives(name="TN"),
      keras.metrics.FalseNegatives(name="FN")
    ]
  )

  return model

def plot_metrics(history):
  metrics = ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()

if __name__ == "__main__":
  # First get a list of all filenames
  load_filenames()

  # Create training dataset containing all filenames
  train_filenames_dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
  # Create labeled training dataset from filename dataset
  train_dataset = train_filenames_dataset.map(
    lambda x: tf.numpy_function(process_path, [x], [tf.float64, tf.int8]),
    num_parallel_calls=AUTOTUNE
  )
  
  train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
  train_dataset = train_dataset.batch(batch_size)
  train_dataset = train_dataset.repeat(epochs)
  train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

  # Create validation dataset containing all filenames
  valid_filenames_dataset = tf.data.Dataset.from_tensor_slices(valid_filenames)
  # Create labeled validation dataset from filename dataset
  valid_dataset = valid_filenames_dataset.map(
    lambda x: tf.numpy_function(process_path, [x], [tf.float64, tf.int8]),
    num_parallel_calls=AUTOTUNE
  )

  valid_dataset = valid_dataset.batch(batch_size)
  valid_dataset = valid_dataset.repeat(epochs)
  valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

  # Load model
  model = get_model()
  print(model.summary())

  # Start training
  history = model.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=len(train_filenames) // batch_size,
    validation_data=valid_dataset,
    validation_steps=len(valid_filenames) // batch_size
  )
