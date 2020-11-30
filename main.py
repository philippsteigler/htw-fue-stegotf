import os
import numpy as np
from PIL import Image

#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#import keras
#import efficientnet.keras as efn
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.applications.efficientnet as efn

train_path = "/home/aw4/ALASKA"
images_per_class = 1000

classes = []
filenames = []
for subdir in os.listdir(train_path):
  if os.path.isdir(os.path.join(train_path, subdir)):
    classes.append(subdir)
    for image in os.listdir(os.path.join(train_path, subdir))[:images_per_class]:
      filenames.append(os.path.abspath(os.path.join(train_path, subdir, image)))

num_classes = len(classes)
print("Found %d images belonging to %d classes" % (len(filenames), num_classes))
print("Classes: ", classes)

img_width = 512
img_height = 512
batch_size = 16
steps_per_epoch = len(filenames) // batch_size
epochs = 10

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  classname = parts[-2].numpy().decode("utf-8")
  index = classes.index(classname)
  label = np.zeros(num_classes)
  label[index] = 1.
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
  label.set_shape((num_classes))
  return image, label

def get_model():
  # load EfficientNet as base
  conv_base = efn.EfficientNetB0(
    weights = "imagenet",
    include_top = False,
    classes = num_classes,
    input_shape = (img_height, img_width, 3)
  )

  model = keras.Sequential()
  model.add(conv_base)

  # add custom top layers for classification
  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(num_classes, activation="softmax"))

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

if __name__ == "__main__":
  # Create dataset containing all filenames
  filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  # Create labeled dataset from filename dataset
  train_dataset = filenames_dataset.map(lambda x: tf.numpy_function(process_path, [x], [tf.float64, tf.float64]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.map(lambda i, l: set_shapes(i, l))
  
  train_dataset = train_dataset.shuffle(buffer_size=500, reshuffle_each_iteration=True)
  train_dataset = train_dataset.batch(batch_size)
  train_dataset = train_dataset.repeat(epochs)
  train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  
  print("Dataset Shape:", train_dataset.element_spec)

  # Load model
  model = get_model()
  print(model.summary())

  # Start training
  model.fit(
    train_dataset,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs
  )