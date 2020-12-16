import os
import pathlib
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.applications.efficientnet as efn
AUTOTUNE = tf.data.experimental.AUTOTUNE

home_path = pathlib.Path("/home/phst757c/ALASKA3")
train_path = pathlib.Path("/projects/p_ml_steg_steigler/ALASKA2/train")

image_count = len(list(train_path.glob("*/*.jpg")))
class_names = np.array(sorted([item.name for item in train_path.glob("*") if item.name != ".DS_Store"]))

img_width = 512
img_height = 512
batch_size = 32
epochs = 20

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  one_hot = parts[-2] == class_names
  return one_hot

def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3)
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def get_model():
  model = keras.Sequential()

  # Load EfficientNet as base
  conv_base = efn.EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3)
  )
  model.add(conv_base)

  # Add custom top layers for classification
  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.25))
  model.add(keras.layers.Dense(len(class_names), activation="softmax"))

  # Finally compile the model
  model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[
      keras.metrics.CategoricalAccuracy(name="CatAcc"),
      keras.metrics.AUC(name="AUC"),
      keras.metrics.TruePositives(name="TP"),
      keras.metrics.FalsePositives(name="FP"),
      keras.metrics.TrueNegatives(name="TN"),
      keras.metrics.FalseNegatives(name="FN")
    ]
  )

  return model

if __name__ == "__main__":
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    print("Classes: ", class_names)
    print("Total images: ", image_count)

    list_ds = tf.data.Dataset.list_files(str(train_path/"*/*"), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    val_size = int(image_count * 0.1)
    train_ds = list_ds.skip(val_size)
    valid_ds = list_ds.take(val_size)

    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    valid_ds = valid_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    valid_ds = configure_for_performance(valid_ds)

    # Load model
    model = get_model()
    print(model.summary())

    # Create a callback that saves the model's weights
    checkpoint_path = home_path/"saves/session-01/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      save_weights_only=True,
      save_freq=(image_count-val_size) // batch_size,
      verbose=1
    )

    """
    # Load weights from previous session
    checkpoint_dir = home_path/"saves/session-01/"
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    """

    # Start training
    model.fit(
      train_ds,
      validation_data=valid_ds,
      epochs=epochs,
      callbacks=[cp_callback],
      max_queue_size=40,
    )
