import os
import pathlib
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.applications.efficientnet as efn
AUTOTUNE = tf.data.experimental.AUTOTUNE

home_path = "/home/phst757c/ALASKA3"
train_path = "/projects/p_ml_steg_steigler/ALASKA2/train"

img_width = 512
img_height = 512
batch_size = 32
epochs = 20

def get_generators():
  image_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=False,
    vertical_flip=False,
    validation_split=0.1
  )

  train_generator = image_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    class_mode="categorical",
    batch_size=batch_size,
    subset="training"
  )

  valid_generator = image_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    class_mode="categorical",
    batch_size=batch_size,
    subset="validation"
  )

  return train_generator, valid_generator

def get_model(num_classes):
  model = keras.Sequential()

  # Load EfficientNet as base
  conv_base = efn.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3)
  )
  model.add(conv_base)

  # Add custom top layers for classification
  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.25))
  model.add(keras.layers.Dense(num_classes, activation="softmax"))

  # Finally compile the model
  model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[
      keras.metrics.CategoricalAccuracy(name="CatAcc"),
      keras.metrics.AUC(name="AUC"),
      keras.metrics.Precision(name="Pre C0", class_id=0),
      keras.metrics.Precision(name="Pre C1", class_id=1),
      keras.metrics.Precision(name="Pre C2", class_id=2),
      keras.metrics.Precision(name="Pre C3", class_id=3),
      keras.metrics.Recall(name="Rec C0", class_id=0),
      keras.metrics.Recall(name="Rec C1", class_id=1),
      keras.metrics.Recall(name="Rec C2", class_id=2),
      keras.metrics.Recall(name="Rec C3", class_id=3),
    ]
  )

  return model

if __name__ == "__main__":
  # Define distribution stategy
  strategy = tf.distribute.MirroredStrategy()
  print("Number of GPUs: ", strategy.num_replicas_in_sync)
  
  # Prepare dataset
  train_gen, valid_gen = get_generators()
  print("Classes: ", train_gen.class_indices)

  with strategy.scope():
    # Load model
    model = get_model(len(train_gen.class_indices))
    print(model.summary())

    # Create a callback that saves the model's weights
    checkpoint_path = home_path + "/saves/session-01/cp-{epoch:04d}.ckpt"
    cp_callback = keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      save_weights_only=True,
      save_freq=train_gen.samples // batch_size,
      verbose=1
    )

    """
    # Load weights from previous session
    checkpoint_dir = home_path + "/saves/session-01/"
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    """

    # Start training
    model.fit(
      train_gen,
      steps_per_epoch=train_gen.samples // batch_size,
      validation_data=valid_gen,
      validation_steps=valid_gen.samples // batch_size,
      epochs=epochs,
      callbacks=[cp_callback],
      max_queue_size=64,
      use_multiprocessing=True,
      workers=16
    )