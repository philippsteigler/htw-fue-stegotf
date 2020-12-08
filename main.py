# Import Tensorflow
import tensorflow as tf
from tensorflow import keras
AUTOTUNE = tf.data.experimental.AUTOTUNE
import tensorflow.keras.applications.efficientnet as efn

home_path = "/projects/p_ml_steg_steigler"
train_path = home_path + "/train"

img_width = 512
img_height = 512
batch_size = 16
epochs = 50

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

def get_model():
  model = keras.Sequential()

  # Load EfficientNet as base
  conv_base = efn.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    classes=4,
    input_shape=(img_height, img_width, 3)
  )
  model.add(conv_base)

  # Add custom top layers for classification
  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.25))
  model.add(keras.layers.Dense(4, activation="softmax"))

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
  # Get image dataset generators
  train_gen, valid_gen = get_generators()
  print("Classes: ", train_gen.class_indices)

  # Load model
  model = get_model()
  print(model.summary())

  # Create a callback that saves the model's weights
  checkpoint_path = home_path + "/saves/session-01/cp-{epoch:04d}.ckpt"
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_freq=train_gen.samples // batch_size,
    verbose=1
  )

  # Load weights from previous session
  checkpoint_dir = home_path + "/saves/session-01/"
  latest = tf.train.latest_checkpoint(checkpoint_dir)
  model.load_weights(latest)

  # Start training
  model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // batch_size,
    validation_data=valid_gen,
    validation_steps=valid_gen.samples // batch_size,
    epochs=epochs,
    callbacks=[cp_callback]
  )
