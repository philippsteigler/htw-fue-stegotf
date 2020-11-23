import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import efficientnet.keras as efn

train_path = "/Users/philipp/ALASKA2/train/"
img_width = 512
img_height = 512
batch_size = 32
epochs = 10

def get_generators(num_classes: int):
  image_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
  )

  if num_classes == 4:
    cm = "categorical"
  else:
    cm = "binary"

  train_generator = image_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    class_mode=cm,
    batch_size=batch_size,
    subset="training"
  )

  valid_generator = image_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    class_mode=cm,
    batch_size=batch_size,
    subset="validation"
  )

  return train_generator, valid_generator

def get_model(num_classes: int):
  # load EfficientNet as base
  conv_base = efn.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    classes=num_classes,
    input_shape=(img_height, img_width, 3)
  )

  model = keras.Sequential()
  model.add(conv_base)

  # add custom top layers for classification
  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.5))

  # add last layer to classify cover vs. 3 stego classes OR cover vs. stego
  # also adjust loss and accuracy parameter for compilation
  if num_classes == 4:
    model.add(keras.layers.Dense(4, activation="softmax"))
    ls = "categorical_crossentropy"
    accuracy = keras.metrics.CategoricalAccuracy(name="Acc")
  else:
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    ls = "binary_crossentropy"
    accuracy = keras.metrics.BinaryAccuracy(name="Acc")

  # finally compile the model
  model.compile(
    optimizer="adam",
    loss=ls,
    metrics=[
      accuracy,
      keras.metrics.AUC(name="AUC"),
      keras.metrics.TruePositives(name="TP"),
      keras.metrics.FalsePositives(name="FP"),
      keras.metrics.TrueNegatives(name="TN"),
      keras.metrics.FalseNegatives(name="FN")
    ]
  )

  return model

if __name__ == "__main__":
  # define class mode: 4 = all classes, 2 = binary
  num_classes = 2

  train_generator, valid_generator = get_generators(num_classes)
  model = get_model(num_classes)
  print(model.summary())

  model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = valid_generator,
    validation_steps = valid_generator.samples // batch_size,
    epochs = epochs
  )
