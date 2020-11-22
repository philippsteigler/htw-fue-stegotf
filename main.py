import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import efficientnet.keras as efn

train_path = "/Users/philipp/ALASKA2/train2/"
cover_label = "Cover_75"
stego_label = "UERD_75"
cover_path = train_path + cover_label
stego_path = train_path + stego_label

img_width = 512
img_height = 512
batch_size = 32
epochs = 10
image_count = len(os.listdir(cover_path) + os.listdir(stego_path))

if __name__ == "__main__":
  # create train and validation dataset
  image_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

  train_generator = image_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    class_mode="binary",
    batch_size=batch_size,
    subset="training")

  valid_generator = image_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    class_mode="binary",
    batch_size=batch_size,
    subset="validation")

  # load EfficientNet as base
  conv_base = efn.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    classes=2,
    input_shape=(img_height, img_width, 3))

  #conv_base.trainable = False

  model = keras.Sequential()
  model.add(conv_base)

  # add custom top layers for classification
  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(1, activation="sigmoid"))

  # compile final model
  model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"])
  
  print(model.summary())

  # train model
  model.fit_generator(
    generator = train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = valid_generator,
    validation_steps = valid_generator.samples // batch_size,
    epochs = epochs)
