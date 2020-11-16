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
steps_per_epoch = image_count // batch_size

if __name__ == "__main__":
  # load images from directory
  train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)

  train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    class_mode="binary",
    batch_size=batch_size)

  # Load EfficientNet
  conv_base = efn.EfficientNetB3(
    weights="imagenet",
    include_top=False,
    classes=2,
    input_shape=(img_height, img_width, 3))

  conv_base.trainable = False

  model = keras.Sequential()
  model.add(conv_base)

  # Add custom top layers for classification
  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(1, activation="sigmoid"))

  # Compile final model
  model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"])
  
  print(model.summary())

  # train model
  model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)
