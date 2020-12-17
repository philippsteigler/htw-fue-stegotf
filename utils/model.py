from tensorflow import keras
import tensorflow.keras.applications.efficientnet as efn

# Function for decaying the learning rate
def decay(epoch):
  if epoch < 4:
    return 1e-3
  elif epoch >= 4 and epoch < 12:
    return 1e-4
  else:
    return 1e-5

# Callback for printing the LR at the end of each epoch
class PrintLR(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print("Learning rate for epoch {} is {}".format(epoch + 1, model.optimizer.lr.numpy()))

def get_model(img_width, img_height, num_classes):
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
      keras.metrics.TruePositives(name="TP"),
      keras.metrics.FalsePositives(name="FP"),
      keras.metrics.TrueNegatives(name="TN"),
      keras.metrics.FalseNegatives(name="FN")
    ]
  )

  return model