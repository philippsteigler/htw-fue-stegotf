import tensorflow as tf
from utils import data
from utils import model

home_path = "/home/phst757c/ALASKA2"
train_path = "/Users/philipp/ALASKA2/train"

img_width = 512
img_height = 512
batch_size = 32
epochs = 20

if __name__ == "__main__":
  # Get classes and paths to images
  class_names, train_files, valid_files = data.load_files(train_path)

  # Get Sequences as datasets
  train_ds = data.AlaskaSequence(train_files, class_names, batch_size)
  valid_ds = data.AlaskaSequence(valid_files, class_names, batch_size)

  # Load model
  model = model.get_model(img_width, img_height)
  print(model.summary())

  # Create a callback that saves the model's weights
  checkpoint_path = home_path + "/saves/session-01/cp-{epoch:04d}.ckpt"
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_freq=len(train_files) // batch_size,
    verbose=1
  )

  # Start training
  model.fit(
    train_ds,
    steps_per_epoch=len(train_files) // batch_size,
    validation_data=valid_ds,
    validation_steps=len(valid_files) // batch_size,
    epochs=epochs,
    callbacks=[cp_callback]
  )
  