import tensorflow as tf
from utils import data
from utils import model

home_path = "/home/phst757c/ALASKA3"
train_path = "/projects/p_ml_steg_steigler/ALASKA2/train"

img_width = 512
img_height = 512
batch_size = 32
epochs = 20

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

if __name__ == "__main__":
  # Define distribution stategy
  strategy = tf.distribute.MirroredStrategy()
  print("Number of GPUs: ", strategy.num_replicas_in_sync)

  # Get classes and paths to images
  class_names, train_files, valid_files = data.load_files(train_path)

  # Get Sequences as datasets
  train_ds = data.AlaskaSequence(train_files, class_names, batch_size)
  valid_ds = data.AlaskaSequence(valid_files, class_names, batch_size)

  with strategy.scope():
    # Load model
    model = model.get_model(img_width, img_height, len(class_names))
    print(model.summary())

    # Create a callback that saves the model's weights
    checkpoint_path = home_path + "/saves/session-01/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      save_weights_only=True,
      save_freq=len(train_files) // batch_size,
      verbose=1
    )

    """
    # Load weights from previous session
    checkpoint_dir = home_path + "/saves/session-01/"
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    """

    callbacks=[
      cp_callback,
      keras.callbacks.LearningRateScheduler(decay),
      PrintLR()
    ]

    # Start training
    model.fit(
      train_ds,
      steps_per_epoch=len(train_files) // batch_size,
      validation_data=valid_ds,
      validation_steps=len(valid_files) // batch_size,
      epochs=epochs,
      callbacks=callbacks,
      max_queue_size=256,
      use_multiprocessing=True,
      workers=8
    )
