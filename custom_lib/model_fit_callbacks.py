
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback



def callback_earlystop_val_loss(resolution = 0.01, patience = 3):
    cb = EarlyStopping(monitor='val_loss', min_delta=resolution, patience=patience, restore_best_weights=True)
    return cb


class callback_display_prediction_each_epoch(Callback):
  def on_epoch_end(self, epoch, logs=None):
    print ('Sample Prediction after epoch {}'.format(epoch+1))

## Call back to visualize improvement after each epoch
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print ('Sample Prediction after epoch {}'.format(epoch+1))