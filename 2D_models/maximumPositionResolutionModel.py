from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.models import Model, Input
from keras.layers import Concatenate, concatenate
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
import math
import seaborn as sns

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

df1 = pd.read_csv('recon.csv')
df2 = pd.read_csv('labels.csv')
df3 = pd.read_csv('angles.csv')

X = df1.values
y = df2.values
angleValues = df3.values

n = 650000

X = np.reshape(X, (n,13,21,1))

path = os.getcwd()
print(path)

#only want to predict x-entry and y-entry
df2.drop('z-entry', axis=1, inplace=True)
df2.drop('n_x', axis=1, inplace=True)
df2.drop('n_y', axis=1, inplace=True)
df2.drop('n_z', axis=1, inplace=True)
df2.drop('cotAlpha',axis = 1, inplace=True)
df2.drop('cotBeta',axis = 1, inplace=True)
df2.drop('number_eh_pairs', axis=1, inplace=True)

#reset y since you dropped columns
y = df2.values

#https://keras.io/api/callbacks/#csvlogger
#from https://keras.io/guides/writing_your_own_callbacks/
class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

   # def on_predict_batch_begin(self, batch, logs=None):
    #    keys = list(logs.keys())
     #   print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

#    def on_predict_batch_end(self, batch, logs=None):
 #       keys = list(logs.keys())
  #      print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


#set up samples
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)
X_train, X_test, X_angle_train, X_angle_test, y_train, y_test = train_test_split(X,angleValues,y, test_size = 0.20, random_state = 0)

print(X.shape, y.shape)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_angle_train.shape, X_angle_test.shape)

#scale input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
X_angle_train = scaler.fit_transform(X_angle_train.reshape(-1, X_angle_train.shape[-1])).reshape(X_angle_train.shape)
X_angle_test = scaler.fit_transform(X_angle_test.reshape(-1, X_angle_test.shape[-1])).reshape(X_angle_test.shape)

#make model
input1 = Input(shape=(13,21,1))
conv1 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(input1)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(pool1)
flattenLayer = Flatten()(conv2)

input2 = Input(shape=(2,))
merge = concatenate([flattenLayer, input2])

dense1 = Dense(32, activation='relu')(merge)
dropout1 = Dropout(0.1)(dense1)
dense2 = Dense(2, activation='linear')(dropout1)

model = Model(inputs =[input1, input2], outputs=dense2)
model.summary()

checkpoint_path = "cp.ckpt"

# Create a callback that saves the model's weights
# currently, model weights are saved for each training
# to do - update for early stopping
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
#patience: Number of epochs with no improvement after which training will be stopped
earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

csv_logger = CSVLogger('log.csv', append=True, separator=';')

batch_size = 64
epochs = 200

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer='adam',
              metrics=['mean_squared_error'])

history = model.fit(
            x=(X_train,X_angle_train),
            y = y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=([X_test,X_angle_test], y_test),
          callbacks=[cp_callback, csv_logger, earlyStop_callback],
          )

res = model.evaluate(
    x=(X_test, X_angle_test), y = y_test, batch_size=batch_size,
)

predictions = model.predict((X_test, X_angle_test), batch_size=batch_size, callbacks=[CustomCallback()])

#save model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
  json_file.write(model_json)

#save full model in hd5 format
model.save('my_model.h5')

df_predict = pd.DataFrame(predictions, columns=['x_entry', 'y_entry'])

trueLabels = pd.DataFrame(y_test, columns=['x_entry', 'y_entry'])

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('loss.png')

plt.plot(history.history['mean_squared_error'], label='MSE')
plt.legend(loc='upper right')
plt.savefig('mse.png')

df_predict.to_csv('predictions.csv')
trueLabels.to_csv('trueLabels.csv')

#for plotting a picture of the model
#from tensorflow.keras.utils import plot_model
#plot_model(model, to_file='model.png')
