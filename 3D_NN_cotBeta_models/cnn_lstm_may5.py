from google.colab import drive
drive.mount('/content/drive')

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM, Conv3D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import CSVLogger
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
import math
import seaborn as sns
from collections import deque
import copy

BATCH_SIZE = 2 
EPOCHS = 1
NUMBER_TEST_SET = 10 #will do train/test split before this, and pass already-made sets in
NUMBER_TRAIN_SET = 20
TEMPORAL_LENGTH = 8 #use first 8 frames (these are 16 frames in each video)

def shuffle_data(samples):
    data = shuffle(samples,random_state=2)
    return data

def data_generator(data,batch_size=BATCH_SIZE,temporal_padding='same',shuffle=True):               
    num_samples = len(data)
    if shuffle:
        data = shuffle_data(data)
    while True:   
        for offset in range(0, num_samples, batch_size):
            print ('starting index: ', offset) 
            batch_samples = data[offset:offset+batch_size]
            
            X_train = []
            y_train = []
            
            for batch_sample in batch_samples: 
                print(batch_sample)
                # Load image (X)
                x = batch_sample[0] #image
                y = batch_sample[1] #label
                temp_data_list = []
                for img in x:
                    try:
                        img = np.load(img)                        
                        temp_data_list.append(img)
                    except Exception as e:
                        print (e)
                        print ('error reading in frame: ',img)                      

                X_train.append(temp_data_list)
                y_train.append(y)

            X_train = np.array(X_train)   
            y_train = np.array(y_train)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        
            print(X_train.shape)               
            yield X_train, y_train

trainData = np.load('/content/drive/MyDrive/testGCP-DEBUG/TrainingData.npy', allow_pickle=True)
testData = np.load('/content/drive/MyDrive/testGCP-DEBUG/TestData.npy', allow_pickle=True)

trainData[0]

train_generator = data_generator(trainData,batch_size=BATCH_SIZE,shuffle=True)
test_generator = data_generator(testData,batch_size=BATCH_SIZE,shuffle=False) 
#x,y = next(train_generator)
#xx,yy = next(train_generator)

def get_model():
    model = Sequential()
    model.add(
    TimeDistributed(
        Conv2D(16, (3,3), activation='relu'), 
        input_shape=(8, 13, 21, 1))
    )
    model.add(
    TimeDistributed(
        Conv2D(32, (3,3), activation='relu')
        )
    )
    model.add(
    TimeDistributed(
        MaxPooling2D()
        )
    )         
    model.add(
    TimeDistributed(
        Flatten()
        )
    )        
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(.1))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='Huber', optimizer=Adam(), metrics=['mean_squared_error'])
    model.summary()
    return model

model = get_model()

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

checkpoint_path = "/content/drive/MyDrive/testGCP/cp.ckpt"
earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)
csv_logger = CSVLogger('/content/drive/MyDrive/testGCP/log.csv', append=True, separator=';')



hist = model.fit(train_generator,
                 steps_per_epoch=(NUMBER_TRAIN_SET/BATCH_SIZE),
                 epochs=EPOCHS, validation_data=(test_generator), validation_steps=(NUMBER_TEST_SET)/BATCH_SIZE,
                 callbacks=[cp_callback, csv_logger, earlyStop_callback])

truthB = []
predB = []

#need to fix "limit" depending on specific batch size otherwise generator will loop without stopping
limit = NUMBER_TEST_SET/BATCH_SIZE

batches = 0
for i in test_generator:
  predB.append(model.predict(i[0]))
  truthB.append(i[1]) 
  batches += 1
  if batches > limit:
    break

predBATCHED = np.concatenate(predB)
truthBATCHED = np.concatenate(truthB)

df_predict2 = pd.DataFrame(predBATCHED, columns=['cotBeta'])
df_true = pd.DataFrame(truthBATCHED, columns=['cotBeta'])

#fix appropriate paths to below
df_predict.to_csv('predictions.csv')
df_true.to_csv('trueLabels.csv')

sns.distplot(df_true['cotBeta']-df_predict['cotBeta'], kde=False, bins=50)
plt.xlabel('cotBeta residual')
plt.ylabel('frequency')
plt.xlim([-.3,.3])
plt.title('Cot Beta Residual')
plt.savefig('cotBeta-resolution.png')
