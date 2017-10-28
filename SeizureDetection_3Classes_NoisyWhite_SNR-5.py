
# coding: utf-8

# In[1]:

# Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten, TimeDistributed, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras import metrics
from keras import backend

get_ipython().magic('matplotlib inline')


# In[2]:

# Define the performance metrics of rmse, sensitivity, recall, specificity, and precision

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis = -1))

def sensitivity(y_true, y_pred):  
     y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
     y_pos = backend.round(backend.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
     tp = backend.sum(y_pos * y_pred_pos)
     tn = backend.sum(y_neg * y_pred_neg)
     fp = backend.sum(y_neg * y_pred_pos)
     fn = backend.sum(y_pos * y_pred_neg)
     sensitivity = tp / (tp + fn)
     return sensitivity

# Recall is the same as the sensitivity
def recall(y_true, y_pred):  
     y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
     y_pos = backend.round(backend.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
     tp = backend.sum(y_pos * y_pred_pos)
     tn = backend.sum(y_neg * y_pred_neg)
     fp = backend.sum(y_neg * y_pred_pos)
     fn = backend.sum(y_pos * y_pred_neg)
     recall = tp / (tp + fn)
     return recall

def specificity(y_true, y_pred):  
     y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
     y_pos = backend.round(backend.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
     tp = backend.sum(y_pos * y_pred_pos)
     tn = backend.sum(y_neg * y_pred_neg)
     fp = backend.sum(y_neg * y_pred_pos)
     fn = backend.sum(y_pos * y_pred_neg)
     specificity = tn / (tn + fp)
     return specificity


def precision(y_true, y_pred):  
     y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
     y_pos = backend.round(backend.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
     tp = backend.sum(y_pos * y_pred_pos)
     tn = backend.sum(y_neg * y_pred_neg)
     fp = backend.sum(y_neg * y_pred_pos)
     fn = backend.sum(y_pos * y_pred_neg)
     precision = tp / (tp + fp)
     return precision


# In[3]:

#import data 
# data=pd.read_csv('/home/ramyh/Documents/ClassDeepLearning/EEGdata.csv', names=['f1':'f2049'])
data=pd.read_csv('/home/ramyh/Documents/ClassDeepLearning/EEGData_3Classes_NoisyWhite_SNR-5.csv') # dataset including 1000 samples from each class


# In[4]:

data["f4097"].value_counts()


# In[5]:

# shuffle the data
data=data.iloc[np.random.permutation(len(data))]


# In[6]:

# reset the data index
data=data.reset_index(drop=True)


# In[7]:

# Time Steps of LSTM
data_length = 4096
timesteps = 2048
data_dim = data_length//timesteps
data_dim


# In[8]:

# training data
X_train=data.loc[0:2399,data.columns != 'f4097']
# training data.reshape
# X_train=X_train.values.reshape([X_train.shape[0], -1, 1])
# X_train=X_train.values.reshape([X_train.shape[0], 64, 64])
X_train=X_train.values.reshape([X_train.shape[0], timesteps, data_dim])

temp=data['f4097']
y_train=temp[0:2400]
# map data into arrays
y_train=np_utils.to_categorical(y_train, num_classes=3)

# test data
X_test=data.loc[2400:2999,data.columns != 'f4097']

# test data.reshape
# X_test = X_test.values.reshape([X_test.shape[0], -1, 1])
# X_test = X_test.values.reshape([X_test.shape[0], 64, 64])
X_test = X_test.values.reshape([X_test.shape[0], timesteps, data_dim])

y_test=temp[2400:3000]
# map data into arrays
y_test=np_utils.to_categorical(y_test, num_classes=3)

# https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# https://faroit.github.io/keras-docs/0.3.3/examples/
# https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/


# In[9]:

X_test.shape


# In[10]:

# create the model
model = Sequential()
# model.add(LSTM(100, input_shape= (4096, 1)))
# model.add(LSTM(100, input_shape= (64, 64)))
# model.add(Dropout(0.1, input_shape= (timesteps, data_dim)))
# model.add(LSTM(100, return_sequences = True))

model.add(LSTM(100, input_shape= (timesteps, data_dim), return_sequences = True))
# model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(50)))
# model.add(GlobalMaxPooling1D())
model.add(GlobalAveragePooling1D())
# model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[sensitivity, specificity, 'accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=40)


# In[11]:

# Plot the sensitivity along with epochs
plt.plot(history.history['sensitivity'], 'g--')
plt.title('Model Sensitivity')
plt.ylabel('Sensitivity')
plt.xlabel('Epoch #')
plt.show()


# In[12]:

# Plot the specificity along with epochs
plt.plot(history.history['specificity'], 'b--')
plt.title('Model Specificity')
plt.ylabel('Specificity')
plt.xlabel('Epoch #')
plt.show()


# In[13]:

# Plot the classification accuracy along with epochs
plt.plot(history.history['acc'], 'r--')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch #')
plt.show()


# In[14]:

# evaluate the classification accuracy
scores = model.evaluate(X_test, y_test, verbose=0)
print("Sensitivity = %.2f%%" % (scores[1]*100))
print("Specificity = %.2f%%" % (scores[2]*100))
print("Classification Accuracy = %.2f%%" % (scores[3]*100))


# In[ ]:



