#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hanna Born
# ANLY 590 Fall 2021
# reference: course code at https://github.com/jh2343/590-CODES


# In[3]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import keras
from numpy import load
from sklearn.model_selection import train_test_split
from keras import preprocessing
from tensorflow.keras.utils import to_categorical


# In[4]:


# load in the data for use
np_load_old = np.load # save np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k) # modify np.load default parameters
data = np.load('data.npy')
labels = np.load('labels.npy')
np.load = np_load_old # restore np.load for future normal usage


# In[5]:


# train/test/val split
x_train, x_test, y_train, y_test = train_test_split(data, labels, 
                                                    test_size=0.20,
                                                    random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                    test_size=0.25,
                                                    random_state=42) # .25 * .8 = .2


# In[6]:


# minor processing steps before use with model
max_features = 30000
maxlen = 20

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
x_val = preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


# In[7]:


# load first model and evaluate
model_cnn = tf.keras.models.load_model('pre_trained_1D_model_WIKI.h5')
results = model_cnn.evaluate(x_test, y_test)
# print out formatted metrics from evaluation
print('Model Performance Metrics:'+
      '\n\t Loss: \t\t ' + str(results[0]) +
      '\n\t Accuracy: \t ' + str(results[1]) +
      '\n\t MSE: \t\t ' + str(results[2]) +
      '\n\t AUC: \t\t ' + str(results[3]))


# In[8]:


# load second model and evaluate
model_rnn = tf.keras.models.load_model('pre_trained_simpleRNN_model_WIKI.h5')
results2 = model_rnn.evaluate(x_test, y_test)
# print out formatted metrics from evaluation
print('Model Performance Metrics:'+
      '\n\t Loss: \t\t ' + str(results2[0]) +
      '\n\t Accuracy: \t ' + str(results2[1]) +
      '\n\t MSE: \t\t ' + str(results2[2]) +
      '\n\t AUC: \t\t ' + str(results2[3]))

