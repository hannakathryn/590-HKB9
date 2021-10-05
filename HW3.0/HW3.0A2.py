#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hanna Born 
# ANLY 590: Neural Nets and Deep Learning
# HW3.0A2.py

# references:
# Textbook (Collet, Deep Learning with Python. Manning Press.) Ch.3 p.68


# In[2]:


"""
Code Binary Classification using KERAS
Train on IMDB data set
"""

import warnings
warnings.filterwarnings("ignore")

# LOAD THE IMDB DATSET -------------------------------------------------
from keras.datasets import imdb

# 50,000 highly polarized views from Internet movie database
# 25,000 training, 25,000 testing, each 50% negative / 50% positive

# num_words - 10000: only keep top 10k most frequently occurring words
# discarding rare wors keeps vectors at manageable size
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000) 


# In[3]:


# the variables train_data and test_data are lists of reviews; 
# each review is a list of word indices 
# (encoding a sequence of words) 
train_data[0]


# In[4]:


# train_labels and test_labels are lists of 0s and 1s, 
# where 0 stands for negative and 1 stands for positive
train_labels[0]


# In[5]:


# no word index will exceed 10000 (restriced to top 10k words)
max([max(sequence) for sequence in train_data])


# In[6]:


# # TO DECODE REVIEW BACK TO ENGLISH IF DESIRED
# word_index = imdb.get_word_index()
# reverse_word_index = dict(
#     [(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join(
#     [reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# In[7]:


# ENCODE THE INTEGER SEQUENCES INTO A BINARY MATRIX --------------------
# cannot feed a list of integers into a neural network
# must turn lists into tensors
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[8]:


# check what samples look like now
# array([ 0.,  1.,  1., ...,  0.,  0.,  0.])
x_train[0] 


# In[9]:


# VECTORIZE LABELS -----------------------------------------------------
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# now the data is ready to be fed into a neural network


# In[10]:


# BUILD THE NETWORK ----------------------------------------------------
# input data is vectors
# labels are scalars (1s and 0s)

# DEFINE THE MODEL
from keras import models
from keras import layers

# ReLU activation function for non-linearity
# otherwise the dense layer would only consist of two linear operations
# (a dot product and an addition)
# ... so the layer could only learn linear transformations
model = models.Sequential()
# two intermediate layers
# number of hidden units of the layers = 16 in each
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
# third layer that will output the scalar prediction
# regarding sentiment of current review
model.add(layers.Dense(1, activation='sigmoid'))


# In[11]:


# COMPILE THE MODEL
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[12]:


# CONFIGURE THE OPTIMIZER
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[13]:


# USING CUSTOM LOSSES AND METRICS
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])


# In[14]:


# VALIDATION SET
# create a validation set using 10000 samples from training data
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[15]:


# TRAIN THE MODEL ------------------------------------------------------
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
# train the model for 20 epochs (iterations)
# over all samples in the x_train and y_train tensors
# in min-batches of 512 samples
# simultaneously monitor loss and accuracy on validation samples
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512, 
                    validation_data=(x_val, y_val))


# In[16]:


# call to model.fit() returns a History object. This object has a member history, 
# which is a dictionary containing data about everything that happened during training
# [u'acc', u'loss', u'val_acc', u'val_loss']
history_dict = history.history
history_dict.keys()


# In[17]:


# PLOT THE TRAINING AND VALIDATION LOSS --------------------------------
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[18]:


# # PLOT THE TRAINING AND VALIDATION ACCURACY ----------------------------
# plt.clf()
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

