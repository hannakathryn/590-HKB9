#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hanna Born 
# ANLY 590: Neural Nets and Deep Learning
# HW3.0A3.py

# references:
# Textbook (Collet, Deep Learning with Python. Manning Press.) Ch.3 p.78


# In[2]:


"""
Code Multi-class classification using KERAS
Train on newswire data set
"""

import warnings
warnings.filterwarnings("ignore")

# LOAD THE NEWSWIRE DATSET ---------------------------------------------
from keras.datasets import reuters

# dataset of short newswires and their topics
# 46 different topics, each with at least 10 examples in training set

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000) # restricts data to 10000 most frequently occuring words


# In[3]:


len(train_data) # 8982 training examples
len(test_data) # 2246 test examples

# each example is a list of integers (word indices):
train_data[10] # [1, 245, 273, ..., 12] 


# In[4]:


# # DECODING NEWSWIRES BACK TO TEXT IF DESIRED -------------------------
# word_index = reuters.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in
#     train_data[0]])
# train_labels[10] # 3 


# In[5]:


# label associated with an example is an integer 0 to 45
# represents a topic index
train_labels[10] # 3


# In[6]:


# PREPARING THE DATA ... vectorize the data ----------------------------
import numpy as np
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# now the data is ready to be fed into a neural network


# In[7]:


# ENCODING THE DATA ----------------------------------------------------
# cannot feed a list of integers into a neural network
# must turn lists into tensors
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[9]:


# ONE HOT ENCODING -----------------------------------------------------

# One Hot Encoding from Scratch
# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results

# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)


# One Hot Encoding using KERAS built in option
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


# In[10]:


# MODEL DEFINITION -----------------------------------------------------
# the number of output classes has gone from 2 to 46. 
# dimensionality of the output space is much larger than binary problem

from keras import models
from keras import layers

model = models.Sequential()
# in stack of dense layers, each layer can only access information 
# in output of the previous layer. if a layer drops information 
# relevant to the classification problem, it can never be recovered
# by later layers: each layer can potentially become an information bottleneck
# 16 dimensional space may be too limitted to learn to separate 
# 46 different classes (small layers may be information bottlenecks)
# for this reason, use larger layers -- 64 units
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))

# end network with dense layer of size 46
# so for each input sample, network outpus a 46-dimensional vector
# each entry(dimension) in the vector encodes different output class
model.add(layers.Dense(46, activation='softmax'))

# note: last layer uses softmax
# means network will output a probablility distribution over 46d
# so output[i] is probability sample belongs to class i
# the 46 scores will sum to 1 !


# In[11]:


# COMPILING THE MODEL --------------------------------------------------
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[12]:


# VALIDATION SET--------------------------------------------------------
# use 1000 samples from training data for validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


# In[15]:


# TRAINING THE MODEL ---------------------------------------------------
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# In[16]:


# PLOT TRAINING AND VALIDATION LOSS ------------------------------------
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[17]:


# # PLOT TRAINING AND VALIDATION ACCURACY IF DESIRED ---------------------
# plt.clf()

# acc = history.history['acc']
# val_acc = history.history['val_acc']

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
