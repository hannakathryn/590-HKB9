#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hanna Born
# ANLY 590 Fall 2021
# reference: course code at https://github.com/jh2343/590-CODES


# In[2]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
from numpy import load
from sklearn.model_selection import train_test_split
from keras import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.layers import Embedding, Flatten, Dense, SimpleRNN
from keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[3]:


# https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa/56062555
np_load_old = np.load # save np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k) # modify np.load default parameters

# load the data saved from clean.py
data = np.load('data.npy')
labels = np.load('labels.npy')

np.load = np_load_old # restore np.load for future normal usage


# In[4]:


# train, test, validation split
# set seed for consistency (for recreating same validation set)
x_train, x_test, y_train, y_test = train_test_split(data, labels, 
                                                    test_size=0.20,
                                                    random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                    test_size=0.25,
                                                    random_state=42) # .25 * .8 = .2


# In[5]:


# instantiate the embedding layer
embedding_layer = Embedding(1000, 64)


# In[6]:


max_features = 10000
maxlen = 20

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
x_val = preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)


# In[7]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


# In[8]:


# use an embedding layer and classifier on the data
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))

model.add(Flatten())

model.add(Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
              metrics=['acc','mse', tf.keras.metrics.AUC(from_logits=True)])
model.summary()

history = model.fit(x_val, y_val,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.25)


# In[9]:


# 1D CNN model
# define the model
# training a models without pretrained word embeddings
# conv_input_train = np.expand_dims(x_train, axis=1)

max_words = 10000    # 10000
embedding_dim = 100  # 100

model = Sequential()
# model.add(tf.keras.layers.Conv1D(32, 3, activation='relu')(conv_input_train))
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
              metrics=['acc','mse', tf.keras.metrics.AUC(from_logits=True)])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))


# In[10]:


# # save the trained model for future use
# model.save_weights('pre_trained_1D_model.h5')
model.save('pre_trained_1D_model_WIKI.h5')


# In[11]:


# # load and evaluate the model on the test set
# model = Sequential()
# model.load_weights('pre_trained_1D_model.h5')
model = tf.keras.models.load_model('pre_trained_1D_model_WIKI.h5')
model.evaluate(x_test, y_test)


# In[12]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
mse = history.history['mse']
val_mse = history.history['val_mse']
auc = history.history['auc_1']
val_auc = history.history['val_auc_1']

epochs = range(1, len(acc) + 1)

# plot and save fig: ACC
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('cnn_acc.png')
plt.figure()

# plot and save fig: Loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('cnn_loss.png')
plt.figure()

# plot and save fig: MSE 
plt.plot(epochs, mse, 'bo', label='Training mse')
plt.plot(epochs, val_mse, 'b', label='Validation mse')
plt.title('Training and validation mse')
plt.legend()
plt.savefig('cnn_mse.png')
plt.figure()

# plot and save fig: AUC
plt.plot(epochs, auc, 'bo', label='Training auc')
plt.plot(epochs, val_auc, 'b', label='Validation auc')
plt.title('Training and validation auc')
plt.legend()
plt.savefig('cnn_auc.png')
plt.show()


# In[13]:


# simpleRNN model
# train the model with embedding and simpleRNN

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32, 
                    dropout = 0.1)) # include dropout regularization ... tried 0.05, 0.1, 0.2
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['acc', 'mse', tf.keras.metrics.AUC(from_logits=True)])
history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=128,
                    validation_split=0.25)


# In[14]:


# save the trained model for future access
model.save('pre_trained_simpleRNN_model_WIKI.h5')

# ## evaluate the model on the test set
# model = tf.keras.models.load_model('pre_trained_simpleRNN_model_WIKI.h5')
# model.evaluate(x_test, y_test)


# In[15]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
mse = history.history['mse']
val_mse = history.history['val_mse']
auc = history.history['auc_2']
val_auc = history.history['val_auc_2']

epochs = range(1, len(acc) + 1)

# plot and save fig: ACC
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('rnn_acc.png')
plt.figure()

# plot and save fig: Loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('rnn_loss.png')
plt.figure()

# plot and save fig: MSE 
plt.plot(epochs, mse, 'bo', label='Training mse')
plt.plot(epochs, val_mse, 'b', label='Validation mse')
plt.title('Training and validation mse')
plt.legend()
plt.savefig('rnn_mse.png')
plt.figure()

# plot and save fig: AUC
plt.plot(epochs, auc, 'bo', label='Training auc')
plt.plot(epochs, val_auc, 'b', label='Validation auc')
plt.title('Training and validation auc')
plt.legend()
plt.savefig('rnn_auc.png')
plt.show()


# In[16]:


# write metrics to a log file, log.txt
# include training info and final metrics
with open('log.txt', 'w') as f:
    f.write('\n Training Metrics:' +
            '\n\t acc: \t ' + str(acc) +
           '\n\t loss: \t ' + str(loss) +
           '\n\t mse: \t ' + str(mse) +
           '\n\t auc: \t ' + str(auc) +
           '\n\n Final Metrics:' +
            '\n\t acc: \t ' + str(np.mean(val_acc)) +
           '\n\t loss: \t ' + str(np.mean(val_loss)) +
           '\n\t mse: \t ' + str(np.mean(val_mse)) +
           '\n\t auc: \t ' + str(np.mean(val_auc)))


# In[ ]:




