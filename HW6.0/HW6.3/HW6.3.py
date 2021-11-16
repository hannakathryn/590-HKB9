#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hanna Born
# ANLY 590 Fall 2021
# HW6.0

# references: 
# code on course github at https://github.com/jh2343/590-CODES/
# https://blog.keras.io/building-autoencoders-in-keras.html


# In[2]:


# import packages
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import MinMaxScaler

import keras
from keras import models
from keras import layers
from keras import Model

from keras.datasets import cifar10, cifar100


# In[4]:


# REPEAT 6.2 but swap out
# MNIST for CIFAR-10
# and 
# MNIST-Fashion for CIFAR100


# # write output to log file
# old_stdout = sys.stdout
# log_file = open("HW6.3_log_file.log","w")
# sys.stdout = log_file


# ----------------------------
# USER PARAMETERS 
# for MNIST data
# ----------------------------
EPOCHS          =   10
NKEEP           =   5000        # DOWNSIZE DATASET
# NKEEP           =   50000       # FULL SIZE DATASET
BATCH_SIZE      =   128
N_channels=3
PIX=32


#GET DATA
(X, _), (test_images, _) = cifar10.load_data()
(X_100, Y_train_100), (test_images_100, test_labels_100) = cifar100.load_data()


# REMOVE "TRUCKS" FROM CIFAR100 DATA BEFORE TRAINING
# (TRUCK overlaps with cifar10 and cifar100)
# from https://www.cs.toronto.edu/~kriz/cifar.html
# The classes are completely mutually exclusive. 
# There is no overlap between automobiles and trucks. 
# "Automobile" includes sedans, SUVs, things of that sort. 
# "Truck" includes only big trucks. Neither includes pickup trucks.
# not sure which label corresponds to truck?


#NORMALIZE AND RESHAPE
X = X.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.
X_100 = X_100.astype('float32') / 255.
Y_train_100 = Y_train_100.astype('float32') / 255.


#DOWNSIZE TO RUN FASTER AND DEBUG
print("BEFORE",X.shape)
X=X[0:NKEEP]
test_images=test_images[0:NKEEP]
print("AFTER",X.shape)


# ----------------------------
# MODEL
# BUILD CNN-AE MODEL
# ----------------------------
input_img = keras.Input(shape=(PIX, PIX, N_channels))

#ENCODER
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

#DECODER
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)
    
    
    
# ----------------------------
# COMPILE
# ----------------------------
model = keras.Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']);
model.summary()

# ----------------------------
# TRAIN
# and track training history
# ----------------------------
history = model.fit(X, X,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(test_images, test_images))


# # # save the trained model for future use
model.save('HW6.3_trained_model_50000.h5')

# # load and evaluate the model on the test set
# model = tf.keras.models.load_model('HW6.3_trained_model_2500.h5')
# model.evaluate(test_images, test_images)


# ----------------------------
# HISTORY PLOT
# ----------------------------
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()


# -------------------------------
# MAKE PREDICTIONS FOR TEST DATA
# -------------------------------
decoded_imgs = model.predict(test_images)
decoded_imgs_100 = model.predict(test_images_100)


# #COMPILE AND FIT
# model.compile(optimizer='rmsprop', 
#               loss='mean_squared_error',
#               metrics=['accuracy'])
# model.summary()
# model.fit(X, X, epochs=10, batch_size=1000,validation_split=0.2)


# --------------------------------
# ORIGINAL AND RECONSTRUCTED 
# --------------------------------
X1=model.predict(X)
X1_100=model.predict(X_100)


# ----------------------------
# DEFINE ERROR THRESHOLD
# ----------------------------
# define error threshold for anomaly detection
# if(err > threshold) --> anomaly
threshold = 4*model.evaluate(X,X, batch_size=X.shape[0])
# print('threshold', threshold)


#-------------------------------------
# EVALUATE ON TEST DATA (CIFAR)
# and
# EVALUATE ON TEST DATA (CIFAR-100)
#-------------------------------------
train_loss, train_acc = model.evaluate(X, X, batch_size=X.shape[0])
test_loss, test_acc = model.evaluate(test_images, test_images, batch_size=test_images.shape[0])
test_loss_100, test_acc_100 = model.evaluate(test_images_100, 
                                                     test_images_100, 
                                                     batch_size=test_images_100.shape[0])

print('\nthreshold: \t\t\t', threshold)

print('\ntrain_loss: \t\t\t', train_loss)
print('test_loss (CIFAR): \t\t', test_loss)
print('test_loss (CIFAR-100): \t', test_loss_100)
print('\ntrain_acc: \t\t\t', train_acc)
print('test_acc (CIFAR): \t\t', test_acc)
print('test_acc (CIFAR-100): \t', test_acc_100)


# -------------------------------------------
# COMPARE ORIGINAL AND RECONSTRUCTED IMAGES
# -------------------------------------------
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(test_images[i].reshape(PIX, PIX,N_channels))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(PIX, PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



# ---------------------------
# ANOMALY DETECTION
# ---------------------------
# reference: https://www.analyticsvidhya.com/blog/2021/05/anomaly-detection-using-autoencoders-a-walk-through-in-python/

def anomaly_predictions(model, test_images, threshold):
    predictions = model.predict(test_images)
    errors = tf.keras.losses.mean_squared_logarithmic_error(predictions, 
                                  test_images)
    errors = tf.math.reduce_sum(errors, axis=1, keepdims=False, name=None)
    errors = tf.math.reduce_sum(errors, axis=1, keepdims=False, name=None)/32
    print(errors)
    anomaly_mask = pd.Series(errors > np.mean(threshold))
    preds = anomaly_mask.map(lambda x: 1.0 if x == True else 0.0) # 0 normal, 1 anomaly
    percent_anomalies = sum(preds)/len(preds) * 100
    # return preds # to get predictions
    return print(f'-- {sum(preds):.0f} anomalies / {len(preds)} total \n-- % anomalies:\t {percent_anomalies:.2f}%')
    

# threshold
print(f"threshold: \t {np.mean(threshold):2f}")

# get anomaly predictions for CIFAR (model trained on CIFAR)
print("\nanomaly predictions for (CIFAR trained on CIFAR):")
anomaly_predictions(model, decoded_imgs, threshold)

# get anomaly predictions for CIFAR-100 (model trained on CIFAR)
print("\nanomaly predictions for (CIFAR-100 trained on CIFAR):")
anomaly_predictions(model, decoded_imgs_100, threshold)



# # finish wirting output to log file and close
# sys.stdout = old_stdout
# log_file.close()


# In[ ]:




