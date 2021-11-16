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

from keras.datasets import mnist, fashion_mnist #, cifar10


# In[3]:


# REPEAT 6.1 but instead using a convolutional auto-encoder
# write output to log file
old_stdout = sys.stdout
log_file = open("HW6.1_log_file.log","w")
sys.stdout = log_file


# ----------------------------
# USER PARAMETERS 
# for MNIST data
# ----------------------------
N_channels=1
PIX=28
EPOCHS          =   10          # 35
NKEEP           =   2500        # DOWNSIZE DATASET
# NKEEP           =   60000       # FULL SIZE DATASET
BATCH_SIZE      =   128


# ----------------------------
# GET MNIST DATASET
# ----------------------------
(X, Y), (test_images, test_labels) = mnist.load_data()
# (x_train, _), (x_test, _) = mnist.load_data()

#NORMALIZE AND RESHAPE
X = X.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.


# ----------------------------
# GET MNIST-FASHION DATA
# ----------------------------
from keras.datasets import fashion_mnist

(X_fashion, Y_fashion), (test_images_fashion, test_labels_fashion) = fashion_mnist.load_data()

# NORMALIZE AND RESHAPE
X_fashion=X_fashion.astype('float32') / 255.
test_images_fashion=test_images_fashion.astype('float32') / 255.


# ----------------------------
# MODEL
# BUILD CNN-AE MODEL
# ----------------------------
input_img = keras.Input(shape=(PIX, PIX, N_channels))

# ENCODER --------------------
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# AT THIS POINT THE REPRESENTATION IS (4, 4, 8) I.E. 128-DIMENSIONAL
 
# DECODER --------------------
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
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
# model.save('HW6.1_trained_model_2500.h5')

# # load and evaluate the model on the test set
# model = tf.keras.models.load_model('HW6.1_trained_model_2500.h5')
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
decoded_imgs_fashion = model.predict(test_images_fashion)


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
X1_fashion=model.predict(X_fashion)


# ----------------------------
# DEFINE ERROR THRESHOLD
# ----------------------------
# define error threshold for anomaly detection
# if(err > threshold) --> anomaly
threshold = 4*model.evaluate(X,X, batch_size=X.shape[0])
# print('threshold', threshold)


#-------------------------------------
# EVALUATE ON TEST DATA (MNIST)
# and
# EVALUATE ON TEST DATA (MNIST-FASHION)
#-------------------------------------
train_loss, train_acc = model.evaluate(X, X, batch_size=X.shape[0])
test_loss, test_acc = model.evaluate(test_images, test_images, batch_size=test_images.shape[0])
test_loss_fashion, test_acc_fashion = model.evaluate(test_images_fashion, 
                                                     test_images_fashion, 
                                                     batch_size=test_images_fashion.shape[0])

print('\nthreshold: \t\t\t', threshold)

print('\ntrain_loss: \t\t\t', train_loss)
print('test_loss (MNIST): \t\t', test_loss)
print('test_loss (MNIST-FASHION): \t', test_loss_fashion)
print('\ntrain_acc: \t\t\t', train_acc)
print('test_acc (MNIST): \t\t', test_acc)
print('test_acc (MNIST-FASHION): \t', test_acc_fashion)


# -------------------------------------------
# COMPARE ORIGINAL AND RECONSTRUCTED IMAGES
# -------------------------------------------
fig, axs = plt.subplots(1,6)
axs[0].imshow(test_images[11].reshape(PIX, PIX))
axs[1].imshow(decoded_imgs[11].reshape(PIX, PIX))
axs[2].imshow(decoded_imgs_fashion[11].reshape(PIX, PIX))
axs[3].imshow(test_images[46].reshape(PIX, PIX))
axs[4].imshow(decoded_imgs[46].reshape(PIX, PIX))
axs[5].imshow(decoded_imgs_fashion[46].reshape(PIX, PIX))


# ---------------------------
# ANOMALY DETECTION
# ---------------------------
# reference: https://www.analyticsvidhya.com/blog/2021/05/anomaly-detection-using-autoencoders-a-walk-through-in-python/

def anomaly_predictions(model, test_images, threshold):
    predictions = model.predict(test_images)
    errors = tf.keras.losses.mean_squared_logarithmic_error(predictions, 
                                  test_images)
    errors = tf.math.reduce_sum(errors, axis=1, keepdims=False, name=None)
    errors = tf.math.reduce_sum(errors, axis=1, keepdims=False, name=None)/28
    print(errors)
    anomaly_mask = pd.Series(errors > np.mean(threshold))
    preds = anomaly_mask.map(lambda x: 1.0 if x == True else 0.0) # 0 normal, 1 anomaly
    percent_anomalies = sum(preds)/len(preds) * 100
    # return preds # to get predictions
    return print(f'-- {sum(preds):.0f} anomalies / {len(preds)} total \n-- % anomalies:\t {percent_anomalies:.2f}%')
    

# threshold
print(f"threshold: \t {np.mean(threshold):2f}")

# get anomaly predictions for MNIST (model trained on MNIST)
print("\nanomaly predictions for (MNIST trained on MNIST):")
anomaly_predictions(model, decoded_imgs, threshold)

# get anomaly predictions for MNIST-FASHION (model trained on MNIST)
print("\nanomaly predictions for (MNIST-FASHION trained on MNIST):")
anomaly_predictions(model, decoded_imgs_fashion, threshold)



# finish wirting output to log file and close
sys.stdout = old_stdout
log_file.close()


# In[ ]:




