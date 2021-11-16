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

from keras import models
from keras import layers
from keras import Model

from keras.datasets import mnist
from keras.datasets import fashion_mnist


# In[4]:


# # write output to log file
# old_stdout = sys.stdout
# log_file = open("HW6.1_log_file.log","w")
# sys.stdout = log_file


# ----------------------------
# GET MNIST DATA
# ----------------------------
(X, Y), (test_images, test_labels) = mnist.load_data()

# NORMALIZE AND RESHAPE
X=X/np.max(X) 
X=X.reshape(60000,28*28); 
test_images=test_images/np.max(test_images) 
test_images=test_images.reshape(10000,28*28)


# ----------------------------
# GET MNIST-FASHION DATA
# ----------------------------
(X_fashion, Y_fashion), (test_images_fashion, test_labels_fashion) = fashion_mnist.load_data()

# NORMALIZE AND RESHAPE
X_fashion=X_fashion/np.max(X_fashion) 
X_fashion=X_fashion.reshape(60000,28*28); 
test_images_fashion=test_images_fashion/np.max(test_images) 
test_images_fashion=test_images_fashion.reshape(10000,28*28)


# ----------------------------
# BUILD MODEL
# ----------------------------
n_bottleneck=100

# # SHALLOW
# model = models.Sequential()
# model.add(layers.Dense(n_bottleneck, activation='linear', input_shape=(28 * 28,)))
# model.add(layers.Dense(28*28,  activation='linear'))

#DEEPER
model = models.Sequential()
NH=300
model.add(layers.Dense(n_bottleneck, activation='linear', input_shape=(28 * 28,)))
model.add(layers.Dense(NH, activation='relu'))
# model.add(layers.Dense(NH, activation='relu', input_shape=(28 * 28,)))
# model.add(layers.Dense(n_bottleneck, activation='relu'))
# model.add(layers.Dense(n_bottleneck, activation='relu'))
# model.add(layers.Dense(NH, activation='relu'))
# model.add(layers.Dense(NH, activation='relu'))
model.add(layers.Dense(28*28,  activation='linear'))


# ----------------------------
# COMPILE AND FIT
# ----------------------------
model.compile(optimizer='rmsprop', 
              loss='mean_squared_error',
              metrics=['accuracy'])
model.summary()
model.fit(X, X, epochs=10, batch_size=1000,validation_split=0.2)


# EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
extract = Model(model.inputs, model.layers[-2].output) # Dense(128,...)
X1 = extract.predict(X)
print(X1.shape)
X1_fashion = extract.predict(X_fashion)

#2D PLOT
plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
plt.show()

#3D PLOT
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=X1[:,0], 
    ys=X1[:,1], 
    zs=X1[:,2], 
    c=Y, 
    cmap='tab10'
)
plt.show()


# --------------------------------
# PLOT ORIGINAL AND RECONSTRUCTED 
# using MNIST and MNIST-FASHION
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
print('test_loss (MNIST-FASHIOIN): \t', test_loss_fashion)
print('\ntrain_acc: \t\t\t', train_acc)
print('test_acc (MNIST): \t\t', test_acc)
print('test_acc (MNIST-FASHIOIN): \t', test_acc_fashion)


# ----------------------------
# RESHAPE
# ----------------------------
X=X.reshape(60000,28,28); #print(X[0])
X1=X1.reshape(60000,28,28); #print(X[0])

X1_fashion=X1_fashion.reshape(60000,28,28); #print(X[0])


# ----------------------------
#COMPARE ORIGINAL 
# ----------------------------
f, ax = plt.subplots(1,6)
I1=11; I2=46
ax[0].imshow(X[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(X1_fashion[I1])  # fashion ... visible error
ax[3].imshow(X[I2])
ax[4].imshow(X1[I2])
ax[5].imshow(X1_fashion[I2])  # fashion ... visible error
plt.show()


# # # save the trained model for future use
# model.save('HW6.1_trained_model.h5')

# # In[]:

# # # load and evaluate the model on the test set
# # model = tf.keras.models.load_model('HW6.1_trained_model.h5')
# # model.evaluate(test_images, test_images)


# ------------------------------------------
# ANOMALY DETECTION
# ---- define function to detect anomalies
# ------------------------------------------
# reference: https://www.analyticsvidhya.com/blog/2021/05/anomaly-detection-using-autoencoders-a-walk-through-in-python/

def anomaly_predictions(model, test_images, threshold):
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    obj=min_max_scaler.fit(test_images)
    test_images_scaled = min_max_scaler.transform(test_images.copy())
    predictions = model.predict(test_images_scaled)
    errors = tf.keras.losses.mean_squared_logarithmic_error(predictions, 
                                  test_images_scaled)
    anomaly_mask = pd.Series(errors) > np.mean(threshold)
    preds = anomaly_mask.map(lambda x: 1.0 if x == True else 0.0) # 0 normal, 1 anomaly
    percent_anomalies = sum(preds)/len(preds) * 100
    # return preds # to get predictions
    return print(f'-- {sum(preds):.0f} anomalies / {len(preds)} total \n-- % anomalies:\t {percent_anomalies:.2f}%')
    

# -----------------------------------
# DETECT ANOMALIES
# in MNIST and MNIST-FASHION
# using model trained on MNIST data
# -----------------------------------
# threshold
print(f"threshold: \t {np.mean(threshold):2f}")

# get anomaly predictions for MNIST (model trained on MNIST)
print("\nanomaly predictions for (MNIST trained on MNIST):")
anomaly_predictions(model, test_images, threshold)

# get anomaly predictions for MNIST-FASHION (model trained on MNIST)
print("\nanomaly predictions for (MNIST-FASHION trained on MNIST):")
anomaly_predictions(model, test_images_fashion, threshold)



# # finish wirting output to log file and close
# sys.stdout = old_stdout
# log_file.close()

