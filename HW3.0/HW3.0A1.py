#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hanna Born 
# ANLY 590: Neural Nets and Deep Learning
# HW3.0A1.py
# code ANN regression using KERAS, train on the Boston housing dataset

# references:
# Textbook (Collet, Deep Learning with Python. Manning Press.) Ch.3 p.85


# In[2]:


"""
Code ANN regression using KERAS
Train on the Boston housing dataset
"""

import warnings
warnings.filterwarnings("ignore")

# LOAD THE BOSTON HOUSING DATSET ------------------------------------
from keras.datasets import boston_housing

# 506 samples total ... 404 training, 102 test
# note: 13 numerical features in the input data ... each has a different scale
(train_data, train_targets), (test_data, 
                              test_targets) = boston_housing.load_data()


# In[3]:


train_data.shape


# In[4]:


test_data.shape


# In[5]:


# targets are median vals of homes in thousands of $
# note: most prices $10k-50k ... 1970s not adjusted for inflation
train_targets


# In[6]:


# NORMALIZING THE DATA ----------------------------------------------
# feature-wise normalization
# note that quantities used to normalize test data are computed using
# training data! shouldnt use quantities computed on test set in workflow
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# In[7]:


# MODEL DEFINITION --------------------------------------------------
# scalar regression
# small network w/ 2 hidden layers, 64 units each
# since the training data has so few samples, a small network is one
# way to mitigate overfitting

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    # hidden layer 1: 64 units
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    # hidden layer 2: 64 units
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# In[8]:


# K FOLD VALIDATION -------------------------------------------------
# evaluating network while adjusting hyper-parameters (num_epochs)
# is tricky when so little data is available
# small validation sets might lead to high variance 
# with regard to the validation split
# to mitigate we can use K-fold CV to partition and evaluate

Chollet, Francois. Deep Learning with Python (p. 87). Manning. Kindle Edition. 

import numpy as np
k = 4

num_val_samples = len(train_data) // k
num_epochs = 50
all_scores = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


# In[9]:


all_scores


# In[10]:


np.mean(all_scores)


# In[11]:


# SAVE THE VALIDATION LOGS AT EACH FOLD -----------------------------
num_epochs = 300  # reduced epochs from 500 in book code to preserve kernel
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)


# In[12]:


average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# In[13]:


average_mae_history


# In[14]:


# import matplotlib.pyplot as plt
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()


# In[15]:


# PLOTTING VALIDATION SCORES ----------------------------------------
# -------------------------- (excluding the first 10 data points) ---
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# In[16]:


# TRAIN THE FINAL RESULT --------------------------------------------
model = build_model()
model.fit(train_data, train_targets,
          epochs=300, batch_size=16, verbose=0) # book used epochs = 80
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

test_mae_score
# book result: 
# 2.5532484335057877 ... off by ~ $2,550

# my result:
# 2.34812268614769 ... off by ~ $2,350 ... slightly better

