#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## to access data
# url_test = 'https://www.kaggle.com/c/dogs-vs-cats/data?select=test1.zip'
# url_train = 'https://www.kaggle.com/c/dogs-vs-cats/data?select=train.zip'


# In[1]:


import warnings
warnings.filterwarnings("ignore")

## GET DATA READY -------------------------------------------------------------
# copy images to training, validation and test directories

# note: full dataset contains 25000 images of dogs and cats (12500 per class)
# Below, creating a new dataset containing three subsets: 
# -- training set with 1000 samples of each class
# -- validation set with 500 samples of each class
# -- test set with 500 samples of each class

import os, shutil

## ADJUST FOR APPROPRIATE PATH DIRECTORY --------------------------------------
# using train.zip from Kaggle
original_dataset_dir = '/Users/hanna/Downloads/kaggle_original_data_train'

base_dir = '/Users/hanna/Downloads/cats_and_dogs_small'
os.mkdir(base_dir)
## ----------------------------------------------------------------------------


# create three new directories for three train/val/test subsets of data
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# TRAIN .. subset for each of 2 classes
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# VALIDATION .. subset for each of 2 classes
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# TEST .. subset for each of 2 classes
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# standard naming for train/val/test subsets for Class cat
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst) 
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# standard naming for train/val/test subsets for Class dog
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)  
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


# In[2]:


## CHECK TRAIN/VALIDATION/TEST SPLIT ------------------------------------------
# check # of pictures in each training split (train/validation/test):
print('total training cat images:', len(os.listdir(train_cats_dir)))
# total training cat images: 1000
print('total training dog images:', len(os.listdir(train_dogs_dir)))
# total training dog images: 1000
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
# total validation cat images: 500
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
# total validation dog images: 500
print('total test cat images:', len(os.listdir(test_cats_dir)))
# total test cat images: 500
print('total test dog images:', len(os.listdir(test_dogs_dir)))
# total test dog images: 500

# 2000 training images, 1000 validation images, 1000 test images
# each split contains same # of samples from each class: 
# this is a balanced binary-classification problem, 
# so classification accuracy will be an appropriate measure of success


# In[ ]:


## BUILD NETWORK --------------------------------------------------------------
# binary-classification problem
# end network w/ a single unit (Dense layer of size 1) and a sigmoid activation
# which will encode probability network is looking at one class or the other

## INSTANTIATE SMALL COVNET FOR CLASSIFICATION (Dog vs. Cat)
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# check how dimensions of feature maps change w/ every successive layer
model.summary()

## CONFIGURE MODEL FOR TRAINING -----------------------------------------------
# compilation step: RMSprop optimizer
# loss: binary cross_entropy because ended network w/ a single sigmoid unit
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

## DATA PREPROCESSING
# before being fed into network, data should be formatted into 
# appropriately preprocessed floating-point tensors 
# Current format is JPEG so will:
# -- read picture files
# -- decode JPEG content to RGB grids of pixels 
# -- convert into floating-point tensors
# -- rescale pixel values (between 0 and 255) to the [0, 1] interval 
#    since Neural Networks prefer small input values
# use ImageDataGenerator to read images from directories
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# check output of one of these generators: 
# yields batches of 150 × 150 RGB images (shape (20, 150, 150, 3)) 
# and binary labels (shape (20,)). 
# 20 samples in each batch (the batch size)
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    # Note: generator yields these batches indefinitely...
    # it loops endlessly over images in target folder. 
    # For this reason, you need to break the iteration loop at some point
    break
    
    
## FIT MODEL USING A BATCH GENERATOR ------------------------------------------
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=10, #30,
      validation_data=validation_generator,
      validation_steps=50) 
      # validation_steps tells process how many batches to draw from 
      # the validation generator for evaluation.

## SAVE THE MODEL -------------------------------------------------------------
model.save('cats_and_dogs_small_1.h5') # save model after training


## TRAINING AND VALIUDATION ACCURACY ------------------------------------------
# plot loss and accuracy of model over training & validation data during training
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# note: these plots are characteristic of overfitting
# training accuracy increases linearly over time, until it reaches nearly 100%, 
# whereas validation accuracy stalls at 70–72%. 
# validation loss reaches its minimum after only five epochs then stalls, 
# whereas training loss keeps decreasing linearly until it reaches nearly 0



## DATA AUGMENTATION --------------------------------------------------------
# overfitting caused by having too few samples to learn from, 
# rendering you unable to train a model that can generalize to new data
# if indefinite data, model exposed to every possibility; couldnt overfit

# data augmentation to help with overfitting
# generates more training data from existing training samples
# by augmenting the samples via a number of random transformations 
# that yield believable-looking images
# then for training, model will never see the exact same picture twice
# helps expose model to more aspects of the data and generalize better

# set up data augmentation configuration via ImageGenerator
datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# ^ rotation_range: value in degrees (0–180)
# -- a range within which to randomly rotate pictures
# width_shift, height_shift: ranges (as a fraction of total width or height) 
# -- within which to randomly translate pictures vertically or horizontally
# shear_range: for randomly applying shearing transformations
# zoom_range: for randomly zooming inside pictures
# horizontal_flip: for randomly flipping half the images horizontally
# -- relevant when there are no assumptions of horizontal asymmetry 
# -- (for example, real-world pictures)
# fill_mode: the strategy used for filling in newly created pixels
# -- which can appear after a rotation or a width/height shift.

## GENERATE CAT PICTURES VIS RANDOM DATA AUGMENTATION
from keras.preprocessing import image
fnames = [os.path.join(train_cats_dir, fname) for
     fname in os.listdir(train_cats_dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

# note: after data augmentation, the inputs seen are still heavily intercorrelated,
# because they come from a small number of original images
# —you can’t produce new information, you can only remix existing information
# so it may not be enough to completely get rid of overfitting
# to fight overfitting, can also add a Dropout layer to model, 
# right before the densely connected classifier.

## DEFINE A NEW COVNET THAT INCLUDES DROPOUT
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

## TRAIN COVNET USING DATA-AUGMENTATION GENERATORS
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10, #100,
    validation_data=validation_generator,
    validation_steps=50)

model.save('cats_and_dogs_small_2.h5') # save the model


# plot the results again
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# note: w/ data augmentation and dropout, are no longer overfitting
# -- training curves closely tracking the validation curves
# -- ~82% accuracy is ~15% improvement from non-regularized model


# SKIP FEATURE EXTRACTION AND FINE-TUNING SECTIONS (require a GPU)

## VISUALIZING INTERMEDIATE ACTIVATIONS ---------------------------------------
# displaying feature maps that are output by 
# various convolution and pooling layers in a network, given a certain input 
# (output of a layer often called its activation, output of activation function). 
# gives view into how an input is decomposed into different filters learned by network
# want to visualize feature maps with three dimensions: 
# -- width, height, and depth (channels)
# -- each channel encodes relatively independent features
#    so proper way to visualize these feature maps is by 
#    independently plotting contents of every channel as a 2D image

from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5') # load model saved earlier
model.summary() # take a look at model


## PREPROCESS A SINGLE IMAGE
img_path = '/Users/hanna/Downloads/cats_and_dogs_small/test/cats/cat.1700.jpg'

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

## display the test picture
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

## INSTANTIATE MODEL FROM AN INPUT TENSOR AND A LIST OF OUTPUT TENSORS -------
# when fed an image input, this model returns the values of the layer activations 
# in the original model 
# a multi-output model: 1 input and 8 outputs (one output per layer activation)
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

## RUNNING THE MODEL IN PREDICT MODE
activations = activation_model.predict(img_tensor)

# activation of the first convolutional layer for the cat image input
# a 148 × 148 feature map with 32 channels
first_layer_activation = activations[0]
print(first_layer_activation.shape) # (1, 148, 148, 32)

# plot fourth channel of activation of the first layer of the original model
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

# visualize seventh channel 
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')


## VISUALIZING EVERY CHANNEL IN EVERY INTERMEDIATE ACTIVATION ----------------
# plot a complete visualization of all the activations in the network
# extract and plot every channel in each of the eight activation maps
# stack the results in one big image tensor, with channels stacked side by side

# first layer acts as collection of various edge detectors
# -- activations here still retain almost all information present in initial picture
# as go higher, activations become increasingly abstract / less visually interpretable
# -- begin to encode higher-level concepts ('cat ear', 'cat eye', etc.)
# -- carry incresingly less information about visual contents of image
# -- increasingly more information related to the class of the image
# sparsity of activations increases with the depth of the layer
# -- in first layer, all filters are activated by the input image
# -- but in following layers, more and more filters are blank
# --     means the pattern encoded by the filter isn’t found in the input image

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    
    size = layer_activation.shape[1]
    
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


# In[ ]:




