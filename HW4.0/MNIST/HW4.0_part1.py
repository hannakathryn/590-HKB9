#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Hanna Born
# ANLY 590 Fall 2021

# references:
# Chollett Chapters 2-5
# code examples on course github at https://github.com/jh2343/590-CODES

import warnings
warnings.filterwarnings("ignore")

from keras import layers 
from keras import models
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame

# Flag for which dataset to train on
# Uncomment to specify which dataset to use: MNIST, MNIST Fashion, CIFAR
#----------------------------------------------------------------------------------
## SELECTIONS
#----------------------------------------------------------------------------------
## SELECT DATA TO USE
dataset = "MNIST"
# dataset = "Fashion"
# dataset = "CIFAR"

## SELECT MODEL TYPE
# model_type = "DFF" # Dense Feed forward ANN (as a reference benchmark)
model_type = "CNN" # A convolutional NN (CNN)

# SELECT DATA AUGMENTATION flag as True or False
# data_aug = True
data_aug = False
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
## BUILD MODEL
#----------------------------------------------------------------------------------
if model_type == "CNN":
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

elif model_type == "DFF":
    # two Dense layers
    # second/last layer is a 10-way softmax layer, 
    # so it will return an array of 10 probability scores (summing to 1)
    # each score is P(current digit image belongs to one of the 10 digit classes)

    #INITIALIZE MODEL
        # Sequential model --> plain stack of layers
        # each layer has exactly one input tensor and one output tensor.
    model = models.Sequential()
    #ADD LAYERS
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    #SOFTMAX  --> 10 probability scores (summing to 1
    model.add(layers.Dense(10,  activation='softmax'))
    #COMPILATION (i.e. choose optimizer, loss, and metrics to monitor)
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    # rmsprop optimizer
    # -- mechanism network uses to update itself based on seen data & loss function
    # categorical crossentropy loss function
    # -- for network to measure its performance on the training data, 
    # -- and thus how it will be able to steer itself in the right direction.
    # accuracy Metric (the fraction of the images that were correctly classified)
    # -- to monitor during training and testing ... here, only care about accuracy
model.summary()

#----------------------------------------------------------------------------------
## GET DATA AND REFORMAT
#----------------------------------------------------------------------------------
from keras.datasets import mnist
from keras.utils import to_categorical
# if statements to determine which one it loads
if dataset == "MNIST":
    # load data: images are encoded as numpy arrays, labels are an array of digits 0-9
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
elif dataset == "Fashion":
    from keras.datasets import fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
elif dataset == "CIFAR":
    from keras.datasets import cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.reshape((50000, 32, 32, 1))
    test_images = test_images.reshape((10000, 32, 32, 1))


# look at train data
train_images.shape    # MNIST: (60000, 28, 28)
len(train_labels)     # MNIST 60000
train_labels          # MNIST array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)

# look at test data
test_images.shape     # MNIST (10000, 28, 28)
len(test_labels)      # MNIST 10000
test_labels           # MNIST array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)


#QUICK INFO ON IMAGE
def get_info(image):
    print("\n------------------------")
    print("INFO")
    print("------------------------")
    print("SHAPE:",image.shape)
    print("MIN:",image.min())
    print("MAX:",image.max())
    print("TYPE:",type(image))
    print("DTYPE:",image.dtype)
    # print(DataFrame(image))

get_info(train_images)
get_info(train_labels)


#----------------------------------------------------------------------------------   
## VISUALIZE AN IMAGE
#----------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Add function to visualize a random (or specified) image in the database
def visualize_image(image_num):
    digit = train_images[image_num]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    
visualize_image(4) # call function to visualize an image in database



# The workflow will be as follows: 
# First, we’ll feed the neural network the training data, train_images and train_labels. 
# The network will then learn to associate images and labels. 
# Finally, we’ll ask the network to produce predictions for test_images, 
# and we’ll verify whether these predictions match the labels from test_labels.

#NORMALIZE
# noteL before training, preprocessing data by reshaping into shape the network expects 
# and scaling it so that all values are in the [0, 1] interval
# note: training images previously stored in array of shape (60000, 28, 28) 
# of type uint8 with values in the [0, 255] interval
# transform it into float32 array of shape (60000, 28 * 28) w/ vals btw 0 and 1
train_images = train_images.astype('float32') / 255 
test_images = test_images.astype('float32') / 255  

#DEBUGGING
if dataset != "CIFAR":
    NKEEP=60000
    batch_size=int(0.05*NKEEP)
    epochs=10 # 20
    print("batch_size",batch_size)
    rand_indices = np.random.permutation(train_images.shape[0])
    val_images=train_images[rand_indices[48001:NKEEP],:,:] # 20% val
    val_labels=train_labels[rand_indices[48001:NKEEP]]
    train_images=train_images[rand_indices[0:48000],:,:] # 80% test
    train_labels=train_labels[rand_indices[0:48000]]
else:
    NKEEP=60000
    batch_size=int(0.05*NKEEP)
    epochs=10 # 20
    print("batch_size",batch_size)
    rand_indices = np.random.permutation(train_images.shape[0])
    val_images=train_images[rand_indices[40001:NKEEP],:,:] # 20% val
    val_labels=train_labels[rand_indices[40001:NKEEP]]
    train_images=train_images[rand_indices[0:40000],:,:] # 80% test
    train_labels=train_labels[rand_indices[0:40000]]


#----------------------------------------------------------------------------------
#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
#----------------------------------------------------------------------------------
# preparing the labels: categorically encode labels
tmp=train_labels[0]
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)
print(tmp, '-->',train_labels[0])
print("train_labels shape:", train_labels.shape)


#----------------------------------------------------------------------------------
#COMPILE AND TRAIN MODEL
#----------------------------------------------------------------------------------
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)


#----------------------------------------------------------------------------------
#EVALUATE ON TEST DATA
#----------------------------------------------------------------------------------
train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)


#----------------------------------------------------------------------------------
## DATA AUGMENTATION IF DESIRED 
#----------------------------------------------------------------------------------
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

## GENERATE NEW PICTURES VIA RANDOM DATA AUGMENTATION
if data_aug == True:
    from keras.preprocessing import image
    img = image.load_img(train_images[img_num], target_size=(150, 150))
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


# In[14]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
val_generator = train_datagen.flow(val_images, val_labels, batch_size=batch_size)


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
    
#----------------------------------------------------------------------------------    
## FIT MODEL USING A BATCH GENERATOR
#----------------------------------------------------------------------------------
# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=10, #30,
#       validation_data=validation_generator,
#       validation_steps=50) 
#       # validation_steps tells process how many batches to draw from 
#       # the validation generator for evaluation.

history=model.fit_generator(train_generator, 
                            steps_per_epoch=100, 
                            epochs=10,
                            validation_data=val_generator, 
                            validation_steps=50,
                            use_multiprocessing=True)


#----------------------------------------------------------------------------------
## DEFINE METHOD TO SAVE MODEL
#----------------------------------------------------------------------------------
def save_model(model_to_save, filename):
    model_to_save.save(filename) # save model after training
save_model(model, 'HW4_part1_model.h5')

#----------------------------------------------------------------------------------
## DEFINE METHOD TO READ MODEL FROM FILE
#----------------------------------------------------------------------------------
def get_model(filename):
    from keras.models import load_model
    model = load_model(filename) # load model saved earlier
    model.summary() # take a look at model
    
#----------------------------------------------------------------------------------
## TRAINING AND VALIUDATION ACCURACY HISTORY PLOT
#----------------------------------------------------------------------------------
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

#----------------------------------------------------------------------------------
## FUNCTION TO VISUALIZE WHAT CNN IS DOING INSIDE
## EVERY CHANNEL IN EVERY INTERMEDIATE ACTIVATION
#----------------------------------------------------------------------------------
# plots a complete visualization of all the activations in the network
# extract and plot every channel in each activation map
# stack results in one big image tensor, with channels stacked side by side
# higher activations have less info about image, more info about classification
def cnn_inside_vis(model):
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




