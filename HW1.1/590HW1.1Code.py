#!/usr/bin/env python
# coding: utf-8

# Hanna Born,
# ANLY 590 HW1.1
# 
# References:  code files provided on class github at https://github.com/jh2343/590-CODES
# 
# I struggled with the assignment initially using OOP and class structure. My submission below reflects reworking through the regression workflow according to the basic structure code example provided on the class github which helped my understanding of this task significantly.

# In[31]:


# imports
import json
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

filepath = '/Users/hanna/Documents/ANLY590/590-CODES/DATA/weight.json'
# filepath='weight.json'

# read file into dictionary
f = open(filepath,)
data = json.load(f)
f.close()

# ---- MODEL SELECTION ---------------------------------------------------------
# choose model type by uncommenting
# model_type="linear";   NFIT=2; xcol=1; ycol=2;       # age vs. weight (linear)
model_type="logistic"; NFIT=4; xcol=1; ycol=2;       # age vs weight (logistic)
# model_type="logistic";   NFIT=4; xcol=2; ycol=0;     # weight vs. adult/child
# ------------------------------------------------------------------------------

# define a function to calculate mse for calculating test/validation losses
def get_mse(pred, actu): 
    return (np.mean((pred-actu)**2.0)) 


X=[]; # initialize matrix to make from input
for key in data.keys():  # loop through dictionary to select desired keys
    if(key in ['x','is_adult','y']): X.append(data[key])
X=np.transpose(np.array(X)) # rows = sample_dimension

# select training columns corresponding to model type selected above
# required columns for each model specified in code line for model selection above
x=X[:,xcol]
y=X[:,ycol]

if(model_type=="linear"): # linear model fit only to children (age < 18)
    y=y[x[:]<18]
    x=x[x[:]<18] 

    
# normalize
# this helps with selecting initial guess since we know the scale
x=(x-np.mean(x))/np.std(x)
y=(y-np.mean(y))/np.std(y) 


# partition into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# models
def model(x,p):
    if(model_type=="linear"):
        return  p[0]*x+p[1]  
    if(model_type=="logistic"):
        return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

# track and define globally for use in plotting
iteration=0; iterations=[]; loss_train=[];  loss_val=[] 

def loss(p): # loss function
    global iterations,loss_train,loss_val,iteration

    # training loss (MSE)
    yp=model(x_train,p) # model predictions for given parameterization p
    training_loss = get_mse(yp, y_train)

    # validation loss (MSE)
    yp=model(x_test,p) #model predictions for given parameterization p
    validation_loss= get_mse(yp, y_test)

    # look at training and validation loss for every 50th iteration
    if(iteration==0):
        print("iteration \t training_loss \t validation_loss") 
    if(iteration%50==0):
        print(iteration,"\t",training_loss,"\t",validation_loss) 
    
    #RECORD FOR PLOTING
    loss_train.append(training_loss)
    loss_val.append(validation_loss)
    iterations.append(iteration)
    iteration+=1

    return training_loss

initial_guess=np.random.uniform(0.1,1.,size=NFIT) # initial random guess

# use minimize to train model and print out optimal parameters
res = minimize(loss, initial_guess, method='BFGS', tol=1e-15)
popt=res.x
print("optimal parameters:",popt)

# predictions
xm=np.array(sorted(x_train))
yp=np.array(model(xm,popt))

# reverse the normalization to have predictions in real scale
def unnorm_x(x): 
    return np.std(x)*x+np.mean(x)  
def unnorm_y(y): 
    return np.std(y)*y+np.mean(y) 

if(True): # function plots of real scale (un-normalized)
    fig, ax = plt.subplots()
    ax.plot(unnorm_x(x_train), unnorm_y(y_train), 'o', label='train')
    ax.plot(unnorm_x(x_test), unnorm_y(y_test), 'x', label='test')
    ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Function Plot")
    plt.show()

if(True): # parity plot y actual predicted
    fig, ax = plt.subplots()
    ax.plot(model(x_train,popt), y_train, 'o', label='train')
    ax.plot(model(x_test,popt), y_test, 'x', label='test')
    plt.xlabel('y predicted')
    plt.ylabel('y actual')
    plt.legend()
    plt.title("Parity Plot")
    plt.show()

if(True): # training and validation loss through iterations
    fig, ax = plt.subplots()
    ax.plot(iterations, loss_train, '.', label='training loss')
    ax.plot(iterations, loss_val, '.', label='validation loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.title("Losses")
    plt.show()


# In[ ]:




