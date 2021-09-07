#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


filepath = "/Users/hanna/Documents/ANLY590/590-CODES/DATA/weight.json"
# filepath = "./DATA/weight.json"

f = open(filepath,)
data = json.load(f)
f.close()
df = pd.DataFrame(data)
# data = np.array(data)
# labels = np.array('is_adult')


# In[3]:


my_filepath = "/Users/hanna/Documents/ANLY590/590-CODES/DATA/weight.json"


# In[ ]:


### Defining the problem and assembling a dataset (I already did this for you weight.json)
# Make a class called “Data” which 
    # reads the json, partitions the data, and has functions (methods) to
    # visualize data (similar to lecture example)
    
class Data:
    
    filepath = "./DATA/weight.json"
    # filepath = "/Users/hanna/Documents/ANLY590/590-CODES/DATA/weight.json"
    
    def read_json_file(self, filepath): # read data
        f = open(filepath,)
        data = json.load(f)
        f.close()
        df = pd.DataFrame(data)
        # data = np.array(data)
        # labels = np.array(labels)
    
    def __init__(self):
        self.age = data[0]
        self.weight = data[1]
        self.is_adult = data[2]
        
    


# In[ ]:


### Choosing a measure of success
# For this exercise use mean square error (MSE) or sum of square error (SSE) as the loss function 


# In[6]:


plt.scatter(data= df, x='x', y='y')
plt.xlabel("age(years)")
plt.ylabel("weight(lb)")
plt.show()


# In[7]:


plt.scatter(data= df, x='y', y='is_adult')
plt.xlabel("weight(lb)")
plt.ylabel("ADULT=1 CHILD=0")
plt.show()


# In[40]:


### Preparing your data
# Normalize inputs (features) and outputs as needed using “standard scalar”
# X_i --> X~_i = (x_i - mu_x)/sigma_x
df['x_norm']=(df['x']-df['x'].mean())/df['x'].std()
df['y_norm']=(df['y']-df['y'].mean())/df['y'].std()


# In[41]:


### Deciding on an evaluation protocol
# For this exercise just break the data into 80% training 20% test
from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
train_df, test_df = train_test_split(df,test_size=0.2)


# In[13]:


### Developing a model
# Linear regression (fit to age<18)
# reference: https://medium.com/geekculture/linear-regression-from-scratch-in-python-without-scikit-learn-a06efe5dedb6

df_child = df[df['x']<18]
mean_x=df_child['x'].mean()
mean_y=df_child['y'].mean()

m = len(df_child['x']) # total no.of input values

# using the formula to calculate m & c ... for when x (age) is under 18
numer = 0
denom = 0
for i in range(m):
    numer += (df_child['x'][i] - mean_x) * (df_child['y'][i] - mean_y)
    denom += (df_child['x'][i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)

print (f'm = {m} \nc = {c}')


# In[27]:


# calculating line values x and y
x = np.linspace (0, 100, 100)
y = c + m * x

plt.plot(x, y,c="red",label='Linear Model')
plt.scatter(train_df['x'], train_df['y'], label='train set')
plt.scatter(test_df['x'], test_df['y'], c="orange", label='test set')

plt.xlabel('age(years)')
plt.ylabel('weight(lb)')
plt.legend()
plt.ylim(0,250)
plt.show()


# In[26]:


# calculating line values x and y
x = np.linspace (0, 100, 100)
y = c + m * x

plt.plot(x, y,c="red",label='Linear Model')
plt.scatter(train_df['x'], train_df['y'], label='train set')
plt.scatter(test_df['x'], test_df['y'], c="orange", label='test set')

plt.xlabel('age(years)')
plt.ylabel('weight(lb)')
plt.legend()
plt.ylim(0,180)
plt.xlim(0,18)
plt.show()


# In[28]:


# -----------------------------------------------------------------------------
# # For example, for the case of linear regression, please do the following

# # 1) Define a model function that takes a vector x and vector p. 
# #        M(x,p) where p=[b,m]=[p[0],p[1]] and returns p[0]+p[1]*x
# #        NFIT=2

# def Mod1(x, p):
#     p=[b,m]=[p[0],p[1]]
#     return p[0]+p[1]*x

# NFIT=2

# # 2) Define an objective (loss) function that takes only a vector of 
# #    model parameters p and returns the loss (mean square error MSE)
# #        def loss(p):
# #             yp=M(x,p) 
# #             ..... 
# #             loss=MSE
# #             return loss
# #       -This should be general, p could be m,b for linear regression, 
# #        or 4 parameters for logistic regression. 

# def loss(p):
#     yp = Mod1(x, p)
#     MSE = np.square(np.subtract(Y_true,Y_pred)).mean()
#     # or # mse = np.sum((y_pred - y_actual)**2)
#     loss = MSE
#     return loss

# # 3) use scipy optimizer to minimize the loss function and obtain the optimal m,b parameters 
# #  #RANDOM INITIAL GUESS FOR FITTING PARAMETERS
# po=np.random.uniform(0.5,1.,size=NFIT)
# #  #TRAIN MODEL USING SCIPY OPTIMIZER
# from scipy.optimize import minimize
# # res = minimize(loss, po, method=OPT_ALGO, tol=1e-15)
# res = minimize(loss, po, method="Nelder-Mead", tol=1e-15)
# popt=res.x
# print("OPTIMAL PARAM:",popt)
# -----------------------------------------------------------------------------


# In[30]:


# B) To visualize training/test loss as a function of the optimizer iterations, 
# you can simply define 3 global arrays that update every time SciPy evaluates 
# the loss function. And then plot these at the end 

#SAVE HISTORY FOR PLOTTING AT THE END
iterations=[]; loss_train=[];  loss_val=[]

iteration=0
def loss(p):
    global iteration,iterations,loss_train,loss_val
    loss_train.append(training_loss)
    loss_val.append(validation_loss)
    iterations.append(iteration)
    iteration+=1


# In[31]:


# C) Since you have to train 3 models, you might want to consider creating 3 python scripts, 
# one for each model. It is possible to do it all in one python file but it requires more scripting. 
# You could get the entire process dialed in and working for linear regression, 
# then cp this file to a new one, modify which data it uses, and change the model to output the 
# logistic (sigmoid) function instead of a linear function. Or combine both and have a flag 
# to specify which you want to use. 

def model(x,p):
    global model_type
    if(model_type=="linear"):
        return p[0]*x+p[1]  
    if(model_type=="logistic"):
        return p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))


# In[35]:


# D) To undo the normalization simply rearrange the normalization formula
# Assume that you normalize both input and output data 

# The model makes predictions in the x' y' space 
# (i.e. for a given x' the model outputs M(x'|P)=ypred' (where ypred' is in y' space)) 
# Then convert back to the original space by inverting the normalization formula 
df['x_undo_norm'] = (df['x_norm']*df['x'].std()) + df['x'].mean() # should be same as original 'x' column preserved
# df['y_pred_undo_norm'] = (df['y_norm']*df['y_pred'].std()) + df['y'].mean()


# In[42]:


train_df.head()


# In[45]:


# Logistic Regression: weight
# reference: https://financial-engineering.medium.com/logistic-regression-without-sklearn-107e9ea9a9b6
def sigmoid(x): # activation function
    return 1/(1+np.exp(-x))

def sq_loss(y_pred, target): # loss function
    return np.mean(pow((y_pred-target),1))

X_tr, y_tr = train_df.x, train_df['is_adult']
X_te, y_te = test_df.x, test_df['is_adult']


# In[49]:


lr = 0.01
W = np.random.uniform(0,1)
b=0.1

for i in range(1000):
    z = np.dot(X_tr, W) + b

y_pred = sigmoid(z)

l = sq_loss(y_pred, y_tr)

gradient_W = np.dot((y_pred - y_tr).T, X_tr)/X_tr.shape[0]
gradient_b = np.mean(y_pred-y_tr)

W = W-lr * gradient_W
b = b - lr * gradient_b


# In[108]:


# Logistic Regression: adult/child
from   scipy.optimize import curve_fit

x = X_tr
yn = y_tr

##FITTING MODEL
def model(x,p1,p2,p3,p4,p5,p6,p7):
    return p1*np.exp(-((x-p2)/p3)**2.0)+p4*np.exp(-((x-p5)/p6)**2.0)+p7
popt, pcov = curve_fit(model, x, yn) #,[0.1,0.1,0.1])

fig, ax = plt.subplots()
ax.plot(train_df['x'], train_df['y'], 'o', label='train')
ax.plot(test_df['x'], test_df['y'], 'o', label='test')
ax.plot(X_tr, y_pred, 'o', label='pred')

ax.plot(x, model(x, *popt), 'r-', label="Model")

plt.show()


# In[69]:


z[-1]


# In[72]:


# calculating line values x and y
plt.plot(y,c="red",label='Linear Model')
plt.scatter(train_df['x'], train_df['y'], label='train set')
plt.scatter(test_df['x'], test_df['y'], c="orange", label='test set')

plt.xlabel('age(years)')
plt.ylabel('weight(lb)')
plt.legend()
plt.ylim(0,250)
plt.show()


# In[ ]:


# After training (see next slide), Generate plots similar to the following
# Use SciPy optimizer to train the parameters (not SciPy curve fit)
#      (similar to in D1.1.2-SciPy-SINGLE-VARIABLE-OPTIMIZER.py)
# Train 3 models, one for each plot in the previous slide
# Visualize your results
#    make at least three plots similar to those shown on the last slide

# skip the last 2 steps ... these only apply for complex models such as ANN
#    as well as a parity plot that plots the model predictions yp as a function of data y
#    Unnormalized the model output so it can predict weight(age) to predict age(weight) rather than the
#      normalized output (see below)

