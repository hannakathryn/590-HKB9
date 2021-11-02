#!/usr/bin/env python
# coding: utf-8

# Hanna Born
# ANLY 590 Fall 2021
# reference: course code at https://github.com/jh2343/590-CODES

# In[]:


import warnings
warnings.filterwarnings('ignore')

from urllib.request import urlopen
import numpy as np




# In[]:


# url locations for three novels from Project Gutenberg
url_frankenstein = "https://gutenberg.org/files/84/84-0.txt"
url_wotw = "https://gutenberg.org/files/36/36-0.txt" # war of the worlds
url_greatExpectations = "https://gutenberg.org/files/1400/1400-0.txt"


# In[]:


# read in only a small sample of the data for code development
# n_chars = 100000  # for testing

# data_frank = urlopen(url_frankenstein).read(n_chars) # read in only up to n_chars
# data_frank = data_frank.decode('utf-8') 
# data_frank = data_frank.split('\n') # split it into lines

# data_wotw = urlopen(url_wotw).read(n_chars) # read in only up to n_chars
# data_wotw = data_wotw.decode('utf-8') 
# data_wotw = data_wotw.split('\n') # split it into lines

# data_greatExpectations = urlopen(url_greatExpectations).read(n_chars) # read in only up to n_chars
# data_greatExpectations = data_greatExpectations.decode('utf-8') 
# data_greatExpectations = data_greatExpectations.split('\n') # split it into lines


# In[]:


# read in full size data (full novel text)
# split into chunks based on newlines
data_frank = urlopen(url_frankenstein).read()
data_frank = data_frank.decode('utf-8') 
data_frank = data_frank.split('\n') # split it into lines

data_wotw = urlopen(url_wotw).read() # read in only up to n_chars
data_wotw = data_wotw.decode('utf-8') 
data_wotw = data_wotw.split('\n') # split it into lines

data_greatExpectations = urlopen(url_greatExpectations).read() # read in only up to n_chars
data_greatExpectations = data_greatExpectations.decode('utf-8') 
data_greatExpectations = data_greatExpectations.split('\n') # split it into lines


# In[]:


# define function to map words to word indexes
def form_dictionary(samples):
    token_index = {};  
    #FORM DICTIONARY WITH WORD INDICE MAPPINGS
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    transformed_text=list()
    for sample in samples:
        tmp=list()
        for word in sample.split():
            tmp.append(token_index[word])
        transformed_text.append(tmp)
        
    return [token_index,transformed_text]

[vocab_frank,x_frank]=form_dictionary(data_frank)
[vocab_wotw,x_wotw]=form_dictionary(data_wotw)
[vocab_greatExpectations,x_greatExpectations]=form_dictionary(data_greatExpectations)


# # look at tokenized samples from each novel
# print(x_frank[95])
# print(x_wotw[95])
# print(x_greatExpectations[95])


# In[5]:


# full dataset with index mappings from all three novels
x = x_frank + x_wotw + x_greatExpectations


# In[6]:

# create an array of lists for index mapping from all three novels
y=np.array(x, dtype='object')



# In[8]:


# y_labels_frank = [list([1,0,0]) for i in range(len(x_frank))]
# y_labels_wotw = [list([0,1,0]) for i in range(len(x_wotw))]
# y_labels_greatExpectations = [list([0,0,1]) for i in range(len(x_greatExpectations))]

# create labels to identify which novel each text index entry comes from
y_labels_frank = [0] * len(x_frank)
y_labels_wotw = [1] * len(x_wotw)
y_labels_greatExpectations = [2] * len(x_greatExpectations)
y_labels_all = y_labels_frank + y_labels_wotw + y_labels_greatExpectations
y_labels = np.array(y_labels_all)
y_labels


# In[9]:


# check that each input has a label
len(y) == len(y_labels)


# In[10]:


# save small pre-processed datasets so don't have to pre-process every time to train
from numpy import save

# save pre-processed data and labels as numpy files
save('data.npy', y)
save('labels.npy', y_labels)


# In[ ]:


# BELOW: save small/full train/test datasets independently


# from sklearn.model_selection import train_test_split

# data_train, data_test, labels_train, labels_test = train_test_split(y, y_labels, 
#                                                                     test_size=0.20,
#                                                                     random_state=42)



# In[ ]:


# # save small pre-processed datasets so don't have to pre-process every time to train
# from numpy import save

# # train
# save('small_data_train.npy', data_train)
# save('small_labels_train.npy', labels_train)

# # test
# save('small_data_test.npy', data_test)
# save('small_labels_test.npy', labels_test)


# In[ ]:


# # save full pre-processed data
# from numpy import save

# # train
# save('data_train.npy', data_train)
# save('labels_train.npy', labels_train)

# # test
# save('data_test.npy', data_test)
# save('labels_test.npy', labels_test)


