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
from numpy import save


# In[3]:


# conda install -c conda-forge wikipedia
# conda install -c conda-forge wordcloud
# pip install wikipedia_sections

import wikipedia

# see https://meta.wikimedia.org/wiki/List_of_Wikipedias
# for langages prefixes
# wikipedia.set_lang('es') #es=spanish en=english 

wikipedia.set_lang('en') # set language to english


## ------------------------------------
## GET DATA
## ------------------------------------

# Getting content from three wikipedia pages about 
# ---- FIS World Cup (Alpine Skiing)
# ---- FIFA World Cup (Soccer)
# ---- and Olympic Games
# to see if NN can properly classify which page the text is from


# ['FIS Alpine Ski World Cup']
# ['FIFA World Cup']
# ['Olympic Games']


fis = wikipedia.page(wikipedia.search("Skiing World Cup", results = 1)).content  # FIS Alpine Ski World Cup
fifa = wikipedia.page(wikipedia.search("Soccer World Cup", results = 1)).content # FIFA World Cup 
olympics = wikipedia.page(wikipedia.search("Olympics", results = 1)).content     # Olympic Games


# In[4]:


## -------------------------------------
## PREPROCESS
## -------------------------------------
# fis = fis.split('.') # split it into lines
# fifa = fifa.split('.') # split it into lines
# olympics = olympics.split('.') # split it into lines


# define function to map words to word indexes
def form_dictionary(samples):
    token_index = {};  
    #FORM DICTIONARY WITH WORD INDICE MAPPINGS
    for sample in samples:
        for word in sample.split(' '):
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    transformed_text=list()
    for sample in samples:
        tmp=list()
        for word in sample.split(' '):
            tmp.append(token_index[word])
        transformed_text.append(tmp)
        
    return [token_index,transformed_text]


# In[5]:


[vocab_fis,x_fis]=form_dictionary(fis)
[vocab_fifs,x_fifa]=form_dictionary(fifa)
[vocab_olympics,x_olympics]=form_dictionary(olympics)


# In[7]:


## -------------------------------------
## CREATE LABELED DATASET (text, topic)
## -------------------------------------

# full dataset with index mappings from all three wikipedia pages
x = x_fis + x_fifa + x_olympics
# create an array of lists for index mapping from all three wikipedia pages
y = np.array(x, dtype='object')

# create labels to identify which novel each text index entry comes from
y_labels_fis = [0] * len(x_fis)
y_labels_fifa = [1] * len(x_fifa)
y_labels_olympics = [2] * len(x_olympics)
y_labels_all = y_labels_fis + y_labels_fifa + y_labels_olympics
y_labels = np.array(y_labels_all)
y_labels


# check that each input has a label
len(y) == len(y_labels)


# In[8]:


## -------------------------------------
## SAVE PRE-PROCESSED DATA
## -------------------------------------
# save small pre-processed datasets so don't have to pre-process every time to train
# save pre-processed data and labels as numpy files
save('data.npy', y)
save('labels.npy', y_labels)


# In[ ]:




