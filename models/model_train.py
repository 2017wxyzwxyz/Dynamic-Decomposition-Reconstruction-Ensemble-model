# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:33:19 2021

@author: co9527de

train model 

"""
# -- coding: utf-8 --

from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential

def create(input_shape,nodes, Dropout_value):
    
    timesteps=input_shape
    #Sequential model 
    model = Sequential()
    model.add(Dense(nodes, input_shape=(timesteps,) ))
    model.add(Dense(nodes))
    model.add(Dropout(Dropout_value))
    model.add(Dense(nodes))
    model.add(Dense(1))
    model.summary()
    
    return model
    
    
    
    