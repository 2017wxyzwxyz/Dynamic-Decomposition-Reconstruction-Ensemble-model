# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:33:19 2021

@author: co9527de

ICEEMDAN-BPNN-SE-TPE

Predicts and returns prediction accuracy for all components

"""

# -- coding: utf-8 --
import numpy as np
from numpy import concatenate
import pandas as pd
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers 

from model_train import create_tpe 

def predict_dec(data,inputn,steps):
   
    width = data.shape[1]
    length = data.shape[0]   
    imf = 0
    result =np.zeros((steps,width))##storeimf prediction components
    
    for imf in range(width):
        data_imf = data[:,imf]
        data_imf = data_imf.reshape(-1,1)
        #  Normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data_imf)
        dk = scaled  
        # Preparation of input and output data
        x, y = [] * 2, [] * 2
        for i in range(len(dk) - inputn):
            x.append(dk[i:i + inputn])
            y.append(dk[i + inputn])
        x = np.array(x)
        y = np.array(y)
           
       # Divide the training set and test set, 90% as training set and validation, overall 7:2:1
        train_size= len(x)-steps
            
        X_train = x[:train_size]#training set 
        y_train = y[:train_size]
        
        X_test = x[train_size:len(x)] #validation set
        y_test = y[train_size:len(x)]
           
        X_train = X_train[:,:,0]  
        y_train = y_train[:,0]
        X_test = X_test[:,:,0]
        y_test = y_test[:,0]
    
        model,best=create_tpe(inputn,X_train , y_train , X_test, y_test)
        # Using the optimization algorithm ADAM and the optimized minimum mean square error loss function
        model.compile(loss='mean_squared_error', optimizer=Adam())
        
        # early stoppping
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
        
        train_modle = model.fit(X_train, y_train,
                                epochs=200, batch_size=24,
                                verbose=2, shuffle=False, callbacks=[early_stopping])
        # Prediction of test data X_test
        y_pred = model.predict(X_test)
        # Inverse normalization of predicted values
        inv_ypre = scaler.inverse_transform(y_pred)  
        
        inv_ypre = inv_ypre[:, 0]  
        result[:, imf] = inv_ypre
        
        
    # components filtering
    predict= np.sum(result, axis=1) #The sum of the terms is the final prediction
    
    return predict

    
    
    
    
    
    
    
    
