# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:33:19 2021

@author: co9527de

Train the model and classify the decomposed components

#High prediction accuracy with low complexity, retained and reconfigured
#High prediction accuracy with high complexity, retention
#Low prediction accuracy and low complexity, refactoring and waiting to be decomposed again
#Low prediction accuracy and high complexity, waiting to be decomposed again
"""

# -- coding: utf-8 --
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy import concatenate
import pandas as pd


from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import r2_score#R 2
from sklearn.metrics import explained_variance_score#EVC
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers 

from models.model_train import create
from dec import iceemdan 
from pec import pe 

def train_dec(data,inputn):

    data_dec=iceemdan(data,0.05,100,5000,2)
    width = data_dec.shape[1]
    length = data_dec.shape[0]
    
    imf = 0
    result =np.zeros((length-int(length/0.9 * 0.7),width))#存储imf预测分量
    mae_all=np.zeros(width)
    pe_imf=np.zeros(width)
    
    for imf in range(width):
        data_imf = data_dec[:,imf]
        data_imf = data_imf.reshape(-1,1)
        
        pe_imf[imf]=pe(data_imf,4,2)#permuation entropy
        
        #  Normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data_imf)
        dk = scaled  
        #  Preparation of input and output data
        x, y = [] * 2, [] * 2
        for i in range(len(dk) - inputn):
            x.append(dk[i:i + inputn])
            y.append(dk[i + inputn])
        x = np.array(x)
        y = np.array(y)
           
       # Divide the training set and test set, 90% as training set and validation, overall 7:2:1
        train_size= int(len(x)/0.9 * 0.7)
        test_size= len(x)-train_size
    
        
        X_train = x[:train_size]#training set 
        y_train = y[:train_size]
        
        X_test = x[train_size:len(x)] #validation set
        y_test = y[train_size:len(x)]
           
        X_train = X_train[:,:,0]  
        y_train = y_train[:,0]
        X_test = X_test[:,:,0]
        y_test = y_test[:,0]
    
        model=create(inputn,10, 0.3)
        # Using the optimization algorithm ADAM and the optimized minimum mean square error loss function
        model.compile(loss='mean_squared_error', optimizer=Adam())
        
        # early stoppping
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
        
        train_modle = model.fit(X_train, y_train,
                                epochs=200, batch_size=24,
                                validation_data=(X_test, y_test),
                                verbose=2, shuffle=False, callbacks=[early_stopping])
        #  Prediction of test data X_test
        y_pred = model.predict(X_test)
        MAE = mean_absolute_error(y_test,y_pred)#imf的MAE
        mae_all[imf] =  MAE 
        # Inverse normalization of predicted values
        inv_ypre = scaler.inverse_transform(y_pred)  
        
        inv_ypre = inv_ypre[:, 0]  
        result[:, imf] = inv_ypre
        
        
    # components filtering
    predict= np.sum(result, axis=1) #The sum of the terms is the final prediction
    
    
    #Normalization
    dy_nor = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    inv_y_nor =dy_nor[inputn+train_size:inputn+train_size]#Normalized values of the real data corresponding to the validation set
    pre_nor = (predict - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))#Normalized value of the data corresponding to the validation of the prediction
    
    fiducial_value_all = mean_absolute_error(inv_y_nor,pre_nor)#MAE
    fiducial_value_based=fiducial_value_all   
    
    pass_imf= np.zeros(shape=(length,width))
    fail_imf= np.zeros(shape=(length,width))
    pass_result=np.zeros(shape=(test_size,width))
           
    imf =1
    record_fiducial=np.zeros(width)
    record_pe=np.zeros(width)
    
    while imf < width:
        a= mae_all[imf]
        b= pe_imf[imf]
        if (a < fiducial_value_based and b < 0.5): #High prediction accuracy with low complexity, retained and reconfigured
            record_fiducial[imf]=1
            record_pe[imf]=1
            pass_imf[:,imf] = data_dec[:,imf]
            pass_result[:,imf] = result[:,imf]
        if (a < fiducial_value_based and b > 0.5): # High prediction accuracy with high complexity, retention
            record_fiducial[imf]=1
            record_pe[imf]=0
            pass_imf[:,imf] = data_dec[:,imf]
            pass_result[:,imf] = result[:,imf]
        if (a > fiducial_value_based and b < 0.5): #Low prediction accuracy and low complexity, refactoring and waiting to be decomposed again
            record_fiducial[imf]=0
            record_pe[imf]=1
            fail_imf[:,imf] = data_dec[:,imf]
        else:                                          #Low prediction accuracy and high complexity, waiting to be decomposed again
            fail_imf[:,imf] = data_dec[:,imf]
        imf+=1
    
    pass_result=np.sum(pass_result, axis=1) # Retained components
        
    record_pe_Matrix=np.tile(record_pe, length).reshape(-1,width)      
            
    fail_imf_re=np.sum(fail_imf*record_pe_Matrix, axis=1) #components to be decomposed
    
    fail_imf=fail_imf-fail_imf*record_pe_Matrix # Remove the Reconstruction components 
    #Delete columns with all 0 elements      
    idx = np.argwhere(np.all(fail_imf[..., :] == 0, axis=0))
    fail_imf = np.delete(fail_imf, idx, axis=1)        
            
    fail_imf= np.concatenate((fail_imf,fail_imf_re),axis=1) # All components to be decomposed
    
    return record_fiducial, record_pe, pass_result, fail_imf
            
    
    
    
    
    
    
    
    
    
