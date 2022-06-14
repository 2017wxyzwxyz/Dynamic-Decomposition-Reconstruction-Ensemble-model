# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:33:19 2021

@author: co9527de

forecasting model 

"""
# -- coding: utf-8 --
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy import concatenate
import pandas as pd


from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers 

from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import r2_score#R 2
from sklearn.metrics import explained_variance_score#EVC
from sklearn.preprocessing import MinMaxScaler

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def create_tpe(input_shape,X_train , y_train , X_test, y_test):
    
    #Declare a search space
    ''' 需要优化的参数
    activation
    Dropout
    lr
    hidden_unit
    '''
    a_hidden_unit= list(range(5,50+1))
    space = {
    'activation': hp.choice('activation', ['tanh','relu','sigmoid','linear']),
    'hidden_unit': hp.choice('hidden_unit', a_hidden_unit),
    'Dropout': hp.uniform('Dropout', 0.0,0.5),
    'lr': hp.uniform('lr', 0.0001,0.1)
    }
    
    #Functions to be minimized
   
    acc = 0
    timesteps =input_shape
    
    def f(params):
        global X_train , y_train , X_test, y_test
        global acc
        print('Params testing: ', params)
        model = Sequential()
        model.add(Dense(params['hidden_unit'], input_shape=(timesteps, ), activation= params['activation']))
        model.add(Dense(params['hidden_unit'],activation= params['activation']))
        model.add(Dropout(params['Dropout']))
        model.add(Dense(params['hidden_unit'],activation= params['activation']))
        model.add(Dense(1))
        model.summary()
          
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=params['lr']),metrics=['acc'])
    
        # early stoppping
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
        model.fit(X_train, y_train,
                            epochs=200, batch_size=128,
                            validation_split=0.1,
                            verbose=2, shuffle=False, callbacks=[early_stopping])
        
        y_pred = model.predict(X_test)
        test_acc = explained_variance_score(y_test, y_pred)
        if test_acc > acc:
            acc = test_acc
           
        print(acc)
        return {
            'loss': -test_acc,
            'status': STATUS_OK
        }
    
    trials = Trials()
    best = fmin(f, space, algo=tpe.suggest, max_evals=50, trials=trials)
    print('best')
    print(best) 
    s1 = pd.Series(best)
    s=s1.values # a numpy array    
    
    #Hyperparameter extraction  
    
    activation_set = ['tanh','relu','sigmoid','linear']
    hidden_unit_set = list(range(5,50+1))
    
    Dropout_bys = s[0]
    
    activation_bys = s[1]
    activation_bys = activation_bys.astype(int)
    activation_bys = activation_set[activation_bys]
       
    
    hidden_unit_bys = s[2]
    hidden_unit_bys = hidden_unit_bys.astype(int)
    hidden_unit_bys =  hidden_unit_set[hidden_unit_bys]
    
    lr_bys = s[3]
    
    #Sequential model
    model = Sequential()
    model.add(Dense(hidden_unit_bys, input_shape=(timesteps, ),
                      activation=activation_bys ))
    model.add(Dense(hidden_unit_bys, activation=activation_bys ))
    model.add(Dropout(Dropout_bys))   
    model.add(Dense(hidden_unit_bys, activation=activation_bys ))         
    model.add(Dense(1))       
    model.summary()    

    
    return model,best
    
    
    