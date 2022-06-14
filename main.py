# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:33:19 2021

@author: co9527de

ICEEMDAN-BPNN-SE-TPE

Predicts and returns prediction accuracy for all components

"""
                
# -- coding: utf-8 --
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy import concatenate
import pandas as pd
import tensorflow as tf

import models
import process
from models import model_train,train_dec,predict_dec,model_forecast
from process import dec,pec

from model_train import create 
from model_forecast import create_tpe
from train_dec import train_dec 
from predict_dec import predict_dec
from dec import iceemdan 
from pec import pe 
 
### Try to limit GPU memory to fit ensembles on RTX 2080Ti
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=9000)])
    except RuntimeError as e:
        print(e)


data_nsw = pd.read_excel('/data/New South Wales.xlsx')#Import raw data
data_nsw.drop('TIME',axis=1)

Threshold=0.05
inputn = 48# Input Dimension


#train process
size= int((len(data_nsw)-inputn) * (0.7+0.2))#  70% of the data is used for training, 20% for validation       
data= data_nsw[:size+inputn]
data= data.values# Training set + validation set

record_fiducial, record_pe, pass_result, fail_imf = train_dec(data,inputn)

print("Done decomposing!")

# Judgment whether the requirement is met
var_imf=np.var(fail_imf, axis = 0)
var=np.var(data)
t=np.sum(var_imf) / var     #Variance share of components to be decomposed after one decomposition        
   
# Used to save results     
record_fiducial_second =[]
record_pe_second =[]
pass_result_second =[]
fail_imf_second =[]
# Secondary decomposition
for i in range(len(fail_imf)):
    record_fiducial_multi, record_pe_multi, pass_result_multi, fail_imf_multi = train_dec(fail_imf[:,i],inputn)
    record_fiducial_second=np.concatenate((record_fiducial_second,record_fiducial_multi),axis=0)
    record_pe_second=np.concatenate((record_pe_second,record_pe_multi),axis=0)
    pass_result_second=pass_result_second+pass_result_multi
    fail_imf_second=np.concatenate((fail_imf_second,fail_imf_multi),axis=1)

print("Done Secondary decomposing!")
var_imf =np.var(fail_imf_second, axis = 0)  
t=np.sum(var_imf) / var     #Variance share of components to be decomposed after Secondary decomposition

#After Secondary decomposition, the NSW dataset satisfies the condition: t<Threshold

# Third decomposition
'''
record_fiducial_third =[]
record_pe_third =[]
pass_result_third =[]
fail_imf_third =[]
for i in range(len(fail_imf_second)):
    record_fiducial_multi, record_pe_multi, pass_result_multi, fail_imf_multi = train_dec(fail_imf_second[:,i],inputn)
    record_fiducial_third=np.concatenate((record_fiducial_third,record_fiducial_multi),axis=0)
    record_pe_third=np.concatenate((record_pe_third,record_pe_multi),axis=0)
    pass_result_third=pass_result_third+pass_result_multi
    fail_imf_third=np.concatenate((fail_imf_third,fail_imf_multi),axis=1)


print("Done Third decomposing!")

var_imf =np.var(fail_imf_third, axis = 0)  
t=np.sum(var_imf) / var     #Variance share of components to be decomposed after Third decomposition

After Third decompositions, all data sets satisfy the condition: t<Threshold
'''
    
#forecast process   

def get_dec_result(data,record_pe,record_fiducial):
    
    data_dec=iceemdan(data,0.05,100,5000,2)
    width = data_dec.shape[1]#columns
    length = data_dec.shape[0]#rows
    
    
    record_pe_inv=[ 1-x for x in record_pe]
    record_fiducial_inv=[ 1-x for x in record_fiducial]
            
    pass_re=record_fiducial*record_pe # High prediction accuracy and low complexity, the value of the component retained and reconstructed is 1
    pass_re_Matrix=np.tile(pass_re, length).reshape(-1,width)
    pass_imf_re=np.sum( data_dec*pass_re_Matrix, axis=1)
    
    pass_save=record_fiducial*record_pe_inv  #High prediction accuracy and high complexity, with a retained component value of 1
    pass_save_Matrix=np.tile(pass_save, length).reshape(-1,width)
    pass_imf_save= data_dec*pass_save_Matrix
    
    pass_all=np.concatenate((pass_imf_re,pass_imf_save),axis=1) #all Retained components
    
    
    fail_re=record_fiducial_inv*record_pe #Low prediction accuracy and low complexity, the value of the component waiting to be decomposed again after performing reconstruction is 1
    fail_re_Matrix=np.tile(fail_re, length).reshape(-1,width)
    fail_imf_re=np.sum( data_dec*fail_re_Matrix, axis=1)
    
    fail=record_fiducial_inv*record_pe_inv #Low prediction accuracy and high complexity, the component waiting to be decomposed again has a value of 1
    fail_Matrix=np.tile(fail, length).reshape(-1,width)
    fail_imf= data_dec*fail_Matrix
    
    fail_all=np.concatenate((fail_imf,fail_imf_re),axis=1) #All components to be decomposed
    
    return pass_all, fail_all


  
# # Set parameters for cyclic scrolling prediction
cyc = 5 #Number of predicted cycles
test_size = len(data_nsw)- size# Number of predictions    
predictions = np.zeros((test_size,cyc)) # Storage predictions   
   

#  cyclic scrolling prediction
for j in range (cyc):
    for i in range(test_size):
        print(i)
        
        data=data_nsw[-(len(data_nsw) ):-(test_size  - i)]
        pass_all, fail_all=get_dec_result(data,record_pe,record_fiducial)
        
        pass_all_second=[]
        fail_all_second=[]
        
        for i in range(len(fail_all)):
            pass_all_multi, fail_all_multi=get_dec_result(fail_all[:,i],record_pe_second[:,i],record_fiducial_second[:,i])
            pass_all_second=np.concatenate((pass_all_second,pass_all_multi),axis=0)
            fail_all_second=np.concatenate((fail_all_second,fail_all_multi),axis=0)
            
        all_imf=np.concatenate((pass_all,pass_all_second,fail_all_second),axis=0)
        predict=predict_dec(all_imf,inputn,1)
        
        predictions[i,j] = predict
        
predictions= np.sum(predictions,axis=0)  #the average of multiple predictions
predictions = pd.DataFrame(predictions)
predictions .to_csv('NSW_result.csv')  

print("Done forecasting!")
       
    
    

    
    
    
    
    
    
    
