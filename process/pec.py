# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:33:19 2021

@author: co9527de

permuation entropy

"""

import matlab
import matlab.engine
eng = matlab.engine.start_matlab()


def pe(I,m,t):
    
    '''
    I: time series;
    m: order of permuation entropy
    t: delay time of permuation entropy
    '''
    
    pe, hist = eng.pec(I,m,t)
    #pe:    permuation entropy
    #hist:  the histogram for the order distribution
    return pe

