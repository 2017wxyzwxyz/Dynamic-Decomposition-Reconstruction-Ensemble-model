# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:33:19 2021

@author: co9527de

decompose the input by ICEEMDAN

"""

import matlab
import matlab.engine
eng = matlab.engine.start_matlab()


def iceemdan(I,Nstd,NR,MaxIter,SNRFlag):
    '''
    #I: signal to decompose
    #Nstd: noise standard deviation
    #NR: number of realizations
    #MaxIter: maximum number of sifting iterations allowed.
    #SNRFlag: if equals 1, then the SNR increases for every stage, as in [1].
    #           If equals 2, then the SNR is the same for all stages, as in [2].
    
    #[1] Colominas MA, Schlotthauer G, Torres ME. "Improve complete ensemble EMD: A suitable tool for biomedical signal processing" 
    #       Biomedical Signal Processing and Control vol. 14 pp. 19-29 (2014)
    
    #The CEEMDAN algorithm was first introduced at ICASSP 2011, Prague, Czech Republic
    
    #The authors will be thankful if the users of this code reference the work
    #where the algorithm was first presented:
    
    #[2] Torres ME, Colominas MA, Schlotthauer G, Flandrin P. "A Complete Ensemble Empirical Mode Decomposition with Adaptive Noise"
    #       Proc. 36th Int. Conf. on Acoustics, Speech and Signa Processing ICASSP 2011 (May 22-27, Prague, Czech Republic)
    '''
    
    dec, its = eng.ceemdan(x,Nstd,NR,MaxIter,SNRFlag)
    #dec: contain the obtained modes in a matrix with the rows being the modes  
    #its: contain the sifting iterations needed for each mode for each realization (one row for each realization)
    return dec

