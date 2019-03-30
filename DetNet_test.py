# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:42:14 2019

@author: zfzdr
"""
#***********************************************************************
#***********************************************************************

import tensorflow as tf
import time as tm
import datetime

#自定义模块
from model import *
from settings import Settings

# Define the parameter of the mode
DetNet_settings = Settings()
Nr = DetNet_settings.Nr    #number of receive antennas
Nt = DetNet_settings.Nt     #number of transmit antenns
mod_Name = DetNet_settings.mod_Name
batchsize_test = DetNet_settings.batchsize_test
maxEpochs_test = DetNet_settings.maxEpochs_test
M = DetNet_settings.M
SNR_st = DetNet_settings.SNR_st
SNR_ed = DetNet_settings.SNR_ed


numbits_perSymbol = int(np.log2(M))
source_length = batchsize_test * maxEpochs_test * Nt * numbits_perSymbol

loc1 = os.getcwd() + '/model_save_10/DetNetmodel.cpkt'
loc2 = os.getcwd() + '/model_save_20/DetNetmodel.cpkt'
loc3 = os.getcwd() + '/model_save_30/DetNetmodel.cpkt'
loc4 = os.getcwd() + '/model_save_100/DetNetmodel.cpkt'

predict1 = Predict(loc1)
predict2 = Predict(loc2)
predict3 = Predict(loc3)
predict4 = Predict(loc4)

time_ZF = 0
time_DetNet_10 = 0
time_DetNet_20 = 0
time_DetNet_30 = 0
time_DetNet_100 = 0

N = SNR_ed - SNR_st 
ber_ZF = np.zeros(N)
ber_DetNet_10 = np.zeros(N)
ber_DetNet_20 = np.zeros(N)
ber_DetNet_30 = np.zeros(N)
ber_DetNet_100 = np.zeros(N)

time_st = tm.time()
for SNR in np.arange(SNR_st, SNR_ed, 1):
    now_time = datetime.datetime.now().strftime('%X')
    print(SNR, now_time)
    
    biterrorsum_ZF = 0
    biterrorsum_DetNet_10 = 0
    biterrorsum_DetNet_20 = 0
    biterrorsum_DetNet_30 = 0  
    biterrorsum_DetNet_100 = 0  
    for i in range(maxEpochs_test):
        source_test, x_test, y_test, H_test = generate_testdata(
            batchsize_test, Nt, Nr, mod_Name, SNR)

        time1 = tm.time()
        x_ZF = predict1.ZF_detector(y_test, H_test)
        time2 = tm.time()
        time_ZF = time_ZF + time2 - time1
        
        time1 = tm.time()
        x_DetNet_10 = predict1.DetNet_detector(y_test, H_test)
        time2 = tm.time()
        time_DetNet_10 = time_DetNet_10 + time2 - time1
        
        time1 = tm.time()
        x_DetNet_20 = predict2.DetNet_detector(y_test, H_test)
        time2 = tm.time()
        time_DetNet_20 = time_DetNet_20 + time2 - time1
        
        time1 = tm.time()
        x_DetNet_30 = predict3.DetNet_detector(y_test, H_test)
        time2 = tm.time()
        time_DetNet_30 = time_DetNet_30 + time2 - time1
        
        time1 = tm.time()
        x_DetNet_100 = predict4.DetNet_detector(y_test, H_test)
        time2 = tm.time()
        time_DetNet_100 = time_DetNet_100 + time2 - time1
        
        
        biterrorsum_ZF = biterrorsum_ZF + bit_error(
            source_test, x_ZF, Nt, batchsize_test, mod_Name)
        biterrorsum_DetNet_10 = biterrorsum_DetNet_10 + bit_error(
            source_test, x_DetNet_10, Nt, batchsize_test, mod_Name)
        biterrorsum_DetNet_20 = biterrorsum_DetNet_20 + bit_error(
            source_test, x_DetNet_20, Nt, batchsize_test, mod_Name)
        biterrorsum_DetNet_30 = biterrorsum_DetNet_30 + bit_error(
            source_test, x_DetNet_30, Nt, batchsize_test, mod_Name)
        biterrorsum_DetNet_100 = biterrorsum_DetNet_100 + bit_error(
            source_test, x_DetNet_100, Nt, batchsize_test, mod_Name)
        
        
    #计算不同误码率下ZF检测和DetNet检测的性能   
    ber_ZF[SNR-SNR_st]  = biterrorsum_ZF / source_length   
    ber_DetNet_10[SNR-SNR_st] = biterrorsum_DetNet_10 / source_length
    ber_DetNet_20[SNR-SNR_st] = biterrorsum_DetNet_20 / source_length
    ber_DetNet_30[SNR-SNR_st] = biterrorsum_DetNet_30 / source_length
    ber_DetNet_100[SNR-SNR_st] = biterrorsum_DetNet_100 / source_length
     
time_ed = tm.time()

print('ber_ZF:', ber_ZF)
print('ber_DetNet_10:', ber_DetNet_10)
print('ber_DetNet_20:', ber_DetNet_20)
print('ber_DetNet_30:', ber_DetNet_30)
print('ber_DetNet_100:', ber_DetNet_100)

            
print('ZF_detector time is:'+str(time_ZF))
print('DetNet_10_detector time is:'+str(time_DetNet_10))
print('DetNet_20_detector time is:'+str(time_DetNet_20))
print('DetNet_30_detector time is:'+str(time_DetNet_30))
print('DetNet_100_detector time is:'+str(time_DetNet_100))

print("Total time is: "+str(time_ed-time_st))


