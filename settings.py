# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:21:11 2019

@author: zfzdr
"""
import os
class Settings():
    def __init__(self):
        self.Nr = 16
        self.Nt = 8
        self.mod_Name = 'qpsk'
        self.M = 4
        self.batchsize = 2000
        self.maxEpochs = 30000
        self.Learning_rate = 0.001
        self.L_size = 10
        self.savedir =  os.getcwd() + '/model_save_%d/'%self.L_size
        self.batchsize_test = 3000
        self.maxEpochs_test = 200
        self.SNR_st = 1
        self.SNR_ed = 16