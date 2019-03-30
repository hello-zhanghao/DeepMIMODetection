# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:56:54 2019

@author: zfzdr
"""
import numpy as np
import tensorflow as tf
#modulation and demodulation:bpsk/qpsk
table_bpsk = np.array([1 + 0j, -1 + 0j])  #modType=2
table_qpsk = np.array(
    [-0.707107 + 0.707107j, -0.707107 - 0.707107j,
    0.707107 + 0.707107j, 0.707107 - 0.707107j])  # modType=4
mod_Table = {'bpsk': table_bpsk, 'qpsk': table_qpsk}

#modulation function
#Input
#sourceSeq: binary sequence, the times of log2(modulation order)
#mod_Name:modulation method:bpsk/qpsk
#Output
#modSeq = modulated signal,dtype='complex',shape=(num_Symbol,1)
def modulation(sourceSeq, mod_Name):
    mod = mod_Table[mod_Name]
    mod_Type = int(mod.size) # Calcuate the Modulation order
    # Calcuate the Symbol number of sourceSeq
    num_Symbol = int(sourceSeq.size * 2 / mod_Type)
    mod_Seq = np.zeros((num_Symbol, 1), dtype='complex128')
    if mod_Type == 2:
        for i in range(num_Symbol):
            index = sourceSeq[i][0]
            mod_Seq[i][0] = mod[index]
    if mod_Type == 4:
        for i in range(num_Symbol):
            index = sourceSeq[i * 2][0] * 2 + sourceSeq[i *2 + 1][0]
            mod_Seq[i][0] = mod[index]
    return mod_Seq

#demodulation
#Input
#receiveSeq:received signal,dtype='complex',shape=(num_Symbol,1)
#mod_Name:modulation method:bpsk/qpsk
#Output
#demod_Seq:demodulated signal,dtype='int',shape=(num_Symbol*2,1)
def demodulation(receiveSeq, mod_Name):
    mod_Type = int(mod_Table[mod_Name].size)
    num_Symbol = int(receiveSeq.size)
    if mod_Type == 2:
        demod_Seq = ((receiveSeq.real < 0) * 1)
    if mod_Type == 4:
        demod_Seq = np.zeros((num_Symbol * 2, 1))
        for i in range((num_Symbol)):
            demod_Seq[i * 2][0] = (receiveSeq[i].real > 0) * 1
            demod_Seq[i * 2 + 1][0] = (receiveSeq[i].imag < 0) * 1
    return demod_Seq

# Define function to generate training data
# Input
# batchsize:num of symbols transmitted per antenna in a frame
# Nt:transmit antennas
# Nr:receive antennas
# mod_Name:modulation method:bpsk/qpsk
# output
# source:base data,binary data
# x_real:modulated signal and transfer real number,dtype='float32',
# y_real:received signal and transfer real number,dtype='float32'
# H_real:channel matrix(real nummber)
def generate_traindata(batchsize, Nt, Nr, mod_Name):
    mod_Type = int(mod_Table[mod_Name].size)
    #H_complex are randomly generated，and H_complex are 归一化的
    H_complex = (np.random.randn(Nr,Nt) + 1j*np.random.randn(Nr,Nt))/np.sqrt(2)
    H_complex = (np.sqrt(Nr)) / np.sqrt(np.trace(np.dot(
            np.transpose(np.conjugate(H_complex)),H_complex))) * H_complex
    H_part1 = np.concatenate((np.real(H_complex), np.imag(H_complex)))
    H_part2 = np.concatenate((-np.imag(H_complex), np.real(H_complex)))
    H_real = np.concatenate((H_part1,H_part2), axis=1).astype('float32')
    numbits_perSymbol = np.log2(mod_Type).astype(int)
    # generate 0,1 bits
    source = np.random.randint(0, 2, ((batchsize * Nt) * numbits_perSymbol, 1))
    # BPSK/QPSK modulation
    x_complex = modulation(source, mod_Name)
    x_complex = np.reshape(x_complex, (Nt, batchsize))
    # Noise Vector with independent,zero mean Gaussian variables of variance 1
    w = np.zeros((Nr, batchsize), dtype='complex')
    for m in range(batchsize):
        SNR = np.random.uniform(8, 16, 1)  # 8dB-13dB uniform distribution
        sigma = np.sqrt(1 / (10 ** (SNR /10)))
        wpart = sigma / np.sqrt(2) * np.random.randn(Nr) + \
                1j * sigma / np.sqrt(2) * np.random.randn(Nr)
        w[:, m] = wpart
    y_complex = np.dot(H_complex, x_complex) + w
    x_real = np.concatenate((np.real(x_complex), np.imag(x_complex)), axis=0)
    y_real = np.concatenate((np.real(y_complex), np.imag(y_complex)), axis=0)
    return source, x_real, y_real, H_real

# Define function to generate test data
# Input
# batchsize:num of symbols transmitted per antenna in a frame
# Nt:transmit antennas
# Nr:receive antennas
# mod_Name:modulation method:bpsk/qpsk
# SNR:specify the SNR to generate different receive signal
# output
# source:base data,binary data
# x_real:modulated signal and transfer real number,dtype='float32',
# y_real:received signal and transfer real number,dtype='float32'
# H_real:channel matrix(real nummber)
def generate_testdata(batchsize, Nt, Nr, mod_Name,SNR):
    mod_Type = int(mod_Table[mod_Name].size)
    #H are randomly generated
    H_complex = (np.random.randn(Nr,Nt) + 1j*np.random.randn(Nr,Nt))/np.sqrt(2)
    H_complex = (np.sqrt(Nr)) / np.sqrt(np.trace(np.dot(
            np.transpose(np.conjugate(H_complex)),H_complex))) * H_complex
    H_part1 = np.concatenate((np.real(H_complex), np.imag(H_complex)))
    H_part2 = np.concatenate((-np.imag(H_complex), np.real(H_complex)))
    H_real = np.concatenate((H_part1,H_part2), axis=1).astype('float32')
    numbits_perSymbol = np.log2(mod_Type).astype(int)
    # generate 0,1 bits
    source = np.random.randint(0, 2, ((batchsize * Nt) * numbits_perSymbol, 1))
    # BPSK/QPSK modulation
    x_complex = modulation(source, mod_Name)
    x_complex = np.reshape(x_complex, (Nt, batchsize))
    # Noise Vector with independent,zero mean Gaussian variables of variance 1
    w = np.zeros((Nr, batchsize), dtype='complex')
    for m in range(batchsize):
        sigma = np.sqrt(1 / (10 ** (SNR /10)))
        wpart = sigma / np.sqrt(2) * np.random.randn(Nr) + \
                1j * sigma / np.sqrt(2) * np.random.randn(Nr)
        w[:, m] = wpart
    y_complex = np.dot(H_complex, x_complex) + w
    x_real = np.concatenate((np.real(x_complex), np.imag(x_complex)), axis=0)
    y_real = np.concatenate((np.real(y_complex), np.imag(y_complex)), axis=0)
    return source, x_real, y_real, H_real

# Calculate the number of error bit between source and x_EST
def bit_error(source, x_EST, Nt, numSymbol_perChannel, mod_Name):
    x_EST_complex = x_EST[0:Nt, :] + 1j*x_EST[Nt:2*Nt, :]
    x_EST_complex_seq = np.reshape(x_EST_complex, (numSymbol_perChannel*Nt, 1))
    x_ESTbit_seq = demodulation(x_EST_complex_seq, mod_Name)
    num_biterror = np.sum(abs(x_ESTbit_seq - source))
    return num_biterror

##python创建目录文件夹
#def mkdir(dir):
#    isExists = os.path.exists(dir)
#    if not isExists:
#        os.mkdir(dir)
#        print(dir+'创建成功')
#        return True
#    else:
#        print(dir+'目录已存在')
#        return false
 
    

##载入参数
#saver = tf.train.import_meta_graph(savedir+'DetNetmodel.cpkt.meta')
#graph = tf.get_default_graph()
#y = graph.get_tensor_by_name(name='receiver:0')
#H = graph.get_tensor_by_name(name='channel:0')
#ZF_detector = graph.get_tensor_by_name(name='ZF_detector:0')
#DetNet_detector = graph.get_tensor_by_name(name='DetNet_detector:0')
class Predict():
    def __init__(self, loc):
        self.graph=tf.Graph()#为每个类(实例)单独创建一个graph
        with self.graph.as_default():
             self.saver=tf.train.import_meta_graph(loc+'.meta')#创建恢复器
             #注意！恢复器必须要在新创建的图里面生成,否则会出错。
        self.sess=tf.Session(graph=self.graph)#创建新的sess
        with self.sess.as_default():
             with self.graph.as_default():
                 self.saver.restore(self.sess,loc)#从恢复点恢复参数
    
    def DetNet_detector(self, y_test, H_test):
        return self.sess.run('DetNet_detector:0',feed_dict={'receiver:0':y_test,'channel:0':H_test})
    
    def ZF_detector(self, y_test, H_test):
        return self.sess.run('ZF_detector:0',feed_dict={'receiver:0':y_test,'channel:0':H_test})
        