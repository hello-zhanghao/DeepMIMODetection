#from tensorflow.python.tools.inspect_checkpoint import  \
#print_tensors_in_checkpoint_file as pt
#
#from tensorflow.python import pywrap_tensorflow
#
#savedir = os.getcwd() + '/model_save_%d/'%5
#savefile = savedir+'/DetNetmodel.cpkt'
#pt(savedir+'DetNetmodel.cpkt', None, True, True)
##reader = pywrap_tensorflow.NewCheckpointReader(savefile)
##
##var_to_shape_map = reader.get_variable_to_shape_map()
##
##for key in var_to_shape_map:
##
##  print("tensor_name: ", DetNet_V3.2g
import tensorflow as tf
import time as tm
import datetime
import os

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
#savedir = DetNet_settings.savedir
#savedir = savedir.replace('\\','/')
#savedir = r'C:\Users\zfzdr\Desktop\DeepMIMODetection_V3.3\model_save_30\\'
M = DetNet_settings.M


numbits_perSymbol = int(np.log2(M))
source_length = batchsize_test * maxEpochs_test * Nt * numbits_perSymbol


c1 = tf.constant(1)
print(c1.graph)
g1 = tf.Graph()
g2 = tf.Graph()
g3 = tf.Graph()
print(g1)
print(g2)
print(g3)
sess = tf.Session(graph=g1)
loc1 = os.getcwd() + '/model_save_10/DetNetmodel.cpkt'
loc2 = os.getcwd() + '/model_save_20/DetNetmodel.cpkt'
predict1 = Predict(loc1)
predict2 = Predict(loc2)
time_ZF = 0
time_DetNet = 0
ber_DetNet_10 = np.zeros((1,15))
ber_DetNet_20 = np.zeros((1,15))
ber_ZF = np.zeros((1,15))
time_st = tm.time()
for SNR in range(1,16):
    biterrorsum_DetNet_10 = 0
    biterrorsum_DetNet_20 = 0
    biterrorsum_ZF = 0
    for i in range(maxEpochs_test):
        source_test, x_test, y_test, H_test = generate_testdata(
            batchsize_test, Nt, Nr, mod_Name, SNR)
#            #计算ZF检测的误码率和时间复杂度
#            time_ZF_st = tm.time()
#            x_ZF = sess.run(ZF_detector,feed_dict={y:y_test, H:H_test})
#            biterrorsum_ZF = biterrorsum_ZF + bit_error(
#                    source_test, x_ZF, Nt, batchsize_test, mod_Name)
#            time_ZF_ed = tm.time()
#            time_ZF = time_ZF + time_ZF_ed - time_ZF_st        
#                #计算DetNet检测的误码率和时间复杂度
            
        time_DetNet_st = tm.time()
        x_DetNet_10 = predict1.DetNet_detector(y_test, H_test)
        x_DetNet_20 = predict2.DetNet_detector(y_test, H_test)
        x_ZF = predict1.ZF_detector(y_test, H_test)
        
        biterrorsum_DetNet_10 = biterrorsum_DetNet_10 + bit_error(
            source_test, x_DetNet_10, Nt, batchsize_test, mod_Name)
        biterrorsum_DetNet_20 = biterrorsum_DetNet_20 + bit_error(
            source_test, x_DetNet_20, Nt, batchsize_test, mod_Name)
        biterrorsum_ZF = biterrorsum_ZF + bit_error(
            source_test, x_ZF, Nt, batchsize_test, mod_Name)
        
        time_DetNet_ed = tm.time()
        time_DetNet = time_DetNet + time_DetNet_ed - time_DetNet_st
#        计算不同误码率下ZF检测和DetNet检测的性能   
    ber_DetNet_10[:, SNR-1] = biterrorsum_DetNet_10 / source_length
    ber_DetNet_20[:, SNR-1] = biterrorsum_DetNet_20 / source_length
    ber_ZF[:, SNR-1]  = biterrorsum_ZF / source_length

print('ber_DetNet_10:', ber_DetNet_10[0])
print('ber_DetNet_20:', ber_DetNet_20[0])
print('ber_ZF:', ber_ZF)
            
time_ed = tm.time()
print("Total time is: "+str(time_ed-time_st))
    
    
#with g1.as_default():
#    saver = tf.train.import_meta_graph(loc + '.meta')
#    saver.restore(sess, loc)      
#    time_ZF = 0
#    time_DetNet = 0
#    ber_DetNet_10 = np.zeros((1,15))
#    ber_ZF_10 = np.zeros((1,15))
#    for SNR in range(1,16):
#        biterrorsum_DetNet = 0
#        biterrorsum_ZF = 0
#        for i in range(maxEpochs_test):
#            source_test, x_test, y_test, H_test = generate_testdata(
#                    batchsize_test, Nt, Nr, mod_Name, SNR)
#            #计算ZF检测的误码率和时间复杂度
#            time_ZF_st = tm.time()
#            x_ZF = sess.run(ZF_detector,feed_dict={y:y_test, H:H_test})
#            biterrorsum_ZF = biterrorsum_ZF + bit_error(
#                    source_test, x_ZF, Nt, batchsize_test, mod_Name)
#            time_ZF_ed = tm.time()
#            time_ZF = time_ZF + time_ZF_ed - time_ZF_st        
#                #计算DetNet检测的误码率和时间复杂度
#            time_DetNet_st = tm.time()
#            x_DetNet = sess.run(DetNet_detector,feed_dict={y:y_test,H:H_test})
#            biterrorsum_DetNet = biterrorsum_DetNet + bit_error(
#                    source_test, x_DetNet, Nt, batchsize_test, mod_Name)
#            time_DetNet_ed = tm.time()
#            time_DetNet = time_DetNet + time_DetNet_ed - time_DetNet_st
#        #计算不同误码率下ZF检测和DetNet检测的性能   
#        ber_DetNet_10[:, SNR-1] = biterrorsum_DetNet / source_length
#        ber_ZF_20[:, SNR-1]  = biterrorsum_ZF/source_length
#            
#        
#        
#        
#time_ed = tm.time()
#print("Total time is: "+str(time_ed-time_st))
##with tf.Session as sess:
##    print(g1)
##    print(g2)
#    print(g3)
    