### -*- coding: utf-8 -*-
##"""
##Created on Wed Feb 27 11:05:05 2019
##
##@author: zfzdr
##"""
#***********************************************************************
#***********************************************************************


#from __future__ import division
import tensorflow as tf
#import numpy as np
#import math as ms
import time as tm
#import scipy.io as scio
import datetime
#自定义模块
from model import *
from settings import Settings


# Define the parameter of the mode
DetNet_settings = Settings()
Nr = DetNet_settings.Nr    #number of receive antennas
Nt = DetNet_settings.Nt     #number of transmit antenns
mod_Name = DetNet_settings.mod_Name
batchsize = DetNet_settings.batchsize
maxEpochs = DetNet_settings.maxEpochs
Learning_rate = DetNet_settings.Learning_rate
L_size = DetNet_settings.L_size
savedir = DetNet_settings.savedir

# Define the Deep MIMO Detection model of the system
tf.reset_default_graph()
# 1 Define the placeholder
x = tf.placeholder(tf.float32, shape=(2 * Nt, None), name = 'transmit')
y = tf.placeholder(tf.float32, shape=(2 * Nr, None), name = 'receiver')
H = tf.placeholder(tf.float32, shape=(2*Nr, 2*Nt), name ='channel')

# 2 Define the parameters of the network
para_dic = {}
for i in range(L_size):
    para_dic['w{}'.format(i+1)] = tf.Variable(tf.truncated_normal(
            [2 * Nt, 6 * Nt], stddev=1, seed=1, dtype='float32'))

    para_dic['b{}'.format(i+1)] = tf.Variable(tf.constant(0.001,
             shape=[2*Nt, 1], dtype='float32'))

    para_dic['t{}'.format(i+1)]=tf.Variable(tf.constant(0.5, dtype='float32'))


# 3 Define the process of the forward propagation
output_dic = {}
for k in range(L_size):
    if k == 0:
        w = para_dic['w{}'.format(k+1)]
        b = para_dic['b{}'.format(k+1)]
        t = para_dic['t{}'.format(k+1)]
        Stack1 = tf.matmul(tf.transpose(H), y)
        s = tf.shape(Stack1)
        Stack2 = tf.zeros([2 * Nt, s[1]], dtype='float32')
        Stack3 = tf.matmul(tf.matmul(tf.transpose(H), H), Stack2)
        Stack = tf.concat([Stack1, Stack2, Stack3], 0)
        Z = tf.matmul(w, Stack) + b
        x_EST = -1 + tf.nn.relu(Z + t)/abs(t) - tf.nn.relu(Z - t)/abs(t)
        output_dic['x_EST_{}'.format(k+1)] = x_EST
    else:
        w = para_dic['w{}'.format(k+1)]
        b = para_dic['b{}'.format(k+1)]
        t = para_dic['t{}'.format(k+1)]
        Stack1 = tf.matmul(tf.transpose(H), y)
        Stack2 = output_dic['x_EST_{}'.format(k)]
        Stack3 = tf.matmul(tf.matmul(tf.transpose(H), H), Stack2)
        Stack = tf.concat([Stack1, Stack2, Stack3], 0)
        Z = tf.matmul(w, Stack) + b
        x_EST = -1 + tf.nn.relu(Z + t)/abs(t) - tf.nn.relu(Z - t)/abs(t)
        output_dic['x_EST_{}'.format(k+1)] = x_EST
#在使用模型时，我们通常需要知道该模型的输入输出，所以记得将输出添加到一个集合中
tf.add_to_collection('DetNet_detector',output_dic['x_EST_{}'.format(k+1)] )
DetNet_detector = tf.add(output_dic['x_EST_{}'.format(k+1)], 0,
                                    name='DetNet_detector')

# 4 Define the loss function and backpropagation algorithm
xwave_part1 = tf.matrix_inverse(tf.matmul(tf.transpose(H), H))
xwave_part2 = tf.matmul(xwave_part1, tf.transpose(H))
ZF_detector = tf.matmul(xwave_part2, y, name='ZF_detector')

loss = 0
for m in range(L_size):
    loss = loss + tf.log(m+1.0) * tf.reduce_sum(
            tf.squared_difference(x,output_dic['x_EST_{}'.format(m+1)])) \
            / tf.reduce_sum(tf.squared_difference(x,ZF_detector))
tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(
            learning_rate = Learning_rate).minimize(loss=loss)

# 5 save the model
saver = tf.train.Saver()

# 6 Create a session to run TensorFlow to train the model
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(savedir,sess.graph)
    saver.restore(sess,savedir+'DetNetmodel.cpkt')
    time_st =tm.time()
    now_time = datetime.datetime.now().strftime('%X')
    print("Start time is: "+now_time)
    for epoch in range(maxEpochs):
       source, x_Feed, y_Feed, H_Feed = generate_traindata(
           batchsize, Nt, Nr, mod_Name)
       sess.run(train,feed_dict={x:x_Feed, y:y_Feed, H:H_Feed})
       if epoch%10==0:
            Loss_Value=sess.run(loss,feed_dict={x:x_Feed, y:y_Feed, H:H_Feed})
            now_time = datetime.datetime.now().strftime('%X')
            print()
            print("After %d training step(s), Current time is: "%epoch+now_time)
            summary_str = sess.run(merged_summary_op, feed_dict={
                x:x_Feed, y:y_Feed, H:H_Feed})
            summary_writer.add_summary(summary_str, epoch)
            x_DetNet = sess.run(DetNet_detector,feed_dict={
                y:y_Feed, H:H_Feed})
            x_ZF = sess.run(ZF_detector,feed_dict={y:y_Feed, H:H_Feed})
            testbiterrorsum10 = bit_error(
                    source, x_DetNet, Nt, batchsize, mod_Name)
            biterrorsum_ZF = bit_error(
                    source, x_ZF, Nt, batchsize, mod_Name)
            ber_ZF = biterrorsum_ZF / batchsize / Nt / 2
            ber_DetNet = testbiterrorsum10 / batchsize / Nt / 2
            print("Loss=%g Acc_ZF=%g  Acc_DetNet=%g"%(
                Loss_Value, ber_ZF,ber_DetNet))
       if epoch%50==0:
           saver.save(sess,savedir+'DetNetmodel.cpkt')
time_ed = tm.time()
print("Total time is: "+str(time_ed-time_st))


