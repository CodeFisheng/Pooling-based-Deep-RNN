import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib as mp
import random as rd
import argparse
import os, sys
import csv
import math
import time
import matplotlib.pyplot as pl
class rnn_base:
    def __init__(self,nn,conf):
        time1 = time.time() # set up counter to record run time
        data_dir = './data/' # directory contains input data
        num_epoches = conf[nn,6] # training epoches for each customer samples
        n_steps = conf[nn,5] # input size
        cus_num = conf[nn,3]
        test_batch_size = 70*48*cus_num # days of a batch
        train_batch_size = 50*48
        feature_size = 1 # same time of a week
        n_hidden = conf[nn,2] # input size
        num_layers = conf[nn,1]
        n_output = 1
        Rs = conf[nn,4]

        def train_data_gen(totaltraindays,x_data,y_data,steps = 48, n_batch = train_batch_size):
            X = np.zeros((n_batch,steps,feature_size))
            Y = np.zeros((n_batch,feature_size))
            rang = range(totaltraindays) # test day sample range
            train_days_list = rd.sample(rang,n_batch) # pick unduplicated n indexes as examples
            #print totaltraindays
            tmpX = [x_data[i,0-steps:] for i in train_days_list]
            tmpY = [y_data[i,:] for i in train_days_list]
            X = np.array(tmpX).reshape(n_batch,steps,feature_size)
            Y = np.array(tmpY).reshape(n_batch,feature_size)
            return (X,Y)

        def test_data_gen(x_data,y_data,steps = 48, n_batch = test_batch_size):
            X = np.zeros((n_batch,steps,feature_size))
            Y = np.zeros((n_batch,feature_size))
            #print x_data[:,0-steps:].shape,y_data.shape
            #print n_batch, steps
            X = x_data[:,0-steps:].reshape(n_batch,steps,feature_size)
            Y = y_data.reshape(n_batch,feature_size)
            return (X,Y)

        # create placeholder for x and y
        x = tf.placeholder("float",[None,n_steps,feature_size])
        istate = tf.placeholder("float",[None,num_layers*2*n_hidden])
        y = tf.placeholder("float",[None,n_output])

        # Define weights
        weights = {
            'hidden': tf.Variable(tf.random_normal([feature_size, n_hidden])), # Hidden layer weights
            'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([n_hidden])),
            'out': tf.Variable(tf.random_normal([n_output]))
        }

        def RNN(_X, _istate, _weights, _biases):

            # input shape: (batch_size, n_steps, n_input)
            _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
            # Reshape to prepare input to hidden activation
            _X = tf.reshape(_X, [-1, feature_size]) # (n_steps*batch_size, n_input)
            # Linear activation
            _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

            # Define a lstm cell with tensorflow
            lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
            stacked_lstm_cell = rnn_cell.MultiRNNCell([lstm_cell]*num_layers)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

            # Get lstm cell output
            outputs, states = rnn.rnn(stacked_lstm_cell, _X, initial_state=_istate)

            # Linear activation
            # Get inner loop last output
            return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

        pred = RNN(x, istate, weights, biases)

        #cost function 
        cost = tf.reduce_mean(tf.pow(pred-y,2)) # cost function of this batch of data
        cost2 = tf.abs(pred-y) # 
        #compute parameter updates
        #train_op = tf.train.GradientDescentOptimizer(0.008).minimize(cost)
        optimizer = tf.train.RMSPropOptimizer(0.005, 0.3).minimize(cost)
        optimizer2 = tf.train.RMSPropOptimizer(0.005, 0.3).minimize(cost2)

        ## iterating among all customers to find current training customer
        result_final = []
        for i in range(0,cus_num):
            test_x_name = data_dir + 'test_x_' + str(i) + '.csv'
            test_y_name = data_dir + 'test_y_' + str(i) + '.csv'
            train_x_name = data_dir + 'train_x_' + str(i) + '.csv'
            train_y_name = data_dir + 'train_y_' + str(i) + '.csv'

            tmp_data = np.array(pd.read_csv(test_x_name,header = None))
            if i == 0:
                test_x_data = tmp_data[:,1:]
            else:
                test_x_data = np.concatenate((test_x_data,tmp_data[:,1:]),axis=0)

            # print test_x_data.dtype  data are stored as float64 double precision format
            tmp_data = np.array(pd.read_csv(test_y_name,header = None))
            if i == 0:
                test_y_data = tmp_data[:,1:]
            else:
                test_y_data = np.concatenate((test_y_data,tmp_data[:,1:]),axis=0)

            tmp_data = np.array(pd.read_csv(train_x_name,header = None))
            if i == 0:
                train_x_data = tmp_data[:,1:]
            else:
                train_x_data = np.concatenate((train_x_data,tmp_data[:,1:]),axis=0)

            tmp_data = np.array(pd.read_csv(train_y_name,header = None))
            if i == 0:
                train_y_data = tmp_data[:,1:]
            else:
                train_y_data = np.concatenate((train_y_data,tmp_data[:,1:]),axis=0)

        traindays = train_y_data.shape[0]

        for i in range(0,1):
            # generate test data
            test_x,test_y = test_data_gen(test_x_data,test_y_data,n_steps,test_batch_size)
            test_x = test_x.reshape(test_batch_size,n_steps,feature_size)
            ### Execute
            # Initializing the variables
            init = tf.initialize_all_variables()
            outp = []
            outlist = np.zeros([Rs,test_batch_size])
            with tf.Session() as sess:
                # Create a summary to monitor cost function
                #tf.scalar_summary("loss", cost)
                #tf.scalar_summary("loss2",cost2)
                # Merge all summaries to a single operator
                #merged_summary_op = tf.merge_all_summaries()

                # tensorboard info.# Set logs writer into folder /tmp/tensorflow_logs
                #summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs', graph_def=sess.graph_def)

                #initialize all variables in the model
                sess.run(init)
                for k in range(num_epoches):
                    #Generate Data for each epoch
                    #What this does is it creates a list of of elements of length seq_len, each of size [batch_size,input_size]
                    #this is required to feed data into rnn.rnn
                    #print traindays
                    X,Y = train_data_gen(traindays,train_x_data,train_y_data,n_steps,train_batch_size)
                    X = X.reshape(train_batch_size,n_steps,feature_size)


                    #Create the dictionary of inputs to feed into sess.run
                    if k < 100:
                        sess.run(optimizer2,feed_dict={x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))})
                    else:
                        sess.run(optimizer,feed_dict={x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))})   
                    #perform an update on the parameters

                    #loss1 = sess.run(cost, feed_dict = {x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))} )
                    #loss2 = sess.run(cost, feed_dict = {x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )            #compute the cost on the validation set
                    #output_tmp = sess.run(pred,feed_dict = {x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))} )
                                                                    #outp_train = output_tmp
                    if k >= num_epoches-Rs:
                        output_tmp = sess.run(pred,feed_dict = {x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )
                        outp_test = output_tmp
                    if k >= num_epoches-Rs:
                        outlist[k-num_epoches+Rs,:] = outp_test.copy().T

                    # Write logs at every iteration
                    #summary_str = sess.run(merged_summary_op, feed_dict={x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )
                    #summary_writer.add_summary(summary_str, k)
                    #print "Iter " + str(k) + ", Minibatch Loss ---- Train = " + "{:.6f}".format(loss1)# + "; Test = " + "{:.6f}".format(loss2)
                    print "Iter " + str(k) + " ---- Process: " + "{:.2f}".format(100*float(k)/float(num_epoches)) + "%"
                                                                            #print "haha{}".format(outp)

        R = []
        RR  = []
        for i in range(Rs):
            out = np.array(outlist[i])
            R.append(np.corrcoef(out.T,test_y.T)[0,1])
            RR.append(np.corrcoef(out.T,test_y.T)[0,1]**2)
        print R
        RRR = np.mean(R)# average Rs R in this time of train
        # run time
        time2 = time.time()
        print 'total running time cost:{}s'.format(time2-time1)

        # append R
        result_final.append(RRR)
        self.rf = result_final
    

