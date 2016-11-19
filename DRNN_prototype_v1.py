import tensorflow as tf
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

data_dir = './data/' # directory contains input data
num_epoches = 5000 # training epoches for each customer samples
n_steps = 48 # input size
test_batch_size = 70*48 # days of a batch
train_batch_size = 2*48
feature_size = 1 # same time of a week
n_hidden = 30 # input size
num_layers = 2
n_output = 1
Rs = 20

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

def maxe(predictions, targets):
    return max(abs(predictions-targets))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# create placeholder for x and y
#with tf.device('/gpu:0'):
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
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)
    # Get lstm cell output
    outputs, states = tf.nn.rnn(stacked_lstm_cell, _X, initial_state=_istate)
    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, istate, weights, biases)
#cost function 
cost = tf.reduce_mean(tf.pow(pred-y,2)) # cost function of this batch of data
#compute parameter updates
#optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

## iterating among all customers to find current training customer
#cus_list = [4,5,8,9]
cus_count = 100
cus_list = [3,10,11,19,28,30,32,57,96,97,98,100,125,131,133,134,153,164,177,179,184,202,205,215,231,237,243,246,253,256,261,264,269,291,312,314,339,342,344,362,381,387,417,421,429,431,432,454,460,465,471,472,486,491,502,511,513,514,519,542,553,565,579,588,592,599,602,607,619,624,637,646,658,662,667,668,671,677,679,683,704,705,708,710,714,728,746,749,786,819,822,833,840,847,850,851,853,854,878,899]
time_record = np.zeros(cus_count)
RList = np.zeros([(num_epoches/10),cus_count])
rmseList = np.zeros([(num_epoches/10),cus_count])
maxeList = np.zeros([(num_epoches/10),cus_count])
for i in range(0,100):
    time1 = time.time()
    print i
    outlist = np.zeros([(num_epoches/10),test_batch_size])
    kind = 0
    ii = cus_list[i]
    test_x_name = data_dir + 'test_x_' + str(ii) + '.csv'
    test_y_name = data_dir + 'test_y_' + str(ii) + '.csv'
    train_x_name = data_dir + 'train_x_' + str(ii) + '.csv'
    train_y_name = data_dir + 'train_y_' + str(ii) + '.csv'
    tmp_data = np.array(pd.read_csv(test_x_name,header = None))
    test_x_data = tmp_data[:,1:]
    # print test_x_data.dtype  data are stored as float64 double precision format
    tmp_data = np.array(pd.read_csv(test_y_name,header = None))
    test_y_data = tmp_data[:,1:]
    tmp_data = np.array(pd.read_csv(train_x_name,header = None))
    train_x_data = tmp_data[:,1:]
    tmp_data = np.array(pd.read_csv(train_y_name,header = None))
    train_y_data = tmp_data[:,1:]
    #log them
    #test_x_data = np.log(test_x_data)
    #test_y_data = np.log(test_y_data)
    #train_x_data = np.log(train_x_data)
    #train_y_data = np.log(train_y_data)
    
    traindays = train_y_data.shape[0]
    # generate test data
    test_x,test_y = test_data_gen(test_x_data,test_y_data,n_steps)
    test_x = test_x.reshape(test_batch_size,n_steps,feature_size)
    ### Execute
    # Initializing the variables
    init = tf.initialize_all_variables()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Create a summary to monitor cost function
        #tf.scalar_summary("loss", cost)
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
            X,Y = train_data_gen(traindays,train_x_data,train_y_data,n_steps)
            X = X.reshape(train_batch_size,n_steps,feature_size)


            #Create the dictionary of inputs to feed into sess.run
            #if k < 0:
            #    sess.run(optimizer2,feed_dict={x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))})
            #else:
            sess.run(optimizer,feed_dict={x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))})   
            #perform an update on the parameters

            # Write logs at every iteration
            #if k>50 & k%10 == 0:
            #    summary_str = sess.run(merged_summary_op, feed_dict={x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )
            #    summary_writer.add_summary(summary_str, k)
            
            #if k % 10 == 0:
            if k % 10 == 0:
                output_tmp_ex = sess.run(pred,feed_dict = {x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )  
                print "Iter " + str(k) + " ---- Process: " + "{:.2f}".format(100*float(k)/float(num_epoches)) + "%"
                outp_test = output_tmp_ex
                outlist[kind,:] = outp_test.copy().T
                kind = kind + 1
            #    print ktmp
            #if k % 10 == 0:
            #    output_tmp_ex = sess.run(pred,feed_dict = {x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )
            #    print "Iter " + str(k)# + ", Minibatch Loss ---- Train = " + "{:.6f}".format(loss1) + "; Test = " + "{:.6f}".format(loss2)
        #print "haha{}".format(outp)
            #    ktmp = np.corrcoef(output_tmp_ex.T,test_y.T)[0,1]
            #    accuracy1.append(ktmp)
            #    print ktmp
    time2 = time.time()
    time_record[i] = time2-time1
    ## evaluation
    for j in range(kind):
        out = np.array(outlist[j])
        tmp = out.T.reshape((1,test_batch_size))
        RList[j][i] = np.corrcoef(tmp[0,:],test_y.T[0,:])[0,1]
        rmseList[j][i] = rmse(tmp[0,:],test_y.T[0,:])
        maxeList[j][i] = maxe(tmp[0,:],test_y.T[0,:])
    
    ## serialize
    prefix = './pes-result/'
    postfix = '-house-' + str(num_layers) + '-' + str(n_hidden) + '.csv'
DataFrame(RList).to_csv(prefix + 'R' + postfix)
DataFrame(rmseList).to_csv(prefix + 'RMSE' + postfix)
DataFrame(maxeList).to_csv(prefix + 'MAXE' + postfix)
DataFrame(time_record).to_csv(prefix + 'TimeLog2.csv')
    
    
