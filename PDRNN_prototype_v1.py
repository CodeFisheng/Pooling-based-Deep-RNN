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

time1 = time.time() # set up counter to record run time
data_dir = '../../database/data/' # directory contains input data
num_epoches = 50000 # training epoches for each customer samples
n_steps = 48 # input size
cus_num = 1
test_batch_size = 70*48*cus_num # days of a batch
test_minibatch_size = 70*48
train_batch_size = 5*48
feature_size = 1 # same time of a week
n_hidden = 30 # input size
num_layers = 3
n_output = 1
cus_label_list = []

def findTrLabel(index,n):
    res = 0
    tmp = 0
    for i in range(0,cus_num):
        tmp = tmp + cus_label_list[i]
        if tmp > index:
            res = i
            break
    
    retur = np.zeros((n,cus_num))
    retur[:,res] = 1
    
    #retur = np.zeros((cus_num))
    #retur[res] = 1
    
    #retur = float(retur)
    return retur

def findTsLabel(index,n):
    res = 0
    res = np.floor(index/test_minibatch_size)
    
    retur = np.zeros((n,cus_num))
    retur[:,res] = 1
    
    #retur = np.zeros((cus_num))
    #retur[res] = 1
    
    #retur = float(retur)
    return retur

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
    tmpZ = [findTrLabel(i,steps) for i in train_days_list]
    Z = np.array(tmpZ).reshape(n_batch,steps,cus_num)
        
    return (X,Y,Z)

def test_data_gen(x_data,y_data,steps = 48, n_batch = test_batch_size):
    X = np.zeros((n_batch,steps,feature_size))
    Y = np.zeros((n_batch,feature_size))
    tmpZ = [findTsLabel(i,steps) for i in range(test_batch_size)]
    #print tmpZ
    Z = np.array(tmpZ).reshape(n_batch,steps,cus_num)
    #print x_data[:,0-steps:].shape,y_data.shape
    #print n_batch, steps
    X = x_data[:,0-steps:].reshape(n_batch,steps,feature_size)
    Y = y_data.reshape(n_batch,feature_size)
    
    return (X,Y,Z)

# create placeholder for x and y
x = tf.placeholder("float",[None,n_steps,feature_size])
z = tf.placeholder("float",[None,n_steps,cus_num])
#z = tf.placeholder("float",[None,cus_num])
istate = tf.placeholder("float",[None,num_layers*2*n_hidden])
y = tf.placeholder("float",[None,n_output])


# Define weights
weights = {
    'inp': tf.Variable(tf.random_normal([cus_num, n_hidden])),
    'hidden': tf.Variable(tf.random_normal([feature_size, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'inp': tf.Variable(tf.random_normal([n_hidden])),
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

def RNN(_X, _Z, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _Z = tf.transpose(_Z, [1, 0, 2])
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, feature_size]) # (n_steps*batch_size, n_input)
    _Z = tf.reshape(_Z, [-1, cus_num])
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    _Z = tf.matmul(_Z, _weights['inp']) + _biases['inp']
    _Q = tf.add(_X,_Z)
    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers)
    
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _Q = tf.split(0, n_steps, _Q) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(stacked_lstm_cell, _Q, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, z, istate, weights, biases)

#cost function 
cost = tf.reduce_mean(tf.pow(pred-y,2)) # cost function of this batch of data
#cost2 = tf.abs(pred-y) # 
#compute parameter updates
#train_op = tf.train.GradientDescentOptimizer(0.008).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
#optimizer2 = tf.train.RMSPropOptimizer(0.005, 0.3).minimize(cost2)

## iterating among all customers to find current training customer
result_final = []
#cus_id_forselect[cus_num]
sim_id_forecast = [0,266,295,431,465,597,615,627,736,798]
dis_id_forecast = [0,230,460,487,520,655,754,767,818,907]
#cus_list = [8,9,11,18,29,45,48,49,58,60,64,65,66,68]
#cus_list = [8,9,11,18,45,49,58,64,65,68]
#cus_list = [3,10,11,19,28,30,32,57,96,97,98,100,125,131,133,134,153,164,177,179,184,202,205,215,231,237,243,246,253,256,261,264,269,291,312,314,339,342,344,362,381,387,417,421,429,431,432,454,460,465,471,472,486,491,502,511,513,514,519,542,553,565,579,588,592,599,602,607,619,624,637,646,658,662,667,668,671,677,679,683,704,705,708,710,714,728,746,749,786,819,822,833,840,847,850,851,853,854,878,899]
starti = 0
cus_list= [312]
endi = cus_num
for i in range(starti,endi):
    #ii = cus_id[i]
    ii = cus_list[i]#sim_id_forecast[i]
    test_x_name = data_dir + 'test_x_' + str(ii) + '.csv'
    test_y_name = data_dir + 'test_y_' + str(ii) + '.csv'
    train_x_name = data_dir + 'train_x_' + str(ii) + '.csv'
    train_y_name = data_dir + 'train_y_' + str(ii) + '.csv'
    leng_list = []
    tmp_data = np.array(pd.read_csv(test_x_name,header = None))
    if i == starti:
        test_x_data = tmp_data[:,1:]
    else:
        test_x_data = np.concatenate((test_x_data,tmp_data[:,1:]),axis=0)
    
    # print test_x_data.dtype  data are stored as float64 double precision format
    tmp_data = np.array(pd.read_csv(test_y_name,header = None))
    if i == starti:
        test_y_data = tmp_data[:,1:]
    else:
        test_y_data = np.concatenate((test_y_data,tmp_data[:,1:]),axis=0)
    
    tmp_data = np.array(pd.read_csv(train_x_name,header = None))
    if i == starti:
        train_x_data = tmp_data[:,1:]
    else:
        train_x_data = np.concatenate((train_x_data,tmp_data[:,1:]),axis=0)
        
    tmp_data = np.array(pd.read_csv(train_y_name,header = None))
    if i == starti:
        train_y_data = tmp_data[:,1:]
    else:
        train_y_data = np.concatenate((train_y_data,tmp_data[:,1:]),axis=0)
    cus_label_list.append(tmp_data[:,1:].shape[0])
    
traindays = train_y_data.shape[0]

outlist = np.zeros([(num_epoches/10),test_batch_size])
kind = 0
for i in range(0,1):
    # generate test data
    [test_x,test_y,test_z] = test_data_gen(test_x_data,test_y_data,n_steps,test_batch_size)
    test_x = test_x.reshape(test_batch_size,n_steps,feature_size)
    print test_x.shape,test_z.shape
    test_z = test_z.reshape(test_batch_size,n_steps,cus_num)
    ### Execute
    # Initializing the variables
    init = tf.initialize_all_variables()
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
            [X,Y,Z] = train_data_gen(traindays,train_x_data,train_y_data,n_steps,train_batch_size)
            X = X.reshape(train_batch_size,n_steps,feature_size)
            Z = Z.reshape(train_batch_size,n_steps,cus_num)

            #Create the dictionary of inputs to feed into sess.run
            
            sess.run(optimizer,feed_dict={x:X,z:Z,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))})   
            #perform an update on the parameters

            #loss1 = sess.run(cost, feed_dict = {x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))} )
            #loss2 = sess.run(cost, feed_dict = {x:test_x,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )            #compute the cost on the validation set
            #output_tmp = sess.run(pred,feed_dict = {x:X,y:Y,istate:np.zeros((train_batch_size,num_layers*2*n_hidden))} )
            #outp_train = output_tmp
            if k % 10 == 0:
                output_tmp = sess.run(pred,feed_dict = {x:test_x,z:test_z,y:test_y,istate:np.zeros((test_batch_size,num_layers*2*n_hidden))} )
                outp_test = output_tmp
                outlist[kind,:] = outp_test.copy().T
                kind = kind + 1
                print "Iter " + str(k) + " ---- Process: " + "{:.2f}".format(100*float(k)/float(num_epoches)) + "%"


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def maxe(predictions, targets):
    return max(abs(predictions-targets))
def mae(prediction, targets):
    return (abs(prediction-targets)).mean()
def mape(prediction, targets):
    return (abs(prediction-targets)*100.0/(targets+0.0001)).mean()
RList = np.zeros([(num_epoches/10),cus_num])
rmseList = np.zeros([(num_epoches/10),cus_num])
maxeList = np.zeros([(num_epoches/10),cus_num])
maeList = np.zeros([(num_epoches/10),cus_num])
mapeList = np.zeros([(num_epoches/10),cus_num])
for i in range(kind):
    for j in range(cus_num):
        out = np.array(outlist[i])
        tmp = out.T.reshape((1,test_batch_size))
        RList[i,j] = np.corrcoef(tmp[0,test_minibatch_size*j:test_minibatch_size*(j+1)],test_y.T[0,test_minibatch_size*j:test_minibatch_size*(j+1)])[0,1]
        rmseList[i,j] = rmse(tmp[0,test_minibatch_size*j:test_minibatch_size*(j+1)],test_y.T[0,test_minibatch_size*j:test_minibatch_size*(j+1)])
        maxeList[i,j] = maxe(tmp[0,test_minibatch_size*j:test_minibatch_size*(j+1)],test_y.T[0,test_minibatch_size*j:test_minibatch_size*(j+1)])
        maeList[i,j] = mae(tmp[0,test_minibatch_size*j:test_minibatch_size*(j+1)],test_y.T[0,test_minibatch_size*j:test_minibatch_size*(j+1)])
        if j == 5:
            continue
        mapeList[i,j] = mape(tmp[0,test_minibatch_size*j:test_minibatch_size*(j+1)],test_y.T[0,test_minibatch_size*j:test_minibatch_size*(j+1)])
#print R
R2 = np.mean(RList,axis=1)
rmse2 = np.mean(rmseList,axis=1)
maxe2 = np.mean(maxeList,axis=1)
mae2 = np.mean(maeList,axis=1)
mape2 = np.mean(mapeList,axis=1)
print 10
postfix = 'multi_5_30_final.csv'
DataFrame(R2).to_csv('./PDRNN-response-result/R2_'+postfix)
DataFrame(RList).to_csv('./PDRNN-response-result/RList_'+postfix)
DataFrame(rmse2).to_csv('./PDRNN-response-result/rmse2_'+postfix)
DataFrame(rmseList).to_csv('./PDRNN-response-result/rmseList_'+postfix)
DataFrame(maxe2).to_csv('./PDRNN-response-result/maxe2_'+postfix)
DataFrame(maxeList).to_csv('./PDRNN-response-result/maxeList_'+postfix)
DataFrame(mae2).to_csv('./PDRNN-response-result/mae2_'+postfix)
DataFrame(maeList).to_csv('./PDRNN-response-result/maeList_'+postfix)
DataFrame(mape2).to_csv('./PDRNN-response-result/mape2_'+postfix)
DataFrame(mapeList).to_csv('./PDRNN-response-result/mapeList_'+postfix)
time2 = time.time()
time = time2-time1
time
mapeList
DataFrame(out).to_csv('./PDRNN-response-result/out3'+postfix)
DataFrame(test_y).to_csv('./PDRNN-response-result/testy3'+postfix)
