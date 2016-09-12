import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib as mp
import argparse
import os, sys
import csv
import math
import time
import matplotlib.pyplot as pl
import random as rd
time1 = time.time()

CIL = np.zeros([1000,1]) # customer ID list
CIN = 0 # customer ID numbers 929
NDC = np.zeros([1000,1]) # number of days in customer data
TDIC = np.zeros([1000,70]) # testing days index of customers
FDIC = [] # flatten data of index of customers

test_days = 70 # number of test days of one customer
max_input_days = 7 # for varing input size, this is the max possible input days

dataframe = pd.read_csv('./SelectedDataFor1EE.csv')
#dataframe

IDframe = DataFrame(dataframe,columns =  ['CustomeID'])
#IDframe

ids = np.array(IDframe)
print '{}'.format(ids.shape)
itermax = ids.shape[0]
old_id = 0
for i in range(itermax):
    new_id = ids[i,0]
    if new_id == old_id:
        continue
    old_id = new_id
    CIL[CIN,0] = old_id
    CIN = CIN + 1
#CIL

for i in range(CIN):
    cus_id = CIL[i,0]
    #print cus_id
    subframe = dataframe[dataframe['CustomeID'] == cus_id]
    #print subframe
    days = subframe.shape[0]
    #print days
    subdata = np.array(subframe)
    subdata = subdata[:,2:]# drop the first two cols --- index and date
    nrows,ncols = subdata.shape
    NDC[i,0] = nrows
    #print nrows,ncols
    flattendata = subdata.reshape((1,nrows*ncols)) # this is the flattened data for customer id
    data_length = flattendata.shape[1] # total data points in single point layout
    train_days = nrows - test_days # traindays
    min_day = max_input_days # lower bound for test days selection
    max_day = nrows # upper bound for test days selection
    
    
    rang = range(min_day,max_day) # test day sample range
    TDIC[i,:] = rd.sample(rang,test_days) # pick unduplicated n indexes as examples
    TDIC[i,:].sort() # produce selected sampling day indexes for test
    ##print TDIC[i,:]
    FDIC.append(flattendata)
    
def test_sample_generate(cusI):
    cus_i = cusI
    print cus_i
    flatdata = FDIC[cusI]
    input0 = np.zeros([0,max_input_days*48])
    label = np.zeros([test_days*48,1])
    
    
    for i in range(test_days):
        day_ind = TDIC[cus_i,i]
        for j in range(48):
            label[i*48+j,0] = flatdata[0,day_ind*48+j]
            tmp = []
            tmp = flatdata[0,day_ind*48+j-max_input_days*48:day_ind*48+j]
            #print tmp.shape
            input1 = np.vstack((input0,tmp))
            input0 = input1
            #print input0.shape
        print i
    return [input0,label]

def train_sample_generate(cusI):
    cus_i = cusI
    flatdata = FDIC[cusI]
    len_tmp = flatdata.shape[1]
    len_tmp = len_tmp/48
    #print len_tmp
    train_days = len_tmp-max_input_days-test_days
    #print train_days
    input0 = np.zeros([0,max_input_days*48])
    label = np.zeros([train_days*48,1])
    
    ind = 0
    for i in range(max_input_days,len_tmp):
        if i in TDIC[cus_i,:]:
            continue # find a day_ind not included in TDIC
        for j in range(48):
            label[ind*48+j,0] = flatdata[0,i*48+j]
            tmp = []
            tmp = flatdata[0,i*48+j-max_input_days*48:i*48+j]
            #print tmp.shape
            input1 = np.vstack((input0,tmp))
            input0 = input1
            #print input0.shape
        ind = ind + 1
        print ind
    return [input0,label]

for i in range(248,277):
    [input_x,label_y] = test_sample_generate(i)
    str_tmp_x = './data/test_x_' + str(i) + '.csv'
    str_tmp_y = './data/test_y_' + str(i) + '.csv'
    input_x = np.array(input_x)
    x = DataFrame(input_x)
    y = DataFrame(label_y)
    #print x
    x.to_csv(str_tmp_x,header = None)
    y.to_csv(str_tmp_y,header = None)
    
    [input_x,label_y] = train_sample_generate(i)
    str_tmp_x = './data/train_x_' + str(i) + '.csv'
    str_tmp_y = './data/train_y_' + str(i) + '.csv'
    input_x = np.array(input_x)
    x = DataFrame(input_x)
    y = DataFrame(label_y)
    #print x
    x.to_csv(str_tmp_x,header = None)
    y.to_csv(str_tmp_y,header = None)
    
# run time
time2 = time.time()
print time2-time1
