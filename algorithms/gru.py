# -*- coding: utf-8 -*-
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import math, datetime
import os, sys

def create_dataset(dataset, lookback=1):
    dataX, dataY = [], [] # create 2 empty lists

    # go through the lenght of dataset, subtract the lookback and 1. 2 steps before the end of dataset, 
    #because we predict 1 step to the future
    for i in range(len(dataset)-lookback-1):
        a = dataset[i:(i+lookback),0]
        dataX.append(a)
        dataY.append(dataset[i+lookback,0]) # get the next value
    return np.array(dataX), np.array(dataY)

# receive the parameters by command line
filename, BATCH_SIZE, NB_EPOCHS, LAYER1, LAYER2 = str(sys.argv[1]), int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
#perc = float(sys.argv[6])

# Dataset configuration
dataset = pd.read_csv(filename, usecols = [1], header=None)
dataset.columns = ["request"]
dataset = dataset.values #convert to the array
dataset = dataset.astype('float32') # convert to float

# length of our dataset
training_size = int(len(dataset)*0.67)
testing_size = len(dataset)-training_size

# split the data set
train, test = dataset[0:training_size:], dataset[training_size:len(dataset),:]

# one time step to the future
lookback = 1
trainX, trainY = create_dataset(train, lookback)
testX, testY = create_dataset(test, lookback)

# Scaling dataset
x_train, y_train = trainX, trainY 
x_test, y_test = testX, testY

# scaling values for model
scaleX = MinMaxScaler()
scaleY = MinMaxScaler()

trainX = scaleX.fit_transform(x_train)
trainX = trainX.reshape((-1,1,1))

trainY = scaleY.fit_transform(y_train.reshape(-1,1))

testX  = scaleX.fit_transform(x_test)
testX = testX.reshape((-1,1,1))

testY  = scaleY.fit_transform(y_test.reshape(-1,1))

# creating model using Keras
model_name = 'requests_GRU'
model = Sequential()
model.add(GRU(units=LAYER1,
              return_sequences=True,
              input_shape=(1, 1)))
model.add(Dropout(0.2))
model.add(GRU(units=LAYER2))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#test with less samples	
#size = int(training_size * (perc /100))
#trainX = trainX[0:training_size - size:]
#trainY = trainY[0:training_size - size:]
timer = datetime.datetime.now() 
# Compilation and training
start = datetime.datetime.now() 
model.compile(loss='mean_squared_error', optimizer='adam')

print "Compilation Time : ",  datetime.datetime.now()  - start
start = datetime.datetime.now() 
model.fit(trainX,trainY,batch_size=BATCH_SIZE, epochs=NB_EPOCHS, validation_split=0.1, verbose=0)
print "Training time : ",  datetime.datetime.now()  - start
#model.save("{}.h5".format(model_name))

# Making predictions
yhat = model.predict(trainX)
yhat = scaleX.inverse_transform(yhat)
y_test = scaleX.inverse_transform(trainY)
print "AMOSTRAS",len(trainY)
train_score = mean_squared_error(y_test, yhat)
print ('Trainscore: %.2f' %(math.sqrt(train_score)))

start = datetime.datetime.now() 

yhat = model.predict(testX)
print "test time: ", datetime.datetime.now() -start

print "Timer: ",  datetime.datetime.now()  - timer
yhat = scaleY.inverse_transform(yhat)
y_test = scaleY.inverse_transform(testY)

#RMSE score
test_score = mean_squared_error(y_test, yhat)
#R square score
r2 = r2_score(y_test,yhat)

print('Filename:',filename)
print('BATCH_SIZE:',BATCH_SIZE)
print('NB_EPOCHS: ',NB_EPOCHS) 
print('LAYER1: ',LAYER1)
print('LAYER2: ',LAYER2) 

print('R2: ',r2) 

print ('Testscore: %.2f MSE (%.2f RMSE)' %(test_score, math.sqrt(test_score)))

# Save into a file the command of executions and its scores
f=open("GRU_Execuções.txt", "a")
f.write("\n python gru.py %s %d %d %d %d - R2: %.2f,Trainscore: %.2f MSE (%.2f RMSE), Testscore: %.2f MSE (%.2f RMSE)" %(filename, BATCH_SIZE,NB_EPOCHS, LAYER1,LAYER2, r2,train_score, math.sqrt(train_score), test_score, math.sqrt(test_score))) 

#print 'actual,  predicted'
#for i in range(0, len(y_test)):
#  print('%.2f, %.2f' % (y_test[i],yhat[i+1]))

# plot some samples.
#plt.plot(yhat[-100:], label='Predicted')
#plt.plot(y_test[-100:], label='Current')
#plt.legend()
#plt.grid()

#plt.ylabel('Requests')
#plt.xlabel('Time')
#plt.show()'''
