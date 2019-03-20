# -*- coding: utf-8 -*-
#!pip install hyperas
from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperas.utils import eval_hyperopt_space, space_eval

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from keras.utils import np_utils
import os, sys
import pandas as pd
import numpy as np

def data():
    '''
    Data providing function:
    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''

    # good practice to set the seed state, starting point
    #receive command line parameter
    filename = str(sys.argv[1])
    dataset = pd.read_csv(filename, usecols=[1], header=None)
    dataset = dataset.values #convert to the array
    dataset = dataset.astype('float32') # convert to float

    # lenth of our data set
    training_size = int(len(dataset)*0.67)
    #testing_size = len(dataset)-training_size

    # split the data set
    train, test = dataset[0:training_size,:], dataset[training_size:len(dataset),:]

    # one time step to the future
    lookback = 1
    #dataX, dataY = [], [] # create 2 empty list
    #------------------------------------------
    dataX_train, dataY_train = [], [] # create 2 empty list
    dataX_test, dataY_test = [], [] # create 2 empty list
    np.random.seed(7)
    
    for i in range(len(dataset)-lookback-1):
        a = dataset[i:(i+lookback),0]
        dataX_train.append(a)
        dataY_train.append(dataset[i+lookback,0]) # get the next value
    
        X_train, Y_train = np.array(dataX_train), np.array(dataY_train)

    for i in range(len(dataset)-lookback-1):
        a = dataset[i:(i+lookback),0]
        dataX_test.append(a)
        dataY_test.append(dataset[i+lookback,0]) # get the next value
    
        X_test, Y_test = np.array(dataX_test), np.array(dataY_test)
 
    return X_train, Y_train, X_test, Y_test

def model(X_train, Y_train, X_test, Y_test):
   
    lookback = 1
    model=Sequential()
    model.add(Dense({{choice([30,60,80])}}, input_dim=lookback, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense({{choice([10,30,60])}}, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mae'])

    model.fit(X_train, Y_train,
              batch_size={{choice([130,180])}},
              nb_epoch={{choice([130,200])}},
              verbose=2,
              validation_data=(X_test, Y_test))
    score, mae = model.evaluate(X_test, Y_test, verbose=0)
    print('Test mae:', mae)
    return {'loss': -mae, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    trials=Trials()
    best_run, best_model, space = optim.minimize(model=model,
                                                data=data,
                                                algo=tpe.suggest,
                                                max_evals=5,
                                                trials=trials,
                                                eval_space=True, 
                                                return_space=True)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print('')
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print('')
    print(filename)
    print('BATCH_SIZE =', best_run.get('batch_size'))
    print('NB_EPOCHS =', best_run.get('nb_epoch'))
    print('LAYER1 =',best_run.get('Dense'))
    print('LAYER2 =',best_run.get('Dense_1'))
    print('')

    print("python gru.py %s %d %d %d %d" %(filename, best_run.get('batch_size'), best_run.get('nb_epoch'), best_run.get('Dense'), best_run.get('Dense_1'))) 
    
    f=open("MLP_hyperparameters.txt", "a")
    f.write("\n python mlp.py %s %d %d %d %d " %(filename, best_run.get('batch_size'), best_run.get('nb_epoch'), best_run.get('Dense'), best_run.get('Dense_1'))) 
        
