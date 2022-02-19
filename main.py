# imports
# from ctypes.wintypes import HMODULE
import os
# from grpc import ChannelCredentials
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
# from model import neural_network_model
from kerasModel import neural_network_LSTM, neural_network_LSTM_2, neural_network_BLSTM


Channels=21
batch_size=21
time_steps = 1
features = 105

# def train_model(train_data, model = False):
#     X=np.array([i[0] for i in train_data]).reshape([-1,Channels,batch_size,1])
#     # X=np.array(input_data).reshape([-1,21,2500,1])
#     Y=[i[1] for i in train_data]
#     Y=np.array(Y)
#     print(X.shape,Y.shape)
#     if not model:
#         model = neural_network_model(Channels,batch_size)
    
#     model.fit({'input':X},{'targets':Y},n_epoch=350,show_metric=True,run_id='EEG_Classifier_plv')
#     return model


def train_model_LSTM(train_data, model = False):
    X=np.array([i[0] for i in train_data]).reshape([-1,time_steps,features])
    # X=np.array(input_data).reshape([-1,21,2500,1])
    Y=[i[1] for i in train_data]
    Y=np.array(Y)
    print(X.shape,Y.shape)
    if not model:
        model = neural_network_BLSTM(time_steps,features)
    
    model.fit(X, Y, epochs=1000)
    return model


training_data=np.load('data/training_data_entropy_1D.npy',allow_pickle=True)
model = train_model_LSTM(training_data)
model.save('model_entropy_BLSTM')

# print(training_data.shape[0])