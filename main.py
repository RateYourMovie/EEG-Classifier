# imports
from ctypes.wintypes import HMODULE
import os
from grpc import ChannelCredentials
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
from model import neural_network_model


Channels=21
batch_size=21

def train_model(train_data, model = False):
    X=np.array([i[0] for i in train_data]).reshape([-1,Channels,batch_size,1])
    # X=np.array(input_data).reshape([-1,21,2500,1])
    Y=[i[1] for i in train_data]
    Y=np.array(Y)
    print(X.shape,Y.shape)
    if not model:
        model = neural_network_model(Channels,batch_size)
    
    model.fit({'input':X},{'targets':Y},n_epoch=350,show_metric=True,run_id='EEG_Classifier_plv')
    return model


training_data=np.load('data/training_data_plv_2sec.npy',allow_pickle=True)
model = train_model(training_data)
model.save('model_plv_2sec.tflearn')

# print(training_data.shape[0])