import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model


def test_model(test_data, model = False):
    X=np.array([i[0] for i in test_data]).reshape([-1,1,105])
    # X=np.array(input_data).reshape([-1,21,2500,1])
    Y=[i[1] for i in test_data]
    Y=np.array(Y)
    print(X.shape,Y.shape)
    if not model:
        model = load_model('model_entropy_BLSTM')

    return model.evaluate(X,Y, verbose=2)

testing_data=np.load('data/testing_data_entropy_1D.npy',allow_pickle=True)

score = test_model(testing_data)
print('The accuracy is: ', score)
