from model import neural_network_model
import numpy as np

def test_model(test_data, model = False):
    X=np.array([i[0] for i in test_data]).reshape([-1,21,2500,1])
    # X=np.array(input_data).reshape([-1,21,2500,1])
    Y=[i[1] for i in test_data]
    Y=np.array(Y)
    # print(X.shape,Y.shape)
    if not model:
        model = neural_network_model(21,2500)
    
    model.load('model_1.tflearn')
    return model.evaluate(X,Y)

testing_data=np.load('data/testing_data_1.npy',allow_pickle=True)

score = test_model(testing_data)
print("The final Prediction is: ",score)