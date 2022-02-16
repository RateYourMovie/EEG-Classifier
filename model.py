import tflearn
from tflearn.layers.core import input_data,fully_connected,dropout,flatten
from tflearn.layers.conv import conv_2d,max_pool_2d,conv_3d,max_pool_3d
from tflearn.layers.estimator import regression


def neural_network_model(row,col):
    network = input_data(shape = [None,row,col,1],name='input')

    

    # network = conv_2d(network,128, 5,activation = 'relu')
    # network = max_pool_2d(network,5)  

    # network = conv_2d(network,32, 5,activation = 'relu')
    # network = max_pool_2d(network,5)

    # network = conv_2d(network,64,5,activation = 'relu')
    # network = max_pool_2d(network,5)
    # network = conv_2d(network,32, 5,activation = 'relu')
    # network = max_pool_2d(network,5) 


    # network = conv_2d(network,128, 5,activation = 'relu')
    # network = max_pool_2d(network,5) 
    # network = conv_2d(network,32, 5,activation = 'relu')
    # network = max_pool_2d(network,5) 

    network = conv_2d(network,32,5,activation = 'relu')
    
    network = conv_2d(network,64,5,activation = 'relu')
    
    network = max_pool_2d(network,2)

    network = conv_2d(network,128, 3,activation = 'relu')
       
    network=flatten(network,name='Flatten')
    
    network = fully_connected(network, 64, activation = 'relu')
    
    network = dropout(network, 0.25)

    network = fully_connected(network, 16, activation = 'relu' )
    
    network = fully_connected(network, 2, activation = 'softmax' )
    
    network = regression(network, optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='targets')

    # model = tflearn.DNN(network, max_checkpoints=1, tensorboard_verbose=0)
    model = tflearn.DNN(network, checkpoint_path='/tmp/tflearn_logs/', tensorboard_verbose=0)
    return model

