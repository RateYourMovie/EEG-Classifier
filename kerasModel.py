import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional

def neural_network_LSTM(time_steps, features):
    n_outputs = 2

    model = Sequential()
    
    model.add(LSTM(128, input_shape = (time_steps, features), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(n_outputs, activation='softmax'))

    #optimizer
    opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def neural_network_LSTM_2(time_steps, features):
    n_outputs = 2
    print('The time steps are: ', time_steps, features)
    model = Sequential()
    
    model.add(LSTM(64, input_shape = (time_steps, features), activation='relu', return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))

    # model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    #optimizer
    opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def neural_network_LSTM_3(time_steps, features):
    n_outputs = 2
    print('The time steps are: ', time_steps, features)
    model = Sequential()
    
    model.add(LSTM(64, input_shape = (time_steps, features), activation='relu', return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu'))

    # model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    #optimizer
    opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def neural_network_BLSTM(time_steps, features):
    n_outputs = 2
    print('The time steps are: ', time_steps, features)
    model = Sequential()
    
    model.add(Bidirectional(LSTM(64, input_shape = (time_steps, features), activation='relu', return_sequences=True)))
    # model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32, activation='relu')))

    # model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    #optimizer
    opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model