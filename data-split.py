import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle


result_array_three=[]
result_array_one=[]
three_minute_data=np.load('./data/three_file_path/MI_three_minute_data_2sec.npy')
one_minute_data=np.load('./data/one_file_path/MI_one_minute_data_2sec.npy')

# print(three_minute_data.shape)
# print(one_minute_data.shape)

for i in range(0,three_minute_data.shape[0]):
    result_array_three.append([1,0])


for i in range(0,one_minute_data.shape[0]):
    result_array_one.append([0,1])



X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
three_minute_data, result_array_three, test_size=0.3, random_state=0)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
one_minute_data, result_array_one, test_size=0.3, random_state=0)

def fun(x1,x2):
    return np.append(x1,x2,axis=0)

X_train=fun(X_train_1,X_train_3)
y_train=fun(y_train_1,y_train_3)
X_test=fun(X_test_1,X_test_3)
y_test=fun(y_test_1,y_test_3)

training_data=[]
testing_data=[]
for i in range(0,len(X_train)):
    training_data.append([X_train[i],y_train[i]])

for i in range(0,len(X_test)):
    testing_data.append([X_test[i],y_test[i]])


shuffle(training_data)
training_data=np.array(training_data)
testing_data=np.array(testing_data)

# print(y_train_3.shape)
print(training_data.shape,testing_data.shape)
np.save('data/training_data_MI_Matrix.npy',training_data)
np.save('data/testing_data_MI_Matrix.npy',testing_data)
