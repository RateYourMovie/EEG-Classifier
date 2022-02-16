import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

hilbert_three=np.load('../data/three_file_path/hilbert_three_minute_data_2sec.npy')
hilbert_one=np.load('../data/one_file_path/hilbert_one_minute_data_2sec.npy')

def phase_locking_value(theta1, theta2):
    complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
    plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return plv

plv_three=[]
plv_one=[]

for i in hilbert_three:
    temp=np.zeros((21,21))
    for j in range(0,len(i)):
        for k in range(0,len(i)):
            temp[j][k]=(phase_locking_value(i[j],i[k]))
    plv_three.append(temp)

for i in hilbert_one:
    temp=np.zeros((21,21))
    for j in range(0,len(i)):
        for k in range(0,len(i)):
            temp[j][k]=(phase_locking_value(i[j],i[k]))
    plv_one.append(temp)

plv_one=np.array(plv_one)
plv_three=np.array(plv_three)

processed_three_epochs_array=[]
processed_one_epochs_array=[]

for i in plv_three:
    scaler = MinMaxScaler()
    scaler.fit(i)
    processed_three_epochs_array.append(scaler.transform(i))

for i in plv_one:
    scaler = MinMaxScaler()
    scaler.fit(i)
    processed_one_epochs_array.append(scaler.transform(i))

for i in plv_one:
    scaler = MinMaxScaler()
    scaler.fit(i)
    processed_one_epochs_array.append(scaler.transform(i))

for i in plv_one:
    scaler = MinMaxScaler()
    scaler.fit(i)
    processed_one_epochs_array.append(scaler.transform(i))

processed_three_epochs_array=np.array(processed_three_epochs_array)
processed_one_epochs_array=np.array(processed_one_epochs_array)
print(processed_three_epochs_array.shape,processed_one_epochs_array.shape)
# print(plv_three.shape)

np.save('../data/three_file_path/plv_three_minute_data_2sec.npy',processed_three_epochs_array)
np.save('../data/one_file_path/plv_one_minute_data_2sec.npy',processed_one_epochs_array)