from glob import glob
import numpy as np
import mne
import os
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from importData import read_data,three_file_path,one_file_path

three_epochs_array=np.array([read_data(i,"180s") for i in three_file_path])
one_epochs_array=np.array([read_data(i,"60s") for i in one_file_path])


print(one_epochs_array.shape)
processed_three_epochs_array=[]
processed_one_epochs_array=[]


for i in three_epochs_array:
    for j in i:
        scaler = MinMaxScaler()
        scaler.fit(j)
        processed_three_epochs_array.append(scaler.transform(j))

for i in one_epochs_array:
    for j in i:
        scaler = MinMaxScaler()
        scaler.fit(j)
        processed_one_epochs_array.append(scaler.transform(j))

processed_three_epochs_array=np.array(processed_three_epochs_array)
processed_one_epochs_array=np.array(processed_one_epochs_array)
print(processed_three_epochs_array.shape,processed_one_epochs_array.shape)

np.save('data/three_file_path/three_minute_data.npy',processed_three_epochs_array)
np.save('data/one_file_path/one_minute_data.npy',processed_one_epochs_array)