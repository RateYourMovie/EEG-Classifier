from glob import glob
import numpy as np
import mne
import os
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from default import DATASET

# all_file_path=glob(r'C:\\Users\\goenk\\Desktop\\Machine-Learning\\EEG-Classifier\\eeg-during-mental-arithmetic-tasks-1.0.0\\eeg-during-mental-arithmetic-tasks-1.0.0\\*.edf')
all_file_path=glob(DATASET)

three_file_path=[i for i in all_file_path if '1' in i.split('_')[1]]
one_file_path=[i for i in all_file_path if '2' in i.split('_')[1]]
three_file_path.sort()
one_file_path.sort()

def read_data(file_path,time):
  data= mne.io.read_raw_edf(file_path, preload=True)
  data.set_eeg_reference()
  data.filter(l_freq=0.5, h_freq=50, filter_length=time)
  epochs=mne.make_fixed_length_epochs(data, duration=2,overlap=0)
  array=epochs.get_data()
  return array

def hilbert_transform_data (file_path,time):
  data= mne.io.read_raw_edf(file_path, preload=True)
  data.set_eeg_reference()
  data.filter(l_freq=0.5, h_freq=50, filter_length=time)
  data.apply_hilbert()
  epochs=mne.make_fixed_length_epochs(data, duration=2,overlap=0)
  array=epochs.get_data()
  return array

# three_epochs_array=np.array([hilbert_transform_data(i,"180s") for i in three_file_path])
# one_epochs_array=np.array([hilbert_transform_data(i,"60s") for i in one_file_path])


# print(one_epochs_array.shape)
# processed_three_epochs_array=[]
# processed_one_epochs_array=[]


# for i in three_epochs_array:
#     for j in i:
#         scaler = MinMaxScaler()
#         scaler.fit(j)
#         processed_three_epochs_array.append(scaler.transform(j))

# for i in one_epochs_array:
#     for j in i:
#         scaler = MinMaxScaler()
#         scaler.fit(j)
#         processed_one_epochs_array.append(scaler.transform(j))

# processed_three_epochs_array=np.array(processed_three_epochs_array)
# processed_one_epochs_array=np.array(processed_one_epochs_array)
# print(processed_three_epochs_array.shape,processed_one_epochs_array.shape)

# np.save('data/three_file_path/three_minute_data.npy',processed_three_epochs_array)
# np.save('data/one_file_path/one_minute_data.npy',processed_one_epochs_array)