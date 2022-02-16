from importData import hilbert_transform_data, three_file_path, one_file_path
import numpy as np

three_epochs_array=np.array([hilbert_transform_data(i,"180s") for i in three_file_path])
one_epochs_array=np.array([hilbert_transform_data(i,"60s") for i in one_file_path])


print(one_epochs_array.shape)
processed_three_epochs_array=[]
processed_one_epochs_array=[]


for i in three_epochs_array:
    for j in i:
        processed_three_epochs_array.append(j)

for i in one_epochs_array:
    for j in i:
        processed_one_epochs_array.append(j)

processed_three_epochs_array=np.array(processed_three_epochs_array)
processed_one_epochs_array=np.array(processed_one_epochs_array)
print(processed_three_epochs_array.shape,processed_one_epochs_array.shape)

np.save('data/three_file_path/hilbert_three_minute_data_2sec.npy',processed_three_epochs_array)
np.save('data/one_file_path/hilbert_one_minute_data_2sec.npy',processed_one_epochs_array)