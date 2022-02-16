import antropy as ant
import numpy as np

entropy_three_raw=np.load('../data/three_file_path/entropy_three_minute_data.npy')
entropy_one_raw=np.load('../data/one_file_path/entropy_one_minute_data.npy')

def entropy_cal(x):
    a=[]
    # Permutation entropy
    a.append(ant.perm_entropy(x, normalize=True))
    #print(ant.perm_entropy(x, normalize=True))
    # Spectral entropy
    a.append(ant.spectral_entropy(x, sf=100, method='welch', normalize=True))
    ##print(ant.spectral_entropy(x, sf=100, method='welch', normalize=True))
    # Singular value decomposition entropy
    a.append(ant.svd_entropy(x, normalize=True))
    #print(ant.svd_entropy(x, normalize=True))
    # Approximate entropy
    a.append(ant.app_entropy(x))
    #print(ant.app_entropy(x))
    # Sample entropy
    a.append(ant.sample_entropy(x))
    #print(ant.sample_entropy(x))
    return a


def fun(entropy_array):
    temp=[]
    ct=0
    for i in entropy_array:
        y=[]
        for j in range(0,len(i)):
            x=i[j]
            y.append(entropy_cal(x))
        temp.append(y)
        ct=ct+1
        print(ct)
        # break
    return temp

entropy_three=np.array(fun(entropy_three_raw))
entropy_one=np.array(fun(entropy_one_raw))

np.save('../data/three_file_path/entropy_three_minute_data_processed.npy',entropy_three)
np.save('../data/one_file_path/entropy_one_minute_data_processed.npy',entropy_one)

print(entropy_one.shape)
print(entropy_three.shape)