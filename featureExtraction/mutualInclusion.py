import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

# maa chuda randi saale 
# def calc_MI(x, y, bins,j,k):
#     # print(x,y)  
#     c_xy = np.histogram2d(x, y, bins)[0]
#     print(j,k,c_xy.shape)
#     g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
#     print(j,k,g)
#     mi = 0.5 * g / c_xy.sum()
#     # print(mi)
#     return mi

MI_three_raw=np.load('../data/three_file_path/MI_three_minute_data.npy')
MI_one_raw=np.load('../data/one_file_path/MI_one_minute_data.npy')

MI_three=[]
MI_one=[]
cnt=0
for i in MI_three_raw:
    temp=np.zeros((21,21))
    for j in range(0,len(i)):
        for k in range(0,len(i)):
            # print(i[0].shape,i[2].shape)
            temp[j][k]=calc_MI(i[j],i[k],21)
    MI_three.append(temp)
    print(cnt)
    cnt=cnt+1
    
cnt=0
print(np.array(MI_three).shape)
for i in MI_one_raw:
    temp=np.zeros((21,21))
    for j in range(0,len(i)):
        for k in range(0,len(i)):
            temp[j][k]=calc_MI(i[j],i[k],21)
    MI_one.append(temp)
    print(cnt)
    cnt=cnt+1

print(np.array(MI_one).shape)
MI_one=np.array(MI_one)
MI_three=np.array(MI_three)
np.save('../data/three_file_path/MI_three_minute_data_2sec.npy',MI_three)
np.save('../data/one_file_path/MI_one_minute_data_2sec.npy',MI_one)