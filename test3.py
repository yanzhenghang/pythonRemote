import pandas as pd
import numpy as np
import gc
print('begin!')
remotePath = '/home/zhangyifan/yanzhenghang/pythonProject/'
# mat = pd.read_csv(remotePath+'mergeToDo_replaced_dropnad_merged_addftured.csv')
# mat = mat.drop(['Unnamed: 0'],axis=1)
#
# tmp_i = mat.shape[0]//2
# mat.loc[tmp_i:mat.shape[0],:].to_csv(remotePath+'after_half.csv', encoding="utf_8_sig", index=False,sep=',')
# gc.collect()
# mat = mat.loc[0:tmp_i,:]
# mat.to_csv(remotePath+'before_half0.csv', encoding="utf_8_sig", index=False,sep=',')
# gc.collect()
# print('after_half.csv  finished!')
mat = pd.read_csv(remotePath+'before_half0.csv')

def half_address(mat):
    mat = mat.astype('str')
    str_nan = str(np.nan)
    mat = mat.replace('$null$', str_nan)
    print('replace finished!')

    columns_tmp = mat.columns[1:(mat.shape[1])]
    mat[columns_tmp] = mat[columns_tmp].astype('float')

    thresh = int(mat.shape[0] * 0.95)
    mat.dropna(thresh=thresh, inplace=True, axis=1)
    mat.dropna(inplace=True, axis=0)
    print('dropna finished!')
    return mat


mat = half_address(mat)
mat_before_colums = mat.columns
mat.to_csv(remotePath+'before_half.csv', encoding="utf_8_sig", index=False,sep=',')
print('mat_before  saved!')
del mat
gc.collect()
mat = pd.read_csv(remotePath+'after_half.csv')
mat = mat[mat_before_colums]
mat = mat.astype('str')
str_nan = str(np.nan)
mat = mat.replace('$null$', str_nan)
print('replace_2nd finished!')
mat.to_csv(remotePath+'after_half_plus.csv', encoding="utf_8_sig", index=False,sep=',')

gc.collect()

mat0=pd.read_csv(remotePath+'before_half.csv')
mat0 = mat0.append(mat)
del mat
gc.collect()
mat = mat0
mat = half_address(mat)
print('all finished, waiting to save!')
# for m in mat1_colums:
#     if m == ''
#     mat[m] = mat[m].fillna(mat[m].mean())
#
#     columns_tmp = mat.columns[1:(mat.shape[1])]
#     mat[columns_tmp] = mat[columns_tmp].astype('float')

mat.to_csv(remotePath+'FINAL_OUT_ALL.csv',encoding="utf_8_sig",index=False, sep=',')
print('FINAL_OUT_ALL.csv finished!')
def sample_ramdom(Data, nofSamples = 1,filename = 'sample_ramdom.csv'):
    import random
    random.seed(5)
    ind = random.sample(range(0,Data.shape[0]),nofSamples)
    out = Data.loc[ind,:]
    out.to_csv(filename[0:-4]+'_'+str(nofSamples)+'_sampled.csv', encoding="utf_8_sig", index=False)
sample_ramdom(mat, 120000,filename=remotePath+'FINAL_OUT_120000_V1.csv')
print('FINAL_OUT_120000_V1.csv finished!')
