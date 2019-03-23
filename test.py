# import pandas as pd
# import numpy as np
# mat = pd.read_csv('mergeToDo.csv')
# print('1')
# thresh = int(mat.shape[0] * 0.98)
# mat.dropna(thresh=thresh, inplace=True,axis=1)
# print('2')
# thresh = int(mat.shape[1] * 0.95)
# mat.dropna(thresh=thresh, inplace=True, axis=0)
# print('3')
# mat.to_csv('mergeToDo2.csv',index=False, sep=',')
import pandas as pd
import numpy as np
print('begin!')
remotePath = '/home/zhangyifan/yanzhenghang/pythonProject/'
mat = pd.read_csv(remotePath+'mergeToDo2.csv')
print('read_finished!')
# mat = mat.astype('str')
# mat = mat.replace('$null$', str(np.nan))
# print('replaced!')
# columns_tmp = mat.columns[2:(mat.shape[1])]
# mat = mat[columns_tmp].astype('float')
# mat.to_csv('mergeToDo_replaced.csv',index=False, sep=',')
# print('mergeToDo_replaced.csv-----readfinished')
# thresh = int(mat.shape[0] * 0.90)
# mat.dropna(thresh=thresh, inplace=True,axis=1)
# thresh = int(mat.shape[1] * 0.90)
# mat.dropna(thresh=thresh, inplace=True, axis=0)
# print('dropna2nd_col_row_finished')
# mat.to_csv('mergeToDo_replaced_dropnad.csv',index=False, sep=',')
# print('mergeToDo_replaced_dropnad  saved!!!')

AB = pd.read_csv(remotePath +'final_merge_data_V1.csv')
def add_feature(data):
    #1st
    data['IS_WEEKEND'] = data['S_DAY']
    data['IS_WEEKEND'] = data['IS_WEEKEND'].astype('int')
    tmp = ( (data['IS_WEEKEND']- 12)%7 <= 4)
    data['IS_WEEKEND'].loc[tmp] = -1
    data['IS_WEEKEND'].loc[~tmp] = 1

    #2nd
    data['WEEK_DAY'] = data['S_DAY']
    data['WEEK_DAY'] = data['WEEK_DAY'].astype('int')
    tmp_day = []
    for i in range(0,7):
        tmp =  ((data['WEEK_DAY'] - 12) % 7 == i)
        tmp_day.append(tmp)
    for m in range(0,7):
        data['WEEK_DAY'].loc[tmp_day[m]] = m+1

    return data
def merge_all(D_big, D_little, filename = 'merge_all.csv',needsave = True):
    data = pd.merge(D_big, D_little, left_on='LC_NAME', right_on = '小区别名', how='inner')
    data = data.drop(['小区别名','RELATED_ENODEB','小区标识'],axis = 1)
    data.to_csv(filename,index=False, sep=',')
    return data
def sample_ramdom(Data, nofSamples = 1,filename = 'sample_ramdom.csv'):
    import random
    random.seed(5)
    ind = random.sample(range(0,Data.shape[0]),nofSamples)
    out = Data.loc[ind,:]
    out.to_csv(filename[0:-4]+'_'+str(nofSamples)+'_sampled.csv', encoding="utf_8_sig", index=False)

ABC = merge_all(mat,AB, filename='mergeToDo_replaced_dropnad_merged.csv')
print('mergeToDo_replaced_dropnad_merged.csv  finished!')
ABC = add_feature(ABC)
ABC.reset_index(drop=True, inplace=True)
ABC.to_csv('mergeToDo_replaced_dropnad_merged_addftured.csv',index=False, sep=',')
print('mergeToDo_replaced_dropnad_merged_addftured.csv  finished!')
sample_ramdom(ABC, 60000,filename='mergeToDo_replaced_dropnad_merged_addftured_sampled.csv')
print('mergeToDo_replaced_dropnad_merged_addftured_sampled.csv  finished!')



