'''该文件用于AIIA测试复赛结果'''

import pandas as pd
import numpy as np
haveRealResult = True
jieguo = pd.read_csv('dfOut_2nd.csv',encoding = 'gbk')
filename = '320data_test.csv'
if haveRealResult == True:
    filename = '320data.csv'

yuanshi = pd.read_csv(filename) #encoding = 'gbk'

loading = True
NUM = yuanshi.shape[0]
# cnt = 0
i = 0
if loading == False:
    while i < NUM:
        if i%10000 == 1:
            print(i)
        per_name_jieguo = jieguo['LC_NAME'].iloc[i]
        per_name_yuanshi = yuanshi['LC_NAME'].iloc[i]
        if per_name_jieguo == per_name_yuanshi:
            i = i + 1
        else:
            print(str(i) + ':', per_name_yuanshi, per_name_jieguo)
            jieguo.drop(index = i, inplace= True)
            jieguo.reset_index(drop=True, inplace=True)

    jieguo.reset_index(drop=True, inplace=True)

    jieguo1 = jieguo.drop(['Unnamed: 0'],axis=1,inplace=False)

    jieguo1.to_csv('dfOut_2nd_xiugai2.csv',encoding='gbk',index=True)
else:
    jieguo1 = pd.read_csv('dfOut_2nd_xiugai2.csv', encoding='gbk')

if haveRealResult == True:
    result = pd.concat([yuanshi['F0823'], jieguo1['F0823']], ignore_index=True, join='outer', axis=1)


    result = result.astype('str')
    str_nan = str(np.nan)
    result = result.replace('$null$', str_nan)
    result = result.astype('float')

    result1 = result.copy()
    result1.iloc[:,0].loc[result1.iloc[:,0]<10]=0
    result1.iloc[:,0].loc[result1.iloc[:,0]>=10]=1
    # result1 = result1.fillna(1)
    result1 = result1.dropna(axis=0,inplace=False)

    result1.reset_index(drop=True, inplace=True)

    # result1.loc[result1.iloc[:,0]==1].iloc[:,1].sum()



    TP = result1.loc[result1.iloc[:,0]==0].shape[0] -  result1.loc[result1.iloc[:,0]==0].iloc[:,1].sum() #原本正预正，复赛要求负预负
    FP = result1.loc[result1.iloc[:,0]==1].shape[0] - result1.loc[result1.iloc[:,0]==1].iloc[:,1].sum() #原本负预正，复赛要求正预负
    FN = result1.loc[result1.iloc[:,0]==0].iloc[:,1].sum()#原本正预负，复赛要求负预正

    P=TP/(TP+FP)
    R=TP/(TP+FN)
    F1=2*P*R/(P+R)






