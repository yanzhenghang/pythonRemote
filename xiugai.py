'''该文件用于AIIA测试初赛结果'''

import pandas as pd
import numpy as np
dfOut = pd.read_csv('dfOut.csv',encoding="gbk")#utf_8_sig
dat319 = pd.read_csv('319data.csv',encoding="utf_8_sig")#utf_8_sig
loading =True
yuanshi = dat319.copy()
jieguo = dfOut.copy()
jieguo.reset_index(drop=True, inplace=True)

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
            jieguo.drop(index = i, inplace= True)
            jieguo.reset_index(drop=True, inplace=True)
            print(str(i) + ':', per_name_yuanshi, per_name_jieguo)
    jieguo.reset_index(drop=True, inplace=True)
    jieguo1 = jieguo.drop(['Unnamed: 0'], axis=1, inplace=False)

    jieguo1.to_csv('dfOut_xiugai2.csv', encoding='gbk', index=True)
else:
    jieguo1 = pd.read_csv('dfOut_xiugai2.csv',encoding='gbk')

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



TP = result1.loc[result1.iloc[:,0]==1].iloc[:,1].sum() #正预正
FP = result1.loc[result1.iloc[:,0]==0].iloc[:,1].sum() #负预正
FN =  result1.loc[result1.iloc[:,0]==1].shape[0]  - result1.loc[result1.iloc[:,0]==1].iloc[:,1].sum()#正预负

P=TP/(TP+FP)
R=TP/(TP+FN)
F1=2*P*R/(P+R)


