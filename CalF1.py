import pandas as pd
import numpy as np
dat320 = pd.read_csv('320data_test.csv',encoding="gbk")#utf_8_sig
dfOut_2nd = pd.read_csv('dfOut_2nd.csv',encoding="gbk")#utf_8_sig    dfOut_2nd_xiugai

yuanshi = dat320.copy()
jieguo = dfOut_2nd.copy()
NUM = yuanshi.shape[0]
# cnt = 0
i = 0
while i < NUM:
    if i%1000 == 1:
        print(i)
    per_name_jieguo = jieguo['LC_NAME'].iloc[i]
    per_name_yuanshi = yuanshi['LC_NAME'].iloc[i]
    if per_name_jieguo == per_name_yuanshi:
        i = i + 1
        continue

    else:
        print(str(i) + ':', per_name_yuanshi, per_name_jieguo)
        jieguo.drop(index = i, inplace= True)
        #删除jieguo的这一行

jieguo.reset_index(drop=True, inplace=True)