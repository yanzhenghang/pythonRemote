import pandas as pd
import numpy as np
def preprocess_cellBaseData(filename = 'F:\小区基础数据.xlsx', needsave = True):

    '''[小区别名, 小区标识, 所属eNodeB标识, 所属行政区域类型, 天线数, 双工方式, 采用的cp类型,
        子帧配置类型, 特殊子帧配置类型, 是否为RRU小区, 上行频点, 下行频点, 小区配置的载频发射功率,
         参考信号（RS）的每RE平均发射功率, A类符号功率比值, B类符号功率比值, 广播信道功率,
          最大传输功率, 运行状态, 小区覆盖类型, 小区覆盖范围, 中心载频的信道号, 带宽, 下行循环前缀长度,
           上行循环前缀长度, 上行带宽, 下行带宽, 小区激活状态, 高速小区指示, 发送和接收模式, 工作模式,
            小区是否闭塞, CSFB回落网络优先级, 是否载波聚合小区, 载波聚合类型, 载波聚合频段组合]'''
    # 80784 * 36

    # 继续删除上行CP和下行CP的两个特征；CSFB回落网络优先级特征删除；
    # 删除了特征的候选级小于2的特征，处理了特征之间有矛盾的数据

    # 删除的特征：小区别名, 以及上述的CP类型等特征的删除
    #filename = 'F:\小区基础数据.xlsx'
    '''columName_cellBaseData = ['小区标识', '所属eNodeB标识', '所属行政区域类型', '天线数', '双工方式', '是否为RRU小区', '上行频点', '下行频点',
                              '小区配置的载频发射功率', '参考信号（RS）的每RE平均发射功率', 'A类符号功率比值', 'B类符号功率比值', '广播信道功率', '最大传输功率', '运行状态',
                              '小区覆盖类型', '小区覆盖范围', '中心载频的信道号', '带宽', '上行带宽', '下行带宽', '小区激活状态', '高速小区指示', '发送和接收模式',
                              '工作模式', '小区是否闭塞', '是否载波聚合小区', '载波聚合类型', '载波聚合频段组合']'''
    print('正在读入：'+filename)
    df = pd.read_excel(filename, sheet_name=0, header=0)#names = columName_cellBaseData
    print('读入完成！')

    #去除重复项
    df.drop_duplicates(keep='first', inplace=True)

    #删除无效特征  df.drop(['B', 'C'], axis=1)
    df.drop(columns=['采用的cp类型','子帧配置类型','特殊子帧配置类型','下行循环前缀长度','上行循环前缀长度', 'CSFB回落网络优先级'], inplace=True)



    #对某些特征的NA取众数，平均值


    # 删除特征的数据存在错误的行
    #thresh = int(df.shape[1] * 0.6)
    #df.dropna(thresh=thresh, inplace=True)

    df = df.dropna()

    df.发送和接收模式 = df.发送和接收模式.str.replace('两发两收', '2T2R(两发两收)')
    df.发送和接收模式 = df.发送和接收模式.str.replace('一发一收', '1T1R(一发一收)')
    df.发送和接收模式 = df.发送和接收模式.str.replace('2T2R(2T2R(两发两收))', '2T2R(两发两收)')
    df.发送和接收模式 = df.发送和接收模式.str.replace('1T1R(1T1R(一发一收))', '1T1R(一发一收)')

    df = df[~df['发送和接收模式'].isin(['四发四收'])]



    df = datatype_transfer_for_cellbase(df, needsave)
    #df.loc[:, ['天线数', '带宽']] = df.loc[:, ['天线数', '带宽']].astype('float')
    #df.loc[:, ['天线数', '带宽']].dtypes

    #df['小区标识'] = df['小区标识'].str.replace('.', '')
    df['小区标识'] = df['小区标识'].str.replace(',', '.')


    #df = df.astype('float64')
    #df['小区标识'] = df['小区标识'].astype(dtype='int64')

    df.reset_index(drop=True, inplace=True)
    # df = pd.read_excel(filename, header=[0, 1], index_col=[0, 1])
    if needsave == True:
        df.to_csv(filename[0:-5]+'_preprocessed.csv',encoding="utf_8_sig",index=False)
    return df
def preprocess_cellAntennaData(filename = 'F:\小区天线参数数据.xlsx', needsave = True):
    '''columName_cellBaseData = ['']'''
    print('正在读入：' + filename)
    df = pd.read_excel(filename, sheet_name=0, header=0)  # names=columName_cellBaseData
    print('读入完成！')

    # 去除重复项
    df.drop_duplicates(keep='first', inplace=True)

    df.drop(columns=['天线类型','美化类型','塔桅类型'], inplace=True,axis= 1)

    tmp = df.发送和接收模式.value_counts().index
    df.发送和接收模式 = df.发送和接收模式.replace(tmp, list(range(1, len(tmp) + 1)))

    df = df.dropna()

    #df['所属小区标识'] = df['所属小区标识'].str.replace('.', '')
    df['所属小区标识'] = df['所属小区标识'].str.replace('，', '.')

    #df = df.astype('float64')
    #df['所属小区标识'] = df['所属小区标识'].astype(dtype='int64')

    df.reset_index(drop=True, inplace=True)

    # df = pd.read_excel(filename, header=[0, 1], index_col=[0, 1])
    if needsave == True:
        df.to_csv(filename[0:-5] + '_preprocessed.csv',encoding="utf_8_sig",index=False)
    return df
def merge_cellBaseData_cellAntennaData(Base, Ante, filename = 'F:/final_merge_data_V1.csv',needsave = True): #传递参数类型为pd.read_excel的返回对象


    #basetmp = Base.小区标识
    #atmp = Ante.小区标识

    #basetmp = basetmp.str.replace('.', '').astype(dtype='int64')
    #atmp = atmp.str.replace('.', '').astype(dtype='int64')

    # Base['小区标识'] = Base['小区标识'].str.replace('.', '')
    # Base['小区标识'] = Base['小区标识'].str.replace(',', '').astype(dtype='int64')
    # Ante['所属小区标识'] = Ante['所属小区标识'].str.replace('.', '')
    # Ante['所属小区标识'] = Ante['所属小区标识'].str.replace('，', '').astype(dtype='int64')

    data = pd.merge(Base,Ante,left_on='小区标识',right_on='所属小区标识',how='inner')
    data = data.drop('所属小区标识',axis = 1)

    if needsave == True:
        data.to_csv(filename,encoding="utf_8_sig",index=False)
    print('Base和Antenna数据融合完成！')
    return  data







    #pass
def datatype_transfer_for_cellbase(csvdata,needsave=True):
    tmp = csvdata.发送和接收模式.value_counts().index
    csvdata.发送和接收模式 = csvdata.发送和接收模式.replace(tmp, list(range(1, len(tmp)+1) ))

    #csvdata.小区标识
    #csvdata.所属eNodeB标识 = csvdata.所属eNodeB标识.astype(dtype='int64')
    tmp = csvdata.所属行政区域类型.value_counts().index
    csvdata.所属行政区域类型 = csvdata.所属行政区域类型.replace(tmp, list(range(1, len(tmp)+1) ))
    #csvdata.天线数 = csvdata.天线数.astype(dtype='int64')
    tmp = csvdata.双工方式.value_counts().index
    csvdata.双工方式 = csvdata.双工方式.replace(tmp,list(range(1, len(tmp)+1) ))
    tmp = csvdata.是否为RRU小区.value_counts().index
    csvdata.是否为RRU小区 = csvdata.是否为RRU小区.replace(tmp,list(range(1, len(tmp)+1) ))
    #csvdata.上行频点 = csvdata.上行频点.astype(dtype='int64')
    #csvdata.下行频点 =csvdata.下行频点.astype(dtype='int64')
    #csvdata.小区配置的载频发射功率 =csvdata.小区配置的载频发射功率.astype()
    #。。。。。。。。。
    tmp = csvdata.运行状态.value_counts().index
    csvdata.运行状态 = csvdata.运行状态.replace(tmp,list(range(1, len(tmp)+1) ))
    tmp = csvdata.小区覆盖类型.value_counts().index
    csvdata.小区覆盖类型 = csvdata.小区覆盖类型.replace(tmp,list(range(1, len(tmp)+1) ))
    tmp = csvdata.小区覆盖范围.value_counts().index
    csvdata.小区覆盖范围 = csvdata.小区覆盖范围.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.小区激活状态.value_counts().index
    csvdata.小区激活状态 = csvdata.小区激活状态.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.高速小区指示.value_counts().index
    csvdata.高速小区指示 = csvdata.高速小区指示.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.工作模式.value_counts().index
    csvdata.工作模式 = csvdata.工作模式.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.小区是否闭塞.value_counts().index
    csvdata.小区是否闭塞 = csvdata.小区是否闭塞.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.是否载波聚合小区.value_counts().index
    csvdata.是否载波聚合小区 = csvdata.是否载波聚合小区.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.载波聚合类型.value_counts().index
    csvdata.载波聚合类型 = csvdata.载波聚合类型.replace(tmp,list(range(1, len(tmp)+1) ))
    tmp = csvdata.载波聚合频段组合.value_counts().index
    csvdata.载波聚合频段组合 = csvdata.载波聚合频段组合.replace(tmp,list(range(1, len(tmp)+1) ) )
    print('小区基站数据量化完成！')
    return csvdata
def excelinfo_to_txt(df,txt_name = 'F:/data.txt'):
    '''
    :param df: 是 pandas 库的读入数据
    :param txt_name: txt保存路径
    :return: None
    '''
    str_out = ''
    for i in range(0, len(df.columns)):
        data_txt = df[df.columns[i]].value_counts()
        data_txt = str(data_txt)
        str_out = str_out + '\n-----------------------------------\n' + data_txt
        print('-----------------------------------')
        print(data_txt)
    print(str_out)
    f = open(txt_name, 'a') #a表示连续写入
    f.write(str_out)
def process_bigData(mode = -1,needsave = True):
    '''
    注释中的代码适用于Python 2
    mat312 = pd.read_csv('mat312.csv')
print mat312.shape
##读取csv文件

str_out = ''
for i in range(0, len(mat312.columns)):
    data_txt = mat312[mat312.columns[i]].value_counts()
    data_txt = str(data_txt)
    str_out = str_out + '\n-----------------------------------\n' + data_txt
    print(data_txt)
pd.set_option('display.max_rows',60)
print(str_out)
f = open("mat4.txt", 'a')
f.write(str_out)
##对该csv文件的各列进行初步统计，分析选出null值较多的行删除

test = mat312['F0823'].astype('string')
mat312 = mat312[ ~ test.str.contains('null') ]
mat312['F0823'].astype('float')
##对某些列的null值行进行删除，列名需要手动输入

mat313 = pd.read_csv('mat313.csv')
print mat313.shape
mat314 = pd.read_csv('mat314.csv')
print mat314.shape
mat315 = pd.read_csv('mat315.csv')
print mat315.shape
mat316 = pd.read_csv('mat316.csv')
print mat316.shape
mat317 = pd.read_csv('mat317.csv')
print mat317.shape
mat318 = pd.read_csv('mat318.csv')
print mat318.shape
##对其他csv文件读取，再次进行上述一步操作

mat=mat312.append([mat313, mat314, mat315, mat316, mat317, mat318])
print mat.shape
##对处理后的7个csv文件进行合并

mat1 = mat.dropna(axis=1)
print mat1.shape
##删除有NA的列

str_out = ''
for i in range(0, len(mat1.columns)):
    data_txt = mat312[mat1.columns[i]].value_counts()
    data_txt = str(data_txt)
    str_out = str_out + '\n-----------------------------------\n' + data_txt
    print(data_txt)
pd.set_option('display.max_rows',60)
print(str_out)
f = open("mat_analyse.txt", 'a')
f.write(str_out)
##对整体dataframe进行分析，分别找到应删除的特征列和应处理的特征列

mat = mat1.drop(['LC_NAME','START_TIME','S_MONTH','S_WEEK','F0057','F0080','F0125','F0190','F0191','F0192','F0202','F0272','F0274','F0334','F0335','F0422','F0423','F0510','F0514','F0517','F0707','F0740','F0766','F0822'], axis=1)
##删除无用列

test = mat['F0147'].astype('string')
print test
#test=chunk()['F0817'].fillna('NA')
test.loc[test.str.contains('$null$')]= 0
#print test
test1=test.replace(0,np.nan)
print test1
test1=test1.astype('float')
test2 = test1.fillna(test1.mean())
#test2 = test1.fillna(test1.median())
print test2
mat['F0147'] = test2
print mat['F0147']
##对特定列的null值进行平均值填充或中值填充或0填充

mat1=mat.drop(['Unnamed: 0'], axis = 1)
mat1=mat.drop(['Unnamed: 0.1'], axis = 1)
mat1=mat.drop(['Unnamed: 0.1.1'], axis = 1)
mat1=mat.drop(['Unnamed: 0.1.1.1'], axis = 1)
##删除生成的无用列

pd.set_option('display.max_rows',3000)
mat1.F0823.value_counts( )
##可对某列再次进行分析，第一行解决了print中无法全部显示的问题

#mat1 = mat1.astype('string')
mat1['F0823'].loc[mat1['F0823']<10.0] = -1
mat1['F0823'].loc[mat1['F0823']>=10.0] = 1
mat1.sort_index()
print mat1
##对标签列进行处理，根据题目要求，大于等于10的替换成1，小于10的替换成-1

print mat1.shape
mat1.to_csv("mat.csv",index=False ,sep=',')

    :return:
    '''
    mat = 0
    #mode:  #等于1表示，正常模式；等于-1表示测试模式
    if mode == 1:
        NofFiles = 7
        print('读取开始！')
        mat = pd.read_csv(remotePath + '312data.csv',encoding='gbk')

        test = mat['F0823'].astype('str')
        mat = mat[~ test.str.contains('null')]
        mat['F0823'] = mat['F0823'].astype('float')
        test = mat['RELATED_ENODEB'].astype('str')
        mat = mat[~ test.str.contains('null')]

        thresh = int(mat.shape[1] * 0.85)
        mat.dropna(thresh=thresh, inplace=True)

        mat = mat.drop(
            ['START_TIME', 'S_MONTH', 'S_WEEK', 'F0057', 'F0080', 'F0125', 'F0190', 'F0191', 'F0192', 'F0202',
             'F0272', 'F0274', 'F0334', 'F0335', 'F0422', 'F0423', 'F0510', 'F0514', 'F0517', 'F0707', 'F0740', 'F0766',
             'F0822'], axis=1)

        print('312data.csv')
        for i in range(1,NofFiles):
            print('读取开始！')
            tmp = i+2
            mat_tmp = pd.read_csv(remotePath + '31' + str(tmp) +'data'+ '.csv',encoding='gbk')
            # 为了减小存储
            test = mat_tmp['F0823'].astype('str')
            mat_tmp = mat_tmp[~ test.str.contains('null')]
            mat_tmp['F0823'] = mat_tmp['F0823'].astype('float')
            test = mat_tmp['RELATED_ENODEB'].astype('str')
            mat_tmp = mat_tmp[~ test.str.contains('null')]
            thresh = int(mat_tmp.shape[1] * 0.85)
            mat_tmp.dropna(thresh=thresh, inplace=True)

            mat_tmp = mat_tmp.drop(
                ['START_TIME', 'S_MONTH', 'S_WEEK', 'F0057', 'F0080', 'F0125', 'F0190', 'F0191', 'F0192', 'F0202',
                 'F0272', 'F0274', 'F0334', 'F0335', 'F0422', 'F0423', 'F0510', 'F0514', 'F0517', 'F0707', 'F0740',
                 'F0766',
                 'F0822'], axis=1)


            mat =  mat.append(mat_tmp)
            print('31' + str(tmp) +'data'+ '.csv')
    elif mode == -1:
        filename = 'mat_test_sampled.csv'
        mat = pd.read_csv(filename,encoding='gbk')

    # mat312 = pd.read_csv('mat312.csv')
    # #print mat312.shape
    # ##读取csv文件
    # mat313 = pd.read_csv('mat313.csv')
    # #print mat313.shape
    # mat314 = pd.read_csv('mat314.csv')
    # #print mat314.shape
    # mat315 = pd.read_csv('mat315.csv')
    # #print mat315.shape
    # mat316 = pd.read_csv('mat316.csv')
    # #print mat316.shape
    # mat317 = pd.read_csv('mat317.csv')
    # #print mat317.shape
    # mat318 = pd.read_csv('mat318.csv')
    # #print mat318.shape
    # ##对其他csv文件读取，再次进行上述一步操作


    #pd.set_option('display.max_rows', 60)
    #excelinfo_to_txt(mat, txt_name='F:/bigDataInfo_mat.txt')


    ##对该csv文件的各列进行初步统计，分析选出null值较多的行删除
    # test = mat['F0823'].astype('str')
    # mat = mat[~ test.str.contains('null')]
    # mat['F0823'] = mat['F0823'].astype('float')
    # ##对某些列的null值行进行删除，列名需要手动输入
    #
    # test = mat['RELATED_ENODEB'].astype('str')
    # mat = mat[~ test.str.contains('null')]

    #删除整行为NA的行
    #mat.dropna(axis=0, how='all', inplace=True)
    #删除正行NA较多的行


    #删除列

    f = open('beforeDropna.txt', 'a')  # a表示连续写入
    str_tmp0 = ''
    for tmp_j in mat.columns:
        str_tmp0 = str_tmp0 + '\',\'' + tmp_j
    f.write(str_tmp0)





    # mat['F0823'].dropna(axis=0)
    # mat = mat[]
    #     mat['RELATED_ENODEB'].dropna(axis=0, inplace=True,)



    # mat = mat312.append([mat313, mat314, mat315, mat316, mat317, mat318])
    # print mat.shape
    ##对处理后的7个csv文件进行合并

    #mat = mat.dropna(axis=1)
    #print mat.shape
    ##删除有NA的列

    # excelinfo_to_txt(mat, txt_name='F:/bigDataInfo_mat1.txt')
    ##对整体dataframe进行分析，分别找到应删除的特征列和应处理的特征列



    # mat = mat.drop(
    #     ['F0011','F0014','F0015','F0078','F0127','F0137','F0147','F0268','F0269','F0270','F0352','F0355','F0356','F0443','F0442',
    #      'F0429','F0703','F0780','F0790','F0800','F0808','F0809','F0810','F0811','F0812','F0813',
    #      'F0818','F0821','F0824','F0825'], axis=1)

    thresh = int(mat.shape[0] * 0.98)
    mat.dropna(thresh=thresh, inplace=True,axis=1)

    thresh = int(mat.shape[1] * 0.95)
    mat.dropna(thresh=thresh, inplace=True, axis=0)

    mat = mat.astype('str')
    str_nan = str(np.nan)
    mat = mat.replace('$null$', str_nan)
    #mat = mat.astype('float')
    columns_tmp = mat.columns[1:(mat.shape[1])]
    mat[columns_tmp] = mat[columns_tmp].astype('float')

    thresh = int(mat.shape[0] * 0.9)
    mat.dropna(thresh=thresh, inplace=True, axis=1)







    #mat.loc[mat[m].str.contains('null')] = np.nan


    tmp = -1
    tmp_columns = mat.columns
    for m in tmp_columns:
        tmp = tmp + 1
        if tmp == 0 or tmp == 1:
            continue
        mat[m] = mat[m].astype('str')
        mat[m].loc[mat[m].str.contains('null')] = np.nan
        #mat[m] = mat[m].replace(-3.1415, np.nan)
        mat[m] = mat[m].astype('float32')

        thresh = int(mat.shape[0] * 0.85)
        if mat[m].notnull().sum()<thresh:
            mat = mat.drop([m], axis=1)
            continue

        mat[m] = mat[m].fillna(mat[m].mean())
        print(m)

    f = open('afterDropna.txt', 'a')  # a表示连续写入
    str_tmp = ''
    for tmp_i in mat.columns:
        str_tmp = str_tmp + '\',\'' + tmp_i
    f.write(str_tmp)




    # test.loc[test.str.contains('$null$')] = 0
    # test.loc[test.str.contains('$null$')] = 0
    # ##删除无用列
    #
    # test = mat['F0147'].astype('string')
    #
    # #print test
    # # test=chunk()['F0817'].fillna('NA')
    # test.loc[test.str.contains('$null$')] = 0
    # # print test
    #
    #
    # test1 = test.replace(0, np.nan)
    # print test1
    # test1 = test1.astype('float')
    # test2 = test1.fillna(test1.mean())
    # # test2 = test1.fillna(test1.median())
    # print test2
    # mat['F0147'] = test2
    # print mat['F0147']
    # ##对特定列的null值进行平均值填充或中值填充或0填充
    #
    # # mat1 = mat.drop(['Unnamed: 0'], axis=1)
    # # mat1 = mat.drop(['Unnamed: 0.1'], axis=1)
    # # mat1 = mat.drop(['Unnamed: 0.1.1'], axis=1)
    # # mat1 = mat.drop(['Unnamed: 0.1.1.1'], axis=1)
    # ##删除生成的无用列
    #
    # pd.set_option('display.max_rows', 3000)
    # mat1.F0823.value_counts()
    # ##可对某列再次进行分析，第一行解决了print中无法全部显示的问题

    # mat1 = mat1.astype('string')

    mat.reset_index(drop=True, inplace=True)
    mat.to_csv("mat_continuous.csv", index=False, sep=',')
    mat2 = mat

    mat2['F0823'].loc[mat2['F0823'] < 10.0] = -1
    mat2['F0823'].loc[mat2['F0823'] >= 10.0] = 1
    #mat.sort_index()
    #print mat1
    ##对标签列进行处理，根据题目要求，大于等于10的替换成1，小于10的替换成-1

    mat2.to_csv("mat_discrete.csv", index=False, sep=',')
    return mat, mat2
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

def test():
    filename = 'F:\小区天线参数数据.xlsx'
    print('正在读入：' + filename)
    df = pd.read_excel(filename, sheet_name=0, header=0)  # names=columName_cellBaseData
    print('读入完成！')
    mat = df
    tmp = -1
    for m in mat.columns:
        tmp = tmp + 1
        if tmp == 0 or tmp == 1:
            continue
        if tmp > 3:
            break
        mat[m] = mat[m].astype('str')
        mat[m].loc[mat[m].str.contains('$null$')] = -3.1415
        mat[m] = mat[m].replace(-3.1415, np.nan)
        mat[m] = mat[m].astype('float')
        mat[m] = mat[m].fillna(mat[m].mean())



def multi_pro(mat):
    from multiprocessing import Pool
    import os

    # def long_time_task(name):
    #     print('Run task %s (%s)...' % (name, os.getpid()))
    #     start = time.time()
    #     time.sleep(random.random() * 3)
    #     end = time.time()
    #     print('Task %s runs %0.2f seconds.' % (name, (end - start)))


    print('Parent process %s.' % os.getpid())
    p = Pool(19)
    # for i in range(20):
    #     p.apply_async(long_time_task, args=(i,))
    fenpei(mat, 20, p)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')





def fenpei(mat,nofgroups,pool):
    tmp = mat.shape[1]
    group_len = (tmp//nofgroups)+1

    tmp_columns = mat.columns
    for i in range(0,nofgroups):
        if i == (nofgroups-1):
            tmp1 = tmp_columns[(group_len * i + 0):-1]
        else:
            tmp1 = tmp_columns[ (group_len*i+0):(group_len*(i+1)) ]
        pool.apply_async(NA2MEAN, args=(mat.loc[:,tmp1],))

def NA2MEAN(mat_sub):
    # tmp = -1
    tmp_columns = mat_sub.columns

    mat = mat_sub

    for m in tmp_columns:
        if m == 'LC_NAME' or m == 'RELATED_ENODEB':
            continue
        # tmp = tmp + 1
        # if tmp == 0 or tmp == 1:
        #     continue
        mat[m] = mat[m].astype('str')
        mat[m].loc[mat[m].str.contains('null')] = np.nan
        # mat[m] = mat[m].replace(-3.1415, np.nan)
        mat[m] = mat[m].astype('float32')

        thresh = int(mat.shape[0] * 0.85)
        if mat[m].notnull().sum() < thresh:
            mat = mat.drop([m], axis=1)
            continue

        mat[m] = mat[m].fillna(mat[m].mean())
        print(m)

def NA2MEAN0(mat,start_col,end_col):
    #tmp = -1
    tmp_columns = mat.columns
    tmp_columns = tmp_columns[start_col:end_col]


    for m in tmp_columns:
        if m == 'LC_NAME' or m == 'RELATED_ENODEB':
            continue
        # tmp = tmp + 1
        # if tmp == 0 or tmp == 1:
        #     continue
        mat[m] = mat[m].astype('str')
        mat[m].loc[mat[m].str.contains('null')] = np.nan
        # mat[m] = mat[m].replace(-3.1415, np.nan)
        mat[m] = mat[m].astype('float32')

        thresh = int(mat.shape[0] * 0.85)
        if mat[m].notnull().sum() < thresh:
            mat = mat.drop([m], axis=1)
            continue

        mat[m] = mat[m].fillna(mat[m].mean())
        print(m)


remotePath = '/home/zhangyifan/yanzhenghang/pythonProject/'

#for test data
def preprocess_cellBaseDataForTest(filename = '小区基础数据_验证.xlsx', needsave = True):

    '''[小区别名, 小区标识, 所属eNodeB标识, 所属行政区域类型, 天线数, 双工方式, 采用的cp类型,
        子帧配置类型, 特殊子帧配置类型, 是否为RRU小区, 上行频点, 下行频点, 小区配置的载频发射功率,
         参考信号（RS）的每RE平均发射功率, A类符号功率比值, B类符号功率比值, 广播信道功率,
          最大传输功率, 运行状态, 小区覆盖类型, 小区覆盖范围, 中心载频的信道号, 带宽, 下行循环前缀长度,
           上行循环前缀长度, 上行带宽, 下行带宽, 小区激活状态, 高速小区指示, 发送和接收模式, 工作模式,
            小区是否闭塞, CSFB回落网络优先级, 是否载波聚合小区, 载波聚合类型, 载波聚合频段组合]'''
    # 80784 * 36

    # 继续删除上行CP和下行CP的两个特征；CSFB回落网络优先级特征删除；
    # 删除了特征的候选级小于2的特征，处理了特征之间有矛盾的数据

    # 删除的特征：小区别名, 以及上述的CP类型等特征的删除
    #filename = 'F:\小区基础数据.xlsx'
    '''columName_cellBaseData = ['小区标识', '所属eNodeB标识', '所属行政区域类型', '天线数', '双工方式', '是否为RRU小区', '上行频点', '下行频点',
                              '小区配置的载频发射功率', '参考信号（RS）的每RE平均发射功率', 'A类符号功率比值', 'B类符号功率比值', '广播信道功率', '最大传输功率', '运行状态',
                              '小区覆盖类型', '小区覆盖范围', '中心载频的信道号', '带宽', '上行带宽', '下行带宽', '小区激活状态', '高速小区指示', '发送和接收模式',
                              '工作模式', '小区是否闭塞', '是否载波聚合小区', '载波聚合类型', '载波聚合频段组合']'''
    print('正在读入：'+filename)
    df = pd.read_excel(filename, sheet_name=0, header=0)#names = columName_cellBaseData
    print('读入完成！')

    #去除重复项
    #df.drop_duplicates(keep='first', inplace=True)

    #删除无效特征  df.drop(['B', 'C'], axis=1)
    df.drop(columns=['采用的cp类型','子帧配置类型','特殊子帧配置类型','下行循环前缀长度','上行循环前缀长度', 'CSFB回落网络优先级'], inplace=True)



    #对某些特征的NA取众数，平均值


    # 删除特征的数据存在错误的行
    #thresh = int(df.shape[1] * 0.6)
    #df.dropna(thresh=thresh, inplace=True)

    #df = df.dropna()

    df.发送和接收模式 = df.发送和接收模式.str.replace('两发两收', '2T2R(两发两收)')
    df.发送和接收模式 = df.发送和接收模式.str.replace('一发一收', '1T1R(一发一收)')
    df.发送和接收模式 = df.发送和接收模式.str.replace('2T2R(2T2R(两发两收))', '2T2R(两发两收)')
    df.发送和接收模式 = df.发送和接收模式.str.replace('1T1R(1T1R(一发一收))', '1T1R(一发一收)')

    df = df[~df['发送和接收模式'].isin(['四发四收'])]



    df = datatype_transfer_for_cellbaseForTest(df, needsave)
    #df.loc[:, ['天线数', '带宽']] = df.loc[:, ['天线数', '带宽']].astype('float')
    #df.loc[:, ['天线数', '带宽']].dtypes

    #df['小区标识'] = df['小区标识'].str.replace('.', '')
    df['小区标识'] = df['小区标识'].str.replace(',', '.')


    #df = df.astype('float64')
    #df['小区标识'] = df['小区标识'].astype(dtype='int64')

    df.reset_index(drop=True, inplace=True)
    # df = pd.read_excel(filename, header=[0, 1], index_col=[0, 1])
    if needsave == True:
        df.to_csv(filename[0:-5]+'_preprocessed.csv',encoding="utf_8_sig",index=False)
    return df
def preprocess_cellAntennaDataForTest(filename = '小区天线参数数据_验证.xlsx', needsave = True):
    '''columName_cellBaseData = ['']'''
    print('正在读入：' + filename)
    df = pd.read_excel(filename, sheet_name=0, header=0)  # names=columName_cellBaseData
    print('读入完成！')

    # 去除重复项
    #df.drop_duplicates(keep='first', inplace=True)

    df.drop(columns=['天线类型','美化类型','塔桅类型'], inplace=True,axis= 1)

    tmp = df.发送和接收模式.value_counts().index
    df.发送和接收模式 = df.发送和接收模式.replace(tmp, list(range(1, len(tmp) + 1)))

    #df = df.dropna()

    #df['所属小区标识'] = df['所属小区标识'].str.replace('.', '')
    df['所属小区标识'] = df['所属小区标识'].str.replace('，', '.')

    #df = df.astype('float64')
    #df['所属小区标识'] = df['所属小区标识'].astype(dtype='int64')

    df.reset_index(drop=True, inplace=True)

    # df = pd.read_excel(filename, header=[0, 1], index_col=[0, 1])
    if needsave == True:
        df.to_csv(filename[0:-5] + '_preprocessed.csv',encoding="utf_8_sig",index=False)
    return df
def merge_cellBaseData_cellAntennaDataForTest(Base, Ante, filename = 'final_merge_data_test_V1.csv',needsave = True): #传递参数类型为pd.read_excel的返回对象


    #basetmp = Base.小区标识
    #atmp = Ante.小区标识

    #basetmp = basetmp.str.replace('.', '').astype(dtype='int64')
    #atmp = atmp.str.replace('.', '').astype(dtype='int64')

    # Base['小区标识'] = Base['小区标识'].str.replace('.', '')
    # Base['小区标识'] = Base['小区标识'].str.replace(',', '').astype(dtype='int64')
    # Ante['所属小区标识'] = Ante['所属小区标识'].str.replace('.', '')
    # Ante['所属小区标识'] = Ante['所属小区标识'].str.replace('，', '').astype(dtype='int64')

    data = pd.merge(Base,Ante,left_on='小区标识',right_on='所属小区标识',how='outer')
    data = data.drop('所属小区标识',axis = 1)
    data.reset_index(drop=True, inplace=True)

    colname = data.columns
    colname = colname.drop(['小区标识'])
    colname = colname.drop(['小区别名'])
    colname = colname.drop(['所属eNodeB标识'])
    print('calculate mean')
    mean_all = data[colname].mean(axis=0)
    print('fillna begin!')
    data[colname] = data[colname].fillna(mean_all)
    print('fillna finished!')

    if needsave == True:
        data.to_csv(filename,encoding="utf_8_sig",index=False)
    print('Base和Antenna数据融合完成！')
    return  data







    #pass
def datatype_transfer_for_cellbaseForTest(csvdata,needsave=True):
    tmp = csvdata.发送和接收模式.value_counts().index
    csvdata.发送和接收模式 = csvdata.发送和接收模式.replace(tmp, list(range(1, len(tmp)+1) ))

    #csvdata.小区标识
    #csvdata.所属eNodeB标识 = csvdata.所属eNodeB标识.astype(dtype='int64')
    tmp = csvdata.所属行政区域类型.value_counts().index
    csvdata.所属行政区域类型 = csvdata.所属行政区域类型.replace(tmp, list(range(1, len(tmp)+1) ))
    #csvdata.天线数 = csvdata.天线数.astype(dtype='int64')
    tmp = csvdata.双工方式.value_counts().index
    csvdata.双工方式 = csvdata.双工方式.replace(tmp,list(range(1, len(tmp)+1) ))
    tmp = csvdata.是否为RRU小区.value_counts().index
    csvdata.是否为RRU小区 = csvdata.是否为RRU小区.replace(tmp,list(range(1, len(tmp)+1) ))
    #csvdata.上行频点 = csvdata.上行频点.astype(dtype='int64')
    #csvdata.下行频点 =csvdata.下行频点.astype(dtype='int64')
    #csvdata.小区配置的载频发射功率 =csvdata.小区配置的载频发射功率.astype()
    #。。。。。。。。。
    tmp = csvdata.运行状态.value_counts().index
    csvdata.运行状态 = csvdata.运行状态.replace(tmp,list(range(1, len(tmp)+1) ))
    tmp = csvdata.小区覆盖类型.value_counts().index
    csvdata.小区覆盖类型 = csvdata.小区覆盖类型.replace(tmp,list(range(1, len(tmp)+1) ))
    tmp = csvdata.小区覆盖范围.value_counts().index
    csvdata.小区覆盖范围 = csvdata.小区覆盖范围.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.小区激活状态.value_counts().index
    csvdata.小区激活状态 = csvdata.小区激活状态.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.高速小区指示.value_counts().index
    csvdata.高速小区指示 = csvdata.高速小区指示.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.工作模式.value_counts().index
    csvdata.工作模式 = csvdata.工作模式.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.小区是否闭塞.value_counts().index
    csvdata.小区是否闭塞 = csvdata.小区是否闭塞.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.是否载波聚合小区.value_counts().index
    csvdata.是否载波聚合小区 = csvdata.是否载波聚合小区.replace(tmp,list(range(1, len(tmp)+1) ) )
    tmp = csvdata.载波聚合类型.value_counts().index
    csvdata.载波聚合类型 = csvdata.载波聚合类型.replace(tmp,list(range(1, len(tmp)+1) ))
    tmp = csvdata.载波聚合频段组合.value_counts().index
    csvdata.载波聚合频段组合 = csvdata.载波聚合频段组合.replace(tmp,list(range(1, len(tmp)+1) ) )
    print('小区基站数据量化完成！')
    return csvdata




def main(needsave = True):
    #test()

    needsave = True
    #数据预处理第一部分
    A = preprocess_cellBaseData(needsave = needsave)
    B = preprocess_cellAntennaData(needsave = needsave)
    AB = merge_cellBaseData_cellAntennaData(A, B, needsave = needsave)
    # excelinfo_to_txt(AB, txt_name='F:/data_AB.txt')

    A_test = preprocess_cellBaseDataForTest(filename=remotePath+'小区基础数据_验证.xlsx',needsave = needsave)
    B_test = preprocess_cellAntennaDataForTest(filename=remotePath+'小区天线参数数据_验证.xlsx',needsave  =needsave)
    AB_test = merge_cellBaseData_cellAntennaDataForTest(A_test, B_test, filename=remotePath+'final_merge_data_test_V1.csv',needsave = needsave)




    AB = pd.read_csv(remotePath +'final_merge_data_V1.csv')
    #数据预处理第二部分
    #mode = 1正常模式；mode = -1测试模式
    C_continuous, C_discrete = process_bigData(mode=1,needsave=needsave)
    C_continuous = add_feature(C_continuous)
    C_discrete = add_feature(C_discrete)
    # #C = pd.read_csv('mat.csv')
    ABC_continuous = merge_all(C_continuous,AB, filename='final_continuous_v1.csv')
    ABC_discrete = merge_all(C_discrete,AB, filename='final_discrete_v1.csv')
    print('数据预处理完成！')
    sample_ramdom(ABC_continuous, 60000,filename='final_continuous.csv')
    sample_ramdom(ABC_discrete, 60000,filename='final_discrete.csv')
    print('样本抽取完成')

if __name__ == '__main__':

    main(needsave = True)






