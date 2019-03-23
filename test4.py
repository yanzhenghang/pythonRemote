import pandas as pd
import numpy as np
remotePath = '/home/zhangyifan/yanzhenghang/pythonProject/'
def process_big_each(mat):
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

    thresh = int(mat.shape[0] * 0.95)
    mat.dropna(thresh=thresh, inplace=True, axis=1)

    thresh = int(mat.shape[1] * 0.95)
    mat.dropna(thresh=thresh, inplace=True, axis=0)

    mat = mat.astype('str')
    str_nan = str(np.nan)
    print('begin replace!')

    mat = mat.replace('$null$', str_nan)
    print('End replace!')
    columns_tmp = mat.columns[1:(mat.shape[1])]
    mat[columns_tmp] = mat[columns_tmp].astype('float')


    thresh = int(mat.shape[0] * 0.9)
    mat.dropna(thresh=thresh, inplace=True, axis=1)

    thresh = int(mat.shape[1] * 0.9)
    mat.dropna(thresh=thresh, inplace=True, axis=0)
    return mat

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



needsave = True
#数据预处理第一部分
A = preprocess_cellBaseData(needsave = needsave)
B = preprocess_cellAntennaData(needsave = needsave)
AB = merge_cellBaseData_cellAntennaData(A, B, needsave = needsave)
# excelinfo_to_txt(AB, txt_name='F:/data_AB.txt')

A_test = preprocess_cellBaseDataForTest(filename=remotePath+'小区基础数据_验证.xlsx',needsave = needsave)
B_test = preprocess_cellAntennaDataForTest(filename=remotePath+'小区天线参数数据_验证.xlsx',needsave  =needsave)
AB_test = merge_cellBaseData_cellAntennaDataForTest(A_test, B_test, filename=remotePath+'final_merge_data_test_V1.csv',needsave = needsave)


def readbigfile():
    NofFiles = 7
    for i in range(1, NofFiles):
        print('读取开始！')
        tmp = i + 2
        mat = pd.read_csv(remotePath + '31' + str(tmp) + 'data' + '.csv', encoding='gbk')
        print('31' + str(tmp) + 'data' + '.csv')
        mat = process_big_each(mat)
        print('Begin to save!')
        mat.to_csv(remotePath+'31' + str(tmp) + 'data_v1.csv',encoding="utf_8_sig",index=False, sep=',')
        print('31' + str(tmp) + 'data' + '.csv'+'  saved!')



def put7filesTogether():
    NofFiles = 7
    print('put7filesTogether()读取开始！')
    mat = pd.read_csv(remotePath + '312data_v1' + '.csv',encoding="utf_8_sig")
    print('312data_v1' + '.csv reading finished!')
    for i in range(1, NofFiles):
        print('put7filesTogether()读取开始！')
        tmp = i + 2
        mat_tmp = pd.read_csv(remotePath + '31' + str(tmp) + 'data_v1' + '.csv',encoding="utf_8_sig")
        print('31' + str(tmp) + 'data_v1' + '.csv reading finished!')
        mat = mat.append(mat_tmp,ignore_index = True)
        #mat = pd.concat([mat,mat_tmp],ignore_index = True,join='inner',axis=0)

        print('concated '+str(i))
    return mat

# mat = put7filesTogether()
# AB = pd.read_csv(remotePath +'final_merge_data_V1.csv')
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
    if needsave:
        data.to_csv(filename,index=False, sep=',')
    return data

def sample_ramdom(Data, nofSamples = 1,filename = 'sample_ramdom.csv'):
    import random
    random.seed(5)
    ind = random.sample(range(0,Data.shape[0]),nofSamples)
    out = Data.loc[ind,:]
    out.to_csv(filename[0:-4]+'_'+str(nofSamples)+'_sampled.csv', encoding="utf_8_sig", index=False)

def fillna_mean(ABC):
    print('Begin to dropna again!')
    thresh = int(ABC.shape[0] * 0.9)
    ABC.dropna(thresh=thresh, inplace=True, axis=1)

    # thresh = int(ABC.shape[1] * 0.95)
    # ABC.dropna(thresh=thresh, inplace=True, axis=0)
    ABC.dropna(inplace=True, axis=0)


    #columns_tmp = ABC.columns
    #columns_tmp = columns_tmp.drop(['RELATED_ENODEB'])  #前面已经去掉了这个


    # print('Begin to calculate the mean!')
    # mean_value = ABC[columns_tmp].mean()
    # print('Mean finished! Begin to fillna!')
    # ABC[columns_tmp] = ABC[columns_tmp].fillna(mean_value)
    # print('fillna Finished! ')

    ABC.reset_index(drop=True, inplace=True)
    return ABC

readbigfile()
mat = put7filesTogether()
AB = pd.read_csv(remotePath +'final_merge_data_V1.csv')
ABC = merge_all(mat,AB, filename='merged.csv',needsave=False)
ABC = add_feature(ABC)

del mat
del AB

ABC = pd.read_csv(remotePath +'ABC_v1.csv')
ABC = fillna_mean(ABC)
print('Begin to save!')
ABC.to_csv('FINAL_OUT_v1.csv',encoding="utf_8_sig",index=False, sep=',')
print('Begin to sample!')
ABC = sample_ramdom(ABC, nofSamples = 200000,filename = 'FINAL_OUT_20w_samples_v1.csv')
ABC_colname = ABC.loc[0:1,:]
ABC_colname.to_csv('FINAL_OUT_v1_colname.csv',encoding="utf_8_sig",index=False, sep=',')
print('Finished all!')
# readbigfileForTest()

def readbigfileForTest():
    NofFiles = 1
    for i in range(0, NofFiles):
        print('读取开始！')
        tmp = i + 9
        mat = pd.read_csv(remotePath + '31' + str(tmp) + 'data_验证' + '.csv', encoding='gbk')
        print('31' + str(tmp) + 'data_验证' + '.csv')
        mat = process_big_eachForTest(mat)
        print('Begin to save!')
        mat.to_csv(remotePath+'31' + str(tmp) + 'data_验证_v1.csv',encoding="utf_8_sig",index=False, sep=',')
        print('31' + str(tmp) + 'data_验证_v1' + '.csv'+'  saved!')
def process_big_eachForTest(mat):
    # test = mat['F0823'].astype('str')
    # mat = mat[~ test.str.contains('null')]
    # mat['F0823'] = mat['F0823'].astype('float')
    mat = mat.drop(['F0823'],axis = 1)#  测试数据集里面显示为  隐藏
    # test = mat['RELATED_ENODEB'].astype('str')
    # mat = mat[~ test.str.contains('null')]

    # thresh = int(mat.shape[1] * 0.85)
    # mat.dropna(thresh=thresh, inplace=True)

    # mat = mat.drop(
    #     ['START_TIME', 'S_MONTH', 'S_WEEK', 'F0057', 'F0080', 'F0125', 'F0190', 'F0191', 'F0192', 'F0202',
    #      'F0272', 'F0274', 'F0334', 'F0335', 'F0422', 'F0423', 'F0510', 'F0514', 'F0517', 'F0707', 'F0740', 'F0766',
    #      'F0822'], axis=1)

    mat = mat.drop(
        ['F0057', 'F0080', 'F0125', 'F0190', 'F0191', 'F0192', 'F0202',
         'F0272', 'F0274', 'F0334', 'F0335', 'F0422', 'F0423', 'F0510', 'F0514', 'F0517', 'F0707', 'F0740', 'F0766',
         'F0822'], axis=1)

    # thresh = int(mat.shape[0] * 0.95)
    # mat.dropna(thresh=thresh, inplace=True, axis=1)
    #
    # thresh = int(mat.shape[1] * 0.95)
    # mat.dropna(thresh=thresh, inplace=True, axis=0)

    mat = mat.astype('str')
    str_nan = str(np.nan)
    print('begin replace!')

    mat = mat.replace('$null$', str_nan)
    print('End replace!')
    # columns_tmp = mat.columns[1:(mat.shape[1])]
    # mat[columns_tmp] = mat[columns_tmp].astype('float')


    # thresh = int(mat.shape[0] * 0.9)
    # mat.dropna(thresh=thresh, inplace=True, axis=1)
    #
    # thresh = int(mat.shape[1] * 0.9)
    # mat.dropna(thresh=thresh, inplace=True, axis=0)
    return mat
def processForTest(ABC_test):
    ABC_colname = pd.read_csv(remotePath + 'FINAL_OUT_v1_colname.csv', encoding="utf_8_sig")

    ABC_part0 = ABC_test['LC_NAME']
    # ABC_test = ABC_test.drop(['LC_NAME'],axis=1)
    ABC_part1 = ABC_test['RELATED_ENODEB']
    ABC_part2 = ABC_test['START_TIME']
    ABC_part3 = ABC_test['S_MONTH']
    ABC_part4 = ABC_test['S_WEEK']

    ABC_colname = ABC_colname.drop(['F0823','LC_NAME'],axis=1)
    #ABC_colname = ABC_colname.drop(['LC_NAME'])   #ABC_colname里面已经不含有LC_NAME
    col_names = ABC_colname.columns
    ABC_part_rest = ABC_test[col_names]
    # name_tmp = ['LC_NAME','RELATED_ENODEB','START_TIME','S_MONTH','S_WEEK']
    # name_tmp = name_tmp + col_names.tolist()#, names=name_tmp

    ABC_test = pd.concat([ABC_part0, ABC_part1, ABC_part2, ABC_part3, ABC_part4, ABC_part_rest], ignore_index=False,join='outer', axis=1)

    ABC_test.reset_index(drop=True, inplace=True)

    # print('calculate mean')
    # mean_all = ABC_test[col_names].mean(axis=0)
    # print('fillna begin!')
    # ABC_test[col_names] = ABC_test[col_names].fillna(mean_all)
    # print('fillna finished!')

    for name in col_names:
        cnt = ABC_test[name].loc[ABC_test[name].isna()].shape[0]
        if cnt>0:
            print('Begin to calculate mean!')
            print(name)
            mean_each = ABC_test[name].mean(axis=0)
            print( ' calculate mean finished!')
            ABC_test[name] = ABC_test[name].fillna(mean_each)
            print(' fillna finished!')
    ABC_test.reset_index(drop=True, inplace=True)
    return ABC_test
def merge_allForTest(D_big, D_little, filename = 'merge_all.csv',needsave = True):
    data = pd.merge(D_big, D_little, left_on='LC_NAME', right_on = '小区别名', how='left')
    data = data.drop(['小区别名','小区标识'],axis = 1)
    if needsave:
        data.to_csv(filename,index=False, sep=',')
    return data

print('Begin!')
mat_test = pd.read_csv(remotePath+'319data_验证_v1.csv',encoding="utf_8_sig")
AB_test = pd.read_csv(remotePath +'final_merge_data_test_V1.csv')
print('test data reading finished!')
ABC_test = merge_allForTest(mat_test,AB_test, filename='merged_test_all.csv',needsave=False)
print('test data merging finished!')
ABC_test = add_feature(ABC_test)
print('test data adding finished!')
ABC_test = processForTest(ABC_test)
print('Saving FINAL_OUT_v1_test.csv')
ABC_test.to_csv(remotePath+'FINAL_OUT_test_v1.csv',encoding="utf_8_sig",index=False, sep=',')
print('Finished!')


