from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import scipy.stats

load_model = True
train_continue =True
trainFileName = 'FINAL_OUT_training3nd_data320.csv'
# trainFileName = 'FINAL_OUT_training2nd_v1_all.csv' #'FINAL_OUT_v1.csv'
# trainFileName = 'FINAL_OUT_20w_samples_training2nd_v1_200000_sampled.csv'
#testFileName = 'FINAL_OUT_test2nd_v1.csv'   #'FINAL_OUT_test_v1.csv'
testFileName = 'FINAL_OUT_test3nd_data320.csv'
# testFileName = 'FINAL_OUT_test2nd_v2.csv'
train = pd.read_csv(trainFileName)
if train_continue == True:
    load_model = True
# train.drop(['Unnamed: 0', 'Unnamed: 0.1'],inplace=True, axis=1)
print('readOK!')
train['F0823'].loc[train['F0823'] < 10.0] = -1
train['F0823'].loc[train['F0823'] >= 10.0] = 1
print('OK1')
target='F0823' # F0823的值就是二元分类输出，即标签
IDcol= 'RELATED_ENODEB'
train['F0823'].value_counts()
train = train.drop(['LC_NAME'],axis=1)

x_columns = [x for x in train.columns if x not in [target,IDcol]]
X = train[x_columns]
y = train[target]

X_train, X_test, y_train, y_test = cross_validation.train_test_split( X, y, test_size=0.25, random_state=0)
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


model_name = 'my_randomForest_model.m'
model_name_new = 'my_randomForest_model_v2.m'
rf0 = []

if (load_model == True) & (train_continue == True):
    rf0 = joblib.load(model_name)
    print('Continue to train!')
    rf0.fit(X_train, y_train)
    print('Training ended!')
    joblib.dump(rf0, model_name_new)
elif (load_model == True) & (train_continue == False):
    rf0 = joblib.load(model_name_new)
else:
    #使用默认参数拟合数据
    rf0 = RandomForestClassifier(oob_score=True, random_state=10,n_estimators = 100,n_jobs=20)
    print('Begin to train!')
    rf0.fit(X_train,y_train)
    print('Training ended!')
    joblib.dump(rf0, model_name)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(X_train)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train,y_predprob))
accurate=rf0.score(X_test, y_test)
print('accurate:' + str(accurate))
y_test_predict = rf0.predict(X_test)

F1 = metrics.f1_score(y_test,y_test_predict,labels=[-1,1],average='micro')
print('F1: '+str(F1))

y_test.reset_index(drop=True, inplace=True)
y_test_predict = pd.DataFrame(y_test_predict)
y_test.loc[y_test == -1] = 0

y_test_predict = y_test_predict.iloc[:,0]
y_test_predict.loc[y_test_predict == -1] = 0

result1 = pd.concat([y_test, y_test_predict], ignore_index=True, join='outer', axis=1)

TP = result1.loc[result1.iloc[:,0]==0].shape[0] -  result1.loc[result1.iloc[:,0]==0].iloc[:,1].sum() #原本正预正，复赛要求负预负
FP = result1.loc[result1.iloc[:,0]==1].shape[0] - result1.loc[result1.iloc[:,0]==1].iloc[:,1].sum() #原本负预正，复赛要求正预负
FN = result1.loc[result1.iloc[:,0]==0].iloc[:,1].sum()#原本正预负，复赛要求负预正

P=TP/(TP+FP)
R=TP/(TP+FN)
F1_2=2*P*R/(P+R)

importances = rf0.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, X_train.columns[indices[f]], importances[indices[f]]))


dataframe = pd.read_csv(testFileName)
colname = X_train.columns
tmp_name = ['LC_NAME','RELATED_ENODEB','START_TIME','S_MONTH','S_WEEK','S_DAY','S_HOUR']
dfHead = dataframe.loc[:,tmp_name]

dfForTest = dataframe[colname]
dfForTest = dfForTest.astype('float')

y_out_label = rf0.predict(dfForTest)
y_out_label = (y_out_label+1)/2

dfLabel = pd.DataFrame(y_out_label,columns=['F0823'])


#num分析重要特征前多少个
num = 10
feature_importance_name = X_train.columns[indices[0:num]]
Positive = ( dfLabel['F0823'] == 1 )
Negative = ( dfLabel['F0823'] == 0 )
Positive_mean = [] #正例均值
Negative_mean = [] #负例均值
Positive_std = [] #正例标准差
Negative_std = [] #负例标准差

# dataToDo = dfForTest[feature_importance_name.tolist()].loc[Negative]
dataPanduan = dfHead['LC_NAME'].loc[Negative]
dataToPanduan = dfHead['LC_NAME'].loc[Negative]
num_of_samples = 20   #只分析20个负例,   指负例的个数
DF2SPSS_N = pd.DataFrame()
DF2SPSS_P = pd.DataFrame()

for name in feature_importance_name:
    print(name+':')
    DF2SPSS_N = pd.concat([DF2SPSS_N, dfForTest[name].loc[Negative]], ignore_index=False,
                            join='outer', axis=1)
    DF2SPSS_P = pd.concat([DF2SPSS_P, dfForTest[name].loc[Positive]], ignore_index=False,
                          join='outer', axis=1)

    #name = name.tolist()[0]
    # Positive_tmp = Positive.rename(columns={'F0823': name}, inplace=False)
    # Negative_tmp = Negative.rename(columns={'F0823': name}, inplace=False)
    PM = dfForTest[name].loc[Positive].mean()
    NM = dfForTest[name].loc[Negative].mean()
    PS = dfForTest[name].loc[Positive].std()
    NS = dfForTest[name].loc[Negative].std()
    Positive_mean.append(PM)
    Negative_mean.append(NM)
    Positive_std.append(PS)
    Negative_std.append(NS)

    data_tmp = dfForTest[name].loc[Negative]
    dataPanduan = pd.concat([dataPanduan, data_tmp], ignore_index=False,
                         join='outer', axis=1)
    dataToPanduan = pd.concat([dataToPanduan, data_tmp], ignore_index=False,
                         join='outer', axis=1)
    for i in range(0,num_of_samples):   #len(data_tmp)   只分析20个负例
        print(i)
        tmp = data_tmp.iloc[i]
        tmp_out = -1

        if PM > NM:
            if tmp >= PM:
                tmp_out = 1
            elif tmp <= NM:
                tmp_out = 0
            else:
                value_P = 0.5-scipy.stats.norm(PM, PS).cdf(tmp)
                value_N = scipy.stats.norm(NM, NS).cdf(tmp)-0.5
                tmp_out = value_N/(value_P + value_N)
        else:
            if tmp <= PM:
                tmp_out = 1
            elif tmp >= NM:
                tmp_out = 0
            else:
                value_P = scipy.stats.norm(PM, PS).cdf(tmp)-0.5
                value_N = 0.5 - scipy.stats.norm(NM, NS).cdf(tmp)
                tmp_out = value_N / (value_P + value_N)
        dataPanduan[name].iloc[i] = tmp_out
dataToPanduan = dataToPanduan.iloc[0:num_of_samples]   #num_of_samples个负例的重要特征的原始值
dataPanduan = dataPanduan.iloc[0:num_of_samples]       #num_of_samples个负例的重要特征的判决值，在0-1之间，越靠近1，表明该特征值使得该样本偏向于正例；相反，其越靠近0，特征值使得该样本偏向于负例。

print(dataToPanduan)
print(dataPanduan)

DF2SPSS_N.to_csv('DF2SPSS_N_l3.csv',encoding='gbk',index=False)
DF2SPSS_P.to_csv('DF2SPSS_P_l3.csv',encoding='gbk',index=False)

dataToPanduan.to_csv('dataToPanduan_l3.csv',encoding='gbk',index=False)
dataPanduan.to_csv('dataPanduan_l3.csv',encoding='gbk',index=False)


print('Positive_mean:')
print(Positive_mean)
print('Negative_mean:')
print(Negative_mean)
print('Positive_std:')
print(Positive_std)
print('Negative_std:')
print(Negative_std)


dfOut = pd.concat([dfHead,dfLabel],ignore_index=False,join='outer', axis=1)
print(dfOut)
dfOut.to_csv('dfOut_3nd_l3.csv',encoding='gbk',index=True)


'''
#以下部分为调参部分代码
#对n_estaimators进行网格搜索
param_test1= {'n_estimators':range(10,71,10)}
gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100, min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),param_grid =param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X_train,y_train)
print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_)
#得到最佳弱学习器迭代次数

#对最大深度max_depth和min_samples_split进行网格搜索
param_test2= {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,
                                 min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10),
   param_grid = param_test2,scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X_train,y_train)
print(gsearch2.grid_scores_,gsearch2.best_params_, gsearch2.best_score_)

#用现有的三个参数，查看模型的袋外分数
rf1= RandomForestClassifier(n_estimators= 70, max_depth=13, min_samples_split=50,
                                 min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10)
rf1.fit(X_train,y_train)
print(rf1.oob_score_)

#再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
param_test3= {'min_samples_split':range(50,130,20), 'min_samples_leaf':range(10,60,10)}
gsearch3= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,max_depth=13,
                                 max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test3,scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X_train,y_train)
print(gsearch3.grid_scores_,gsearch2.best_params_, gsearch2.best_score_)

#最后我们再对最大特征数max_features做调参:
param_test4= {'max_features':range(3,11,2)}
gsearch4= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 70,max_depth=13, min_samples_split=50,
                                 min_samples_leaf=10 ,oob_score=True, random_state=10),
   param_grid = param_test4,scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X_train,y_train)
print(gsearch4.grid_scores_,gsearch4.best_params_, gsearch4.best_score_)

#测试调好参数的模型袋外分数
rf2= RandomForestClassifier(n_estimators= 70, max_depth=13, min_samples_split=50,
                                 min_samples_leaf=10,max_features=9 ,oob_score=True, random_state=10)
rf2.fit(X_train,y_train)
print(rf2.oob_score_)

#看测试集结果
clf = RandomForestClassifier(n_estimators= 70, max_depth=13, min_samples_split=50,
                                 min_samples_leaf=10,max_features=9 ,oob_score=True, random_state=10)
clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))
'''