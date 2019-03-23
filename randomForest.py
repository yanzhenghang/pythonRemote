from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
import pandas as pd
train = pd.read_csv('FINAL_OUT_v1.csv')
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

X_train, X_test, y_train, y_test = cross_validation.train_test_split( X, y, test_size=0.3, random_state=0)
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


#使用默认参数拟合数据
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X_train,y_train)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(X_train)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train,y_predprob))
accurate=rf0.score(X_test, y_test)
print('accurate:' + str(accurate))
y_test_predict = rf0.predict(X_test)

F1 = metrics.f1_score(y_test,y_test_predict,labels=[-1,1],average='micro')
print('F1: '+str(F1))

dataframe = pd.read_csv('FINAL_OUT_test_v1.csv')

colname = X_train.columns

tmp_name = ['LC_NAME','RELATED_ENODEB','START_TIME','S_MONTH','S_WEEK','S_DAY','S_HOUR']
dfHead = dataframe.loc[:,tmp_name]

dfForTest = dataframe[colname]
dfForTest = dfForTest.astype('float')

y_out_label = rf0.predict(dfForTest)
y_out_label = (y_out_label+1)/2

dfLabel = pd.DataFrame(y_out_label,columns=['F0823'])
dfOut = pd.concat([dfHead,dfLabel],ignore_index=False,join='outer', axis=1)
print(dfOut)
dfOut.to_csv('dfOut.csv',encoding='gbk',index=True)


#一下哦部分为调参部分代码
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