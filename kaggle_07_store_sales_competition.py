# -*- coding:utf8 -*-
# @TIME : 2018/5/8 上午06:38
# @Author : Allen
# @File : kaggle_07_store_sales_competition.py

#Rossmann Store Sales: https://www.kaggle.com/c/rossmann-store-sales
#Forecast sales using store, promotion, and competitor data

#学习的点
#numpy tolist()的用法：https://blog.csdn.net/lilong117194/article/details/78437224
#Python isinstance() 函数:http://www.runoob.com/python/python-func-isinstance.html
#Python中append和extend的区别:https://www.cnblogs.com/subic/p/6553187.html,
#使用sklearn之LabelEncoder将Label标准化:https://blog.csdn.net/u010412858/article/details/78386407
#sklearn的cross_validation不能使用，cross_validation不能用：https://www.cnblogs.com/alanma/p/6877354.html
#python os.path模块常用方法详解:https://www.cnblogs.com/wuxie1989/p/5623435.html
#'list' object has no attribute 'shape':https://stackoverflow.com/questions/21015674/list-object-has-no-attribute-shape


#代码思路：
#【1】定义一些变换和评判标准，调整loss function
    #传入一个numpy数组，把它的非零值变为平方的倒数
    #平均值（y-yhat的差值平方*1/y平方），最后开根号
    #这个函数最后要在xgboost中要调用，这个是自己定义的
    #打印store、train、test的数据出来看看

#【2】定义函数用来加载数据：分为数值型与非数值型
    #导入store数据，
    #合并数据作为train， test
    #定义数值型特征和非数值型特征
    #返回train，test, test中提取出的feature，feature中的非数值型变量

#【3】数据与特征处理
    #定义year, month, day数据，观察每一个门店，列出1月份～12月份，有活动商店的名单
    #先把noisy特征去掉, 目前所有的features是test中的列名提取出的,数值型特征&非数值型特征
    #加入新的特征'year', 'month', 'day', 注意这里append是当作整体添加到其中，extend是当作一个序列，与原序列合并放在后面
    #预处理numberic_values, 填充缺失值，定义一个class
    #预处理non-numberic的值，LabelEncoder可以将标签分配一个0—n_classes-1之间的编码

#【4】使用xgboost训练与分析数据 & to_csv文件 & plot画图


# 导入库
import os
import pandas as pd
import numpy as np
import datetime
import csv
import scipy as sp
import xgboost as xgb
import itertools
import operator
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import cross_validation
from sklearn.model_selection import train_test_split #这里需要更新
from matplotlib import pylab as plt
plot = True

goal = 'Sales'
myid = 'Id'

#【1】定义一些变换和评判标准，调整loss function
#传入一个numpy数组，把它的非零值变为平方的倒数
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w
#平均值（y-yhat的差值平方*1/y平方），最后开根号
def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return  rmspe

#这个函数最后要在xgboost中要调用，这个是自己定义的
def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) -1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

os.chdir('/Users/a1/Desktop/learning/kaggle_07/')

store = pd.read_csv('store.csv')
#打印store的数据出来看看
print(store.head())
print("**************")

#打印train中的内容看看
train_df = pd.read_csv('train.csv')
print(train_df.head())
print('*****###########******')

#打印test中的内容
test_df = pd.read_csv('test.csv')
print(test_df.head())
print('**************')

#【2】定义函数用来加载数据
def load_data():
    '''
    加载数据，分为数值型与非数值型
    '''
    #导入store数据，
    store = pd.read_csv('store.csv')
    train_org = pd.read_csv('train.csv', dtype={'StateHoliday':pd.np.string_})
    test_org = pd.read_csv('test.csv', dtype={'StateHoliday':pd.np.string_})
    #合并数据作为train， test
    train = pd.merge(train_org, store, on='Store', how='left')
    test = pd.merge(test_org, store, on='Store', how = 'left')
    #定义数值型特征和非数值型特征
    features = test.columns.tolist() #把array格式转化为list格式
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features_numeric = test.select_dtypes(include=numerics).columns.tolist()
    features_non_numeric = [f for f in features if f not in features_numeric]
    #返回train，test, test中提取出的feature，feature中的非数值型变量
    return (train, test, features, features_non_numeric)

#【3】数据与特征处理
def process_data(train, test, features, features_non_numeric):
    '''
    Feature engineering and selection.
    :param train:
    :param test:
    :param features:
    :param features_non_numeric:
    :return:
    '''
    train = train[train['Sales'] > 0]

    for data in [train, test]:
        #year, month, day
        data['year'] = data.Date.apply(lambda x: x.split('-')[0])
        data['year'] = data['year'].astype(float)
        data['month'] = data.Date.apply(lambda x: x.split('-')[1])
        data['month'] = data['month'].astype(float)
        data['day'] = data.Date.apply(lambda x: x.split('-')[2])
        data['day'] = data['day'].astype(float)

        #观察每一个门店，列出1月份～12月份，有活动商店的名单
        #先判断变量的类型用isinstance(x, float)， 如果哪家门店1月份打折，最后它结果返回1
        data['promojan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jan" in x else 0)
        data['promofeb'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Feb" in x else 0)
        data['promomar'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Mar" in x else 0)
        data['promoapr'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Apr" in x else 0)
        data['promomay'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "May" in x else 0)
        data['promojun'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jun" in x else 0)
        data['promojul'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jul" in x else 0)
        data['promoaug'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Aug" in x else 0)
        data['promosep'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Sep" in x else 0)
        data['promooct'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Oct" in x else 0)
        data['promonov'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Nov" in x else 0)
        data['promodec'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Dec" in x else 0)

        #Feature set
        #先把noisy特征去掉, 目前所有的features是test中的列名提取出的,数值型特征&非数值型特征
        noisy_features = [myid, 'Date']
        features = [c for c in features if c not in noisy_features]
        features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]

        #加入新的特征'year', 'month', 'day', 注意这里append是当作整体添加到其中，extend是当作一个序列，与原序列合并放在后面
        features.extend(['year', 'month', 'day'])

        #预处理numberic_values, 填充缺失值
        ## http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
        class DataFrameImputer(TransformerMixin):
            #初始化
            def __init__(self):
                '''
                Inpute missing values.
                columns of dtype object are inputed with the most frequent value in column.
                columns of other types are inpute with mean of column
                '''
            #满足类型，填充出现次数最多的，不满足，填充均值，这个应该是针对数值型特征和非数值型特征来设计的
            def fit(self, X, y=None):
                self.fill = pd.Series([X[c].value_counts().index[0]  # mode
                                       if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],  # mean
                                      index=X.columns)
                return self
            def transform(self, X, y=None):
                return X.fillna(self.fill)

        train = DataFrameImputer().fit_transform(train)
        test = DataFrameImputer().fit_transform(test)

        #预处理non-numberic的值
        #LabelEncoder可以将标签分配一个0—n_classes-1之间的编码
        le = LabelEncoder()
        for col in features_non_numeric:
            le.fit(list(train[col]) + list(test[col]))
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
        #做归一化操作用StandardScaler(), 把需要做归一化的列挑出来
        scaler = StandardScaler()
        for col in set(features) - set(features_non_numeric) - set([]):
            scaler.fit(np.array(list(train[col]) + list(test[col])).reshape(1,-1))
            #ValueError: Expected 2D array, got 1D array instead:https://blog.csdn.net/dongyanwen6036/article/details/78864585
            train[col] = scaler.transform(train[col])
            test[col] = scaler.transform(test[col])
        return (train, test, features, features_non_numeric)


#【4】训练与分析数据,
# 这里用用xgboost可以拿到特征的重要程度，可以进一步优化
def XGB_native(train, test, features, features_non_numeric):
    depth = 13
    eta = 0.01
    ntrees = 8000
    mcw = 3
    params = {
        "objective": "reg:linear",
        "booster": "gbtree",
        "max_depth": depth,
        "min_child_weight": mcw,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "silent": 1
    }
    print("Running with params :" + str(params))
    print("Running with ntree :" + str(ntrees))
    print("Running with features :" + str(features))

    #Train model with local split
    tsize = 0.05
    X_train, X_test = train_test_split(train, test_size=tsize)
    dtrain = xgb.DMatrix(X_train[features], np.log(X_train[goal] + 1))
    dvalid = xgb.DMatrix(X_test[features], np.log(X_test[goal] + 1))
    #这里有疑问？？，这是什么参数
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    #xbgoost训练
    gbm = xgb.train(params, dtrain, ntrees, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

    #这里也有疑问？
    train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test[goal].values)
    print(error)

    #predct and Export
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    submission = pd.DataFrame({myid: test[myid], goal:np.exp(test_probs) - 1})

    #生成csv文件
    if not os.path.exists('result/'):
        os.makedirs('result/')
    submission.to_csv("./result/data-xgb_d%s_eta%s_nntree%s_mcw%s_tsize%s.csv" % (str(depth), str(eta), str(mcw), str(tsize)))

    #feature importance 特征重要程度
    if plot:
      outfile = open('xgb.fmap', 'w')
      i = 0
      for feat in features:
          outfile.write('{0}\t{1}\tq\n'.format(i, feat))
          i = i + 1
      outfile.close()
      importance = gbm.get_fscore(fmap='xgb.fmap')
      importance = sorted(importance.items(), key=operator.itemgetter(1))
      df = pd.DataFrame(importance, columns=['feature', 'fscore'])
      df['fscore'] = df['fscore'] / df['fscore'].sum()
      # Plotitup
      plt.figure()
      df.plot()
      df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
      plt.title('XGBoost Feature Importance')
      plt.xlabel('relative importance')
      plt.gcf().savefig('Feature_Importance_xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.png' % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize)))

#运行程序
print(" ==> 载入数据...")
train, test, features, features_non_numeric = load_data()
print(" ==> 处理数据与特征工程...")
train, test, features, features_non_numeric = process_data(train, test, features, features_non_numeric)
print(" ==> 使用XGBoost建模...")
XGB_native(train, test, features, features_non_numeric)

















