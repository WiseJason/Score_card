#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : WiseJason
import pandas as pd
import numpy as np
import re
import time
import datetime
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

def plt_null_col(data):
    train_isnull = data.isnull().mean()  # 缺失值的比例
    print(train_isnull[train_isnull>0])#查看有缺失的数据
    train_isnull=train_isnull[train_isnull>0].sort_values(ascending=False)
    train_isnull.plot.bar(figsize=(12, 8), title='数据缺失情况')
    # plt.show()
def plt_null_row(data):
    plt.figure(figsize=(12, 8))
    plt.scatter(np.arange(data.shape[0]),
    data.isnull().sum(axis=1).sort_values().values)  # 每行缺失的个数
    plt.show()

path=os.getcwd()+'/application.csv'
data=pd.read_csv(path,header=0)
# print(data.head())
# print(data['loan_status'].unique())
# print(data.groupby(['loan_status'])['member_id'].count())
data['term']=data['term'].apply(lambda x:int(x.replace('months',"")))
data['y']=data['loan_status'].apply(lambda x: int(x=='Charged Off'))
# print(data.groupby(['y'])['member_id'].count())
data=data.loc[data.term==36]
print(data.shape)
plt_null_col(data)
# plt_null_row(data)
# print(data.groupby(['y'])['member_id'].count())
train_data,test_data=train_test_split(data,test_size=0.4)
train_data['int_rate_clean']=train_data.loc[:,'int_rate'].map(lambda x: float(x.replace('%',''))/100)
def Year(x):
    x=str(x)
    if x.find('n/a'):
        return -1
    elif x.find('10+'):
        return 11
    else:
        return int(re.sub("\D", "", x))
train_data['emp_length_clean']=train_data['emp_length'].map(Year)
def DescExisting(x):
    x2=str(x)
    if x2=='nan':
        return 'no desc'
    else:
        return 'desc'
train_data['desc_clean']=train_data['desc'].map(DescExisting)
def datemanage(x,format):
    if str(x)=='nan':
        return datetime.datetime.strptime('9900-1','%Y-%m')
    else:
        return datetime.datetime.strptime(x,format)
train_data['app_date_clean']=train_data['issue_d'].map(lambda x:datemanage(x,'%Y/%m/%d'))
train_data['earliest_cr_line_clean']=train_data['earliest_cr_line'].map(lambda x:datemanage(x,'%Y/%m/%d'))
def MakeupMissing(x):#用-1填充空值
    if np.isnan(x):
        return -1
    else:
        return x
train_data['mths_since_last_delinq_clean'] = train_data['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))
train_data['mths_since_last_record_clean'] = train_data['mths_since_last_record'].map(lambda x:MakeupMissing(x))
train_data['pub_rec_bankruptcies_clean'] = train_data['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))
train_data['limit_income'] = train_data.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)


def MonthGap(earlyDate,lateDate):
    if lateDate>earlyDate:
        yr=lateDate.year-earlyDate.year
        end_month=yr*12+lateDate.month
        delta = end_month - earlyDate.month
        return delta
    else:
        return 0
train_data['earliest_cr_to_app'] = train_data.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)

num_features = ['int_rate_clean','emp_length_clean','annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_to_app','inq_last_6mths', \
                'mths_since_last_record_clean', 'mths_since_last_delinq_clean','open_acc','pub_rec','total_acc']

cat_features = ['home_ownership', 'verification_status','desc_clean', 'purpose', 'zip_code','addr_state','pub_rec_bankruptcies_clean']

def cat_feature(cat_features):
    more_value_features = []
    less_value_features = []
    for var in cat_features:
        valueCounts=len(train_data[var].unique())
        if valueCounts>5:
            more_value_features.append(var)#超过5个需要处理
        else:
            less_value_features.append(var)
    return more_value_features,less_value_features
more_value_features=cat_feature(cat_features)[0]
less_value_features=cat_feature(cat_features)[1]
print(more_value_features)
print(less_value_features)






