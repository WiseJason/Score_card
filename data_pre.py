#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : WiseJason
import pandas as pd
import numpy as np
import re
import time
import datetime
from dateutil.relativedelta import relativedelta
import os
from sklearn.model_selection import train_test_split
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
path=os.getcwd()+'/application.csv'
data=pd.read_csv(path,header=0)
# print(data.head())
# print(data['loan_status'].unique())
# print(data.groupby(['loan_status'])['member_id'].count())
data['term']=data['term'].apply(lambda x:int(x.replace('months',"")))
data['y']=data['loan_status'].apply(lambda x: int(x=='Charged Off'))
# print(data.groupby(['y'])['member_id'].count())
data=data.loc[data.term==36]
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
train_data['desc_clearn']=train_data['desc'].map(DescExisting)
def datemanage(x,format):
    if str(x)=='nan':
        return datetime.datetime.strptime('9900-1','%Y-%m')
    else:
        return datetime.datetime.strptime(x,format)
train_data['app_date_clean']=train_data['issue_d'].map(lambda x:datemanage(x,'%Y/%m/%d'))
train_data['earliest_cr_line_clean']=train_data['earliest_cr_line'].map(lambda x:datemanage(x,'%Y/%m/%d'))
def MakeupMissing(x):
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
#
#
# num_features = ['int_rate_clean','emp_length_clean','annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_to_app','inq_last_6mths', \
#                 'mths_since_last_record_clean', 'mths_since_last_delinq_clean','open_acc','pub_rec','total_acc']
# cat_features = ['home_ownership', 'verification_status','desc_clean', 'purpose', 'zip_code','addr_state','pub_rec_bankruptcies_clean']
# more_value_features=[]
# less_value_features=[]
#
# for var in cat_features:
#     valueCounts=len(set(train_data[var]))
#     if valueCounts>5:
#         more_value_features.append(var)
#     else:
#         less_value_features.append(var)
#
# def BinbadRate(df,col,target,grantRatwIndicator=0):
#     total=df.groupby([col])[target].count()
#     total=pd.DataFrame({'total':total})
#     bad=df.groupby([col])[target].sum()
#     bad=pd.DataFrame({"bad":bad})
#     regroup=total.merge(bad,left_index=True,right_index=True,how='left')
#     regroup.reset_index(inplace=True)
#     regroup['bad_rate']=regroup.apply(lambda x:x.bad*1.0/x.total,axis=1)
#     dicts=dict(zip(regroup[col],regroup['bad_rate']))
#     if grantRatwIndicator==0:
#         return (dicts,regroup)
#     N=sum(regroup['total'])
#     B=sum(regroup['bad'])
#     overallRate=B*1.0/N
#     return (dicts,regroup,overallRate)
# def MergeBad(df,col,target):
#     regroup=BinbadRate(df,col,target)[1]
#     regroup=regroup.sort_values(by='bad_rate')
#     col_regroup=[[i] for i in regroup[col]]
#     for i  in range(regroup.shape[0]-1):
#         col_regroup[i+1]=col_regroup[i]+col_regroup[i+1]
#         col_regroup.pop(i)
#         if regroup['bad_rate'][i+1]>0:
#             break
#     newGroup={}
#     for i  in range(len(col_regroup)):
#         for g2 in col_regroup[i]:
#             newGroup[g2]='Bin '+str(i)
#     return newGroup
# merge_bin_dict={}
# var_bin_list=[]
# for col in less_value_features:
#     binBadRate=BinbadRate(train_data,col,'y')[0]
#     if min(binBadRate.values())==0:
#         print('{} need to be combined due to 0 bad rate'.format(col))
#         combine_bin=MergeBad(train_data,col,'y')
#         merge_bin_dict[col]=combine_bin
#         newVar=col+'_Bin'
#         train_data[newVar]=train_data[col].map(combine_bin)
#         var_bin_list.append(newVar)
#     if max(binBadRate.values())==1:
#         print('{} need to be combined due to 0 good rate'.format(col))
#         combine_bin=MergeBad(train_data,col,'y',direction='good')
#         merge_bin_dict[col]=combine_bin
#         newVar=col+'_Bin'
#         train_data[newVar]=train_data[col].map(combine_bin)
#         var_bin_list.append(newVar)
# def BadRateEncoding(df,col,target):
#     regroup=BinbadRate(df,col,target,grantRatwIndicator=0)[1]
#     br_dict=regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')
#     for k,v in br_dict.items():
#         br_dict[k]=v['bad_rate']
#     binBadRateEncoding=df[col].map(lambda x:br_dict[x])
#     return {'encoding':binBadRateEncoding,'bad_rate':br_dict}
# br_encoding_dict={}
# for col in more_value_features:
#     br_encoding=BadRateEncoding(train_data,col,'y')
#     print(br_encoding)
#     train_data[col + '_br_encoding'] = br_encoding['encoding']
#     br_encoding_dict[col] = br_encoding['bad_rate']
#     num_features.append(col + '_br_encoding')