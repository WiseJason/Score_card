import pandas as pd
import numpy as np
import os
from scipy.stats import chi2
import warnings
import math
from random import randint
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)
data=pd.read_csv("分箱.csv",index_col=0)
positive_sum=data["positive_num"].sum()
negtive_sum=data['negtive_num'].sum()
data["WOE"]=data.apply(lambda x:math.log(np.divide(x['positive_num']/positive_sum,x['negtive_num']/negtive_sum),math.e),axis=1)
data["系数"]=np.subtract(data['positive_num']/positive_sum,data['negtive_num']/negtive_sum)
data['IV']=(np.multiply(data["WOE"],data['系数'])).round(2)
data["WOE"]=(data["WOE"].map(lambda x:abs(x))).round(2)
del data["系数"]
data.to_csv("编码.csv", index='int_rate_clean')