import pandas as pd
import numpy as np
import os
from scipy.stats import chi2
import warnings
from random import randint
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

def get_chiSquare_distuibution(dfree=1, cf=0.05):
    percents = [ 0.95, 0.90, 0.5,0.1, 0.05, 0.025, 0.01, 0.005]
    df = pd.DataFrame(np.array([chi2.isf(percents, df=i) for i in range(1, 30)]))
    df.columns = percents
    df.index = df.index+1
    pd.set_option('precision', 3)
    chiSquare_threashhold=df.loc[dfree, cf]
    return chiSquare_threashhold
def calc_chiSquare(num_table):
    num_table_cal_chi=num_table.copy()
    num_table_chi=np.array([])
    for i in range(num_table_cal_chi.shape[0]-1):
        chi=(num_table_cal_chi[i,2]-num_table_cal_chi[i,4])**2/num_table_cal_chi[i,4]+(num_table_cal_chi[i,3]-num_table_cal_chi[i,5])**2/num_table_cal_chi[i,5]
        +(num_table_cal_chi[i+1,2]-num_table_cal_chi[i+1,4])**2/num_table_cal_chi[i+1,4]+(num_table_cal_chi[i+1,3]-num_table_cal_chi[i+1,5])**2/num_table_cal_chi[i+1,5]
        num_table_chi=np.append(num_table_chi,chi)
    return num_table_chi

def ChiMerge(data, feature_colname, target_colname,max_bins,sample=None):
    if sample != None:
        df = data.sample(n=sample)
    else:
        data
    num_table =get_num_table(data,feature_colname,target_colname)
    num_table_chiSquare=calc_chiSquare(num_table)
    chi_threshold=get_chiSquare_distuibution(4,0.1)
    min_chiSquare=min(num_table_chiSquare)
    while min_chiSquare<chi_threshold:
        min_index=np.where(num_table_chiSquare==min(num_table_chiSquare))[0]
        num_table_chiSquare, num_table = merge_chiSquare(num_table, min_index)
        min_chiSquare=min(num_table_chiSquare)
    bins=num_table_chiSquare.shape[0]
    while bins>max_bins:
        min_index = np.where(num_table_chiSquare == min(num_table_chiSquare))[0]
        num_table_chiSquare, num_table = merge_chiSquare(num_table, min_index)
        min_chiSquare = min(num_table_chiSquare)
        bins = num_table_chiSquare.shape[0]
    while not monotonicity(num_table):
        min_index = np.where(num_table_chiSquare == min(num_table_chiSquare))[0]
        num_table_chiSquare, num_table = merge_chiSquare(num_table, min_index)
        min_chiSquare = min(num_table_chiSquare)
    result_table = pd.DataFrame(num_table, columns=["int_rate_clean", "total_num", "positive_num", "negtive_num",
                                                 "theory_positive_num", "theory_negtive_num"])
    result_table['chiSquare']=1.0
    result_table['chiSquare'][0]=float("inf")
    result_table['chiSquare'][1:]=num_table_chiSquare
    result_table.to_csv("分箱.csv",index=None)
    return 1

def monotonicity(num_table):
    num_table=pd.DataFrame(num_table,columns=["int_rate_clean","total_num","positive_num","negtive_num","theory_positive_num","theory_negtive_num"])
    num_table['bad_rate']=num_table['positive_num']/num_table['total_num']
    num_table['bad_rate_dif']=num_table['bad_rate'].diff(1)
    num_table=num_table.fillna(0)
    if len(num_table[num_table['bad_rate_dif']>0])==(num_table.shape[0]-1) or len(num_table[num_table['bad_rate_dif']<0])==(num_table.shape[0]-1):
        return True
    else:
        return False

def merge_chiSquare(num_table,min_index):
    num_table_merge=num_table.copy()
    num_table_merge[min_index,0]=num_table_merge[min_index+1,0]
    num_table_merge[min_index, 1] = num_table_merge[min_index + 1, 1]+num_table_merge[min_index,1]
    num_table_merge[min_index, 2] = num_table_merge[min_index + 1, 2]+num_table_merge[min_index,2]
    num_table_merge[min_index, 3] = num_table_merge[min_index + 1, 3]+num_table_merge[min_index,3]
    num_table_merge[min_index, 4] = num_table_merge[min_index + 1, 4]+num_table_merge[min_index,4]
    num_table_merge=np.delete(num_table_merge,min_index+1,axis=0)
    num_table_chiSquare=calc_chiSquare(num_table_merge)
    return num_table_chiSquare,num_table_merge


def get_num_table(data,feature_col,tag_col):
    total_num = data.groupby([feature_col])[tag_col].count()
    total_num = pd.DataFrame({'total_num': total_num})
    positive_num = data.groupby([feature_col])[tag_col].sum()
    positive_num = pd.DataFrame({'positive_num': positive_num})
    positive_rate=data[tag_col].sum()/data.shape[0]
    negtive_rate=1-positive_rate
    num_table = pd.merge(total_num, positive_num, left_index=True, right_index=True,
                         how='inner')
    num_table.reset_index(inplace=True)
    num_table['negtive_num'] = num_table['total_num'] - num_table['positive_num']  # 统计需分箱变量每个值负样本数
    num_table['theory_positive_num']=num_table['total_num']*positive_rate
    num_table['theory_negtive_num'] = num_table['total_num'] * negtive_rate
    num_table=np.array(num_table)
    return num_table


path=os.getcwd()+'/application.csv'
data=pd.read_csv(path)
data['y'] = data['loan_status'].apply(lambda x: int(x == 'Charged Off'))
data['int_rate_clean']=data.loc[:,'int_rate'].map(lambda x: float(x.replace('%',''))/100)
data=data[['int_rate_clean','y']][0:200]
num_table=ChiMerge(data, 'int_rate_clean', 'y',10)
