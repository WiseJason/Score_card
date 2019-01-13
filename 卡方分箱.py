import pandas as pd
import numpy as np
import os
from scipy.stats import chi2
import warnings
from random import randint
warnings.filterwarnings("ignore")
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

def get_chiSquare_distuibution(dfree=1, cf=0.05):
    '''
    根据自由度和置信度得到卡方分布和阈值
    dfree:自由度，分类类别-1，默认为4
    cf:显著性水平，默认10%
    '''
    percents = [ 0.95, 0.90, 0.5,0.1, 0.05, 0.025, 0.01, 0.005]
    df = pd.DataFrame(np.array([chi2.isf(percents, df=i) for i in range(1, 30)]))
    df.columns = percents
    df.index = df.index+1
    # 显示小数点后面数字
    pd.set_option('precision', 3)
    chiSquare_threashhold=df.loc[dfree, cf]
    return chiSquare_threashhold
def calc_chiSquare(num_table):
    num_table_cal_chi=num_table.copy()
    # print(num_table_cal_chi.shape)
    num_table_chi=np.array([])
    for i in range(num_table_cal_chi.shape[0]-1):
        chi=(num_table_cal_chi[i,2]-num_table_cal_chi[i,4])**2/num_table_cal_chi[i,4]+(num_table_cal_chi[i,3]-num_table_cal_chi[i,5])**2/num_table_cal_chi[i,5]
        +(num_table_cal_chi[i+1,2]-num_table_cal_chi[i+1,4])**2/num_table_cal_chi[i+1,4]+(num_table_cal_chi[i+1,3]-num_table_cal_chi[i+1,5])**2/num_table_cal_chi[i+1,5]
        num_table_chi=np.append(num_table_chi,chi)
    return num_table_chi

def ChiMerge(data, feature_colname, target_colname,max_bins,sample=None):
    '''
    运行前需要 import pandas as pd 和 import numpy as np
    df:传入一个数据框仅包含一个需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
    feature_colname:需要卡方分箱的变量名称（字符串）
    flag：正负样本标识的名称（字符串）
    confidenceVal：置信度水平（默认是不进行抽样95%）
    bin：最多箱的数目
    sample: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
    '''
    # 进行是否抽样操作


    if sample != None:
        df = data.sample(n=sample)
    else:
        data
    num_table =get_num_table(data,feature_colname,target_colname)
    # print(num_table.shape)
    # print(num_table[2,5])
    num_table_chiSquare=calc_chiSquare(num_table)
    # chi_threshold=get_chiSquare_distuibution(4,0.1)
    chi_threshold=3.81
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
    # a=np.array(float("inf")).reshape(1,)
    # num_table_chiSquare=np.row_stack((a,num_table_chiSquare))
    # print(num_table_chiSquare.shape)
    # print(num_table_chiSquare)
    # print(num_table.shape)
    print("finish")
    result_table = pd.DataFrame(num_table, columns=["int_rate_clean", "total_num", "positive_num", "negtive_num",
                                                 "theory_positive_num", "theory_negtive_num"])
    result_table['chiSquare']=1.0
    result_table['chiSquare'][0]=float("inf")
    result_table['chiSquare'][1:]=num_table_chiSquare
    # num_table_chiSquare=np.append(num_table,num_table_chiSquare)
    # num_table_chiSquare=pd.DataFrame(num_table_chiSquare,columns=["int_rate_clean","total_num","positive_num","negtive_num","theory_positive_num","theory_negtive_num","chiSquare"])
    # num_table_chiSquare.to_csv("卡方分箱结果.csv",index=None)
    # return num_table
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
    total_num = data.groupby([feature_col])[tag_col].count()  # 统计需分箱变量每个值数目
    total_num = pd.DataFrame({'total_num': total_num})  # 创建一个数据框保存之前的结果
    positive_num = data.groupby([feature_col])[tag_col].sum()  # 统计需分箱变量每个值正样本数
    positive_num = pd.DataFrame({'positive_num': positive_num})  # 创建一个数据框保存之前的结果
    positive_rate=data[tag_col].sum()/data.shape[0]
    negtive_rate=1-positive_rate
    num_table = pd.merge(total_num, positive_num, left_index=True, right_index=True,
                         how='inner')  # 组合total_num与positive_class
    num_table.reset_index(inplace=True)
    num_table['negtive_num'] = num_table['total_num'] - num_table['positive_num']  # 统计需分箱变量每个值负样本数
    num_table['theory_positive_num']=num_table['total_num']*positive_rate
    num_table['theory_negtive_num'] = num_table['total_num'] * negtive_rate
    num_table=np.array(num_table)
    return num_table
# def save_result(num_table):


path=os.getcwd()+'/application.csv'
data=pd.read_csv(path)
data['y'] = data['loan_status'].apply(lambda x: int(x == 'Charged Off'))
data['int_rate_clean']=data.loc[:,'int_rate'].map(lambda x: float(x.replace('%',''))/100)
data=data[['int_rate_clean','y']][0:200]
num_table=ChiMerge(data, 'int_rate_clean', 'y',10)
