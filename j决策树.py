import pandas as pd
import numpy as np


#计算信息熵
def caculateEntropy(data):
    pass


if __name__=="__main__":
    data=pd.read_csv('application.csv')
    print(data.info())
    caculateEntropy(data)


