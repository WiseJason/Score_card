import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def setK(eigValue,rate):
    eigValue=np.argsort(-eigValue)
    a=0
    b=eigValue.sum()
    for i in range(1,eigValue.size+1):
        topK=eigValue[i-1]
        eigVal=eigValue[topK]
        a+=eigVal
        if a/b>rate:
            break
        return i
def pca(x,rate):
    n_samples,n_features=x.shape
    x_mean=x.mean()
    x=x.sub(x_mean,axis=1)
    plt.plot(x,label=x.columns)
    plt.legend()
    plt.show()
    x_cov=x.cov()
    eigValue,eigVec=np.linalg.eig(x_cov)
    eig_pairs=[(np.abs(eigValue[i]),eigVec[:,i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True)
    k=setK(eigValue,rate)
    feature=np.array([ele[1] for ele in eig_pairs[:k]])
    data=np.dot(x,feature.T)
    plt.plot(data)
    plt.legend()
    plt.show()






