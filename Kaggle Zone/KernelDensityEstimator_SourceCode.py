import numpy as np
import pandas as pd
import copy
import scipy.stats as st
import math
from collections import defaultdict

mu, sigma = 10, 5
s1 = st.norm(mu, sigma).rvs(100000)
s2 = st.norm(mu-4, sigma+4).rvs(100000)

df_1 = pd.DataFrame(s1,columns=["feat"]  )
df_1['Y'] = 0

df_2 = pd.DataFrame(s2,columns=["feat"]  )
df_2['Y'] = 1
df = pd.concat([df_1,df_2],axis=0).reset_index(drop=True)

#不同Y之間的interval不同,要傳入原始df的最大值&最小值
xlen =0.5

def feat_fit_func(df,X,X_len): #創造mapping dict
    feat_X = df.sort_values(by=[X]).reset_index(drop=True)[X].to_numpy()
    interval_n = math.ceil((feat_X.max()-feat_X.min())/X_len) #區間數
    feat_Xmin = feat_X.min() #最小值
    
    print("interval_n: ",interval_n)
    print("feat_Xmin: ",feat_Xmin)

    for i in range(interval_n,-1,-1):
        feat_X[(feat_Xmin + (i)*X_len <= feat_X) &\
               (feat_X < feat_Xmin + (i+1)*X_len) ] = feat_Xmin + (i+1)*X_len #向右靠齊
    unique, counts = np.unique(feat_X, return_counts=True)
    counts_propotion = counts/len(feat_X)

    my_dict=defaultdict(int)
    for i in range(len(unique)):
        my_dict[unique[i]]=counts_propotion[i]

    return my_dict

feat_dict_0 = feat_fit_func(df[df['Y']==0].reset_index(drop=True),'feat',xlen)
feat_dict_1 = feat_fit_func(df[df['Y']==1].reset_index(drop=True),'feat',xlen)

#fit 固定範圍,transform就可以直接用
def feat_transform_func(df,X,X_len,fit_dict): #放入df & mapping dict, Return Feat pdf
    feat_X = copy.deepcopy(df[X]).to_numpy()
    
    #區間+1才會是原長度
    interval_n = math.ceil((max(list(fit_dict.keys())) - min(list(fit_dict.keys())) )/X_len)+1 
    feat_Xmin = min(list(fit_dict.keys()))-X_len #向右靠齊所以要往前推一格

    for i in range(interval_n,-1,-1):
        feat_X[(feat_Xmin + (i)*X_len <= feat_X) &\
               (feat_X < feat_Xmin + (i+1)*X_len) ] = feat_Xmin + (i+1)*X_len #向右靠齊
    
    return feat_X
    
df['feat_Y=0'] = feat_transform_func(df,'feat',xlen,feat_dict_0)
df['feat_Y=0_pdf'] = [feat_dict_0[i] for i in df['feat_Y=0']]

df['feat_Y=1'] = feat_transform_func(df,'feat',xlen,feat_dict_1)
df['feat_Y=1_pdf'] = [feat_dict_1[i] for i in df['feat_Y=1']]

df['diff_pdf'] = df['feat_Y=1_pdf'] - df['feat_Y=0_pdf']



Y= 'Y'
X='feat'
bins = math.ceil((df[X].max()-df[X].min())/xlen)-2

df = df.sort_values(by=[X]).reset_index(drop=True)
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
plt.figure(figsize=(15,10),dpi=60,linewidth = 2)
plt.hist(df.loc[df[Y]==0,X],bins=bins,density=True)
plt.hist(df.loc[df[Y]==1,X],bins=bins,density=True)

plt.plot(df['feat'],df['feat_Y=1_pdf'],'o-',color = 'r', label="pdf(Y=1)")
plt.plot(df['feat'],df['feat_Y=0_pdf'],'o-',color = 'g', label="pdf(Y=0)")
plt.plot(df['feat'],df['diff_pdf'],'o-',color = 'b', label="diff_pdf")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 標示x,y軸(labelpad代表與圖片的距離)
plt.xlabel("feat", fontsize=20, labelpad = 15)
plt.ylabel("pdf", fontsize=20, labelpad = 20)
# 顯示出線條標記位置
plt.legend(loc = "best", fontsize=20)

plt.show()

