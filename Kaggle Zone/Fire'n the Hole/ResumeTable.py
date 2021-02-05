%reset -f
import pandas as pd
import numpy as np
import copy
import seaborn as sns

import scipy.stats as st
mu, sigma = 10, 5
s1 = st.norm(mu, sigma).rvs(10000)
s2 = st.norm(mu-2, sigma+2).rvs(1000)

df_1 = pd.DataFrame(s1,columns=["feat"]  )
df_1['Y'] = 0
df_2 = pd.DataFrame(s2,columns=["feat"]  )
df_2['Y'] = 1

df = pd.concat([df_1,df_2],axis=0)
df = df.reset_index(drop=True)
df.head(5)

import random
cnt=0
for i in range(10):
    N = random.uniform(0,20)
    df['feat_%i' %i] = st.norm(mu+N, sigma+N).rvs(len(df))
    cnt+=1

df.head(5)

from scipy import stats
def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),5) 

    return summary

resumetable(df)