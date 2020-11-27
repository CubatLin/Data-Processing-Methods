
"""
2019年10月底台灣權值前100股票名單與資料: TW100.xlsx

Robo Advisor with 
1. Alpha/Beta strategies for picking stocks
2. MV for optimization
3. Rebalance asset every year

"""

import os
from os.path import join
import pandas as pd
import numpy as np
import datetime
import csv
import scipy.optimize as sco

path = os.getcwd()
src = join(path , 'input')
out = join(path , 'output')

'''Picking asset strategies'''
# 'alpha' or 'beta'
bench = 'alpha'

'''period'''
start = datetime.datetime(2006,1,1)
end = datetime.datetime(2018,12,31)

TWII = pd.read_excel(join(src, 'TWII.xlsx'), index_col = 0).loc[start:end,:]
TWII.index = pd.to_datetime(TWII.index)

'''input data'''
TW100 = pd.read_excel(join(src, 'TW100.xlsx'), index_col = 0).loc[start:end,:]
pd.read_excel(join(src , 'TW100.xlsx'), index_col = 0).loc[start:end,:]

stock = TW100.dropna(axis = 1) 
stock.to_csv(join(out , 'tw100_clean.csv') , index = True)

tw100_clean = pd.DataFrame(stock)
tw100_clean.to_csv(join(out , 'tw100_clean.csv') , index = True)

N = int(len(stock)/13)

log_stock = np.log(stock/stock.shift(N)).dropna(how = 'all')
log_TWII = np.log(TWII/TWII.shift(N)).dropna(how = 'all')


# alpha and beta
rf = 0
def df_reg(df_year , df_tw):
    df = {'alpha':[] , 'beta':[]}
    for col in df_year:

        Cov = np.cov(df_year[col], df_tw['TWII'])
        B = Cov[0,1] / Cov[1,1]
        
        # ALPHA
        A = (np.mean(df_year[col]) - rf) - B * (np.mean(df_tw['TWII']) - rf)
        
        df['alpha'].append(A)
        df['beta'].append(B)
        
    df = pd.DataFrame(df , index = df_year.columns)
    return df

# return and volatility functions
def Portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights )
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) 
    return std, returns


# volatility function of portfolio
def Portfolio_volatility(weights, mean_returns, cov_matrix):
    return Portfolio_performance(weights, mean_returns, cov_matrix)[0]


# MV model
def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0,1)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(Portfolio_volatility, num_assets*[1/num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


y = list(np.unique(log_stock.index.year))

value = {'year': [] , 'Portfolio' : [100] , 'TWII' : [100]}


for i in range(len(y)-1):
    year = i + y[0]
    
    data = log_stock[log_stock.index.year == year]
    twii = log_TWII[log_TWII.index.year == year]
    
    # pick the first five alpha or beta stocks
    df = df_reg(data , twii)
    T = df[bench].sort_values(ascending = True)
    T = list(T[:5].index)
    
    # data
    tarprice = TW100[TW100.index.year == year+1][T]
    if i == 0:
        value['year'].append(tarprice.index[0].date())
    
    value['year'].append(tarprice.index[-1].date())    
   
    tarreturn = log_stock[log_stock.index.year == year][T]
    
    # Markowitz MV Optimization
    mean_return = tarreturn.mean()
    cov_matrix = tarreturn.cov()
    
    w = min_variance(mean_return , cov_matrix)["x"]
    
    # calculate value
    buy_price = tarprice.iloc[0]
    sell_price = tarprice.iloc[-1]
    
    units = (value['Portfolio'][i] * w) / buy_price
    spread = sell_price - buy_price
    value['Portfolio'].append( value['Portfolio'][i] + (spread * units).sum())
    
    # TW index
    V_TWII = TWII[TWII.index.year == year+1]
    TWII_buy = V_TWII.iloc[0]
    TWII_sell = V_TWII.iloc[-1]
    
    U_TWII = float(value['TWII'][i] / TWII_buy)
    Spread_TWII = float(TWII_sell - TWII_buy)
    
    value['TWII'].append(value['TWII'][i] + Spread_TWII * U_TWII)

value = pd.DataFrame(value)
value.to_csv(join(out , 'Portfolio vs TWII.csv') , index = False)

# 繪圖
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

plt.figure(figsize=(10, 6))
plt.plot( 'year', 'Portfolio', data=value, color='#F5B041', linewidth=2)
plt.plot( 'year', 'TWII', data=value, color='#5DADE2', linewidth=2)
plt.legend()
plt.grid(True)
plt.title('Portfolio vs TWII')
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.xticks(value['year'])
plt.xlabel('Date')
plt.ylabel('Value')
plt.gcf().autofmt_xdate()
plt.savefig(join(out ,'Portfolio vs TWII.png'))

file = open(join(out , 'Portfolio vs TWII.csv') , 'a' , newline='')
w = csv.writer(file)
irr_P = round(np.power(value['Portfolio'][len(y)-1]/value['Portfolio'][0],1/len(y)) - 1, 4)
irr_tw = round(np.power(value['TWII'][len(y)-1]/value['TWII'][0],1/len(y)) - 1 , 4)

w.writerow([])
w.writerow(['' , 'IRR'])
w.writerow(['Portfolio' , irr_P ])
w.writerow(['TWII' , irr_tw ])
file.close()
