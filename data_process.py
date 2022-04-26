# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:54:03 2022

@author: cui
"""
import os
import numpy as np
import pandas as pd
from progressbar import progressbar
import yfinance as yf
import matplotlib.pyplot as plt

cwd = r"C:\st_sim\blog_general\ml_book"
os.chdir(cwd)

# Load the data
data_raw = pd.read_pickle('data_ml.pkl', compression='bz2')


# %% field pivot 

data_dict = {};
for i in progressbar(range(2, data_raw.shape[1])):
    field_name = data_raw.columns[i]
    df_mat = data_raw.pivot(index='date',columns='stock_id',values=field_name)
    df_mat.columns.name = None;
    df_mat.index.name = None;
    data_dict[field_name] = df_mat
    
    
# %% Replication of benchmarks

# Rebalance announcements are made on the first Friday of March, June, Sept, and Dec. 
# This change becomes effective prior to the market open on Friday, September 3, 2021

# Conclusion: 
    # Top 500 EW ptf based on ML_data.xlsx is not perfect replication of S&P 500 EW.
    # Firstly, market_cap is 3M average but not latest value
    # Secondly, daily is only monthly but not weekly
    # Thirly, more check to do for return outlier since avg(ret) is not robust.


df_mktcap = data_dict['Mkt_Cap_3M_Usd']#.truncate(after='2019-02-27')
df_ret_3M = data_dict['R3M_Usd']
df_ret_ptf = {};
for i, date_i in enumerate(df_mktcap.index):
    if date_i.month%3 == 2:
        print(date_i)
        df_cap_i    = df_mktcap.loc[date_i, :].sort_values(ascending=False)
        flag_sp500  = df_cap_i[:500].index
        ret_i = df_ret_3M.loc[date_i, flag_sp500]
        df_ret_ptf[date_i] = ret_i.mean()
    
df_ret_ptf  = pd.Series(df_ret_ptf).shift(1)    
    

# "RSP" is ticker of "Invesco S&P 500 Equal Weight ETF"
df_spxew_init = yf.download('RSP', start = str(df_mktcap.index.date[0]), 
               end = str(df_mktcap.index.date[-1]), progress = False)

df_spxew = df_spxew_init['Adj Close']   
df_spxew.index.name = None
    
df_ret_ptf_use = df_ret_ptf.truncate(before = df_spxew.index[0])
df_ret_ptf_use.values[0] = np.nan 
   
df_spxew_use = df_spxew.reindex(df_ret_ptf_use.index, method='ffill')
    
df_spxew_use_ret = df_spxew_use.pct_change(1)  
    
df_comp = pd.DataFrame({'price_index':df_spxew_use_ret, 'ptf':df_ret_ptf_use})    
    
df_comp.plot.scatter(x=0, y=1)


df_nav = (df_comp+1).cumprod()
df_nav.plot()

# NAV plotting
fig  = plt.figure(figsize=(12, 6))
ax_use = fig.gca();
df_nav.plot(ax=ax_use)
plt.grid()
# plt.title('Comparison NAVs (High dvd started in 2018)')
plt.ylabel('Nav (rebased to 100)')
plt.legend()


# %% Momentum strategy among SPX

df_mktcap = data_dict['Mkt_Cap_3M_Usd']#.truncate(after='2019-02-27')
df_ret_3M = data_dict['R3M_Usd']
df_mom_12M = data_dict['Mom_Sharp_11M_Usd']
df_mom_6M  = data_dict['Mom_Sharp_5M_Usd']
df_mom_use = df_mom_6M + df_mom_12M


df_ret_ptf = {};
for i, date_i in enumerate(df_mktcap.index):
    if date_i.month%3 == 2:
        print(date_i)
        df_cap_i        = df_mktcap.loc[date_i, :].sort_values(ascending=False)
        flag_sp500      = df_cap_i[:500].index
        
        df_mom_6M_i     = df_mom_6M.loc[date_i, flag_sp500]
        df_mom_12M_i    = df_mom_12M.loc[date_i, flag_sp500]
        df_mom_i        = pd.DataFrame({'mom_6M':df_mom_6M_i, 'mom_12M':df_mom_12M_i})
        df_mom_avg_i    = df_mom_i.mean(axis=1)
        flag_fac        = df_mom_avg_i[:250].index
        
        ret_i = df_ret_3M.loc[date_i, flag_fac]
        
        
        df_ret_ptf[date_i] = ret_i.mean()

df_ret_ptf  = pd.Series(df_ret_ptf).shift(1)   
 
df_ret_ptf_use = df_ret_ptf.truncate(before = df_spxew.index[0])
df_ret_ptf_use.values[0] = np.nan 

df_spxew_use = df_spxew.reindex(df_ret_ptf_use.index, method='ffill')
df_spxew_use_ret = df_spxew_use.pct_change(1) 

df_comp = pd.DataFrame({'price_index':df_spxew_use_ret, 'ptf':df_ret_ptf_use})    
    
df_comp.plot.scatter(x=0, y=1)


df_nav = (df_comp+1).cumprod()
df_nav.plot()

# NAV plotting
fig  = plt.figure(figsize=(12, 6))
ax_use = fig.gca();
df_nav.plot(ax=ax_use)
plt.grid()
plt.title('Momentum factor')
plt.ylabel('Nav (rebased to 100)')
plt.legend()


# %% Next step: If volume 3M confirm momentum strategy ?


















