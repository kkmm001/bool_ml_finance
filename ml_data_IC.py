# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 23:08:28 2022

@author: cui
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from progressbar import progressbar

# import scipy
# import stats
# import usr.methodo.backtests.backtest_module as bt 

import book_ml_finance.setup as setup #import setup_DB


# %% field pivot 

data_ml_raw, features = setup.setup_DB()

data_ml, df_miss = setup.field_pivot(data_ml_raw)

data_ml_raw.columns

df_mktcap = data_ml['Mkt_Cap_3M_Usd']#.truncate(after='2019-02-27')

# "RSP" is ticker of "Invesco S&P 500 Equal Weight ETF"
df_spxew_init = yf.download('RSP', start = str(df_mktcap.index.date[0]), 
               end = str(df_mktcap.index.date[-1]), progress = False)

df_spxew = df_spxew_init['Adj Close']   
df_spxew.index.name = None


# %% correlation calculation - v1
# Equivalent IC among SPX


# df_ret_1M = data_ml['R1M_Usd']
# df_ret_3M = data_ml['R3M_Usd']

# df_corr_1M = {};
# for i in progressbar(range(len(df_mktcap.index))):
    
#     date_i = df_mktcap.index[i]
#     df_cap_i        = df_mktcap.loc[date_i, :]
#     df_cap_clean_i  = df_cap_i.dropna()
#     seuil_i         = df_cap_clean_i.quantile(500/df_cap_clean_i.size)

#     flag_sp500      = df_cap_i <= seuil_i
#     df_ret_use_i    = df_ret_1M.loc[date_i, flag_sp500]
    
#     df_corr_1M_i = [];
#     for j, feature_j in enumerate(features):
#         # print(feature_j)
#         df_factor_i = data_ml[feature_j].loc[date_i, flag_sp500]
#         # scipy.stats.spearmanr(df_ret_1M_i, df_factor_i, nan_policy='omit')
    
#         df_corr_1M_ij = df_ret_use_i.corr(df_factor_i, method='spearman')
#         df_corr_1M_i.append(df_corr_1M_ij)
        
#     df_corr_1M_i = pd.Series(df_corr_1M_i, index=features)
#     df_corr_1M[date_i] = df_corr_1M_i 
    
# df_corr_1M = pd.DataFrame(df_corr_1M).T        
 

# %% correlation calculation - v2
# Equivalent IC among SPX
     
df_ret_1M_SPX = data_ml['R1M_Usd'].copy()
df_ret_3M_SPX = data_ml['R3M_Usd'].copy()


df_mktcap_fill  = data_ml['Mkt_Cap_3M_Usd'].fillna(100)

seuil_sp500     = df_mktcap_fill.quantile(500/df_mktcap_fill.shape[1], axis=1)
df_flag_sp500   = df_mktcap_fill.le(seuil_sp500, axis=0)


df_ret_1M_SPX[~df_flag_sp500] = np.nan;
df_ret_3M_SPX[~df_flag_sp500] = np.nan;

df_vol3Y_raw = data_ml['Vol3Y_Usd'].copy();

df_corr_1M = {};
df_corr_3M = {};

# for j, feature_j in enumerate(features):
for j in progressbar(range(len(features))):
    feature_j = features[j]
    df_corr_1M[feature_j] = df_ret_1M_SPX.corrwith(data_ml[feature_j], axis=1, method='spearman')
    df_corr_3M[feature_j] = df_ret_3M_SPX.corrwith(data_ml[feature_j], axis=1, method='spearman')

df_corr_1M = pd.DataFrame(df_corr_1M)      
df_corr_3M = pd.DataFrame(df_corr_3M)      




# %%


# Goal 1: similarity between factor
# => the similarity can be measured by correlation between factors
# => PCA may be a suitable way.
# See the article of the bird.


# Goal 2: How is better, who is worse.


# Q1: std vs. mean
# Chart 2: std vs. mean    

# Q2: Skewness

# Chart 2: Pct. positive sign
# Chart 3: Skewness measure




# Chart 2: std vs. mean  
# Period of crisis
# Period of calm

df_mean_Y   = df_corr_1M.resample('Y').mean()
df_median_Y = df_corr_1M.resample('Y').median()
df_std_Y    = df_corr_1M.resample('Y').std()


# df_IC_raw   = ((df_corr_1M+1)/2).agg(['mean', 'std', 'median']).T
df_stats     = df_corr_1M.agg(['mean', 'median', 'skew']).T
df_stats = df_stats.sort_values('skew')

fig  = plt.figure(figsize=(12, 6))
ax_use = fig.gca();
df_stats.plot.scatter(x='median', y='mean', ax=ax_use)
plt.title('IC between signal and 1M fwd return')
plt.legend()
plt.axhline(y=0, color='r', linestyle='-')
plt.axvline(x=0, color='r', linestyle='-')



df_corr_1M.skew()


# %% Next step: If volume 3M confirm momentum strategy ?




