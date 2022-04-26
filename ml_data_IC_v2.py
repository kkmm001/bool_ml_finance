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
import statsmodels.api as sm
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


# %% correlation calculation - raw returns
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

dict_corr = {};

dict_corr['1M'] = df_corr_1M
dict_corr['3M'] = df_corr_3M

# t_stats_1M = df_corr_1M*(np.sqrt(500))/(np.sqrt(1-df_corr_1M**2))
# t_stats_3M = df_corr_3M*(np.sqrt(500))/(np.sqrt(1-df_corr_3M**2))

def t_stat(df_corr):
    return(df_corr*(np.sqrt(500))/(np.sqrt(1-df_corr**2)))

# %% correlation calculation - beta neutral version
# Equivalent IC among SPX
     
df_ret_1M_raw = data_ml['R1M_Usd'].copy()
df_ret_3M_raw = data_ml['R3M_Usd'].copy()

df_ret_1M_mkt = df_ret_1M_SPX.mean(axis=1)
df_ret_3M_mkt = df_ret_3M_SPX.mean(axis=1)


Y = df_ret_1M_raw
X = df_ret_1M_mkt

df_res = {};
for i, col_i in enumerate(df_ret_1M_raw.columns):
    print(i)
    X_cst = sm.add_constant(X).copy()
    model   = sm.OLS(Y.iloc[:, i], X_cst, missing='drop')
    results = model.fit()
    
    res         = results.params.copy();
    res['R2']   = results.rsquared
    df_res[col_i] = res

df_res = pd.DataFrame(df_res).T

beta = df_res.iloc[:, 1].copy()
beta[beta>=1.2] = 1.2
beta[beta<=0.8] = 0.8



df_ret_1M_mkt_repeated = pd.concat([df_ret_1M_mkt]*len(beta), axis=1)
df_ret_1M_mkt_repeated.columns = df_ret_1M_SPX.columns

df_ret_3M_mkt_repeated = pd.concat([df_ret_3M_mkt]*len(beta), axis=1)
df_ret_3M_mkt_repeated.columns = df_ret_3M_SPX.columns

df_ret_1M_SPX_neutral = df_ret_1M_SPX - beta*df_ret_1M_mkt_repeated
df_ret_3M_SPX_neutral = df_ret_3M_SPX - beta*df_ret_3M_mkt_repeated


df_vol3Y_raw = data_ml['Vol3Y_Usd'].copy();

df_corr_1M_neutral = {};
df_corr_3M_neutral = {};

# for j, feature_j in enumerate(features):
for j in progressbar(range(len(features))):
    feature_j = features[j]
    df_corr_1M_neutral[feature_j] = df_ret_1M_SPX_neutral.corrwith(data_ml[feature_j], axis=1, method='spearman')
    df_corr_3M_neutral[feature_j] = df_ret_3M_SPX_neutral.corrwith(data_ml[feature_j], axis=1, method='spearman')

df_corr_1M_neutral = pd.DataFrame(df_corr_1M_neutral)      
df_corr_3M_neutral = pd.DataFrame(df_corr_3M_neutral)      

dict_corr_neutral = {};

dict_corr_neutral['1M'] = df_corr_1M_neutral
dict_corr_neutral['3M'] = df_corr_3M_neutral



# t_stats_1M_neutral = df_corr_1M_neutral*(np.sqrt(500))/(np.sqrt(1-df_corr_1M_neutral**2))
# t_stats_3M_neutral = df_corr_3M_neutral*(np.sqrt(500))/(np.sqrt(1-df_corr_1M_neutral**2))



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


# %% 

# fig  = plt.figure(figsize=(12, 6))
# ax_use = fig.gca();
# df_stats.plot.scatter(x='skew', y='avg_std', ax=ax_use)
# plt.title('IC between signal and 1M fwd return')
# plt.legend()
# plt.axhline(y=0, color='r', linestyle='-')
# plt.axvline(x=0, color='r', linestyle='-')

# df_corr_1M.skew()


dict_corr_use = dict_corr_neutral

## # Chart 1: Comparison of active risk
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(hspace=0.25, wspace=0.25);

ax_counter = 0

for key, df_corr_use in dict_corr_use.items():
    # print(key)
    
    df_stats    = df_corr_use.agg(['mean', 'median', 'skew', 'std']).T
    
    ax_use      = ax[ax_counter]
    df_stats.plot.scatter(x='median', y='mean', ax=ax_use)
    ax_use.set_title('Correlation SP between factor \nand %s fwd return (market neutral)'%key)
    ax_use.axhline(y=0, color='r', linestyle='-')
    ax_use.axvline(x=0, color='r', linestyle='-')
    
    ax_counter = ax_counter +1
    ax_use.set_ylim([-0.05,0.05])
    ax_use.set_xlim([-0.05,0.05])
    
    

## # Chart 2: Consistency of IC sign
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(hspace=0.25, wspace=0.25);

ax_counter = 0

for key, df_corr_use in dict_corr_use.items():
    # print(key)
    
    df_stats    = df_corr_use.agg(['std']).T
    df_stats['pct_positive_corr']    = (df_corr_use>0).mean()

    ax_use      = ax[ax_counter]
    df_stats.plot.scatter(x='std', y='pct_positive_corr', ax=ax_use)
    ax_use.set_title('Correlation SP between factor \nand %s fwd return (market neutral)'%key)
    ax_use.axhline(y=0.5, color='r', linestyle='-')
    # ax_use.axvline(x=0, color='r', linestyle='-')
    
    ax_counter = ax_counter +1
    # ax_use.set_ylim([-0.05,0.05])
    # ax_use.set_xlim([-0.05,0.05])    

# %% Next step: If volume 3M confirm momentum strategy ?




