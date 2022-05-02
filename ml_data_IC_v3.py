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



def t_stat(df_corr):
    return(df_corr*(np.sqrt(500))/(np.sqrt(1-df_corr**2)))

def ret_SPX(data_ml, return_type=['R1M_Usd', 'R3M_Usd'], risk_adj=False):
    
    df_mktcap_fill  = data_ml['Mkt_Cap_3M_Usd'].fillna(100)
    seuil_sp500     = df_mktcap_fill.quantile(500/df_mktcap_fill.shape[1], axis=1)
    df_flag_sp500   = df_mktcap_fill.le(seuil_sp500, axis=0)    
    
    dict_ret = {};
    
    for type_i in return_type:
        # print(type_i)
        df_ret_SPX = data_ml[type_i].copy()   
        
        if risk_adj:
            df_ret_SPX = df_ret_SPX/df_ret_SPX.std()
        
        df_ret_SPX[~df_flag_sp500] = np.nan;
        dict_ret[type_i.replace("_Usd", "")] =  df_ret_SPX

    return dict_ret

def IC_compute(dict_ret, data_ml, features):
    
    dict_corr = {};
    dict_stat = {};
    for key, df_ret in dict_ret.items():
        # print('%s\n'%key)
        df_corr_xxx = {};
        
        for j in progressbar(range(len(features))):
            feature_j = features[j]
            df_corr_xxx[feature_j] = df_ret.corrwith(data_ml[feature_j], axis=1, method='spearman')
        
        df_corr_xxx = pd.DataFrame(df_corr_xxx) 
        
        dict_corr[key] = df_corr_xxx   
        dict_stat[key] = t_stat(df_corr_xxx)
        
    return dict_corr, dict_stat


# %% field pivot 

data_ml_raw, features = setup.setup_DB()
data_ml_raw.columns = data_ml_raw.columns.str.replace('Total_Liabilities_Total_Assets', 'Leverage')
features = [xxx.replace('Total_Liabilities_Total_Assets', 'Leverage') for xxx in features]

data_ml, df_miss = setup.field_pivot(data_ml_raw)

df_mktcap       = data_ml['Mkt_Cap_3M_Usd'] #.truncate(after='2019-02-27')


# "RSP" is ticker of "Invesco S&P 500 Equal Weight ETF"
df_spxew_init   = yf.download('RSP', start = str(df_mktcap.index.date[0]), 
                  end = str(df_mktcap.index.date[-1]), progress = False)

df_spxew = df_spxew_init['Adj Close']   
df_spxew.index.name = None


df_AAPL_init = yf.download('AAPL', 
          start = str(df_mktcap.index.date[0]), 
          end = str(df_mktcap.index.date[-1]), 
          progress = False)

df_AAPL = df_AAPL_init['Adj Close'].dropna().resample('M').last()
df_AAPL.index.name = None

df_ret_AAPL = df_AAPL.pct_change(1)

df_corr = data_ml['R1M_Usd'].shift(-1).corrwith(df_ret_AAPL)


# %% correlation calculation - raw returns
# Equivalent IC among SPX

dict_ret_raw = ret_SPX(data_ml, risk_adj=False)
dict_ret_adj = ret_SPX(data_ml, risk_adj=True)

dict_corr_raw, dict_stat_raw = IC_compute(dict_ret_raw, data_ml, features)
dict_corr_adj, dict_stat_adj = IC_compute(dict_ret_adj, data_ml, features)

dict_corr = {'raw':dict_corr_raw, 'adj':dict_corr_adj}

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

dict_corr_use = dict_corr_raw

## # Chart 1: Comparison of active risk
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(hspace=0.25, wspace=0.25);

ax_counter = 0

for key, df_corr_use in dict_corr_use.items():
    # print(key)
    
    df_stats    = df_corr_use.agg(['mean', 'median', 'skew', 'std']).T
    
    ax_use      = ax[ax_counter]
    df_stats.plot.scatter(x='median', y='mean', ax=ax_use)
    ax_use.set_title('IC of factor - %s (raw returns)'%key)
    ax_use.axhline(y=0, color='r', linestyle='-')
    ax_use.axvline(x=0, color='r', linestyle='-')
    
    ax_counter = ax_counter +1
    ax_use.set_ylim([-0.05,0.05])
    ax_use.set_xlim([-0.05,0.05])
    
    
dict_corr_use = dict_corr_adj

## # Chart 1: Comparison of active risk
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(hspace=0.25, wspace=0.25);

ax_counter = 0

for key, df_corr_use in dict_corr_use.items():
    # print(key)
    
    df_stats    = df_corr_use.agg(['mean', 'median', 'skew', 'std']).T
    
    ax_use      = ax[ax_counter]
    df_stats.plot.scatter(x='median', y='mean', ax=ax_use)
    ax_use.set_title('IC of factor - %s (risk-adjusted returns)'%key)
    ax_use.axhline(y=0, color='r', linestyle='-')
    ax_use.axvline(x=0, color='r', linestyle='-')
    
    ax_counter = ax_counter + 1
    ax_use.set_ylim([-0.05,0.05])
    ax_use.set_xlim([-0.05,0.05]) 
 

IC_median_raw = dict_corr_raw['R3M'].median();
IC_median_adj = dict_corr_adj['R3M'].median();

flag_selection = np.abs(IC_median_adj) > 2/100

res_topIC = IC_median_adj[flag_selection]


# field_use = ['Mkt_Cap_3M_Usd', 'Mom_5M_Usd', 'Mom_11M_Usd', 'Pb', 'Ebitda_Margin', 'Ocf_Margin', 
#              'Op_Margin', 'Roa', 'Roc', 'Asset_Turnover', 'Nd_Ebitda', 'Tot_Debt_Rev', 
#              'Total_Liabilities_Total_Assets', 'Div_Yld', 'Vol1Y_Usd', 'Vol3Y_Usd'];


dict_field  = {'size':['Mkt_Cap_3M_Usd'], 
               'mom': ['Mom_5M_Usd', 'Mom_11M_Usd', 'Mom_Sharp_5M_Usd', 'Mom_Sharp_11M_Usd'], 
               'value':['Pb', 'Pe'], 'margin':['Ebitda_Margin', 'Ocf_Margin', 'Op_Margin'], 
               'profitability':['Roa', 'Roc', 'Asset_Turnover'], 
               'leverage': ['Nd_Ebitda', 'Tot_Debt_Rev', 'Leverage'],
               'dvd':['Div_Yld'], 'vol':['Vol1Y_Usd', 'Vol3Y_Usd']};

field_use   = [yyy for xxx in dict_field.values() for yyy in xxx]
df_corr_use = dict_corr_adj['R3M'].loc[:, field_use]

df_plot     = df_corr_use.quantile([0.25, 0.5, 0.75])

fig     = plt.figure(figsize=(13, 9))
ax_use  = fig.gca();
df_corr_use.boxplot(ax=ax_use)

df_IC = IC_median_raw[field_use].to_frame('raw')
df_IC['adj'] = IC_median_adj[field_use]

df_IC.plot.scatter(x='raw', y='adj')


# %% Next step: If volume 3M confirm momentum strategy ?




