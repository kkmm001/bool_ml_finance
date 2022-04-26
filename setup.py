# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:20:08 2022

@author: cui
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar

plt.style.use("ggplot")

def fct_bucket(df_test, N=5):
    
    df_tmp  = df_test.dropna().copy()
    if df_tmp.notna().sum() > N:
        bins    = df_tmp.quantile(np.arange(0, N+1, 1)/10)
        labels  = np.arange(1, N+1, 1)/10
        df_output = pd.cut(df_tmp, bins, labels=labels).astype(float)
    else:
        df_output = df_tmp
        df_output.values[:] = np.nan
        
    df_output = df_output.reindex(df_test.index)
    return df_output

def setup_DB():
    
    cwd = r'C:\st_sim\book_ml_finance\data'
    os.chdir(cwd)    
    # data_raw = pd.read_pickle('data_ml.pkl', compression='bz2')
    data_raw    = pd.read_pickle('data_ml_raw.pkl').set_index('date').sort_index()
    data_ml     = data_raw.truncate(before = '1999-12-31', after = '2019-01-01' )

    # data_ml.sort_values(["stock_id","date"], inplace=True)
    data_ml     = data_ml.reset_index().sort_values(["stock_id","date"])
    features    = list(data_ml.iloc[:,2:95].columns)

    # No need to determine "cat", if should be fixed by given universe
    
    # ret_field = ['R1M_Usd', 'R3M_Usd', 'R6M_Usd', 'R12M_Usd']
    # data_ml_group = data_ml.groupby(['date'])
    # for ret_field_i in ret_field:
    #     # print(ret_field_i)
    #     data_ml['%s_cat'%ret_field_i] = data_ml_group[ret_field_i].apply(fct_bucket)

    return data_ml, features




def field_pivot(data_raw, index='date', columns='stock_id', begin_date='2000-01-01', end_date='2018-09-30'):
    # data_all = {};
    data = {};
    df_miss = {};
    for i in progressbar(range(2, data_raw.shape[1])):
        field_name  = data_raw.columns[i]
        df_mat      = data_raw.pivot(index=index, columns=columns, values=field_name)
        df_mat.columns.name = None;
        df_mat.index.name   = None;
        # data_all[field_name] = df_mat
        data[field_name]    = df_mat.truncate(before = begin_date, after=end_date)
        df_miss[field_name] = df_mat.isna().mean(axis=1)
        
    df_miss = pd.DataFrame(df_miss)    
    return data, df_miss


# %%

# quality problem of 3M returns, two much zeros
# df_test = data_ml[['date', 'stock_id', 'R3M_Usd']].set_index(['date', 'stock_id']).R3M_Usd.unstack().iloc[-1, :]


# (df == 0).mean(axis=1).plot()

# (df.isna()).mean(axis=1).plot()


# df_test = data_ml[ret_field_i].
# df_test = fct_bucket(data_ml['R3M_Usd'])






