# -*- coding: utf-8 -*-
"""
Created on Sun May  1 11:31:21 2022

@author: cui
"""

import os, re
import numpy as np
import pandas as pd

import csv
import datetime
import yfinance as yf
from progressbar import progressbar
import pickle


import requests
from json import loads

key = 'N7ITOCFKA6AHTH9I' 

def read_daily_price(symbol = 'IBM', function = 'TIME_SERIES_DAILY', key = 'N7ITOCFKA6AHTH9I'):
    # function: 
        
        # TIME_SERIES_DAILY
        # TIME_SERIES_DAILY_ADJUSTED (Premium)
        # TIME_SERIES_WEEKLY
        # TIME_SERIES_WEEKLY_ADJUSTED
        # TIME_SERIES_MONTHLY
        # TIME_SERIES_MONTHLY_ADJUSTED
        
    url = 'https://www.alphavantage.co/query?function=%s&symbol=%s&outputsize=full&apikey=%s'%(function, symbol, key)
    data_raw    = requests.get(url).json()
    data_clean  = data_raw[list(data_raw.keys())[1]]
    data        = pd.DataFrame(data_clean).T
    
    # Keep only letters
    data.columns = [re.sub("[^a-zA-Z]+", "", xxx) for xxx in data.columns]
    
    return data




def read_price_intraday(symbol = 'IBM', interval = '5min', key = 'N7ITOCFKA6AHTH9I'):
            
    # Time Interval:
        # Time interval between two consecutive data points in the time series. 
        # The following values are supported: 1min, 5min, 15min, 30min, 60min
        
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=%s&outputsize=full&apikey=%s'%(symbol, interval, key)
    data_raw    = requests.get(url).json()
    data_clean  = data_raw[list(data_raw.keys())[1]]
    data        = pd.DataFrame(data_clean).T
    
    # Keep only letters
    data.columns = [re.sub("[^a-zA-Z]+", "", xxx) for xxx in data.columns]

    return data



def read_price_intraday_extend(symbol = 'IBM', interval = '5min', timeslice = 'year1month1', key = 'N7ITOCFKA6AHTH9I'):
        
    # To ensure optimal API response speed, the trailing 2 years of intraday data is evenly divided into 24 "slices" - 
        # year1month1, year1month2, year1month3, ..., 
        # year1month11, year1month12, year2month1, year2month2, year2month3, ..., 
        # year2month11, year2month12. 
        # Each slice is a 30-day window, with year1month1 being the most recent and year2month12 being the farthest from today. 
        # By default, slice=year1month1.
            
    # Time Interval:
        # Time interval between two consecutive data points in the time series. 
        # The following values are supported: 1min, 5min, 15min, 30min, 60min
        
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=%s&interval=%s&slice=%s&apikey=%s'%(symbol, interval, timeslice, key)
    data_raw    = requests.get(url).json()
    data_clean  = data_raw[list(data_raw.keys())[1]]
    data        = pd.DataFrame(data_clean).T
    
    data.columns = [re.sub("[^a-zA-Z]+", "", xxx) for xxx in data.columns]

    return data



def Listing_status():

    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    # CSV_URL = 'https://www.alphavantage.co/query?function=LISTING_STATUS&date=2014-07-10&state=delisted&apikey=demo'
    CSV_URL = 'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo'
    
    with requests.Session() as s:
        download    = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        cr          = csv.reader(decoded_content.splitlines(), delimiter=',')
        
        cr  = pd.DataFrame(cr)
        cr  = cr.T.set_index(0).T
        
        df_listing  = cr.loc[cr.assetType=='Stock', :]
        # my_list = pd.DataFrame(list(cr))
    return df_listing



# %% Function validation

df_test_intraday    = read_price_intraday(key='demo')
df_test_daily       = read_daily_price(key='demo')


# %%




# %% Load Reference Data

cwd = r"C:\st_sim\Alpha_avantage\blackrock_csv_msci_USA"
os.chdir(cwd)    

list_holding    = pd.read_pickle('raw_holding_msci_usa.pkl')
df_ref          = list_holding['20220428']

df_all_stocks   = Listing_status()


# %% 

flag_valid = np.in1d(df_ref['Issuer Ticker'], df_all_stocks['symbol']) 
df_ref.loc[~flag_valid, :]


# %% Yahoo

ISIN_list   = df_ref['ISIN'].to_list()
Ticker_list = df_ref['Issuer Ticker'].to_list()


df_close  = {};
df_volume = {};

# for ISIN_i in ISIN_list:
for i in progressbar(range(len(ISIN_list))):
    
    ISIN_i  = ISIN_list[i];
    Ticker_i  = Ticker_list[i];

    symbol_use = Ticker_i
    # symbol_use = 'HEIA'   # Example to Test for 
    
    ticker  = yf.Ticker(symbol_use)
    df_i    = ticker.history(period="max")

    if len(df_i) == 0:
        ticker  = yf.Ticker(symbol_use[:-1]+"-"+symbol_use[-1])
        df_i    = ticker.history(period="max")


    df_close[symbol_use]  = df_i['Close']
    df_volume[symbol_use] = df_i['Volume']


with open('mkt_data_msci_usa.pkl', 'wb') as f:
    pickle.dump(df_close, f)
    pickle.dump(df_volume, f)
    pickle.dump(df_ref, f)







