# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:54:03 2022

@author: cui
"""
import os
import pandas as pd


cwd = r"C:\st_sim\blog_general\ml_book"
os.chdir(cwd)

# Load the data
data_raw=pd.read_excel('data_ml.xlsx')       
data_raw.to_pickle('data_ml.pkl', compression='bz2')

