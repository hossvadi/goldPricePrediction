# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:36:42 2022

@author: Hossein.JvdZ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# loading to a Pandas DataFrame
gold_data = pd.read_csv('dataset.csv')

#show 5 first rows
#gold_data.head()

#show 5 last rows
#gold_data.tail()

#missing values
#gold_data.isnull().sum()

# getting statistical measure of data
#gold_data.describe()

#Possitive or Negative Correlation
correlation = gold_data.corr()
# construct a heatmap to understand the correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size:8'}, cmap='Blues')




