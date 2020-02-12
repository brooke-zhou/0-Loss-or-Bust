#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 01:15:36 2020

@author: yacong.zhou@gmail.com
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import data_processing

# import dand clean ata
original_data = np.load('../data/all_train.npy')
np_data = data_processing.missing_values(original_data, 
                                            method='zeros')
data = pd.DataFrame(np_data)
X = data.iloc[:,:-1]  
y = data.iloc[:,-1]   

#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
#plot heat map
plt.figure(figsize=(20,20))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")