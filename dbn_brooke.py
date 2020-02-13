#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 01:23:56 2020

@author: yacong.zhou@gmail.com
Adapted from: https://github.com/albertbup/deep-belief-network/

Requirements:
    numpy>=1.12.0
    scipy>=0.18.1
    scikit-learn>=0.18.1
    tensorflow>=1.5.0 
"""

import numpy as np

from sklearn.metrics.classification import accuracy_score
from sklearn import preprocessing

from dbn import SupervisedDBNClassification

import data_processing

# load data
original_train_set, original_test_set = data_processing.split_data(
    npy_data_file='../data/all_train.npy', train_portion=0.01, 
    split_mode='first', save='npy')
#original_train_set = np.load('../data/all_train.npy')
#original_test_set = np.load('../data/test_set.npy')

# fill in missing values
train_set = data_processing.missing_values(original_train_set, 
                                           method='median')
test_set = data_processing.missing_values(original_test_set, 
                                            method='median')

# get X and y
X_train = train_set[1:,1:-1]
#X_scaled_train = preprocessing.scale(X_train)
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled_train = min_max_scaler.fit_transform(X_train)
y_train = train_set[1:,-1]
X_test = test_set[:5000,1:-1]
#X_scaled_test = preprocessing.scale(X_test)
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled_test = min_max_scaler.fit_transform(X_test)
y_test = test_set[:5000,-1]

# Training
clf = SupervisedDBNClassification(hidden_layers_structure=[1024, 512],
                                  learning_rate_rbm=0.05,
                                  learning_rate=0.1,
                                  n_epochs_rbm=3,
                                  n_iter_backprop=10,
                                  batch_size=128,
                                  activation_function='sigmoid', # relu->error
                                  dropout_p=0.2)
clf.fit(X_train, y_train)

# Save the model
clf.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# Test
y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(y_test, y_pred))