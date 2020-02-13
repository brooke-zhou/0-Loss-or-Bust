#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:06:03 2020

@author: yacong.zhou@gmail.com
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

import data_processing
from helper_brooke import classification_err

# partition data to get training and test sets
# original_train_set, original_test_set = data_processing.split_data(
#      data_file_name='../data/train.csv', train_portion=0.1, 
#      split_mode='first', save='npy')
# original_train_set, original_test_set = data_processing.split_data(
#     npy_data_file='../data/all_train.npy', train_portion=0.8, 
#     split_mode='random', save='npy')
# original_train_set, original_test_set = data_processing.split_data(
#     data_file_name='../data/debug.csv', train_portion=0.8, 
#     split_mode='random', save='npy')
# original_train_set, original_test_set = data_processing.split_data(
#     npy_data_file='../data/all_debug.npy', train_portion=0.8, 
#     split_mode='random', save='npy')

# load partitioned data from saved files
#original_train_set = np.load('../data/default_partition/train_set.npy')
#original_test_set = np.load('../data/default_partition/test_set.npy')
# original_train_set = np.load('../data/train_set.npy')
# original_test_set = np.load('../data/test_set.npy')
original_train_set = np.load('../data/all_train.npy')

# fill in missing values
train_set = data_processing.missing_values(original_train_set, 
                                           method='median')
# test_set = data_processing.missing_values(original_test_set, 
#                                             method='median')

# get X and y
# remove first row. Scale by mean and stdev
X_train = train_set[1:,1:-1]
X_scaled_train = preprocessing.scale(X_train)
#min_max_scaler = preprocessing.MinMaxScaler()
#X_scaled_train = min_max_scaler.fit_transform(X_train)
y_train = train_set[1:,-1]
# X_test = test_set[:,1:-1]
# X_scaled_test = preprocessing.scale(X_test)
#min_max_scaler = preprocessing.MinMaxScaler()
#X_scaled_test = min_max_scaler.fit_transform(X_test)
# y_test = test_set[:,-1]


# Initialize kfold cross-validation object.
num_folds = 5
kf = KFold(n_splits=num_folds)

######################################
# random forest model: min leaf size #
######################################

# hyperparameters
n_estimators = 500 # number of trees in the forest
criterion = 'gini' # impurity function
min_samples_leaf = np.array([1000,800,600,400,200])

# Initialize kfold cross-validation object.
num_folds = 5
kf = KFold(n_splits=num_folds)
train_err = np.zeros(len(min_samples_leaf))
test_err = np.zeros(len(min_samples_leaf))


for train_index, test_index in kf.split(X_train):
    
    # Training and testing data points for this fold:
    x_train_kf, x_test_kf = X_train[train_index], X_train[test_index]
    y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

    # define the classifier
    clf = RandomForestClassifier(n_estimators = n_estimators, 
                                criterion = criterion)
    
    # clf_list = []
    
    # train the model
    for i,min_leaf_size in enumerate(min_samples_leaf):
        
        # train a decision tree model
        clf.set_params(min_samples_leaf = min_leaf_size)
        clf.fit(x_train_kf, y_train_kf)
        
        # clf_list.append(clf)
        
        # calculate and save training error
        y_predict = clf.predict(x_train_kf)
        train_err[i] += classification_err(y_predict, y_train_kf)
        
        # calculate and save test error
        y_predict = clf.predict(x_test_kf)
        test_err[i] += classification_err(y_predict, y_test_kf)

# normalize error
train_err /= num_folds
test_err /= num_folds
for i,min_leaf_size in enumerate(min_samples_leaf):
    print('minimum leaf size = {}'.format(min_leaf_size))
    print('In-sample error = {:.6f}'.format(train_err[i]))
    print('Out-of-sample error = {:.6f}'.format(test_err[i]))
    
# plot results
plt.figure()
plt.plot(min_samples_leaf, test_err, label='Testing error')
plt.plot(min_samples_leaf, train_err, label='Training error')
plt.xlabel('Minimum Node Size')
plt.ylabel('Classification error')
plt.title('Random Forest with Gini Impurity and Minimum Node Size')
plt.legend(loc=0, shadow=True, fontsize='x-large')
plt.show()


######################################
# random forest model: max leaf size #
######################################

# hyperparameters
n_estimators = 500 # number of trees in the forest
criterion = 'gini' # impurity function
max_depth = np.array([2,4,6,8,10,12,14,16,18,20,22,24,26,28,30])

# Initialize kfold cross-validation object.
num_folds = 5
kf = KFold(n_splits=num_folds)
train_err = np.zeros(len(max_depth))
test_err = np.zeros(len(max_depth))


for train_index, test_index in kf.split(X_train):
    
    # Training and testing data points for this fold:
    x_train_kf, x_test_kf = X_train[train_index], X_train[test_index]
    y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

    # define the classifier
    clf = RandomForestClassifier(n_estimators = n_estimators, 
                             criterion = criterion)

    # clf_list = []
    
    # train the model
    for i,max_depth_val in enumerate(max_depth):
        
        # train a decision tree model
        clf.set_params(max_depth = max_depth_val)
        clf.fit(x_train_kf, y_train_kf)
        
        # clf_list.append(clf)
        
        # calculate and save training error
        y_predict = clf.predict(x_train_kf)
        train_err[i] += classification_err(y_predict, y_train_kf)
        
        # calculate and save test error
        y_predict = clf.predict(x_test_kf)
        test_err[i] += classification_err(y_predict, y_test_kf)
   
# normalize error
train_err /= num_folds
test_err /= num_folds
for i,max_depth_val in enumerate(max_depth):
    print('maximus depth = {}'.format(max_depth_val))
    print('In-sample error = {:.6f}'.format(train_err[i]))
    print('Out-of-sample error = {:.6f}'.format(test_err[i]))
    
# plot results
plt.figure()
plt.plot(max_depth, test_err, label='Testing error')
plt.plot(max_depth, train_err, label='Training error')
plt.xlabel('Maximum Depth')
plt.ylabel('Classification error')
plt.title('Random Forest with Gini Impurity and Maximum Depth')
plt.legend(loc=0, shadow=True, fontsize='x-large')
plt.show()