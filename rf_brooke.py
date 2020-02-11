#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:06:03 2020

@author: yacong.zhou@gmail.com
"""

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import data_processing
from helper_brooke import classification_err

# partition data to get training and test sets
# original_train_set, original_test_set = data_processing.split_data(
    # data_file_name='../data/train.csv', train_portion=0.002, 
    # split_mode='random', save='npy')
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
original_train_set = np.load('../data/default_partition/train_set.npy')
original_test_set = np.load('../data/default_partition/test_set.npy')
# original_train_set = np.load('../data/train_set.npy')
# original_test_set = np.load('../data/test_set.npy')

# fill in missing values
train_set = data_processing.missing_values(original_train_set, 
                                           method='median')
test_set = data_processing.missing_values(original_test_set, 
                                            method='median')

# get X and y
X_train = train_set[:,:-1]
y_train = train_set[:,-1]
X_test = test_set[:,:-1]
y_test = test_set[:,-1]


# # Initialize kfold cross-validation object.
# num_folds = 5
# kf = KFold(n_splits=num_folds)
# # Initialize training and test error arrays:
# training_err_array = np.array([])
# test_err_array = np.array([])
# for train_index, test_index in kf.split(X_train):
#     # Training and testing data points for this fold:
#     x_train_kf, x_test_kf = x_data[train_index], x_data[test_index]
#     y_train_kf, y_test_kf = y_data[train_index], y_data[test_index]



######################################
# random forest model: min leaf size #
######################################
# hyperparameters
n_estimators = 500 # number of trees in the forest
criterion = 'gini' # impurity function


# define the classifier
clf = RandomForestClassifier(n_estimators = n_estimators, 
                             criterion = criterion)

# minimal leaf node size as early stopping criterion
min_samples_leaf = np.array([200,150,100,50])
train_err = np.zeros(len(min_samples_leaf))
test_err = np.zeros(len(min_samples_leaf))

# train the model
for i,min_leaf_size in enumerate(min_samples_leaf):
    print('minimum leaf size = {}'.format(min_leaf_size))
    
    # train a decision tree model
    clf.set_params(min_samples_leaf = min_leaf_size)
    clf.fit(X_train, y_train)
    
    # calculate and save training error
    y_predict = clf.predict(X_train)
    train_err[i] = classification_err(y_predict, y_train)
    print('In-sample error = {:.6f}'.format(train_err[i]))
    
    # calculate and save test error
    y_predict = clf.predict(X_test)
    test_err[i] = classification_err(y_predict, y_test)
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