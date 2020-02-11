#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:49:05 2020

@author: yacong.zhou@gmail.com
"""

def classification_err(y, real_y):
    """
    This function returns the classification error between two equally-sized vectors of 
    labels; this is the fraction of samples for which the labels differ.
    
    Inputs:
        y: (N, ) shaped array of predicted labels
        real_y: (N, ) shaped array of true labels
    Output:
        Scalar classification error
    """
    #==============================================
    # TODO: Implement the classification_err function,
    # based on the above instructions.
    #==============================================   
    
    if len(y) == len(real_y):
        # initlaize counter
        differ_label_counter = 0
        for i in range(len(y)):
            # test if a data point has differerent labels 
            if y[i] != real_y[i]:
                differ_label_counter += 1
    else:
        print('length does not match')
    return differ_label_counter/len(y)