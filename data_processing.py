import numpy as np

def split_data(data_file_name='../data/train.csv', 
               train_portion=0.8, split_mode='random', save='npy'):
    """
    Split original data into training and testing sets. 
    
    Parameters
    ----------
    data_file_name : str, optional
        Name of data set. The default is '../data/train.csv'.
    train_portion : float in [0,1], optional
        The portion of data to be in training set. The default is 0.8.
    split_mode : str, optional
        Split method. 
        'random': randomly select data to be in each subset. 
        'first': first train_portion in all data to be in training set. 
        'last': first train_portion in all data to be in training set. 
    save : str, optional
        Save the output arrays or not and in which format.
        'npy' : save as binary files in NumPy .npy format.
        'csv' : save as .csv format.
        'no' : do not save output.

    Returns
    -------
    train_set : numpy array of size (train_set, 1)
    test_set : numpy array of size (test_set, 1)

    """
    all_data = np.genfromtxt(data_file_name, delimiter = ',')
    data_size = len(all_data)
    train_size = round(data_size*train_portion)
    train_mask = np.full(data_size, False)
    
    if split_mode == 'random':
        import random
        random.seed(1552020)
        train_mask[:train_size] = True
        np.random.shuffle(train_mask)
    elif split_mode == 'first':
        train_mask[:train_size] = True
    elif split_mode == 'last':
        train_mask[data_size-train_size:] = True
    else:
        print('Not a valid split mode.')
        
    test_mask = [not i for i in train_mask]
    train_set = all_data[train_mask]
    test_set = all_data[test_mask]
    
    if save == 'npy':
        np.save('../data/train_set.npy',train_set)
        np.save('../data/test_set.npy',test_set)
    elif save == 'csv':
        np.savetxt('../data/train_set.csv', train_set, delimiter=',')
        np.savetxt('../data/test_set.csv', test_set, delimiter=',')
    elif save == 'no':
        print('Splitted data not saved to disk.')
    else: 
        print('Not a valid save option.')
        
    return train_set, test_set

def missing_values(original_data, method='omit', 
                   supply_data=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                                15,16,17,18,19,20,21,22,23,24,25,26]):
    """
    Replace missing values in original data according to given rules.

    Parameters
    ----------
    original_data : numpy array
        The data set containing NaN.
    method : str, optional
        'omit' : remove rows containing NaN. Default.
        'mean' : replace NaN by the mean of its column.
        'median' : replace NaN by the median of its column.
        'zeros' : replace NaN by 0.
        'change_and_add_flags' : replace NaN by the values specified in 
         supply_data at each corresponding columns. Then add new columns 
         with 0 = not NaN and 1 = is NaN.
    supply_data : list of floats, optional
        values to replace NaN in each column. The default is 
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26].
        'imputation' : fill in missing values by simple machine learning

    Returns
    -------
    new_data : numpy array of size (row_original_data, 
               column_original_data + n_column_containing_missing)
        The processed data array.

    """
    if method == 'omit':
        new_data = original_data[~np.isnan(original_data).any(axis=1)]
        
    elif method == 'mean':
        non_nan_data = original_data[~np.isnan(original_data).any(axis=1)]
        mean_row = np.mean(non_nan_data, axis=0)
        for i_column in range(len(mean_row)):
            original_data[:,i_column] = np.nan_to_num(original_data[:,i_column], 
                                                      nan=mean_row[i_column])
            new_data = original_data
            
    elif method == 'median':
        non_nan_data = original_data[~np.isnan(original_data).any(axis=1)]
        median_row = np.median(non_nan_data, axis=0)
        for i_column in range(len(median_row)):
            original_data[:,i_column] = np.nan_to_num(original_data[:,i_column], 
                                                      nan=median_row[i_column])
            new_data = original_data
            
    elif method == 'zeros':
        new_data = np.nan_to_num(original_data, nan=0.0)
        
    elif method == 'change_and_add_flags':
        import numpy.ma as ma
        for i_column in range(27): # 27 columns in total, not including y
            new_column = np.zeros(len(original_data[:,i_column]))
            mask = np.ma.masked_invalid(original_data[:,i_column]).mask
            new_column[mask] = 1
            if np.sum(new_column) != 0:
                new_column = np.expand_dims(new_column, axis=0)
                new_column = new_column.transpose()
                np.append(original_data,new_column,axis=1)
                original_data[:,i_column] = np.nan_to_num(original_data[:,i_column], 
                                                      nan=supply_data[i_column])
                new_data = original_data
                
    elif method == 'imputation':
        # to do
        pass
    
    else: 
        print('Invalid option for treating missing data.')
    
    return new_data