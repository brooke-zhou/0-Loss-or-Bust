import xgboost as xgb
import numpy as np
import pandas as pd
# If I want to split my sets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn import preprocessing, svm, linear_model, cluster
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from data_processing import missing_values

import csv

# Needs further tinkering with
# https://xgboost.readthedocs.io/en/latest/parameter.html
# This model might currently be overfitting, but it should
# be a really good boosting algorithm.
def xgboost(dtrain, dtest):
    labels = dtrain[:, -1]
    dtrain = dtrain[:, :-1]
    dtrain = xgb.DMatrix(dtrain, label=labels)

    dtest = xgb.DMatrix(dtest)
    # specify parameters via map
    param = {'max_depth':6, 'normalize_type':'forest','objective':'binary:logistic' , 'eval_metric':'auc'}
    num_round = 10
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    preds = bst.predict(dtest)
    #print(preds)
    return preds

# Standard sklearn gaussian Naive Bayes
def gaussianNaiveBayes(dtrain, dtest):
    # can use split?
    y_train = dtrain[:, -1]
    x_train = dtrain[:, :-1]
    gnb = GaussianNB()
    predictions = gnb.fit(x_train, y_train).predict(dtest)
    return predictions

# Standard sklearn multinomial Naive Bayes
def multinomialNaiveBayes(dtrain, dtest):
    # can use split?
    y_train = dtrain[:, -1]
    x_train = dtrain[:, :-1]
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    gnb = MultinomialNB()
    predictions = gnb.fit(x_train, y_train).predict(dtest)
    return predictions

# Standard sklearn complement Naive Bayes
def complementNaiveBayes(dtrain, dtest):
    # can use split?
    y_train = dtrain[:, -1]
    x_train = dtrain[:, :-1]
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    gnb = ComplementNB()
    predictions = gnb.fit(x_train, y_train).predict(dtest)
    return predictions

# Standard sklearn bernoulli Naive Bayes
def bernoulliNaiveBayes(dtrain, dtest):
    # can use split?
    y_train = dtrain[:, -1]
    x_train = dtrain[:, :-1]
    gnb = BernoulliNB()
    predictions = gnb.fit(x_train, y_train).predict(dtest)
    return predictions

# Standard sklearn categorical Naive Bayes
def categoricalNaiveBayes(dtrain, dtest):
    # can use split?
    y_train = dtrain[:, -1]
    x_train = dtrain[:, :-1]
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    newscaler = preprocessing.MinMaxScaler()
    newscaler.fit(dtest)
    dtest = newscaler.transform(dtest)
    #print(x_train[x_train < 0])
    gnb = CategoricalNB()
    gnb.fit(x_train, y_train)
    print("GNB Features")
    print("GNB cat count Features")
    print(gnb.category_count_)
    print("GNB class count Features")
    print(gnb.class_count_)
    print("GNB feature log prob Features")
    print(gnb.feature_log_prob_)
    print("GNB n Features")
    print(gnb.n_features_)
    print("Length test")
    print(len(dtest[0]))
    predictions = gnb.predict(dtest)
    return predictions

# Standard sklearn sgdClassifier
def sgdClassifier(dtrain, dtest):
    clf = linear_model.SGDClassifier()
    clf.fit(dtrain[:, :-1], dtrain[:, -1])
    return clf.predict(dtest)

# Standard sklearn linearSVC
# didn't converge
def linearSVC(dtrain, dtest):
    clf = svm.LinearSVC(C=np.exp(-9))
    clf.fit(dtrain[:, :-1], dtrain[:, -1])
    return clf.predict(dtest)

# Standard sklearn kmeans
def kmeans(dtrain, dtest):
    # fix that the labels work properly
    clf = cluster.KMeans(n_clusters = 2)
    clf.fit(dtrain[:, :-1])
    return clf.predict(dtest)

# Standard sklearn lin regression
def linReg(dtrain, dtest):
    clf = linear_model.LinearRegression().fit(dtrain[:, :-1], dtrain[:, -1])
    return clf.predict(dtest)

# Standard sklearn logistic regression
def logReg(dtrain, dtest):
    clf = linear_model.LogisticRegression().fit(dtrain[:, :-1], dtrain[:, -1])
    return clf.predict(dtest)

# sklearn extratrees - I might play with this.
def extraTrees(dtrain, dtest):
    clf = ExtraTreesClassifier()
    clf.fit(dtrain[:, :-1], dtrain[:, -1])
    return clf.predict(dtest)

# MSFT's open source lightgbm - note: seems to produce
# similar results to xgboost
""" So far, lightgbm has produced the most optimal results as """
""" far as I can tell. It's a boosting algorithm, I think. """
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
def lightgbm(dtrain, dtest):
    train_data = lgb.Dataset(dtrain[:, :-1], label=dtrain[:, -1])
    param = {'num_leaves': 80, 'max_depth':7,'objective': 'binary', 'learning_rate':0.01}
    param['metric'] = 'auc'
    num_round = 100
    # separate command for k-fold cross validation:lgb.cv(...)
    bst = lgb.train(param, train_data, num_round)
    return bst.predict(dtest)

""" This imitates the test metric we will be judged on, the """
""" roc_auc metric. Cross validation is used for this.  """
def testNative(dtrain):
    # only test open source and linear
    # Initialize kfold cross-validation object with 10 folds:
    num_folds = 10
    kf = KFold(n_splits=num_folds)
    # Iterate through cross-validation folds:
    i = 1
    errors = [0, 0, 0, 0, 0, 0, 0]
    for train_index, test_index in kf.split(dtrain):
        # Print out test indices:
        print('Fold ', i, ' of ', num_folds, ' test indices:', test_index)

        # Training and testing data points for this fold:
        train, x_test = dtrain[train_index], dtrain[test_index]
        x_test, y_test = dtrain[:, :-1], dtrain[:, -1]
        yXGB = xgboost(train, x_test)
        yLin = linReg(train, x_test)
        yLight = lightgbm(train, x_test)
        yXGBErr = roc_auc_score(y_test, yXGB)
        yLinErr = roc_auc_score(y_test, yLin)
        yLightErr = roc_auc_score(y_test, yLight)
        xgbLinErr = roc_auc_score(y_test, (yXGB + yLin)/2)
        linLight = roc_auc_score(y_test, (yLight + yLin)/2)
        xgbLight = roc_auc_score(y_test, (yXGB + yLight)/2)
        errors[0] = errors[0] + yXGBErr
        errors[1] = errors[1] + yLinErr
        errors[2] = errors[2] + yLightErr
        errors[3] = errors[3] + xgbLinErr
        errors[4] = errors[4] + linLight
        errors[5] = errors[5] + xgbLight
        errors[6] = errors[6] + roc_auc_score(y_test, (yXGB + yLight + yLin)/3)



        i += 1
    print("XGB SCORE: " + str(errors[0]/10))
    print("Lin SCORE: " + str(errors[1]/10))
    print("LightGBM SCORE: " + str(errors[2]/10))
    print("1 and 2 SCORE: " + str(errors[3] / 10))
    print("2 and 3 SCORE: " + str(errors[4] / 10))
    print("1 and 3 SCORE: " + str(errors[5] / 10))
    print("all SCORE: " + str(errors[6] / 10))

    #print(xgboost(dtrain, dtest))

    #print(gaussianNaiveBayes(dtrain, dtest))

    #print(multinomialNaiveBayes(dtrain, dtest))

    #print(complementNaiveBayes(dtrain, dtest))

    #print(bernoulliNaiveBayes(dtrain, dtest))

    # doesn't work
    # print(categoricalNaiveBayes(dtrain, dtest))

    # Didn't converge one of two trials
    # print(sgdClassifier(dtrain, dtest))

    # didn't converge
    # print(linearSVC(dtrain, dtest))

    #print(linReg(dtrain, dtest))

    #print(logReg(dtrain, dtest))

    #print(extraTrees(dtrain, dtest))

    #print(lightgbm(dtrain, dtest))


""" The main engine that runs everything """
if __name__ == "__main__":
    df_train = pd.read_csv('../train.csv', index_col=0)
    df_test = pd.read_csv('../test.csv', index_col=0)

    #df_test = df_test.dropna()
    #df_train = df_train.dropna()
    # don't need labels
    dtrain = df_train.values
    dtrain = missing_values(dtrain, method="mean")
    print(dtrain)
    dtest = df_test.values
    dtest = missing_values(dtest, method="mean")
    #print(dtest)
    #indices = df_test['id']
    #testNative(dtrain)
    yres = lightgbm(dtrain, dtest)
    #print(indices)
    #print(yres)
    indices = range(len(dtrain), len(dtrain) + len(dtest))
    print(indices)
    print(len(indices))
    df = pd.DataFrame({'id': indices, 'Predicted': yres})
    #print(df['id'])
    #print(df['Predicted'])
    df.to_csv("submission.csv", index=False)

    #df.to_csv('submission.csv')
    #with open('submission.csv', 'w', newline='') as csvfile:


    #print(gaussianNaiveBayes(dtrain, dtest))

    #print(multinomialNaiveBayes(dtrain, dtest))

    #print(complementNaiveBayes(dtrain, dtest))

    #print(bernoulliNaiveBayes(dtrain, dtest))

    # doesn't work
    #print(categoricalNaiveBayes(dtrain, dtest))

    # Didn't converge one of two trials
    # print(sgdClassifier(dtrain, dtest))

    # didn't converge
    # print(linearSVC(dtrain, dtest))

    #print(linReg(dtrain, dtest))

    #print(logReg(dtrain, dtest))

    #print(extraTrees(dtrain, dtest))

    #print(lightgbm(dtrain, dtest))