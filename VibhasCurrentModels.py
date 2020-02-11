import xgboost as xgb
import numpy as np
import pandas as pd
# If I want to split my sets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn import preprocessing, svm, linear_model, cluster
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb

# Needs further tinkering with
# https://xgboost.readthedocs.io/en/latest/parameter.html
def xgboost(dtrain, dtest):
    labels = dtrain[:, -1]
    dtrain = dtrain[:, :-1]
    dtrain = xgb.DMatrix(dtrain, label=labels)

    dtest = xgb.DMatrix(dtest)
    # specify parameters via map
    param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
    num_round = 2
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
    clf = svm.LinearSVC()
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

# Standard sklearn logistric regression
def logReg(dtrain, dtest):
    clf = linear_model.LogisticRegression().fit(dtrain[:, :-1], dtrain[:, -1])
    return clf.predict(dtest)

# sklearn extratrees
def extraTrees(dtrain, dtest):
    clf = ExtraTreesClassifier()
    clf.fit(dtrain[:, :-1], dtrain[:, -1])
    return clf.predict(dtest)

# MSFT's open source lightgbm - note: seems to produce
# similar results to xgboost
def lightgbm(dtrain, dtest):
    train_data = lgb.Dataset(dtrain[:, :-1], label=dtrain[:, -1])
    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = 10
    # separate command for k-fold cross validation:lgb.cv(...)
    bst = lgb.train(param, train_data, num_round)
    return bst.predict(dtest)


if __name__ == "__main__":
    df_train = pd.read_csv('../train.csv', index_col=0)
    df_test = pd.read_csv('../test.csv', index_col=0)

    df_test = df_test.dropna()
    df_train = df_train.dropna()
    # don't need labels
    dtrain = df_train.values[1:]
    print(len(dtrain))
    dtest = df_test.values[1:]

    print(xgboost(dtrain, dtest))

    print(gaussianNaiveBayes(dtrain, dtest))

    print(multinomialNaiveBayes(dtrain, dtest))

    print(complementNaiveBayes(dtrain, dtest))

    print(bernoulliNaiveBayes(dtrain, dtest))

    # doesn't work
    #print(categoricalNaiveBayes(dtrain, dtest))

    # Didn't converge one of two trials
    # print(sgdClassifier(dtrain, dtest))

    # didn't converge
    # print(linearSVC(dtrain, dtest))

    print(linReg(dtrain, dtest))

    print(logReg(dtrain, dtest))

    print(extraTrees(dtrain, dtest))

    print(lightgbm(dtrain, dtest))