import xgboost as xgb
import numpy as np
import pandas as pd
# If I want to split my sets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn import preprocessing, svm, linear_model, cluster
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import statistics
from Ensemble_Attempts import get_trans_data
import random

#from numba import jit
from data_processing import missing_values
import random

import csv

# Needs further tinkering with
# https://xgboost.readthedocs.io/en/latest/parameter.html
# This model might currently be overfitting, but it should
# be a really good boosting algorithm.
defaults = {'objective': 'binary:logistic', 'tree_method': 'gpu_hist', 'gpu_id': 0, 'eta': 0.01, 'eval_metric': 'auc'}

def xgboost(dtrain, dtest, num_round=500, param=defaults, labels1=None):
    labels = dtrain[:, -1]
    dtrain = dtrain[:, :-1]
    dtrain = xgb.DMatrix(dtrain, label=labels)

    if labels1 is not None:
        dtest = xgb.DMatrix(dtest, label=labels1)
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        # specify parameters via map
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
    else:
        dtest = xgb.DMatrix(dtest)
        # specify parameters via map
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

# Standard sklearn lin regression + normalization
def linReg(dtrain, dtest):
    dtrain[:, :-1] = dtrain[:, :-1] / sum(dtrain[:, :-1])
    dtest = dtest /sum(dtest)
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
lightDefault = {'objective': 'binary', 'learning_rate':0.01, 'metric':'auc'}

def lightgbm(dtrain, dtest, num_round=100, param=lightDefault, labels=None):
    train_data = lgb.Dataset(dtrain[:, :-1], label=dtrain[:, -1])
    # separate command for k-fold cross validation:lgb.cv(...)
    bst = lgb.train(param, train_data, num_round)
    return bst.predict(dtest)

""" This imitates the test metric we will be judged on, the """
""" roc_auc metric. Cross validation is used for this.  """
def testNative(train):
    # only test open source and linear
    # Initialize kfold cross-validation object with 10 folds:
    dtrain = train
    #random.shuffle(dtrain)
    num_folds = 3
    kf = KFold(n_splits=num_folds)
    # Iterate through cross-validation folds:
    i = 1
    errors = [0, 0, 0, 0, 0, 0, 0]
    for train_index, test_index in kf.split(dtrain):
        # Print out test indices:
        print('Fold ', i, ' of ', num_folds, ' test indices:', test_index)

        # Training and testing data points for this fold:
        train, x_test = dtrain[train_index], dtrain[test_index]
        x_test, y_test = x_test[:, :-1], x_test[:, -1]
        yXGB = xgboost(train, x_test, labels1=y_test)
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
    print("XGB SCORE: " + str(errors[0]/num_folds))
    print("Lin SCORE: " + str(errors[1]/num_folds))
    print("LightGBM SCORE: " + str(errors[2]/num_folds))
    print("1 and 2 SCORE: " + str(errors[3] / num_folds))
    print("2 and 3 SCORE: " + str(errors[4] / num_folds))
    print("1 and 3 SCORE: " + str(errors[5] / num_folds))
    print("all SCORE: " + str(errors[6] / num_folds))


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
#@jit(nopython=True, parallel=True)
def testLong(dtrain, num_folds = 3):
    # models
    i = 0
    #kf = KFold(n_splits=num_folds)
    xgboostMods = ["gbtree", "gblinear", "dart"]
    treeMethod = "gpu_hist"
    etas = [0.01, 0.05, 0.1, 0.2, 0.3]
    numRounds = [500, 500, 300, 200, 100]
    scaleWeights = list(range(10, 0, -1)) + \
                   [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    max_bin = [256, 300, 400, 512]
    parallelTrees = range(1, 11)
    maxDepth = range(1, 21)
    minChildWeight = range(1, 21)
    subsamples = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Following only for dart
    dtWt = ["uniform", "weighted"]
    nmTyp = ["tree", "forest"]
    rateDrop = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #featureSel = ["cyclic", "shuffle", "random", "greedy", "thrifty"]
    # Light gbm
    lgbm = ["gbdt", "rf", "dart", "goss"]
    min_data_in_leaf = list(range(10, 100, 10)) + list(range(1000, 10000, 1000))
    max_depth = range(1, 20)
    bagging_fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    bagging_freq = range(1, 11)
    feature_fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    drop_rate = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    max_drop = list(range(40, 100, 10)) + list(range(100, 1000, 100))
    lightDefault = {'objective': 'binary', 'learning_rate': 0.01, 'metric': 'auc', 'device': 'gpu',
                    'gpu_platform_id': 0, 'gpu_device_id': 0}
    defaultXGB = {'objective': 'binary:logistic', 'eta': 0.01, 'tree_method': 'gpu_hist', 'gpu_id': 0,
                  'predictor': 'gpu_predictor', 'eval_metric': 'auc'}

    fileErrs = "XGB tuning"+str(num_folds)+".txt"
    file2Errs = "lightGBMtuning"+str(num_folds)+".txt"
    prefix = str(num_folds) + " "
    #indices = [[dtrain[len(dtrain)/3:], dtrain[:len(dtrain)/3]],
    #    [dtrain[:len(dtrain)/3] +dtrain[2*len(dtrain)/3:],
    #dtrain[len(dtrain)/3:2*len(dtrain)/3]], [dtrain[:2*len(dtrain)], dtrain[2*len(dtrain):]]]
    kf = KFold(n_splits=num_folds)
    for train_index, test_index in kf.split(dtrain):
        # Training and testing data points for this fold:
        train, x_test = dtrain[train_index], dtrain[test_index]
        x_test, y_test = x_test[:, :-1], x_test[:, -1]
        for method in xgboostMods:
            print(defaultXGB)
            testMethod = defaultXGB.copy()
            testMethod['booster'] = method
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB " + method + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB method: " + method + " error: " + str(err) + "\n")
            f.close()
        print(defaultXGB)
        for nRd in numRounds:
            for eVal in etas:
                testMethod = defaultXGB.copy()
                testMethod['eta'] = eVal
                yVals = xgboost(train, x_test, num_round=nRd, param=testMethod, labels1=y_test)
                err = roc_auc_score(y_test, yVals)
                fileYs = prefix + "XGB eval numRounds " + str(eVal) + ", " + str(nRd) + ".csv"
                df = pd.DataFrame({'Predicted': yVals})
                df.to_csv(fileYs, index=False)
                f = open(fileErrs, "a")
                f.write("The XGB evals numRounds " + str(eVal) + ", " + str(nRd) + " error: " + str(err) + "\n")
                f.close()
        for sWts in scaleWeights:
            testMethod = defaultXGB.copy()
            testMethod['scale_pos_weight'] = sWts
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB scale positive weights " + str(sWts) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB sample weights " + str(sWts) + " error: " + str(err) + "\n")
            f.close()
        for mBin in max_bin:
            testMethod = defaultXGB.copy()
            testMethod['max_bin'] = mBin
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB scale max bins " + str(mBin) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB max bin " + str(mBin) + " error: " + str(err) + "\n")
            f.close()
        for nTree in parallelTrees:
            testMethod = defaultXGB.copy()
            testMethod['num_parallel_tree'] = nTree
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB scale n trees " + str(nTree) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB number of trees " + str(nTree) + " error: " + str(err) + "\n")
            f.close()
        for mD in maxDepth:
            testMethod = defaultXGB.copy()
            testMethod['max_depth'] = mD
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB scale max depths " + str(mD) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB max depth " + str(mD) + " error: " + str(err) + "\n")
            f.close()
        for mCW in minChildWeight:
            testMethod = defaultXGB.copy()
            testMethod['min_child_weight'] = mCW
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB scale min child weights " + str(nTree) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB min child weights " + str(mCW) + " error: " + str(err) + "\n")
            f.close()
        for sSamples in subsamples:
            testMethod = defaultXGB.copy()
            testMethod['subsample'] = sSamples
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB scale subsamples " + str(sSamples) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB subsamples " + str(sSamples) + " error: " + str(err) + "\n")
            f.close()
        for wt in dtWt:
            testMethod = defaultXGB.copy()
            testMethod['booster'] = 'dart'
            testMethod['sample_type'] = wt
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB " + wt + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB sample type " + wt + " error: " + str(err) + "\n")
            f.close()
        for typ in nmTyp:
            testMethod = defaultXGB.copy()
            testMethod['booster'] = 'dart'
            testMethod['normalize_type'] = typ
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB " + typ + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB types " + typ + " error: " + str(err) + "\n")
            f.close()
        for rt in rateDrop:
            testMethod = defaultXGB.copy()
            testMethod['booster'] = 'dart'
            testMethod['rate_drop'] = rt
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB rate drop " + str(rt) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB rates " + str(rt) + " error: " + str(err) + "\n")
            f.close()

        for method in lgbm:
            testMethod = lightDefault.copy()
            testMethod['boosting'] = method
            if method == 'rf':
                for fract in bagging_fraction:
                    for freq in bagging_freq:
                        testMethod = lightDefault.copy()
                        testMethod['bagging_fraction'] = fract
                        testMethod['bagging_freq'] = freq
                        yVals = lightgbm(train, x_test, param=testMethod)  # , labels1=y_test)
                        err = roc_auc_score(y_test, yVals)
                        fileYs = prefix + "Light GBM booster bag frac " + str(fract) \
                                 + " bag freq " + str(freq) + ".csv"
                        df = pd.DataFrame({'Predicted': yVals})
                        df.to_csv(fileYs, index=False)
                        f = open(file2Errs, "a")
                        f.write("The Light GBM with rf and bagging frac: " + str(fract)
                                + "bagging freq" + str(freq) + " error: "
                                + str(err) + "\n")
                        f.close()
            else:
                yVals = lightgbm(train, x_test, param=testMethod)#, labels1=y_test)
                err = roc_auc_score(y_test, yVals)
                fileYs = prefix + "Light GBM booster " + method + ".csv"
                df = pd.DataFrame({'Predicted': yVals})
                df.to_csv(fileYs, index=False)
                f = open(file2Errs, "a")
                f.write("The Light GBM method: " + method + " error: " + str(err) + "\n")
                f.close()
        for minD in min_data_in_leaf:
            testMethod = lightDefault.copy()
            testMethod['min_data_in_leaf'] = minD
            yVals = lightgbm(train, x_test, param=testMethod)#, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "Light GBM booster min data " + str(minD) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(file2Errs, "a")
            f.write("The Light GBM min data leaf: " + str(minD) + " error: " + str(err) + "\n")
            f.close()
        for maxD in max_depth:
            testMethod = lightDefault.copy()
            testMethod['max_depth'] = maxD
            yVals = lightgbm(train, x_test, param=testMethod)#, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "Light GBM booster max depth " + str(maxD) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(file2Errs, "a")
            f.write("The Light GBM max data leaf: " + str(maxD) + " error: " + str(err) + "\n")
            f.close()
        for fract in bagging_fraction:
            for freq in bagging_freq:
                testMethod = lightDefault.copy()
                testMethod['bagging_fraction'] = fract
                testMethod['bagging_freq'] = freq
                yVals = lightgbm(train, x_test, param=testMethod)#, labels1=y_test)
                err = roc_auc_score(y_test, yVals)
                fileYs = prefix + "Light GBM booster bag frac " + str(fract) \
                         + " bag freq " + str(freq) + ".csv"
                df = pd.DataFrame({'Predicted': yVals})
                df.to_csv(fileYs, index=False)
                f = open(file2Errs, "a")
                f.write("The Light GBM bagging frac: " + str(fract)
                        + "bagging freq" + str(freq) + " error: "
                        + str(err) + "\n")
                f.close()
        for ff in feature_fraction:
            testMethod = lightDefault.copy()
            testMethod['feature_fraction'] = ff
            yVals = lightgbm(train, x_test, param=testMethod)#, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "Light GBM booster feature fraction " + str(ff) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(file2Errs, "a")
            f.write("The Light GBM feature fraction: " + str(ff) + " error: " + str(err) + "\n")
            f.close()
        for dr in drop_rate:
            testMethod = lightDefault.copy()
            testMethod['booster'] = 'dart'
            testMethod['drop_rate'] = dr
            yVals = lightgbm(train, x_test, param=testMethod)#, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "Light GBM booster drop rate " + str(dr) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(file2Errs, "a")
            f.write("The Light GBM drop rate: " + str(dr) + " error: " + str(err) + "\n")
            f.close()
        for maxD in max_drop:
            testMethod = lightDefault.copy()
            testMethod['max_drop'] = maxD
            yVals = lightgbm(train, x_test, param=testMethod)#, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "Light GBM booster max drop " + str(maxD) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(file2Errs, "a")
            f.write("The Light GBM max drop: " + str(maxD) + " error: " + str(err) + "\n")
            f.close()


def filteredData(dataStuff):
    dataStuff.loc[dataStuff[['opened_position_qty ']].isnull().any(axis=1), 'opened_position_qty '] = 0
    dataStuff.loc[dataStuff[['closed_position_qty']].isnull().any(axis=1), 'closed_position_qty'] = 0
    #for key in list(dataStuff.columns):
    #    print("." + key + ".")
    return dataStuff

def testBlend(dtrain, num_folds = 3, xgbParams=[], lightgbmParams = []):
    scores = [0 for i in range(12)]
    kf = KFold(n_splits=num_folds)
    i = 0
    lgbm = ["gbdt", "rf", "dart", "goss"]
    fileName = "Results of test " + str(num_folds) + ".txt"
    print(fileName)
    #file2Errs = "lightGBM results" + str(num_folds) + ".txt"
    for train_index, test_index in kf.split(dtrain):
        f = open(fileName, "a")
        f.write("KFold: " + str(i) + " of " + str(num_folds) + ".\n")
        f.close()
        i = i + 1

        # Training and testing data points for this fold:
        train, x_test = dtrain[train_index], dtrain[test_index]
        x_test, y_test = x_test[:, :-1], x_test[:, -1]
        collectScores = []
        lightScores = []

        for param in xgbParams:
            yVals = xgboost(train, x_test, param=param, labels1=y_test)
            collectScores.append(yVals.tolist())
        for param in lightgbmParams:
            yVals = lightgbm(train, x_test, param=param, labels=y_test)
            lightScores.append(yVals.tolist())
            print([statistics.mean([collectScores[0][i]] + [lightScores[j][i] for j in range(len(lightScores))]) for i in range(len(lightScores[0]))])
        print(len(collectScores[0]))
        print(len(lightScores[0]))
        print((collectScores[0][0]))

        agg1 = [statistics.mean([collectScores[0][i], lightScores[0][i]]) for i in range(len(lightScores[0]))]
        agg2 = [statistics.mean([collectScores[0][i], lightScores[1][i]]) for i in range(len(lightScores[0]))]
        agg3 = [statistics.mean([collectScores[0][i]] + [lightScores[j][i] for j in range(len(lightScores))]) for i in range(len(lightScores[0]))]
        agg4 = [statistics.mean([lightScores[j][i] for j in range(len(lightScores))]) for i in range(len(lightScores[0]))]
        agg5 = [statistics.mean([lightScores[j][i] for j in range(1)]) for i in range(len(lightScores[0]))]
        agg6 = [statistics.mean([lightScores[0][i]] + [lightScores[2][i]]) for i in range(len(lightScores[0]))]
        agg7 = [statistics.median([collectScores[0][i]] + [lightScores[j][i] for j in range(len(lightScores))]) for i in range(len(lightScores[0]))]
        currScores = [roc_auc_score(y_test, collectScores[0]), roc_auc_score(y_test, lightScores[0]),
                      roc_auc_score(y_test, lightScores[1]),
                      roc_auc_score(y_test, lightScores[2]), roc_auc_score(y_test, lightScores[3]),
                      roc_auc_score(y_test, agg1),
                      roc_auc_score(y_test, agg2),
                  roc_auc_score(y_test, agg3),
                  roc_auc_score(y_test, agg4), roc_auc_score(y_test, agg5),
                      roc_auc_score(y_test, agg6), roc_auc_score(y_test, agg7)]
        for i in range(len(scores)):
            scores[i] = scores[i] + currScores[i]


        f = open(fileName, "a")
        f.write("The stats " + str(currScores) + ".\n")
        f.close()
    for i in range(len(scores)):
        scores[i] = scores[i] / (num_folds)
    f = open(fileName, "a")
    f.write("The total stats " + str(scores) + ".\n")
    f.close()

def fineTuneTesting(dtrain, num_folds = 3):
    # models
    xgboostMods = ["gbtree", "gblinear", "dart"]
    etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
    numRounds = [500, 400, 300, 200, 100]
    scaleWeights = list(range(10, 0, -1)) + \
                   [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    max_bin = [256, 300, 400, 512, 1000]
    maxDepth = range(3, 8)
    #minChildWeight = range(1, 21)
    subsamples = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # For LightGBM only
    rateDrop = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Light gbm
    lgbm = ["gbdt", "rf"]

    max_depth = range(3, 8)
    bagging_fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    bagging_freq = range(1, 20)
    feature_fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    drop_rate = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    max_drop = list(range(40, 100, 10)) + list(range(100, 1000, 100))
    lightDefault = {'objective': 'binary', 'learning_rate': 0.01, 'metric': 'auc', 'device': 'gpu',
                    'gpu_platform_id': 0, 'gpu_device_id': 0}
    defaultXGB = {'objective': 'binary:logistic', 'eta': 0.01, 'tree_method': 'gpu_hist', 'gpu_id': 0,
                  'predictor': 'gpu_predictor', 'eval_metric': 'auc'}

    fileErrs = "XGB params results" + str(num_folds) + ".txt"
    file2Errs = "lightGBM results" + str(num_folds) + ".txt"
    prefix = str(num_folds) + " "

    i = 0

    kf = KFold(n_splits=num_folds)
    for train_index, test_index in kf.split(dtrain):
        f = open(fileErrs, "a")
        f.write("KFold: " + str(i) + " of " + str(num_folds) + ".\n")
        f.close()

        f = open(file2Errs, "a")
        f.write("KFold: " + str(i) + " of " + str(num_folds) + ".\n")
        f.close()
        i = i + 1
        # Training and testing data points for this fold:
        train, x_test = dtrain[train_index], dtrain[test_index]
        x_test, y_test = x_test[:, :-1], x_test[:, -1]
        maxResult = 0
        maxRd = 0
        maxE = 0
        for nRd in numRounds:
            for eVal in etas:
                testMethod = defaultXGB.copy()
                testMethod['eta'] = eVal
                yVals = xgboost(train, x_test, num_round=nRd, param=testMethod, labels1=y_test)
                err = roc_auc_score(y_test, yVals)
                if err > maxResult:
                    maxRd, maxE = nRd, eVal
        f = open(fileErrs, "a")
        f.write("The XGB evals numRounds " + str(maxE) + ", " + str(maxRd) + " error: " + str(err) + "\n")
        f.close()

        maxWeight = 0
        maxScale = 0
        for sWts in scaleWeights:
            testMethod = defaultXGB.copy()
            testMethod['scale_pos_weight'] = sWts
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            if err > maxWeight:
                maxWeight, maxScale = sWts, err
        f = open(fileErrs, "a")
        f.write("The XGB sample weights " + str(maxWeight) + " error: " + str(maxScale) + "\n")
        f.close()

        aBin = 0
        binRes = 0
        for mBin in max_bin:
            testMethod = defaultXGB.copy()
            testMethod['max_bin'] = mBin
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            if err > binRes:
                aBin, binRes = mBin, err
        f = open(fileErrs, "a")
        f.write("The XGB max bin " + str(aBin) + " error: " + str(binRes) + ".\n")
        f.close()

        mDep = 0
        mVal = 0
        for mD in maxDepth:
            testMethod = defaultXGB.copy()
            testMethod['max_depth'] = mD
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            if err > mVal:
                mVal, mDep = err, mD
        f = open(fileErrs, "a")
        f.write("The XGB max depth " + str(mDep) + " error: " + str(mVal) + "\n")
        f.close()
        '''for mCW in minChildWeight:
            testMethod = defaultXGB.copy()
            testMethod['min_child_weight'] = mCW
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            fileYs = prefix + "XGB scale min child weights " + str(nTree) + ".csv"
            df = pd.DataFrame({'Predicted': yVals})
            df.to_csv(fileYs, index=False)
            f = open(fileErrs, "a")
            f.write("The XGB min child weights " + str(mCW) + " error: " + str(err) + "\n")
            f.close()'''
        maxScore = 0
        maxSamp = 0
        for sSamples in subsamples:
            testMethod = defaultXGB.copy()
            testMethod['subsample'] = sSamples
            yVals = xgboost(train, x_test, param=testMethod, labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            if err > maxScore:
                maxScore, maxSamp = err, sSamples
        f = open(fileErrs, "a")
        f.write("The XGB subsamples " + str(maxSamp) + " error: " + str(maxScore) + "\n")
        f.close()

        maxScore = 0
        mFra = 0
        mFre = 0
        for fract in bagging_fraction:
            for freq in bagging_freq:
                testMethod = lightDefault.copy()
                testMethod['boosting'] = 'rf'
                testMethod['bagging_fraction'] = fract
                testMethod['bagging_freq'] = freq
                yVals = lightgbm(train, x_test, param=testMethod)  # , labels1=y_test)
                err = roc_auc_score(y_test, yVals)
                if err > maxScore:
                    maxScore, mFra, mFre = err, fract, freq

        f = open(file2Errs, "a")
        f.write("The Light GBM with rf and bagging frac: " + str(mFra)
                + "bagging freq" + str(mFre) + " error: "
                + str(maxScore) + "\n")
        f.close()

        maxScore = 0
        mD = 0
        for maxD in max_depth:
            testMethod = lightDefault.copy()
            testMethod['max_depth'] = maxD
            yVals = lightgbm(train, x_test, param=testMethod)  # , labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            if err > maxScore:
                maxScore, mD = err, maxD
        f = open(file2Errs, "a")
        f.write("The Light GBM max data leaf: " + str(mD) + " error: " + str(maxScore) + "\n")
        f.close()

        maxScore = 0
        mf = 0
        for ff in feature_fraction:
            testMethod = lightDefault.copy()
            testMethod['feature_fraction'] = ff
            yVals = lightgbm(train, x_test, param=testMethod)  # , labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            if err > maxScore:
                maxScore, mf = err, ff
        f = open(file2Errs, "a")
        f.write("The Light GBM feature fraction: " + str(mf) + " error: " + str(maxScore) + "\n")
        f.close()

        maxScore = 0
        mDr = 0
        for dr in drop_rate:
            testMethod = lightDefault.copy()
            testMethod['booster'] = 'dart'
            testMethod['drop_rate'] = dr
            yVals = lightgbm(train, x_test, param=testMethod)  # , labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            if err > maxScore:
                maxScore, mDr = err, dr
        f = open(file2Errs, "a")
        f.write("The Light GBM drop rate: " + str(mDr) + " error: " + str(maxScore) + "\n")
        f.close()

        maxScore = 0
        mD = 0
        for maxD in max_drop:
            testMethod = lightDefault.copy()
            testMethod['max_drop'] = maxD
            yVals = lightgbm(train, x_test, param=testMethod)  # , labels1=y_test)
            err = roc_auc_score(y_test, yVals)
            if err > maxScore:
                maxScore, mD = err, maxD
        f = open(file2Errs, "a")
        f.write("The Light GBM max drop: " + str(mD) + " error: " + str(maxScore) + "\n")
        f.close()



""" The main engine that runs everything """
if __name__ == "__main__":
    # Read the data in
    df_train = pd.read_csv('../train.csv', index_col=0)
    df_test = pd.read_csv('../test.csv', index_col=0)
    # Take care of missing values
    df_train = filteredData(df_train)
    # Needed for writing indices to our submission
    indices = range(len(df_train), len(df_train) + len(df_test))
    # Handle missing values
    df_test = filteredData(df_test)

    # don't need labels
    dtrain = df_train.values[1:, :-1]
    labels = df_train.values[1:, -1]
    dtest = df_test.values

    # normalize/scale data
    # try shuffling
    dtrain = get_trans_data(dtrain)
    dtrain = np.array([dtrain[i] + [labels[i]] for i in range(len(dtrain))])
    #random.shuffle(dtrain)
    dtest = np.array(list(get_trans_data(dtest)))

    # This was me testing all feasible parameters.
    # Takes forever to run, even on GPU.
    #for i in [10, 6, 5, 3]:
    #    testLong(dtrain, num_folds=i)

    # This was me testing a few parameters I couldn't
    # fully pinpoint a pattern for and seemed worth
    # testing.
    #fineTuneTesting(dtrain, num_folds=10)

    # Parameters for nearly all models for XGBoost.
    defaultXGB = {'objective': 'binary:logistic', 'eta': 0.01, 'tree_method': 'gpu_hist', 'gpu_id': 0,
                 'predictor': 'gpu_predictor', 'eval_metric': 'auc'}
    # Parameters for nearly all models for LightGBM.
    lightDefault = {'objective': 'binary', 'learning_rate': 0.01, 'metric': 'auc', 'device': 'gpu',
                    'gpu_platform_id': 0, 'gpu_device_id': 0}
    dartXGB = defaultXGB.copy()
    regLight = lightDefault.copy()
    rf = lightDefault.copy()
    dartL = lightDefault.copy()
    goss = lightDefault.copy()
    defaultXGB['eta'] = 0.01
    nRound = 100

    # Below are models used for the XGBoost
    # package.

    # Default: Gradient boosting tree

    # Balance of weights: 10 brute force tested
    # seemed ideal for all cuts.
    defaultXGB['scale_pos_weight'] = 10
    # Depth of tree. Needs to be high enough
    # to fit the model, but can't be too deep
    # or complexity will lead to overfitting.
    defaultXGB['depth'] = 7
    # Randomly sample 0.2 of the training data before
    # growing trees to avoid overfitting.
    defaultXGB['subsample'] = 0.2

    # Dart: Gradient boosting with dropping trees
    # to reduce over-fitting
    dartXGB['booster'] = 'dart'
    # Sample type is weighted
    dartXGB['sample_type'] = 'weighted'
    # Normalize algorithm is tree as it seemed to
    # empirically do well, though it's a possibility that
    # in theory forest would do better.
    dartXGB['normalize_type'] = 'forest'
    # Dropout rate to avoid overfitting
    dartXGB['rate_drop'] = 0.1

    # Below are models used for Microsoft's
    # LightGBM open source software.

    # Default Gradient Boosting Decision tree
    # Minimal number of data in one leaf. Can help
    # with overfitting
    regLight['min_data_in_leaf'] = 2000
    # Again, we want the model to be complex enough
    # to fit the data but not enough to overfit.
    regLight['max_depth'] = 7
    # Randomly select part of features before
    # training each tree.
    regLight['feature_fraction'] = 0.5
    regLight['num_leaves'] = 70

    # Random forest, with 0.1 samples bagged/randomly
    # selected, for each iteration.
    rf['boosting'] = 'rf'
    rf['bagging_fraction'] = 0.1
    rf['bagging_freq'] = 1

    # Default dart booster for light gbm
    # Dropouts meet Multiple Additive Regression Trees
    dartL['boosting'] = 'dart'
    # Gradient-based One-Side Sampling
    goss['boosting'] = 'goss'


    # This is to test blending
    #for i in [10, 6, 3]:
    #    testBlend(dtrain, num_folds=i, xgbParams=[defaultXGB], lightgbmParams=[regLight, rf, dartL, goss])

    # Here is actual code to generate the data
    # with the ideal blending/model necessary.
    #print("GBC...")
    #random_state = 54321
    #model1 = GradientBoostingClassifier(random_state=random_state)
    #model1.fit(dtrain[:, :-1], labels)
    #gbRes = model1.predict(dtest)
    print("Starting...")
    xgbResult = xgboost(dtrain, dtest, param = defaultXGB)
    print("XGBoost done...")
    xgbDart = xgboost(dtrain, dtest, param = dartXGB)
    print("XGBoost dart done...")
    regResult = lightgbm(dtrain, dtest, param=regLight)


    print("LightGBM done...")
    #rfResult = lightgbm(dtrain, dtest, param=rf)
    print("RF  done...")
    #dartLight = lightgbm(dtrain, dtest, param=dartL)
    print("LightGBM dart done...")
    #gossRes = lightgbm(dtrain, dtest, param=goss)
    print("Goss done...")

    #yres = regResult
    yres = (xgbResult + xgbDart + regResult)/3

    df = pd.DataFrame({'id': indices, 'Predicted': yres})

    df.to_csv("submission.csv", index=False)




    '''lightDefault = {'objective': 'binary', 'learning_rate': 0.01, 'metric': 'auc', 'device': 'gpu',
                    'gpu_platform_id': 0, 'gpu_device_id': 0}
    rfLight = lightDefault.copy()
    goss = lightDefault.copy()
    lightDefault['min_data_in_leaf'] = 8000
    #lightDefault['max_depth'] = 11
    lightDefault['bagging_fraction'] = 0.4
    lightDefault['bagging_freq'] = 1
    lightDefault['feature_fraction'] = 0.6
    lightDefault['drop_rate'] = 0.2
    lightDefault['max_drop'] = 70'''
    #lightRes = lightgbm(dtrain, dtest)

    '''rfLight['boosting'] = 'rf'
    rfLight['bagging_fraction'] = 0.3
    rfLight['bagging_freq'] = 1
    rfRes = lightgbm(dtrain, dtest, param=rfLight, labels=labels)'''


    '''goss['boosting'] = 'goss'
    gossRes = lightgbm(dtrain, dtest, param=goss, labels=labels)'''

    #yres = (lightRes)


    #df = pd.DataFrame({'id': indices, 'Predicted': yres})

    #df.to_csv("submission.csv", index=False)


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