#import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

from KagglegymEmulation import make
from KagglegymEmulation import r_score


import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)

import sklearn.decomposition as decomp

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import LabelBinarizer

from scipy import optimize


from sklearn.cluster import KMeans

excl = ["id", "sample", "y", "timestamp"]

# --------------------------------------------------------------------------------

def checkNewIdfromFeatures(train,test):

    df_test = test.copy()
    df_train = train.copy()

    uniq_timestamp = df_test["timestamp"].unique()
    uniq_id = sorted(df_test["id"].unique())
    #print uniq_id

    print("-- selected features timestamp: %d" % uniq_timestamp)

    train_uniq_id = sorted(df_train["id"].unique())
    train_uniq_timestamp = sorted(   df_train["timestamp"].unique()  )

    new_test_uniq_id = np.array( [ test_id for test_id in uniq_id if test_id not in train_uniq_id] )

    missing_training_id = [miss_id for miss_id in train_uniq_id if miss_id not in uniq_id]

    #print("   train timestamp length %d , train id length %d" % ( len(train_uniq_timestamp),len(train_uniq_id) ) )
    print("   test id length %d" % ( len(uniq_id) ) )
    #print("   new test unique id %s" % new_test_uniq_id)
    print("   new test unique id lengyj %d" % len(new_test_uniq_id))


    print("   missing training id count from features  %s" % len(missing_training_id) )



def TraingDataStd(train,test):

    df_train = train.copy()
    df_test = test.copy()

    uniq_timestamp = df_train["timestamp"].unique()

    col = [c for c in df_train.columns if c not in excl]

    TrainStds = []
    for timestamp_id in uniq_timestamp:

        df_one = df_train[df_train.timestamp == timestamp_id].copy()
        y_one = df_one['y'].values.tolist()

        TrainStds.append(np.std(y_one))


    #plt.scatter(range(len(TrainStds)),TrainStds ,c='r'  )
    #plt.show()

    train_feat = df_train[col]
    train_feat = train_feat.fillna(train_feat.mean())
    X = np.array(train_feat)

    null_number_by_row = train_feat.isnull().sum(axis=1)

    X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)




    Xtrain = X
    km = KMeans(n_clusters=20)
    km.fit(Xtrain)

    class_predict = km.predict(Xtrain)

    training =  df_train.copy()
    #training.iloc[:,2:110] = Xtrain.copy()
    training.loc[:,"kcls"] = class_predict.astype(np.int)

    print training.head()

    #fg , ax = plt.figure(figsize=(12,10))
    """
    g = sns.FacetGrid(training, col="kcls", hue="id")
    g.map(plt.scatter, "timestamp", "y", alpha=.7)
    g.add_legend();

    #sns.factorplot(x="timestamp",y="id",hue="kcls",data=training)
    plt.show()
    """
    test_feat = df_test[col]
    test_feat = test_feat.fillna(test_feat.mean())
    Xtest = np.array(test_feat)
    Xtest = (Xtest - np.mean(Xtest,axis=0)) / np.std(Xtest,axis=0)
    class_predict_test = km.predict(Xtest)

    df_test.iloc[:,2:] = Xtest.copy()
    df_test.loc[:,"kcls"] = class_predict_test.astype(np.int)

    sns.factorplot(x="kcls",y="id",data=df_test)
    plt.show()

def TrainFeatCorr(train,test):

    df_train = train.copy()
    df_test = test.copy()

    # unique timestamp of training data
    uniq_timestamp = df_train["timestamp"].unique()
    test_uniq_timestamp = df_test["timestamp"].unique()
    print("------ test id ------> %d " % test_uniq_timestamp)

    col = [c for c in df_train.columns if c not in excl]

    # ###
    test_feat = df_test[col]
    test_feat = test_feat.fillna(test_feat.mean(axis=0))
    test_feat = test_feat.fillna(.0)


    test_feat_shape = test_feat.shape
    n = test_feat.isnull().sum(axis=1)
    h,w = test_feat_shape

    # normalize
    Xtest = np.array(test_feat)
    Xtest = (Xtest - np.mean(Xtest, axis = 0)) / np.std(Xtest, axis = 0)
    # ###

    uniq_timestamp = df_train["timestamp"].unique()

    top_correlations = []
    for uniq_id in uniq_timestamp:

        #print("-- select timestamp: %d" % uniq_id)

        df_select = df_train.loc[df_train.timestamp == uniq_id,:].copy()
        train_feat = df_select[col].copy()
        train_feat = train_feat.fillna(train_feat.mean(axis=0))
        #
        # failed to fill with mean, then fill with zero
        #
        train_feat = train_feat.fillna(.0)

        num_ = train_feat.isnull().sum(axis=1)
        train_feat.loc[:,"znull"] = num_

        #print train_feat.loc[train_feat.znull > 0,:].head()

        train_feat = ( train_feat - train_feat.mean(axis=0) ) / train_feat.std(axis=0)
        train_feat_shape = train_feat.shape

        if h < train_feat_shape[0]:
            h_min = test_feat_shape[0]
        else:
            h_min = train_feat_shape[0]

        Xtrain = np.array(train_feat)

        corrs = []
        for col_id in range(w):


            train_col_data = Xtrain[:h_min,col_id]
            test_col_data = Xtest[:h_min,col_id]

            if len(train_col_data) != len(test_col_data):
                print(h_, len(train_col_data), len(test_col_data) )

            corr = np.corrcoef(train_col_data,test_col_data)[0][1]
            #print corr
            if not np.isnan(corr):
                corrs.append(corr)
            else:
                corrs.append(.0)


        corrs = np.array(corrs)
        top_correlation = np.argsort(corrs)[::-1][0]

        #print top_correlation,corrs[top_correlation]
        top_correlations.append(   corrs[top_correlation]   )

    top_correlations = np.array(  top_correlations )
    top_corr_timestamp = np.argsort(top_correlations)[::-1][:10]

    print top_corr_timestamp


def dataPCA(data,n_comp=5):

    pca = decomp.PCA(n_components = n_comp)
    pca.fit(data)
    transformed = pca.transform(data)
    E = pca.explained_variance_ratio_
    #print np.cumsum(E)[::-1][0]
    #print transformed.shape

    m = data.shape[0]
    X = np.hstack( ( np.ones((m, 1)), transformed ) )
    #print X.shape
    return X

# --------------------------------------------------------------------------------


env = make()

observation = env.reset()
train = observation.train.copy()



while True:



    features = observation.features.copy()

    TrainFeatCorr(train,features)

    #      checkNewIdfromFeatures(train,features)


#    y_hat = gmodel_test.predict2(observation.features.copy())

    target = observation.target

    #target['y'] = y_hat

    timestamp = observation.features["timestamp"][0]


    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        y_true = env.temp_test_y

        print("-- reward %.5f" % reward  )



    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)

    if done:
        print("Public score: {}".format(info["public_score"]))
        break
