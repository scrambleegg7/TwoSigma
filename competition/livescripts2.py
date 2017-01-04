#  import kagglegym
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV


import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf

import sklearn.decomposition as decomp

"""
    unnecessary lines for live pane
"""
import seaborn as sns
import matplotlib.pyplot as plt

from KagglegymEmulation import make
from KagglegymEmulation import r_score
from collections import Counter
from logging import getLogger, StreamHandler, DEBUG
from logClass import MyHandler
#############################################################

def computeCost(X, y, theta):
    m = len(y)
    tmp = np.dot(X, theta) - y
    J = 1.0 / (2 * m) * np.dot(tmp.T, tmp)
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []
    for iter in range(iterations):
        theta = theta - alpha * (1.0 / m) * np.dot(X.T, np.dot(X, theta) - y)
        J_history.append(computeCost(X, y, theta))
    return theta, J_history


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

    print("   new test unique id %s" % new_test_uniq_id)

    print("   missing training id count from features  %s" % len(missing_training_id) )


def testAalysis4(train,test,y_test_true):

    df_test = test.copy()
    df_train = train.copy()

    print("test shape %s" % (df_test.shape,))
    print("y_test_true shape %s" % (y_test_true.shape,))

    uniq_timestamp = df_test["timestamp"].unique()
    uniq_id = sorted(df_test["id"].unique())
    #print uniq_id

    print("-- selected features timestamp: %d" % uniq_timestamp)

    train_uniq_id = sorted(df_train["id"].unique())
    train_uniq_timestamp = sorted(   df_train["timestamp"].unique()  )

    new_test_uniq_id = np.array( [ test_id for test_id in uniq_id if test_id not in train_uniq_id] )

    print("test id length %d" % (len(uniq_id) ) )
    print("train id length %d" %  len(train_uniq_id)  )

    print("length of new test unique id %d" % new_test_uniq_id.shape)

    Xtrain = np.array( df_train )
    Xtest = np.array( df_test )

    print("* training data shape %s" %  (Xtrain.shape,) )
    print("* test data shape %s" % (Xtest.shape,) )
    train_id = Xtrain[:,0]

    scores = {}
    y_test_pred_dict = {}
    for cnt,idx in enumerate(uniq_id):
        mask =  Xtrain[:,0] == idx
        select_one_id = Xtrain[mask,:].copy()

        mask = select_one_id[:,1] > (uniq_timestamp - 30)

        select_one_id = select_one_id[mask,:]
        #print select_one_id.shape

        data = select_one_id[:,2:110]
        y = select_one_id[:,110].ravel()
        y = np.cumsum(y)

        if cnt % 100 == 0:
            print("-- trainig counter:%d test_id:%d" % (cnt,idx) )

        #
        #  data model used from PCA analysis 5 dimentions
        #
        y_pred, theta, final_cost = fitting(data,y)
        #
        if not np.isnan(final_cost):
            #print idx,r_score(y,y_pred)
            scores[idx] = r_score(y,y_pred)

            mask = Xtest[:,0] == idx
            select_one_test_id = Xtest[mask,:].copy()
            test_data = select_one_test_id[:,2:]

            data_stacked = np.vstack(  (data[1:,:],test_data )  )
            Xtt = dataPCA(data_stacked)
            y_test_diff = np.dot(Xtt,theta)
            #y_test_diff = np.diff(y_test)

            y_test_prediction = y_test_diff[-1]
            y_test_pred_dict[idx] = y_test_prediction
            #print(y_test_true[cnt],y_test_prediction)

        else:
            scores[idx] = 0.0
            y_test_pred_dict[idx] = 0.0

    print("-- total counter %d" % (cnt + 1)  )
    r_score_value = scores.values()
    print np.mean(r_score_value)



    return y_test_pred_dict



def dataPCA(data,n_comp=5):

    #
    # set ZERO for NaN
    #
    data[ np.isnan(data)   ] = .0
    #
    # normalize
    #
    mu = np.mean(data)
    sigma = np.std(data)
    data = (data - mu) / sigma

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

def fitting(data,y):


    X = dataPCA(data)


    theta = np.zeros(n_comp + 1)
    iterations = 10000
    alpha = 0.1

    # calculate cost
    initialCost = computeCost(X, y, theta)
    #print "initial cost:", initialCost

    # gradients descent
    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
    #print "theta:", theta
    #print "final cost:", J_history[-1]

    y_pred = np.dot(X, theta)

    return y_pred, theta, J_history[-1]



def yplot(train):

    df_train = train.copy()

    df_train_0time = df_train[ df_train.id == 15].copy()
    Xt = np.array(df_train_0time)

    #ax = fig.add_subplot(5,5,idx+1)
    plt.scatter(np.arange( Xt.shape[0] ),  np.cumsum(Xt[:,110]) , c='r'   )
    plt.show()



def trainingAalysis(train):

    df_train = train.copy()
    #print df_train.columns.values

    print "training size", len(train["y"])

    uniq_timestamp = train["timestamp"].unique()
    uniq_id = sorted(train["id"].unique())

    print("train unique timestamp %d, train unique id %d" % len(uniq_timestamp),len(uniq_id) )

    i = len(uniq_timestamp) / 2

    timesplit = uniq_timestamp[i]

    #
    # copy is necessary for data frame manipulation .....
    #
    df_train_1time = df_train[ df_train.timestamp == 1].copy()

    df_train_1time['y_cumsum'] = np.cumsum( df_train_1time['y'].values )
    df_train_1time['serialno'] = np.arange( df_train_1time.shape[0]  )

    mycolumns = ['id','serialno','y_cumsum','y']
    df_train_compact = df_train_1time.loc[:, mycolumns ].copy()
    print df_train_compact.describe()
    X = np.array(df_train_compact)

    iddict = Counter(X[:,0])

    fig = plt.figure(figsize=(8,6))
    for idx,i in enumerate(uniq_id[:25]):
        df_train_0time = df_train[ df_train.id == i].copy()
        Xt = np.array(df_train_0time)

        ax = fig.add_subplot(5,5,idx+1)
        ax.scatter(np.arange( Xt.shape[0] ),  np.cumsum(Xt[:,110]) , c='r'   )

        print("id:%d  time length:%d" % (i, Xt.shape[0]))
        #ax.tick_params(labelbottom="off")
        #ax.tick_params(labelleft="off")


        corrs = []
        for j in range(2,110):

            param = Xt[:,j]
            param[ np.isnan(param)   ] = .0

            corr = np.corrcoef(Xt[:,110],Xt[:,j])[0][1]

            if not np.isnan(corr):
                corrs.append(corr)

        top5 = np.argsort(corrs)[::-1][:5]
        print top5

    plt.show()
    #sns.factorplot(x="serialno", y="y_cumsum", hue="id", data=df_train_compact)
    #subset = df_test[df_test.timestamp == timesplit]

    #plt.plot(X[:,1],X[:,2],c='r')
    #plt.show()


"""
remove these lines when copying into live panges of kaggle
"""
log = getLogger("root")
log.setLevel(DEBUG)
log.addHandler(MyHandler())

##############################################


# The "environment" is our interface for code competitions
#
# env = kagglegym.make()

#columns = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']

env = make()


# We get our initial observation by calling "reset"
observation = env.reset()

#
#trainingAalysis(train_data)
#

train_data = observation.train.copy()

y_test_true = env.temp_test_y
test_data = observation.features.copy()

#yplot(train_data)

mycounter = 0
while True:

    test_data = observation.features.copy()
    target = observation.target

    #y_pred_dict = testAalysis4(train_data,test_data,y_test_true)
    #for k,v in y_pred_dict.items():
    #    target.loc[target.id == k ,"y"] = v


    checkNewIdfromFeatures(train_data,test_data)

    mycounter += 1
    if mycounter > 5:
        break

    observation, reward, done, info = env.step(target)

    #print("reward: %.5f" %reward)

    if done:
        print("Public score: {}".format(info["public_score"]))
        break



"""
    temporary exit
"""
exit()
"""
    temporary exit
"""

#gmodel_test = glmModel(train_data, columns)
#gmodel_test.BuildModel2()


#elasticmodel = ElasticNetCV()
#gmodel_test = mModel(elasticmodel,train_data,columns)



print("Train has {} rows".format(len(observation.train)))
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))


rewards = []
reward = .0
while True:



    features = observation.features.copy()




#    y_hat = gmodel_test.predict2(observation.features.copy())

    target = observation.target

    target['y'] = y_hat

    timestamp = observation.features["timestamp"][0]


    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        y_true = env.temp_test_y

        #y_true = np.exp(y_true)

        score_ = r_score(y_true,y_hat)
        rewards.append(score_)

        print("-- score %.5f" % np.mean(rewards)  )
        print("-- reward %.5f" % reward  )



    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)

    if done:
        print("Public score: {}".format(info["public_score"]))
        break
