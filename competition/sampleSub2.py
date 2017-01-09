

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

# ---------------------------------------------------------------------------

def randInitializeWeights(L_in, L_out):
    """
    """

    epsilon_init = 1.
    #W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    W = np.random.rand(L_out, 1 + L_in)

    return W

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def safe_log(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))

def nnCostFunction(nn_params, *args):

    in_size, hid_size, num_labels, X, y, lam = args

    Theta1 = nn_params[0:(in_size + 1) * hid_size].reshape((hid_size, in_size + 1))
    Theta2 = nn_params[(in_size + 1) * hid_size:].reshape((num_labels, hid_size + 1))


    #print(Theta1.shape)
    #print(Theta2.shape)

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    m = X.shape[0]

    X = np.hstack((np.ones((m, 1)), X))

    lb = LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    J = 0
    for i in range(m):
        xi = X[i, :]
        yi = y[i]
        # forward propagation
        a1 = xi
        z2 = np.dot(Theta1, a1)
        a2 = sigmoid(z2)
        a2 = np.hstack((1, a2))
        z3 = np.dot(Theta2, a2)
        a3 = sigmoid(z3)

        #print("-- a3 shape %s" % (a3.shape,))
        #print("-- yi shape %s" % (yi.shape,))

        J += sum(-yi * safe_log(a3) - (1 - yi) * safe_log(1 - a3))
        # backpropagation
        delta3 = a3 - yi
        delta2 = np.dot(Theta2.T, delta3) * sigmoidGradient(np.hstack((1, z2)))
        delta2 = delta2[1:]  #


        delta2 = delta2.reshape((-1, 1))
        delta3 = delta3.reshape((-1, 1))
        a1 = a1.reshape((-1, 1))
        a2 = a2.reshape((-1, 1))

        Theta1_grad += np.dot(delta2, a1.T)
        Theta2_grad += np.dot(delta3, a2.T)
    J /= m

    temp = 0.0;
    for j in range(hid_size):
        for k in range(1, in_size + 1):  #
            temp += Theta1[j, k] ** 2
    for j in range(num_labels):
        for k in range(1, hid_size + 1): #
            temp += Theta2[j, k] ** 2
    J += lam / (2.0 * m) * temp;

    #
    Theta1_grad /= m
    Theta1_grad[:, 1:] += (lam / m) * Theta1_grad[:, 1:]
    Theta2_grad /= m
    Theta2_grad[:, 1:] += (lam / m) * Theta2_grad[:, 1:]

    #
    grad = np.hstack((np.ravel(Theta1_grad), np.ravel(Theta2_grad)))

    print "J =", J
    return J, grad

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    # forward propagation
    X = np.hstack((np.ones((m, 1)), X))
    h1 = sigmoid(np.dot(X, Theta1.T))

    h1 = np.hstack((np.ones((m, 1)), h1))
    h2 = sigmoid(np.dot(h1, Theta2.T))

    return np.argmax(h2, axis=1)


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


#env = kagglegym.make()

env = make()

o = env.reset()
excl = ["id", "sample", "y", "timestamp"]
col = [c for c in o.train.columns if c not in excl]

#train = pd.read_hdf('../input/train.h5')
df_train = o.train
train = df_train[col]
d_mean= train.median(axis=0)
d_mu = train.mean(axis=0)

train = o.train[col]
n = train.isnull().sum(axis=1)

y_train = df_train['y'].values.tolist()
y_train = np.array(y_train)

mu = np.mean(y_train)
std = np.std(y_train)
y_cdfs = np.array( [  norm.cdf(x=y_, loc=mu, scale=std) for y_ in y_train ] )

y_prob =  np.around(y_cdfs,decimals=1) * 10.

# fill with mean
train = train.fillna(d_mu)

X = np.array(train)
print("-- X shape %s" % (X.shape,))

in_size = X.shape[1]
hid_size = 30
num_labels = 11

X = X.astype(np.float64)

#for c in range(X.shape[1]):
#    X[:,c] *= 255.0 / np.max(X[:,c])

X /= X.max()

initial_Theta1 = randInitializeWeights(in_size, hid_size)
initial_Theta2 = randInitializeWeights(hid_size, num_labels)

initial_nn_params = np.hstack((np.ravel(initial_Theta1), np.ravel(initial_Theta2)))

lam = .05
X = X[:10000,:]
J, grad = nnCostFunction(initial_nn_params, in_size, hid_size, num_labels, X, y_prob, lam)


res = optimize.minimize(fun=nnCostFunction, x0=initial_nn_params, method="CG", jac=True,
                                  options={'maxiter':70, 'disp':True},
                                  args=(in_size, hid_size, num_labels, X, y_prob, lam))
nn_params = res.x

Theta1 = nn_params[0:(in_size + 1) * hid_size].reshape((hid_size, in_size + 1))
Theta2 = nn_params[(in_size + 1) * hid_size:].reshape((num_labels, hid_size + 1))

Xt = X[:10,:]
#rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
y_pred = predict(Theta1, Theta2, Xt)




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

<<<<<<< 4c6e4cd8bda6225134432bbbebf58570f4204b12
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
=======
#y_pred = model1.predict(Xt)
print(y_prob[:10])
print(y_pred)

while True:
    df_test = o.features
    test = o.features[col]
    test = test.fillna(d_mean)
>>>>>>> sample sub2 change : neuro network program
