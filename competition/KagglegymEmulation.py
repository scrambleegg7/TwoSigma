import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNetCV

import os
import platform as pl


from envParam import envParamLinux
from envParam import envParamOsxBk
from envParam import envParam

from MyLoggerDeconstClass import MyLoggerDeconstClass



import math

def r_score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    r = np.sign(r2) * np.sqrt(np.abs(r2))
    return max(-1, r)


class Observation(object):
    def __init__(self, train, target, features):
        self.train = train
        self.target = target
        self.features = features


class Environment(MyLoggerDeconstClass):
    def __init__(self):
        
        super(Environment,self).__init__()

        
        if pl.system() == "Linux":
            self.env = envParamLinux()
        elif pl.system() == "Darwin":
            self.env = envParamOsxBk()
        else:
            self.env = envParam()

        datadir = self.env.datadir
        
        filename = "train.h5" 
        fileinfo = os.path.join(datadir,filename)
        
        self.log.info("Loading train.h5 .......")
        
        with pd.HDFStore(fileinfo, "r") as hfdata:
                
        #with pd.HDFStore("../input/train.h5", "r") as hfdata:
            self.timestamp = 0
            fullset = hfdata.get("train")
            self.unique_timestamp = fullset["timestamp"].unique()


            
            
            # Get a list of unique timestamps
            # use the first half for training and
            # the second half for the test set
            n = len(self.unique_timestamp)
            i = int(n/2)
            timesplit = self.unique_timestamp[i]

            self.log.info("length of unique timestamp %d" % n )
            self.log.info("training size = unique_idx %d" % i )


            self.n = n
            self.unique_idx = i
            self.train = fullset[fullset.timestamp < timesplit]
            self.test = fullset[fullset.timestamp >= timesplit]
            
            self.log.info("train size %s" % (self.train.shape,) )
            self.log.info("test size %s" % (self.test.shape,) )            

            # Needed to compute final score
            self.full = self.test.loc[:, ['timestamp', 'y']]
            self.full['y_hat'] = 0.0
            self.temp_test_y = None

        self.log.info("Loading done .......")
        

    def reset(self):
        timesplit = self.unique_timestamp[self.unique_idx]

        self.log.info("timesplit : %d for subset" % timesplit )

        self.unique_idx = int(self.n / 2)
        self.unique_idx += 1
        subset = self.test[self.test.timestamp == timesplit]

        self.log.info("unique index %d" % self.unique_idx )
        


        # reset index to conform to how kagglegym works
        target = subset.loc[:, ['id', 'y']].reset_index(drop=True)
        self.log.info("subset shape %s" % (target.shape,) )
        
        
        self.temp_test_y = target['y']
        self.log.info("temp test size %s" % (self.temp_test_y.shape,) )
        
        target.loc[:, 'y'] = 0.0  # set the prediction column to zero
        self.log.info("set target y to zero")

        # changed bounds to 0:110 from 1:111 to mimic the behavior
        # of api for feature
        features = subset.iloc[:, :110].reset_index(drop=True)
        
        #self.colNames = features.columns.values.tolist()[2:]
        self.log.info("features size %s" % (features.shape,) )
        #self.log.info("features col names %s" % (self.colNames) )
        
        observation = Observation(self.train, target, features)
        return observation

    def step(self, target):
        
        
        timesplit = self.unique_timestamp[self.unique_idx-1]
        
        #self.log.info("timesplit-1 %d" % timesplit )
        
        # Since full and target have a different index we need
        # to do a _values trick here to get the assignment working
        y_hat = target.loc[:, ['y']]
        
        self.full.loc[self.full.timestamp == timesplit, ['y_hat']] = y_hat._values

        #self.log.info("update full y_hat with input target y ")

        #self.log.info("unique index --> %d" % self.unique_idx )


        if self.unique_idx == self.n:


            self.log.info("unique index == %d" % self.n )
            
            done = True
            observation = None
            reward = r_score(self.temp_test_y, target.loc[:, 'y'])
            
            score = r_score(self.full['y'], self.full['y_hat'])

            info = {'public_score': score}
        
        else:

            reward = r_score(self.temp_test_y, target.loc[:, 'y'])
            #self.log.info("r2 score for temp_test_y with target y ")
            #self.log.info("r2 score %.5f " % reward)


            done = False

            info = {}
            timesplit = self.unique_timestamp[self.unique_idx]

            self.unique_idx += 1
            
            #self.log.info("next unique index %d" % self.unique_idx )

            subset = self.test[self.test.timestamp == timesplit]

            # reset index to conform to how kagglegym works

            target = subset.loc[:, ['id', 'y']].reset_index(drop=True)

            self.temp_test_y = target['y']

            # set the prediction column to zero
            target.loc[:, 'y'] = 0

            # column bound change on the subset
            # reset index to conform to how kagglegym works
            features = subset.iloc[:, 0:110].reset_index(drop=True)

            observation = Observation(self.train, target, features)

        return observation, reward, done, info


    def __str__(self):
        return "Environment()"


def make():
    return Environment()