# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNetCV
#   import kagglegym
import math

from KagglegymEmulation import make
from KagglegymEmulation import r_score

from logging import getLogger, StreamHandler, DEBUG
from logClass import MyHandler

from KaggleDataAnalysisClass import KaggleDataAnalysisClass


from sklearn.linear_model import ElasticNetCV


import matplotlib.pyplot as plt

import seaborn as sns



class fitModel():
    def __init__(self, model, train, columns):

        # first save the model ...
        self.model   = model
        self.columns = columns
        
        # Get the X, and y values, 
        y = np.array(train.y)
        
        X = train[columns]
        self.xMeans = X.mean(axis=0) # Remember to save this value
        self.xStd   = X.std(axis=0)  # Remember to save this value

        #X = np.array(X.fillna( self.xMeans ))
        
        X = np.array(X.fillna( 0.0 ))        
        
        X = (X - np.array(self.xMeans))/np.array(self.xStd)
        
        # fit the model
        self.model.fit(X, y)
        
        return
    
    def predict(self, features):
        
        X = features[self.columns]

        #X = np.array(X.fillna( self.xMeans ))
        X = np.array(X.fillna( 0.0 ))

        X = (X - np.array(self.xMeans))/np.array(self.xStd)

        return self.model.predict(X)



class TrainData(object):
    
    def __init__(self, train):

        self.train = train
        
        self.train.fillna( 0.0  )
        
        self.uniq_id = sorted(self.train["id"].unique())
        
        #print self.uniq_id
        self.y = self.train["y"]
        
        self.idvsTech30()
        
    
    def pickOneId(self,topid):
        
        #uniq_id = np.random.permutation(uniq_id)
        #topid = uniq_id[0]
        #print "random top id",topid
    
        self.topid_train = self.train.loc[ self.train["id"] == topid, :]
    
        #print "shape", self.topid_train.shape
        
        #topid_uniq_timestamp = self.topid_train["timestamp"].unique()

        #print "topid uniq timestamp length",  len(topid_uniq_timestamp)
        
    def idvsTech30(self):

        corrs = []
        
        for train_id in self.uniq_id:
            
            self.pickOneId(train_id)
            tech30 = self.topid_train["technical_30"]
            y = self.topid_train["y"]
                        
            corr = np.corrcoef(tech30,y)[0][1]
            
            if np.isnan(corr):
                corr = .0
            
            corrs.append(corr)
            
        sort_corr = np.argsort(corrs)[::-1][:10]
        
        print "top corr", np.array(corrs)[ sort_corr  ]
        print "top id", np.array(self.uniq_id)[ sort_corr  ]
        #print  sort_corr 
        
    def graph(self,topid):
        
        self.pickOneId(topid)
        
        h,w = self.topid_train.shape
        
        y = self.topid_train["y"]
        tech30 = self.topid_train["technical_30"]
        
        plt.scatter(range(h), y)
        plt.show()
        
        sns.jointplot(tech30, y)
        plt.show()
        
        
        

def proc2(log):
    
    
    
    
    env = make()
    
    observation_test = env.reset()
    
    #emcv = ElasticNetCV()
    
    #columns = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']

    columns = ['technical_30', 'technical_20', 'fundamental_11']


    trainCls = TrainData( observation_test.train.copy() ) 
    
    
    trainCls.graph(1276)
    trainCls.graph(1083)    



def proc1(log):
    
    
    
    
    env = make()
    
    observation_test = env.reset()
    
    emcv = ElasticNetCV()
    
    #columns = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']

    columns = ['technical_30', 'technical_20', 'fundamental_11']

    train_data = observation_test.train.copy()
    

    model_test = fitModel(emcv, train_data, columns)
    
    """
    train_data = observation_test.train.copy()

    features_data = observation_test.features.copy()
        
    feat_colNames = features_data.columns.values.tolist()[2:]
        
    #train_data = observation_test.features.copy    
    
    kaggleAnalysis = KaggleDataAnalysisClass(train_data,True)
    
    kaggleAnalysis.corrCheck(feat_colNames)    
    
    #emcv = ElasticNetCV(fit_intercept = True)
    
    
    kaggleAnalysis.modelfit(emcv)
    """

      
    
    while True:


        prediction_test  = model_test.predict(observation_test.features.copy())    

        target_test      = observation_test.target

        target_test['y'] = prediction_test


        """
        features_data = observation_test.features.copy()

        prediction_test = kaggleAnalysis.predict(features_data)  

        target_test      = observation_test.target

        target_test['y'] = prediction_test


        timestamp_ = observation_test.features["timestamp"][0]
    
        log.info("timestamp : %d " % timestamp_)

        """
        timestamp_ = observation_test.features["timestamp"][0]



        rewards = []
        if timestamp_ % 100 == 0:
            print(timestamp_)
            
            y_true = env.temp_test_y 
            
            score_ = r_score(y_true,prediction_test)
            rewards.append(score_)
            
            log.info("score %.5f" % np.mean(rewards)  )
            
    
        observation_test, reward_test, done_test, info_test = env.step(target_test)
    
        #log.info("reward_test : %.5f " % reward_test)
    
        if done_test:
            print('Info-test:',info_test['public_score'])
    
            break

    

def main():

    log = getLogger("root")
    log.setLevel(DEBUG)
    log.addHandler(MyHandler())


    
    proc2(log)
    #proc1(log)    
    


if __name__ == "__main__":
    main()