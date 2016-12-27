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

from TrainDataClass import TrainData
from FitModelClass import fitModel
from FitModelClass import glmModel



def proc2(log):
    
    
    
    
    env = make()
    
    observation_test = env.reset()
    
    #emcv = ElasticNetCV()
    
    #columns = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']

    columns = ['technical_30', 'technical_20', 'fundamental_11']


    trainCls = TrainData( observation_test.train.copy() ) 
    
    
    trainCls.graph(2047)
    #trainCls.graph(1083)    



def proc1(log):
    
    
    
    
    env = make()
    
    observation_test = env.reset()
    
    emcv = ElasticNetCV()
    
    #columns = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']

    columns = ['technical_30', 'technical_20', 'fundamental_11']

    train_data = observation_test.train.copy()
    

    #model_test = fitModel(emcv, train_data, columns)
    model_test = glmModel(train_data, columns)    
    y_hat = model_test.BuildModel()
    
    print len(y_hat)
    print y_hat[:10]
    
    y_true = model_test.df["y"]

    print len(y_true)    
    print y_true[:10]        
    #score_ = r_score(y_true, y_hat)
    
    #print score_

    
    return 1
    
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


    
    proc1(log)
    #proc2(log)    
    


if __name__ == "__main__":
    main()