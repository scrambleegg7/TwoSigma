# -*- coding: utf-8 -*-




from DataClass import DataClass

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from pandas.tools.plotting import scatter_matrix

from MyLoggerDeconstClass import MyLoggerDeconstClass

import time

from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso

from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

#from sklearn import linear_model as lm
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import scale


class KaggleDataAnalysisClass(MyLoggerDeconstClass):
    
    def __init__(self,train,normalized=False):
        
        super(KaggleDataAnalysisClass,self).__init__()
        
        self.train = train
        
        mean_train = self.train.mean(axis=0)
        
        self.log.info("mean shape %s" % (mean_train.shape,))
        
        self.train = self.train.fillna(mean_train)
        
        self.X = np.array(self.train)
        self.y = np.array(self.train.y)
        
        startidx = 2 # start
        endidx = 110
        
        self.top20 = None
        
        self.features108 = self.X[:,startidx:endidx].copy()
        mean_features108 = np.array( mean_train[startidx:endidx] ) 
        std_features108 = np.std(self.features108,axis=0)        
    
        if normalized:
            self.features108 = (self.features108 - mean_features108) / std_features108

            self.log.info("normalized features108 shape %s" % (self.features108.shape,))        
        else:
            self.log.info("features108 shape %s" % (self.features108.shape,))        
            
        #self.X_scale = self.X.copy()
        #self.X_scale[:,startidx:endidx] = scale(self.X[:, startidx:endidx  ], axis=0)
        
    def corrCheck(self,colNames):

        self.log.info("colNames length %d" % len(colNames))
        
        corrY_list = []        
        
        h,w = self.features108.shape
        
        y_train = self.X[:,110]
        
        for i in range(w):
            
            corr  =  np.corrcoef( y_train, self.features108[:,i] )[0][1]
            #print corr
            corrY_list.append(corr)

        corrY_absolute = np.array( [   np.abs(c) for c in corrY_list   ] )
        mycorr_top20_abs = np.argsort( corrY_absolute )[::-1][:20]
        
        
        self.log.info("--top4 out of 20 absolute corr vs y (target)--\n")
        for idx in mycorr_top20_abs[:4]:
            self.log.info("idx:%d - %s:%.4f" % (idx, colNames[idx], corrY_list[idx]))
            
        self.top20 = mycorr_top20_abs
        
    def modelfit(self,model):
        
        mytop4 = self.top20[:3]
        self.log.info("top4 index %s" % (mytop4))        
    
        self.features_top4 = self.features108[:,mytop4]
        self.log.info("train top4 shape %s" % (self.features_top4.shape,))        
    
        X = self.features_top4.copy()
        y = self.y
        
        #emcv = ElasticNetCV(fit_intercept = True)

        self.log.info("Start to train top4")

        self.model = model        
 
        self.model.fit(X,y)
        
        self.log.info("End to train")    
    
    def predict(self,features):

        mytop4 = self.top20[:3]      
          
        features_top4 = features.iloc[:,mytop4]
        
        mean_feat = features_top4.mean(axis=0)
        
        features_top4 = features_top4.fillna(mean_feat)

        self.log.info("test top4 shape %s" % (features_top4.shape,))    
        
        X = np.array(features_top4)
        
        X = (X - np.array(mean_feat)) / np.std(X, axis=0)
        
        
        
        
        
        return self.model.predict( X )
        
        

        
        



        


        
        
        