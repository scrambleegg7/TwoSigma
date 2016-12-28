# -*- coding: utf-8 -*-

import numpy as np


#

import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf

#


class glmModel():
    
    def __init__(self, train, columns):

        # first save the model ...
        self.model   = None
        self.columns = columns
        self.train = train
        

        
        self.X = None
        self.y = None        
        
        self.dataNormalize()
        
        # fit the model
        #self.model.fit(self.X, self.y)
        
        self.df["y"] = self.y
        self.df["y_hat"] = 0.0
        
        print self.df.head()      
        
        #self.BuildModel()

    def BuildModel(self):

        
        ols_model = 'y ~ technical_30 + technical_20 + fundamental_11' 
        
        mod = smf.ols(formula=ols_model, data= self.df)
        
        res = mod.fit()

        print res.params
        print res.summary()    
        
        return res
        
    def dataNormalize(self):
        
        # Get the X, and y values, 
        #        
        


        self.train_drop = self.train.dropna()        
        print self.train_drop.describe()

        
        X = self.train[self.columns]
        
        self.df = X.copy()

        
        self.xMeans = X.mean(axis=0) # Remember to save this value
        self.xStd   = X.std(axis=0)  # Remember to save this value

        #X = np.array(X.fillna( self.xMeans ))
        
        X = np.array(X.fillna( 0.0 ))        
        
        self.X = (X - np.array(self.xMeans))/np.array(self.xStd)        
        self.y = np.array(self.train.y)
        
    
    def predict(self, features):
        
        X = features[self.columns]

        #X = np.array(X.fillna( self.xMeans ))
        X = np.array(X.fillna( 0.0 ))

        X = (X - np.array(self.xMeans))/np.array(self.xStd)

        return self.model.predict(X)






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
