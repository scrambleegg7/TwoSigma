#  import kagglegym
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV


import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf




from KagglegymEmulation import make
from KagglegymEmulation import r_score


class glmModel():
    
    def __init__(self, train, columns):

        # first save the model ...
        self.model   = None
        self.columns = columns
        self.train = train
        

        self.y = np.array(train.y)
        self.Xt = train[columns]

        
        self.X = None
        self.normalize()
        
        # fit the model
        #self.model.fit(self.X, self.y)
        
        self.df = pd.DataFrame(self.X,columns=columns)
        
        self.df["y"] = self.y
        
        #self.df["y"] = np.exp( self.y )
        
        self.df["y_hat"] = 0.0
        
        print( self.df.head() )     
        
        
    def normalize(self):


        self.xMeans = self.Xt.mean(axis=0) 
        self.xStd   = self.Xt.std(axis=0)  

        #X = np.array(X.fillna( self.xMeans ))
        
        Xt = np.array(self.Xt.fillna( 0.0 ))        
        
        self.X = (Xt - np.array(self.xMeans))/np.array(self.xStd)
        

        

    def BuildModel2(self):

        
        ols_model = 'y ~ technical_30 + technical_20 + fundamental_11 + technical_19' 
        
        ols_model_poly = 'y ~ technical_30 + technical_20 + I(technical_30 ** 2.0) + I(technical_20 ** 2.0)' 
        
        
        Xc = sm.add_constant(self.X)

        #mod = sm.OLS(self.train["y"],Xc)        
        
        mod = smf.ols(formula=ols_model_poly , data= self.df)
        
        self.res = mod.fit()
        

    def predict2(self,features):
        
        self.Xt = features[self.columns]
        self.normalize()
        
        i, w0, w1, w0_2, w1_2 = self.res.params        
        wmtx = [w0, w1, w0_2, w1_2]
        
        #print wmtx

        Xpoly = np.column_stack( ( self.X, self.X[:,0] ** 2.0  )  )        
        Xpoly = np.column_stack( ( Xpoly, self.X[:,1] ** 2.0  )  )        
        #print Xpoly.shape
        
        y_hat = np.sum(Xpoly * wmtx,axis=1) + i 

        #print("-- length of y hat %d" %  len(y_hat))

        #y_hat = 

        return ( y_hat )
                    

    def BuildModel(self):

        
        ols_model = 'y ~ technical_30 + technical_20 + fundamental_11 + technical_19' 
        
        Xc = sm.add_constant(self.X)

        mod = sm.OLS(self.train["y"],Xc)        
        
        #mod = smf.ols(formula=ols_model, data= self.df)
        
        self.res = mod.fit()
    
    def predict(self,features):
        
        self.Xt = features[self.columns]
        self.normalize()
        
        i, w0, w1, w2, w3 = self.res.params        
        wmtx = [w0, w1, w2, w3]
        
        y_hat = np.sum(self.X * wmtx,axis=1) + i 

        #print("-- length of y hat %d" %  len(y_hat))

        #y_hat = 

        return ( y_hat )
        

class mModel():
    
    def __init__(self, model, train, columns):

        # first save the model ...
        self.model   = model
        self.columns = columns
        
        # Get the X, and y values, 
        self.y = np.array(train.y)
        self.Xt = train[columns]
        
        self.normalize()
        
        
        # fit the model
        self.model.fit(self.X, self.y)
        
        
    
    def normalize(self):


        self.xMeans = self.Xt.mean(axis=0) 
        self.xStd   = self.Xt.std(axis=0)  

        #X = np.array(X.fillna( self.xMeans ))
        
        Xt = np.array(self.Xt.fillna( 0.0 ))        
        
        self.X = (Xt - np.array(self.xMeans))/np.array(self.xStd)
        
        
    def predict2(self, features):
        
        X = features[self.columns]

        #X = np.array(X.fillna( self.xMeans ))
        X = np.array(X.fillna( 0.0 ))

        X = (X - np.array(self.xMeans))/np.array(self.xStd)

        return self.model.predict(X)






# The "environment" is our interface for code competitions
# 
# env = kagglegym.make()

env = make()



# We get our initial observation by calling "reset"
observation = env.reset()

#columns = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']
columns = ['technical_30', 'technical_20']

train_data = observation.train.copy()

#gmodel_test = glmModel(train_data, columns)        
#gmodel_test.BuildModel2()


elasticmodel = ElasticNetCV()
gmodel_test = mModel(elasticmodel,train_data,columns)



print("Train has {} rows".format(len(observation.train)))
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))



#y_hat = gmodel_test.predict(observation.features.copy())    


#return 1

rewards = []

while True:
    
    features = observation.features.copy()
    
        
    
    
    y_hat = gmodel_test.predict2(observation.features.copy())    
    
    target = observation.target
    
    
    #target_test      = observation_test.target

    target['y'] = y_hat
    
    
    timestamp = observation.features["timestamp"][0]
    
    
    
    
    if timestamp % 100 == 0:
        
        print("Timestamp #{}".format(timestamp))
        
        y_true = env.temp_test_y 
        

        #y_true = np.exp(y_true)        
        
        score_ = r_score(y_true,y_hat)
        rewards.append(score_)
            
        print("-- score %.5f" % np.mean(rewards)  )
            
        
        
        

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break