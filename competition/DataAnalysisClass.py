# -*- coding: utf-8 -*-



from DataClass import DataClass

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from MyLoggerDeconstClass import MyLoggerDeconstClass

#from logging import getLogger, StreamHandler, DEBUG

class DataAnalysisClass(MyLoggerDeconstClass):
    
    def __init__(self):
        
        super(DataAnalysisClass,self).__init__()
        
        
        self.data = DataClass()
        
        self.train_data = self.data.getTrain()
        
        self.y_train = self.data.getTarget()
        
        self.timestamp = self.data.getTimeStamp()
        
    
    def timeStampAnalysis(self):
        
        
        self.log.info("length of timestamp:%d" % len(self.timestamp))
        
        timestamp_np = np.array(self.timestamp)
        
        mask = np.isnan(timestamp_np)
        
        nan_rate = np.sum(mask == True) / np.float(  len(mask)  )
        
        self.log.info("nan of timestamp: %.4f" % nan_rate)
    

    def featuresAnalyss(self):
        
        
        

    def normalizedData(self):
          
        pass
        
    
    def __str__(self):
        "DataAnalysisClass"
        
    def DataNormalize(self):
        pass
    
    
    def getColumns(self):
        return self.data.getColumns()
    
    def hist(self):
        
        self.log.info("y train size %d" % len(self.y_train))        
        
        plt.hist(self.y_train)
        plt.show()
        
        sns.distplot(self.y_train, kde=False, rug=False, bins=1000) 
        plt.show()

        self.log.info( "-- y train mean %.4f "% np.mean(self.y_train) )
        self.log.info( "-- y train std %.4f "% np.std(self.y_train) ) 
        
        #sns.distplot(self.timestamp, kde=False, rug=False, bins=10000) 
        #plt.show()
        
    
    def correlationData(self):
        pass