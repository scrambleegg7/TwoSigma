# -*- coding: utf-8 -*-



from DataClass import DataClass

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np


class DataAnalysisClass(object):
    
    def __init__(self):
        
        self.data = DataClass()
        
        self.train_data = self.data.getTrain()
        
        self.y_train = self.data.getTarget()
    
    
    def __str__(self):
        "DataAnalysisClass"
        
    def DataNormalize(self):
        pass
    
    
    
    def hist(self):
        
        print "-- y traiing size", len(self.y_train)
        
        plt.hist(self.y_train)
        plt.show()
        
        sns.distplot(self.y_train, kde=False, rug=False, bins=1000) 
        plt.show()

        print "-- y train mean ", np.mean(self.y_train)
        print "-- y train std ", np.std(self.y_train)        