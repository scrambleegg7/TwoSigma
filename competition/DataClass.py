# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import platform as pl
import os

import pandas as pd

from envParam import envParamLinux
from envParam import envParamOsxBk
from envParam import envParam

class DataClass(object):
    
    def __init__(self):
        
        if pl.system() == "Linux":
            self.env = envParamLinux()
        elif pl.system() == "Darwin":
            self.env = envParamOsxBk()
        else:
            self.env = envParam()
            
        self.loadData()
        self.getTrain()
        
    def loadData(self,filename="train.h5"):
        
        datadir = self.env.datadir
        
        fileinfo = os.path.join(datadir,filename)
        
        self.df = pd.HDFStore(fileinfo,"r")
        
    def getTrain(self):
        
        self.train_data = self.df.get("train")
        
        
        return self.train_data
        
    def getTarget(self):
        
        return self.train_data.ix[:,"y"].tolist()
        
    def getTimeStamp(self):

        return self.train_data.ix[:,"timestamp"].tolist()
        
    
    def getColumns(self):
        return self.train_data.columns.tolist()
        
    def getDf(self):
        return self.df
        
        
        
        
        
        
    
    