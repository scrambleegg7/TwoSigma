# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import platform as pl
import os

import pandas as pd

from envParam import envParamLinux
from envParam import envParamOsxBk
from envParam import envParam


from MyLoggerDeconstClass import MyLoggerDeconstClass

class DataClass(MyLoggerDeconstClass):
    
    def __init__(self):
        
        super(DataClass,self).__init__()

        
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

        self.log.info("Loading train.h5 .......")
        
        self.df = pd.HDFStore(fileinfo,"r")
        
        self.log.info("end to load data .....")
        
        
        
    def loadCsvData(self,filename="my_df.csv"):

        datadir = self.env.datadir
        
        fileinfo = os.path.join(datadir,filename)

        my_df = pd.read_csv(fileinfo,encoding='cp932')
        
        return my_df
        
        
    def saveCsvData(self,my_df,filename="my_df.csv"):
        
        datadir = self.env.datadir
        
        fileinfo = os.path.join(datadir,filename)

        my_df.to_csv(fileinfo,encoding='cp932',index=False)

    def getTrain(self):
        
        self.train_data = self.df.get("train")
        
        return self.train_data
        
    def getId(self):
        
        return self.train_data.ix[:,"id"].values.tolist()
        
    def getTarget(self):
        
        return self.train_data.ix[:,"y"].values.tolist()
        
    def getTimeStamp(self):

        return self.train_data.ix[:,"timestamp"].values.tolist()
        
    
    def getColumns(self):
        return self.train_data.columns.values.tolist()
        
    def getDf(self):
        return self.df
        
        
        
        
        
        
    
    