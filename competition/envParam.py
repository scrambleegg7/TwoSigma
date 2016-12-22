# -*- coding: utf-8 -*-

import numpy as np
import platform as pl



class envParam(object):
    
    def __init__(self,test=False):
        
        self.test = test
        self.setParam()
    
    def setParam(self):
        
        self.datadir = "."
            

class envParamLinux(envParam):
    
    def __init__(self,test=False):
        super(envParamLinux,self).__init__(test=False)
        
    def setParam(self):
        
        self.datadir = "/home/hideaki/SynologyNFS/myProgram/pythonProj/KaggleData/TwoSigma"
        
    
class envParamOsxBk(envParam):
    
    def __init__(self,test=False):
        super(envParamOsxBk,self).__init__(test=False)
        
    def setParam(self):
        
        self.datadir = "/Users/donchan/Documents/myData/KaggleData/TwoSigma"
        