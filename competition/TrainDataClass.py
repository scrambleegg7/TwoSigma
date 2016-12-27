# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class TrainData(object):
    
    def __init__(self, train):

        self.train = train
        
        self.train.fillna( 0.0  )
        
        self.uniq_id = sorted(self.train["id"].unique())
        
        #print self.uniq_id
        self.y = self.train["y"]
        
        

        print self.train["id"].value_counts()[:10]        
        #self.idvsTech30()
        
        
        
    
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
        
        plt.scatter(range(h), tech30)
        plt.show()
        
        sns.jointplot(tech30, y)
        plt.show()
        
        
