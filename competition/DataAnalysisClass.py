# -*- coding: utf-8 -*-



from DataClass import DataClass

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import numpy as np

from pandas.tools.plotting import scatter_matrix
import pandas

from MyLoggerDeconstClass import MyLoggerDeconstClass

#from logging import getLogger, StreamHandler, DEBUG

class DataAnalysisClass(MyLoggerDeconstClass):
    
    def __init__(self):
        
        super(DataAnalysisClass,self).__init__()
        
        
        self.data = DataClass()
        
        self.train_data = self.data.getTrain()
        
        self.id = self.data.getId()
        
        self.y_train = self.data.getTarget()
        
        self.timestamp = self.data.getTimeStamp()
        
        
        self.my_df = None
        
    def loadCsvData(self,filename="my_df.csv"):
        
        self.my_df = self.data.loadCsvData(filename)
        self.my_df["id"] = self.id
        self.my_df['y'] = self.y_train
        
        print self.my_df.head()
        
    
    def timeStampAnalysis(self):
        
        
        self.log.info("length of timestamp:%d" % len(self.timestamp))
        
        timestamp_np = np.array(self.timestamp)
        
        mask = np.isnan(timestamp_np)
        
        nan_rate = np.sum(mask == True) / np.float(  len(mask)  )
        
        self.log.info("nan of timestamp: %.4f" % nan_rate)
    
    def pcaAnalysis(self):
        pass
        


    def startAnalysis(self):
        
        
        #scatter_matrix(self.my_df,alpha=0.2,figsize=(6,6),diagonal='kde')
        
        f13 = self.my_df["fundamental_13"]
        f31 = self.my_df["fundamental_31"]
        plt.scatter(f13,f31)        
        plt.show()
        
        plt.scatter(f31,self.y_train)
        plt.show()
        #sns.lmplot(x="fundamental_13",y="y",data=self.my_df,logistic=True)
        #sns.lmplot(x="fundamental_13",y="fundamental_31",data=self.my_df)
        #plt.show()
        
        sns.distplot(self.id, kde=False, rug=False, bins=1000) 

        plt.show()
    #
    # building data model fpr feature analysis
    #
    def BuildDataFeaturesAnalyss(self):
        
        startidx = 2 # start
        endidx = 110
        
        dff = self.train_data.iloc[:,startidx:endidx]
        
        # features from derived_0 to technical_44
        
        self.log.info("shape of features original :%d %d" % (dff.shape[0],dff.shape[1]   ))
        
        
        #dff_drop = dff.dropna()
        #self.log.info("shape of features drop NaN :%d %d" % (dff_drop.shape[0],dff_drop.shape[1]   ))
        #print dff_drop.head()
                
        #dff_zero = dff.fillna(0)
        #self.log.info("shape of features replace NaN with 0 :%d %d" % (dff_zero.shape[0],dff_zero.shape[1]   ))
        #print dff_zero.head()

        dff_mean = dff.fillna(   dff.mean()   )
        self.log.info("shape of features replace NaN with mean :%d %d" % (dff_mean.shape[0],dff_mean.shape[1]   ))

        print dff_mean.head()
        
        y_train = np.array(self.y_train)
        
        h,w = dff_mean.shape
        
        # numpy array list used ....
        mycolumns = np.array( dff_mean.columns.values.tolist() )

        
        # make matrix         
        mtr = np.ones( (w,w)  )
        
        # pickup upper triangle matrix from matrix
        # diag should be ZERO
        mtr_upper = np.triu(mtr,1)
        
        u_h, u_w = mtr_upper.shape
        new_combination_dict = {}
                
        for i in range(u_w):
            for j in range(u_w):

                if mtr_upper[i,j] == 1.:                
                    cor_ = np.corrcoef( dff_mean.iloc[:,i], dff_mean.iloc[:,j] )[0][1]
                    if cor_ != 1.:
                        new_combination_dict[ (i,j) ] = cor_
                
                    if (i * u_w + j) % 1000 == 0:
                        self.log.info("loop index 000s %d "  %  (i * u_w + j) )

                else:
                    continue                    
            
        self.log.info("lenth of new corr combination matix %d "  % len(new_combination_dict))
        
        # convert numpy from list 
        vals = np.array( new_combination_dict.values() ) 
        keys = np.array( new_combination_dict.keys() )
        
                
        mycorr_top10 = np.argsort( vals )[::-1][:10]
        mykeys_top10 = keys[ mycorr_top10 ]
        
        
        self.log.info("\n\n--top 10 correlation for features")
        
        
        top10_colnames = []
        for idx, (i,j) in enumerate(mykeys_top10):
            
            col1name = mycolumns[i]
            col2name = mycolumns[j]
            
            top10_colnames.append(col1name)
            top10_colnames.append(col2name)
            
            
            my_index = mycorr_top10[idx]
            
            self.log.info("%s:%s --> %.5f"  %  ( col1name,col2name, vals[my_index]    )   )
            
        top10_colnames = list(set(top10_colnames)) 
        print top10_colnames   
        
        #dff_top10 = dff_mean.iloc[:,top10_colnames]
        dff_top10 = dff_mean[top10_colnames]
        
        print dff_top10.head()

        self.log.info(" save fillna with mean high corr matix")

        self.data.saveCsvData(dff_top10,"df_top10corr.csv")

        self.log.info(" save done ...........")
        
        #g = sns.PairGrid(dff_top10)
        #g.map_diag(plt.hist)
        #g.map_offdiag(plt.scatter)

    
    
    def normalizedData(self):
          
        pass
        
    
    def __str__(self):
        "DataAnalysisClass"
        
    def DataNormalize(self):
        pass
    
    
    def getColumns(self):
        return self.data.getColumns()
        
    def yGraph(self):
        
        pass
    
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