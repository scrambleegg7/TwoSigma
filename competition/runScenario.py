# -*- coding: utf-8 -*-

from DataClass import DataClass
from DataAnalysisClass import DataAnalysisClass

from logging import getLogger, StreamHandler, DEBUG
from logging import Formatter


from logClass import MyHandler


def proc1():
    
    
    
    pass

def analysis(log):
    
    analysisCls = DataAnalysisClass()
    
    analysisCls.timeStampAnalysis()
    
    
    analysisCls.featuresAnalyss()
    
    colnames = analysisCls.getColumns()
    
    #analysisCls.hist()


    #log.info(colnames)
    log.info(" length of columnas : %d" % len(colnames) )

    #return colnames


def loadCsvData(log):
    
    analysisCls = DataAnalysisClass()
    analysisCls.loadCsvData("df_top10corr.csv")
    
    log.info("file has been loaded............. " )


    analysisCls.startAnalysis()
def main():

    log = getLogger("root")
    log.setLevel(DEBUG)
    log.addHandler(MyHandler())

    
    
    #proc1()
    #analysis(log)

    loadCsvData(log)




if __name__ == "__main__":
    main()